#!/usr/bin/env python
#
#
# ex_toy3
#
# Train a network to calculate the phi angle from
# images where each pixel represents a wire in a
# toy detector model.
#
# See README.md for details.
#


import os
import sys
import gzip
import pandas as pd
import numpy as np
import math

# If running on Google Colaboratory you can uncomment the
# following and modify to use your Google Drive space.
#from google.colab import drive
#drive.mount('/content/gdrive')
#workdir = '/content/gdrive/My Drive/work/2019.03.26.trackingML/eff100_inverted'
#os.chdir( workdir )


from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Reshape, Flatten, Input, Lambda
from keras.optimizers import SGD, Adamax, Adadelta
from keras.initializers import glorot_normal
from keras.callbacks import Callback, TensorBoard
from keras.utils.training_utils import multi_gpu_model
import keras.backend as K
import keras.losses
import tensorflow as tf


width  = 36
height = 100
EPOCHS = 1000
BS     = 2000
GPUS   = 0
Nouts  = 60

PHIMIN   = -12.0
PHIMAX   =  12.0
PHI_BINSIZE = (PHIMAX-PHIMIN)/Nouts

ZMIN   = -10.0
ZMAX   =  10.0
Z_BINSIZE = (ZMAX-ZMIN)/Nouts

# Open labels files so we can get number of samples and pass the
# data frames to the generators later
traindf = pd.read_csv('TRAIN/track_parms.csv')
validdf = pd.read_csv('VALIDATION/track_parms.csv')
STEP_SIZE_TRAIN = len(traindf)/BS
STEP_SIZE_VALID = len(validdf)/BS

#-----------------------------------------------------
# generate_arrays_from_file
#-----------------------------------------------------
# Create generator to read in images and labels
# (used for both training and validation samples)
def generate_arrays_from_file( path, labelsdf ):

	images_path = path+'/images.raw.gz'
	print( 'generator created for: ' + images_path)

	batch_input      = []
	batch_labels_phi = []
	batch_labels_z   = []
	idx = 0
	ibatch = 0
	while True:  # loop forever, re-reading images from same file
		with gzip.open(images_path) as f:
			while True: # loop over images in file
			
				# Read in one image
				bytes = f.read(width*height)
				if len(bytes) != (width*height): break # break into outer loop so we can re-open file
				data = np.frombuffer(bytes, dtype='B', count=width*height)
				pixels = np.reshape(data, [width, height, 1], order='F')
				pixels_norm = np.transpose(pixels.astype(np.float) / 255., axes=(1, 0, 2) )
				
				# Read one label
				phi = labelsdf.phi[idx]
				z   = labelsdf.z[idx]
				idx += 1

				# Add to batch and check if it is time to yield
				batch_input.append( pixels_norm )
				batch_labels_phi.append( phi )
				batch_labels_z.append( z )
				if len(batch_input) == BS :
					ibatch += 1
					
					# Since we are training multiple loss functions we must
					# pass the labels back as a dictionary whose keys match
					# the layer their corresponding values are being applied
					# to.
					labels_dict = {
						'phi_output' :  np.array(batch_labels_phi ),
						'z_output'   :  np.array(batch_labels_z   ),		
					}
					
					yield ( np.array(batch_input), labels_dict )
					batch_input      = []
					batch_labels_phi = []
					batch_labels_z   = []

			idx = 0
			f.close()

#-----------------------------------------------------
# MyWeightedAvg
#
# This is used by the final Lambda layer in each branch
# of the network. It defines the formula for calculating
# the weighted average of the inputs from the previous
# layer.
#-----------------------------------------------------
def MyWeightedAvg(inputs, binsize, xmin):
	ones = K.ones_like(inputs[0,:])                       # [1, 1, 1, 1....]   (size Nouts)
	idx  = K.cumsum(ones)                                 # [1, 2, 3, 4....]   (size Nouts)
	norm = K.sum(inputs, axis=1, keepdims=True)           # normalization of all outputs by batch. shape is 1D array of size batch (n.b. keepdims=True is critical!)
	wsum = K.sum(idx*inputs, axis=1, keepdims=True)/norm  # array of size batch with weighted avg. of mean in units of bins (n.b. keepdims=True is critical!)
	output = (binsize*(wsum-0.5)) + xmin                  # convert from bins to physical units (shape batch,1)

	print('MyWeightedAvg:')
	print('       binsize = %f' % binsize)
	print('          xmin = %f' % xmin)
	print('   input shape = %s' % str(inputs.shape))
	print('  output shape = %s' % str(output.shape))
	
	return output

#-----------------------------------------------------
# DefineCommonModel
#-----------------------------------------------------
def DefineCommonModel(inputs):
	x = Flatten(name='top_layer1')(inputs)
	x = Dense(int(Nouts*5), name='common_layer1', activation='linear', kernel_initializer="glorot_uniform")(x)
	return x

#-----------------------------------------------------
# DefinePhiModel
#-----------------------------------------------------
def DefinePhiModel(inputs):
	x = Dense(Nouts, name='phi_output_dist', activation='relu', kernel_initializer="glorot_uniform")(inputs)
	x = Lambda(MyWeightedAvg, output_shape=(1,), name='phi_output', arguments={'binsize':PHI_BINSIZE, 'xmin':PHIMIN})(x)
	return x
	
#-----------------------------------------------------
# DefineZModel
#-----------------------------------------------------
def DefineZModel(inputs):
	x = Dense(Nouts, name='z_output_dist', activation='relu', kernel_initializer="glorot_uniform")(inputs)
	x = Lambda(MyWeightedAvg, output_shape=(1,), name='z_output', arguments={'binsize':Z_BINSIZE, 'xmin':ZMIN})(x)
	return x

#-----------------------------------------------------
# DefineModel
#-----------------------------------------------------
# This is used to define the model. It is only called if no model
# file is found in the model_checkpoints directory.
def DefineModel():
	
	# If GPUS==0 this will force use of CPU, even if GPUs are present
	# If GPUS>1 this will force the CPU to serve as orchestrator
	# If GPUS==1 this will do nothing, allowing GPU to act as its own orchestrator
	if GPUS!=1: tf.device('/cpu:0')

	# Here we build the network model.
	# This model is made of multiple parts. The first handles the
	# inputs and identifies common features. The rest are branches with
	# each determining an output parameter from those features.
	inputs      = Input(shape=(height, width, 1), name='image_inputs')
	commonmodel = DefineCommonModel(inputs)
	phimodel    = DefinePhiModel( commonmodel )
	zmodel      = DefineZModel( commonmodel )
	model       = Model(inputs=inputs, outputs=[phimodel, zmodel])
	model.summary()

	# Here we specify a different loss function for every output branch.
	# We also specify a weight for each branch. The weights allow us to 
	# specify that it is more important to minimize certain losses more
	# than others.
	sigma_phi = 0.011  # estimated resolution in degrees (from previous training)
	sigma_z   = 0.100  # estimated resolution in cm (from previous training)
	losses = {
		'phi_output'      :  'mean_squared_error',
		'z_output'        :  'mean_squared_error',		
	}
	lossWeights = {
		'phi_output'      :  1.0/(sigma_phi*sigma_phi),
		'z_output'        :  1.0/(sigma_z*sigma_z),		
	}
	
	# Compile the model, possibly using multiple GPUs
	opt = Adadelta(clipnorm=1.0)
	if GPUS<=1 :
		final_model = model
	else:
		final_model = multi_gpu_model( model, gpus=GPUS )

	final_model.compile(loss=losses, loss_weights=lossWeights, optimizer=opt, metrics=['mae', 'mse', 'accuracy'])
	
	return final_model


#===============================================================================


# Here we want to check if a model has been saved due to previous training.
# If so, then we read it in and continue training where it left off. Otherwise,
# we define the model and start fresh. 

# Look for most recent saved epoch
epoch_loaded = -1
for f in os.listdir('model_checkpoints'):
	if f.startswith('model_epoch') and f.endswith('.h5'):
		e = int(f[11:-3])
		if e > epoch_loaded:
			epoch_loaded = e
			fname = 'model_checkpoints/model_epoch%03d.h5' % epoch_loaded

if epoch_loaded > 0:
	print('Loading model: ' + fname)
	model = load_model( fname )
else:
	print('Unable to find saved model. Will start from scratch')
	model = DefineModel()

# Print summary of model
model.summary()

#===============================================================================

# Create training and validation generators
train_generator = generate_arrays_from_file('TRAIN', traindf)
valid_generator = generate_arrays_from_file('VALIDATION', validdf)

# Use tensorboard to log training. To view the training log with the
# tensorboard gui you can run tensorboard to fire up a web server
# so you can use your browser to view the results.
#
# Note: you may need to move the log file to your
# local desktop and run tensorboard there.
#
#  tensorboard --logdir=./logs
tensorboard=TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=BS*GPUS, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

#-----------------------------------------------------
# class checkpointModel
#-----------------------------------------------------
# There is a bug in keras that causes an error when trying to save a model
# trained on multiple GPUs. The work around is to save the original model
# at the end of every epoch using a callback. See
#    https://github.com/keras-team/kersas/issues/8694
if not os.path.exists('model_checkpoints'): os.mkdir('model_checkpoints')
class checkpointModel(Callback):
	def __init__(self, model):
		self.model_to_save = model
	def on_epoch_end(self, epoch, logs=None):
		myepoch = epoch_loaded + epoch +1
		fname = 'model_checkpoints/model_epoch%03d.h5' % myepoch
		old_fname = 'model_checkpoints/model_epoch%03d.h5' % (myepoch-1)
		if os.path.exists( old_fname ):
			print('removing old model: %s' % old_fname)
			os.remove( old_fname )
		print('saving model: %s' % fname)
		self.model_to_save.save(fname)
cbk = checkpointModel( model )

cbk.on_epoch_end(-1)

# Fit the model
history = model.fit_generator(
  generator        = train_generator
  ,steps_per_epoch  = STEP_SIZE_TRAIN
  ,validation_data  = valid_generator
  ,validation_steps = STEP_SIZE_VALID
  ,epochs=EPOCHS
  ,use_multiprocessing=False
  ,callbacks=[tensorboard, cbk]
)

