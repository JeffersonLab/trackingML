#!/usr/bin/env python
#
#
# ex_toy4
#
# Building on toy3 example, this adds drift distance
# information to the pixel color. It also adds a 
# random z-vertex position in addition to the phi
# angle. 
#
# The network is defined with 2 branches to calculate
# the phi and z. They share a common input layer and
# initial Dense layer then implement their own dense
# layers.
#
# Another difference from toy3 is that a final dense
# layer with a single neuron is added to each of the
# branches to calculate phi(z) parameters directly
# rather than doing that outside of the network. To
# help this, the weights feeding that last neuron are
# set to fixed weights (bin centers) and are marked
# as non-trainable.
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
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Dropout, BatchNormalization, Input, Activation
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

ZMIN   = -9.0
ZMAX   =  9.0
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

	batch_input           = []
	batch_labels_phi      = []
	batch_labels_z        = []
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
				
				# Labels
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
# DefineCommonModel
#-----------------------------------------------------
def DefineCommonModel(inputs):
	x = Flatten()(inputs)
	x = Dense(int(Nouts*5), name='top_layer1', activation='linear', kernel_initializer="glorot_uniform")(x)
	return x

#-----------------------------------------------------
# DefinePhiModel
#-----------------------------------------------------
def DefinePhiModel(inputs):
	x = Dense(int(Nouts*2), name='phi_input', activation='linear', kernel_initializer="glorot_uniform")(inputs)
	x = Dense(Nouts, name='phi_output_dist', activation='relu', kernel_initializer="glorot_uniform")(x)
	x = Dense(  1  , name='phi_output', activation='linear', trainable=False )(x)  # non-trainable layer used to calculate weighted average

	return x
	
#-----------------------------------------------------
# DefineZModel
#-----------------------------------------------------
def DefineZModel(inputs):

	x = Dense(int(Nouts*3), name='z_input', activation='linear', kernel_initializer="glorot_uniform")(inputs)
	x = Dense(Nouts, name='z_output_dist', activation='relu', kernel_initializer="glorot_uniform")(x)
	x = Dense(1, name='z_output', activation='linear', trainable=False )(x)  # non-trainable layer used to calculate weighted average

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

	# The last single neuron layer in each branch is a non-trainable
	# layer used to calculate the weighted average of the Nouts outputs
	# of the previous layer. We set the weights and bias here such that
	# it will be the weighted average of the previous layer's outputs.
	# Well, almost. The values actually need to be divided by the sum
	# of the outputs of the previous layer. Those will be changing
	# during training so we don't know those ahead of time. We have to
	# rely on the network training the upstream parameters to do this
	# normalization for us (or else do it in a custom loss function 
	# which we're not doing here).
	#
	# The shape of the weights is kind of funny. It includes Nouts
	# weights and a single bias. It is easiest to get the initial
	# weights first and use the returned structure to set them.
	weights = model.get_layer('phi_output').get_weights()
	for i in range(0, Nouts): weights[0][i][0] = PHIMIN + (i+0.5)*PHI_BINSIZE
	weights[1][0] = 0.0  # bias
	model.get_layer('phi_output').set_weights(weights)

	weights = model.get_layer('z_output').get_weights()
	for i in range(0, Nouts): weights[0][i][0] = ZMIN + (i+0.5)*Z_BINSIZE
	weights[1][0] = 0.0  # bias
	model.get_layer('z_output').set_weights(weights)

	# Here we specify a different loss function for every output branch.
	# We also specify a weight for each branch. The weights allow us to 
	# specify that it is more important to minimize certain losses more
	# than others.
	losses = {
		'phi_output'      :  'mean_squared_error',
		'z_output'        :  'mean_squared_error',		
	}
	lossWeights = {
		'phi_output'      :  1.0,
		'z_output'        :  1.0,		
	}
	
	# Compile the model, possibly using multiple GPUs
	opt = Adadelta(clipnorm=1.0)
	if GPUS<=1 :
		final_model = model
	else:
		final_model = multi_gpu_model( model, gpus=GPUS )

	final_model.compile(loss=losses, loss_weights=lossWeights, optimizer=opt, metrics=['mae', 'mse', 'accuracy'])
	
	return final_model

#-----------------------------------------------------
# class checkpointModel
#-----------------------------------------------------
# There is a bug in keras that causes an error when trying to save a model
# trained on multiple GPUs. The work around is to save the original model
# at the end of every epoch using a callback. See:
#    https://github.com/keras-team/kersas/issues/8694
#
# Note that in order to save disk space, we remove the previous model
# as we save a new one. We do keep every 20th epoch though in case 
# we want to go back and look at the evolution of the training.
if not os.path.exists('model_checkpoints'): os.mkdir('model_checkpoints')
class checkpointModel(Callback):
	def __init__(self, model):
		self.model_to_save = model
	def on_epoch_end(self, epoch, logs=None):
		myepoch = epoch_loaded + epoch +1
		fname = 'model_checkpoints/model_epoch%03d.h5' % myepoch
		old_fname = 'model_checkpoints/model_epoch%03d.h5' % (myepoch-1)
		if os.path.exists( old_fname ) and ((myepoch%20)!=1):
			print('removing old model: %s' % old_fname)
			os.remove( old_fname )
		print('saving model: %s' % fname)
		self.model_to_save.save(fname)

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


#===============================================================================
# Fit the model
history = model.fit_generator(
  generator        = train_generator
  ,steps_per_epoch  = STEP_SIZE_TRAIN
  ,validation_data  = valid_generator
  ,validation_steps = STEP_SIZE_VALID
  ,epochs=EPOCHS
  ,use_multiprocessing=False
  ,callbacks=[tensorboard, checkpointModel( model )]
)


