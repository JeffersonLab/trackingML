#!/usr/bin/env python3
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
from keras.layers import Dense, Reshape, Flatten, Input, Lambda, Layer
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

				# Here, we pull off just the wireid for each layer so we cn reduce the
				# 3600 inputs down to just 36. This assumes only one hit per layer.
				wires = np.argmax(pixels_norm, axis=0).flatten()

				# Read one label
				phi = np.arctan( labelsdf.phi[idx] )
				z   = labelsdf.z[idx]
				idx += 1

				# Add to batch and check if it is time to yield
				batch_input.append( wires )
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

# -----------------------------------------------------
# MyProduct
#
# This is used to make a layer that has the products of
# all elements of the input layer. Thus the output layer
# has N^2 nodes if the input layer has N nodes
# -----------------------------------------------------
def MyProduct(inputs):

	N = inputs.shape[1]
	outputs = []
	for i in range(0, N):
		for j in range(0, N):
			idx = i + j*N
			outputs.append( inputs[:, i] * inputs[:, j] )

	output = K.transpose( K.stack( outputs ) )
	print('MyProduct:')
	print('   input shape = %s' % str(inputs.shape))
	print('  output shape = %s' % str(output.shape))

	return output

# -----------------------------------------------------
# MyRatio
#
# This is used for layers where a simple ratio from two nodes
# in the previous layer are needed.
# -----------------------------------------------------
def MyRatio(inputs):

	output = inputs[:,0]/inputs[:,1]

	print('MyRatio:')
	print('   input shape = %s' % str(inputs.shape))
	print('  output shape = %s' % str(output.shape))

	return output


#-----------------------------------------------------
# MyFirstLayer
#
# inputs are 36 wire ids
# outputs are N, Sx, Sy, Sxx, Sxy
#
# Sx is made from sum of x values of each layer. Since
# x-values are static information, we let the network
# train to find them. Yes, we could just enter them
# here, but we're trying to get the network to train
# them in hopes this mechanism can be generalized later.
#
# Thus, this custom layer has 36 weights which should
# train to the x_i values. We sum those to get the
# value for Sx which is used to calculate the second 36
# values.
#-----------------------------------------------------
class MyFirstLayer(Layer):

	def __init__(self, output_dim=5, **kwargs):
		self.output_dim = output_dim
		print('self.output_dim: ' + str(self.output_dim))
		super(MyFirstLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		# n.b. input_shape[0] is should be batch size.
		# output_dim should be defined as 72 when this layer
		# is added to the model below.
		print('input_shape.shape: ' + str(input_shape))
		self.kernel = self.add_weight(name='kernel',
									  shape=(width,1),
									  initializer='uniform',
									  trainable=True)

		# Initialize the weights
		x = 30.0
		x_weights = self.get_weights()
		x_vals = []
		for igroup in range(0,6):
			for ichamber in range(0,6):
				idx = ichamber + (igroup*6)
				x_weights[0][idx] = x
				x += 1.0
			x += 49.0 # already pushed to 1 past last plane of previous group
		self.set_weights( x_weights )
		print(self.get_weights())

		super(MyFirstLayer, self).build(input_shape)  # Be sure to call this at the end

	def call(self, y):
		# The 36 weights being fit in the kernel are the x-values of the planes
		N   = width*K.ones_like(y[:,0])
		Sx  = K.sum(self.kernel)*K.ones_like(y[:,0])
		Sxx = K.sum(self.kernel*self.kernel)*K.ones_like(y[:,0])
		Sy  = K.sum(y, axis=1)
		Sxy = K.sum(K.transpose(y)*self.kernel, axis=0)
		print('     y: ' + str(   y ) + '  (transpose:' + str(K.transpose(y)) + ')')
		print('     N: ' + str(   N ))
		print('    Sx: ' + str(  Sx ))
		print('   Sxx: ' + str( Sxx ))
		print('    Sy: ' + str(  Sy ))
		print('   Sxy: ' + str( Sxy ))
		print('kernel: ' + str(self.kernel) )
		out = K.transpose(K.stack( (N, Sx, Sy, Sxx, Sxy) ))
		print('   out: ' +  str(out) )
		return out

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

#-----------------------------------------------------
# DefineCommonModel
#-----------------------------------------------------
def DefineCommonModel(inputs):
#def DefineCommonModel():

	# Here we set up to essentially calculate the Sxy and Sy values
	# as would be done in a linear regression. The values of Sx and
	# Sxx are considered constants in that each event should have
	# 36 hits, each with the same x-value as for every other event.
	# Thus, the x values are static and should be contained in the
	# trained weights.
	#
	# inputs: 36 x 1 array of inputs. Each value is wireid which
	#         is proportional to y
	#
	# first layer: 72 node Lamda layer. This copies the inputs twice
	#         The first 36 values are
	#
	# first dense layer (common_layer1): Contains the y_i and x_i*y_i
	# values that need to be summed to form Sy and Sxy.
	#
	# The second dense layer (common_layer2)
	# is products of these. Sepcifically Sxy, SxSy, SxxSy, and SxSxy.

	#x = Input(shape=(width,), name='inputs')
	x = MyFirstLayer(5, name='N_Sx_Sy_Sxx_Sxy')(inputs)
	x = Lambda(MyProduct, output_shape=(25,), name='S_products')(x)
	return x

#-----------------------------------------------------
# DefinePhiModel
#-----------------------------------------------------
def DefinePhiModel(inputs):
	x = Dense(2, name='phi_output_sums', activation='linear', kernel_initializer="glorot_uniform")(inputs)
	x = Lambda(MyRatio, output_shape=(1,), name='phi_output')(x)
	return x
	
#-----------------------------------------------------
# DefineZModel
#-----------------------------------------------------
def DefineZModel(inputs):
	x = Dense(2, name='z_output_dist', activation='linear', kernel_initializer="glorot_uniform")(inputs)
	x = Lambda(MyRatio, output_shape=(1,), name='z_output')(x)
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
	inputs      = Input(shape=( width, ), name='image_inputs')
	commonmodel = DefineCommonModel(inputs)
	#commonmodel = DefineCommonModel()
	phimodel    = DefinePhiModel( commonmodel )
	zmodel      = DefineZModel( commonmodel )
	model       = Model(inputs=inputs, outputs=[phimodel, zmodel])
	model.summary()

	# Here we specify a different loss function for every output branch.
	# We also specify a weight for each branch. The weights allow us to 
	# specify that it is more important to minimize certain losses more
	# than others.
	sigma_phi = 1000.011  # estimated resolution in degrees (from previous training)
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

#keras.layers.MyFirstLayer = MyFirstLayer

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
	model = load_model( fname, custom_objects={'MyFirstLayer':MyFirstLayer} )
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
		super(Callback, self).__init__()
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

