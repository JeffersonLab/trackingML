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
from keras.models import Sequential
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
EPOCHS = 10
BS     = 2000
GPUS   = 1
Nouts  = 60

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

	batch_input  = []
	batch_labels = []
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
				
				# Make labels be array of size Nouts with contents set to Gaussian with
				# amplitude 10.0 and sigma 2 bins. The loss function will extract the
				# true phi from this. It is more complicated than just passing the phi
				# itself as the label, but allows training the outputs to be Gaussian.
				phi = labelsdf.phi[idx]
				idx += 1
				labels = [phi]

				# Add to batch and check if it is time to yield
				batch_input.append( pixels_norm )
				batch_labels.append( labels )
				if len(batch_input) == BS :
					ibatch += 1
					#print('\nyielding batch %d for %s' % (ibatch, path))
					yield ( np.array(batch_input), np.array(batch_labels) )
					batch_input  = []
					batch_labels = []

			idx = 0
			f.close()


#-----------------------------------------------------
# DefineModel
#-----------------------------------------------------
# This is used to define the model. It is only called if no model
# file is found in the model_checkpoints directory.
def DefineModel():
	
	# If GPUS==0 this will force use of CPU, even if GPUs are present
	# If GPUS>1 this will force the CPU to server as orchestrator
	# If GPUS==1 this will do nothing, allowing GPU to act as its own orchestrator
	if GPUS!=1: tf.device('/cpu:0')

	# Here we build the network model.
	model = Sequential()
	model.add(Conv2D(16, (3,3), activation="tanh", kernel_initializer="glorot_normal", strides=(1,1), input_shape=(height, width, 1), padding="same", data_format="channels_last") )
	model.add(Conv2D(16, (3,3), activation="tanh", kernel_initializer="glorot_normal", strides=(1,1), input_shape=(height, width, 1), padding="same", data_format="channels_last") )
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(1, activation='linear', kernel_initializer="glorot_normal"))
	model.summary()

	if GPUS<=1 :
		parallel_model = model
	else:
		parallel_model = multi_gpu_model( model, gpus=GPUS )
	
	# Compile the model and print a summary of it
	opt = Adadelta(clipnorm=1.0)
	parallel_model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse', 'accuracy'])
	
	return parallel_model

#===============================================================================


# Here we want to check if a model has been saved due to previous training.
# If so, then we read it in and continue training where it left off. Otherwise,
# we define the model and start fresh. This is perhaps more complicated than
# this example needs so you can feel free to delete this section.

# Look for most recent saved epoch
for epoch_loaded in range(2000, -1, -1): # there is probably a more efficient way!
	fname = 'model_checkpoints/model_epoch%03d.h5' % epoch_loaded
	if os.path.exists( fname ): break

if epoch_loaded > 0:
	print('Loading model: ' + fname)
	keras.losses.customLoss = customLoss
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
