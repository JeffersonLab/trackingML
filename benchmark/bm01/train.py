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
EPOCHS = 1000
BS     = 2000
GPUS   = 0
Nouts  = 60

XMIN   = -12.0
XMAX   =  12.0
BINSIZE = (XMAX-XMIN)/Nouts

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
				labels = np.empty(Nouts)
				for i in range(0,Nouts):
					phi_bin = XMIN + (i+0.5)*BINSIZE
					phi_diff = phi - phi_bin
					sigma = 2.0*BINSIZE
					g = phi_diff/sigma
					labels[i] = -0.5*g*g
				labels = 10.0*np.exp(labels)

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
# customLoss
#-----------------------------------------------------
# Define custom loss function that compares calcukated phi
# to true
def customLoss(y_true, y_pred):
	
	# For y_pred shape is (batch, Nouts)  so axis=0 for batch, axis=1 for Nouts
	# For y_true shape is (batch, 1)
	
	# Calculate weighted sum of prediction
	ones = K.ones_like(y_pred[0,:])   # [1, 1, 1, 1....]   (size Nouts)
	idx  = K.cumsum(ones)             # [1, 2, 3, 4....]   (size Nouts)
	norm_pred = K.sum(y_pred, axis=1)    # normalization of all outputs by batch. shape is 1D array of size batch
	wsum_pred = K.sum(idx*y_pred, axis=1)/norm_pred  # array of size batch with weighted avg. of mean in units of bins

	# Calculate weighted sum of label (this gives us true phi)
	norm_true = K.sum(y_true, axis=1)    # normalization of all outputs by batch. shape is 1D array of size batch
	wsum_true = K.sum(idx*y_true, axis=1)/norm_true  # array of size batch with weighted avg. of mean in units of bins

	# loss_weighted_avg is mean squared difference between weighted avg.
	# of true and prediction (i.e. measures in direction of phi, not in bin height)
	# This should force accuracy in phi prediction
	nbatch = K.sum(K.ones_like(y_pred[:,0]))  # number of values in THIS batch (may be less than BS)
	loss_weighted_avg = BINSIZE*K.sum(K.square(wsum_true - wsum_pred))/nbatch

	# loss_mse is the mean squared error of the heights for the labels compared to the
	# the prediction summed over bins and batches. This should force a Guassian shape
	loss_mse = BINSIZE*K.sum(K.square(y_true - y_pred))/nbatch/Nouts
	
	# total loss is sum of both losses. MSE needs to be scaled down
	# by about 30 to make them contribute equally. The boost_wavg factor gives
	# additional boost to weighted average to make it count more than mse.
	boost_wavg = 500.0  # make 1 to make wavg and mse contribute equally, make larger to make wavg count more
	loss_total = loss_weighted_avg + loss_mse/30.0/boost_wavg

	# Print some info. about the shapes of some tensors.
	# Note that this function only gets called once. Tensorflow knows how to
	# calculate loss itself after calling this.
	print('y_true shape: '    + str(y_true.shape)    )
	print('y_pred shape: '    + str(y_pred.shape)    )
	print('idx shape: '       + str(idx.shape)       )
	print('norm_pred shape: ' + str(norm_pred.shape) )
	print('wsum_pred shape: ' + str(wsum_pred.shape) )
	print('out shape: '       + str(loss_total.shape))
	return loss_total

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
	model.add(Conv2D(1, (1,1), activation="relu", kernel_initializer="glorot_uniform", strides=(1,1), input_shape=(height, width, 1), padding="same", data_format="channels_last") )
	model.add(Flatten())
	model.add(Dense(int(Nouts*5), activation='linear', kernel_initializer="glorot_uniform"))
	model.add(Dense(Nouts, activation='relu', kernel_initializer="glorot_uniform"))
	model.summary()

	if GPUS<=1 :
		parallel_model = model
	else:
		parallel_model = multi_gpu_model( model, gpus=GPUS )
	
	# Compile the model and print a summary of it
	opt = Adadelta(clipnorm=1.0)
	parallel_model.compile(loss=customLoss, optimizer=opt, metrics=['mae', 'mse', 'accuracy'])
	
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


# The following was used to test the custom loss function at one point.
# It is left here as an example.
#testdf = pd.read_csv('TEST/track_parms.csv')
#test_generator = generate_arrays_from_file('TEST', testdf)
#x = Input(shape=(None,))
#y = Input(shape=(None,))
#loss_func = K.Function([x,y], [customLoss(x,y)])
#myinputs1, mylabel1 = next(test_generator)
#myinputs2, mylabel2 = next(test_generator)
#print('label shape: ' + str(mylabel1.shape))
#print('mylabel1:')
#print(mylabel1)
#print('mylabel2:')
#print(mylabel2)
#myloss = loss_func([mylabel1, mylabel2])
#print(myloss)
#print('------------------------------')


