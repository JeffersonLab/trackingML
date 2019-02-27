#!/usr/bin/env python
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


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Dropout, BatchNormalization, Input, Activation
from keras.optimizers import SGD, Adamax, Adadelta
from keras.initializers import glorot_normal
from keras.callbacks import Callback, TensorBoard
from keras.utils.training_utils import multi_gpu_model
import keras.backend as K
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3


width  = 36
height = 100
EPOCHS = 10
BS     = 1000
GPUS   = 4
Nouts  = 200


# Open labels files so we can get number of samples and pass the
# data frames to the generators later
traindf = pd.read_csv('TRAIN/track_parms.csv')
validdf = pd.read_csv('VALIDATION/track_parms.csv')
STEP_SIZE_TRAIN = len(traindf)/BS
STEP_SIZE_VALID = STEP_SIZE_TRAIN/10


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
					
				# Read in one set of labels.
				# Here, we convert the phi value into an array with
				# just one element set to 1.
				phi = labelsdf.phi[idx]/10.0  # normalize to units of 10 degrees so -1<=phi<=1
				idx += 1
				labels = np.zeros(Nouts)
				myidx = int( float(Nouts)*(phi+1.0)/2.0 )
				if myidx>=0 and myidx<Nouts:
					labels[myidx] = 1.0    # one hot
				else:
					continue

				# Add to batch and check if it is time to yield
				batch_input.append( pixels_norm )
				batch_labels.append( labels )
				if len(batch_input) == BS :
					ibatch += 1
					yield ( np.array(batch_input), np.array(batch_labels) )
					batch_input  = []
					batch_labels = []

			idx = 0
			f.close()



with tf.device('/cpu:0'):

	# Here we build the network model.
	model = Sequential()
	model.add(Conv2D(128, (3,3), activation="relu", kernel_initializer="glorot_normal", strides=(1,1), input_shape=(height, width, 1), padding="same", data_format="channels_last") )
	model.add(Conv2D(128, (3,3), activation="relu", kernel_initializer="glorot_normal", strides=(1,1), input_shape=(height, width, 1), padding="same", data_format="channels_last") )
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
	model.add(Flatten())
	model.add(Dense(Nouts, activation='softmax'))

if GPUS<=1 :
	parallel_model = model
else:
	parallel_model = multi_gpu_model( model, gpus=GPUS )

# Define custom loss function that compares calcukated phi to true
def customLoss(y_true, y_pred):
	ones = K.ones_like(y_true[0,:])  # [1, 1, 1, 1....]
	idx = K.cumsum(ones)             # [1, 2, 3, 4....]
	norm_true = K.sum(y_true, axis=1)    # normalization of all outputs by batch. shape is 1D array of size batch
	norm_pred = K.sum(y_pred, axis=1)    # normalization of all outputs by batch. shape is 1D array of size batch
	wsum_true = K.sum(idx*y_true, axis=1)/norm_true  # array of size batch with weighted avg.
	wsum_pred = K.sum(idx*y_pred, axis=1)/norm_pred  # array of size batch with weighted avg.
	out = K.sum(K.square(wsum_pred - wsum_true))/K.sum(ones)
	print('y_pred shape: ' + str(y_pred.shape) ) 
	print('idx shape: ' + str(idx.shape) ) 
	print('norm_true shape: ' + str(norm_true.shape) ) 
	print('wsum_true shape: ' + str(wsum_true.shape) ) 
	return out

# Below is code to test the custom loss function above.
# It is commented out since the output can be pretty verbose.
# It is left here as an example though since loss functions
# are kind of tricky which makes testing them at least as tricky.
#
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

# Compile the model and print a summary of it
opt = Adadelta(clipnorm=1.0)
parallel_model.compile(loss=customLoss, optimizer=opt, metrics=['mae', 'mse'])
parallel_model.summary()

# Create training and validation generators
train_generator = generate_arrays_from_file('TRAIN', traindf)
valid_generator = generate_arrays_from_file('VALIDATION', validdf)

# Use tensorboard to log training
tensorboard=TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=BS*GPUS, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# There is a bug in keras that causes an error when trying to save a model
# trained on multiple GPUs. The work around is to save the original model
# at the end of every epoch using a callback. See 
#    https://github.com/keras-team/kersas/issues/8694
class checkpointModel(Callback):

	def __init__(self, model):
		self.model_to_save = model
		if not os.path.exists('model_checkpoints'): os.mkdir('model_checkpoints')

	def on_epoch_end(self, epoch, logs=None):
		self.model_to_save.save('model_checkpoints/model_epoch%03d.h5' % epoch)

cbk = checkpointModel( model )


# Fit the model
history = parallel_model.fit_generator(
    generator        = train_generator
   ,steps_per_epoch  = STEP_SIZE_TRAIN
   ,validation_data  = valid_generator
   ,validation_steps = STEP_SIZE_VALID
   ,epochs=EPOCHS
	,use_multiprocessing=False
	,callbacks=[tensorboard, cbk]
)

# Save model (disabled due to bug. See checkpointModel above)
#parallel_model.save('model.h5')	

# To view the training log with the tensorboard gui
# you can run tensorboard to fire up a web server
# so you can use your browser to view the results.
#
# Note: you may need to move the log file to your
# local desktop and run tensorboard there.
#
#  tensorboard --logdir=./logs


