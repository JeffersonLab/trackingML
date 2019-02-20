#!/usr/bin/env python
#
# ex_toy2
#
# Train a network on images of straight tracks to find
# the angle of the line in the image. Labels are the
# angle in radians.
#
# Training and validation sets should be in the 
# TRAIN and VALIDATION directories respectively. There
# should be a file "images_raw.gz" and "track_parms.csv"
# in each directory. These can be created with the 
# mkphiimages program. See README.md for more details.
#
# This treats the problem as a classification problem 
# rather than a regression problem (at least as far as
# the network is concerned). The network is designed to
# output 360 values corresponding to 1 degree bins.
# The labels are then arrays of 360 numbers with all of
# them zero except the bin in which the true value falls
# which is set to 1. On prediction, the bin with the 
# largest value is taken as the angle limiting the
# resolution to 1 degree/sqrt(12). 
#
# n.b. it is possible to get better resolution by combining
# information from multiple bins in the prediction, but
# that is not done in this example.

import sys
import gzip
import pandas as pd
import numpy as np
import math


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Dropout, BatchNormalization, Input, Activation
from keras.optimizers import SGD, Adamax
from keras.initializers import glorot_normal
from keras.callbacks import TensorBoard

width = 200
height = 200
BS = 32

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
	print 'generator created for: ' + images_path 

	batch_input  = []
	batch_labels = []
	idx = 0
	while True:  # loop forever, re-reading images from same file
		with gzip.open(images_path) as f:
			while True: # loop over images in file
			
				# Read in one image
				bytes = f.read(width*height)
				if len(bytes) != (width*height): break # break into outer loop so we can re-open file
				data = np.frombuffer(bytes, dtype='B', count=width*height)
				pixels = np.reshape(data, [width, height, 1], order='F')
				pixels_norm = pixels.astype(np.float) / 255.
				
				# Read in one set of labels.
				# Here, we convert the phi value into an array with
				# just one element set to 1.
				phi = labelsdf.phi[idx]
				phi_degrees = round( math.degrees(phi + math.pi) ) # 0-360 degrees
				labels = np.zeros(360)
				myidx = int(phi_degrees)
				if myidx>=0 and myidx<=359: labels[myidx] = 1.0    # one hot

				# Add to batch and check if it is time to yield
				batch_input.append( pixels_norm )
				batch_labels.append( labels )
				idx += 1
				if len(batch_input) == BS :
					yield ( np.array(batch_input), np.array(batch_labels) )
					batch_input  = []
					batch_labels = []

			idx = 0
			f.close()



# Here we build the network model.
model = Sequential()
model.add(Conv2D(2, (3, 3), activation="relu", kernel_initializer="glorot_normal", strides=(1,1), input_shape=(200, 200, 1), padding="same", data_format="channels_last") )
model.add(Conv2D(1, (3, 3), activation="relu", kernel_initializer="glorot_normal", strides=(1,1), padding="same", data_format="channels_last") )
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(360, activation='softmax'))


from keras import backend as K
def nll1(y_true, y_pred):
	return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

# Compile the model and print a summary of it
sgd = SGD()
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['mae', 'mse'])
model.summary()

# Create training and validation generators
train_generator = generate_arrays_from_file('TRAIN', traindf)
valid_generator = generate_arrays_from_file('VALIDATION', validdf)

# Use tensorboard to log training
tensorboard=TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=BS, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# Fit the model
history = model.fit_generator(
    generator        = train_generator
   ,steps_per_epoch  = STEP_SIZE_TRAIN
   ,validation_data  = valid_generator
   ,validation_steps = STEP_SIZE_VALID
   ,epochs=200
	,use_multiprocessing=True
	,callbacks=[tensorboard]
)

# Save model
model.save('model_classification.h5')	

# To view the training log with the tensorboard gui
# you can run tensorboard to fire up a web server
# so you can use your browser to view the results.
#
# Note: you may need to move the log file to your
# local desktop and run tensorboard there.
#
#  tensorboard --logdir=./logs


