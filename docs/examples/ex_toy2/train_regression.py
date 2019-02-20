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
# This trains a regression network to output a single
# floating point number representing the phi angle.
# This works OK, but not great. As it turns out, framing
# the problem as a classification problem uses many
# fewer parameters and gives much more accurate results.
# This is left here as an example though.
# Note: One key thing needed to make this work was to use
# linear activation functions. Others were tried, but
# had trouble converging.

import sys
import gzip
import pandas as pd
import numpy as np


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Dropout, BatchNormalization, Input, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

width = 200
height = 200
BS = 64

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
	print('generator created for: ' + images_path) 

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
				
				# Read in one set of labels normalized to units of pi
				labels =  [ labelsdf.phi[idx]/3.14 ]

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
model.add(Dense(200, input_shape=(200,200,1,), activation='linear'))
model.add(Flatten())
model.add(Dense(5, activation='linear'))
model.add(Dense(1, activation='linear'))


# Compile the model and print a summary of it
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd, metrics=['mae', 'mse'])
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
   ,epochs=30
	,use_multiprocessing=True
	,callbacks=[tensorboard]
)

# Save model
model.save('model_regression.h5')	

# To view the training log with the tensorboard gui
# you can run tensorboard to fire up a web server
# so you can use your browser to view the results.
#
# Note: you may need to move the log file to your
# local desktop and run tensorboard there.
#
#  tensorboard --logdir=./logs


