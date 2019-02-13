#!/usr/bin/env python
#
# ex_toy1
#
# Train a network to add 3 numbers together. This is
# a trivial example meant to test a keras+tensorflow
# installation. It also demonstrates how one can:
#
# 1. Inject input with a custom generator
# 2. Save the model for later use
# 3. Log the training with tensorboard for later inspection
#


import sys
import numpy as np
import random
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

BS = 32                               # batch size
STEP_SIZE_TRAIN = 10000/BS            # number of batches to train on
STEP_SIZE_VALID = STEP_SIZE_TRAIN/10  # number of batches to validate on


# Data generator
def generate_arrays( ):

	while True:  # loop forever, re-reading images from same file
	
		batch_input  = []
		batch_labels = []
		for ibatch in range (0,BS):

			# Inputs are 3 random numbers from 0-1
			# Labels are product of 3 numbers
			inputs = []
			for i in range(0,30): inputs.append( random.random() )
		
			label = sum(inputs)
		
			batch_input.append( inputs )
			batch_labels.append( label )
			
		yield ( np.array(batch_input), np.array(batch_labels) )



# Here we build the network model. A single layer with linear activation.
model = Sequential()
model.add(Dense(1, input_shape=(30,), activation='linear'))

# Compile the model and print a summary of it
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd, metrics=['mae', 'mse'])
model.summary()

# Create training and validation generators
train_generator = generate_arrays()
valid_generator = generate_arrays()

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
model.save('model2.h5')		

# To view the training log with the tensorboard gui
# you can run tensorboard to fire up a web server
# so you can use your browser to view the results.
#
# Note: you may need to move the log file to your
# local desktop and run tensorboard there.
#
#  tensorboard --logdir=./logs

