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

import sys
import gzip
import pandas as pd
import numpy as np
import math
np.random.seed(123)  # for reproducibility


from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Dropout, BatchNormalization, Input, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

width  = 36
height = 100
BS = 1



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
				pixels_norm = np.transpose(pixels.astype(np.float) / 255., axes=(1, 0, 2) )
				
				# Read in one set of labels
				phi = labelsdf.phi[idx]
				labels = [phi]

				# Add to batch and check if it is time to yield
				batch_input.append( pixels_norm )
				batch_labels.append( labels )
				idx += 1
				if len(batch_input) == BS :
					#print 'pos = %d  %f  %f  idx=%d' % (f.tell(), np.sum(batch_input), np.sum(batch_labels), idx)
					yield ( np.array(batch_input), np.array(batch_labels) )
					batch_input  = []
					batch_labels = []

			idx = 0
			f.close()


# Load model
epoch=1
model = load_model('model_checkpoints/model_epoch%03d.h5' % epoch)
model.summary()

# Open labels files so we can get number of samples and pass the
# data frames to the generators later
testdf = pd.read_csv('TEST/track_parms.csv')
NTEST = len(testdf)

# Create test generator
test_generator = generate_arrays_from_file('TEST', testdf)

# Open file for writing network output for a few events
fnetout = open('network_output.dat', 'w')

# Open output file (simple ascii)
with open('phi_test.dat', 'w') as f:

	# Loop over test samples
	print ('Writing test results to output ...')
	for i in range(0, NTEST):
		(x,y) = next(test_generator)
		image = x[0][0]
		phi = y[0][0]
		model_prediction = model.predict(x)
		
		# Write full outputs for first 100 samples
		if i<100 :
			for v in model_prediction[0]: fnetout.write('%f ' % v)
			fnetout.write('\n')
		
		# Get weighted average of bins
		Nouts = len(model_prediction[0])
		norm = np.sum(model_prediction[0])
		idx = np.arange(1, Nouts+1) 
		phi_model = np.dot( model_prediction[0], idx) / norm
		
		# Convert back into degrees
		phi_model = ((2.0*phi_model/len(model_prediction[0])) - 1.0)*10.0
		
		# Correct 1/2 bin systematic shift in the predicted values
		phi_model -= 20.0/Nouts/2.0
		
		f.write('%f %f\n' % (phi, phi_model))
		if (i%100) == 0:
			sys.stdout.write('  %d/%d written  \r' % (i, NTEST))
			sys.stdout.flush()

	f.close()
	fnetout.close()
	print( '\nDone')
