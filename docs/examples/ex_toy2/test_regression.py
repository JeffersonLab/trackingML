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
np.random.seed(123)  # for reproducibility


from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Flatten, Dropout, BatchNormalization, Input, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

width = 200
height = 200
BS = 1



# Create generator to read in images and labels
# (used for both training and validation samples)
def generate_arrays_from_file( path, labelsdf ):

	images_path = path+'/images.raw.gz'
	print('generator created for: ' + images_path )

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
				
				# Read in one set of labels. normalize to units of pi
				labels =  [ labelsdf.phi[idx]/3.14 ]

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
model = load_model("model2.h5")
model.summary()

# Open labels files so we can get number of samples and pass the
# data frames to the generators later
testdf = pd.read_csv('TEST/track_parms.csv')
NTEST = len(testdf)

# Create test generator
test_generator = generate_arrays_from_file('TEST', testdf)

# Open output file (simple ascii)
with open('phi_test.dat', 'w') as f:

	# Loop over test samples
	print 'Writing test results to output ...'
	for i in range(0, NTEST):
		(x,y) = next(test_generator)
		image = x[0][0]
		phi = y[0][0]*3.14
		phi_model = model.predict(x)*3.14
		f.write('%f %f\n' % (phi, phi_model))
		if (i%100) == 0:
			sys.stdout.write('  %d/%d written  \r' % (i, NTEST))
			sys.stdout.flush()

	f.close()
	print('\nDone')
