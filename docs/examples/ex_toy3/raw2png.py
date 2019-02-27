#!/usr/bin/env python
#
# This script will read in a gzipped "images.raw.gz" file
# and create PNG files from it. The intent of this is to
# allow a visual check of the track "images" being fed
# into the AI
#
# The images.raw.gz file contains images of tracks made
# by the TrackMLextract plugin. The raw file format is
# just 8bit greyscale images, each width*height bytes.
#
# If a track_parms.csv file is present, it is read and
# the filename column used to name the files. Otherwise
# they are given a generic name based on position in the
# file.

import gzip
import png
import os
import sys
import numpy as np
import pandas as pd

#------------------ input parameters --------------------
width = 36
height = 100

iMIN = 0    # First image to grab starting from zero
iNUM = 25  # Number of images to grab

idx_fname = 'track_parms.csv'
outputdir = './'
#--------------------------------------------------------

# Read in file names if index file if present
fnames = []
if os.path.exists( idx_fname ):
	df = pd.read_csv( idx_fname )
	fnames = df.filename

# Make sure output directory exists
if not os.path.exists(outputdir): os.mkdir( outputdir )

# Open gzipped images file and loop over images
Ncreated = 0
with gzip.open('images.raw.gz') as f:

	# Skip iMIN images at front of file
	if iMIN>0:
		print 'Skipping %d images ...' % iMIN
		f.read(width*height*iMIN)
			
	# Create images from next iNUM records
	print 'Creating %d images in %s ...' % (iNUM, outputdir)
	for i in range(0,iNUM):

		# Read record and make it the right shape
		bytes = f.read(width*height)
		if len(bytes) != (width*height):
			print '\n File truncated!'
			break
		data = np.frombuffer(bytes, dtype='B', count=width*height)
		pixels = np.reshape(data, [height, width])
		
		# Image name
		idx = i + iMIN
		fname = 'img%06d.png' % idx 
		if idx < len(fnames): fname = fnames[idx]
		
		# Create image file
		png.fromarray(pixels, 'L').save( outputdir + '/' + fname )

		# Bookkeeping
		Ncreated += 1
		if (Ncreated%10) == 0:
			sys.stdout.write(' created %d/%d images\r' % (Ncreated,iNUM))
			sys.stdout.flush()
	
print 'Created ' + str(Ncreated) + ' image files'
print 'Done'
	
