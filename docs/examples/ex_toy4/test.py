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

import os
import sys
import gzip
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime

# If running on Google Colaboratory you can uncomment the
# following and modify to use your Google Drive space.
#from google.colab import drive
#drive.mount('/content/gdrive')
#workdir = '/content/gdrive/My Drive/work/2019.03.26.trackingML/eff100_inverted'
#os.chdir( workdir )


from keras.models import load_model, Model
import keras.losses
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit


width  = 36
height = 100
BS     = 1
GPUS   = 0
Nouts  = 60

PHIMIN   = -12.0
PHIMAX   =  12.0
PHI_BINSIZE = (PHIMAX-PHIMIN)/Nouts

ZMIN   = -9.0
ZMAX   =  9.0
Z_BINSIZE = (ZMAX-ZMIN)/Nouts

# Open labels file for TEST set
testdf = pd.read_csv('TEST/track_parms.csv')
NTEST = int(len(testdf)/1)


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

				# Read in one set of labels
				phi = labelsdf.phi[idx]
				z = labelsdf.z[idx]
				labels = [phi, z]

				# Add to batch and check if it is time to yield
				batch_input.append( pixels_norm )
				batch_labels.append( labels )
				idx += 1
				if len(batch_input) == BS :
					ibatch += 1
					yield ( np.array(batch_input), np.array(batch_labels) )
					batch_input  = []
					batch_labels = []

			idx = 0
			f.close()


# Find most recent saved model
epoch_loaded = -1
for f in os.listdir('model_checkpoints'):
	if f.startswith('model_epoch') and f.endswith('.h5'):
		e = int(f[11:-3])
		if e > epoch_loaded:
			epoch_loaded = e
			fname = 'model_checkpoints/model_epoch%03d.h5' % epoch_loaded

if epoch_loaded <= 0:
  print('Unable to find saved model.')
  sys.exit(0)

model_saved_time_str = datetime.utcfromtimestamp(os.path.getmtime(fname)-4*3600).strftime('%Y-%m-%d %H:%M:%S')
print('Loading model: ' + fname + '  : ' + model_saved_time_str)
model = load_model( fname, custom_objects={'tf' : tf} )
model.summary()

test_generator = generate_arrays_from_file('TEST', testdf)

# Create arrays to hold results
phi_truth = np.zeros(NTEST)
phi_model = np.zeros(NTEST)
diff_phi  = np.zeros(NTEST)
z_truth   = np.zeros(NTEST)
z_model   = np.zeros(NTEST)
diff_z    = np.zeros(NTEST)

# Loop over test samples
print ('Looping through test events ...')
start_time = time.time()
for i in range(0, NTEST):
	(x,y) = next(test_generator)
	image = x[0][0]
	phi   = y[0][0]
	z     = y[0][1]
	model_prediction = model.predict(x)
	
	phi_truth[i] = phi
	phi_model[i] = model_prediction[0]
	diff_phi[i]  = phi_model[i] - phi_truth[i]
	z_truth[i]   = phi
	z_model[i]   = model_prediction[1]
	diff_z[i]    = z_model[i] - z_truth[i]

	# Print progress info after every 1000 inputs
	if (i%1000) == 0:
		delta_t = time.time() - start_time
		t_inference = delta_t*1.0E3/1000.0
		percent_done = 100.0*float(i)/float(NTEST)
		sys.stdout.write('\r %3.1f%% %d/%d written  %3.3f ms per inference' % (percent_done ,  i, NTEST, t_inference))
		sys.stdout.flush()
		start_time = time.time()
print('\nDone')

#------------ Make plots of results
bins_def_phi = np.linspace(-0.2, 0.2, 201)
bins_def_z   = np.linspace(-10.0, 10.0, 201)

# First, define the figure which consists of two plots, side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
ax1.grid()
ax2.grid()
ax1.set_facecolor((0.9,0.9,0.99))
ax2.set_facecolor((0.9,0.9,0.99))

# Create histograms and save bin contents and definitions for fitting later
n_phi, bins_phi, p_phi = ax1.hist(diff_phi, bins_def_phi, alpha=0.5, label='phi', color='blue')
n_z  , bins_z  , p_z   = ax2.hist(diff_z  , bins_def_z  , alpha=0.5, label='z'  , color='red')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax1.set_title('$\phi$ Resolution after %d epochs' % epoch_loaded)
ax2.set_title('$z$ Resolution after %d epochs' % epoch_loaded)
ax1.set_xlabel('$\delta\phi$ (degrees)')
ax2.set_xlabel('$\delta z$ (cm)')

# Simple, unweighted gaussian fit
(mu, sigma_phi) = scipy.stats.norm.fit(diff_phi)
(mu, sigma_z  ) = scipy.stats.norm.fit(diff_z  )

#---- Fit gaussian with 1/sqrt(N) bin uncertainty
#---- There must be a simpler way to do this!

# Define Gaussian function
def fgaus(x, A, x0, sigma):
    """ Gaussian with amplitude A """
    #print('A:%f  x0:%f  sigma:%f' % (A, x0,sigma))
    delta = (x-x0)/sigma
    return A * np.exp( -(delta**2)/2.0 )

# Calculate sigmas for each bin
sigmas_phi = np.sqrt(n_phi)
sigmas_z   = np.sqrt(n_z  )

# Prevent division by zero by making all empty bins have huge sigmas
sigmas_phi[sigmas_phi==0.0] = 1.0E6
sigmas_z[sigmas_z==0.0] = 1.0E6

# The array returned from the hist method above are the bin edges so
# contain number of bins+1 elements. Convert to array with values at
# bin centers
bins_phi = bins_phi[:-1] + 0.5*(bins_phi[1]-bins_phi[0])
bins_z   = bins_z[:-1] + 0.5*(bins_z[1]-bins_z[0])


# Do the weighted fit
try:
	popt_phi, pcov = curve_fit(fgaus, bins_phi, n_phi, (np.amax(n_phi), 0.0, 0.015), sigma=sigmas_phi, absolute_sigma=True)
except Exception as E:
	print('Exception  thrown while curve fitting phi.')
	print(E)
	print(bins_phi)
	print(n_phi)
try:
	popt_z, pcov = curve_fit(fgaus, bins_z, n_z, (np.amax(n_z), 0.0, 0.030), sigma=sigmas_z, absolute_sigma=True)
except Exception as E:
	print('Exception  thrown while curve fitting z.')
	print(E)
	print(bins_z)
	print(n_z)

# Plot curve
fit_phi = scipy.stats.norm.pdf(bins_phi, popt_phi[1], popt_phi[2])*popt_phi[0]*(math.sqrt(2.0*math.pi)*popt_phi[2]) # norm properly normalizes gaussian so we need to remove that and apply amplitude parameter
fit_z   = scipy.stats.norm.pdf(bins_z, popt_z[1], popt_z[2])*popt_z[0]*(math.sqrt(2.0*math.pi)*popt_z[2]) # norm properly normalizes gaussian so we need to remove that and apply amplitude parameter
ax1.plot(bins_phi, fit_phi, 'k', linewidth=2, color='black')
ax2.plot(bins_z, fit_z, 'k', linewidth=2, color='blue')

# Draw fit resolutions on plot
ax1.text(0.10, 0.70*popt_phi[0], '$\sigma_{\phi}$   = %5.4f$^o$' %  popt_phi[2], fontsize=15)
ax2.text(0.10, 0.55*popt_z[0], '$\sigma_{z}$ = %5.4f cm' %  popt_z[2], fontsize=15)

#print('unweighted fit results: sigma_opti=%f  sigma_wavg=%f  sigma_gauss=%f' % (sigma_calc, sigma_wavg, sigma_gauss) )
#print('Model used was saved on: ' + model_saved_time_str )

fig.show()
imgfname = 'resolution_epoch%03d.png' % epoch_loaded
plt.savefig(imgfname)
print('Saved plot to: %s' % imgfname) 
plt.waitforbuttonpress()
