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


from keras.models import load_model
import keras.losses
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit


width  = 36
height = 100
BS     = 1
GPUS   = 1
Nouts  = 60

XMIN   = -12.0
XMAX   =  12.0
BINSIZE = (XMAX-XMIN)/Nouts

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
				phi_calc = labelsdf.phi_calc[idx]
				labels = [phi, phi_calc]

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

# This is not actually used here, but some customLoss needs to be
# defined to prevent errors when the model is loaded.
def customLoss(y_true, y_pred):
	return K.sum(K.square(y_true - y_pred))


# Find most recent saved model
for epoch_loaded in range(2000, -1, -1): # there is probably a more efficient way!
  fname = 'model_checkpoints/model_epoch%03d.h5' % epoch_loaded
  if os.path.exists( fname ): break

if epoch_loaded <= 0:
  print('Unable to find saved model.')
  sys.exit(0)

model_saved_time_str = datetime.utcfromtimestamp(os.path.getmtime(fname)-4*3600).strftime('%Y-%m-%d %H:%M:%S')
print('Loading model: ' + fname + '  : ' + model_saved_time_str)
keras.losses.customLoss = customLoss
model = load_model( fname )
model.summary()

test_generator = generate_arrays_from_file('TEST', testdf)

# Create arrays to hold results
phis_truth = np.zeros(NTEST)
phis_gauss = np.zeros(NTEST)
phis_wavg  = np.zeros(NTEST)
diff_gauss = np.zeros(NTEST)
diff_wavg  = np.zeros(NTEST)
diff_calc  = np.zeros(NTEST)

# Loop over test samples
print ('Looping through test events ...')
start_time = time.time()
for i in range(0, NTEST):
	(x,y) = next(test_generator)
	image = x[0][0]
	phi = y[0][0]
	phi_calc = y[0][1]
	model_prediction = model.predict(x)
	
	# Assume Gaussian shape of outputs with A=10 and sigma=2bins
	# We use this to calculate the Gaussian center. It is complicated
	# a bit by both edge effects that may cut off part of the
	# Gaussian and ambiguity in whether the true value is to the left
	# or right of the bin center for the max bin
	A = 10.0     # amplitude is 10 (unless a bin value is greater)
	sigma = 2.0  # sigma is 2 bins
	Nouts = len(model_prediction[0])
	ibin_max = np.argmax( model_prediction )
	if model_prediction[0][ibin_max] > A : A = model_prediction[0][ibin_max]
	x_vals  = []  # list of Gaussian centers as calculated from all bins
	r_center = 0  # magnitude of deviation of true value from bin center for max bin
	for ibin in range( ibin_max-5, ibin_max+6):
		if ibin < 0: continue
		if ibin >= Nouts: continue
		x_i = float(ibin)+0.5          # bin center in units of bins
		y_i = model_prediction[0][ibin]   # value of bin (via network output)
		if y_i/A < 0.05: continue
		r = sigma*math.sqrt(2.0*math.log(A/y_i)) # magnitude of deviation from bin center of Gaussian center
		if ibin<ibin_max : x_vals  += [x_i + r]
		if ibin>ibin_max : x_vals  += [x_i - r]
		if ibin==ibin_max: r_center = r

	
	# Use x vals from bins on either side of max bin to determine
	# whether true value is to left or right of max bin's center
	x_tmp = np.average( x_vals )  # average x without center bin
	x_center = float(ibin_max)+0.5
	if x_tmp > x_center:
		x_vals += [x_center + r_center]
	else:
		x_vals += [x_center - r_center]

	x_avg = np.average( x_vals )  # average x with center bin (in units of bins)

	# Get weighted average of bins
	norm = np.sum(model_prediction[0])
	idx = np.arange(0, Nouts)
	phi_wavg = np.dot( model_prediction[0], idx) / norm
	
	# Convert back into degrees
	phi_model = (x_avg*BINSIZE) + XMIN
	phi_wavg = ((phi_wavg+0.5)*BINSIZE) + XMIN

	phis_truth[i] = phi
	phis_gauss[i] = phi_model
	phis_wavg[i] = phi_wavg
	diff_gauss[i] = phi_model - phi
	diff_wavg[i] = phi_wavg - phi
	diff_calc[i] = phi_calc - phi

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
bins = np.linspace(-0.2, 0.2, 201)

# First, define the figure which consists of two plots, side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
ax1.grid()
ax2.grid()
ax1.set_facecolor((0.9,0.9,0.99))
ax2.set_facecolor((0.9,0.9,0.99))

# Create histograms and save bin contents and definitions for fitting later
n_wavg, bins_wavg, p_wavg = ax1.hist(diff_wavg, bins, alpha=0.5, label='weighted avg.', color='blue')
n_gaus, bins_gaus, p_gaus = ax1.hist(diff_gauss, bins, alpha=0.5, label='gauss', color='red')
n_calc, bins_calc, p_calc = ax1.hist(diff_calc, bins, alpha=0.5, label='optimal', color='black')
ax1.legend(loc='upper right')
ax1.set_title('Resolution after %d epochs' % epoch_loaded)

ax2.scatter(phis_truth, phis_wavg, marker='.', alpha=0.5, label='weighted avg.', color='blue')
ax2.scatter(phis_truth, phis_gauss, marker='.', alpha=0.5, label='gauss', color='red')
ax2.legend(loc='upper left')
ax2.set_title('Correlation after %d epochs' % epoch_loaded)
ax2.set_xlabel('phi truth (degrees)')
ax2.set_ylabel('phi model (degrees)')

# Simple, unweighted gaussian fit
(mu, sigma_calc) = scipy.stats.norm.fit(diff_calc)
(mu, sigma_wavg) = scipy.stats.norm.fit(diff_wavg)
(mu, sigma_gauss) = scipy.stats.norm.fit(diff_gauss)

#---- Fit gaussian with 1/sqrt(N) bin uncertainty
#---- There must be a simpler way to do this!

# Define Gaussian function
def fgaus(x, A, x0, sigma):
    """ Gaussian with amplitude A """
    #print('A:%f  x0:%f  sigma:%f' % (A, x0,sigma))
    delta = (x-x0)/sigma
    return A * np.exp( -(delta**2)/2.0 )

# Calculate sigmas for each bin
sigmas_calc = np.sqrt(n_calc)
sigmas_wavg = np.sqrt(n_wavg)
sigmas_gaus = np.sqrt(n_gaus)

# Prevent division by zero by making all empty bins have huge sigmas
sigmas_calc[sigmas_calc==0.0] = 1.0E6
sigmas_wavg[sigmas_wavg==0.0] = 1.0E6
sigmas_gaus[sigmas_gaus==0.0] = 1.0E6

# The array returned from the hist method above are the bin edges so
# contain number of bins+1 elements. Convert to array with values at
# bin centers
bins_calc = bins_calc[:-1] + 0.5*(bins_calc[1]-bins_calc[0])


# Do the weighted fit
try:
	popt_calc, pcov = curve_fit(fgaus, bins_calc, n_calc, (np.amax(n_calc), 0.0, 0.015), sigma=sigmas_calc, absolute_sigma=True)
except Exception as E:
	print('Exception  thrown while curve fitting calc.')
	print(E)
	print(bins_calc)
	print(n_calc)
try:
	popt_wavg, pcov = curve_fit(fgaus, bins_calc, n_wavg, (np.amax(n_wavg), 0.0, 0.030), sigma=sigmas_wavg, absolute_sigma=True)
except Exception as E:
	print('Exception  thrown while curve fitting wavg.')
	print(E)
	print(bins_calc)
	print(n_wavg)
try:
	popt_gaus, pcov = curve_fit(fgaus, bins_calc, n_gaus, (np.amax(n_gaus), 0.0, 0.150), sigma=sigmas_gaus, absolute_sigma=True)
except Exception as E:
	print('Exception  thrown while curve fitting gaus.')
	print(bins_calc)
	print(n_gaus)
	print(E)

# Plot curve
fit_calc = scipy.stats.norm.pdf(bins_calc, popt_calc[1], popt_calc[2])*popt_calc[0]*(math.sqrt(2.0*math.pi)*popt_calc[2]) # norm properly normalizes gaussian so we need to remove that and apply amplitude parameter
fit_wavg = scipy.stats.norm.pdf(bins_calc, popt_wavg[1], popt_wavg[2])*popt_wavg[0]*(math.sqrt(2.0*math.pi)*popt_wavg[2]) # norm properly normalizes gaussian so we need to remove that and apply amplitude parameter
fit_gaus = scipy.stats.norm.pdf(bins_calc, popt_gaus[1], popt_gaus[2])*popt_gaus[0]*(math.sqrt(2.0*math.pi)*popt_gaus[2]) # norm properly normalizes gaussian so we need to remove that and apply amplitude parameter
ax1.plot(bins_calc, fit_calc, 'k', linewidth=2, color='black')
ax1.plot(bins_calc, fit_wavg, 'k', linewidth=2, color='blue')
ax1.plot(bins_calc, fit_gaus, 'k', linewidth=2, color='red')

# Draw fit resolutions on plot
ax1.text(0.10, 0.70*popt_calc[0], '$\sigma_{opti}$   = %5.4f$^o$' %  popt_calc[2], fontsize=15)
ax1.text(0.10, 0.55*popt_calc[0], '$\sigma_{wavg}$ = %5.4f$^o$' %  popt_wavg[2], fontsize=15)
ax1.text(0.10, 0.40*popt_calc[0], '$\sigma_{gaus}$  = %5.4f$^o$' %  popt_gaus[2], fontsize=15)

print('unweighted fit results: sigma_opti=%f  sigma_wavg=%f  sigma_gauss=%f' % (sigma_calc, sigma_wavg, sigma_gauss) )
print('Model used was saved on: ' + model_saved_time_str )

fig.show()
imgfname = 'resolution_epoch%03d.png' % epoch_loaded
plt.savefig(imgfname)
print('Saved plot to: %s' % imgfname) 
plt.waitforbuttonpress()
