#!/usr/bin/env python

import os
import sys
import gzip
import pandas as pd
import numpy as np
import math



from keras.models import load_model
from keras.models import Sequential, Model
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

PHIMIN   = -12.0
PHIMAX   =  12.0
PHI_BINSIZE = (PHIMAX-PHIMIN)/Nouts

ZMIN   = -9.0
ZMAX   =  9.0
Z_BINSIZE = (ZMAX-ZMIN)/Nouts


#-----------------------------------------------------
# DefineEntryModel
#-----------------------------------------------------
def DefineEntryModel(inputs):
	x = Flatten()(inputs)
	x = Dense(int(Nouts*5), name='top_layer1', activation='linear', kernel_initializer="glorot_uniform")(x)
	return x

#-----------------------------------------------------
# DefinePhiModel
#-----------------------------------------------------
def DefinePhiModel(inputs):
	x = Dense(int(Nouts*2), name='phi_input', activation='linear', kernel_initializer="glorot_uniform")(inputs)
	x = Dense(Nouts, name='phi_output', activation='relu', kernel_initializer="glorot_uniform")(x)
	x = Dense(1, name='phi_wavg', activation='relu', kernel_initializer="glorot_uniform")(x)
	return x
	
#-----------------------------------------------------
# DefineZModel
#-----------------------------------------------------
def DefineZModel(inputs):

	x = Dense(int(Nouts*3), name='z_input', activation='linear', kernel_initializer="glorot_uniform")(inputs)
	x = Dense(Nouts, name='z_output', activation='relu', kernel_initializer="glorot_uniform")(x)
	x = Dense(1, name='z_wavg', activation='relu', kernel_initializer="glorot_uniform")(x)
	return x


#===============================================================================


# Build full model from multiple parts

inputs = Input(shape=(height, width, 1), name='image_inputs')
x = DefineEntryModel(inputs)
y = DefinePhiModel( x )
z = DefineZModel( x )

model = Model(inputs=inputs, outputs=[y, z])
model.summary()

weights = z.get_layer('z_wavg').get_weights()
print( weights )
print( len(weights) )
print( type(weights[0]) )
print( type(weights[1]) )
print( weights[0].shape )
print( weights[1].shape )

myweights = weights
myweights[1][0] = 0.123

XMIN = -10.0
XMAX = +10.0
NBINS = len(myweights[0])
BINSIZE = (XMAX-XMIN)/NBINS
for i in range(0, NBINS):
	 myweights[0][i][0] = XMIN + (i+0.5)*BINSIZE

model.get_layer('phi_wavg').set_weights(myweights)

print(model.get_layer('phi_wavg').get_weights())

#===============================================================================

