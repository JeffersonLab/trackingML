#!/usr/bin/env python
from __future__ import print_function, division

import tensorflow as tf
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.convolutional import MaxPooling2D

from keras import backend as K
from keras.layers import Input

from keras.preprocessing import image
from keras.layers import Input, Dense, Flatten, Dropout, Reshape, Concatenate
from keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

from keras.datasets import cifar10
import keras.backend as K

import matplotlib.pyplot as plt

import sys
import numpy as np

N_IMAGES = 100
GPUS = 1



def generate_noise(n_samples, noise_dim):
  X = np.random.normal(0, 1, size=(n_samples, noise_dim))
  return X

tags = ['img']

config = tf.ConfigProto()

#below to force cpu
#config = tf.ConfigProto(
#        device_count = {'GPU': 0}
#    )
print("generating "+str(N_IMAGES)+" images")
print("running with "+str(GPUS)+" gpus")
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.95

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

# GAN creation
model = load_model("test")


if GPUS<=1 :
    ganparallel_model = model
else:
    ganparallel_model = multi_gpu_model( model, gpus=GPUS )

# # Get training images



for epoch in range(N_IMAGES):
  noise_data = generate_noise(1, 100)  
  generated_images = ganparallel_model.predict([noise_data])

  print("===================")
  #print('\tEpoch: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch+1, cum_g_loss/num_batches, cum_d_loss/num_batches))
  #print(epoch+1)
  #print(epoch+1%10)
