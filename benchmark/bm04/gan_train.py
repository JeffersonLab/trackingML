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

BATCH_SIZE = 25
N_EPOCHS = 100
GPUS = 1

def get_generator(input_layer):
  print("generator model")
  print(input_layer.shape)
  #print(condition_layer.shape)

  #merged_input = Concatenate()([input_layer, condition_layer])
  
  hid = Dense(int(128 * img_height * img_width/4), activation='relu')(input_layer)#(merged_input)
  hid = BatchNormalization(momentum=0.9)(hid)
  hid = LeakyReLU(alpha=0.1)(hid)
  hid = Reshape((int(img_height/2), int(img_width/2), 128))(hid)

  hid = Conv2D(128, kernel_size=4, strides=1,padding='same')(hid)
  hid = BatchNormalization(momentum=0.9)(hid)    
  hid = LeakyReLU(alpha=0.1)(hid)

  hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
  hid = BatchNormalization(momentum=0.9)(hid)
  hid = LeakyReLU(alpha=0.1)(hid)

  #hid = Conv2D(256, kernel_size=5, strides=1,padding='same')(hid)
  #hid = BatchNormalization(momentum=0.9)(hid)    
  #hid = LeakyReLU(alpha=0.1)(hid)

  #hid = Conv2D(256, kernel_size=5, strides=1,padding='same')(hid)
  #hid = BatchNormalization(momentum=0.9)(hid)    
  #hid = LeakyReLU(alpha=0.1)(hid)

  #hid = Conv2D(256, kernel_size=5, strides=1, padding='same')(hid)
  #hid = BatchNormalization(momentum=0.9)(hid)
  #hid = LeakyReLU(alpha=0.1)(hid)

  hid = Conv2D(256, kernel_size=2, strides=1, padding='same')(hid)
  hid = BatchNormalization(momentum=0.9)(hid)
  hid = LeakyReLU(alpha=0.1)(hid)
                      
  
  hid = Conv2D(3, kernel_size=5, strides=1, padding="same")(hid)
  hid = Flatten()(hid)
  hid = Dense(img_height*img_width*img_channels)(hid)

  hid = Reshape((img_height, img_width, img_channels))(hid)

  out = Activation("tanh")(hid)

  print(out.shape)
  model = Model(inputs=[input_layer], outputs=out)
  #model = Model(inputs=[input_layer, condition_layer], outputs=out)
  model.summary()

  if GPUS<=1 :
    parallel_model = model
  else:
    parallel_model = multi_gpu_model( model, gpus=GPUS )
  
  return parallel_model

def get_discriminator(input_layer):
  
  depth = 128
  dropout = 0.1
  #model = InceptionV3(include_top=True, weights=None, input_tensor=input_layer, input_shape=None, pooling=None, classes=1)
  model=Sequential()
  input_shape = (img_height, img_width, img_channels)

  model.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(dropout))

  model.add(Conv2D(depth*2, 5, strides=2, padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(dropout))

  model.add(Conv2D(depth*4, 5, strides=2, padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(dropout))

  model.add(Conv2D(depth*8, 5, strides=1, padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(dropout))

  # Out: 1-dim probability
  model.add(Flatten())
  model.add(Dense(256))
  #model.add(Dense(128))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  model.summary()
  if GPUS<=1 :
    discparallel_model = model
  else:
    discparallel_model = multi_gpu_model( model, gpus=GPUS )
  return discparallel_model

def one_hot_encode(y):
  z = np.zeros((len(y), 10))
  idx = np.arange(len(y))
  z[idx, y] = 1
  return z

def generate_noise(n_samples, noise_dim):
  X = np.random.normal(0, 1, size=(n_samples, noise_dim))
  return X

def generate_random_labels(n):
  y = np.random.choice(10, n)
  y = one_hot_encode(y)
  return y

tags = ['img']

def get_image_batch():
  img_batch = real_image_generator.next()
  #print(img_batch)
  #plt.imshow(img_batch[0])
  #plt.show()

  # keras generators may generate an incomplete batch for the last batch in an epoch of data
  if len(img_batch) != BATCH_SIZE:
    img_batch = real_image_generator.next()

  assert img_batch.shape == (BATCH_SIZE, img_height, img_width, img_channels), img_batch.shape
  return img_batch
  
def show_samples(batchidx):
  #fig, axs = plt.subplots(5, 6, figsize=(img_width))
  #plt.subplots_adjust(hspace=0.3, wspace=0.1)
  #fig, axs = plt.subplots(5, 6)
  #fig.tight_layout()
  #print("loop")

  noise_data = generate_noise(BATCH_SIZE, 100)
  print(noise_data.shape)
  random_labels = generate_random_labels(BATCH_SIZE)
  # We use same labels for generated images as in the real training batch
  gen_imgs = generator.predict([noise_data])
  #for classlabel in range(1):
  #  row = int(classlabel / 2)
  #  coloffset = (classlabel % 2) * 3
  #  lbls = np.ones(1) #one_hot_encode([classlabel] * 3)
  #  noise = generate_noise(3, 100)
  #  print("make image")
  #  gen_imgs = generator.predict([noise, lbls])

  img = image.array_to_img(gen_imgs[0], scale=True)
  plt.imshow(img)
  plt.draw()
  plt.savefig("out_imgs/"+batchidx+".png")
  plt.close()

config = tf.ConfigProto()

#below to force cpu
#config = tf.ConfigProto(
#        device_count = {'GPU': 0}
#    )
print("running "+str(N_EPOCHS)+" Epochs")
print("using batch size of "+str(BATCH_SIZE))
print("running with "+str(GPUS)+" gpus")
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.95

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))


img_height=64 #472
img_width=64 #696
img_channels=3


# GAN creation
img_input = Input(shape=(img_height,img_width,img_channels))

discriminator = get_discriminator(img_input)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False

noise_input = Input(shape=(100,))
generator = get_generator(noise_input)
#generator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

gan_input = Input(shape=(100,))
#X_train = generator(gan_input)
#gan_out = discriminator(X_train)
#gan = Model(inputs=[gan_input], output=gan_out)
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
gan.summary()
if GPUS<=1 :
    ganparallel_model = gan
else:
    ganparallel_model = multi_gpu_model( gan, gpus=GPUS )
ganparallel_model.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# # Get training images
data_generator = ImageDataGenerator(rescale=1./255)

flow_from_directory_params = {'target_size': (img_height, img_width),
                              'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                              'class_mode': None,
                              'batch_size': BATCH_SIZE}

real_image_generator = data_generator.flow_from_directory(
        directory="./training_set",
        **flow_from_directory_params
)

 
num_batches = int(real_image_generator.n//real_image_generator.batch_size)


for epoch in range(N_EPOCHS):
  dloss=[0.,0.]
  aloss=[0.,0.]
  for batch_idx in range(num_batches):
    # Get the next set of real images to be used in this iteration
    images = get_image_batch()# X_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
    plt.imshow(images[0])
    #plt.show()
    labels = np.ones(BATCH_SIZE)#  y_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]


    noise_data = generate_noise(BATCH_SIZE, 100)
    #print(noise_data.shape)
    # We use same labels for generated images as in the real training batch
    generated_images = generator.predict([noise_data])

    
    #images_train = images[np.random.randint(0,images.shape[0], size=BATCH_SIZE), :, :, :]
    x = np.concatenate((images, generated_images))
    y = np.ones([2*BATCH_SIZE, 1])
    y[BATCH_SIZE:, :] = 0
    d_loss = discriminator.train_on_batch(x, y)
    dloss[0]+=d_loss[0]
    dloss[1]+=d_loss[1]
    y = np.ones([BATCH_SIZE, 1])
    noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, 100])
    a_loss = gan.train_on_batch(noise, y)
    aloss[0]+=a_loss[0]
    aloss[1]+=a_loss[1]
  log_mesg = "%d: [D loss: %f, acc: %f]" % (epoch, dloss[0], dloss[1]/num_batches)
  log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, aloss[0], aloss[1]/num_batches)
  print(log_mesg)
  #print("===================")
  #print('\tEpoch: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch+1, cum_g_loss/num_batches, cum_d_loss/num_batches))
  #print(epoch+1)
  #print(epoch+1%10)
  if((epoch+1) % 10 == 0):
    print("SHOW")
    show_samples("epoch" + str(epoch))

generator.save("test")
