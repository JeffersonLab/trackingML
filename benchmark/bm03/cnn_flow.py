#!/usr/bin/env python

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import TensorBoard
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import tensorflow as tf
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.models import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model



GPUS = 1


class SmallVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
 
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# (CONV => RELU) * 2 => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# (CONV => RELU) * 3 => POOL layer set
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		
        
        # first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
 
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
 
		# return the constructed network architecture
		return model

class myNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
 
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL layer set
		model.add(Conv2D(64, (2, 2), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))
		# (CONV => RELU) * 2 => POOL layer set
		model.add(Conv2D(64, (2, 2), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (2, 2), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# (CONV => RELU) * 3 => POOL layer set
		model.add(Conv2D(128, (2, 2), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		#model.add(Conv2D(256, (3, 3), padding="same"))
		#model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		#model.add(MaxPooling2D(pool_size=(4, 4)))
		#model.add(Dropout(0.25))
		
        
        # first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
 
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
 
		# return the constructed network architecture
		return model

class myNet2:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
 
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL layer set
		model.add(Conv2D(128, (2, 2), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))
		# (CONV => RELU) * 2 => POOL layer set
		model.add(Conv2D(128, (2, 2), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (2, 2), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# (CONV => RELU) * 3 => POOL layer set
		model.add(Conv2D(128, (2, 2), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		#model.add(Conv2D(256, (3, 3), padding="same"))
		#model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		#model.add(MaxPooling2D(pool_size=(4, 4)))
		#model.add(Dropout(0.25))
		
        
        # first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
 
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
 
		# return the constructed network architecture
		return model

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-lm", "--load-model", required=False,
	help="path to model to load for training")
ap.add_argument("-e", "--epochs", required=False,
	help="how many epochs to run")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
ap.add_argument("-g", "--gpus", required=False,
	help="number of GPUs to use (0 for none)")
args = vars(ap.parse_args())


###################################
# TensorFlow wizardry
#config = tf.ConfigProto()

#below to force cpu
#config = tf.ConfigProto(
#        device_count = {'GPU': 0}
#    )

# Don't pre-allocate memory; allocate as-needed
#config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
#config.gpu_options.per_process_gpu_memory_fraction = 0.95

# Create a session with the above options specified.
#K.tensorflow_backend.set_session(tf.Session(config=config))
###################################

#data_gen_args = dict(featurewise_center=True,
#                     featurewise_std_normalization=True,
#                     rotation_range=0,
#                     width_shift_range=0.,
#                     height_shift_range=0.,
#                     zoom_range=0)
#train_datagen = ImageDataGenerator(**data_gen_args)
#valid_datagen = ImageDataGenerator(**data_gen_args)
#test_datagen = ImageDataGenerator(**data_gen_args)

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# initialize our initial learning rate and # of epochs to train for
#INIT_LR = 0.01 # Learning rate
INIT_LR = 0.01 # Learning rate
EPOCHS = 50 # num epochs
if(args["epochs"] is not None):
	EPOCHS=int(args["epochs"])
if(args['gpus'] is not None):
	GPUS=int(args["gpus"])

BS=5

train_generator = train_datagen.flow_from_directory(
    directory=r"./training_set/",
    target_size=(472,696),
    color_mode="rgb",
    batch_size=BS,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_generator = valid_datagen.flow_from_directory(
    directory=r"./validation_set/",
    target_size=(472,696),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=r"./TEST_set/",
    target_size=(472,696),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)
 
# initialize our VGG-like Convolutional Neural Network
#model = SmallVGGNet.build(width=696, height=472, depth=3,classes=4)

if GPUS !=1 : tf.device('/cpu:0')
input_tensor = Input( shape=(472,696,3) )
model = InceptionV3(include_top=True, weights=None, input_shape=None, input_tensor=input_tensor, pooling=None, classes=3)
#model = Sequential()
#model.add(Activation("relu", input_shape=(height, width,1)))
#model.add(Conv2D(128, (3,3), activation="relu", kernel_initializer="glorot_normal", strides=(1,1), input_shape=(472,696,3), padding="same", data_format="channels_last") )
#model.add(Conv2D(128, (3,3), activation="relu", kernel_initializer="glorot_normal", strides=(1,1), input_shape=(472,696,3), padding="same", data_format="channels_last") )
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
#model.add(Flatten())
#model.add(Dense(int(Nouts/5), activation='relu'))
#model.add(Dense(int(Nouts/5), activation='relu'))
#model.add(Dense(int(Nouts/5), activation='relu'))
#model.add(Dense(int(Nouts/5), activation='relu'))
#model.add(Dense(3, activation='softmax'))
#model.add(BatchNormalization())
#model = SmallVGGNet.build(width=696, height=472, depth=3,classes=3)
if(args["load_model"] is not None):
	print("LOADING MODEL")
	model = load_model(args["load_model"])

if GPUS<=1 :
	parallel_model = model
else:
	parallel_model = multi_gpu_model( model, gpus=GPUS )

parallel_model.summary()

 
# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt =Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#opt = Adagrad(lr=0.01, epsilon=None, decay=0.0)
#opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
parallel_model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

tensorboard=TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# train the neural network
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
H = parallel_model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_generator, validation_steps=STEP_SIZE_VALID, epochs=EPOCHS,callbacks=[tensorboard])

# evaluate the network
print("[INFO] evaluating network...")

parallel_model.evaluate_generator(generator=valid_generator, verbose=1, steps=STEP_SIZE_VALID)

test_generator.reset()
pred=parallel_model.predict_generator(test_generator,verbose=1,steps=test_generator.n)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
#print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
 


# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
print(labels)
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(labels))
f.close()

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)
