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
from keras.models import Model
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
from keras import losses
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.recurrent import LSTM, LSTMCell
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.layers import Input


class myNet:
    @staticmethod
    def build(inparm, outparm):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        inputs = Input(shape=(inparm,))
        #x = Dense(64, activation='tanh')(inputs)
        x = Dense(512, activation='tanh')(inputs) 
        x = Dense(512, activation='tanh')(x)
        x = Dense(512, activation='tanh')(x)
        x = Dense(512, activation='tanh')(x)
        x = Dense(512, activation='tanh')(x)
        x = Dense(512, activation='tanh')(x)
        x = Dense(512, activation='tanh')(x)
        x = Dense(512, activation='tanh')(x)
        x = Dense(512, activation='tanh')(x)
        x = Dense(512, activation='tanh')(x)
        x= Dropout(.2)(x)
        predictions = Dense(outparm, activation='linear')(x)

        model = Model(inputs=inputs, outputs=predictions)
        if GPUS<=1 :
            parallel_model = model
        else:
            parallel_model = multi_gpu_model( model, gpus=GPUS )
        return parallel_model

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=False,
    help="path to output trained model")
ap.add_argument("-lm", "--load-model", required=False,
    help="path to model to load for training")
ap.add_argument("-d", "--data", required=False,
    help="Name of directory containing the data sets")
ap.add_argument("-e", "--epochs", required=False,
    help="how many epochs to run")
ap.add_argument("-l", "--label-bin", required=False,
    help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=False,
    help="path to output accuracy/loss plot")
args = vars(ap.parse_args())


###################################
# TensorFlow wizardry
config = tf.ConfigProto()

#below to force cpu
#config = tf.ConfigProto(
#        device_count = {'GPU': 0}
#    )

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.95

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))
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
#test_datagen = ImageDataGenerator(rescale=1./255)

# initialize our initial learning rate and # of epochs to train for
#INIT_LR = 0.01 # Learning rate
INIT_LR = 0.01 # Learning rate
EPOCHS = 5 # num epochs
if(args["epochs"] is not None):
    EPOCHS=int(args["epochs"])

BS=100000
GPUS=1

inDATA=[] #np.array([]) #np.array()
outDATA=[]

with open("mlin.dat") as fp, open("mlout.dat") as fpo:  
    line = fp.readline()
    lineo = fpo.readline()
    inparams=len(line.strip().split(",")[:])+1
    outparams=len(lineo.strip().split(",")[:])-1
    #for buckets in range(0,inparams):
    #    inDATA.append([])
    #print(inDATA)
    while line and lineo:
        #listv=[]
        for pl in line.strip().split(",")[:]:
            inDATA.append(float(pl))

        ind=0
        for plo in lineo.strip().split(",")[:]:
            if ind==0:
                inDATA.append(float(plo))
            else:
                outDATA.append(float(plo))
            ind=ind+1

        line= fp.readline()
        lineo=fpo.readline()


inDATA=np.reshape(inDATA,(-1,inparams))
#inDATA.reshape(21,)

#with open("mlout.dat") as fp:  
#    line = fp.readline()
#    outparams=len(line.strip().split(","))
#    while line:
#        for pl in line.strip().split(","):
#            outDATA.append(float(pl))
#        #outDATA.append(line.strip().split(","))
#        line= fp.readline()

outDATA=np.reshape(outDATA,(-1,outparams))

#print(inDATA)
#print(outDATA)
#exit(1)

# initialize our VGG-like Convolutional Neural Network
#model = SmallVGGNet.build(width=696, height=472, depth=3,classes=4)
model = myNet.build(inparm=inparams, outparm=outparams)
# this could also be the output a different Keras model or layer

#input_tensor = Input(shape=imgshape)

if(args["load_model"] is not None):
    print("LOADING MODEL")
    model = load_model(args["load_model"])

model.summary()

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
#opt =Adadelta(lr=100.0, rho=0.95, epsilon=None, decay=0.0)
#opt = Adagrad(lr=0.01, epsilon=None, decay=0.0)
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="MSE", optimizer=opt)

tensorboard=TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=BS, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# train the neural network
#STEP_SIZE_TRAIN=train_generator.n/train_generator.batch_size
#STEP_SIZE_VALID=valid_generator.n/valid_generator.batch_size

H= model.fit(x=inDATA, y=outDATA, batch_size=BS, epochs=EPOCHS, verbose=1, callbacks=None, validation_split=0.25, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
#H = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_generator, validation_steps=STEP_SIZE_VALID, epochs=EPOCHS,callbacks=[tensorboard])

print("PREDICTION~~~~~~~~~~")
model.predict(inDATA,batch_size=1)

plotname="plot"

if(args["plot"]):
    plotname=args["plot"]


# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(plotname)

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")

model_name=args["data"]

if(args["model"]):
    model_name=args["model"]

model.save(model_name)


