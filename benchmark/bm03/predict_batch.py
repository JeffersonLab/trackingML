#!/usr/bin/env python

# import the necessary packages
from keras.models import load_model
import argparse
import pickle
import cv2
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to model to use")
ap.add_argument("-o", "--output", required=True,
	help="name of ouput csv")
ap.add_argument("-i", "--input", required=True,
	help="path to root directory of input images")
args = vars(ap.parse_args())

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=r"./training_set/",
    target_size=(472,696),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=args["input"],
    target_size=(472,696),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
#labels = pickle.loads(open("labels", "rb").read())

test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1,steps=test_generator.n)
print(pred)
predicted_class_indices=np.argmax(pred,axis=1)
#labels = (train_generator.class_indices)
print("labels")
labels = (train_generator.class_indices)
print(labels)
labelin=labels
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

goodCon=[]
badCon=[]
unknownCon=[]
cosmicCon=[]
ledCon=[]
for cons in pred:
    goodCon.append(cons[1])
    badCon.append(cons[0])
    unknownCon.append(cons[2])
    #cosmicCon.append(cons[1])
    #ledCon.append(cons[2])

#print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

filenames=test_generator.filenames
results=pd.DataFrame({"aFilename":filenames,
                      "bPredictions":predictions,
                      "cGoodConfiden":goodCon,
                      "dBadConfiden":badCon,
                      "eUnknownConfiden":unknownCon})
                      #"fcosmicConfiden":cosmicCon,
                      #"fledConfiden":ledCon})
results.to_csv(args["output"],index=False)
