import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


train_dataset_path=r"C:\Users\Ariya Rayaneh\Desktop\mask"
width=height=224
INIT_LR = 1e-4
EPOCHS = 250
BS = 32

idg=ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    brightness_range=(0.8,1.2),
    zoom_range=0.15,
    shear_range=0.3,
    rotation_range=20,
    validation_split=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="nearest"
)


train_data=idg.flow_from_directory(
    train_dataset_path,
    target_size=(width,height),
    class_mode='categorical',
    batch_size=BS,
    subset='training'
)

val_data=idg.flow_from_directory(
    train_dataset_path,
    target_size=(width,height),
    class_mode='categorical',
    batch_size=BS,
    subset='validation'
)

class_names = ['correct_mask_wearing', 'incorrect_mask_wearing']

baseModel = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling='avg',
    input_tensor=Input(shape=(224, 224, 3)))

model = tf.keras.Sequential([
    baseModel,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])


for layer in baseModel.layers:
	layer.trainable = False

opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


H = model.fit(
    train_data,
	steps_per_epoch=len(train_data) // BS,
	validation_data=val_data,
	validation_steps=len(val_data) // BS,
	epochs=EPOCHS)


model.save(r'C:\Users\Ariya Rayaneh\Desktop\my_model_new.h5',save_format="h5")

new_model = tf.keras.models.load_model(r'C:\Users\Ariya Rayaneh\Desktop\mask_detector (1).model')
























