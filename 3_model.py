# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 19:02:31 2016
@author: syamprasadkr
"""
# This is a program to train a CNN model to predict facial expressions

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import cv2
import numpy as np
import theano
import os
from keras import backend as K
K.set_image_dim_ordering('th')

# Fix seed for reproducibility
seed = 7
np.random.seed(seed)

# Set directory paths
PATH = 'data'
EXT1 = 'train'
EXT2 = 'val'
EXT3 = 'test'
PATH1 = os.path.join(PATH, EXT1)
PATH2 = os.path.join(PATH, EXT2)
PATH3 = os.path.join(PATH, EXT3)

img_width, img_height = 48, 48
#kaggle dataset
nb_train_samples = 28703
nb_validation_samples = 7170

# Number of training epochs
nb_epoch = 20 

#CNN model
model = Sequential()
model.add(Convolution2D(64, 3, 3, input_shape = (1, img_width, img_height)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Convolution2D(128, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Convolution2D(256, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Convolution2D(512, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Image augmentation to make the training more robust
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)
                                 
test_datagen = ImageDataGenerator(rescale = 1./255)

# Load data in batches to train and validate
train_generator = train_datagen.flow_from_directory(PATH1,
                                                    target_size = (img_width, img_height),
                                                    color_mode = 'grayscale',                                                    
                                                    batch_size = 100,
                                                    class_mode = 'categorical')  
                                    
validation_generator = test_datagen.flow_from_directory(PATH2,
                                                    target_size = (img_width, img_height),
                                                    color_mode = 'grayscale',                                                    
                                                    batch_size = 100,
                                                    class_mode = 'categorical')                                    
                
# Perform the training (fits the model on data)
history = model.fit_generator(train_generator,
                    samples_per_epoch = nb_train_samples,
                    nb_epoch = nb_epoch,
                    validation_data = validation_generator,
                    nb_val_samples = nb_validation_samples)

# Save the weights to be used in inference model
model.save_weights('3_model.h5')





