# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:15:18 2016

@author: Ulkesh
"""


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
import os
import numpy


from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

# dimensions of our images.
img_width, img_height = 48,48

train_data_dir = 'train'
validation_data_dir = 'val'
nb_classes = 7
nb_train_samples = 28703
nb_validation_samples = 7170
nb_epoch = 20


model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(1, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])
print(model.summary())

# this is the augmentation configuration we will use for training
#train_datagen = ImageDataGenerator(rescale=1./255, zca_whitening=True, rotation_range=90.0, 
#                                   horizontal_flip=True, vertical_flip=True)

train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        color_mode = "grayscale",
        batch_size=50,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        color_mode = "grayscale",
        batch_size=50,
        class_mode='categorical')

history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        verbose = 2,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
i=0
while 1:
    i = i+1
    if os.path.isfile(str(i) + '_try.h5') == False:
        model.save_weights(str(i) + '_try.h5')
        print (i, "try")
        break;
        



#import cv2
#
#im = cv2.imread('train/0/2.jpg')
#cv2.imshow("resized", im)
#cv2.waitKey(0)
#print(im.shape)
