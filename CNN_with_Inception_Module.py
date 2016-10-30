import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import cv2
import os

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam


# Some parameters
PATH_DATASET = 'Dataset/'
NB_EPOCH = 30
BATCH_SIZE = 200
VALIDATION_SPLIT = 0.3
VERBOSE = 2
TEST_TRAIN_SPLIT = 0.1


def get_data(test_train_split):
    dataset = []
    
    for roots, dirs, files in os.walk(PATH_DATASET):
        for file in files:
            if file[-4:] == '.jpg':
                path = '{}Gesture_{}/{}'.format(PATH_DATASET, file[0], file)
                pic = (cv2.imread(path)).astype(int)
                dataset.append((pic, int(file[0])))

    filter(None, dataset)

    # Shuffling and splitting the dataset into training, validation and testing
    np.random.shuffle(dataset)
    
    x = []
    y = []

    for i in range(len(dataset)):
        print dataset[i]
        try:
            item = list(dataset[i])
            if np.any(None, item) != True:
                x.append(item[0])
                y.append(item[1])
            else:
                continue
        except:
            continue

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_train_split)

    return X_train, Y_train, X_test, Y_test


def create_model():
    inp = Input(shape=(3, 224, 224), name='Input')
    conv_1 = Convolution2D(32, 3, 3, border_mode='same', activation='relu', name='conv_1')(inp)
    conv_2 = Convolution2D(32, 3, 3, border_mode='same', activation='relu', name='conv_2')(conv_1)

    # Inception Module 1
    max_pool_1 = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_1')(conv_2)
    inception1_conv1_1x1 = Convolution2D(32, 1, 1, border_mode='same', activation='relu', name='inception1_conv1_1x1')(max_pool_1)

    inception1_conv2_3x3 = Convolution2D(32, 3, 3, border_mode='same', activation='relu', name='inception1_conv2_3x3')(inception1_conv1_1x1)

    inception1_conv3_5x5 = Convolution2D(32, 5, 5, border_mode='same', activation='relu', name='inception1_conv3_5x5')(inception1_conv1_1x1)

    inception1_merge = merge([inception1_conv1_1x1, inception1_conv2_3x3, inception1_conv3_5x5], mode='concat', concat_axis=1, name='inception1_merge')

    #Inception Module 2
    max_pool_2 = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_2')(inception1_merge)
    inception2_conv1_1x1 = Convolution2D(32, 1, 1, border_mode='same', activation='relu', name='inception2_conv1_1x1')(max_pool_2)
    
    inception2_conv2_3x3 = Convolution2D(32, 3, 3, border_mode='same', activation='relu', name='inception2_conv2_3x3')(inception2_conv1_1x1)
    
    inception2_conv3_5x5 = Convolution2D(32, 5, 5, border_mode='same', activation='relu', name='inception2_conv3_5x5')(inception2_conv1_1x1)
    
    inception2_merge = merge([inception2_conv1_1x1, inception2_conv2_3x3, inception2_conv3_5x5], mode='concat', concat_axis=1, name='inception2_merge')


    #Inception Module 3
    max_pool_3 = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_3')(inception2_merge)
    inception3_conv1_1x1 = Convolution2D(32, 1, 1, border_mode='same', activation='relu', name='inception3_conv1_1x1')(max_pool_3)
    
    inception3_conv2_3x3 = Convolution2D(32, 3, 3, border_mode='same', activation='relu', name='inception3_conv2_3x3')(inception3_conv1_1x1)
    
    inception3_conv3_5x5 = Convolution2D(32, 5, 5, border_mode='same', activation='relu', name='inception3_conv3_5x5')(inception3_conv1_1x1)
    
    inception3_merge = merge([inception3_conv1_1x1, inception3_conv2_3x3, inception3_conv3_5x5], mode='concat', concat_axis=1, name='inception3_merge')
    
    #Inception Module 4
    max_pool_4 = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_4')(inception3_merge)
    inception4_conv1_1x1 = Convolution2D(32, 1, 1, border_mode='same', activation='relu', name='inception4_conv1_1x1')(max_pool_4)
    
    inception4_conv2_3x3 = Convolution2D(32, 3, 3, border_mode='same', activation='relu', name='inception4_conv2_3x3')(inception4_conv1_1x1)
    
    inception4_conv3_5x5 = Convolution2D(32, 5, 5, border_mode='same', activation='relu', name='inception4_conv3_5x5')(inception4_conv1_1x1)
    
    inception4_merge = merge([inception4_conv1_1x1, inception4_conv2_3x3, inception4_conv3_5x5], mode='concat', concat_axis=1, name='inception4_merge')
    
    #Inception Module 5
    max_pool_5 = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_5')(inception4_merge)
    inception5_conv1_1x1 = Convolution2D(32, 1, 1, border_mode='same', activation='relu', name='inception5_conv1_1x1')(max_pool_5)
    
    inception5_conv2_3x3 = Convolution2D(32, 3, 3, border_mode='same', activation='relu', name='inception5_conv2_3x3')(inception5_conv1_1x1)
    
    inception5_conv3_5x5 = Convolution2D(32, 5, 5, border_mode='same', activation='relu', name='inception5_conv3_5x5')(inception5_conv1_1x1)
    
    inception5_merge = merge([inception5_conv1_1x1, inception5_conv2_3x3, inception5_conv3_5x5], mode='concat', concat_axis=1, name='inception5_merge')

    #Fully Connected Layers
    max_pool_6 = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_6')(inception5_merge)
    flat_1 = Flatten()(max_pool_6)
    hidden_1 = Dense(64, activation='relu', name='hidden_2')(flat_1)
    out = Dense(7, activation='softmax', name='out')(hidden_1)

    model = Model(input=inp, output=out)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


# Getting the data
X_train, Y_train, X_test, Y_test = get_data(TEST_TRAIN_SPLIT)
print 'here'
# Creating the model
model = create_model()
model.summary()

# Training the model
model.fit(X_train, Y_train, validation_split=VALIDATION_SPLIT, nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE)

# Evaluating the model
scores = model.evaluate(X_test, Y_test, verbose=2)
print("\nCNN Accuracy: %.2f%%" % (scores[1]*100))
