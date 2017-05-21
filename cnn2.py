import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import pandas as pd

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

URL_DATASET = 'FacialExpresionDataset/fer2013.csv'

def get_dataset():
    dataset = pd.read_csv(URL_DATASET)

    X_train = []

    X_raw = dataset['pixels']
    for row in X_raw:
        X_train.append(row.split(' '))

    Y_train = np_utils.to_categorical(np.array(dataset['emotion']))
    print 'Number of examples in total:', len(Y_train), '\n'
    X_train = (np.array(X_train)).reshape(len(Y_train), 1, 48, 48)

    return X_train, Y_train

num_classes = 7

# Creating the model
inp = Input(shape=(1, 48, 48), name='Input')

conv_1 = Convolution2D(64, 3, 3, border_mode='same', activation='relu', name='conv_1')(inp)
conv_2 = Convolution2D(64, 3, 3, border_mode='same', activation='relu', name='conv_2')(conv_1)
max_pool_1 = MaxPooling2D(pool_size=(2, 2), name='max_pool_1')(conv_2)

conv_3 = Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='conv_3')(max_pool_1)
conv_4 = Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='conv_4')(conv_3)
max_pool_2 = MaxPooling2D(pool_size=(2, 2), name='max_pool_2')(conv_4)

conv_5 = Convolution2D(256, 3, 3, border_mode='same', activation='relu', name='conv_5')(max_pool_2)
conv_6 = Convolution2D(256, 3, 3, border_mode='same', activation='relu', name='conv_6')(conv_5)
max_pool_3 = MaxPooling2D(pool_size=(2, 2), name='max_pool_3')(conv_6)

conv_7 = Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='conv_7')(max_pool_3)
conv_8 = Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='conv_8')(conv_7)
max_pool_4 = MaxPooling2D(pool_size=(2, 2), name='max_pool_4')(conv_8)

dense_1 = Flatten()(max_pool_4)
dense_2 = Dense(128, activation='relu')(dense_1)
out = Dense(num_classes, activation='sigmoid')(dense_2)
model = Model(input=inp, output=out)
print model.summary()

# Getting thr dataset
X_train, Y_train = get_dataset()

# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train[100:], Y_train[100:], validation_split=0.1, nb_epoch=10, batch_size=100)

# Evaluating the model
scores = model.evaluate(X_train[:100], Y_train[:100], verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
