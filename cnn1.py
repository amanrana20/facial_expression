import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import pandas as pd

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

URL_DATASET = '/Users/amanrana/Desktop/fer2013.csv'
dataset = pd.read_csv(URL_DATASET)

X_train = []

X_raw = dataset['pixels']
for row in X_raw:
    X_train.append(row.split(' '))


#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#print(pd.DataFrame(X_train).skew())


Y_train = np_utils.to_categorical(np.array(dataset['emotion']))
print len(Y_train)
X_train = (np.array(X_train)).reshape(len(Y_train), 1, 48, 48)

num_classes = 7

# Creating the model
model = Sequential()
model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 48, 48), activation='relu'))
model.add(Convolution2D(32, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 5, 5, border_mode='valid', activation='relu'))
model.add(Convolution2D(64, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))
print model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train[100:], Y_train[100:], validation_split=0.1, nb_epoch=10, batch_size=100, verbose=2)

# Evaluating the model
scores = model.evaluate(X_train[:100], Y_train[:100], verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

