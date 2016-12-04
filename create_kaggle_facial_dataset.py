import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os, cv2
from PIL import Image


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


URL_DATASET = 'FacialExpresionDataset/fer2013.csv'
PATH_BASE = 'Data'
EXT_TRAIN = 'train'
EXT_VAL = 'val'
dataset = pd.read_csv(URL_DATASET)
print dataset.head()
X_train = []


X_raw = dataset['pixels']
for row in X_raw:
    X_train.append(row.split(' '))


le = LabelEncoder()
Y_train = le.fit_transform(np.array(dataset['emotion']))
print len(Y_train)
X_train = (np.array(X_train)).reshape(len(Y_train), 48, 48)


# Maiking the base folder
if not os.path.exists(PATH_BASE):
    os.makedirs(PATH_BASE)
    os.makedirs(os.path.join(PATH_BASE, EXT_TRAIN))
    os.makedirs(os.path.join(PATH_BASE, EXT_VAL))

n = len(Y_train)
counter = 0


def write_img(EXT_train_or_val, img_name):
    # Goes into train
    img = np.array(Image.fromarray(x))
    write_path = os.path.join(PATH_BASE, EXT_train_or_val, '{}'.format(int(float(y))))
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    else:
        cv2.imwrite('{}/{}.jpg'.format(write_path, img_name), img)



def save_data(image_name, x, y, train_or_val):
    if train_or_val == 0:
        write_img(EXT_TRAIN, image_name)
    else:
        write_img(EXT_VAL, image_name)

counter_a = 1
counter_b = 2
counter = 0
# Making the datset
for i, x in enumerate(X_train):
    x = np.array(x).astype(np.float32)
    y = Y_train[i]
    
    if i < (0.8 * n):
        # Put into train folder
        save_data(counter_a, x, y, 0)
        counter_a += 1
    else:
        # Put into validation folder
        save_data(counter_b, x, y, 1)
        counter_b += 1
    counter = i

print counter
