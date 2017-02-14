import csv
import numpy as np
import math
import cv2
import json

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import BatchNormalization, Convolution2D, Dense, Input, Activation, Flatten, Dropout


X_train = []
y_train = []

###
# Get photo addresses and steering datas from datas in 'driving_log.csv'.
# And save datas in lists. X_train save photo addresses, y_train save steering datas.
###
with open('driving_log.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        center = row['center']
        X_train.append(center)
        y_train.append(float(row['steering']))
        left = row['left'].strip()
        X_train.append(left)
        y_train.append(float(row['steering']))
        right = row['right'].strip()
        X_train.append(right)
        y_train.append(float(row['steering']))
X_train = np.array(X_train)
y_train = np.array(y_train)
print('finish get input...')

###
# shuffle X_train list and y_train list.
###
X_train, y_train = shuffle(X_train, y_train)
print('finish shuffle...')

###
# split train and valiation set.
###
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
print('finish split...')

###
# generator function : There are so many pictures so heap space can't handle all, so I use python generator function.
###
def generate_arrays_from_list(Xs,Ys,batch_size = 128):
    # batch_train set consist of 4 dim array. (batch_size, height, width, RGB)
    batch_train = np.zeros((batch_size , 66, 200, 3), dtype= np.float32)
    # batch_angle set consist of 2 dim array. (batch_size, steering)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    n= 0
    while True:
        for i in range(batch_size):
            if n == len(Xs):
                n = 0
                batch_train2 = np.zeros((i , 66, 200, 3), dtype= np.float32)
                batch_angle2 = np.zeros((i,), dtype= np.float32)
                # this part handle last batch set.
                for j in range(i):
                    batch_train2[j] = batch_train[j]
                    batch_angle2[j] = batch_angle[j]
                yield batch_train2, batch_angle2
                break
            else:
                # get image from Xs's addresses.
                img = cv2.imread(Xs[n])
                # resize image for NVIDIA model's input.
                img = cv2.resize(img, (200, 66))
                arr = np.array(img, dtype = np.float32).flatten()
                batch_train[i] = np.array(img, dtype = np.float32)
                batch_angle[i] = Ys[n]
                n= n + 1
                if i == (batch_size-1):
                    yield batch_train, batch_angle

###
# Build the NVIDIA model. It consist of 1 normalization layer, 5 convolution layers, 3 fully connected layers and output layer. Also it has 6 dropout layers.
###
model = Sequential()
model.add(BatchNormalization(input_shape=(66,200,3), axis=1))
model.add(Convolution2D(24,5,5, activation='relu',border_mode='valid',subsample=(2,2),input_shape = (66,200,3)))
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5, activation='relu',border_mode='valid',subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5, activation='relu',border_mode='valid',subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation='relu',border_mode='valid',subsample=(1,1)))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation='relu',border_mode='valid',subsample=(1,1)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1164,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(10,activation = 'relu'))
model.add(Dense(1,activation = 'linear'))
# This model use adam optimizer and mean square loss function.
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
model.summary()
#For using python generator, I use Keras fit_generator function.
model.fit_generator(generate_arrays_from_list(X_train,y_train),samples_per_epoch=len(X_train), nb_epoch = 5,validation_data=generate_arrays_from_list(X_valid,y_valid), nb_val_samples=len(X_valid))
print('finish learning...')

###
# save the model.
###
model.save('./model.h5')
print('finish saving files...')
