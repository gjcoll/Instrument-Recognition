from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Lambda
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.backend import spatial_2d_padding
from keras.backend import expand_dims
import matplotlib.pyplot as plt
import utill
import os
import numpy as np


def Han_model(input_shape=(128,44,1),num_classes=11):

    model = Sequential()
    model.add(Lambda(spatial_2d_padding,input_shape=input_shape))

    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='linear',padding = 'same',
                    strides = 1))
    model.add(LeakyReLU(alpha = 0.33))

    model.add(Lambda(spatial_2d_padding))
    model.add(Conv2D(32, (3, 3), activation='linear',padding = 'same',strides = 1))
    model.add(LeakyReLU(alpha = 0.33))

    model.add(MaxPooling2D(pool_size=(3, 3),strides = (3,3)))
    model.add(Dropout(0.25))

    model.add(Lambda(spatial_2d_padding))
    model.add(Conv2D(64, (3,3),activation='linear',strides = 1,padding = 'same'))
    model.add(LeakyReLU(alpha = 0.33))

    model.add(Lambda(spatial_2d_padding))
    model.add(Conv2D(64, (3,3),activation='linear',strides = 1,padding = 'same'))
    model.add(LeakyReLU(alpha = 0.33))

    model.add(MaxPooling2D(pool_size=(3, 3),strides = (3,3)))
    model.add(Dropout(0.25))

    model.add(Lambda(spatial_2d_padding))
    model.add(Conv2D(128, (3,3),activation='linear',padding = 'same',strides = 1))
    model.add(LeakyReLU(alpha = 0.33))

    model.add(Lambda(spatial_2d_padding))
    model.add(Conv2D(128, (3,3),activation='linear',padding = 'same',strides = 1))
    model.add(LeakyReLU(alpha = 0.33))

    model.add(MaxPooling2D(pool_size=(3, 3),strides = (3,3)))
    model.add(Dropout(0.25))

    model.add(Lambda(spatial_2d_padding))
    model.add(Conv2D(256, (3,3),activation='linear',padding = 'same',strides = 1))
    model.add(LeakyReLU(alpha = 0.33))

    model.add(Lambda(spatial_2d_padding))
    model.add(Conv2D(256, (3,3),activation='linear',padding = 'same',strides = 1))
    model.add(LeakyReLU(alpha = 0.33))

    model.add(GlobalMaxPooling2D())

    model.add(Dense(1024, activation='linear'))
    model.add(LeakyReLU(alpha = 0.33))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model
