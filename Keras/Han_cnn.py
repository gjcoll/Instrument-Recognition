'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Lambda
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.backend import spatial_2d_padding
from keras.backend import expand_dims
# import matplotlib
import utill
import os
import numpy as np


# def zero_padding(x):
#         return spatial_2d_padding(x)


batch_size = 128
num_classes = 11
epochs = 24*2




cwd = os.getcwd()
# the data, split between train and test sets
filedir = '\\Keras\\IRMAS_npzs_B\\'
# Should build a function for the npzloading of a whole folder
# cel_set,cel_label=utill.load_npz(cwd+'\\Keras\\IRMAS_npzs\\IRMAS_cel_A.npz')
# cla_set,cla_label=utill.load_npz(cwd+'\\Keras\\IRMAS_npzs\\IRMAS_cla_A.npz')
# flu_set,flu_label=utill.load_npz(cwd+'\\Keras\\IRMAS_npzs\\IRMAS_flu_A.npz')
# gac_set,gac_label=utill.load_npz(cwd+'\\Keras\\IRMAS_npzs\\IRMAS_gac_A.npz')
# gel_set,gel_label=utill.load_npz(cwd+'\\Keras\\IRMAS_npzs\\IRMAS_gel_A.npz')
# org_set,org_label=utill.load_npz(cwd+'\\Keras\\IRMAS_npzs\\IRMAS_org_A.npz')
# pia_set,pia_label=utill.load_npz(cwd+'\\Keras\\IRMAS_npzs\\IRMAS_pia_A.npz')
# sax_set,sax_label=utill.load_npz(cwd+'\\Keras\\IRMAS_npzs\\IRMAS_sax_A.npz')

X,y = utill.read_npz_folder(filedir)

# flu_label = flu_label*7
# X = np.append(cel_set,[cla_set,flu_set,gac_set,gel_set,org_set,pia_set,sax_set], axis = 0)
# y = np.append(cel_label, [cla_label,flu_label,gac_label,gel_label,org_label,pia_label,sax_label],axis = 0)

x_train, x_test, y_train, y_test = \
        train_test_split(X, y, test_size=.15, random_state=42)

# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# input image dimensions
img_rows, img_cols = x_train.shape[1], x_train.shape[2] # Needs to be set to the dimensions of the Nsynth Data

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

print(y_train.shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 # Test with this and without
x_test /= 255 # Test with this and without
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


print(y_test.shape)
# input_shape = (img_rows, 43, 1)

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


 model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)


print('Test loss:', score[0])
print('Test accuracy:', score[1])




# # Plotting of accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# # Plotting of loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()