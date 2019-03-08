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




if __name__ == "__main__":

    batch_size = 128
    num_classes = 11
    epochs = 24*2



    # Loading of data
    cwd = os.getcwd()
    filedir = '\\Keras\\mixed_npz_3219\\'
    X,y = utill.read_npz_folder(filedir)

    x_train, x_test, y_train, y_test = \
            train_test_split(X, y, test_size=.15, random_state=42)

    img_rows, img_cols = x_train.shape[1], x_train.shape[2] # Needs to be set to the dimensions of the Nsynth Data

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255 # Test with this and without
    x_test /= 255 # Test with this and without


    
    score=[0,0]
    model= Han_model(input_shape,num_classes)
    while score[1] < 0.68: 
        

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])

        early_stopping_monitor = EarlyStopping(patience=3)

        history=model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks=[early_stopping_monitor])
        score = model.evaluate(x_test, y_test, verbose=0)


        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


    model_name = input('Please enter the name of the model: ')

    # Plotting of accuracy
    utill.plot_accuracy(history,model_name=model_name)
    # Plotting of loss
    utill.plot_loss(history,model_name=model_name)

    model_json = model.to_json()
    with open(model_name +".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name+"_weights.h5")
    print("Saved model to disk")

    y_pred = model.predict(x_test)
    y_sl_true = utill.mutilabel2single(y_test)
    y_sl_pred = utill.mutilabel2single(y_pred)

    utill.plot_confusion_matrix(y_sl_true,y_sl_pred,utill.CLASS_NAMES,title = model_name)