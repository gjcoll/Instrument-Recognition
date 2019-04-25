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

from keras_model_options import Han_model
#############
# Develop a class based system for robust programming and easy implimentation.
###########

def train_untill(model_func, training_data, validation_data, input_shape, num_classes=11, min_loss:float = 0.55, patience:int = 3):
    
    batch_size = 128 # training batch sizes
    num_classes = 11 # number of classes to measure
    epochs = 24*2 # Maximum number of epochs
    score=[0,0]
    x_train, y_train = training_data[0], training_data[1]
    x_val, y_val = validation_data[0], validation_data[1]



    while score[1] < min_loss: 
        
        model= model_func(input_shape,num_classes)

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])

        early_stopping_monitor = EarlyStopping(patience= patience)

        history=model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping_monitor])
        score = model.evaluate(x_val, y_val, verbose=0)


        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    return model, history

def train_model(model_func, training_data_dir: str, additional_data: list = None):
    

    X, y  = utill.read_npz_folder(training_data_dir) 
    x_train, x_val, y_train, y_val = \
            train_test_split(X, y, test_size=.15)

    if not(additional_data is None):
        for add_data in additional_data:
            X_add, y_add =utill.read_npz_folder(add_data)
            x_train=np.append(x_train,X_add,axis = 0)
            y_train=np.append(y_train,y_add,axis = 0)

    img_rows, img_cols = x_train.shape[1], x_train.shape[2] # Needs to be set to the dimensions of the IRMAS dataset
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    # x_train /= 255 # Test with this and without
    # x_test /= 255 # Test with this and without
    model, history = train_untill(model_func, (x_train,y_train),(x_val,y_val),input_shape)
    return model, history, x_val, y_val


if __name__ == "__main__":
    # Loading of data
    cwd = os.getcwd()
    filedir = 'Keras\\IRMAS_npzs_C\\'
    model,history,x_test,y_test = train_model(Han_model,filedir,['Keras\\Nsynth\\'])


    model_name = input('Please enter the name of the model: ')

    # Plotting of accuracy
    utill.plot_accuracy(history,model_name=model_name)
    # Plotting of loss
    utill.plot_loss(history,model_name=model_name)
    y_pred = model.predict(x_test)
    y_sl_true = utill.mutilabel2single(y_test)
    y_sl_pred = utill.mutilabel2single(y_pred)

    utill.plot_confusion_matrix(y_sl_true,y_sl_pred,utill.CLASS_NAMES,title = model_name)

    answer = input('Would you like to save the model? [y/n]')
    answer=answer.lower()
    while (answer != 'y') and (answer != 'n'):
        answer = input('Not an acceptable answer \n\tWould you like to save the model? [y/n]')

    if answer== 'y':
        model_json = model.to_json()
        with open(model_name +".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_name+"_weights.h5")
        print("Saved model to disk")

