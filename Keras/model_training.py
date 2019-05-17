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
from model_testing import main_model_test
import matplotlib.pyplot as plt
import utill
import os
import numpy as np

from keras_model_options import Han_model

CWD = os.getcwd()
if CWD.split('\\')[-1] != 'Keras':
    CWD = os.path.join(CWD,'Keras')

#############
# Develop a class based system for robust programming and easy implimentation.
###########
class Model_training():

    def __init__(self,model_func,batch_size = 128,num_classes = 11, max_epochs = 48, patience = 3,min_acc:float = 0.55,multilable = False):
        # model set up
        self.batch_size = batch_size # number of batches per epoch
        self.num_classes = num_classes # number of classes to measure
        self.epochs = max_epochs # maximum number of epochs

        # model function
        self.model_func = model_func
        self.patience = patience
        self.min_acc = min_acc
        self.multilable = multilable



    def train_untill(self, training_data, validation_data, input_shape):

        score=[0,0]
        x_train, y_train = training_data[0], training_data[1]
        x_val, y_val = validation_data[0], validation_data[1]

        while score[1] < self.min_acc: 
            
            model= self.model_func(input_shape,num_classes = self.num_classes)
            if self.multilable:
                model.compile(loss=keras.losses.binary_crossentropy,
                            optimizer=keras.optimizers.Adam(),
                            metrics=['accuracy'])
            else:
                model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adam(),
                            metrics=['accuracy'])

            early_stopping_monitor = EarlyStopping(patience= self.patience)

            history=model.fit(x_train, y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=1,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping_monitor])
            score = model.evaluate(x_val, y_val, verbose=0)


            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
        return model, history

    def train_model(self, training_data_dir: str, additional_data: list = None, augment_data = False, complex_reader=False,RandomState=None):
        
        if not complex_reader:
            X, y  = utill.read_npz_folder(training_data_dir)
        else:
            X, y = utill.unpack_npz_folder(training_data_dir)
            
        X_list = []
        y_list = []
        for xx,yy in zip(X,y):
            xx,yy = utill.train_split_spec(xx,yy)
            X_list.append(xx)
            y_list.append(yy)

        X = np.concatenate(X_list)
        del X_list
        y = np.concatenate(y_list)
        del y_list

        x_train, x_val, y_train, y_val = \
                train_test_split(X, y, test_size=.15)
        del X
        del y
        
        if not(additional_data is None):
            for add_data in additional_data:
                X_add, y_add =utill.read_npz_folder(add_data)
                x_train=np.append(x_train,X_add,axis = 0)
                y_train=np.append(y_train,y_add,axis = 0)

        if augment_data:
            x_train = [utill.drop_timenfreq(x,RandomState=RandomState) for x in x_train]

        x_train = np.asarray(x_train)
        x_val = np.asarray(x_val)

        img_rows, img_cols = x_train.shape[1], x_train.shape[2] 
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        model, history = self.train_untill((x_train,y_train),(x_val,y_val),input_shape)
        return model, history, x_val, y_val


if __name__ == "__main__":
    # Loading of data
    filedir = 'IRMAS_trainingData_full\\'
    mt_object = Model_training(Han_model,num_classes=11,min_acc=0.73)
    model,history,x_val,y_val = mt_object.train_model(filedir,augment_data=True,additional_data=['Nsynth\\'])#additional_data=['Nsynth\\']

    answer = input('Would you like to save the model? [y/n]')
    answer=answer.lower()
    while (answer != 'y') and (answer != 'n'):
        answer = input('Not an acceptable answer \n\tWould you like to save the model? [y/n]')    
    if answer== 'y':
        model_name = input('Please enter the name of the model: ')
        # Plotting of accuracy
        utill.plot_accuracy(history,model_name=model_name,save=True)
        # Plotting of loss
        utill.plot_loss(history,model_name=model_name,save=True)
        # Plotting and evaluation of the confusion matrix
        y_pred = model.predict(x_val)
        y_sl_true = utill.mutilabel2single(y_val)
        y_sl_pred = utill.mutilabel2single(y_pred)
        utill.plot_confusion_matrix(y_sl_true,y_sl_pred,utill.CLASS_NAMES,title = model_name, save=True)
        main_model_test(model,'IRMAS_testdata\\',model_name+'.csv')

        # serialize weights to HDF5
        model.save_weights(os.path.join(CWD,'trained_models',model_name+"_weights.h5"))
        print("Saved model to disk")

