"""
A Testing and analysis Framework for Saved keras Models 
"""
from __future__ import print_function
import keras
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import model_from_json, load_model
from keras.utils import plot_model
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import Han_cnn
import matplotlib.pyplot as plt
import utill
import os
import numpy as np

def l_model(model_json, model_weights_h5):
    with open(model_json, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(model_weights_h5)
    return model

def load_model_testcase():
    ## Load Data ##
    cwd = os.getcwd()
    model = Han_cnn.Han_model()
    model_weights_h5 = cwd+"\\Models\\model_03032019_0.h5"
    model.load_weights(model_weights_h5)
    return True

def confusion_matix_testcase():
    y = np.load('y_true.npy')
    y = utill.mutilabel2single(y)
    utill.plot_confusion_matrix(y,y,utill.CLASS_NAMES)


if __name__ == "__main__":
    model = Han_cnn.Han_model()
    plot_model(model, to_file='model.png')
    model_svg=SVG(model_to_dot(model).create(prog='dot', format='svg'))
    display(model_svg)
    input()
    
