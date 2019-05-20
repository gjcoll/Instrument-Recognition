"""
This is a sequential pipeline code that allows for a raw audio input selected by the users to be run through a series of instrument recognition models
"""
import utill
import model_testing
#import animate
import os
import keras
import time

from os import listdir
from os.path import isfile, join
from keras_model_options import Han_model
from keras.callbacks import EarlyStopping
from keras import backend as K
from animator import ResultAnimator

CWD = os.getcwd()
if CWD.split('\\')[-1] != 'Keras':
    CWD = join(CWD,'Keras')

if __name__ == "__main__":
    """
    Section 1:
    Select an audio signal to have IR done on
    """
    #Place holder comment
    """
    Section 2:
    Run the whole audio signal through our preprocessing pipline
    """
    # Ths following section is a place holder for preprocessing
    X,y = utill.read_test_npz_folder(join(CWD,'IRMAS_testdata'))
    # for testing
    X = X[0]
    """
    Section 3:
    Run the preprocessed audio signal through several models, outputing a dictionary of model name to outputs
    """
    weight_path = join(CWD, 'trained_models') 
    weights = [f for f in listdir(weight_path) if isfile(join(weight_path,f))]
    model_name = Han_model
    model_results = {}
    for weight in weights:
        model=model_testing.l_model(model_name,join(weight_path,weight))
        model_results[weight] = model_testing.windowed_predict(model,X,hop_size=22,seconds=10)

    
    """
    Section 4:
    Create an animation object for each of the models,
    Combine with an audio signal. 
    """
    RA =ResultAnimator(utill.CLASS_NAMES)
    for result in model_results:
        RA.animate_results(model_results[result])
        RA.save_animation(join(CWD, 'Model_results\\'+result))
        RA.resest()