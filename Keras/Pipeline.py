"""
This is a sequential pipeline code that allows for a raw audio input selected by the users to be run through a series of instrument recognition models
"""
import utill
import model_testing
#import animate
import os
import keras
import time
import moviepy.editor as mpe
import multiprocessing as mp

from tkinter import *
from tkinter import filedialog
from os import listdir
from os.path import isfile, join
from keras_model_options import Han_model
from keras.callbacks import EarlyStopping
from keras import backend as K
from animator import ResultAnimator
from tkinter.filedialog import askopenfilename

CWD = os.getcwd()
if CWD.split('\\')[-1] != 'Keras':
    CWD = join(CWD,'Keras')

num_threads = mp.cpu_count()
seconds = 19
from moviepy.video.io.VideoFileClip import VideoFileClip
from threading import Timer

if __name__ == "__main__":
    """
    Section 1:
    Select an audio signal to have IR done on
    """
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = join(CWD,'Test_songs'),title = "Select song file",filetypes = (("wav files","*.wav"),("all files","*.*")))
    print(root.filename)
    songfile = root.filename
    """
    Section 2:
    Run the whole audio signal through our preprocessing pipline
    """
    #preprocess single audio file from section 1

    melspecs = utill.preprocess_single(songfile)
    X = melspecs
    
    # uncomment below to open npz folder of preprocessed data
    #     X,y = utill.read_test_npz_folder(join(CWD,'IRMAS_testdata'))
    #     # for testing
    #     X = X[0]
    """
    Section 3:
    Run the preprocessed audio signal through several models, outputing a dictionary of model name to outputs
    """
    weight_path = join(CWD, 'trained_models') 
    weights = [join(weight_path,'Best_weights.h5')]#[f for f in listdir(weight_path) if isfile(join(weight_path,f))]
    model_name = Han_model
    model_results = {}
    for weight in weights:
        model=model_testing.l_model(model_name,join(weight_path,weight))
        model_results[weight] = model_testing.windowed_predict(model,X,hop_size=22,seconds=seconds)

    del model
    del X
    del melspecs
    """
    Section 4:
    Create an animation object for each of the models,
    Combine with an audio signal. 
    """
    RA =ResultAnimator(utill.CLASS_NAMES)
    for result in model_results:
        f_name = songfile.split('/')[-1]
        f_name = f_name.split('.')[0] + '.mp4'
        f_name = os.path.join(CWD,'videos',f_name)
        RA.animate_results(model_results[result])
        RA.save_animation(f_name)
        RA.resest()
        del RA
        my_clip = mpe.VideoFileClip(f_name)
        audio= mpe.AudioFileClip(songfile)
        final_clip = my_clip.set_audio(audio.set_duration(seconds))
        final_clip.write_videofile(f_name, threads = 4)
