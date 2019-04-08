import numpy as np
import glob, os, pickle
from pathlib import Path
import librosa
import scipy
import shutil
import gzip
from utill import load_folder_Test,test_labels_IRMAS,spec_Testing


fpath = 'D:\\SD Project\\IRMAS\\IRMAS-TestingData-Part1\\IRMAS-TestingData-Part1\\Part1' # path to testing data folder
npz_dest = 'D:\\SD Project\\IRMAS\\IRMAS-TestingData-Part1\\IRMAS-TestingData-Part1\\IRMAS_testPt1_' #change to location for npz to be saved 

pickle_in=open("C:\\Users\\anna_\\OneDrive\\Documents\\GitHub\\Instrument-Recognition\\Keras\\datadict.pickle","rb")
label_dict=pickle.load(pickle_in)
count = 0
for file in os.listdir("D:\\SD Project\\IRMAS\\IRMAS-TestingData-Part1\\IRMAS-TestingData-Part1\\Part1") :
    if file.endswith(".wav"):

        lab = label_dict[file]
        dest = npz_dest + str(count)
        spec_Testing(fpath,file,dest,lab)
        count = count + 1


