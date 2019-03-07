import numpy as np
import glob, os
from pathlib import Path
import librosa
import scipy
import shutil
import gzip
from utill import load_folder_Test,test_labels_IRMAS,spec_Testing


fpath = 'D:\\SD Project\\IRMAS\\IRMAS-TestingData-Part1\\IRMAS-TestingData-Part1\\subset' # path to testing data folder
npz_dest = 'D:\\SD Project\\IRMAS\\IRMAS-TestingData-Part1\\IRMAS-TestingData-Part1\\IRMAS_testdata.npz' #change to location for npz to be saved 
t_labels = test_labels_IRMAS
spec_Testing(fpath,npz_dest,t_labels)


