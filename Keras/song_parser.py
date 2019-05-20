import numpy as np
import glob, os, pickle
from pathlib import Path
import librosa
import scipy
import shutil
import gzip
from utill import load_folder_Test,test_labels_IRMAS,spec_Testing
#['cel','cla','flu','gac','gel','org','pia','sax','tru','vio','voi']
fpath = 'C:\\Users\\anna_\\Desktop\\Test_Songs'
npz_dest = 'C:\\Users\\anna_\\Desktop\\Test_Songs\\CultofPersonality.npz'
# pickle_in=open("C:\\Users\\anna_\\OneDrive\\Documents\\GitHub\\Instrument-Recognition\\Keras\\datadict.pickle","rb")
# label_dict=pickle.load(pickle_in)
# count = 0
# for file in os.listdir("D:\\SD Project\\IRMAS\\IRMAS-TestingData-Part1\\IRMAS-TestingData-Part1\\Part1") :
#     if file.endswith(".mp3"):

#         lab = label_dict[file]
#         dest = npz_dest + str(count)
#         spec_Testing(fpath,file,dest,lab)
#         count = count + 1
lab = [0,0,0,0,1,0,0,0,0,0,0]
spec_Testing(fpath,'CultofPersonality.wav',npz_dest,lab)
