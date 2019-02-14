import numpy as np
import glob, os
from pathlib import Path
import librosa
import scipy
import shutil
import gzip

def load_folder(data_path,max_count,done_folder):
    ## Function to load .wav files in a given folder to an array of (data, sample rate) ##
    # INPUTS:
    # data_path = string to folder containing files to be loaded
    # max_count = number of files you wish to load in
        ## max_count can be used to only load the first "max_count" number of files from a folder,
        # if max_count == 0 then whole folder will be loaded
    ## ------------------------------ 
  
    samples = []
    count = 0
    
    if max_count!= 0: #Load first 'max_count' number of files from folder
        for file in glob.glob(os.path.join(data_path,'*.wav')):
            if count < max_count:
                temp,sr = librosa.load(file)
                temp = librosa.util.fix_length(temp,2*sr)
                samples.append([temp,sr])
                shutil.move(file,done_folder)
                count+=1
                
    else:
    #load whole folder
         for file in glob.glob(os.path.join(data_path,'*.wav')):
                temp,sr = librosa.load(file)
                temp = librosa.util.fix_length(temp,2*sr)
                samples.append([temp,sr])
                
    return samples
            
    
def nsy_sep(src_path,dest_path): 
    
    ## Specific function to separate nsynth dataset into folders based on instument family, returns nothing ##
    # INPUTS:
    # src_path = path to folder containing all nsynth files with no separation
    # dest_path = destination path for folder containing premade folders for each instrument family labeled after
    # the first three letters in the family name (ie, bass = bas, brass = bra, vocals = voc etc.)
    ## -------------------------
    for file in glob.glob(os.path.join(src_path,'*.wav')):
        fam = str.split(file,src_path+'\\')[1][:3]
        folder = os.path.join(dest_path,fam)
        shutil.move(file,folder)
        
        

def mel_spec_it(signal,fs):
    ## Function to find Mel Spectrogram of a signal ##
    # INPUTS:
    # signal
    # fs = sample rate
    # Returns an array of shape(n_mel,time)
    ##--------------------------    
    melS = librosa.feature.melspectrogram(signal,fs) # Mel Spectorgram
  
    return melS

def load_npz(npz_file):
    ## Function to load a compressed .npz file into an array of data and labels ##
    ## -------------
    aloda = np.load(npz_file)
    data_arry = aloda['data']
    label_arry = aloda['labels']
    
    
    return data_arry,label_arry

def load_npz_old(npz_file):
    ## use if load_npz returning error about 'data' not being found 
    ## (these are from an oder version of formatting I used when first parsing, 
    # I need to rerun some of these, the mel spec data is correct and useable from this load function)

    aloda = np.load(npz_file)
    data_ary = aloda['a']

    return data_ary
    

def spec_multiple(src_path, dest_path, done_folder, label, max_count = 2000):

    #load in max_count # of samples (defult 2000)
    A_samp = load_folder(src_path,max_count,done_folder)
    # Generate Mel spectrograms (128)
    melspecs_A = [mel_spec_it(x[0],x[1]) for x in A_samp]
    # Generate Label Array based on input label
    A_labels = np.full(shape = np.shape(melspecs_A)[0],fill_value = label)
    # Save to compressed file, file of two arrays 'data' = audio part (mel specs), 'labels' = label array
    np.savez_compressed(dest_path, data = melspecs_A, labels = A_labels)

    return np.shape(melspecs_A)
