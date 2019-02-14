# Nsynth Spectogram Extractor

import numpy as np
import glob, os
from pathlib import Path
import librosa
from utill import load_folder,mel_spec_it,spec_multiple
import shutil



#count number of files in folder
source_path = 'D:\\SD Project\\nsynth\\instr_families\\gui'
num_files = len(os.listdir(source_path))

#run 2000, change letter in dest_folder path
df_path = 'D:\\SD Project\\nsynth\\instr_families\\done_folders\\df_gui'
compress_path = 'D:\\SD Project\\nsynth\\instr_families\\MelSpecs_comp\\gui_2000_'
letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# letter = 'CDEFGHIJKLMNOP' # manually Change starting letter if partially through a file
i = 0 # index for letters
label = 0 # define label 

while num_files > 0: #while there are still files in the folder
    num_files = len(os.listdir(source_path)) #check how many files
    if num_files >= 2000: #if there are more than 2000 left run 2000

        ltr = letter[i] # determine letter for the npz file
        spec_multiple(source_path,(compress_path + ltr + '.npz'),df_path,label) # run specs on 2000 samples and return compressed np file 
        i = i+1 # increase letter index
        
    else: # if there are less than 2000 files left, then just run the rest of the files (the folder name should really be changed, from 2000 but i didnt do that)
        spec_multiple(source_path,(compress_path + ltr + '.npz'),df_path,label,max_count = num_files)
        i = i+1
    # print(i) # uncomment to print just to check program still running and see how many files created