import pandas as pd
import numpy as np
import pickle


filename = 'D:\\SD Project\\OpenMic\\openmic-2018-v1.0.0\\openmic-2018-v1.0.0\\openmic-2018\\openmic-2018-aggregated-labels.csv'
fileinfo = {}
file_csv = pd.read_csv(filename)
inst_labels = file_csv.instrument
sample_name = file_csv.sample_key

inst_classes = ['acc','ban','bas','cel','cla','cym','dru','flu','gui','mal','man','org','pia','sax','syn','tro','tru','uku','vio','voi']

for lab in inst_labels:
    label_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    s = 0
    dex = 0
    for inst in inst_classes:
        if lab[:3] == inst:
            label_array[dex] = 1
        else:
            dex = dex+1
            
    fileinfo[sample_name[s]] = label_array 
    s = s+1
    
pickle_out=open("C:\\Users\\anna_\\OneDrive\\Documents\\GitHub\\Instrument-Recognition\\Keras\\openMic_labels.pickle","wb")
pickle.dump(fileinfo, pickle_out)
pickle_out.close()
