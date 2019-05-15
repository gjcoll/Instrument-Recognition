import numpy as np
import glob, os, pickle
from pathlib import Path
import librosa
import scipy
import shutil
import gzip
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pathlib import Path
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

CLASS_NAMES = ['Cello', 'Clarinet', 'Flute', 'Acoustic Guitar', 'Electric Guitar','Organ','Piano','Saxophone','Trumpet','Violin','Voice'] 


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
    

def spec_multiple(src_path, dest_path, done_folder, label, max_count):
    ## Function that combines loading, mel spec and compressing for multiple samples 
    # src_path = samples (.wav) folder
    # dest_path = path for resulting compressed path (.npz)
    # done_folder = path to folder to move sample files that are done to - this makes knowing where to start another itteration easier
    # label = binary label for samples
    # max_count = number of files to process, default = 2000

    #load in max_count # of samples (defult 2000)
    # A_samp = load_folder(src_path,max_count,done_folder)
    A_samp = load_folder_IRMAS(src_path,max_count,done_folder)

    # normaize by dividiving time domain signal by max value
    A_norm = []
    
    for tds in A_samp:
        tds_max = np.amax(tds[0])
        A_norm.append([tds[0]/tds_max , tds[1]])


    # Generate Mel spectrograms (128)
    melspecs_A = [mel_spec_it(x[0],x[1]) for x in A_norm]

    #compress magnitudes with natural log
    ln_melspecs_A = [np.log(abs(h) + np.finfo(np.float64).eps) for h in melspecs_A]
    # Generate Label Array based on input label
    # A_labels = np.full(shape = np.shape(melspecs_A)[0],fill_value = label)
    A_labels = np.tile(label,[np.shape(melspecs_A)[0],1])
    # Save to compressed file, file of two arrays 'data' = audio part (mel specs), 'labels' = label array
    np.savez_compressed(dest_path, data = ln_melspecs_A, labels = A_labels)

    return np.shape(ln_melspecs_A)
    
def train_split_spec(X:np.array,y:np.array,time_split:int = 44):
    # Splits training and validation data that is longer than a second into 1 second long spec
    # with the same labels 

    # Inputs:
    #
    X_list = []
    y_list = []
    if len(X.T) != 44:
        i = 0
        while i < len(X.T):
            if len(X[:,i:].T) < 44:
                x = X[:,i:]
                dif = 44 - len(x.T)
                X_list.append(np.concatenate([x, np.zeros([len(x), dif])], axis = 1))
                y_list.append(y)
            else:
                X_list.append(X[:,i:i+44])
                y_list.append(y)
            i += 44
    else:
        X_list.append(X)
        y_list.append(y)
    
    return X_list, y_list





def load_folder_IRMAS(data_path,max_count,done_folder):
    ## Downsamples tp 22050 and converts stero to mono and only loads 1sec to make dimensions match the paper

    ## Function to load .wav files in a given folder to an array of (data, sample rate) ##
    # INPUTS:
    # data_path = string to folder containing files to be loaded
    # max_count = number of files you wish to load in
        ## max_count can be used to only load the first "max_count" number of files from a folder,
        # if max_count == 0 then whole folder will be loaded
    ## ------------------------------ 
  
    samples = []
    count = 0
    downsamp = 22050
    
    if max_count!= 0: #Load first 'max_count' number of files from folder
        for file in glob.glob(os.path.join(data_path,'*.wav')):
            if count < max_count:
                # temp,sr = librosa.core.load(file,sr = downsamp,mono = True,duration = 1) # downsample to 22050 and make mono (default)
                temp,sr = librosa.core.load(file,sr = downsamp,mono = True) # downsample to 22050 and make mono (default)
                # temp = librosa.util.fix_length(temp,2*sr);
                samples.append([temp,sr])
                shutil.move(file,done_folder)
                count+=1
                
    else:
    #load whole folder
         for file in glob.glob(os.path.join(data_path,'*.wav')):
                temp,sr = librosa.core.load(file,sr = downsamp,mono = True) # downsample to 22050 and make mono (idk if the mono right)
                # temp = librosa.util.fix_length(temp,2*sr);
                samples.append([temp,sr])
                
    return samples


def get_npz_filenames(filedir):
    ## Function to get npz_filenames given a directory to look it
    # INPUTS:
    # filedir = a file directory containing npz files
    #
    # OUTPUT
    # roots = the root directory
    # npz_files = generator object within root containing npz_files
    cw = os.getcwd()
    for roots, dirs,files in os.walk(os.path.join(cw, filedir)):
        npz_files = (f for f in files)
        return roots, npz_files

def read_npz_folder(filedir):
    root, files = get_npz_filenames(filedir)
    X,y = load_npz(root + next(files))
    for f in files:
        if f == 'IRMAS_flu_A.npz':
            data, label = load_npz(root + f)
            label = label * 7
            X = np.append(X,data, axis =0)
            y = np.append(y,label, axis =0)
        elif f[-4:] == '.npz':
            data, label = load_npz(root + f)
            try:
                X = np.append(X,data, axis =0)
                y = np.append(y,label, axis =0)
            except:
                print('An error has occured when loading file ', f)    
    return X, y

def read_test_npz_folder(filedir):
    root, files = get_npz_filenames(filedir)
    X,y = [],[]
    for f in files:
        data,label = load_npz(os.path.join(root,f))
        X.append(data)
        y.append(label)
    return X,y

def test_labels_IRMAS(folderpath):
    names = [os.path.basename(x) for x in glob.glob(folderpath + '/**/*.wav', recursive=True)]
    fileinfo={}
    i=0
    os.chdir(folderpath)
    for file in os.listdir(folderpath):
        if file.endswith(".txt"):
            with open(file) as myfile:
                # print(myfile.read())
                # cel cla flu gac gel org pia sax tru vio voi nod dru
                fileparams = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                testparams = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for line in myfile.readlines():
                    if 'cel' in line:
                        fileparams[0] = 1
                    if 'cla' in line:
                        if fileparams != testparams:
                            continue
                        fileparams[1] = 1
                    if 'flu' in line:
                        fileparams[2] = 1
                    if 'gac' in line:
                        fileparams[3] = 1
                    if 'gel' in line:
                        fileparams[4] = 1
                    if 'org' in line:
                        fileparams[5] = 1
                    if 'pia' in line:
                        fileparams[6] = 1
                    if 'sax' in line:
                        fileparams[7] = 1
                    if 'tru' in line:
                        fileparams[8] = 1
                    if 'vio' in line:
                        fileparams[9] = 1
                    if 'voi' in line:
                        fileparams[10] = 1
                    if 'nod' in line:
                        fileparams[11] = 1
                    if 'dru' in line:
                        fileparams[12] = 1
                fileinfo[names[i]]=fileparams
                i=i+1

    # Simple pickle, not sure what the end goal with this is but I can change it up to what we need
    #pickle_out=open("datadict.pickle","wb")
    #pickle.dump(fileinfo, pickle_out)
    #pickle_out.close()
    #pickle_in=open("datadict.pickle","rb")
    #example=pickle.load(pickle_in)
    # Example call to get the dict value of a filename out of a dict
    #print(example['01 - Chet Baker - Prayer For The Newborn-8.wav'])

    return fileinfo

def chunks(alist, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(alist), n):
        yield alist[i:i + n]

def load_folder_Test(data_path):
    ## FOR .WAV FILES
    ## Downsamples tp 22050 and converts stero to mono and only loads 1sec to make dimensions match the paper

    ## Function to load .wav files in a given folder to an array of (data, sample rate) ##
    # INPUTS:
    # data_path = string to folder containing files to be loaded
    # max_count = number of files you wish to load in
        ## max_count can be used to only load the first "max_count" number of files from a folder,
        # if max_count == 0 then whole folder will be loaded
    ## ------------------------------ 
  
    samples = []
    downsamp = 22050
    for file in glob.glob(os.path.join(data_path,'*.wav')):
        temp,sr = librosa.core.load(file,sr = downsamp,mono = True) # downsample to 22050 and make mono (default)
        samples.append([temp,sr])
              
                
    return samples


def spec_Testing(fpath,filename,dest_path,label):
    ## Function that combines loading, mel spec and compressing for multiple samples 
    
    #load folder
    
    A_samp,sr = librosa.core.load(fpath + '\\' + filename,sr = 22050, mono = True) # downsample to 22050 and make mono (default)
    # label_all = test_labels_IRMAS(folderpath)

    # normaize by dividiving time domain signal by max value
    # count = 0
    # for song in A_samp:
    #     A_norm = []
    #     tds_max = np.amax(song[0])
    #     A_norm = [song[0] / tds_max , song[1]]
    #     melspec = mel_spec_it(A_norm[0],A_norm[1])
    #     ln_mel = np.log(abs(melspec) + np.finfo(np.float64).eps)

    #     dpath = dest_path + '_' + str(count) + '.npz'
    #     np.savez_compressed(dpath, data = ln_mel, labels = label_all[count])
    #     count = count + 1
    A_norm = A_samp / (np.amax(A_samp))
    mels = mel_spec_it(A_norm,sr)
    ln_mels = np.log(abs(mels)+np.finfo(np.float64).eps)
    np.savez_compressed(dest_path, data = ln_mels,  labels = label )

   

    return ln_mels
    
def mutilabel2single(mutli_label, labels=CLASS_NAMES):
    # A function that returns single labels for use of confusion matrix
    single_label = [None]*len(mutli_label)
    i=0
    for label in mutli_label:
        single_label[i] = labels[int(np.argmax(label))]
        i+=1
    return single_label



## Analysis 
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap= cm.Blues,
                          save:bool = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cmat = confusion_matrix(y_true, y_pred,labels = classes)
    # Only use the labels that appear in the data
    if normalize:
        cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cmat)

    fig, ax = plt.subplots()
    im = ax.imshow(cmat, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cmat.shape[1]),
           yticks=np.arange(cmat.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cmat.max() / 2.
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            ax.text(j, i, format(cmat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cmat[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    if save:
        cwd=os.getcwd()
        plt.savefig(os.path.join(cwd, 'Keras\\Model_images', title +'_CM.png'))
    return ax

def plot_accuracy(history, model_name=None, save:bool = False):
    if model_name is None:
        model_name = ''
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy: ' + model_name)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    if save:
        cwd = os.getcwd()
        fig.savefig(os.path.join(cwd,'Keras\\Model_images',model_name + '_accuracy.png'))

def plot_loss(history, model_name=None, save:bool = False):
    if model_name is None:
        model_name = ''
    fig=plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss: '+ model_name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    if save:
        cwd = os.getcwd()
        fig.savefig(os.path.join(cwd,'Keras\\Model_images',model_name + '_loss.png'))
    

