import soundfile as sf
import librosa
import numpy as np
import pickle,glob,os
from utill import mel_spec_it,spec_multiple

# filename =  'D:\\SD Project\\OpenMic\\openmic-2018-v1.0.0\\openmic-2018-v1.0.0\\openmic-2018\\audio\\000\\000046_3840.ogg'
nums = np.arange(100,156,1)
for i in nums:
    a = str(i)
    data_path = 'D:\\SD Project\\OpenMic\\openmic-2018-v1.0.0\\openmic-2018-v1.0.0\\openmic-2018\\audio\\' + a
    dest_path = 'D:\\SD Project\\OpenMic\\openmic-2018-v1.0.0\\openmic-2018-v1.0.0\\openmic_pre\\openmic_' + a + '.npz'
    labelpick = open('C:\\Users\\anna_\\OneDrive\\Documents\\GitHub\\Instrument-Recognition\\Keras\\openMic_labels.pickle','rb')
    label_def = pickle.load(labelpick)
    # label = label_def['000046_3840']
    A_norm = []
    A_labels = []
    for filenm in glob.glob(os.path.join(data_path,'*.ogg')):
        dp,ext = os.path.split(filenm)
        samp_key = ext[:-4]
        label = label_def[samp_key]

        data = sf.read(filenm)
        T_data= data[0].T
        if np.shape(T_data)[0] == 2:
            mono_data =  np.mean( np.array([ T_data[0], T_data[1] ]), axis=0 )
        else:
            mono_data = T_data
        data_22k = librosa.resample(mono_data, data[1], 22050,mono=True)

            
        tds_max = np.amax(data_22k)
        A_norm.append([data_22k/tds_max , data[1]])
        A_labels.append([label])

    melspecs_A = [mel_spec_it(x[0],x[1]) for x in A_norm]

    #compress magnitudes with natural log
    ln_melspecs_A = [np.log(abs(h) + np.finfo(np.float64).eps) for h in melspecs_A]
    # Generate Label Array based on input label
    # A_labels = np.full(shape = np.shape(melspecs_A)[0],fill_value = label)
    # A_labels = np.tile(label,[np.shape(melspecs_A)[0],1])
    # Save to compressed file, file of two arrays 'data' = audio part (mel specs), 'labels' = label array
    np.savez_compressed(dest_path, data = ln_melspecs_A, labels = A_labels)


    print(np.shape(ln_melspecs_A))