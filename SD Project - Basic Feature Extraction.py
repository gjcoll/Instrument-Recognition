
# coding: utf-8

# In[274]:


import sklearn
from pathlib import Path
import librosa
import numpy
import scipy
import matplotlib.pyplot as plt
import IPython


import mir_eval


# In[275]:


# ls audio


# In[276]:


kickSamples = numpy.array([librosa.load('audio\Kick\RGRS_KICK_HDFL_HT_01.wav'),
               librosa.load('audio\Kick\RGRS_KICK_HDFL_HT_02.wav'),
               librosa.load('audio\Kick\RGRS_KICK_HDFL_HT_03.wav'),
               librosa.load('audio\Kick\RGRS_KICK_SFFL_HT_01.wav'),
               librosa.load('audio\Kick\RGRS_KICK_SFFL_HT_02.wav')])


snareSamples = numpy.array([librosa.load('audio\Snare\L400_SNR_DCFL_HT_01.wav'),
                librosa.load('audio\Snare\L400_SNR_DCFL_HT_02.wav'),
                librosa.load('audio\Snare\L400_SNR_DCFL_HT_03.wav'),
                librosa.load('audio\Snare\L400_SNR_DEFL_HT_01.wav'),
                librosa.load('audio\Snare\L400_SNR_DEFL_HT_02.wav'),
                librosa.load('audio\Snare\L400_SNR_DEFL_HT_03.wav'),
                librosa.load('audio\Snare\L400_SNR_LCFL_HT_01.wav'),
                librosa.load('audio\Snare\L400_SNR_LCFL_HT_02.wav'),
                librosa.load('audio\Snare\L400_SNR_LCFL_HT_03.wav'),
                librosa.load('audio\Snare\L400_SNR_LEFL_HT_01.wav'),
                librosa.load('audio\Snare\L400_SNR_LEFL_HT_02.wav')])              


# In[277]:


#  librosa.feature.zero_crossing_rate(snareSamples[1][0])
    


# In[278]:


def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0,0],#Zero Crossing Rate
        librosa.feature.spectral_centroid(signal)[0,0], #center freq
        librosa.feature.tonnetz(signal)[0,0] #tonal centroid features
    ]


# In[279]:


kickFeat = numpy.array([extract_features(x[0]) for x in kickSamples])
snareFeat = numpy.array([extract_features(x[0]) for x in snareSamples])


# In[280]:


plt.figure(figsize=(14, 5))
plt.hist(kickFeat[:,0], color='g', range=(0, 0.1), alpha=0.5, bins=20)
plt.hist(snareFeat[:,0], color='r', range=(0, 0.1), alpha=0.5, bins=20)
plt.legend(('kicks', 'snares'));
plt.xlabel('Zero Crossing Rate');
plt.ylabel('Count');


# In[281]:


plt.figure(figsize=(14, 5))
plt.hist(kickFeat[:,1], color='g', range=(0, 2000), bins=30, alpha=0.6)
plt.hist(snareFeat[:,1], color='r', range=(0, 2000), bins=30, alpha=0.6)
plt.legend(('kicks', 'snares'))
plt.xlabel('Center Frequency (freq bin)')
plt.ylabel('Count')


# In[282]:


plt.figure(figsize=(14, 5))
plt.hist(kickFeat[:,2], color='g', range=(-.03, .05), bins=30, alpha=0.6)
plt.hist(snareFeat[:,2], color='r', range=(-.03, .05), bins=30, alpha=0.6)
plt.legend(('kicks', 'snares'))
plt.xlabel('Tonal Centroids')
plt.ylabel('Count')


# In[283]:


feature_table = numpy.vstack((kickFeat, snareFeat))
print(feature_table.shape)


# In[284]:


scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
training_features = scaler.fit_transform(feature_table)
print(training_features.min(axis=0))
print(training_features.max(axis=0))


# In[285]:


plt.scatter(training_features[:5,1], training_features[:5,2], c='g') #kick
plt.scatter(training_features[5:,1], training_features[5:,2], c='r') #snare

plt.xlabel('Center Frequency')
plt.ylabel('Tonal Centroid')


# In[286]:


a, sr = librosa.load('audio\simpleloop.wav')


# In[287]:


a_onset = librosa.onset.onset_detect(a)


# In[288]:


a_os_sec = librosa.frames_to_time(a_onset)


# In[289]:


a_os_samp =librosa.frames_to_samples(a_onset)


# In[290]:


clicks = librosa.clicks(frames=a_onset, sr=sr, length=len(a))


# In[291]:


IPython.display.Audio(x + clicks, rate=sr)


# In[292]:


a_feat = extract_features(a) # should segrment first ---- rn just looking at whole file 


# In[270]:


frame_sz = int(sr*0.100)
afeat_split = numpy.array([extract_features(a[i:i+frame_sz]) for i in a_os_samp])


# In[271]:


scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
afeat_feat =scaler.fit_transform(afeat_split)


# In[272]:


afeat_scaled


# In[273]:


plt.scatter(afeat_feat[:2,1], afeat_feat[:2,2], c='g')
plt.xlabel('Center Frequency')
plt.ylabel('Tonal Centroid')


# In[294]:


model = sklearn.cluster.AffinityPropagation()
labels = model.fit_predict(afeat_feat)
print(labels)

