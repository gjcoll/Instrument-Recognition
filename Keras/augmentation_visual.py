#%%
import os
CWD = os.getcwd()
if CWD.split('\\')[-1] != 'Keras':
    CWD = os.path.join(CWD,'Keras')
os.chdir(CWD)
import matplotlib.pyplot as plt
from matplotlib import animation
import utill
import numpy as np


X,y = utill.read_npz_folder('IRMAS_trainingData_full\\')

plt.imshow(X[0])
plt.title('Mel Spec Original')
plt.xlabel('Time Samples')
plt.ylabel('Mel Freq Bins')
plt.show()


#%%
X_prime = utill.drop_timenfreq(X[0],chuncks=1)
plt.imshow(X_prime)
plt.title('Mel Spec Augmented')
plt.xlabel('Time Samples')
plt.ylabel('Mel Freq Bins')
plt.show()
