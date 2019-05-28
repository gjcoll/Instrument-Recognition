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
class Animator():
        def __init__(self):
                self.barcollection = None


        def barlist(self,n): 
                return [1/float(n*k) for k in range(1,6)]

        def animate(self,i):
                y=self.barlist(i+1)
                for i, b in enumerate(self.barcollection):
                        b.set_height(y[i])

        def interpolate(self):
                pass

        def animate_bars(self):
                fig=plt.figure()

                n=50 #Number of frames
                x=range(1,6)
                self.barcollection = plt.bar(x,self.barlist(1))# Generates our bar charts into one collection

                anim=animation.FuncAnimation(fig,self.animate,repeat=False,blit=False,frames=n,
                                        interval=100)

                anim.save('mymovie.mp4',writer=animation.FFMpegWriter(fps=10))
                plt.show()
A = Animator()
A.animate_bars()
#%%