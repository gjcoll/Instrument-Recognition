import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import pickle
import os
import utill

from os.path import join, isfile
from os import listdir

CWD = os.getcwd()
if CWD.split('\\')[-1] != 'Keras':
    CWD = join(CWD,'Keras')

matplotlib.use('TkAgg')

class ResultAnimator():

    def __init__(self,class_names):
        self.x = []
        self.b = [] 
        self.c = []
        self.d = []
        self.e = []
        self.f = []
        self.g = []
        self.u = [] 
        self.aa = [] 
        self.ab = []
        
        self.ani = None
        self.rects = None
        self.class_names = class_names

    def resest(self):
        self.x = []
        self.b = [] 
        self.c = []
        self.d = []
        self.e = []
        self.f = []
        self.g = []
        self.u = [] 
        self.aa = [] 
        self.ab = []
        self.rects = None

        self.ani = None

    def animated_barplot(self,frame):
        y = [self.x[frame][0], self.x[frame][1], self.x[frame][2], self.x[frame][3], self.x[frame][4], self.x[frame][5], self.x[frame][6], self.x[frame][7], self.x[frame][8], self.x[frame][9], self.x[frame][10]]
        for rect, h in zip(self.rects, y):
            rect.set_height(h)
        return self.rects

    def animate_results(self, results):
        length = len(results)
        for i in range(length-1):
            listobj1 = results[i].tolist()
            listobj2 = results[i+1].tolist()
            for m in range(11):
                self.b.append(np.linspace(listobj1[m], listobj2[m], num=5, endpoint=False).tolist())
            for j in range(11):
                self.c.append(self.b[j][0])
                self.d.append(self.b[j][1])
                self.e.append(self.b[j][2])
                self.f.append(self.b[j][3])
                self.g.append(self.b[j][4])
            self.x.append(self.c)
            self.x.append(self.d)
            self.x.append(self.e)
            self.x.append(self.f)
            self.x.append(self.g)
            self.b, self.c, self.d, self.e, self.f, self.g, self.u, self.aa, self.ab = [], [], [], [], [], [], [], [], []

        N = 11
        y = [self.x[0][0], self.x[0][1], self.x[0][2], self.x[0][3], self.x[0][4], self.x[0][5], self.x[0][6], self.x[0][7], self.x[0][8], self.x[0][9], self.x[0][10]]
        fig, ax = plt.subplots()
        self.rects = plt.bar(range(N), y[0],  align='center')
        xticks = self.class_names
        plt.xticks(range(11), xticks, size=8, rotation=45)
        plt.xlim([-1, 11])
        plt.ylim([0, 1])
        plt.rcParams['animation.ffmpeg_path'] = r'PATHTOFFMPEG'
        self.ani = animation.FuncAnimation(fig, self.animated_barplot, blit=True, interval=(23*5),  repeat=False, frames=len(self.x))
        plt.show()

    def save_animation(self,filename):
        Writer = animation.FFMpegWriter
        writer = Writer(metadata=dict(artist='Me'))
        self.ani.save(filename, writer=writer)

if __name__ == "__main__":
    with open(join(CWD,'predictions\\IRMAS_predictions_all.pkl'), 'rb') as f:
        a = pickle.load(f)
    RA=ResultAnimator(utill.CLASS_NAMES)
    RA.animate_results(a[0])