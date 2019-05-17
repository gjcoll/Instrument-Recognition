import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import pickle
import os

from os.path import join, isfile
from os import listdir

CWD = os.getcwd()
if CWD.split('\\')[-1] != 'Keras':
    CWD = join(CWD,'Keras')

matplotlib.use('TkAgg')

with open(join(CWD,'predictions\\IRMAS_predictions_all.pkl'), 'rb') as f:
    a = pickle.load(f)
x, b, c, d, e, f, g, u, aa, ab = [], [], [], [], [], [], [], [], [], []

length = len(a[0])
for i in range(length-1):
    listobj1 = a[0][i].tolist()
    listobj2 = a[0][i+1].tolist()
    for m in range(11):
        b.append(np.linspace(listobj1[m], listobj2[m], num=5, endpoint=False).tolist())
    for j in range(11):
        c.append(b[j][0])
        d.append(b[j][1])
        e.append(b[j][2])
        f.append(b[j][3])
        g.append(b[j][4])
    x.append(c)
    x.append(d)
    x.append(e)
    x.append(f)
    x.append(g)
    b, c, d, e, f, g, u, aa, ab = [], [], [], [], [], [], [], [], []


def animated_barplot(frame):
    y = [x[frame][0], x[frame][1], x[frame][2], x[frame][3], x[frame][4], x[frame][5], x[frame][6], x[frame][7], x[frame][8], x[frame][9], x[frame][10]]
    for rect, h in zip(rects, y):
        rect.set_height(h)
    return rects


Writer = animation.FFMpegWriter
writer = Writer(metadata=dict(artist='Me'))
N = 11
y = [x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], x[0][6], x[0][7], x[0][8], x[0][9], x[0][10]]
fig, ax = plt.subplots()
rects = plt.bar(range(N), y[0],  align='center')
xticks = ['Cello', 'Clarinet', 'Flute', 'A. Guitar', 'E. Guitar', 'Organ', 'Piano', 'Sax', 'Trumpet', 'Violin', 'Voice']
plt.xticks(range(11), xticks, size=8, rotation=45)
plt.xlim([-1, 11])
plt.ylim([0, 1])
plt.rcParams['animation.ffmpeg_path'] = r'PATHTOFFMPEG'
ani = animation.FuncAnimation(fig, animated_barplot, blit=True, interval=(23*5),  repeat=False, frames=len(x))
# ani.save('test.mp4', writer=writer)

plt.show()
