import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.animation as animation
x = [np.linspace(0, 1, num=1000).tolist(), np.concatenate([np.linspace(0, 0.5, num=100, endpoint=False),
                                                                 np.linspace(0.5, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.7, num=100, endpoint=False),
                          np.linspace(0.7, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.2, num=100, endpoint=False),
                          np.linspace(0.2, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.1, num=100, endpoint=False),
                          np.linspace(0.1, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.5, num=100, endpoint=False),
                          np.linspace(0.5, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.05, num=100, endpoint=False),
                          np.linspace(0.05, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.55, num=100, endpoint=False),
                          np.linspace(0.55, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.5, num=100, endpoint=False),
                          np.linspace(0.5, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.5, num=100, endpoint=False),
                          np.linspace(0.5, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.5, num=100, endpoint=False),
                          np.linspace(0.5, 0, num=900)], axis=0).tolist()]


def animated_barplot(frame):
    x = [np.linspace(0, 1, num=1000).tolist(), np.concatenate([np.linspace(0, 0.5, num=100, endpoint=False),
                                                               np.linspace(0.5, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.7, num=100, endpoint=False),
                         np.linspace(0.7, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.2, num=100, endpoint=False),
                         np.linspace(0.2, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.1, num=100, endpoint=False),
                         np.linspace(0.1, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.5, num=100, endpoint=False),
                         np.linspace(0.5, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.05, num=100, endpoint=False),
                         np.linspace(0.05, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.55, num=100, endpoint=False),
                         np.linspace(0.55, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.5, num=100, endpoint=False),
                         np.linspace(0.5, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.5, num=100, endpoint=False),
                         np.linspace(0.5, 0, num=900)], axis=0).tolist(),
         np.concatenate([np.linspace(0, 0.5, num=100, endpoint=False),
                         np.linspace(0.5, 0, num=900)], axis=0).tolist()]
    y = [x[0][frame], x[1][frame], x[2][frame], x[3][frame], x[4][frame], x[5][frame], x[6][frame], x[7][frame], x[8][frame], x[9][frame], x[10][frame]]
    for rect, h in zip(rects, y):
        rect.set_height(h)
    time.sleep(512 / 22050)
    return rects


Writer = animation.FFMpegWriter
writer = Writer(fps=15, metadata=dict(artist='Me'))
N = 11
y = [x[0][0], x[1][0], x[2][0], x[3][0], x[4][0], x[5][0], x[6][0], x[7][0], x[8][0], x[9][0], x[10][0]]
fig, ax = plt.subplots()
rects = plt.bar(range(N), y[0],  align='center')
frames = 1000
plt.xlim([-1, 11])
plt.ylim([0, 1])
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\John Dwyer\Downloads\ffmpeg-20190508-06ba478-win64-static\bin\ffmpeg'
ani = animation.FuncAnimation(fig, animated_barplot, blit=True, interval=0.5, frames=frames, repeat=False)
ani.save('test.mp4', writer=writer)

plt.show()
