import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import pickle


with open('IRMAS_predictions_all.pkl', 'rb') as f:
    a = pickle.load(f)
x, b, c, d, e, f, g, u, aa, ab = [], [], [], [], [], [], [], [], [], [] #every hard coded interpolation point

length = len(a[0])
for i in range(length-1): #runs three times (1=first&2=second,1=second&2=third,1=third&2=fourth)
    listobj1 = a[0][i].tolist() #first pickle value (first array of values)
    listobj2 = a[0][i+1].tolist() #second pickle value (second array of values)
    for m in range(11):
        b.append(np.linspace(listobj1[m], listobj2[m], num=5, endpoint=False).tolist())#Interpolates 5 values, but no the actual second value (endpoint=false)
    for j in range(11):
        c.append(b[j][0]) #grabs first values from b
        d.append(b[j][1]) #grabs first interpolation from b
        e.append(b[j][2]) #so on and so forth
        f.append(b[j][3])
        g.append(b[j][4])
        # u.append(b[j][5])
        # aa.append(b[j][6])
        # ab.append(b[j][7])
    x.append(c) # adds values to main list so they can be drawn.
    x.append(d)
    x.append(e)
    x.append(f)
    x.append(g)
    # x.append(u)
    # x.append(aa)
    # x.append(ab)
    b, c, d, e, f, g, u, aa, ab = [], [], [], [], [], [], [], [], [] #clears all, otherwise the interpolations will just grow


def animated_barplot(frame):
    y = [x[frame][0], x[frame][1], x[frame][2], x[frame][3], x[frame][4], x[frame][5], x[frame][6], x[frame][7], x[frame][8], x[frame][9], x[frame][10]] #setting up the bar plots
    for rect, h in zip(rects, y):
        rect.set_height(h) #drawing the rectangle height
    return rects


Writer = animation.FFMpegWriter
writer = Writer(metadata=dict(artist='Me'))
N = 11
y = [x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], x[0][6], x[0][7], x[0][8], x[0][9], x[0][10]] #requires the first set be declared for some reason, couldn't avoid this
fig, ax = plt.subplots()
rects = plt.bar(range(N), y[0],  align='center')
xticks = ['Cello', 'Clarinet', 'Flute', 'A. Guitar', 'E. Guitar', 'Organ', 'Piano', 'Sax', 'Trumpet', 'Violin', 'Voice']
plt.xticks(range(11), xticks, size=8, rotation=45)
plt.xlim([-1, 11])
plt.ylim([0, 1])
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\John Dwyer\Downloads\ffmpeg-20190508-06ba478-win64-static\bin\ffmpeg'
ani = animation.FuncAnimation(fig, animated_barplot, blit=True, interval=(23*5),  repeat=False, frames=len(x)) #frames is the length of x, so for each array on frame, and interval is 23*5 because of the five interpolation steps
ani.save('test.mp4', writer=writer)

plt.show()
