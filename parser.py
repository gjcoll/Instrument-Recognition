import glob, os, pickle
names = [os.path.basename(x) for x in glob.glob('C:/Users/John Dwyer/Desktop/School/SeniorDesignTutorial/Instrument-Recognition/Data Samples/**/*.wav', recursive=True)]
fileinfo={}
print(len(names));
for i in names:
    # cel cla flu gac gel org pia sax tru vio voi nod dru
    fileparams=[0,0,0,0,0,0,0,0,0,0,0,0,0]
    testparams=[0,0,0,0,0,0,0,0,0,0,0,0,0]
    if '[cel]' in i:
        fileparams[0] = 1
    if '[cla]' in i:
        if fileparams != testparams:
            continue
        fileparams[1] = 1
    if '[flu]' in i:
        fileparams[2] = 1
    if '[gac]' in i:
        fileparams[3] = 1
    if '[gel]' in i:
        fileparams[4] = 1
    if '[org]' in i:
        fileparams[5] = 1
    if '[pia]' in i:
        fileparams[6] = 1
    if '[sax]' in i:
        fileparams[7] = 1
    if '[tru]' in i:
        fileparams[8] = 1
    if '[vio]' in i:
        fileparams[9] = 1
    if '[voi]' in i:
        fileparams[10] = 1
    if '[nod]' in i:
        fileparams[11] = 1
    if '[dru]' in i:
        fileparams[12] = 1
    fileinfo[i]=fileparams

# Simple pickle, not sure what the end goal with this is but I can change it up to what we need
pickle_out=open("datadict.pickle","wb")
pickle.dump(fileinfo, pickle_out)
pickle_out.close()
pickle_in=open("datadict.pickle","rb")
example=pickle.load(pickle_in)
# Example call to get the dict value of a filename out of a dict
print(example['008__[cel][nod][cla]0058__1.wav'])