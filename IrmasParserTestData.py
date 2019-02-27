import glob, os, pickle
names = [os.path.basename(x) for x in glob.glob('C:/Users/John Dwyer/Desktop/School/SeniorDesignTutorial/Instrument-Recognition/Test Data/**/*.wav', recursive=True)]
fileinfo={}
i=0;
os.chdir("C:/Users/John Dwyer/Desktop/School/SeniorDesignTutorial/Instrument-Recognition/Test Data")
for file in os.listdir("C:/Users/John Dwyer/Desktop/School/SeniorDesignTutorial/Instrument-Recognition/Test Data"):
    if file.endswith(".txt"):
        with open(file) as myfile:
            # print(myfile.read())
            # cel cla flu gac gel org pia sax tru vio voi nod dru pop_roc bvgvcf jaz_blu cou_fol lat_sou
            fileparams = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for line in myfile.readlines():

                if 'cel' in line:
                    fileparams[0] = 1
                if 'cla' in line:
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
                if 'pop_roc' in line:
                    fileparams[13] = 1
                if 'jaz_blu' in line:
                    fileparams[14] = 1
                if 'cou_fol' in line:
                    fileparams[15] = 1
                if 'lat_sou' in line:
                    fileparams[16] = 1
            fileinfo[names[i]]=fileparams
            i=i+1;

# Simple pickle, not sure what the end goal with this is but I can change it up to what we need
pickle_out=open("datadict.pickle","wb")
pickle.dump(fileinfo, pickle_out)
pickle_out.close()
pickle_in=open("datadict.pickle","rb")
example=pickle.load(pickle_in)
# Example call to get the dict value of a filename out of a dict
print(example['01 - Chet Baker - Prayer For The Newborn-8.wav'])