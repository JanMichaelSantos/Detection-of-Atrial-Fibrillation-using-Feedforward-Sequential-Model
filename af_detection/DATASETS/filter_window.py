import os
import wfdb as wf
import pandas as pd
import glob
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import csv

sps = 250 # samples per second of ecg recordings 

# set the directory where the data lives
froot = os.getcwd()
fpath = ('\DATASETS\csv_files\\chunks\\')
root_dir =  froot + fpath

#directory where windowed data will be saved
ftarget1 = ('\DATASETS\csv_files\windowed\column1\\')
ftarget2 = ('\DATASETS\csv_files\windowed\column2\\')

#make directory if target path does not exist
if os.path.exists(froot+ftarget1) is False:
    os.makedirs(froot+ftarget1)
if os.path.exists(froot+ftarget2) is False:
    os.makedirs(froot+ftarget2)

# get the patient IDs
filelist = glob.glob(root_dir + '*.csv', recursive=True)
patients  = [(os.path.split(i)[1]).split('.')[0] for i in filelist]

#Highpass Butterworth filter, 0.5Hz cutoff
hpf = signal.butter(6, 0.5, 'hp', fs=250, output='sos')

#Lowpass Butterworth filter, 60Hz cutoff
lpf = signal.butter(6, 60, 'lp', fs=250, output='sos')

n_rows = 64000

for n in range(0,len(filelist)):
    print("Working on: " + patients[n])
    df = pd.read_csv(filelist[n])
    data_c1 = np.asarray(df.iloc[:,0])
    data_c2 = np.asarray(df.iloc[:,1])

    data_c1_bp = signal.sosfilt(lpf, signal.sosfilt(hpf, data_c1))
    data_c2_bp = signal.sosfilt(lpf, signal.sosfilt(hpf, data_c2))
    
    with open(froot+ftarget1+patients[n]+'_c1.csv','w+') as f:
        for i in range(1,n_rows-100):
            d_row = data_c1_bp[i:i+100] #get windowed data
            for d in d_row:
                f.write(str(d)+',')
            f.write("\n")
    f.close()

    with open(froot+ftarget2+patients[n]+'_c2.csv','w+') as f:
        for i in range(1,n_rows-100):
            d_row = data_c2_bp[i:i+100] #get windowed data
            for d in d_row:
                f.write(str(d)+',')
            f.write("\n")
    f.close()

    #For display only
    '''
    t = np.linspace(0, len(data_c1)/250, len(data_c1), False)
    plt.figure(1)
    plt.plot(t,data_c1)
    plt.plot(t,data_c1_bp)
    
    plt.figure(2)
    plt.plot(t,data_c2)
    plt.plot(t,data_c2_bp)
    '''
    #plt.show()






    

            
            


