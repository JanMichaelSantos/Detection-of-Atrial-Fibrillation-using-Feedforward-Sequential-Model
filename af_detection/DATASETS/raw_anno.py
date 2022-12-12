'''
This file can be used for visualization purposes of the Raw ECG data, 
Annotated data, and Location of R-peaks
'''
from time import sleep, time
import os
import wfdb
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import signal as signal
import random
from datetime import datetime

#change working directory
froot = 'F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection'
os.chdir(froot+"/DATASETS/mit-bih-atrial-fibrillation-database-1.0.0/files/")

#Get list of all .dat files in the current folder
dat_files=glob.glob('*.dat') 
patients = [(os.path.split(i)[1]).split('.')[0] for i in dat_files]

ftarget_f_n=  froot + '/DATASETS/csv_files/'

for _ in range(1):
    i = 3
    #INIT RECORD WINDOW SIZE OF 500 SAMPLES
    window_size = 512
    n_windows = 3
    afib_window = np.zeros(window_size)
    nafib_window = np.zeros(window_size)

    print('Time Start: ', datetime.now())
    print('Patient: ',patients[i],' Progress: '+str(i+1)+"/" +str(len(dat_files)))
    #Get ECG1 Data
    record,fields = wfdb.rdsamp(patients[i])
    record = record[:,0] #c1 = ecg1, c2= ecg2, only get ecg1
    
    #Read ATR Annotation File
    record_atr = wfdb.rdann(patients[i],extension ='atr',shift_samps=True)

    #Read QRSC Annotation File
    record_qrsc = wfdb.rdann(patients[i],extension ='qrs',shift_samps=True)
    
    # Plot raw data with annotations

    # Get 1 min recording
    # duration is 60s sampled at 250 samples per second       
    duration = 60*250
    record = record[0:duration]
    atr = record_atr.sample
    
    plt.figure(1,figsize=(8,5))
    plt.title('1 MIN RECORDING OF PATIENT 04126')

    # Plot NORMAL and AFIB WAVEFORMS with different colors
    for anno_i in range(len(atr)-1):

        start = atr[anno_i]
        end = atr[anno_i+1]

        if end > duration: end  = duration

        n = np.linspace(start,end,len(record[start:end]))

        if  record_atr.aux_note[anno_i] == '(N':
            plt.plot(n,record[start:end], color = 'red')

        elif record_atr.aux_note[anno_i] == '(AFIB':
            plt.plot(n,record[start:end], color = 'green')
        

    # Plot R-peaks with dots
    qrsc = record_qrsc.sample
    qrsc = [x for x in qrsc if x <= duration]
    
    plt.scatter(qrsc, record[qrsc], marker = '*',color = 'blue')
    plt.xlabel('ECG Index')
    plt.ylabel('mV')
    plt.show()
