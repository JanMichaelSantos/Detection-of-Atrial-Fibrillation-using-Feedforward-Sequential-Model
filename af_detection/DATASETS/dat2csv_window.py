'''
Read ECG dat files and create a csv file 
with each row having a filtered and normalized 3 RR cycle
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

# NOTE: THESE PATIENTS ONLY HAS 1 LABEL
# patient[13] aux note label is only (N, all data is normal
# patient[14] aux note label is only (AFIB, all data is atrial fib

for i in range(15,len(dat_files)):
#for i in range(x,x+1):
    #INIT Record window size of 500 samples
    window_size = 512
    n_windows = 3
    afib_window = np.zeros(window_size)
    nafib_window = np.zeros(window_size)

    print('Time Start: ', datetime.now())
    print('Patient: ',patients[i],' Progress: '+str(i+1)+"/" +str(len(dat_files)))
    #Get ECG1 Data
    record,fields = wfdb.rdsamp(patients[i])
    record = record[:,0] #c1 = ecg1, c2= ecg2, only get ecg1

    #Read ATR annotation file
    record_atr = wfdb.rdann(patients[i],extension ='atr',shift_samps=True)

    #Read QRSC annotation file
    record_qrsc = wfdb.rdann(patients[i],extension ='qrs',shift_samps=True)
    
    #Get random indeces for RR
    indeces = list(range(1,len(record_qrsc.sample)-n_windows-1))
    #n_sample = len(record_qrsc.sample) #MAX length of record_qrsc.sample

    random.shuffle(indeces)

    #Check if sample index is AFIB or N_AFIB
    for rr_index in indeces:
        rr_i = record_qrsc.sample[rr_index:rr_index+(n_windows+1)]  #get n cycles
        for atr_index in range(0,len(record_atr.sample)-1):
            #Find AUX_NOTE of lower boundary
            if rr_i[-1] < record_atr.sample[atr_index+1] and rr_i[0] >=  record_atr.sample[atr_index]: #Check if window is between N and AFIB
                #Apply processing on chosen record samples
                w1 = signal.resample(record[rr_i[0]:rr_i[-1]],window_size)

                #Filter Signals
                fs_n = 250
                hpf = signal.butter(6, 1, 'high', fs=fs_n, output='sos') #Highpass Butterworth filter, 1Hz cutoff
                lpf = signal.butter(6, 35, 'low', fs=fs_n, output='sos') #Lowpass Butterworth filter, 35Hz cutoff

                record_f = signal.sosfilt(lpf, signal.sosfilt(hpf, w1)) #Apply filter

                #Normalize Signals
                record_f_2c = np.column_stack((record_f,record_f))
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(record_f_2c)
                record_f_n = scaler.transform(record_f_2c)

                if  record_atr.aux_note[atr_index] == '(N': 
                    nafib_window = np.vstack((nafib_window ,record_f_n[:,0]))
                elif record_atr.aux_note[atr_index] == '(AFIB': 
                    afib_window = np.vstack((afib_window ,record_f_n[:,0]))

                break # Exit Finding of ATR index loop 
                    
            else: 
                    pass
    print('Current Size AFIB: ', np.shape(afib_window))
    print('Current Size N: ', np.shape(nafib_window))

    afib_window = afib_window[1:,:]
    nafib_window = nafib_window[1:,:]

    #Save AFIB data with labels to CSV
    path=str(patients[i])+'_AFIB.csv'
    np.savetxt(ftarget_f_n+path, np.hstack((np.ones((len(afib_window),1)),afib_window)),fmt='%1.3f',delimiter=",")
    print(len(afib_window), " rows, AFIB.csv dataset saved on ",ftarget_f_n)

    #Save NO-AFIB data with labels to CSV
    path=str(patients[i])+'_N_AFIB.csv'
    np.savetxt(ftarget_f_n+path, np.hstack((np.zeros((len(nafib_window),1)),nafib_window)),fmt='%1.3f',delimiter=",")
    print(len(nafib_window), " rows, N_AFIB.csv dataset saved on ",ftarget_f_n)

    print('Time End: ', datetime.now())