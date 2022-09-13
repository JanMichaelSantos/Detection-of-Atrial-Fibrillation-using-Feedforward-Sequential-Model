'''
https://physionet.org/content/afdb/1.0.0/
Reference used for filter cutoff frequencies (Data Description Section)

'''
'''
Read ECG dat files and create a csv file 
with each row having a filtered and normalized 3 cycles of pqrst waves
'''
from time import sleep
import os
import wfdb
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import signal as signal
import random

#change working directory
froot = 'E:'
os.chdir(froot+"/DATASETS/mit-bih-atrial-fibrillation-database-1.0.0/files/")

#Get list of all .dat files in the current folder
dat_files=glob.glob('*.dat') 
patients = [(os.path.split(i)[1]).split('.')[0] for i in dat_files]

ftarget_f_n=  froot + '/DATASETS/csv_files/'

#INIT RECORD WINDOW SIZE OF 500 SAMPLES
window_size = 500
afib_window = np.zeros(window_size)
nafib_window = np.zeros(window_size)
for i in range(0,len(dat_files)):
    print('Patient: ',patients[i],' Progress: '+str(i+1)+"/" +str(len(dat_files)))
    #Get ECG1 Data
    record,fields = wfdb.rdsamp(patients[i])
    record = record[:,0]

    #READ ATR ANNOTATION FILE
    record_atr = wfdb.rdann(patients[i],extension ='atr',shift_samps=True)
    #READ QRSC ANNOTATION FILE
    record_qrsc = wfdb.rdann(patients[i],extension ='qrs',shift_samps=True)
    
    #Get random indeces for RR
    n_sample = 20000
    indeces = list(range(1,len(record_qrsc.sample)-5))
    random.shuffle(indeces)
    indeces = indeces[0:n_sample]
    
    #Check if sample index is AFIB or N_AFIB
    for rr_index in indeces:
        rr_i = record_qrsc.sample[rr_index:rr_index+4]
        for atr_index in range(len(record_atr.sample)-1,0,-1):
            #Find AUX_NOTE of lower boundary
            if rr_i[3] < record_atr.sample[atr_index]:
                if  rr_i[0] >=  record_atr.sample[atr_index-1]:
                    #Apply processing on chose record samples
                    w1 = signal.resample(record[rr_i[0]:rr_i[3]],window_size)

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

                    #Make AFIB and N-AFIB data balanced
                    if len(nafib_window) < len(afib_window):
                        afib_window = afib_window[0:len(nafib_window)]
                    break # Exit Finding of ATR index loop 
                        
            else: 
                    pass

# SAVE AFIB DATA WITH LABELS TO CSV
afib_window = afib_window[1:,:]
path='AFIB.csv'
np.savetxt(ftarget_f_n+path, np.hstack((np.ones((len(afib_window),1)),afib_window)),fmt='%1.3f',delimiter=",")
print(len(afib_window), " rows, AFIB.csv dataset saved on ",ftarget_f_n)

# SAVE NO-AFIB DATA WITH LABELS TO CSV
path='N_AFIB.csv'
nafib_window = nafib_window[1:,:]
np.savetxt(ftarget_f_n+path, np.hstack((np.ones((len(nafib_window),1)),nafib_window)),fmt='%1.3f',delimiter=",")
print(len(nafib_window), " rows, N_AFIB.csv dataset saved on ",ftarget_f_n)

