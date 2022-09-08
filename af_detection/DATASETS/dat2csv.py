import os
import wfdb
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#change working directory
cwd = os.getcwd()
os.chdir(cwd+'/DATASETS/mit-bih-atrial-fibrillation-database-1.0.0/files/')

#Get list of all .dat files in the current folder
dat_files=glob.glob('*.dat') 
patients  = [(os.path.split(i)[1]).split('.')[0] for i in dat_files]

for i in range(0,len(dat_files)):
    print("Progress: "+str(i+1)+"/" +str(len(dat_files)))
    record,fields = wfdb.rdsamp(patients[i]) # rdsamp() returns the signal as a numpy array  
    print(fields)
    input()
    record = np.asarray(record[0])
    path=patients[i]+".csv"
    np.savetxt(path,record,delimiter=",") #Writing the CSV for each record
    