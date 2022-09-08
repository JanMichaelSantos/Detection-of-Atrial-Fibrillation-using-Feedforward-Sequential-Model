import os
import wfdb as wf
import pandas as pd
import glob
import numpy as np
import csv


sps = 250 # samples per second of ecg recordings 

# set the directory where the data lives
froot = os.getcwd()
fpath = ('\DATASETS\csv_files\\chunk\\')
root_dir =  froot + fpath

ftarget = ('\DATASETS\csv_files\\filtered\\')

# get the patient IDs
filelist = glob.glob(root_dir + '*.csv', recursive=True)
patients  = [(os.path.split(i)[1]).split('.')[0] for i in filelist]


for n in range(0,len(filelist)):
    print("Working on: " + filelist[n])
    for i,chunk in enumerate(pd.read_csv(filelist[n], chunksize=460000)):
        chunk.to_csv(froot+ftarget+patients[n] +'_chunk{}.csv'.format(i), index=False)


            
            


