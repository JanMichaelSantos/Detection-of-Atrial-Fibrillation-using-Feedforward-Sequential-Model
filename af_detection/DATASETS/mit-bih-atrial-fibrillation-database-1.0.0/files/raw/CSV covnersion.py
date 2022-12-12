import os
import wfdb as wf
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#change working directory
cwd = os.getcwd()
os.chdir(cwd +'/raw')
cwd = os.getcwd()

#convert .dat to csv
dat_files=glob.glob('*.dat') #Get list of all .dat files in the current folder
df  = pd.DataFrame(data = dat_files)
df.to_csv("dat_files_list.csv",index=False, header=None) #Write the list to a CSV file
files = pd.read_csv("dat_files_list.csv",header=None)

for i in range(0,len(files)+1):
    recordname=str(files.iloc[[i]])
    print(recordname[-9:-4])
    recordname_new=recordname[-9:-4] #Extracting just the filename part
    print(recordname_new)

    record = wf.rdsamp(recordname_new) # rdsamp() returns the signal as a numpy array  
    record = np.asarray(record[0])
    path=recordname_new+".csv"
    np.savetxt(path,record,delimiter=",") #Writing the CSV for each record
    print("Files done: %s/%s"% (i,len(files)))

print("Files done: %s/%s"% (i,len(files)))

#print("\nAll files done!")	

#read signals
#signal = wf.rdrecord('04015')
#ann = wf.rdann('04015', extension='atr')

#wf.plot_wfdb(record = signal, annotation = ann, plot_sym=True, time_units='seconds', title= 'MIT-BIH Record 04015', figsize= (10,4), ecg_grids='all' )
