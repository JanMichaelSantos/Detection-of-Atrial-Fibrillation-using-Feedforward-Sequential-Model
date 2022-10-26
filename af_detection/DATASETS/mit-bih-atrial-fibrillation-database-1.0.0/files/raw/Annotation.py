import os
import glob
import wfdb as wf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


cwd = os.getcwd()
os.chdir(cwd +'/raw')


annotation = wf.rdann('04015', extension = 'atr', pn_dir= 'afdb')
print(annotation)
#path = str(annotation) + ".txt"
#np.savetxt(path,annotation,delimiter=",")


#record = wf.rdrecord('04043')
#wf.plot_wfdb(record = record, annotation= annotation, title = 'Record 04043', time_units='seconds')

