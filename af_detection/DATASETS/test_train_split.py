'''
Process data stored on csv file and split it between test and train data to be used for model fitting
'''

from cgi import print_directory
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# fix random seed for reproduciblity
seed = 1337
np.random.seed(seed)

# set the directory where the data lives
froot = 'F:'
fpath = ('/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection/DATASETS/w_500_balanced/')
root_dir =  froot + fpath

print(root_dir)
# get the patient IDs
filelist = glob.glob(root_dir + '*.csv', recursive=True)
patients  = [(os.path.split(i)[1]).split('.')[0] for i in filelist]

# read all the data into a single data frame
frames = [pd.read_csv(p, header=None) for p in filelist]
data   = pd.concat(frames, ignore_index=True)
    
# show what our data looks like
print('we have {0} data points from 23 patients!'.format(data.shape))

# split the data into variables and targets (0 = no af, 1 = af)
x_data = data.iloc[:,1:].values     #WINDOWED
y_data = data.iloc[:,0].values      #LABELS
y_data[y_data > 0] = 1

# save the data with numpy so we can use it later
datafile = './DATASETS/training_data.npz'
np.savez(datafile, 
         x_data=x_data, y_data=y_data)

# count the number of samples exhibitning af
print('there are {0} out of {1} samples that have at least one beat that '
      'is classified as atrial fibrillation'.format(sum(y_data), len(y_data)))

# split the data into training (75%) and validation (25%) to use in our 
# initial model development
#
# note: when we start to properly fine tune the models we should use 10-fold 
# cross validation to evaluate the effects of the model structure and
# parameters 
#
print('Create Test/Train Split')
# create train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    stratify=y_data, 
                                                    test_size=0.25,
                                                    random_state=seed)

# reformat the training and test inputs to be in the format that the lstm 
# wants

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# save the data with numpy so we can use it later
datafile = './DATASETS/training_and_validation.npz'
np.savez(datafile, 
         x_train=x_train, x_test=x_test, 
         y_train=y_train, y_test=y_test)

print('NPZ files successfully saved')