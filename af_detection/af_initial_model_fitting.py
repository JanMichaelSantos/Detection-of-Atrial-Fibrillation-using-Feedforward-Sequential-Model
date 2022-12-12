'''
Training of model
'''

import os
import time
import pickle
from matplotlib import cm
from matplotlib.cbook import flatten
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
import keras.metrics as Kmetrics
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix

seed = 1337
np.random.seed(seed)

headless = False

# load the npz file
data_path = 'F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection/DATASETS/training_and_validation.npz'
af_data   = np.load(data_path)

# extract the training and validation data sets from this data
x_train = af_data['x_train']
y_train = af_data['y_train']
x_test  = af_data['x_test']
y_test  = af_data['y_test']

# set the model parameters
n_timesteps = x_train.shape[1]
n_col = len(x_train[0])
mode = 'concat'
n_epochs = 1000 #1000
batch_size = int(1024*128) #n rows

# sequential
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=n_col))
model.add(Dropout(0.10))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  #sigmoid classification layer

# set the optimiser
opt = Adam()

# compile the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

# set up a model checkpoint callback (including making the directory where to save our weights)
directory = 'F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection/DATASETS/model/initial_runs_{0}/'.format(time.strftime("%Y%m%d_%H%M"))
os.makedirs(directory)
filename  = 'af_sequence_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpointer = ModelCheckpoint(filepath=directory+filename, 
                               verbose=1, 
                               save_best_only=True)

# fit the model
history = model.fit(x_train, y_train,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=[checkpointer])

# get the best validation accuracy
best_accuracy = max(history.history['val_acc'])
print('best validation accuracy = {0:f}'.format(best_accuracy))

# pickle the history so we can use it later
with open(directory + 'training_history', 'wb') as file:
    pickle.dump(history.history, file)

# set matplotlib to use a backend that doesn't need a display if we are 
# running remotely
if headless:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# plot the results

#change working directory to save on folder Plots
os.chdir('F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection/')

f1 = plt.figure()
ax1 = f1.add_subplot(111)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training and Validation Accuracy of AFib diagnosis')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.text(0.4, 0.05, 
         ('Validation Accuracy = {0:.3f}'.format(best_accuracy)), 
         ha='left', va='center', 
         transform=ax1.transAxes)
plt.savefig('Plots/af_sequence_training_accuracy_{0:.3f}%_.png'
            .format(best_accuracy*100)) #time.strftime("%Y%m%d_%H%M")
plt.show()

# loss
f2 = plt.figure()
ax2 = f2.add_subplot(111)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss of AFib Diagnosis')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.text(0.4, 0.05, 
         ('Validation Loss = {0:.3f}'
        .format(min(history.history['val_loss']))), 
         ha='right', va='top', 
         transform=ax2.transAxes)
plt.savefig('Plots/af_sequence_training_loss_{0:.3f}%_.png'
            .format(best_accuracy*100))

print('done')