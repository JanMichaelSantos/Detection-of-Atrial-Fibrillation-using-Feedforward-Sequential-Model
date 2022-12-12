import os
import time
import pickle
import numpy as np

# import keras deep learning functionality
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import GlobalMaxPool1D
from keras.layers import Dense
from keras.layers import Dropout

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

# fix random seed for reproduciblity
seed = 1337
np.random.seed(seed)

#
# get the data
#

# load the npz file
data_path = 'F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection/DATASETS/training_data.npz'
af_data   = np.load(data_path)

# extract the training and validation data sets from this data
x_data = af_data['x_data']
y_data = af_data['y_data']

#
# create and train the model
#

# set the model parameters
n_timesteps = x_data.shape[1]
n_col = len(x_data[0])
mode = 'concat'
n_epochs = 1000 
batch_size = int(1024*256) #n rows

# SQEUENTIAL
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=n_col))
model.add(Dropout(0.10))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
#model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  #sigmoid


# set the optimiser
opt = Adam()

# compile the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

# get the initial weights
initial_weights = model.get_weights()

#
# do stratified k-fold crossvalidation on the model
#

# progress ...
print('doing cross validation ...')

# set the root directory for results
results_dir = 'F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection/model/cross_validation_{0}/'.format(
        time.strftime("%Y%m%d_%H%M"))

# import stratified k-fold functions
from sklearn.model_selection import StratifiedKFold

# create the kfold object with 10 splits
n_folds = 10
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

# store some data from the crossvalidation process
xval_history = list()
final_accuracy = list()
final_loss = list()

# do cross validation
fold = 0
for train_index, test_index in kf.split(x_data, y_data):
  
    # progress ...
    print('evaluating fold {0}'.format(fold))
    
    # set up a model checkpoint callback (including making the directory where  
    # to save our weights)
    directory = results_dir + 'fold_{0}/'.format(fold)
    os.makedirs(directory)
    filename  = 'af_sequence_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpointer = ModelCheckpoint(filepath=directory+filename, 
                                   verbose=0, 
                                   save_best_only=True)    
    # get train and test data
    x_train, y_train = x_data[train_index], y_data[train_index]
    x_test, y_test   = x_data[test_index], y_data[test_index]    
    
    # train the model
    history = model.fit(x_train, 
                        y_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        verbose=0,
                        validation_data=(x_test, y_test),
                        callbacks=[checkpointer])    
    
    # run the final model on the validation data and save the predictions and
    # the true labels so we can plot a roc curve at a later date (af_crossvalidation_evaluation)
    y_predict = model.predict(x_test, batch_size=batch_size, verbose=0)
    np.save(directory + 'test_predictions.npy', y_predict)
    np.save(directory + 'test_labels.npy', y_test)

    # store the training history
    xval_history.append(history.history)
    
    # print the validation result
    final_loss.append(history.history['val_loss'][-1])
    final_accuracy.append(history.history['val_acc'][-1])
    print('validation loss is {0} and accuracy is {1}'.format(final_loss[-1],
          final_accuracy[-1]))
    
    # reset the model weights
    model.set_weights(initial_weights)
        
    # next fold ...
    fold = fold + 1

#
# tidy up ...
#




# print the final results
print('overall performance:')
print('{0:.5f}% (+/- {1:.5f}%)'.format(
        np.mean(final_accuracy), 
        np.std(final_accuracy))
     ) 

# pickle the entire cross validation history so we can use it later
with open(results_dir + 'xval_history', 'wb') as file:
    pickle.dump(xval_history, file)

# we're all done!
print('all done!')