
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# my visualisation utilities for plotting some pretty graphs of classifier 
# performance
import visualisation_utils as my_vis

#
# get the data
#

# load the npz file
data_path = './DATASETS//test_data.npz'
af_data   = np.load(data_path)

# extract the inputs and labels
x_test  = af_data['x_data']
y_test  = af_data['y_data']

# set the model parameters
n_timesteps = x_test.shape[1]
n_col = len(x_test[0])
mode = 'concat'
n_epochs = 1000 
batch_size = int(1024*256) #n rows

# SQEUENTIAL
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=n_col))
model.add(Dropout(0.20))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#
# load saved weights
#

# set the weights to load
weights_path = './model/initial_runs_20220919_0206/af_sequence_weights.828-0.18.hdf5'

# load those weights and compile the model (which is needed before we can make
# any predictions)
model.load_weights(weights_path)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# note: although we have specified an optimiser in our model compilation, it
# is obviously not used ...

#
# predict the labels using the model and evaluate over a range of metrics
#

# use the model to predict the labels
y_predict = model.predict(x_test, batch_size=batch_size, verbose=0)
np.save('./results/blindfold_predictions.npy', y_predict)

#
# evaluate predictions
#

# accuracy
accuracy = accuracy_score(y_test, np.round(y_predict))
print('blind-fold accuracy is {0:.5f}'.format(accuracy))

# precision
precision = precision_score(y_test, np.round(y_predict))
print('blind-fold precision is {0:.5f}'.format(precision))

# recall
recall = recall_score(y_test, np.round(y_predict))
print('blind-fold recall is {0:.5f}'.format(recall))

# f1 score
f1 = f1_score(y_test, np.round(y_predict))
print('blind-fold f1 score is {0:.5f}'.format(f1))

# classification report
print(classification_report(y_test, np.round(y_predict), 
                            target_names=['normal', 'af']))

# set the names of the classes
classes = ['normal', 'af']

# get the confusion matrix and plot both the un-normalised and normalised
# confusion plots 
cm = confusion_matrix(y_test, np.round(y_predict))

plt.figure(figsize=[5,5])
my_vis.plot_confusion_matrix(cm, 
                      classes=classes,
                      title=None)
plt.savefig('./results/blind_confusion_plot.png',
            dpi=600, bbox_inches='tight', pad_inches=0.5)

plt.figure(figsize=[5,5])
my_vis.plot_confusion_matrix(cm, 
                      classes=classes,
                      normalize=True,
                      title=None)
plt.savefig('./results/blind_confusion_plot_normalised.png',
            dpi=600, bbox_inches='tight', pad_inches=0.5)
plt.show()

# calculate and plot the roc curve
plt.figure(figsize=[5,5])
title = 'Receiver operating characteristic curve showing ' \
        'AF diagnostic performance'
my_vis.plot_roc_curve(y_predict, y_test, title=None)        
plt.savefig('./results/blind_roc_curve.png',
            dpi=600, bbox_inches='tight', pad_inches=0.5)
plt.show()