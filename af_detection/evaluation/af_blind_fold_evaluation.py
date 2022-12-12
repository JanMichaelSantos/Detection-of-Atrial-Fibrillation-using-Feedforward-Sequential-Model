
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
data_path = 'F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection/DATASETS/test_data.npz'
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
model.add(Dense(256, activation='relu', input_dim=n_col))
model.add(Dropout(0.10))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  #sigmoid

#
# load saved weights
#

# set the weights to load
weights_path = 'F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/model/initial_runs_20221026_2053/af_sequence_weights.986-0.14.hdf5'

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
np.save('F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection/results/blindfold_predictions.npy', y_predict)

#
# evaluate predictions
#

# accuracy
accuracy = accuracy_score(y_test, np.round(y_predict))
print('blind-fold accuracy is {0:.5f}'.format(accuracy))

# precision
precision = precision_score(y_test, np.round(y_predict))
print('blind-fold precision is {0:.5f}'.format(precision))

# sensitivity
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
print(cm[0][0])
print(cm[1][0])
print(cm[1][1])
print(cm[0][1])

true_negative =  cm[0][0]
false_positive = cm[0][1]

specificity = (true_negative/(true_negative+false_positive))*100

plt.figure(figsize=[5,5])
my_vis.plot_confusion_matrix(cm, 
                      classes=classes,
                      title=None)
plt.savefig('F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection/results/blind_confusion_plot.png',
            dpi=600, bbox_inches='tight', pad_inches=0.5)

plt.figure(figsize=[5,5])
my_vis.plot_confusion_matrix(cm, 
                      classes=classes,
                      normalize=True,
                      title=None)
plt.savefig('F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection/results/blind_confusion_plot_normalised.png',
            dpi=600, bbox_inches='tight', pad_inches=0.5)
plt.show()

recall = recall*100
precision = precision*100
f1 = f1*100
accuracy = accuracy*100

fname_stats = 'Metrics.csv'
with open (fname_stats,'w+') as f:
        f.write('Sensitivity,Specificity,Precision, F1-Score,Accuracy\n')
        f.write(str(recall)+',')
        f.write(str(specificity)+',')
        f.write(str(precision)+',')
        f.write(str(f1)+',')
        f.write(str(accuracy)+'\n')
f.close()


# calculate and plot the roc curve
plt.figure(figsize=[5,5])
title = 'Receiver operating characteristic curve showing ' \
        'AF diagnostic performance'
my_vis.plot_roc_curve(y_predict, y_test, title=None)        
plt.savefig('F:/1_COLLEGE/TERM 9/CAPSTONE/Capstone/af_detection/results/blind_roc_curve.png',
            dpi=600, bbox_inches='tight', pad_inches=0.5)
plt.show()