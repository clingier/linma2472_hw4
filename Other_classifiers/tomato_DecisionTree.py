import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sn
from sklearn.tree import DecisionTreeClassifier

with open('Xtrain.npy', 'rb') as f:
    Xtrain = np.load(f)
with open('ytrain.npy', 'rb') as f:
    ytrain = np.load(f)
with open('Xtest.npy', 'rb') as f:
    Xtest = np.load(f)
with open('ytest.npy', 'rb') as f:
    ytest = np.load(f)

nsamples, nx, ny, nz = Xtrain.shape
Xtrain = Xtrain.reshape((nsamples,nx*ny*nz))
nsamples, nx, ny, nz = Xtest.shape
Xtest = Xtest.reshape((nsamples,nx*ny*nz))

clf = DecisionTreeClassifier(random_state=0)
clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtest)

confusion_matrix = metrics.confusion_matrix(y_true=ytest, y_pred=ypred)

Conf = pd.DataFrame(confusion_matrix)
plt.figure(figsize = (8,8))
labels = ['Bacterial spot', 
          'early blight', 
          'healty', 
          'late blight', 
          'leaf mold', 
          'septoria leaf spot', 
          'spider mite', 
          'target spot', 
          'ToMV', 
          'TYCLV']
ax = sn.heatmap(Conf,xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt='g', cbar = False)
ax.set_xticklabels(labels, rotation=45)
ax.set(title='')
ax.set_xlabel(xlabel='predicted label', fontsize='large', fontweight='bold')
ax.set_ylabel(ylabel='true label', fontsize='large', fontweight='bold')

metrics.accuracy_score(ytest, ypred, normalize=True, sample_weight=None)
