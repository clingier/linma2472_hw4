import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sn
from sklearn.neighbors import KNeighborsClassifier

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

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtest)

confusion_matrix = metrics.confusion_matrix(y_true=ytest, y_pred=ypred)

Conf = pd.DataFrame(confusion_matrix)
#Conf = pd.DataFrame(confusion_matrix, range(10), range(10))
plt.figure(figsize = (10,7))
#labels = ['S_Coral', 'H_coral', 'algae', 'Oth', 'Oth_inv']

ax = sn.heatmap(Conf, annot=True, cmap="viridis", fmt='g')
#ax = sn.heatmap(Conf,xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt='g')

ax.set(title="confusion matrix for random forest",xlabel="predicted label",ylabel="true label",)

metrics.accuracy_score(ytest, ypred, normalize=True, sample_weight=None)