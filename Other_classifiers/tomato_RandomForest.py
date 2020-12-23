import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

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

clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(Xtrain, ytrain)

s = dump(clf, 'r_forest.joblib')