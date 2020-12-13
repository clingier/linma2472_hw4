import PIL
import numpy as np
from os import listdir
from matplotlib import image

##load the images##

LABEL_VALUE = {
    # 'Tomato___Bacterial_spot': 0,
    # 'Tomato___Early_blight': 1,
    # 'Tomato___healthy': 2,
    # 'Tomato___Late_blight': 3,
    # 'Tomato___Leaf_Mold': 4,
    # 'Tomato___Septoria_leaf_spot': 5,
    'Tomato___Spider_mites_Two-spotted_spider_mite': 6,
    # 'Tomato___Target_Spot': 7,
    # 'Tomato___Tomato_mosaic_virus': 8,
    # 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 9,
}

Xtrain = np.empty((0, 64, 64, 3), int)
ytrain = np.array([])

Xtest = np.empty((0, 64, 64, 3), int)
ytest = np.array([])

for folder, value in LABEL_VALUE.items():
    for filename in listdir('train/' + folder):
        Xtrain = np.append(Xtrain, [np.array(PIL.Image.open('train/' + folder + '/' + filename).resize((64, 64), PIL.Image.ANTIALIAS))], axis=0)
        ytrain = np.append(ytrain, [value])
    for filename in listdir('test/' + folder):
        Xtest = np.append(Xtest, [np.array(PIL.Image.open('test/' + folder + '/' + filename).resize((64, 64), PIL.Image.ANTIALIAS))], axis=0)
        ytest = np.append(ytest, [value])
        
##create a train and test database##

shuffler = np.random.permutation(len(Xtrain))
Xtrain = Xtrain[shuffler]
ytrain = ytrain[shuffler]

shuffler_2 = np.random.permutation(len(Xtest))
Xtest = Xtest[shuffler_2]
ytest = ytest[shuffler_2]

X = np.concatenate([Xtrain, Xtest], axis = 0)
y = np.concatenate([ytrain, ytest])

with open('Xtrain.npy', 'wb') as f:
    np.save(f, Xtrain)
with open('ytrain.npy', 'wb') as f:
    np.save(f, ytrain)
    
with open('Xtest.npy', 'wb') as f:
    np.save(f, Xtest)
with open('ytest.npy', 'wb') as f:
    np.save(f, ytest)

with open('X.npy', 'wb') as f:
    np.save(f, X)
with open('y.npy', 'wb') as f:
    np.save(f, y)

##############################################################################
