from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import load
from tensorflow import keras
import numpy as np
from tqdm import tqdm
from dataloader import test_path
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

svc = load('svm.joblib')
r_f = load('r_forest.joblib')
model = keras.models.load_model('models/simple_model_1608571698.2372336.h5')
TEST_BATCH_SIZE = 256


test_gen = ImageDataGenerator()

test_set = test_gen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=TEST_BATCH_SIZE,
)


with open('Xtest.npy', 'rb') as f:
    Xtest = np.load(f)
with open('ytest.npy', 'rb') as f:
    ytest = np.load(f)


nsamples, nx, ny, nz = Xtest.shape
Xtest = Xtest.reshape((nsamples,nx*ny*nz))

ypred_svc = svc.predict(Xtest)
ypred_rf = r_f.predict(Xtest)

#####################################Get the predictions and the labels 
# is pretty hard with keras api, had to cheat a bit

ypred_cnn = np.array([])
ytest_cnn = np.array([])

def generator_with_true_classes(model, generator):
    while True:
        x, y = next(generator)
        yield x, model.predict(x), y

nb_of_samples = 0
for x, y_pred, y_true in tqdm(generator_with_true_classes(model, test_set)):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    ypred_cnn = np.append(ypred_cnn, y_pred)
    ytest_cnn = np.append(ytest_cnn, y_true)
    # do something with data, eg. print it.
    nb_of_samples += TEST_BATCH_SIZE
    print(nb_of_samples)
    if nb_of_samples >= 984:
         break

for i, ytest, ypred in zip(range(3), [ypred_svc, ypred_rf, ypred_cnn], [ytest, ytest, ytest_cnn]):
    cm = confusion_matrix(y_true=ytest, y_pred=ypred)

    Conf = pd.DataFrame(cm)
    plt.figure(figsize = (8,8))
    labels = ['Bacterial', 
            'Early blight', 
            'Healty', 
            'Late blight', 
            'Leaf mold', 
            'Septoria spot', 
            'Spider mite', 
            'Target spot', 
            'TMV', 
            'TYCLV']

    ax = sn.heatmap(Conf,xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt='g', cbar = False)
    ax.set_xticklabels(labels, rotation=45)
    ax.set(title='')
    ax.set_xlabel(xlabel='predicted label', fontsize='large', fontweight='bold')
    ax.set_ylabel(ylabel='true label', fontsize='large', fontweight='bold')
    plt.savefig(f"plots/confusion_matrix_{i}.pdf")
    plt.show()
    print(accuracy_score(ytest, ypred))
