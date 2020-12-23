from sklearn.svm import SVC
from dataloader import training_set, test_set
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


svm = SVC()
scaler = StandardScaler()

for batch in tqdm(training_set):
    X = batch[0]
    y = batch[1]
    svm.fit(X, y)