from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))
print(model.score(X_test, y_test))