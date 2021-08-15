from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

shuffle = np.random.randint(0, 150, 150)

X_train, y_train = datasets.load_iris(return_X_y=True)
X_train = X_train[shuffle]
y_train = y_train[shuffle]
X_train = X_train[:100]
y_train = y_train[:100]
X_test = X_train[-50:]
y_test = y_train[-50:]

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_train, y_train))

filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
