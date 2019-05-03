import numpy as np
from sklearn import linear_model, datasets

iris = datasets.load_breast_cancer()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data
y = iris.target

X = X[:-1]
y = y[:-1]

y [y==0] = -1



print(X.shape, X[0].shape)
print(y.shape, y[0].shape)

Xs = np.split(X,4)
ys = np.split(y,4)
X_train = np.concatenate((Xs[0], Xs[1], Xs[2]))
X_test = Xs[3]

y_train = np.concatenate((ys[0], ys[1], ys[2]))
y_test = ys[3]


clf = linear_model.SGDClassifier(alpha=1.0, max_iter=20000, epsilon=0.1)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

acc_count = 0
for i, y in enumerate(preds):
    if (y_test[i] == (preds[i])):
        acc_count += 1

print("Accuracy : " + str((float(acc_count) / float(len(preds)) * 100)))
# 75.3521126761 default sgd output from SvmIrisSGD.py
