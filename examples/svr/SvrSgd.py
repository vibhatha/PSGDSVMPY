import numpy as np

X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],

])

y = np.array([1.2,2.1,1.3,1.9,1.6])

X_test = np.array([
    [-2.3,4.2,-1.2],
    [4.3,1.1,-1.5],
    [1.7, 6.1, -1.3],
    [2.1, 4.5, -1.6],
    [6.5, 2.6, -1.1],

])

y_test = np.array([1.7,2.3,1.6,1.5,1.9])

def get_accuracy(X_test,y_test, w):
    y_pred = svm_sgd_predict(X_test, w)
    print(y_pred)
    acc_count = 0
    tolerance = 0.1
    for i, y in enumerate(y_pred):
        diff = y_test[i] - y_pred[i]

        if (diff<=tolerance):
            acc_count += 1

    print("Accuracy : " + str((float(acc_count) / float(len(y_pred)) * 100)))


def svm_sgd_predict(X,w):
    print(w.shape)
    print(X.shape)
    y_pred = []
    for i, x in enumerate(X):
        y_pred.append((np.dot(X[i],w)))
    return y_pred

def svm_sgd(X, Y):

    w = np.zeros(len(X[0]))
    eta = 0.0001
    epochs = 100

    for epoch in range(1,epochs):
        for i, x in enumerate(X):
            if (Y[i]*np.dot(X[i], w)) < 1:
                w = w + eta * (w - (X[i] * Y[i]))
            else:
                w = w + eta * w

    return w

w = svm_sgd(X,y)
print(w)

get_accuracy(X_test, y_test, w)
