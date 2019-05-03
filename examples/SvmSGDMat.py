from sklearn import svm, datasets
import numpy as np

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

def svm_sgd(X, Y):
    w = np.zeros(len(X[0]))
    eta = 0.001
    epochs = 10
    conditions = []

    for epoch in range(1,epochs):
        for i, x in enumerate(X):
            condition = Y[i]*np.dot(X[i], w)
            print("Cond1", condition)
            conditions.append(condition)
            if (condition) < 1:
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
            else:
                w = w + eta * (-2  *(1/epoch)* w)

    return w, conditions


def svm_sgd_mat(X,Y):
    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 10000
    conditions = []
    for epoch in range(1, epochs):
        condition = Y * np.dot(X,w)
        #print("Cond Shape", condition[0])
        sum = np.sum(condition)
        conditions.append(sum)
        if (sum < 1):
            b = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)
            print(b)
            w = w + eta * ( b +  (-2 * (1 / epoch) * w))
        else:
            w = w + eta * (-2 * (1 / epoch) * w)

    return w, conditions


def svm_sgd_predict(X,w):
    print(w.shape)
    print(X.shape)
    y_pred = []

    for i, x in enumerate(X):
        y_pred.append(np.sign(np.dot(X[i],w)))
    return y_pred


def get_accuracy(X_test, w):
    y_pred = svm_sgd_predict(X_test, w)
    print(y_pred)
    acc_count = 0
    for i, y in enumerate(y_pred):
        if (y_test[i] == (y_pred[i])):
            acc_count += 1
    print("Accuracy : " + str((float(acc_count) / float(len(y_pred)) * 100)))


w1, conditions1 = svm_sgd_mat(X_train,y_train)
w2, conditions2 = svm_sgd(X_train, y_train)
#get_accuracy(X_test=X_test, w=w1)
#print(w1)
#print("------------------")
#print(w2)

