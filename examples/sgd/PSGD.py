import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
import numpy as np
from comms import Communication

X = np.array([
    [-2.0, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
    [-2.1, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
    [-2.2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
    [-2.3, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],

],dtype='f')

y = np.array([-1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1], dtype='f')


def svm_sgd(X, Y):
    w = np.zeros(len(X[0]),'f')
    eta = 1
    epochs = 10

    for epoch in range(1, epochs):
        for i, x in enumerate(X):
            if (Y[i] * np.dot(X[i], w)) < 1:
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1 / epoch) * w))
            else:
                w = w + eta * (-2 * (1 / epoch) * w)
    return w


def svm_psgd(X, y):
    comms = Communication.Communication()
    rank = comms.comm.Get_rank()
    size = comms.comm.Get_size()
    m = len(X)
    partition_size = m / size
    eta = 1
    epochs = 0
    T = 10
    if size > 1:
        epochs = 5
    else:
        epochs = T
    print("World Size : ", size, epochs)
    start = rank * partition_size
    end = start + partition_size
    X_p = X[start:end,:]
    y_p = y[start:end]
    m1 = len(X_p[0])


    w = np.zeros(m1, 'f')
    grad = np.zeros(m1, 'f')
    grad_r = np.zeros(m1, 'f')

    for epoch in range(1, epochs):
        for i, x in enumerate(X_p):
            if (y_p[i] * np.dot(X_p[i], w)) < 1:
                grad = ((X_p[i] * y_p[i]) + (-2 * (1 / epoch) * w))
            else:
                grad = (-2 * (1 / epoch) * w)
            comms.allreduce(input=grad, output=grad_r, op=comms.mpi.SUM, dtype=comms.mpi.FLOAT)
            print(rank, epoch, grad, grad_r, w)
            if (rank == 0):
                g_global = 0
                if(size > 1):
                    g_global = grad_r / (size * partition_size)
                else :
                    g_global = grad_r / (size)
                w = w + eta * g_global
                comms.bcast(input=w, dtype=comms.mpi.FLOAT, root=0)


    return rank,w





rank,wp = svm_psgd(X,y)
ws = svm_sgd(X, y)
if(rank==0):
    print(ws,wp)
