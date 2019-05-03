import time
from dask.distributed import Client
import numpy as np

constval = 50000000


def inc(x):
    for i in range(0,constval):
        x = x+1
    return x


def dask_test1():
    client = Client()
    exec_time = 0
    exec_time -= time.time()
    a = client.submit(inc, 10)  # calls inc(10) in background thread or process
    b = client.submit(inc, 20)  # calls inc(20) in background thread or process
    print(a.result(), b.result())
    exec_time += time.time()
    print("Dask Time Taken : " + str(exec_time))

def normal_test1():
    exec_time = 0
    exec_time -= time.time()
    a1 = inc(10)
    b1 = inc(20)
    exec_time += time.time()
    print(a1, b1)
    print("Normal Time Taken : " + str(exec_time))


def inner_looper(i, X, B, epoch):
    for j, x in enumerate(X):
        if (i != j):
            kernel_value =  (X[i] - X[j]).T * (X[i] - X[j])
            B[epoch+1][i] = np.sum(np.sum(kernel_value, axis=0))
    return B

def task_parallel():
    start_time = 0
    start_time -= time.time()
    X = np.array(
        [[1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 33, 5, 3]])

    X = np.array([[1, 2], [2, 5], [3, 4], [5, 6], [8, 2], [1, 2], [2, 5], [3, 4], [5, 6], [8, 2]])
    B = np.zeros((10, len(X)))

    for epoch in range(0,10-1):
        for i,x in enumerate(X):
            B = inner_looper(i, X, B, epoch)


    print(np.sum(B[9]))
    start_time += time.time()
    print("Multi Process Time Taken : " + str(start_time))


def data_parallel():
    start_time = 0
    start_time -= time.time()
    X = np.array(
        [[1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1], [0, 1, 2, 55, 1],
         [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
         [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4],
         [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
         [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 33, 5, 3]])

    X = np.array([[1,2],[2,5],[3,4],[5,6],[8,2],[1,2],[2,5],[3,4],[5,6],[8,2]])

    print(X.shape)
    B = np.zeros((10, len(X)))
    epochs = 10
    epoch_list = np.arange(0,epochs-1)
    X_splits = np.split(X,5)
    B_splits = np.split(B,5)
    #epoch_list_splits = np.split(epoch_list,5)
    #client1 = Client(processes=True)
    tasks = []
    B_add = []

    sum = 0
    for Xs in X_splits:
        for epoch in range(0, 10 - 1):
            for i, x in enumerate(Xs):
                B = inner_looper(i, Xs, B, epoch)
        sum += np.sum(B[9])
    print(sum)
    start_time += time.time()
    print("Multi Process Time Taken : " + str(start_time))



def normal_loop_test():
    start_time = 0
    start_time -= time.time()
    X = np.array([[1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1,2,3,4,5],[0,1,3,4,5],[0,4,5,6,4],[0,1,3,4,5],[0,1,3,54,1],[0,1,2,55,1],[0,11,33,5,3],[0,11,11,3,1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1],
                  [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 11, 3, 1], [1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5, 6, 4], [0, 1, 3, 4, 5], [0, 1, 3, 54, 1],
                  [0, 1, 2, 55, 1], [0, 11, 33, 5, 3], [0, 11, 33, 5, 3]])
    B = np.zeros((10, len(X)))
    print(X.shape)
    tasks = []
    for epoch in range(0,10-1):
        for i,x in enumerate(X):
            inner_looper(i, X, B, epoch)


    print(np.sum(B[9]))
    start_time += time.time()
    print("Normal Time Taken : " + str(start_time))



#thread_loop_test()
#data_parallel()
#task_parallel()
dask_test1()
