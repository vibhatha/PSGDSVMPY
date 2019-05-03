import sys
import os
import socket

HOME = os.environ['HOME']
sys.path.insert(1, HOME + '/github/StreamingSVM')
from operations import LoadLibsvm
import numpy as np
from operations import Print
import time
from comms import Communication


def stats(exp_name='', acc=0, time=0, world_size=1):
    Print.Print.result1(exp_name + " Parallel SGD SVM Accuracy : " + str(acc) + "%" + ", " + str(time))

    fp = open("logs/" + socket.gethostname() + "_" + exp_name + "_cores_"+str(world_size) + "_psgd_results.txt", "a")
    # fp.write("alpha : " + str(self.alpha) + ", epochs : " + str(self.epochs) + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
    fp.write(str(acc) + ", " + str(time) + "\n")
    fp.close()


X = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [1, 5], [2, 6], [3, 7], [4, 5]])
y = np.array([1, 1, 1, 1, -1, -1, -1, -1])
X_test = np.array(
    [[1, 1.25], [2.1, 1.15], [3.1, 1.45], [4.23, 1.21], [1.3, 5.25], [2.11, 6.24], [3.3, 7.24], [4.212, 5.78]])
# plt.scatter(X[:,0],X[:,1])
# plt.show()
datasets = ['ijcnn1', 'heart', 'webspam', 'cod-rna', 'phishing', 'breast_cancer', 'w8a', 'a9a', 'real-slim']
n_features = [22, 13, 254, 8, 68, 10, 300, 123, 20958]
splits = [False, False, True, False, True, True, False, False, True]

for dataset, features, split in zip(datasets, n_features, splits):

    base_path = ''
    hostname = socket.gethostname()

    if hostname == 'vibhatha-ThinkPad-P50':
        base_path = '/home/vibhatha/data/svm/'
    else:
        base_path = '/N/u/vlabeyko/data/svm/svm/'

    bulk = False
    dataset = dataset
    training_file = base_path + dataset + '/training.csv'
    testing_file = base_path + dataset + '/testing.csv'
    n_features = features
    split = split
    training_loader = LoadLibsvm.LoadLibSVM(filename=training_file, n_features=n_features)

    x_training = []
    y_training = []
    x_testing = []
    y_testing = []

    if split == True:
        x_all, y_all = training_loader.load_all_data()
        ratio = 0.8
        world_size = len(x_all)
        split_index = int(world_size * ratio)
        x_training = x_all[:split_index]
        x_testing = x_all[split_index:]
        y_training = y_all[:split_index]
        y_testing = y_all[split_index:]

    else:
        training_loader = LoadLibsvm.LoadLibSVM(filename=training_file, n_features=n_features)
        testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_file, n_features=n_features)
        x_training, y_training = training_loader.load_all_data()
        x_testing, y_testing = testing_loader.load_all_data()

    print(x_training.shape)
    X = x_training
    y = y_training

    X = X.astype('f')
    y = y.astype('f')

    # create mini batches


    comms = Communication.Communication()
    rank = comms.comm.Get_rank()
    world_size = comms.comm.Get_size()

    n = len(X)
    indices = np.arange(0, n, 1)
    data_per_machine = n / world_size
    batch_size = 10
    if (world_size > 1):
        batch_size = int(0.10 * data_per_machine)
    else:
        batch_size = n
    data_per_machine_indices = np.array_split(indices, world_size)
    index_set_of_machine = data_per_machine_indices[rank]
    print("Total Size, Data Per Machine, Index Per Machine, Batch Size ", n, data_per_machine ,len(index_set_of_machine), batch_size)
    indices = np.random.choice(index_set_of_machine, batch_size, replace=False)

    eta = 0.01
    epochs = 0
    T = 100
    if world_size > 1:
        epochs = n / (world_size * batch_size)
    else:
        epochs = T
    print("World Size : ", world_size, epochs)


    exp_time = 0
    exp_time -= time.time()


    m1 = len(X[0])

    w = np.zeros(m1, 'f')
    grad = np.zeros(m1, 'f')
    grad_r = np.zeros(m1, 'f')
    w_list = []
    isComplete = False
    S = indices
    T = epochs

    for i in indices:
        for epoch in range(1, epochs):

            if (y[i] * np.dot(X[i], w)) < 1:
                grad = grad + (-1*(X[i] * y[i]) + (w))
            else:
                grad = grad + w
        comms.allreduce(input=grad, output=grad_r, op=comms.mpi.SUM, dtype=comms.mpi.FLOAT)
        if (rank == 0):
            g_global = 0
            if (world_size > 1):
                g_global = grad_r / (world_size * batch_size)
            else:
                g_global = grad_r / (world_size)
            w = w - eta * g_global
            comms.bcast(input=w, dtype=comms.mpi.FLOAT, root=0)
        if (rank == 0 and epoch == epochs - 1):
            isComplete = True

    exp_time += time.time()

    if (isComplete):
        labels = []
        for x in x_testing:
            label = np.sign(np.dot(w.T, x))
            labels.append(label)

        y_pred = np.array(labels)
        # print(labels)
        # print(y_testing)
        correct = (y_pred == y_testing).sum()
        total = len(y_pred)
        acc = float(correct) / float(total) * 100.0
        print("Acc : ", acc)
        print("Time : ", exp_time)
        stats(exp_name=dataset, acc=acc, time=exp_time, world_size=world_size)


