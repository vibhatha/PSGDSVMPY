import sys
import os
import socket

HOME = os.environ['HOME']
sys.path.insert(1, HOME + '/github/StreamingSVM')
import numpy as np
from operations import Print
import time
from comms import Communication
from distributed import DistributedDataLoader
from api import Constant
from api import ExperimentObjectPSGDItems

# per core data distribution testing
def stats(exp_name='', acc=0, time=0, world_size=1, beta1=0.93, beta2=0.99, batch_size=10, epochs=10, repitition=10):
    Print.Print.result1("Repitition " + str(repitition) + ", DataSet : " +
        exp_name + " Parallel SGD SVM Accuracy : " + str(acc) + "%" + ", " + str(time) + ", Epochs : " + str(epochs) + ", " + str(beta1) + ", " + str(beta2))

    fp = open("logs/psgd/adam/" + socket.gethostname() + "_" + exp_name + "_batch_size_" + str(batch_size) + "_cores_" + str(
        world_size) + "_psgd_adam_pcd_results.txt", "a")
    # fp.write("alpha : " + str(self.alpha) + ", epochs : " + str(self.epochs) + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
    fp.write(
        str(epochs) + ", " + str(batch_size) + ", " + str(beta1) + ", " + str(beta2) + ", " + str(acc) + ", " + str(
            time) + "\n")
    fp.close()


comms = Communication.Communication()
rank = comms.comm.Get_rank()
world_size = comms.comm.Get_size()
T = 100
M = world_size


expItem = ExperimentObjectPSGDItems.ExperimentObjectsPSGDItems()
experiments = expItem.getlist()

for experiment in experiments:

    DATA_SET =experiment.dataset
    DATA_SOURCE = experiment.data_soruce
    FEATURES = experiment.features
    SAMPLES = experiment.samples
    SPLIT = experiment.split
    TRAINING_FILE = experiment.training_file
    TESTING_FILE = experiment.testing_file
    TRAINING_SAMPLES = experiment.training_samples
    TESTING_SAMPLES = experiment.testing_samples
    REPITITIONS = 1

    dis = DistributedDataLoader.DistributedDataLoader(source_file=DATA_SOURCE,
                                                      n_features=FEATURES,
                                                      n_samples=SAMPLES, world_size=world_size,
                                                      rank=rank,split=SPLIT, testing_file=TESTING_FILE,
                                                      train_samples=TRAINING_SAMPLES, test_samples=TESTING_SAMPLES)
    x_all, y_all = dis.load_training_data_chunks()
    X_test, y_test = dis.load_testing_data()
    m1 = len(x_all[0])
    epsilon = 0.00000001
    w = np.zeros(m1, 'f')
    w_ar = np.zeros(m1, 'f')
    gradient = np.zeros(m1, 'f')
    gradient_r = np.zeros(m1, 'f')
    w_list = []
    isComplete = False
    v = np.zeros(w.shape, 'f')
    r = np.zeros(w.shape, 'f')
    beta1_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.93, 0.95, 0.99, 0.999]
    beta2_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.93, 0.95, 0.99, 0.999]
    #beta1_range = [0.90]
    #beta2_range = [0.93]

    X = x_all
    y = y_all
    for beta1 in beta1_range:
        for beta2 in beta2_range:
            epochs = np.arange(1, T)
            for rep in np.arange(0,REPITITIONS):
                exp_time = 0
                exp_time -= time.time()
                for epoch in epochs:
                    m = len(X)
                    C = int(m / M)
                    m_real = C * M
                    range = np.arange(0, m_real - 1, M)
                    #print(len(range),M)
                    #if (epoch % 10):
                        #print("Rank " + str(rank) + ", Epoch " + str(epoch))
                    for i in range:
                        Xi = X[i + rank]
                        yi = y[i + rank]
                        condition = yi * np.dot(Xi, w)
                        alpha = 1.0 / (1.0 + float(epoch))
                        coefficient = ((1.0 / 1.0 + float(epoch)))
                        if (condition < 1):
                            gradient = alpha * (-(Xi * yi) + (coefficient * w))
                        else:
                            gradient = alpha * (coefficient * w)
                        v = beta1 * v + (1 - beta1) * gradient
                        v_hat = v / (1 - beta1 ** epoch)
                        r = beta2 * r + (1 - beta2) * (np.multiply(gradient, gradient))
                        r_hat = r / (1 - beta2 ** epoch)
                        w = w - alpha * np.multiply((v_hat), 1.0 / (np.sqrt(r_hat) + epsilon))
                        comms.allreduce(input=w, output=w_ar, op=comms.mpi.SUM, dtype=comms.mpi.FLOAT)
                        w = w_ar / M
                        comms.bcast(input=w, dtype=comms.mpi.FLOAT, root=0)
                        if (epoch == T - 1):
                            isComplete = True
                exp_time += time.time()

                if (rank == 0 and isComplete):
                    labels = []
                    for x in X_test:
                        label = np.sign(np.dot(w.T, x))
                        labels.append(label)

                    y_pred = np.array(labels)
                    # print(labels)
                    # print(y_testing)
                    correct = (y_pred == y_test).sum()
                    total = len(y_pred)
                    acc = float(correct) / float(total) * 100.0
                    print("Acc : ", acc)
                    stats(exp_name=DATA_SET, acc=acc, time=exp_time, world_size=world_size, beta1=beta1, beta2=beta2, batch_size=-1, epochs=T, repitition=rep)