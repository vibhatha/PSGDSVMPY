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


def stats(exp_name='', acc=0, time=0, world_size=1, beta1=0.93, beta2=0.99, batch_size=10, epochs=10):
    Print.Print.result1(exp_name + " Parallel SGD SVM Accuracy : " + str(acc) + "%" + ", " + str(time)
                        + ", Epochs : " + str(epochs) + ", " + str(batch_size) + ", " + str(beta1) + ", " + str(beta2))

    fp = open("logs/psgd/minibatch/" + socket.gethostname() + "_" + exp_name +"_batch_size_"+str(batch_size)+ "_cores_"+str(world_size) + "_minibatch_psgd_adam_results.txt", "a")
    # fp.write("alpha : " + str(self.alpha) + ", epochs : " + str(self.epochs) + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
    fp.write(str(epochs) + ", " +str(batch_size) + ", " + str(beta1) + ", " + str(beta2) + ", " + str(acc) + ", " + str(time) + "\n")
    fp.close()


X = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [1, 5], [2, 6], [3, 7], [4, 5]])
y = np.array([1, 1, 1, 1, -1, -1, -1, -1])
X_test = np.array(
    [[1, 1.25], [2.1, 1.15], [3.1, 1.45], [4.23, 1.21], [1.3, 5.25], [2.11, 6.24], [3.3, 7.24], [4.212, 5.78]])

args = sys.argv

dataset = str(args[1])
features = int(args[2])
split = bool(args[3])

# plt.scatter(X[:,0],X[:,1])
# plt.show()
datasets = ['webspam', 'cod-rna', 'phishing', 'w8a', 'a9a', 'ijcnn1']
n_features = [254, 8, 68, 300, 123, 22]
splits = [True, False, True, False, False, False]
datasets = [dataset]
n_features = [features]
splits = [split]


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

    DATA_TYPE = 'float32'

    X = X.astype(DATA_TYPE)
    y = y.astype(DATA_TYPE)

    # create mini batches


    comms = Communication.Communication()
    rank = comms.comm.Get_rank()
    world_size = comms.comm.Get_size()

    batch_rations = np.arange(0,1,0.1,dtype='f')

    for repitition in np.arange(0,10,1):
        n = len(X)
        indices = np.arange(0, n, 1)
        np.random.shuffle(indices)
        data_per_machine = n / world_size
        batch_size = 10
        for per in [1.00]:
            if (world_size > 1):
                batch_size = int(per * data_per_machine)
            else:
                batch_size = n
            data_per_machine_indices = np.array_split(indices, world_size)
            index_set_of_machine = data_per_machine_indices[rank]
            print("Total Size, Data Per Machine, Index Per Machine, Batch Size ", n, data_per_machine ,len(index_set_of_machine), batch_size)
            indices = np.random.choice(index_set_of_machine, batch_size, replace=False)

            eta = 0.01
            epochs = 0
            T = 200
            if world_size > 1:
                epochs = 200#n / (world_size * batch_size)
            else:
                epochs = T
            print("World Size : ", world_size, epochs)


            m1 = len(X[0])

            w = np.zeros(m1, DATA_TYPE)
            w_r = np.zeros(m1, DATA_TYPE)
            gradient = np.zeros(m1, DATA_TYPE)
            gradient_r = np.zeros(m1, DATA_TYPE)
            w_list = []
            isComplete = False
            S = indices
            T = epochs
            v = np.zeros(w.shape, DATA_TYPE)
            r = np.zeros(w.shape, DATA_TYPE)
            beta1_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.93, 0.95, 0.99, 0.999]
            beta2_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.93, 0.95, 0.99, 0.999]
            tstart = True
            for beta1 in beta1_range:
                for beta2 in beta2_range:
                    beta = 0.90
                    epsilon = 0.00000001
                    exp_time = 0
                    if(rank == 0 and tstart == True):
                        exp_time -= time.time()

                    for index in indices:
                        gradient = 0
                        for epoch in range(1, epochs):
                            condition = y[index] * np.dot(X[index], w)
                            alpha = 1.0 / (1.0 + float(epoch))
                            coefficient = ((1 / float(epoch)))
                            if (condition < 1):
                                gradient = alpha * (-(X[index] * y[index]) + (coefficient * w))  # gradient accumilation issue must be checked
                            else:
                                gradient = alpha * (coefficient * w)

                        v = beta1 * v + (1 - beta1) * gradient
                        v_hat = v / (1 - beta1 ** epoch)
                        grad_mul = (np.multiply(gradient, gradient))
                        #print(gradient[0:2], grad_mul[0:2])
                        r = beta2 * r + (1 - beta2) * grad_mul
                        r_hat = r / (1 - beta2 ** epoch)
                        w = w - alpha * np.multiply((v_hat), 1.0 / (np.sqrt(r_hat) + epsilon))
                        #comms.bcast(input=w, dtype=comms.mpi.FLOAT, root=0)
                        comms.allreduce(input=w, output=w_r, op=comms.mpi.SUM, dtype=comms.mpi.FLOAT)
                        w = w_r/world_size
                        if (rank == 0 and epoch == epochs - 1):
                            isComplete = True
                            tstart = False

                    if(rank == 0 and tstart == False):
                        exp_time += time.time()
                        tstart = True

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
                        print(" Beta1, Beta2, Acc : ",beta1, beta2, acc)
                        print("Time : ", exp_time)
                        stats(exp_name=dataset, acc=acc, time=exp_time, world_size=world_size, beta1=beta1, beta2=beta2, batch_size=batch_size, epochs=epochs)


