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
import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t

def test(int x):
    cdef int y = 0
    cdef int i
    for i in range(x):
        y += i
    return y

def stats(exp_name='', acc=0, time=0, world_size=1, beta1=0.93, beta2=0.99, batch_size=10, epochs=10):
    Print.Print.result1(exp_name + " Parallel SGD SVM Accuracy : " + str(acc) + "%" + ", " + str(time)
                        + ", Epochs : " + str(epochs) + ", " + str(batch_size) + ", " + str(beta1) + ", " + str(beta2))

    fp = open("logs/psgd/minibatch/" + socket.gethostname() + "_" + exp_name + "_batch_size_" + str(
        batch_size) + "_cores_" + str(world_size) + "_minibatch_psgd_adam_results.txt", "a")
    # fp.write("alpha : " + str(self.alpha) + ", epochs : " + str(self.epochs) + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
    fp.write(
        str(epochs) + ", " + str(batch_size) + ", " + str(beta1) + ", " + str(beta2) + ", " + str(acc) + ", " + str(
            time) + "\n")
    fp.close()



args = sys.argv

dataset = str(args[1])
features = int(args[2])
split = False #bool(args[3])

print(dataset, features, split)

comms = Communication.Communication()
rank = comms.comm.Get_rank()
world_size = comms.comm.Get_size()
# plt.scatter(X[:,0],X[:,1])
# plt.show()
datasets = ['webspam', 'cod-rna', 'phishing', 'w8a', 'a9a', 'ijcnn1']
n_features = [254, 8, 68, 300, 123, 22]
splits = [True, False, True, False, False, False]

cdef int n = 35000
cdef int batch_size = 10
cdef int epochs = 200
cdef double eta = 0.0001
cdef int m1 = 22
cdef double per = 1.00

cdef np.ndarray beta1_range = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.93, 0.95, 0.99, 0.999])
cdef np.ndarray beta2_range = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.93, 0.95, 0.99, 0.999])

DATA_TYPE = 'float32'


cdef np.ndarray w = np.random.uniform(0, 1, features)#np.zeros(m1, dtype=DTYPE)
cdef np.ndarray w_r = np.zeros(m1, dtype=DTYPE)
cdef np.ndarray gradient = np.zeros(m1, dtype=DTYPE)
cdef np.ndarray gradient_r = np.zeros(m1, dtype=DTYPE)
cdef np.ndarray v = np.zeros(m1, dtype=DTYPE)
cdef np.ndarray r = np.zeros(m1, dtype=DTYPE)
cdef np.ndarray X = np.zeros((28000,m1), dtype=DTYPE)
cdef np.ndarray y = np.zeros(28000, dtype=DTYPE)
cdef np.ndarray indices = np.arange(0, n, 1, dtype=DTYPE)
cdef int data_per_machine = n / world_size
cdef double epsilon = 0.00000001
cdef double alpha = 0.01
isComplete = False
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

print("X training, ",x_training.shape)
X = x_training
y = y_training


# create mini batches
batch_size = data_per_machine
batch_size = 10

if (world_size > 1):
    batch_size = int(per * data_per_machine)
else:
    batch_size = n

np.random.shuffle(indices)

data_per_machine_indices = np.array_split(indices, world_size)
cdef np.ndarray index_set_of_machine = data_per_machine_indices[rank]
# print("Total Size, Data Per Machine, Index Per Machine, Batch Size , Indices len", n, data_per_machine,
#        len(index_set_of_machine), batch_size, len(indices),len(data_per_machine_indices))
indices = np.random.choice(index_set_of_machine, batch_size, replace=False)
# print("X shape and Y shape Indices Length ", len(X), len(y), len(indices))
# print("World Size : ", world_size, epochs)
#
for beta1 in beta1_range:
     for beta2 in beta2_range:
         exp_time = 0
         exp_time -= comms.mpi.Wtime()
         for epoch in range(1, epochs):
             for index in indices:
                 index = int(index)
                 #print(index, len(indices), len(X), len(y))
                 condition = y[index] * np.dot(X[index], w)
                 coefficient = ((1 / float(epoch)))
                 if (condition < 1):
                     gradient = alpha * (-(X[index] * y[index]) + (coefficient * w))
                 else:
                     gradient = alpha * (coefficient * w)

                 v = beta1 * v + (1 - beta1) * gradient
                 v_hat = v / (1 - beta1 ** epoch)
                 grad_mul = (np.multiply(gradient, gradient))
# #                 #print(gradient[0:2], grad_mul[0:2])
                 r = beta2 * r + (1 - beta2) * grad_mul
                 r_hat = r / (1 - beta2 ** epoch)
                 w = w - alpha * np.multiply((v_hat), 1.0 / (np.sqrt(r_hat) + epsilon))
                #comms.bcast(input=w, dtype=comms.mpi.FLOAT, root=0)
                 comms.allreduce(input=w, output=w_r, op=comms.mpi.SUM, dtype=comms.mpi.DOUBLE)
                 #print(w_r)
                 w = w_r / float(world_size)
             if (rank == 0 and epoch == epochs - 1):
                 isComplete = True
#
         exp_time += comms.mpi.Wtime()
#
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
            print(" Beta1, Beta2, Acc : ", beta1, beta2, acc)
            print("Time : ", exp_time)
            stats(exp_name=dataset, acc=acc, time=exp_time, world_size=world_size, beta1=beta1, beta2=beta2,
                  batch_size=batch_size, epochs=epochs)
