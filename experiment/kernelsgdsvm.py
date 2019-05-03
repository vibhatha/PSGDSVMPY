import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from experiment import BenchmarkSgdSvm
from experiment import BenchmarkPegasosSvm
from experiment import BenchmarkKernelSGDSvm
from experiment import BenchmarkMinibatchSGDSvm
from experiment import BenchmarkSGDMomentum
import numpy as np
import socket

class KernelSGDSVM:

    def __init__(self, exp_name, training_file, testing_file, features=1, alpha=0.01, epochs=100, eta=0.01,
                 kernel='linear', gamma=0.01, degree=2, C=1,
                 ld=0.0001, labelfix=False, split=False, randomize=False, minibatch=False, minibatch_size=1000):
        self.training_file = training_file
        self.split = split
        self.testing_file = testing_file
        self.n_features = features
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.epochs = epochs
        self.C = C
        self.exp_name = exp_name
        self.eta = eta
        self.ld = ld
        self.labelfix = labelfix
        self.randomize = randomize
        self.minibatch = minibatch
        self.minibatch_size = minibatch_size


    def benchmark_kernel_sgd(self):
        bmkernelsgd = BenchmarkKernelSGDSvm.BenchmarkKernelSGDSvm(exp_name=self.exp_name, training_file=self.training_file,
                                                   testing_file=self.testing_file,
                                                   alpha=self.alpha, gamma=self.gamma, degree=self.degree,
                                                   kernel=self.kernel, features=self.n_features, epochs=self.epochs,
                                                   labelfix=self.labelfix, randomize=self.randomize, split=self.split, auto=False)

        bmkernelsgd.load_data()
        bmkernelsgd.train()
        bmkernelsgd.test()
        bmkernelsgd.stats()


args = sys.argv

dataset = str(args[1])
features = int(args[2])
split = bool(args[3])
gamma = float(args[4])

base_path = ''
hostname = socket.gethostname()

if hostname == 'vibhatha-ThinkPad-P50':
	base_path = '/home/vibhatha/data/svm/'
else:
	base_path = '/N/u/vlabeyko/data/svm/svm/'

dataset = dataset
training_file = base_path + dataset + '/training.csv'
testing_file = base_path + dataset + '/testing.csv'
features = features
alpha = 1
epochs = 200
ld = 1
eta = 0.1
labelfix = False
split = split
randomize = True
gamma = gamma
degree = 1
kernel = 'rbf'
minibatch_size = 1000
minibatch = False
C=1

# for epochs in [50]:
#     for al in [0.001]:
#         for batch_size in [20000]:
#             bm = MultiBenchmark(exp_name=dataset, training_file=training_file, testing_file=testing_file, features=features,
#                     alpha=al, epochs=epochs, eta=eta, ld=ld, C=C, gamma=gamma, degree=degree, kernel=kernel,
#                     labelfix=labelfix, randomize=randomize, split=split, minibatch=minibatch, minibatch_size=batch_size)
#             bm.benchmark_custom_minibatch_sgd()



## For Samples with both training and testing data
# bm = Benchmark(exp_name=dataset,training_file=training_file, testing_file=testing_file,
#                features=features, alpha=alpha, epochs=epochs, eta=eta, ld=ld,
#                labelfix=labelfix, randomize=randomize, split=False)
# bm.benchmark_sgd()
# bm.benchmark_pegasos()

## For samples with only a training sample
## Split the data into a ratio (0.6) to get the training and the testing datasets

#bm = MultiBenchmark(exp_name=dataset,training_file=training_file, testing_file=testing_file, features=features,
#                 alpha=alpha, epochs=epochs, eta=eta, ld=ld, gamma=gamma, degree=degree, kernel=kernel,
#                 labelfix=labelfix, randomize=randomize, split=split, minibatch=minibatch, minibatch_size=minibatch_size)
#bm.benchmark_kernel_sgd()
#bm.benchmark_manual_sgd()
#bm.benchmark_momentum_sgd()
#bm.benchmark_minibatch_sgd()
###############################################################
###############################################################
###############################################################
################# Dynamic Benchmarking Suite###################
###############################################################
###############################################################
###############################################################


# for ldi in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
#     bm = MultiBenchmark(exp_name=dataset,training_file=training_file, testing_file=testing_file, features=features,
#                 alpha=alpha, epochs=epochs, eta=eta, ld=ldi,
#                 labelfix=labelfix, randomize=randomize, split=True)
#     bm.benchmark_pegasos()
#
# for alpha in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
#     bm = MultiBenchmark(exp_name=dataset,training_file=training_file, testing_file=testing_file, features=features,
#                 alpha=alpha, epochs=epochs, eta=eta, ld=ldi,
#                 labelfix=labelfix, randomize=randomize, split=True)
#     bm.benchmark_manual_sgd()

# for al in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
#     for gm in [0.9, 1]:
#         bm = MultiBenchmark(exp_name=dataset,training_file=training_file, testing_file=testing_file, features=features,
#                  alpha=al, epochs=epochs, eta=eta, ld=ld, gamma=gm, degree=degree, kernel=kernel,
#                  labelfix=labelfix, randomize=randomize, split=True)
#         bm.benchmark_kernel_sgd()

## Webspam
### Bulky Experiment
# for epochs in [10, 20, 30 ,40, 50, 60, 70, 80, 90 ,100, 150, 200]:
#     for al in [0.00000001, 0.0000001,0.000001,0.00001 ,0.0001, 0.001, 0.01, 0.1, 1]:
#         for batch_size in [10000, 50000, 100000, 200000]:
#             bm = MultiBenchmark(exp_name=dataset, training_file=training_file, testing_file=testing_file, features=features,
#                     alpha=al, epochs=epochs, eta=eta, ld=ld, C=C, gamma=gamma, degree=degree, kernel=kernel,
#                     labelfix=labelfix, randomize=randomize, split=split, minibatch=minibatch, minibatch_size=batch_size)
#             bm.benchmark_minibatch_sgd()
#         #bm.benchmark_manual_sgd()
#         #bm.benchmark_sgd()
#         #bm.benchmark_momentum_factor()
# dataset='ijcnn1'
# split = False
# features = 22
# for epochs in [10, 20, 30 ,40, 50, 60, 70, 80, 90 ,100, 150, 200]:
#     for al in [0.00000001, 0.0000001,0.000001,0.00001 ,0.0001, 0.001, 0.01, 0.1, 1]:
#         for batch_size in [10000, 15000, 20000, 30000, 40000, 50000]:
#             bm = MultiBenchmark(exp_name=dataset, training_file=training_file, testing_file=testing_file, features=features,
#                     alpha=al, epochs=epochs, eta=eta, ld=ld, C=C, gamma=gamma, degree=degree, kernel=kernel,
#                     labelfix=labelfix, randomize=randomize, split=split, minibatch=minibatch, minibatch_size=batch_size)
#             bm.benchmark_minibatch_sgd()
#
#
# dataset='a9a'
# split = False
# features = 123
# for epochs in [10, 20, 30 ,40, 50, 60, 70, 80, 90 ,100, 150, 200]:
#     for al in [0.00000001, 0.0000001,0.000001,0.00001 ,0.0001, 0.001, 0.01, 0.1, 1]:
#         for batch_size in [10000, 15000, 20000, 30000]:
#             bm = MultiBenchmark(exp_name=dataset, training_file=training_file, testing_file=testing_file, features=features,
#                     alpha=al, epochs=epochs, eta=eta, ld=ld, C=C, gamma=gamma, degree=degree, kernel=kernel,
#                     labelfix=labelfix, randomize=randomize, split=split, minibatch=minibatch, minibatch_size=batch_size)
#             bm.benchmark_minibatch_sgd()
#
# dataset='covtype'
# split = True
# features = 54
# for epochs in [10, 20, 30 ,40, 50, 60, 70, 80, 90 ,100, 150, 200]:
#     for al in [0.00000001, 0.0000001,0.000001,0.00001 ,0.0001, 0.001, 0.01, 0.1, 1]:
#         for batch_size in [50000, 100000, 150000, 300000]:
#             bm = MultiBenchmark(exp_name=dataset, training_file=training_file, testing_file=testing_file, features=features,
#                     alpha=al, epochs=epochs, eta=eta, ld=ld, C=C, gamma=gamma, degree=degree, kernel=kernel,
#                     labelfix=labelfix, randomize=randomize, split=split, minibatch=minibatch, minibatch_size=batch_size)
#             bm.benchmark_minibatch_sgd()

bm = KernelSGDSVM(exp_name=dataset, training_file=training_file, testing_file=testing_file, features=features,
                     alpha=alpha, epochs=epochs, eta=eta, ld=ld, C=C, gamma=gamma, degree=degree, kernel=kernel,
                     labelfix=labelfix, randomize=randomize, split=split, minibatch=minibatch, minibatch_size=minibatch_size)
bm.benchmark_kernel_sgd()
