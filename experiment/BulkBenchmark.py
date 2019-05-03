import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from experiment import BenchmarkSgdSvm
from experiment import BenchmarkPegasosSvm
from experiment import BenchmarkKernelSGDSvm
from experiment import BenchmarkMinibatchSGDSvm
from experiment import BenchmarkSGDMomentum
from experiment import BenchmarkSGDAda
import numpy as np
import socket

class BulkBenchmark:

    def __init__(self, exp_name, training_file, testing_file, features=1, alpha=0.01, epochs=100, eta=0.01,
                 kernel='linear', gamma=0.01, degree=2, C=1, bulk = True,
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
        self.bulk = bulk

    def  benchmark_sgd(self):
        bmsvmsgd = BenchmarkSgdSvm.BenchmarkSgdSVM(exp_name=self.exp_name, training_file=self.training_file, testing_file=self.testing_file,
                                             alpha=self.alpha, features=self.n_features, epochs=self.epochs, labelfix=self.labelfix, randomize=self.randomize, split=self.split, auto=True, bulk=self.bulk)
        bmsvmsgd.load_data()
        bmsvmsgd.train()
        bmsvmsgd.test()
        bmsvmsgd.stats()

    def benchmark_pegasos(self):
        bmsvmsgd = BenchmarkPegasosSvm.BenchmarkPegasosSvm(exp_name=self.exp_name, training_file=self.training_file,
                                                   testing_file=self.testing_file,
                                                   eta=self.eta, ld=self.ld, features=self.n_features, epochs=self.epochs, labelfix=self.labelfix, randomize=self.randomize, split=self.split, bulk=self.bulk)
        bmsvmsgd.load_data()
        bmsvmsgd.train()
        bmsvmsgd.test()
        bmsvmsgd.stats()

    def benchmark_manual_sgd(self):
        bmsvmsgd = BenchmarkSgdSvm.BenchmarkSgdSVM(exp_name=self.exp_name, training_file=self.training_file,
                                                   testing_file=self.testing_file,
                                                   alpha=self.alpha, features=self.n_features, epochs=self.epochs,
                                                   labelfix=self.labelfix, randomize=self.randomize, split=self.split, auto=False)
        bmsvmsgd.load_data()
        bmsvmsgd.train()
        bmsvmsgd.test()
        bmsvmsgd.stats()

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


    def benchmark_minibatch_sgd(self):
        bmminisgd = BenchmarkMinibatchSGDSvm.BenchmarkMinibatchSgdSVM(exp_name=self.exp_name, training_file=self.training_file,
                                                   testing_file=self.testing_file,
                                                   alpha=self.alpha, features=self.n_features, epochs=self.epochs,
                                                   labelfix=self.labelfix, randomize=self.randomize, split=self.split,
                                                   auto=True, minibatch=True, batch_size=self.minibatch_size)

        bmminisgd.load_data()
        bmminisgd.train()
        bmminisgd.test()
        bmminisgd.stats()

    def benchmark_custom_minibatch_sgd(self):
        bmminisgd = BenchmarkMinibatchSGDSvm.BenchmarkMinibatchSgdSVM(exp_name=self.exp_name, training_file=self.training_file,
                                                   testing_file=self.testing_file,
                                                   alpha=self.alpha, features=self.n_features, epochs=self.epochs,
                                                   labelfix=self.labelfix, randomize=self.randomize, split=self.split,
                                                   auto=True, minibatch=False, custom_minibatch=True, batch_size=self.minibatch_size)

        bmminisgd.load_data()
        bmminisgd.train()
        bmminisgd.test()
        bmminisgd.stats()



    def benchmark_momentum_factor(self):
        bmsvmsgdmomentum = BenchmarkSGDMomentum.BenchmarkSGDMomentum(exp_name=self.exp_name,
                                                                     training_file=self.training_file,
                                                                     testing_file=self.testing_file,
                                                                     alpha=self.alpha, C=self.C, gamma=self.gamma,
                                                                     features=self.n_features, epochs=self.epochs,
                                                                     labelfix=self.labelfix,
                                                                     randomize=self.randomize, split=self.split, bulk=self.bulk)
        bmsvmsgdmomentum.load_data()
        bmsvmsgdmomentum.train()
        bmsvmsgdmomentum.test()
        bmsvmsgdmomentum.stats()



    def benchmark_adagrad(self):
        bmsvmsgd = BenchmarkSGDAda.BenchmarkSGDAda(exp_name=self.exp_name, training_file=self.training_file,
                                                   testing_file=self.testing_file,
                                                   alpha=self.alpha, features=self.n_features, epochs=self.epochs, labelfix=self.labelfix,
                                                   randomize=self.randomize, split=self.split, auto=True, bulk=self.bulk)
        bmsvmsgd.load_data()
        bmsvmsgd.train()
        bmsvmsgd.test()
        bmsvmsgd.stats()
