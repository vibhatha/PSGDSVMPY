import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from sgd import SVM
from operations import LoadLibsvm
from operations import Bcolors
from operations import Print
from kernel import Kernel
import time
import sklearn.metrics.pairwise
import numpy as np
import socket


class BenchmarkKernelSGDSvm:

    def __init__(self, exp_name ,training_file, testing_file, features=1, alpha=0.01, gamma=0.01, degree =2, batch_size=1000,
                 kernel='linear', epochs=100, labelfix=False, split=False, randomize=False, auto=False):
        self.training_file = training_file
        self.testing_file = testing_file
        self.n_features = features
        self.alpha = alpha
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel
        self.batch_size = batch_size
        self.epochs = epochs
        self.exp_name = exp_name
        self.labelfix = labelfix
        self.split = split
        self.randomize = randomize
        self.auto = auto

    def load_data(self):
        training_filepath = self.training_file
        testing_filepath = self.testing_file
        if not self.split:
            training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=self.n_features)
            testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath, n_features=self.n_features)
            self.x_training, self.y_training = training_loader.load_all_data()
            self.x_testing, self.y_testing = testing_loader.load_all_data()

        if self.split:
            Print.Print.result2("Data Splitting ...")
            training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=self.n_features)
            x_all, y_all = training_loader.load_all_data()
            ratio = 0.8
            size = len(x_all)
            split_index = int(size * ratio)
            self.x_training = x_all[:split_index]
            self.x_testing = x_all[split_index:]
            self.y_training = y_all[:split_index]
            self.y_testing = y_all[split_index:]

        if (self.labelfix):
            self.y_training = self.y_training[self.y_training == 2] = -1
            self.y_testing = self.y_testing[self.y_testing == 2] = -1

        return self.x_training, self.y_training, self.x_testing, self.y_testing

    def generate_kernelized_data(self):
        Xk_train = []
        Xk_test = []
        if (self.kernel == 'linear'):
            Xk_train = sklearn.metrics.pairwise.linear_kernel(self.x_training)
            Xk_test = sklearn.metrics.pairwise.linear_kernel(self.x_testing)

        if (self.kernel == 'rbf'):
            Xk_train = sklearn.metrics.pairwise.rbf_kernel(X=self.x_training, gamma=self.gamma)
            Xk_test = sklearn.metrics.pairwise.rbf_kernel(X=self.x_testing, gamma=self.gamma)

        if (self.kernel == 'poly'):
            Xk_train = sklearn.metrics.pairwise.polynomial_kernel(X=self.x_training, degree=self.degree)
            Xk_test = sklearn.metrics.pairwise.polynomial_kernel(X=self.x_testing, degree=self.degree)

        return Xk_train, Xk_test

    def generate_kerenelized_feature_matrix(self,X):
        Xk = []
        if (self.kernel == 'linear'):
            Xk = sklearn.metrics.pairwise.linear_kernel(X=X)

        if (self.kernel == 'rbf'):
            Xk = sklearn.metrics.pairwise.rbf_kernel(X=X, gamma=self.gamma)

        if (self.kernel == 'poly'):
            Xk = sklearn.metrics.pairwise.polynomial_kernel(X=X, degree=self.degree)

        return Xk

    def train(self):
        self.training_time = 0
        self.svm = SVM.SVM(X=self.x_training, y=self.y_training, alpha=self.alpha, gamma=self.gamma, degree = self.degree,
                           n_features=self.n_features, randomize=self.randomize)
        self.training_time -= (time.time())
        Print.Print.console('Kernel SGD Mode ')
        Print.Print.info1('\t|            |')
        Print.Print.info1('\t|            |')
        Print.Print.info1('\t|            |')
        Print.Print.info1('\t|            |')
        Print.Print.info1('\t|            |')
        Print.Print.info1('\t\\            /')
        Print.Print.info1('\t \\          /')
        Print.Print.info1('\t  \\        /')
        Print.Print.info1('\t   \\      /')
        Print.Print.info1('\t    \\    /')
        Print.Print.info1('\t     \\  /')
        Print.Print.info1('\t      \\/')
        Print.Print.result2('Kernel : ' + self.kernel)
        Print.Print.result2('Gamma : ' + str(self.gamma))
        Print.Print.result2('Degree : ' + str(self.degree))


        self.svm.train_with_simple_kernel(X=self.x_training, y=self.y_training, alpha=self.alpha,
                                     epochs=self.epochs, kernel=self.kernel)
        # X_train, X_test = self.generate_kernelized_data()
        # self.svm.train_with_kernel(Xorg=self.x_training, X=X_train, y=self.y_training, alpha=self.alpha,
        #                             epochs=self.epochs, kernel=self.kernel)
        # self.svm.custom_minibatch_kernel_sgd_train(X=self.x_training, y=self.y_training, alpha=self.alpha,
        #                                            epochs=self.epochs,batch_size=self.batch_size,
        #                                            kernel=self.kernel, gamma=self.gamma, degree=self.degree)
        self.training_time += (time.time())
        return self.svm

    def test(self):
        #y_pred = self.svm.kernel_predict(X=self.x_testing, kernel=self.kernel)
        y_pred = self.svm.predict(X=self.x_testing)
        print(y_pred)
        self.acc = self.svm.get_accuracy(y_test=self.y_testing, y_pred=y_pred)
        return self.acc

    def stats(self):
        print(Bcolors.Bcolors.BOLD + Bcolors.Bcolors.OKGREEN + self.exp_name + " Accuracy : " + str(self.acc) + "%" + Bcolors.Bcolors.BOLD)
        print(Bcolors.Bcolors.BOLD + self.exp_name + " Time : " + str(self.training_time) + " s" + Bcolors.Bcolors.OKBLUE)
        fp = open("logs/"  + socket.gethostname()+"_"  + self.exp_name + "_kernel_sgd_results.txt", "a")
        #fp.write("alpha : " + str(self.alpha) + ", epochs : " + str(self.epochs) + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
        fp.write(self.kernel+','+str(self.alpha) + "," + str(self.epochs) + "," + str(self.acc) + "," + str(self.training_time) + "\n")
        fp.close()



