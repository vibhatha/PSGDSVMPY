import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from sgd import SVM
from operations import LoadLibsvm
from operations import Bcolors
from operations import Print
import time
import socket
import numpy as np

class BenchmarkSGDMomentum:

    def __init__(self, exp_name ,training_file, testing_file, features=1, alpha=0.01, C=1, gamma=1,
                 epochs=100, labelfix=False, split=False, randomize=False, bulk=False):
        self.training_file = training_file
        self.testing_file = testing_file
        self.n_features = features
        self.alpha = alpha
        self.C = C
        self.gamma = gamma
        self.epochs = epochs
        self.exp_name = exp_name
        self.labelfix = labelfix
        self.split = split
        self.randomize = randomize
        self.bulk = bulk

    def load_data(self):
        Print.Print.info1("Loading Data")
        training_filepath = self.training_file
        testing_filepath = self.testing_file
        if not self.split and not self.bulk:
            training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=self.n_features)
            testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath, n_features=self.n_features)
            self.x_training, self.y_training = training_loader.load_all_data()
            self.x_testing, self.y_testing = testing_loader.load_all_data()

        if self.split and not self.bulk:
            Print.Print.info1("Splitting data ...")
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
            # this logic can varied depending on the target labels
            # TODO : Change the logic
            self.y_training = self.y_training[self.y_training == 2] = -1
            self.y_testing = self.y_testing[self.y_testing == 2] = -1

        if self.bulk and not self.split:
            # here we consider a larger testing data set.
            # here we consider a larger testing data set.
            print("Loading Bulk Training File")
            training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=self.n_features)
            self.x_training, self.y_training = training_loader.load_all_data()
            return self.x_training, self.y_training
        else:
            return self.x_training, self.y_training, self.x_testing, self.y_testing
        print("Training and Testing data loaded ...")

    def normalize_data(self):
        Print.Print.info1("Normalizing data")
        if not self.split and not self.bulk:
            mean_train = np.mean(self.x_training)
            std_train = np.std(self.x_training)
            mean_test = np.mean(self.x_testing)
            std_test = np.std(self.x_testing)
            self.x_training = (self.x_training - mean_train) / std_train
            self.x_testing = (self.x_testing - mean_test) / std_test
            return self.x_training, self.x_testing

    def train(self, gamma=0.98):
        Print.Print.info1("Training with SGD + Momentum ...")
        self.gamma = gamma;
        self.training_time = 0
        self.non_momentum_training_time = 0
        self.svm = SVM.SVM(X=self.x_training, y=self.y_training, alpha=self.alpha, C=self.C, gamma=self.gamma, n_features=self.n_features, randomize=self.randomize)
        self.w_init = self.svm.init_weights()
        self.indices_init = self.svm.init_indices(X=self.x_training)
        self.training_time -= (time.time())
        self.w_m = self.svm.train_momentum_cost2(X=self.x_training, y=self.y_training, alpha=self.alpha, epochs=self.epochs, w_init=self.w_init, indices_init=self.indices_init, gamma=gamma)
        self.training_time += (time.time())
        Print.Print.info("Momentum Training Completed !")
        Print.Print.info1("Training with SGD ...")
        self.non_momentum_training_time -= (time.time())
        self.w_sgd = self.svm.auto_sgd_train_cost2(X=self.x_training, y=self.y_training, epochs=self.epochs, w_init=self.w_init, indices_init=self.indices_init)
        self.non_momentum_training_time += (time.time())
        Print.Print.info("SGD Training Completed !")
        return self.svm

    def train_benchmark_light(self, w_init=[], x_train=[], y_train=[], x_test=[], y_test=[], log_file='', tolerance=0.001, indices_init=[], gamma=0.98, alpha=0.01):
        self.training_time = 0
        self.effective_epochs = 0
        self.cost = 0
        self.initial_cost = 0
        self.io_time = 0
        Print.Print.info1("Training with SGD + Momentum ...")
        self.training_time = 0
        self.non_momentum_training_time = 0
        self.svm = SVM.SVM(X=x_train, y=y_train, alpha=self.alpha, C=self.C, gamma=self.gamma,
                           n_features=self.n_features, randomize=self.randomize)
        w_init = w_init
        indices_init = indices_init
        self.training_time -= (time.time())
        self.w, self.effective_epochs, self.cost, self.io_time, self.initial_cost = self.svm.train_sgd_momentum(X=x_train, y=y_train, X_test=x_test, y_test=y_test, alpha=alpha,
                                                epochs=self.epochs, w_init=w_init, indices_init=indices_init, log_file=log_file, tolerance=tolerance, gamma=gamma)
        self.training_time += (time.time())
        self.training_time = self.training_time - self.io_time
        Print.Print.info1("Momentum Training Completed !")
        return self.w, self.effective_epochs, self.initial_cost, self.cost, self.training_time

    def train_benchmark_heavy(self, w_init=[], x_train=[], y_train=[], x_test=[], y_test=[], log_file='', tolerance=0.001, indices_init=[]):
        self.training_time = 0
        self.effective_epochs = 0
        self.cost = 0
        self.initial_cost = 0
        self.io_time = 0
        Print.Print.info1("Training with SGD + Momentum ...")
        self.training_time = 0
        self.non_momentum_training_time = 0
        self.svm = SVM.SVM(X=x_train, y=y_train, alpha=self.alpha, C=self.C, gamma=self.gamma,
                           n_features=self.n_features, randomize=self.randomize)
        w_init = w_init
        indices_init = indices_init
        self.training_time -= (time.time())
        self.w, self.effective_epochs, self.cost, self.io_time, self.initial_cost = self.svm.train_sgd_momentum(X=x_train, y=y_train, X_test=x_test, y_test=y_test, alpha=self.alpha,
                                                epochs=self.epochs, w_init=w_init, indices_init=indices_init, log_file=log_file, tolerance=tolerance)
        self.training_time += (time.time())
        self.training_time = self.training_time - self.io_time
        Print.Print.info1("Momentum Training Completed !")
        return self.w, self.effective_epochs, self.initial_cost, self.cost, self.training_time


    def test(self):
        self.acc_m = 0
        self.acc_sgd = 0
        if self.bulk:
            testing_filepath = self.testing_file
            print("Loading Bulk Testing Files")
            files = os.listdir(testing_filepath)
            print("File Path : " + testing_filepath)
            print(files)
            self.bulk_testing_x = []
            self.bulk_testing_y = []
            for file in files:
                print("Loading Testing Bulk File : " + file)
                testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath + "/" + file,
                                                       n_features=self.n_features)
                x_testing, y_testing = testing_loader.load_all_data()

                y_pred_m = self.svm.custom_predict(x_testing, w=self.w_m)
                self.acc_m += self.svm.get_accuracy(y_test=y_testing, y_pred=y_pred_m)

                y_pred_sgd = self.svm.custom_predict(x_testing, w=self.w_sgd)
                self.acc_sgd += self.svm.get_accuracy(y_test=y_testing, y_pred=y_pred_sgd)
            self.acc_m = self.acc_m / len(files)
            self.acc_sgd = self.acc_sgd / len(files)

        else:
            y_pred_m = self.svm.custom_predict(X=self.x_testing, w=self.w_m)
            self.acc_m = self.svm.get_accuracy(y_test=self.y_testing, y_pred=y_pred_m)

            y_pred_sgd = self.svm.custom_predict(X=self.x_testing, w=self.w_sgd)
            self.acc_sgd = self.svm.get_accuracy(y_test=self.y_testing, y_pred=y_pred_sgd)

        return self.acc_m, self.acc_sgd

    def advanced_test(self, x_testing, y_testing, w):
        #y_pred = self.svm.predict(X=self.x_testing)
        #self.acc = self.svm.get_accuracy(y_test=self.y_testing, y_pred=y_pred)
        self.acc = 0
        if self.bulk:
            testing_filepath = self.testing_file
            print("Loading Bulk Testing Files")
            files = os.listdir(testing_filepath)
            print("File Path : " + testing_filepath)
            print(files)
            self.bulk_testing_x = []
            self.bulk_testing_y = []
            self.acc = 0
            for file in files:
                print("Loading Testing Bulk File : " + file)
                testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath + "/" + file,
                                                       n_features=self.n_features)
                x_testing, y_testing = testing_loader.load_all_data()
                y_pred = self.svm.custom_predict(x_testing, w=w)
                self.acc += self.svm.get_accuracy(y_test=y_testing, y_pred=y_pred)
            self.acc = self.acc / len(files)
        else:
            y_pred = self.svm.custom_predict(X=x_testing, w=w)
            self.acc = self.svm.get_accuracy(y_test=y_testing, y_pred=y_pred)

        return self.acc

    def advanced_stats(self, effective_epochs=0, accuracy=0, training_time=0, initial_cost=0, final_cost=0):
        Print.Print.result1(self.exp_name + "Momentum  Accuracy : " + str(self.acc) + "%")
        Print.Print.result1(self.exp_name + "Momentum Time : " + str(self.training_time) + " s")
        fp = open("logs/benchmark/summary/" + socket.gethostname() + "_" + self.exp_name + "_momentum_results.txt", "a")
        # fp.write("alpha : " + str(self.alpha) + ", epochs : " + str(self.epochs) + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
        fp.write(str(self.alpha) + "," + str(self.epochs) +  "," + str(effective_epochs) + ","  + str(initial_cost) + "," + str(final_cost) + "," + str(accuracy) + "," +  str(training_time) + "\n")
        fp.close()


    def stats(self):
        Print.Print.result1("Results")
        Print.Print.result1(self.exp_name + " Momentum Accuracy : " + str(self.acc_m) + "%")
        Print.Print.result1(self.exp_name + " Momentum Time : " + str(self.training_time) + " s")
        Print.Print.result1(self.exp_name + " SGD Accuracy : " + str(self.acc_sgd) + "%")
        Print.Print.result1(self.exp_name + " SGD Time : " + str(self.non_momentum_training_time) + " s")
        fp = open("logs/" + socket.gethostname()+"_" + self.exp_name + "_sgd_momentum_results.txt", "a")
        #fp.write("alpha : " + str(self.alpha) + ", C : " + str(self.C) + ", gamma : " + str(self.gamma) + ", epochs : " + str(self.epochs)
        #         + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
        fp.write(str(self.alpha) + "," + str(self.C) + "," + str(self.gamma) + "," + str(self.epochs)
                 + "," + str(self.acc_m) + "," + str(self.training_time) + "\n")
        fp.close()
        fp = open("logs/" + socket.gethostname() + "_" + self.exp_name + "_sgd_non_momentum_results.txt", "a")
        # fp.write("alpha : " + str(self.alpha) + ", C : " + str(self.C) + ", gamma : " + str(self.gamma) + ", epochs : " + str(self.epochs)
        #         + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
        fp.write(str(self.alpha) + "," + str(self.C) + "," + str(self.gamma) + "," + str(self.epochs)
                 + "," + str(self.acc_sgd) + "," + str(self.non_momentum_training_time) + "\n")
        fp.close()
