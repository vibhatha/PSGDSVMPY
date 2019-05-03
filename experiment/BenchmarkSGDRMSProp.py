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

class BenchmarkSGDRMSProp:

    def __init__(self, exp_name ,training_file, testing_file, features=1, alpha=0.01,
                 epochs=100, labelfix=False, split=False, randomize=False, auto=False, bulk=False,
                 gamma=1, C=1):
        self.training_file = training_file
        self.testing_file = testing_file
        self.n_features = features
        self.alpha = alpha
        self.epochs = epochs
        self.exp_name = exp_name
        self.labelfix = labelfix
        self.split = split
        self.randomize = randomize
        self.auto = auto
        self.bulk = bulk
        self.gamma = gamma
        self.C = C

    def load_data(self):
        training_filepath = self.training_file
        testing_filepath = self.testing_file
        if not self.split and not self.bulk:
            training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=self.n_features)
            testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath, n_features=self.n_features)
            self.x_training, self.y_training = training_loader.load_all_data()
            self.x_testing, self.y_testing = testing_loader.load_all_data()

        if self.split and not self.bulk:
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

        if self.bulk and not self.split:
            # here we consider a larger testing data set.
            print("Loading Bulk Training File")
            training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=self.n_features)
            self.x_training, self.y_training = training_loader.load_all_data()

            return self.x_training, self.y_training
        else:
            return self.x_training, self.y_training, self.x_testing, self.y_testing
        print("Training and Testing data loaded ...")

    def train(self,beta=0.90):
        self.training_time = 0
        self.beta = beta
        self.svm = SVM.SVM(X=self.x_training, y=self.y_training, alpha=self.alpha, n_features=self.n_features, randomize=self.randomize)
        self.w_init = self.svm.init_weights()
        self.training_time -= (time.time())
        Print.Print.console('RMS PROP SGD Training ...')
        self.w = self.svm.rms_prop_sgd_train(X=self.x_training, y=self.y_training, alpha=self.alpha, epochs=self.epochs, w_init=self.w_init, beta=beta)
        self.training_time += (time.time())
        return self.svm

    def train_benchmark_light(self, w_init=[], x_train=[], y_train=[], x_test=[], y_test=[], log_file='',
                              tolerance=0.001, indices_init=[], beta=0.98, alpha=0.01):
        self.training_time = 0
        self.effective_epochs = 0
        self.cost = 0
        self.initial_cost = 0
        self.beta = beta
        self.io_time = 0
        Print.Print.info1("Training with SGD + RMSProp ...")
        self.training_time = 0
        self.svm = SVM.SVM(X=x_train, y=y_train, alpha=self.alpha, C=self.C, gamma=self.gamma,
                           n_features=self.n_features, randomize=self.randomize)
        w_init = w_init
        indices_init = indices_init
        self.training_time -= (time.time())
        self.w, self.effective_epochs, self.cost, self.io_time, self.initial_cost = self.svm.train_rmsprop_sgd(
            X=x_train, y=y_train, X_test=x_test, y_test=y_test, alpha=alpha, beta=beta,
            epochs=self.epochs, w_init=w_init, indices_init=indices_init, log_file=log_file, tolerance=tolerance)
        self.training_time += (time.time())
        self.training_time = self.training_time - self.io_time
        Print.Print.info1("RMSProp Training Completed !")
        return self.w, self.effective_epochs, self.initial_cost, self.cost, self.training_time


    def train_benchmark_heavy(self, w_init=[], x_train=[], y_train=[], x_test=[], y_test=[], log_file='',
                              tolerance=0.001, indices_init=[], beta=0.90):
        self.training_time = 0
        self.effective_epochs = 0
        self.cost = 0
        self.initial_cost = 0
        self.beta = beta
        self.io_time = 0
        Print.Print.info1("Training with SGD + RMSProp ...")
        self.training_time = 0
        self.svm = SVM.SVM(X=x_train, y=y_train, alpha=self.alpha, C=self.C, gamma=self.gamma,
                           n_features=self.n_features, randomize=self.randomize)
        w_init = w_init
        indices_init = indices_init
        self.training_time -= (time.time())
        self.w, self.effective_epochs, self.cost, self.io_time, self.initial_cost = self.svm.train_rmsprop_sgd(
            X=x_train, y=y_train, X_test=x_test, y_test=y_test, alpha=self.alpha, beta=beta,
            epochs=self.epochs, w_init=w_init, indices_init=indices_init, log_file=log_file, tolerance=tolerance)
        self.training_time += (time.time())
        self.training_time = self.training_time - self.io_time
        Print.Print.info1("RMSProp Training Completed !")
        return self.w, self.effective_epochs, self.initial_cost, self.cost, self.training_time


    def test(self):
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
            for file in files:
                print("Loading Testing Bulk File : " + file)
                testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath + "/" + file,
                                                       n_features=self.n_features)
                x_testing, y_testing = testing_loader.load_all_data()
                y_pred = self.svm.custom_predict(x_testing, w=self.w)
                self.acc += self.svm.get_accuracy(y_test=y_testing, y_pred=y_pred)
            self.acc = self.acc / len(files)
        else:
            y_pred = self.svm.custom_predict(X=self.x_testing, w=self.w)
            self.acc = self.svm.get_accuracy(y_test=self.y_testing, y_pred=y_pred)

        return self.acc


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
        Print.Print.result1(self.exp_name + "RMSPROP Accuracy : " + str(self.acc) + "%")
        Print.Print.result1(self.exp_name + "RMSPROP Time : " + str(self.training_time) + " s")
        fp = open("logs/benchmark/summary/" + socket.gethostname() + "_" + self.exp_name + "_rmsprop_results.txt", "a")
        # fp.write("alpha : " + str(self.alpha) + ", epochs : " + str(self.epochs) + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
        fp.write(str(self.alpha) + "," +str(self.beta) + "," + str(self.epochs) +  "," + str(effective_epochs) + ","  + str(initial_cost) + "," + str(final_cost) + "," + str(accuracy) + "," +  str(training_time) + "\n")
        fp.close()

    def stats(self):
        Print.Print.result1(self.exp_name + " Accuracy : " + str(self.acc) + "%")
        Print.Print.result1(self.exp_name + " Time : " + str(self.training_time) + " s")
        fp = open("logs/" + socket.gethostname()+"_"   + self.exp_name + "_rmsprop_results.txt", "a")
        #fp.write("alpha : " + str(self.alpha) + ", epochs : " + str(self.epochs) + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
        fp.write(str(self.alpha) + "," +str(self.beta) + "," + str(self.epochs) + "," + str(self.acc) + "," + str(self.training_time) + "\n")
        fp.close()


