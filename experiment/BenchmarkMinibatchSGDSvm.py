import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from sgd import SVM
from operations import LoadLibsvm
import time
from operations import Print
import socket


class BenchmarkMinibatchSgdSVM:

    def __init__(self, exp_name ,training_file, testing_file, features=1, alpha=0.01,
                 epochs=100, labelfix=False, split=False, randomize=False, auto=False, minibatch=True, batch_size=10000,
                 custom_minibatch=False,custom_minibatch_without_coeff=False):
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
        self.minibatch = minibatch
        self.custom_minibatch = custom_minibatch
        self.minibatch_size = batch_size
        self.custom_minibatch_without_coeff = custom_minibatch_without_coeff

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

    def train(self):
        self.training_time = 0
        self.svm = SVM.SVM(X=self.x_training, y=self.y_training, alpha=self.alpha, n_features=self.n_features, randomize=self.randomize)
        self.training_time -= (time.time())
        if self.auto and (not (self.minibatch or self.custom_minibatch or self.custom_minibatch_without_coeff)):
            Print.Print.console('Auto Alpha Mode')
            self.svm.train(X=self.x_training, y=self.y_training, alpha=self.alpha, epochs=self.epochs)
        if not self.auto:
            Print.Print.console('Manual Alpha Mode')
            self.svm.manual_sgd_train(X=self.x_training, y=self.y_training, alpha=self.alpha, epochs=self.epochs)
        if self.minibatch or (not self.auto):
            Print.Print.console('Minibatch SGD Mode')
            self.svm.minibatch_sgd_train(X=self.x_training, y=self.y_training, alpha=self.alpha, epochs=self.epochs, batch_size=self.minibatch_size)
        if self.custom_minibatch or (not self.auto):
            Print.Print.console('Custom Minibatch SGD Mode')
            self.svm.custom_minibatch_sgd_train(X=self.x_training, y=self.y_training, alpha=self.alpha, epochs=self.epochs, batch_size=self.minibatch_size)
        self.training_time += (time.time())
        if self.custom_minibatch_without_coeff:
            Print.Print.console('Custom Minibatch SGD Mode')
            self.svm.custom_minibatch_sgd_train_without_coeff(X=self.x_training, y=self.y_training, alpha=self.alpha,
                                                epochs=self.epochs, batch_size=self.minibatch_size)
        return self.svm


    def train_minibatch_version1(self):
        self.training_time = 0
        self.svm = SVM.SVM(X=self.x_training, y=self.y_training, alpha=self.alpha, n_features=self.n_features,
                           randomize=self.randomize)
        self.training_time -= (time.time())
        Print.Print.console('Minibatch Version 1 Training')
        self.svm.minibatch_sgd_train_version1(X=self.x_training, y=self.y_training, alpha=self.alpha, epochs=self.epochs, batch_size=self.minibatch_size)
        self.training_time += (time.time())


    def test(self):
        y_pred = self.svm.predict(X=self.x_testing)
        self.acc = self.svm.get_accuracy(y_test=self.y_testing, y_pred=y_pred)
        return self.acc

    def stats(self):
        Print.Print.result1(self.exp_name + " Accuracy : " + str(self.acc) + "%")
        Print.Print.result2(self.exp_name + " Time : " + str(self.training_time) + " s")
        fp = open("logs/minibatch/sgd/" + socket.gethostname()+"_"  + self.exp_name + "_minibatch_sgd_results.txt", "a")
        #fp.write("alpha : " + str(self.alpha) + ", epochs : " + str(self.epochs) + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
        fp.write(str(self.minibatch_size) + ", " +str(self.alpha) + "," + str(self.epochs) + "," + str(self.acc) + "," + str(self.training_time) + "\n")
        fp.close()


