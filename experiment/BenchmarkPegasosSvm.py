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


class BenchmarkPegasosSvm:


    def __init__(self, exp_name ,training_file, testing_file, features=1, eta=0.01,
                 ld=0.001, epochs=100, labelfix=False, split=False, randomize=False, bulk=False):
        self.training_file = training_file
        self.testing_file = testing_file
        self.n_features = features
        self.eta = eta
        self.ld = ld
        self.epochs = epochs
        self.exp_name = exp_name
        self.labelfix = labelfix
        self.split = split
        self.randomize = randomize
        self.bulk = bulk

    def load_data(self):
        Print.Print.info1("Loading data ...")
        training_filepath = self.training_file
        testing_filepath = self.testing_file
        if (self.split == False and not self.bulk ):
            training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=self.n_features)
            testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath, n_features=self.n_features)
            self.x_training, self.y_training = training_loader.load_all_data()
            self.x_testing, self.y_testing = testing_loader.load_all_data()

        if (self.split == True and not self.bulk):
            Print.Print.info1("Splitting data ... ")
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
            # here we consider a larger testing data set.
            print("Loading Bulk Training File")
            training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=self.n_features)
            self.x_training, self.y_training = training_loader.load_all_data()
            return self.x_training, self.y_training
        else:
            return self.x_training, self.y_training, self.x_testing, self.y_testing
        print("Training and Testing data loaded ...")


    def train(self):
        Print.Print.info1("Training SVM ...")
        self.training_time = 0
        self.svm = SVM.SVM(X=self.x_training, y=self.y_training, eta=self.eta, ld= self.ld, n_features=self.n_features, epochs=self.epochs, randomize=self.randomize)
        self.svm.init_weights()
        self.training_time -= (time.time())
        self.w = self.svm.pegasus_train_cost(X=self.x_training, y=self.y_training)
        self.training_time += (time.time())
        return self.svm

    def test(self):
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

    def stats(self):
        Print.Print.result1(self.exp_name + " Pegasos Accuracy : " + str(self.acc) + "%")
        Print.Print.result1( self.exp_name + " Pegasos Time : " + str(self.training_time) + " s" + Bcolors.Bcolors.OKBLUE)
        fp = open("logs/" + socket.gethostname()+"_"  + self.exp_name + "_pegasos_results.txt", "a")
        #fp.write("eta : " + str(self.eta) + ", lambda : " + str(self.ld) + ", epochs : " + str(self.epochs) + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
        fp.write(str(self.eta) + "," + str(self.ld) + "," + str(self.epochs) + "," + str(self.acc) + "," + str(self.training_time) + "\n")
        fp.close()
