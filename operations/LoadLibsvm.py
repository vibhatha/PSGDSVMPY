import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from numpy import genfromtxt
import numpy as np


class LoadLibSVM:

    def __init__(self, filename, n_features=1):
        self.filename = filename
        self.n_features = n_features

    def load_data(self):
        print("Reading " + self.filename + " with " + str(self.n_features))
        data = genfromtxt(self.filename, delimiter=',')
        rem = len(data) % 5
        if(rem == 0):
            for i in range(0, rem):
                data = np.delete(data, 0, 0)

        arrs = np.array_split(data, 5)

        training = np.concatenate((arrs[0],arrs[1], arrs[2]), axis=0)
        testing = np.concatenate((arrs[3], arrs[4]), axis=0)
        print(len(data),training.shape,training[0].shape)
        y_training = training[:,0]
        x_training = np.delete(training, 0, axis=1)
        y_testing = testing[:, 0]
        x_testing = np.delete(testing, 0, axis=1)
        print(x_training.shape, y_training.shape, x_testing.shape, y_testing.shape)
        return x_training, y_training, x_testing, y_testing

    def load_all_data(self):
        data = genfromtxt(self.filename, delimiter=',')
        y = data[:, 0]
        x = np.delete(data, 0, axis=1)
        return x, y

