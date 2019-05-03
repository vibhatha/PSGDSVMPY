import sys
import os
import socket
HOME = os.environ['HOME']
sys.path.insert(1, HOME + '/github/StreamingSVM')
import csv
from itertools import islice
import numpy as np
from operations import Print

class DistributedDataLoader:

    def __init__(self, source_file, training_file=None, testing_file=None, pos=0, buffer=100, n_features=1, n_samples=100, train_samples=0, test_samples=0, ratio=0.8, world_size=4, rank=0, split=False):
        self.source_file = source_file
        self.pos = pos # starting position to read
        self.buffer = buffer # buffer size to read (number of lines)
        self.n_features = n_features # number of features in a data point
        self.n_samples = n_samples
        self.ratio = ratio
        self.world_size = world_size
        self.rank = rank
        self.split = split
        self.training_file = training_file
        self.testing_file = testing_file
        self.train_samples = train_samples
        self.testing_samples = test_samples


    def getRows(self):
        #reference : https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
        num_lines = sum(1 for line in open(self.source_file))
        return num_lines

    def load_testing_data(self):
        if(self.split == True):
            start_id  = int(self.n_samples * self.ratio)
            end_id = self.n_samples
            Print.Print.info1('Loading Data In Distributed Format In Rank : ' + str(self.rank) + ', Start : ' + str(
                start_id) + ", End : " + str(end_id)+", Testing Samples : " + str(end_id-start_id))
            data_list = []
            with open(self.source_file, 'rt') as f:
                reader = csv.reader(f, delimiter=',')
                for row in islice(reader, start_id, end_id):
                    data_list.append(row)

            data = np.array(data_list,'f')
            np.random.shuffle(data)
            y = data[:, 0]
            x = np.delete(data, 0, axis=1)
            return x,y
        if(self.split == False):
            start_id = 0
            end_id = self.testing_samples
            Print.Print.info1('Loading Data In Distributed Format In Rank : ' + str(self.rank) + ', Start : ' + str(
                start_id) + ", End : " + str(end_id) + ", Testing Samples : " + str(end_id - start_id))
            data_list = []
            with open(self.testing_file, 'rt') as f:
                reader = csv.reader(f, delimiter=',')
                for row in islice(reader, start_id, end_id):
                    data_list.append(row)

            data = np.array(data_list, 'f')
            np.random.shuffle(data)
            y = data[:, 0]
            x = np.delete(data, 0, axis=1)
            return x, y



    def load_training_data_chunks(self):
        if (self.split == True):
            data_list = []
            testing_samples = self.n_samples - int(self.n_samples * self.ratio)
            training_samples = self.n_samples - testing_samples
            data_per_machine = training_samples / self.world_size
            start = self.rank * data_per_machine
            end = start + data_per_machine
            Print.Print.info1('Loading Data In Distributed Format In Rank : ' + str(self.rank) + ', Start : ' + str(start) + ", End : " + str(end))
            with open(self.source_file, 'rt') as f:
                reader = csv.reader(f, delimiter=',')
                for row in islice(reader, start, end):
                    data_list.append(row)

            data = np.array(data_list,'f')
            np.random.shuffle(data)
            y = data[:, 0]
            x = np.delete(data, 0, axis=1)
            return x,y
        if(self.split == False):
            data_list = []
            testing_samples = self.testing_samples
            training_samples = self.train_samples
            data_per_machine = training_samples / self.world_size
            start = self.rank * data_per_machine
            end = start + data_per_machine
            Print.Print.info1('Loading Data In Distributed Format In Rank : ' + str(self.rank) + ', Start : ' + str(
                start) + ", End : " + str(end))
            with open(self.source_file, 'rt') as f:
                reader = csv.reader(f, delimiter=',')
                for row in islice(reader, start, end):
                    data_list.append(row)

            data = np.array(data_list, 'f')
            np.random.shuffle(data)
            y = data[:, 0]
            x = np.delete(data, 0, axis=1)
            return x, y



    def load_training_data_batch_per_core(self):
        all_data_list_x = []
        all_data_list_y = []
        testing_samples = self.n_samples - int(self.n_samples * self.ratio)
        training_samples = self.n_samples - testing_samples
        data_per_machine = training_samples / self.world_size
        for m in range(0,self.world_size):
            data_list=[]
            start = m * data_per_machine
            end = start + data_per_machine
            Print.Print.info1('Loading Data In Distributed Format In Rank : ' + str(m) + ', Start : ' + str(
                start) + ", End : " + str(end))
            with open(self.source_file, 'rt') as f:
                reader = csv.reader(f, delimiter=',')
                for row in islice(reader, start, end):
                    data_list.append(row)

            data = np.array(data_list, 'f')
            np.random.shuffle(data)
            y = data[:, 0]
            x = np.delete(data, 0, axis=1)
            all_data_list_x.append(x)
            all_data_list_y.append(y)
        return all_data_list_x, all_data_list_y











