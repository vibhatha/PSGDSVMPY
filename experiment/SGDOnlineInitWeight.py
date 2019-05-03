import sys
import os
import socket

HOME = os.environ['HOME']
sys.path.insert(1, HOME + '/github/StreamingSVM')
from experiment import BenchmarkOnlineSGDSvm
from operations import Print
from operations import LoadLibsvm
import numpy as np
import datetime

# datasets = ['ijcnn1', 'webspam', 'heart']
# n_features = [22, 254, 13]
# splits = [False, True, False]
datasets = ['ijcnn1', 'heart', 'webspam', 'cod-rna', 'phishing', 'breast_cancer', 'w8a', 'a9a', 'real-slim']
n_features = [22, 13, 254, 8, 68, 10, 300, 123, 20958]
splits = [False, False, True, False, True, True, False, False, True]

datasets = ['phishing']
n_features = [68]
splits = [True]

for dataset, features, split in zip(datasets, n_features, splits):

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
    features = features
    alpha = 1
    epochs = 200
    ld = 1
    eta = 0.1
    labelfix = False
    split = split
    randomize = True
    gamma = 16
    degree = 1
    kernel = 'rbf'
    minibatch_size = 1000
    minibatch = True
    C = 1
    exp_name = dataset
    base_weight_path = '/home/vibhatha/Documents/Research/SVM-SGD/weights/2018_OCT_18/weights/benchmark/'
    weight_file = 'j-106_phishing_2018-10-16_18:20:34.111968_0_final_sgd_weights.txt'


    w = np.loadtxt(base_weight_path+weight_file)

    bmsvmsgd = BenchmarkOnlineSGDSvm.BenchmarkOnlineSGDSvm(exp_name=exp_name, training_file=training_file,
                                                           testing_file=testing_file,
                                                           alpha=alpha, features=features, epochs=epochs,
                                                           labelfix=labelfix, randomize=randomize, split=split,
                                                           auto=True)
    bmsvmsgd.load_data()

    for i in range(0, 10):
        bmsvmsgd.train_sgd_online_init_weight(w)
        bmsvmsgd.test()
        bmsvmsgd.stats()
