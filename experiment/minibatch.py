import sys
import os
import socket

HOME = os.environ['HOME']
sys.path.insert(1, HOME + '/github/StreamingSVM')
from experiment import BenchmarkMinibatchSGDSvm

base_path = ''
hostname = socket.gethostname()

if hostname == 'vibhatha-ThinkPad-P50':
    base_path = '/home/vibhatha/data/svm/'
else:
    base_path = '/N/u/vlabeyko/data/svm/svm/'

dataset = 'cod-rna'
training_file = base_path + dataset + '/training.csv'
testing_file = base_path + dataset + '/testing.csv'
features = 8
alpha = 0.001
epochs = 200
ld = 1
eta = 0.1
labelfix = False
split = False
randomize = True
gamma = 16
degree = 1
kernel = 'rbf'
minibatch_size = 1000
minibatch = True
custom_minibatch_without_coeff = False
C = 1
exp_name = dataset

bmminisgd = BenchmarkMinibatchSGDSvm.BenchmarkMinibatchSgdSVM(exp_name=exp_name,
                                                              training_file=training_file,
                                                              testing_file=testing_file,
                                                              alpha=alpha, features=features,
                                                              epochs=epochs,
                                                              labelfix=labelfix, randomize=randomize,
                                                              split=split,
                                                              auto=True, minibatch=False, custom_minibatch=True,
                                                              batch_size=minibatch_size,
                                                              custom_minibatch_without_coeff=False)

bmminisgd.load_data()
bmminisgd.train_minibatch_version1()
bmminisgd.test()
bmminisgd.stats()
