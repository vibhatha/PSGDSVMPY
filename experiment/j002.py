import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
import socket
from experiment import BulkBenchmark

base_path = ''
hostname = socket.gethostname()

if hostname == 'vibhatha-ThinkPad-P50':
	base_path = '/home/vibhatha/data/svm/'
else:
	base_path = '/N/u/vlabeyko/data/svm/svm/'

bulk = True
dataset = 'rcv1'
training_file = base_path + dataset + '/training.csv'
testing_file = base_path + dataset + '/bulk'
features = 47236
alpha = 1
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
minibatch = False
C=1

for i in range(0, 10):
    for al in [0.1, 0.01, 0.001, 1]:
        bm = BulkBenchmark.BulkBenchmark(exp_name=dataset, training_file=training_file, testing_file=testing_file,
                                         features=features,
                                         alpha=alpha, epochs=epochs, eta=eta, ld=ld, C=C, gamma=gamma, degree=degree,
                                         kernel=kernel,
                                         labelfix=labelfix, randomize=randomize, split=split, minibatch=minibatch,
                                         minibatch_size=minibatch_size, bulk=bulk)
        bm.benchmark_adagrad()


for i in range(0,5):
    bm = BulkBenchmark.BulkBenchmark(exp_name=dataset, training_file=training_file, testing_file=testing_file, features=features,
                   alpha=alpha, epochs=epochs, eta=eta, ld=ld, C=C, gamma=gamma, degree=degree, kernel=kernel,
                   labelfix=labelfix, randomize=randomize, split=split, minibatch=minibatch, minibatch_size=minibatch_size, bulk=bulk)

    bm.benchmark_sgd()
    bm.benchmark_momentum_factor()

for ldi in [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1 ]:
    bm1 = BulkBenchmark.BulkBenchmark(exp_name=dataset, training_file=training_file, testing_file=testing_file, features=features,
                   alpha=alpha, epochs=epochs, eta=eta, ld=ld, C=C, gamma=gamma, degree=degree, kernel=kernel,
                   labelfix=labelfix, randomize=randomize, split=split, minibatch=minibatch, minibatch_size=minibatch_size, bulk=bulk)

    bm1.benchmark_pegasos()
