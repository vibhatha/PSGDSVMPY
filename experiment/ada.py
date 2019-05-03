import sys
import os
import socket
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from experiment import BenchmarkSGDAda

datasets = [ 'ijcnn1', 'heart', 'webspam', 'cod-rna', 'phishing', 'breast_cancer', 'w8a', 'a9a', 'real-slim']
n_features = [22, 13, 254, 8, 68, 10, 300 , 123, 20958]
splits = [False, False, True, False, True, True, False, False, True]
datasets = [ 'ijcnn1']
n_features = [22]
splits = [False]

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
	alpha = 0.01
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
	C=1
	exp_name = dataset

	bmsvmsgd = BenchmarkSGDAda.BenchmarkSGDAda(exp_name=exp_name, training_file=training_file, testing_file=testing_file,
												 alpha=alpha, features=features, epochs=epochs, labelfix=labelfix, randomize=randomize, split=split, auto=True, bulk=bulk)
	bmsvmsgd.load_data()
	for alpha in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]:
		for i in range(0,10):
			bmsvmsgd.train_alpha(alpha)
			bmsvmsgd.test()
			bmsvmsgd.stats()
