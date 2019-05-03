import sys
import os
import socket
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from experiment import BenchmarkSGDAdam

#datasets = ['ijcnn1', 'webspam', 'heart']
#n_features = [22, 254, 13]
#splits = [False, True, False]
datasets = [ 'ijcnn1', 'heart', 'webspam', 'cod-rna', 'phishing', 'breast_cancer', 'w8a', 'a9a', 'real-slim']
n_features = [22, 13, 254, 8, 68, 10, 300 , 123, 20958]
splits = [False, False, True, False, True, True, False, False, True]
datasets = ['cod-rna']
n_features = [8]
splits = [False]
#
# datasets = ['ijcnn1']
# n_features = [22]
# splits = [False]
#beta1_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.93, 0.95, 0.99, 0.999]
#beta2_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.93, 0.95, 0.99, 0.999]
beta1_range = [0.999]
beta2_range = [0.999]
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
	C=1
	exp_name = dataset


	bmsvmsgd = BenchmarkSGDAdam.BenchmarkSGDAdam(exp_name=exp_name, training_file=training_file, testing_file=testing_file,
												 alpha=alpha, features=features, epochs=epochs, labelfix=labelfix, randomize=randomize, split=split, auto=True, bulk=bulk)
	bmsvmsgd.load_data()
	for beta1 in beta1_range:
		for beta2 in beta2_range:
			for i in range(0,10):
				bmsvmsgd.train(beta1=beta1, beta2=beta2)
				bmsvmsgd.test()
				bmsvmsgd.stats()
