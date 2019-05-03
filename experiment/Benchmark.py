import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from experiment import BenchmarkSgdSvm
from experiment import BenchmarkPegasosSvm
from experiment import BenchmarkSGDMomentum

class Benchmark:

    def __init__(self, exp_name, training_file, testing_file, features=1, alpha=0.01, epochs=100, eta=0.01,
                 ld=0.0001, labelfix=False, split=False, randomize=False, C=1, gamma=1):
        self.training_file = training_file
        self.split = split
        self.testing_file = testing_file
        self.n_features = features
        self.alpha = alpha
        self.epochs = epochs
        self.exp_name = exp_name
        self.eta = eta
        self.ld = ld
        self.C = C
        self.gamma = gamma
        self.labelfix = labelfix
        self.randomize = randomize

    def benchmark_sgd(self):
        bmsvmsgd = BenchmarkSgdSvm.BenchmarkSgdSVM(exp_name=self.exp_name, training_file=self.training_file, testing_file=self.testing_file,
                                             alpha=self.alpha, features=self.n_features, epochs=self.epochs, labelfix=self.labelfix,
                                                   randomize=self.randomize, split=self.split, auto=True)
        bmsvmsgd.load_data()
        bmsvmsgd.train()
        bmsvmsgd.test()
        bmsvmsgd.stats()

    def benchmark_pegasos(self):
        bmsvmsgd = BenchmarkPegasosSvm.BenchmarkPegasosSvm(exp_name=self.exp_name, training_file=self.training_file,
                                                   testing_file=self.testing_file,
                                                   eta=self.eta, ld=self.ld, features=self.n_features, epochs=self.epochs, labelfix=self.labelfix,
                                                           randomize=self.randomize, split=self.split)
        bmsvmsgd.load_data()
        bmsvmsgd.train()
        bmsvmsgd.test()
        bmsvmsgd.stats()

    def benchmark_sgd_momentum(self):
        bmsvmsgdmomentum = BenchmarkSGDMomentum.BenchmarkSGDMomentum(exp_name=self.exp_name, training_file=self.training_file, testing_file=self.testing_file,
                                             alpha=self.alpha, C=self.C, gamma=self.gamma, features=self.n_features, epochs=self.epochs, labelfix=self.labelfix,
                                                                     randomize=self.randomize, split=self.split)
        bmsvmsgdmomentum.load_data()
        bmsvmsgdmomentum.train()
        bmsvmsgdmomentum.test()
        bmsvmsgdmomentum.stats()

    def benchmark_manual_sgd(self):
        bmsvmsgd = BenchmarkSgdSvm.BenchmarkSgdSVM(exp_name=self.exp_name, training_file=self.training_file,
                                                   testing_file=self.testing_file,
                                                   alpha=self.alpha, features=self.n_features, epochs=self.epochs,
                                                   labelfix=self.labelfix, randomize=self.randomize, split=self.split, auto=False)
        bmsvmsgd.load_data()
        bmsvmsgd.train()
        bmsvmsgd.test()
        bmsvmsgd.stats()


base_path = '/home/vibhatha/data/svm/'
dataset = 'webspam'
training_file = base_path + dataset + '/training.csv'
testing_file = base_path + dataset + '/testing.csv'
features = 254
alpha = 0.01
epochs = 30
ld = 0.0000001
eta = 0.1
labelfix = False
split = False
randomize = True
C = 1
gamma = 1
## For Samples with both training and testing data
# bm = Benchmark(exp_name=dataset,training_file=training_file, testing_file=testing_file,
#                features=features, alpha=alpha, epochs=epochs, eta=eta, ld=ld,
#                labelfix=labelfix, randomize=randomize, split=False)
# bm.benchmark_sgd()
# bm.benchmark_pegasos()

## For samples with only a training sample
## Split the data into a ratio (0.6) to get the training and the testing datasets

bm = Benchmark(exp_name=dataset,training_file=training_file, testing_file=testing_file, features=features,
                alpha=alpha, epochs=epochs, eta=eta, ld=ld, C=C, gamma=gamma,
                labelfix=labelfix, randomize=randomize, split=True)
bm.benchmark_manual_sgd()
#bm.benchmark_sgd_momentum()
