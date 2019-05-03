import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from experiment import BenchmarkSgdSvm
from experiment import BenchmarkPegasosSvm
import numpy as np
import argparse


class DynamicBenchmark:

    @staticmethod
    def get_bool(input):
        if input == 'True':
            return True
        else:
            return False

    @staticmethod
    def initialize(args):
        randomize = DynamicBenchmark.get_bool(args.randomize)
        labelFix = DynamicBenchmark.get_bool(args.labelFix)
        split = DynamicBenchmark.get_bool(args.split)
        dmb = DynamicBenchmark(exp_name=args.exp_name, training_file=args.trainFile, testing_file=args.testFile,
                               features=args.features,
                               alpha=args.alpha, epochs=args.epochs, eta=args.eta, ld=args.lambdaa,
                               labelfix=labelFix, randomize=randomize, split=split)
        return dmb

    def __init__(self, exp_name, training_file, testing_file, features=1, alpha=0.01, epochs=100, eta=0.01,
                 ld=0.0001, labelfix=False, split=False, randomize=False):
        self.training_file = training_file
        self.split = split
        self.testing_file = testing_file
        self.n_features = features
        self.alpha = alpha
        self.epochs = epochs
        self.exp_name = exp_name
        self.eta = eta
        self.ld = ld
        self.labelfix = labelfix
        self.randomize = randomize

    def benchmark_sgd(self):
        bmsvmsgd = BenchmarkSgdSvm.BenchmarkSgdSVM(exp_name=self.exp_name, training_file=self.training_file, testing_file=self.testing_file,
                                             alpha=self.alpha, features=self.n_features, epochs=self.epochs, labelfix=self.labelfix, randomize=self.randomize, split=self.split, auto=True)
        bmsvmsgd.load_data()
        bmsvmsgd.train()
        bmsvmsgd.test()
        bmsvmsgd.stats()

    def benchmark_manual_sgd(self):
        bmsvmsgd = BenchmarkSgdSvm.BenchmarkSgdSVM(exp_name=self.exp_name, training_file=self.training_file,
                                                   testing_file=self.testing_file,
                                                   alpha=self.alpha, features=self.n_features, epochs=self.epochs,
                                                   labelfix=self.labelfix, randomize=self.randomize, split=self.split,
                                                   auto=False)
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

    @staticmethod
    def genOpts():
        parser = argparse.ArgumentParser()
        parser.add_argument("-tr", "--trainFile", help="Training File Path", type=str, required=True)
        parser.add_argument("-tw", "--testFile", help="Testing File Path", type=str, required=False)
        parser.add_argument("-s", "--split", help="Testing File Path", type=str, required=False)
        parser.add_argument("-f", "--features", help="Testing File Path", type=int, required=False, default=1)
        parser.add_argument("-a", "--alpha", help="Testing File Path", type=float, required=False, default=0.01)
        parser.add_argument("-e", "--epochs", help="Testing File Path", type=int, required=False, default=100)
        parser.add_argument("-x", "--exp_name", help="Testing File Path", type=str, required=True, default='Experiment-01-')
        parser.add_argument("-t", "--eta", help="Testing File Path", type=float, required=False, default=0.01)
        parser.add_argument("-l", "--lambdaa", help="Testing File Path", type=float, required=False, default=0.01)
        parser.add_argument("-lf", "--labelFix", help="Testing File Path", type=str, required=False)
        parser.add_argument("-r", "--randomize", help="Testing File Path", type=str, required=False)
        args = parser.parse_args()
        return args


args = DynamicBenchmark.genOpts()
dmbinit = DynamicBenchmark.initialize(args)
dmbinit.benchmark_manual_sgd()
