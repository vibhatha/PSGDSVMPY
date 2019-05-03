import sys
import os
import socket

HOME = os.environ['HOME']
sys.path.insert(1, HOME + '/github/StreamingSVM')
from experiment import BenchmarkSGDAdam
from experiment import BenchmarkPegasosSvm
from experiment import BenchmarkSGDAda
from experiment import BenchmarkSGDRMSProp
from experiment import BenchmarkSGDMomentum
from experiment import BenchmarkSgdSvm
from operations import Print
from operations import LoadLibsvm
import numpy as np
import datetime


### Non Bulk Data

### loading data

def load_light_data(training_file, testing_file, split=False):
    x_training = []
    x_testing = []
    y_training = []
    y_testing = []

    training_filepath = training_file
    testing_filepath = testing_file
    if not split:
        training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=features)
        testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath, n_features=features)
        x_training, y_training = training_loader.load_all_data()
        x_testing, y_testing = testing_loader.load_all_data()

    if split:
        training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=features)
        x_all, y_all = training_loader.load_all_data()
        ratio = 0.8
        size = len(x_all)
        split_index = int(size * ratio)
        x_training = x_all[:split_index]
        x_testing = x_all[split_index:]
        y_training = y_all[:split_index]
        y_testing = y_all[split_index:]
    print("Training and Testing data loaded ...")
    return x_training, y_training, x_testing, y_testing


datasets = ['ijcnn1', 'heart', 'webspam', 'cod-rna', 'phishing', 'breast_cancer', 'w8a', 'a9a', 'real-slim']
features = [22, 13, 254, 8, 68, 10, 300, 123, 20958]
splits = [False, False, True, False, True, True, False, False, True]

# datasets =['cod-rna', 'phishing', 'breast_cancer', 'w8a', 'a9a']
# features = [8, 68, 10, 300 , 123]
# splits = [False, True, True, False, False]
datasets = ['breast_cancer']
features = [10]
splits = [True]

for dataset, feature, split in zip(datasets, features, splits):

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
    features = feature
    alpha = 1
    epochs = 200
    ld = 1
    eta = 0.1
    labelfix = False
    split = split
    randomize = True
    gamma = 1
    degree = 1
    kernel = 'rbf'
    minibatch_size = 1000
    minibatch = True
    C = 1
    exp_name = dataset
    repetitions = range(0, 1)
    tolerance = 0.000001


    bmsvmsgdada = BenchmarkSGDAda.BenchmarkSGDAda(exp_name=exp_name, training_file=training_file,
                                                  testing_file=testing_file,
                                                  alpha=alpha, features=features, epochs=epochs, labelfix=labelfix,
                                                  randomize=randomize, split=split, auto=True, bulk=bulk)



    x_training, y_training, x_testing, y_testing = load_light_data(training_file=training_file,
                                                                   testing_file=testing_file, split=split)
    w_init = np.random.uniform(0, 1, features)
    m = len(x_training)
    indices_init = np.random.choice(m, m, replace=False)
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.000001, 0.2, 0.02, 0.002, 0.0002, 0.00002, 0.000002,1]

    for repition in repetitions:
        for alpha in alphas:
            now = datetime.datetime.now()
            prefix = str(now.date()) + "_" + str(now.time())
            ########################################LOGS###########################################

            ## Ada Logs
            #######################################################################################
            log_file_ada = "logs/benchmark/epochlog/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
                repition) + "_ada_epoch_results.txt"
            init_weight_file_ada = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
                repition) + "_init_ada_weights.txt"
            final_weight_file_ada = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
                repition) + "_final_ada_weights.txt"

            ####################################### Training #######################################

            ############### ADA ###################
            w_final_ada, epochs_eff_ada, initial_cost_ada, final_cost_ada, overall_training_time_ada = bmsvmsgdada.train_benchmark_light(
                x_train=x_training, y_train=y_training, x_test=x_testing, y_test=y_testing, w_init=w_init,
                log_file=log_file_ada, tolerance=tolerance, indices_init=indices_init)


            ## save weight vectors
            if repition == 0:
                np.savetxt(init_weight_file_ada, w_init)

            np.savetxt(final_weight_file_ada, w_final_ada)


            ## testing
            overall_accuracy_ada = bmsvmsgdada.advanced_test(x_testing=x_testing, y_testing=y_testing,
                                                             w=w_final_ada)

            ## stats
            bmsvmsgdada.advanced_stats(effective_epochs=epochs_eff_ada, accuracy=overall_accuracy_ada,
                                       training_time=overall_training_time_ada, initial_cost=initial_cost_ada,
                                       final_cost=final_cost_ada)

