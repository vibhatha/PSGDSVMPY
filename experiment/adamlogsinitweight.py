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


# datasets = [ 'ijcnn1', 'heart', 'webspam', 'cod-rna', 'phishing', 'breast_cancer', 'w8a', 'a9a', 'real-slim']
# features = [22, 13, 254, 8, 68, 10, 300 , 123, 20958]
# splits = [False, False, True, False, True, True, False, False, True]
datasets = ['cod-rna']
features = [8]
splits = [False]
weight_base_path = '/home/vibhatha/Documents/Research/SVM-SGD/Streaming/weights/streaming/'
weight_file = 'j-100_cod-rna_2018-10-19_16:43:06.182486_8_sgd_weight_final.txt'
# heart : j-100_heart_2018-10-19_16:40:27.315507_9_sgd_weight_final.txt
# ijcnn1 : j-100_ijcnn1_2018-10-19_16:40:26.458297_17_sgd_weight_final.txt
# phishing : j-100_phishing_2018-10-19_16:43:15.924555_15_sgd_weight_final.txt
# w8a : j-100_w8a_2018-10-19_16:43:34.748323_6_sgd_weight_final.txt
# webspam : j-100_webspam_2018-10-19_16:41:55.758648_1_sgd_weight_final.txt
# real-slim : j-100_real-slim_2018-10-19_17:07:18.540189_1_sgd_weight_final.txt
weight_file_path = weight_base_path + weight_file
# datasets =['cod-rna', 'phishing', 'breast_cancer', 'w8a', 'a9a']
# features = [8, 68, 10, 300 , 123]
# splits = [False, True, True, False, False]
# datasets = ['webspam']
# features = [254]
# splits = [True]

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
    repetitions = range(0, 10)
    tolerance = 0.01
    bmsvmsgd = BenchmarkSgdSvm.BenchmarkSgdSVM(exp_name=exp_name, training_file=training_file,
                                               testing_file=testing_file,
                                               alpha=alpha, features=features, epochs=epochs,
                                               labelfix=labelfix, randomize=randomize, split=split,
                                               auto=True)
    bmsvmsgdmomentum = BenchmarkSGDMomentum.BenchmarkSGDMomentum(exp_name=exp_name,
                                                                 training_file=training_file,
                                                                 testing_file=testing_file,
                                                                 alpha=alpha, C=C, gamma=gamma,
                                                                 features=features, epochs=epochs,
                                                                 labelfix=labelfix,
                                                                 randomize=randomize, split=split)

    bmsvmsgdada = BenchmarkSGDAda.BenchmarkSGDAda(exp_name=exp_name, training_file=training_file,
                                                  testing_file=testing_file,
                                                  alpha=alpha, features=features, epochs=epochs, labelfix=labelfix,
                                                  randomize=randomize, split=split, auto=True, bulk=bulk)

    bmsvmsgdrmsprop = BenchmarkSGDRMSProp.BenchmarkSGDRMSProp(exp_name=exp_name, training_file=training_file,
                                                              testing_file=testing_file,
                                                              alpha=alpha, features=features, epochs=epochs,
                                                              labelfix=labelfix,
                                                              randomize=randomize, split=split, auto=True, bulk=bulk)

    bmsvmsgdadam = BenchmarkSGDAdam.BenchmarkSGDAdam(exp_name=exp_name, training_file=training_file,
                                                     testing_file=testing_file,
                                                     alpha=alpha, features=features, epochs=epochs, labelfix=labelfix,
                                                     randomize=randomize, split=split, auto=True, bulk=bulk)

    x_training, y_training, x_testing, y_testing = load_light_data(training_file=training_file,
                                                                   testing_file=testing_file, split=split)
    #w_init = np.random.uniform(0, 1, features)
    w_init = np.loadtxt(weight_file_path)
    m = len(x_training)
    indices_init = np.random.choice(m, m, replace=False)
    gamma_momentum = []
    b_adam = []
    b_rmsprop = []
    beta1_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.93, 0.95, 0.99, 0.999]
    beta2_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.93, 0.95, 0.99, 0.999]

    for b1 in beta1_range:
        for b2 in beta2_range:
            for repition in repetitions:
                now = datetime.datetime.now()
                prefix = str(now.date()) + "_" + str(now.time())
                ########################################LOGS###########################################

                ## Adam Logs
                #######################################################################################
                log_file_adam = "logs/benchmark/epochlog/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
                    repition) + "_adam_epoch_results.txt"
                init_weight_file_adam = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
                    repition) + "_init_adam_weights.txt"
                final_weight_file_adam = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
                    repition) + "_final_adam_weights.txt"

                ####################################### Training #######################################


                w_final_adam, epochs_eff_adam, initial_cost_adam, final_cost_adam, overall_training_time_adam = bmsvmsgdadam.train_benchmark_light(
                    x_train=x_training, y_train=y_training, x_test=x_testing, y_test=y_testing, w_init=w_init,
                    log_file=log_file_adam, tolerance=tolerance, indices_init=indices_init, beta1=b1, beta2=b2)

                ## save weight vectors
                if repition == 0:
                    np.savetxt(init_weight_file_adam, w_init)
                np.savetxt(final_weight_file_adam, w_final_adam)

                ## testing
                overall_accuracy_adam = bmsvmsgdadam.advanced_test(x_testing=x_testing, y_testing=y_testing,
                                                                   w=w_final_adam)

                ## stats
                bmsvmsgdadam.advanced_stats(effective_epochs=epochs_eff_adam, accuracy=overall_accuracy_adam,
                                            training_time=overall_training_time_adam, initial_cost=initial_cost_adam,
                                            final_cost=final_cost_adam)
