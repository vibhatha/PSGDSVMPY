import sys
import os
import socket
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
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

def load_training_data(training_file, split=False):
    x_training = []
    y_training = []
    training_filepath = training_file

    if not split:
        training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=features)
        x_training, y_training = training_loader.load_all_data()

    print("Training data loaded ...")
    return x_training, y_training


def load_random_test_data(testing_filepath):
    print("Loading Bulk Testing Files")
    files = os.listdir(testing_filepath)
    print("File Path : " + testing_filepath)
    print(files)
    bulk_testing_x = []
    bulk_testing_y = []
    file_size = len(files)
    random_id = np.random.randint(0,file_size)
    print("Loading Testing Bulk File : " + files[random_id])
    testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath + "/" + files[random_id],
                                               n_features=1)
    x_testing, y_testing = testing_loader.load_all_data()
    return x_testing, y_testing



datasets = [ 'rcv1']
features = [47236]
splits = [False, False]

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
    testing_file = base_path + dataset + '/bulk'
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
    C=1
    exp_name = dataset
    repetitions = range(0,10)
    tolerance = 0.000001
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
                                                       alpha=alpha, features=features, epochs=epochs, labelfix=labelfix,
                                                       randomize=randomize, split=split, auto=True, bulk=bulk)

    bmsvmsgdadam = BenchmarkSGDAdam.BenchmarkSGDAdam(exp_name=exp_name, training_file=training_file,
                                                 testing_file=testing_file,
                                                 alpha=alpha, features=features, epochs=epochs, labelfix=labelfix,
                                                 randomize=randomize, split=split, auto=True, bulk=bulk)



    x_training, y_training = load_training_data(training_file=training_file, split=split)
    x_testing, y_testing = load_random_test_data(testing_filepath=testing_file)
    w_init = np.random.uniform(0, 1, features)
    m = len(x_training)
    indices_init = np.random.choice(m, m, replace=False)
    for repition in repetitions:
        now = datetime.datetime.now()
        prefix = str(now.date())+"_"+str(now.time())
        ########################################LOGS###########################################
        ## SGD Logs
        log_file_sgd= "logs/benchmark/epochlog/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(repition) + "_sgd_epoch_results.txt"
        init_weight_file_sgd = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(repition) + "_init_sgd_weights.txt"
        final_weight_file_sgd = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(repition) + "_final_sgd_weights.txt"
        ## Momentum Logs
        #######################################################################################
        log_file_mom = "logs/benchmark/epochlog/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_momentum_epoch_results.txt"
        init_weight_file_mom = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_init_momentum_weights.txt"
        final_weight_file_mom = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_final_momentum_weights.txt"
        ## Ada Logs
        #######################################################################################
        log_file_ada = "logs/benchmark/epochlog/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_ada_epoch_results.txt"
        init_weight_file_ada = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_init_ada_weights.txt"
        final_weight_file_ada = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_final_ada_weights.txt"

        ## RMSProp Logs
        #######################################################################################
        log_file_rmsprop = "logs/benchmark/epochlog/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_rmsprop_epoch_results.txt"
        init_weight_file_rmsprop = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_init_rmsprop_weights.txt"
        final_weight_file_rmsprop = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_final_rmsprop_weights.txt"

        ## Adam Logs
        #######################################################################################
        log_file_adam = "logs/benchmark/epochlog/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_adam_epoch_results.txt"
        init_weight_file_adam = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_init_adam_weights.txt"
        final_weight_file_adam = "weights/benchmark/" + socket.gethostname() + "_" + exp_name + "_" + prefix + "_" + str(
            repition) + "_final_adam_weights.txt"

        ####################################### Training #######################################

        ################ SGD #################
        w_final_sgd, epochs_eff_sgd, initial_cost_sgd, final_cost_sgd, overall_training_time_sgd = bmsvmsgd.train_benchmark_heavy(x_train=x_training, y_train=y_training, x_test=x_testing, y_test=y_testing, w_init=w_init, log_file=log_file_sgd, tolerance=tolerance, indices_init=indices_init)
        ####################################

        ############### Momentum #############
        w_final_mom, epochs_eff_mom, initial_cost_mom, final_cost_mom, overall_training_time_mom = bmsvmsgdmomentum.train_benchmark_heavy(
             x_train=x_training, y_train=y_training, x_test=x_testing, y_test=y_testing, w_init=w_init,
             log_file=log_file_mom, tolerance=tolerance, indices_init=indices_init)

        ############### ADA ###################
        w_final_ada, epochs_eff_ada, initial_cost_ada, final_cost_ada, overall_training_time_ada = bmsvmsgdada.train_benchmark_heavy(
             x_train=x_training, y_train=y_training, x_test=x_testing, y_test=y_testing, w_init=w_init,
              log_file=log_file_ada, tolerance=tolerance, indices_init=indices_init)

        ############### RMSPROP ###################
        w_final_rmsprop, epochs_eff_rmsprop, initial_cost_rmsprop, final_cost_rmsprop, overall_training_time_rmsprop = bmsvmsgdrmsprop.train_benchmark_heavy(
             x_train=x_training, y_train=y_training, x_test=x_testing, y_test=y_testing, w_init=w_init,
              log_file=log_file_rmsprop, tolerance=tolerance, indices_init=indices_init)

        w_final_adam, epochs_eff_adam, initial_cost_adam, final_cost_adam, overall_training_time_adam = bmsvmsgdadam.train_benchmark_heavy(
            x_train=x_training, y_train=y_training, x_test=x_testing, y_test=y_testing, w_init=w_init,
             log_file=log_file_adam, tolerance=tolerance, indices_init=indices_init)



        ## save weight vectors
        if repition == 0:
            np.savetxt(init_weight_file_sgd, w_init)
            np.savetxt(init_weight_file_mom, w_init)
            np.savetxt(init_weight_file_ada, w_init)
            np.savetxt(init_weight_file_rmsprop, w_init)
            np.savetxt(init_weight_file_adam, w_init)
        np.savetxt(final_weight_file_sgd, w_final_sgd)
        np.savetxt(final_weight_file_mom, w_final_mom)
        np.savetxt(final_weight_file_ada, w_final_ada)
        np.savetxt(final_weight_file_rmsprop, w_final_rmsprop)
        np.savetxt(final_weight_file_adam, w_final_adam)

        ## testing
        ## iterate through all the test samples.
        overall_accuracy_sgd = 0
        overall_accuracy_mom = 0
        overall_accuracy_ada = 0
        overall_accuracy_rmsprop = 0
        overall_accuracy_adam = 0
        files = os.listdir(testing_file)
        print("File Path : " + testing_file)
        print(files)
        bulk_testing_x = []
        bulk_testing_y = []
        file_size = len(files)
        random_id = np.random.randint(0, file_size)
        print("Loading Testing Bulk File : " + files[random_id])
        for file in files:
            testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_file + "/" + file,
                                               n_features=1)
            x_testing, y_testing = testing_loader.load_all_data()
            overall_accuracy_sgd += bmsvmsgd.advanced_test(x_testing=x_testing, y_testing=y_testing, w=w_final_sgd)
            overall_accuracy_mom += bmsvmsgdmomentum.advanced_test(x_testing=x_testing, y_testing=y_testing, w=w_final_mom)
            overall_accuracy_ada += bmsvmsgdada.advanced_test(x_testing=x_testing, y_testing=y_testing, w=w_final_ada)
            overall_accuracy_rmsprop += bmsvmsgdrmsprop.advanced_test(x_testing=x_testing, y_testing=y_testing, w=w_final_rmsprop)
            overall_accuracy_adam += bmsvmsgdadam.advanced_test(x_testing=x_testing, y_testing=y_testing,
                                                                 w=w_final_adam)

        overall_accuracy_sgd = overall_accuracy_sgd/ file_size
        overall_accuracy_mom = overall_accuracy_mom / file_size
        overall_accuracy_ada = overall_accuracy_ada / file_size
        overall_accuracy_rmsprop = overall_accuracy_rmsprop / file_size
        overall_accuracy_adam = overall_accuracy_adam / file_size

        ## stats
        bmsvmsgd.advanced_stats(effective_epochs=epochs_eff_sgd, accuracy=overall_accuracy_sgd, training_time=overall_training_time_sgd, initial_cost=initial_cost_sgd, final_cost=final_cost_sgd)
        bmsvmsgdmomentum.advanced_stats(effective_epochs=epochs_eff_mom, accuracy=overall_accuracy_mom,
                                 training_time=overall_training_time_mom, initial_cost=initial_cost_mom,
                                 final_cost=final_cost_mom)
        bmsvmsgdada.advanced_stats(effective_epochs=epochs_eff_ada, accuracy=overall_accuracy_ada,
                                  training_time=overall_training_time_ada, initial_cost=initial_cost_ada,
                                  final_cost=final_cost_ada)

        bmsvmsgdrmsprop.advanced_stats(effective_epochs=epochs_eff_rmsprop, accuracy=overall_accuracy_rmsprop,
                                   training_time=overall_training_time_rmsprop, initial_cost=initial_cost_rmsprop,
                                   final_cost=final_cost_rmsprop)
        bmsvmsgdadam.advanced_stats(effective_epochs=epochs_eff_adam, accuracy=overall_accuracy_adam,
                                       training_time=overall_training_time_adam, initial_cost=initial_cost_adam,
                                       final_cost=final_cost_adam)





