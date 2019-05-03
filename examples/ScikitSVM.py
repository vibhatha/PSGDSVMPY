import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from operations import LoadLibsvm
import socket

from sklearn import svm, datasets
import numpy as np


def load_data(training_file, testing_file, split=False, n_features=10):
    training_filepath = training_file
    testing_filepath = testing_file
    if not split:
        training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=n_features)
        testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath, n_features=n_features)
        x_training, y_training = training_loader.load_all_data()
        x_testing, y_testing = testing_loader.load_all_data()

    if split:
        print("Data Splitting ...")
        training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=n_features)
        x_all, y_all = training_loader.load_all_data()
        ratio = 0.6
        size = len(x_all)
        split_index = int(size * ratio)
        x_training = x_all[:split_index]
        x_testing = x_all[split_index:]
        y_training = y_all[:split_index]
        y_testing = y_all[split_index:]


    return x_training, y_training, x_testing, y_testing


base_path = ''
hostname = socket.gethostname()

if hostname == 'vibhatha-ThinkPad-P50':
	base_path = '/home/vibhatha/data/svm/'
else:
	base_path = '/N/u/vlabeyko/data/svm/svm/'

dataset = 'heart'
training_file = base_path + dataset + '/training.csv'
testing_file = base_path + dataset + '/testing.csv'
n_features = 13
split=False

X_train, y_train, X_test, y_test = load_data(training_file=training_file, testing_file=testing_file, split=split, n_features=n_features)

clf = svm.SVC()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

acc_count = 0
for i, y in enumerate(preds):
    if (y_test[i] == (preds[i])):
        acc_count += 1

print("LibSVM Accuracy : " + str((float(acc_count) / float(len(preds)) * 100)))
# 75.3521126761 default sgd output from SvmIrisSGD.py

from sklearn import linear_model
clf = linear_model.SGDClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)


acc_count = 0
for i, y in enumerate(preds):
    if (y_test[i] == (preds[i])):
        acc_count += 1

print(" SGD Accuracy : " + str((float(acc_count) / float(len(preds)) * 100)))
