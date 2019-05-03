import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from sgd import SVM
from operations import LoadLibsvm


# def case1():
#     iris = datasets.load_breast_cancer()
#     X = iris.data
#     y = iris.target
#
#     X = X[:-1]
#     y = y[:-1]
#
#     y[y == 0] = -1
#
#     svm = SVM.SVM(X=X, y=y, alpha=1, n_features=30)
#
#     X_train, y_train, X_test, y_test = svm.split_data(X=X, y=y, percentage=60)
#
#     svm.train(X=X_train, y=y_train, alpha=0.01, epochs=1200)
#
#     y_pred = svm.predict(X=X_test)
#
#     acc = svm.get_accuracy(y_test=y_test, y_pred=y_pred)
#
#     print("Accuracy : " + str(acc) + "%")


def case2():
    training_filepath = "/home/vibhatha/data/svm/ijcnn1/ijcnn1.tr"
    testing_filepath = "/home/vibhatha/data/svm/ijcnn1/ijcnn1.t"
    svm = SVM.SVM(trainPath=training_filepath, testPath=testing_filepath)
    #X_train, y_train, X_test, y_test = svm.load_libsvmdata(n_features=22)
    #print(str(y_train[0]))
    #print(str(X_train[0]))
    #svm.train(X=X_train, y=y_train, alpha=0.01, epochs=1200)

    #y_pred = svm.predict(X=X_test)

    #acc = svm.get_accuracy(y_test=y_test, y_pred=y_pred)

    #print("Accuracy : " + str(acc) + "%")


def case3():
    training_filepath = "/home/vibhatha/data/svm/ijcnn1/ijcnn1_training"
    testing_filepath = "/home/vibhatha/data/svm/ijcnn1/ijcnn1_testing"
    loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=22)
    x_training, y_training, x_testing, y_testing = loader.load_data()
    svm = SVM.SVM(X=x_training, y=y_training, alpha=1, n_features=22)
    # alpha=0.0003, epochs=1000 #good
    alpha = 0.1
    epochs = 500
    svm.train(X=x_training, y=y_training, alpha=alpha, epochs=epochs)

    y_pred = svm.predict(X=x_testing)

    acc = svm.get_accuracy(y_test=y_testing, y_pred=y_pred)

    print("Accuracy : " + str(acc) + "%")

    fp = open("logs/results.txt", "a")
    fp.write("alpha : " + str(alpha) + ", epochs : " + str(epochs) + ", accuracy : " + str(acc) + "%\n")
    fp.close()


case3()
