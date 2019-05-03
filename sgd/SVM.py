import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
import numpy as np
from operations import LoadLibsvm
from operations import Print
from kernel import Kernel
import scipy as sc
from scipy.spatial.distance import pdist, squareform
import time

class SVM:

    def __init__(self, trainPath=None, testPath=None, X=None, y=None, alpha=0.01, C=1, gamma=1, degree=2, n_features=22, eta=0.1, ld=0.01, epochs=100, randomize=False, bulk=False, split=False):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.trainPath = trainPath
        self.testPath = testPath
        self.n_features = n_features
        self.eta = eta
        self.ld = ld
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.epochs = epochs
        self.randomize = randomize
        self.bulk = bulk
        self.split = split

    def init_weights(self):
        self.w = np.zeros(self.n_features)
        self.w = np.random.uniform(0, 1, self.n_features)
        return self.w

    def init_indices(self, X):
        m = len(X)
        indices = np.random.choice(m, m, replace=False)
        return indices

    def split_data(self, X, y, percentage=60):
        size = len(y)
        percentage = int(size * percentage / 100)
        X_train, X_test = X[:percentage,:], X[percentage:,:]
        y_train, y_test = y[:percentage], y[percentage:]
        return X_train, y_train, X_test, y_test

    def train(self, X, y, alpha=0.01, epochs=1000, w_init=[]):
        self.w = np.zeros(self.n_features)
        self.w = np.random.uniform(0,1, self.n_features)
        for epoch in range(1, epochs):
            alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                    if (condition < 1):
                        self.w = self.w + alpha * ((X[i] * y[i]) + (-2 * (1 / epoch) * self.w))
                    else:
                        self.w = self.w + alpha * (-2 * (1 / epoch) * self.w)

            if(self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        self.w = self.w + alpha * ((X[index] * y[index]) + (-2 * (1 / epoch) * self.w))
                    else:
                        self.w = self.w + alpha * (-2 * (1 / epoch) * self.w)
            cost = 0.5 * np.dot(self.w, self.w.T) + self.C * condition
            if (epoch % 1 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))
        return self.w

    def train_sgd(self, X, y, alpha=0.01, epochs=1000, w_init=[]):
        self.w = np.zeros(self.n_features)
        self.w = np.random.uniform(0,1, self.n_features)
        for epoch in range(1, epochs):
            alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                    if (condition < 1):
                        self.w = self.w - alpha * (-1*(X[i] * y[i]) + (self.w))
                    else:
                        self.w = self.w - alpha * (self.w)

            if(self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        self.w = self.w - alpha * (-1 * (X[index] * y[index]) + (self.w))
                    else:
                        self.w = self.w - alpha * (self.w)
            cost = 0.5 * np.dot(self.w, self.w.T) + self.C * condition
            if (epoch % 1 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))
        return self.w

    def train_sgd_online(self, X, y, alpha=0.01, epochs=1000, w_init=[]):
        self.w = np.zeros(self.n_features)
        self.w = np.random.uniform(0,1, self.n_features)
        for epoch in range(1, 2):
            alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                    if (condition < 1):
                        self.w = self.w - alpha * (-1*(X[i] * y[i]) + (self.w))
                    else:
                        self.w = self.w - alpha * (self.w)

            if(self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        self.w = self.w - alpha * (-1 * (X[index] * y[index]) + (self.w))
                    else:
                        self.w = self.w - alpha * (self.w)
            cost = 0.5 * np.dot(self.w, self.w.T) + self.C * condition
            if (epoch % 1 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))
        return self.w

    def train_sgd_online_init_weight(self, X, y, alpha=0.01, epochs=1000, w_init=[]):
        self.w = np.zeros(self.n_features)
        self.w = w_init
        for epoch in range(1, 2):
            alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                    if (condition < 1):
                        self.w = self.w - alpha * (-1*(X[i] * y[i]) + (self.w))
                    else:
                        self.w = self.w - alpha * (self.w)

            if(self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        self.w = self.w - alpha * (-1 * (X[index] * y[index]) + (self.w))
                    else:
                        self.w = self.w - alpha * (self.w)
            cost = 0.5 * np.dot(self.w, self.w.T) + self.C * condition
            if (epoch % 1 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))
        return self.w

    def train_with_simple_kernel(self, X, y, alpha=0.01, kernel='linear', epochs=10000):
        m = len(X)
        B = np.zeros((epochs,m),dtype=float)
        A = np.zeros((epochs, m), dtype=float)
        indices = np.random.choice(m, m, replace=False)
        for epoch in range(0,epochs-1):
            alpha = 1.0 / (1.0 + float(epoch))  # alpha update rule
            A[epoch] = (1.0 / alpha * epoch) * B[epoch]
            for i in indices:
                for j in indices:
                    if (j!=i):
                        B[epoch+1][j] = B[epoch][j]
                        kernel_value = 0
                        if kernel=='rbf':
                            kernel_value = 0.5 * (X[i]- X[j]).T * (X[i]- X[j]) * (1.0/self.gamma**2)
                            kernel_value = np.exp(-1.0 * kernel_value)
                        condition = np.sum(A[epoch][j] * kernel_value, axis=0)
                        if (condition < 1):
                            B[epoch+1][i] = B[epoch][i] + y[i]
                        else:
                            B[epoch+1][i] = B[epoch][i]
            if (epoch % 1 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs))
        A_bar = (1.0/float(epochs)) * np.sum(A, axis=0)
        A_bar = np.reshape(A_bar, (1, len(A_bar)))
        if kernel == 'rbf':
            kv = 0.5 * X * X * (1.0 / self.gamma ** 2)
            kv = np.exp(-1.0 * kv)
            a = A_bar.T * kv
            self.w = np.sum(a, axis=0)

    def train_with_kernel(self, Xorg, X, y, alpha=0.01, epochs=1000, kernel='linear'):
        m = len(X) # number of samples
        a = np.zeros((1,m), dtype=float)
        indices = np.random.choice(m, m, replace=False)
        t = 1
        for epoch in range(1, epochs-1):
            if (epoch % 10 == 0):
                Print.Print.overwrite("Epoch " + str(epoch) + "/" + str(epochs))
            for index in indices:
                yi = y[index]
                Xi = X[index]
                y1 = yi * a[0]
                Xi = np.reshape(Xi, (1,len(Xi)))
                aXi = np.dot(y1, Xi.T)
                print(aXi.T)
                condition = (1.0/(alpha*epoch)) * aXi
                a[0] = a[0] * (1.0 - (1.0/float(t)))
                if (condition < 1):
                    a[0][index] = a[0][index] + (float(yi) / float(alpha * t))
                #else:
                #    a[index] = a[index]
                t = t + 1

        a_nonzero_indices = np.where(a[0]!=0)
        aT = [a[0][index] for index in a_nonzero_indices]

        SV = [Xorg[index] for index in a_nonzero_indices]
        self.a = np.array(aT)[0].reshape(len(aT[0]),1)
        self.SV = np.array(SV)[0]

    def custom_minibatch_kernel_sgd_train(self, X, y, alpha=0.01, kernel='linear', gamma=2, degree=2, epochs=100,
                                          batch_size=10000):
        m = len(X)  # number of samples
        a = np.zeros((1, m), dtype=float)
        for epoch in range(1, epochs):
            alpha = 1.0 / (1.0 + float(epoch))  # alpha update rule
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                    if (condition < 1):
                        self.w = self.w + alpha * ((X[i] * y[i]) + (-2 * (1 / epoch) * self.w))
                    else:
                        self.w = self.w + alpha * (-2 * (1 / epoch) * self.w)

            if (self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                num_minibatches = num_range / batch_size
                minibatch_indices = np.array_split(indices, num_minibatches)
                t = 1
                for minibatch_inds in minibatch_indices:
                    X_sub = np.take(X, minibatch_inds, axis=0)
                    y_sub = np.take(y, minibatch_inds, axis=0)
                    X_sub_k = Kernel.Kernel.generate_kerenelized_feature_matrix(X=X_sub, kernel=kernel, degree=degree,
                                                                                gamma=gamma)
                    a_sub = np.take(a[0], minibatch_inds, axis=0)
                    for i,x in enumerate(X_sub_k):
                        yi = y_sub[i]
                        Xi = X_sub_k[i]
                        y1 = yi * a_sub
                        Xi = np.reshape(Xi, (1, len(Xi)))
                        aXi = np.dot(y1, Xi.T)
                        condition = (1.0 / (alpha * epoch)) * aXi
                        a_sub = a_sub * (1.0 - (1.0 / float(t)))
                        if (condition < 1):
                            a_sub[i] = a_sub[i] + (float(yi) / float(alpha * t))
                        # else:
                        #    a[index] = a[index]
                        t = t + 1
                    sub_itr = 0
                    for index in minibatch_inds:
                        a[0][index] = a_sub[sub_itr]
                        sub_itr = sub_itr + 1

            if (epoch % 1 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs))

        a_nonzero_indices = np.where(a[0] != 0)
        aT = [a[0][index] for index in a_nonzero_indices]

        SV = [X[index] for index in a_nonzero_indices]
        self.a = np.array(aT)[0].reshape(len(aT[0]), 1)
        self.SV = np.array(SV)[0]

    def auto_sgd_train(self, X, y, epochs=1000):
        self.w = np.zeros(self.n_features)
        self.w = np.random.uniform(0,1, self.n_features)
        for epoch in range(1, epochs):
            alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                    if (condition < 1):
                        self.w = self.w + alpha * ((X[i] * y[i]) + (-2 * (1 / epoch) * self.w))
                    else:
                        self.w = self.w + alpha * (-2 * (1 / epoch) * self.w)

            if(self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        self.w = self.w + alpha * ((X[index] * y[index]) + (-2 * (1 / epoch) * self.w))
                    else:
                        self.w = self.w + alpha * (-2 * (1 / epoch) * self.w)

            cost = 0.5 * np.dot(self.w, self.w.T) + self.C * condition
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))
        return self.w




    def custom_minibatch_sgd_train(self,X, y, alpha=0.01, epochs=100, batch_size=10000):
        self.w = np.random.uniform(0, 1, self.n_features)
        for epoch in range(1, epochs):
            alpha = 1.0 / (1.0 + float(epoch))  # alpha update rule
            if (self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                num_minibatches = num_range / batch_size
                minibatch_indices = np.array_split(indices, num_minibatches)
                for minibatch_inds in minibatch_indices:
                    for index in minibatch_inds:
                        condition = y[index] * np.dot(X[index], self.w)
                        if (condition < 1):
                            self.w = self.w + alpha * ((X[index] * y[index]) + (-2 * (1 / epoch) * self.w))
                        else:
                            self.w = self.w + alpha * (-2 * (1 / epoch) * self.w)

            cost = abs(0.5 * np.dot(self.w, self.w.T) + self.C * condition)
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))


    # this algorithm is wrong, because we need to select the xy<1 data samples out of the minibatch :D
    def minibatch_sgd_train_version1(self,X, y, alpha=0.01, epochs=100, batch_size=10000):
        self.w = np.zeros((X.shape[1],1))
        print(self.w.shape)
        v = np.zeros(self.w.shape)
        beta1 = 0.5
        beta2 = 0.5
        epsilon = 0.00000001
        r = np.zeros(self.w.shape)
        num_range = len(X)
        num_minibatches = int(num_range / batch_size)
        real_total_data = num_minibatches * batch_size
        indices = np.random.choice(real_total_data, real_total_data, replace=False)

        minibatch_indices = np.array_split(indices, num_minibatches)
        print(minibatch_indices[0].shape, len(minibatch_indices))
        for epoch in range(1, epochs):
            alpha = 1.0 / (1.0 + float(epoch))  # alpha update rule
            if (self.randomize == True):
                for minibatch_inds in minibatch_indices:
                    Xb = np.take(X, minibatch_inds,axis=0)
                    yb = np.take(y, minibatch_inds,axis=0)
                    Xw = Xb.dot(self.w)
                    #print(minibatch_inds.shape, Xb.shape, yb.shape, self.w.shape, Xw.shape, yb.shape)
                    condition1 = (Xw.T * yb)/float(batch_size)
                    condition = np.sum(condition1)
                    # print("<1",condition)
                    Xy = np.matmul(Xb.T, yb)
                    Xy = np.reshape(Xy, (X.shape[1], 1))
                    Xy_sum = np.sum(Xy,axis=1)/batch_size
                    print(Xy_sum)
                    gradient = Xy_sum
                    gradient = np.reshape(gradient, (gradient.shape[0],1))

                    v = beta1 * v + (1 - beta1) * gradient
                    v_hat = v / (1 - beta1 ** epoch)
                    r = beta2 * r + (1 - beta2) * (np.multiply(gradient, gradient))
                    r_hat = r / (1 - beta2 ** epoch)
                    self.w = self.w - alpha * np.multiply((v_hat), 1.0 / (np.sqrt(r_hat) + epsilon))
                    #print(Xy_sum.shape, self.w.shape)


                    #self.w = self.w + alpha * (Xy_sum + (-2 * (1 / epoch) * self.w))

                    cost = abs(0.5 * np.dot(self.w.T, self.w) + self.C * condition)/batch_size


            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))


    def custom_minibatch_sgd_train_without_coeff(self,X, y, alpha=0.01, epochs=100, batch_size=10000):
        self.w = np.random.uniform(0, 1, self.n_features)
        for epoch in range(1, epochs):
            alpha = 1.0 / (1.0 + float(epoch))  # alpha update rule
            if (self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                num_minibatches = num_range / batch_size
                minibatch_indices = np.array_split(indices, num_minibatches)
                for minibatch_inds in minibatch_indices:
                    for index in minibatch_inds:
                        condition = y[index] * np.dot(X[index], self.w)
                        if (condition < 1):
                            self.w = self.w + alpha * ((X[index] * y[index]) + (self.w))
                        else:
                            self.w = self.w + alpha * (self.w*1)

            cost = abs(0.5 * np.dot(self.w, self.w.T) + self.C * condition)
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))



    def minibatch_sgd_train(self, X, y, alpha=0.01, epochs=1000, batch_size=10000):
        self.w = np.zeros(self.n_features)
        self.w = np.random.uniform(0,1, self.n_features)
        condition = 1
        for epoch in range(1, epochs):
            alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            eta = 1.0 / (alpha * epoch)
            ## optimize the code reusability by taking the algo out of indices generation

            if (self.randomize == False):
                num_range = len(X)
                indices = np.arange(0,num_range,1)
                num_minibatches = num_range / batch_size
                minibatch_indices = np.array_split(indices, num_minibatches)
                for minibatch_inds in minibatch_indices:
                    for index in minibatch_inds:
                        condition = y[index] * np.dot(X[index], self.w)
                        sum_Xy = []
                        if (condition < 1):
                            k1 = X[index] * y[index]
                            sum_Xy.append(k1)

                        w1 = (eta / batch_size) * np.sum(np.array(sum_Xy))
                        w2 = (1 - (eta * alpha)) * self.w
                        self.w = w1 + w2
            cost = 0.5 * np.dot(self.w, self.w.T) + self.C * condition
            if (epoch % 1 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))

            if(self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                num_minibatches = num_range / batch_size
                minibatch_indices = np.array_split(indices, num_minibatches)
                for minibatch_inds in minibatch_indices:
                    for index in minibatch_inds:
                        condition = y[index] * np.dot(X[index], self.w)
                        sum_Xy = []
                        if (condition < 1):
                            k1 = X[index] * y[index]
                            sum_Xy.append(k1)

                        w1 = (eta/batch_size) * np.sum(np.array(sum_Xy))
                        w2 = (1 - (eta * alpha)) * self.w
                        self.w = w1 + w2
            cost = 0.5 * np.dot(self.w, self.w.T) + self.C * condition
            if (epoch % 1 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))

    def manual_sgd_train(self, X, y, alpha=0.01, epochs=1000):
        self.w = np.zeros(self.n_features)
        self.w = np.random.uniform(0,1, self.n_features)
        for epoch in range(1, epochs):
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                    if (condition < 1):
                        self.w = self.w + alpha * ((X[i] * y[i]) + (-2 * (1 / epoch) * self.w))
                    else:
                        self.w = self.w + alpha * (-2 * (1 / epoch) * self.w)

            if(self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        self.w = self.w + alpha * ((X[index] * y[index]) + (-2 * (1 / epoch) * self.w))
                    else:
                        self.w = self.w + alpha * (-2 * (1 / epoch) * self.w)
            cost = 0.5 * np.dot(self.w, self.w.T) + self.C * condition
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                                    + ", Cost : " + str(cost))

        return self.w

    def train_momentum(self, X, y, alpha=0.01, epochs=1000):
        #self.w = np.zeros(self.n_features)
        #self.w = np.random.uniform(0,1, self.n_features)
        gamma = 0.9
        C = self.C
        # momentum portion
        v = np.array(np.zeros(self.w.shape))
        for epoch in range(1, epochs):
            alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            coefficient = 1
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                     #(-2 * (1 / epoch))
                    if (condition < 1):
                        # self.w = self.w + alpha * ((X[i] * y[i]) + (-2 * (1 / epoch) * self.w))
                        v = gamma * v + (1-gamma) * ((X[i] * y[i]) + ( -2 * (1 / epoch) * self.w))
                        self.w = self.w - alpha * v
                    else:
                        v = gamma * v + (1-gamma) * (coefficient * -2 * (1 / epoch) * self.w)
                        self.w = self.w - alpha * v
                cost = 0.5 * np.dot(self.w, self.w.T) + C * condition
                if (epoch % 10 == 0):
                    Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                        + ", Cost : " + str(cost))


            if(self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        v = gamma * v - alpha * (coefficient * self.w - C * X[index] * y[index])
                        self.w = self.w + v
                    else:
                        v = gamma * v - alpha * (coefficient * self.w)
                        self.w = self.w + v
                cost = 0.5 * np.dot(self.w, self.w.T) + C * condition
                if (epoch % 10 == 0):
                    Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                        + ", Cost : " + str(cost))
        return self.w


    def auto_sgd_train_cost(self, X, y, epochs=1000, w_init=[]):
        self.w = w_init
        cost = 10000000
        epoch = 1
        condition = 1
        while(cost > 0.01 and epoch < epochs):
            alpha = 1.0 / (1.0 + float(epoch))  # alpha update rule
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                    if (condition < 1):
                        self.w = self.w - alpha * (self.w - (X[i] * y[i]))
                    else:
                        self.w = self.w - alpha * self.w

            if (self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        self.w = self.w - alpha * ( self.w - self.C * (X[index] * y[index]))
                    else:
                        self.w = self.w - alpha * self.w

            cost = abs(0.5 * np.dot(self.w, self.w.T) + self.C * condition)
            epoch = epoch + 1
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))

        return self.w


    def auto_sgd_train_cost1(self, X, y, epochs=1000, w_init=[]):
        #self.w = np.zeros(self.n_features)
        self.w = w_init
        cost = 10000000
        epoch = 1
        condition = 1
        coefficient = 1#(2 * (1 / epoch))
        while(cost > 0.01 and epoch < epochs):
            alpha = 1.0 / (1.0 + float(epoch))  # alpha update rule
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                    if (condition < 1):
                        self.w = self.w - alpha * (self.w - (X[i] * y[i]))
                    else:
                        self.w = self.w - alpha * self.w

            if (self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        self.w = self.w - alpha * (coefficient * self.w - self.C * (X[index] * y[index]))
                    else:
                        self.w = self.w - alpha *  coefficient *self.w

            cost = abs(0.5 * np.dot(self.w, self.w.T) + self.C * condition)
            epoch = epoch + 1
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))

        return self.w



    def auto_sgd_train_cost2(self, X, y, epochs=1000, w_init=[], indices_init=[]):
        #self.w = np.zeros(self.n_features)
        w = w_init
        indices = indices_init
        cost = 10000000
        epoch = 1
        condition = 1
        coefficient = (2 * (1 / epoch))
        print("Initial Cost : " + str(abs(0.5 * np.dot(w, w.T) + y[indices[0]] * np.dot(X[indices[0]], w))))
        while(cost > 0.01 and epoch < epochs):
            alpha = 1.0 / (1.0 + float(epoch))  # alpha update rule
            for index in indices:
                condition = y[index] * np.dot(X[index], w)
                if (condition < 1):
                    w = w - alpha * (coefficient * w - self.C * (X[index] * y[index]))
                else:
                    w = w - alpha *  coefficient * w
            epoch = epoch + 1
            cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))
        self.w = w

        return self.w


    def train_momentum_cost(self, X, y, alpha=0.01, epochs=1000, indices_init=[], w_init=[]):
        #self.w = np.zeros(self.n_features)
        self.w = np.random.uniform(0,1, self.n_features)
        gamma = 0.99
        C = self.C
        # momentum portion
        v = np.array(np.zeros(self.w.shape))
        epoch = 1
        cost = 100000
        while (cost > 0.01 and epoch < epochs):
            alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            coefficient = 1
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                     #(-2 * (1 / epoch))
                    if (condition < 1):
                        # self.w = self.w + alpha * ((X[i] * y[i]) + (-2 * (1 / epoch) * self.w))
                        v = gamma * v + (1-gamma) * ((X[i] * y[i]) + ( -2 * (1 / epoch) * self.w))
                        self.w = self.w - alpha * v
                    else:
                        v = gamma * v + (1-gamma) * (coefficient * -2 * (1 / epoch) * self.w)
                        self.w = self.w - alpha * v




            if(self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        v = gamma * v + (1-gamma) * (coefficient * self.w - (self.C * X[index] * y[index]) )
                        self.w = self.w - alpha *  v
                    else:
                        v = gamma * v + (1-gamma) * (coefficient * self.w)
                        self.w = self.w - alpha * v


            epoch = epoch + 1
            cost = abs(0.5 * np.dot(self.w, self.w.T) + C * condition)
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                        + ", Cost : " + str(cost))

        return self.w



    def train_momentum_cost1(self, X, y, alpha=0.01, epochs=1000, w_init=[]):
        #self.w = np.zeros(self.n_features)
        #self.w = np.random.uniform(0,1, self.n_features)
        self.w = w_init
        gamma = 0.99
        C = self.C
        # momentum portion
        v = np.array(np.zeros(self.w.shape))
        epoch = 1
        cost = 100000
        condition = 1
        coefficient = (2 * (1 / epoch))
        while (cost > 0.01 and epoch < epochs):
            alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                     #(-2 * (1 / epoch))
                    if (condition < 1):
                        # self.w = self.w + alpha * ((X[i] * y[i]) + (-2 * (1 / epoch) * self.w))
                        v = gamma * v - ((coefficient * self.w) - (X[i] * y[i]))
                        self.w = self.w + alpha * v
                    else:
                        v = gamma * v -  (coefficient * self.w)
                        self.w = self.w + alpha * v


            if(self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        v = gamma * v - (coefficient * self.w - (self.C * X[index] * y[index]))
                        self.w = self.w + alpha *  v
                    else:
                        v = gamma * v - (coefficient * self.w)
                        self.w = self.w + alpha * v


            epoch = epoch + 1
            cost = abs(0.5 * np.dot(self.w, self.w.T) + C * condition)
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                        + ", Cost : " + str(cost))

        return self.w

    def train_momentum_cost2(self, X, y, alpha=0.01, epochs=1000, w_init=[], indices_init=[], gamma=0.999):
        w = w_init
        indices = indices_init
        gamma = gamma
        C = self.C
        # momentum portion
        v = np.array(np.zeros(self.w.shape))
        print("Initial Cost : " + str(abs(0.5 * np.dot(w, w.T) + y[indices[0]] * np.dot(X[indices[0]], w))))
        epoch = 1
        cost = 100000
        condition = 1
        coefficient = (2 * (1 / epoch))
        while (cost > 0.001 and epoch < epochs):
            alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            cost = abs(0.5 * np.dot(w, w.T) + C * condition)
            for index in indices:
                condition = y[index] * np.dot(X[index], w)
                if (condition < 1):
                    v = gamma * v - (1)*(coefficient * w - (self.C * X[index] * y[index]))
                    w = w + alpha *  v
                else:
                    v = gamma * v - (1)*(coefficient * w)
                    w = w + alpha * v
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost))
            epoch = epoch + 1


        self.w = w
        return self.w



    def train_non_momentum(self, X, y, alpha=0.01, epochs=1000):
        #self.w = np.zeros(self.n_features)
        #self.w = np.random.uniform(0,1, self.n_features)
        gamma = self.gamma
        C = self.C
        # momentum portion
        v = np.array(np.zeros(self.w.shape))
        for epoch in range(1, epochs):
            # alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            coefficient = 1
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                     #(-2 * (1 / epoch))
                    if (condition <= 1):
                        self.w = (1-alpha) * self.w + alpha * C * y[i] * X[i]
                    else:
                        self.w = (1-alpha) * self.w

                cost = 0.5 * np.dot(self.w, self.w.T) + C * condition
                if (epoch % 10 == 0):
                    Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                        + ", Cost : " + str(cost))

            if(self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition <= 1):
                        self.w = (1-alpha) * self.w + alpha * C * y[index] * X[index]
                    else:
                        self.w = (1-alpha) * self.w

                cost = 0.5 * np.dot(self.w, self.w.T) + C * condition
                if (epoch % 10 == 0):
                    Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                        + ", Cost : " + str(cost))
        return self.w


    def pegasus_train(self,X, y):
        #self.w = np.zeros(self.n_features)
        for epoch in range(1, self.epochs):
            self.eta = 1.0 / (self.ld * epoch)
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                    if (condition < 1):
                        self.w = (1 - self.eta * self.ld) * self.w + self.eta * y[i] * X[i]
                    else:
                        self.w = (1 - self.eta * self.ld) * self.w
            if (self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        self.w = (1 - self.eta * self.ld) * self.w + self.eta * y[index] * X[index]
                    else:
                        self.w = (1 - self.eta * self.ld) * self.w
            cost = abs(0.5 * self.ld * np.dot(self.w, self.w.T) +  condition)
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(self.epochs)
                                + ", Cost : " + str(cost))


    def pegasus_train_cost(self,X, y):
        #self.w = np.zeros(self.n_features)
        epoch = 1
        cost = 100000
        while(cost > 0.01 and epoch < self.epochs):
            self.eta = 1.0 / (self.ld * epoch)
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], self.w)
                    if (condition < 1):
                        self.w = (1 - self.eta * self.ld) * self.w + self.eta * y[i] * X[i]
                    else:
                        self.w = (1 - self.eta * self.ld) * self.w
            if (self.randomize == True):
                num_range = len(X)
                indices = np.random.choice(num_range, num_range, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], self.w)
                    if (condition < 1):
                        self.w = (1 - self.eta * self.ld) * self.w + self.eta * y[index] * X[index]
                    else:
                        self.w = (1 - self.eta * self.ld) * self.w
            cost = abs(0.5 * self.ld * np.dot(self.w, self.w.T) +  condition)
            epoch = epoch + 1
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(self.epochs)
                                + ", Cost : " + str(cost))
        return self.w

    def train1(self, X, y, alpha=0.01, epochs=1000):
        self.w = np.zeros(self.n_features)
        m = X.shape[0]
        ld = 1.0
        C = 1/(ld*m)
        t = 0
        alpha = 1 / (t + 1)

    def kernel_predict(self, X, kernel):
        print("X shape ", X.shape)
        print("A shape ", self.a.shape)
        print("SV shape ", self.SV.shape)
        SV = self.SV
        a = self.a
        a = a[0]
        y_pred = []
        #print(a)
        print("X[0].shape ", X[0].shape)
        print("SV[0].shape ", SV[0].shape)
        itr = 0
        for sample in X:
            itr = itr + 1
            accumulator = 0
            prediction_label = 1
            for index in range(len(a)):
                kernel_value = 0
                if (kernel == 'linear'):
                    kernel_value = np.inner(SV[index], sample)
                if (kernel == 'rbf'):
                    pairwise_dists = squareform(pdist((np.array(SV[index]) - np.array([sample])), 'euclidean'))
                    K = sc.exp(-pairwise_dists ** 2 / self.gamma ** 2)
                    kernel_value = K
                if (kernel == 'poly'):
                    kernel_value = np.inner(SV[index], sample) ** self.degree
                accumulator += a[index] * kernel_value
            print(accumulator)
            if accumulator < 0:
                prediction_label = -1
            else:
                prediction_label = 1
            y_pred.append(prediction_label)

        # for i, x in enumerate(X):
        #     y_pred.append(np.sign(np.dot(X[i], self.w)))
        print(y_pred)
        return y_pred

    def predict(self, X):
        y_pred = []
        for i, x in enumerate(X):
            y_pred.append(np.sign(np.dot(X[i], self.w)))
        return y_pred

    def custom_predict(self,X, w):
        y_pred = []
        for i, x in enumerate(X):
            y_pred.append(np.sign(np.dot(X[i], w)))
        return y_pred

    def get_accuracy(self, y_test, y_pred):
        acc_count = 0
        for i, y in enumerate(y_pred):
            if (y_test[i] == (y_pred[i])):
                acc_count += 1
        return (float(acc_count) / float(len(y_pred)) * 100)


##################### ADA SVM TRAIN #############################

    def ada_sgd_train(self, X, y, alpha=0.01, epochs=1000, w_init=[]):
        w = w_init
        epsilon = 0.00000001
        r = np.zeros(w.shape)
        num_range = len(X)
        epoch = 1
        cost = 100000
        indices = np.random.choice(num_range, num_range, replace=False)
        while(cost > 0.01 and epoch < epochs):
            gradient = 0
            coefficient = 1
            for index in indices:
                condition = y[index] * np.dot(X[index], w)
                if (condition < 1):
                    gradient = alpha * (-(X[index] * y[index]) + (coefficient * w))
                else:
                    gradient = alpha * (coefficient * w)
                r = r + np.multiply(gradient,gradient)
                r = r + epsilon;
                d1 = np.multiply(gradient,1.0/r)
                w = w - (alpha * d1)

            cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            epoch = epoch + 1
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                + ", Cost : " + str(cost))
        self.w = w
        return w


###################### RMS PROP ############################################
    def rms_prop_sgd_train(self, X, y, alpha=0.01, epochs=1000, w_init=[], beta=0.90):
        w = w_init
        beta = beta
        epsilon = 0.00000001
        r = np.zeros(w.shape)
        num_range = len(X)
        cost = 100000000
        epoch = 1
        indices = np.random.choice(num_range, num_range, replace=False)
        while(cost > 0.01 and epoch < epochs):
            gradient = 0
            alpha = 1.0 / (1.0 + float(epoch))
            coefficient = 1#(-1 / (float(epoch) + epsilon))
            for index in indices:
                condition = y[index] * np.dot(X[index], w)
                if (condition < 1):
                    gradient = alpha * (-(X[index] * y[index]) + (coefficient * w))
                else:
                    gradient = alpha * (coefficient * w)
                r = (beta**epoch) * r + (1-(beta**epoch)) * np.multiply(gradient, gradient)
                r = np.sqrt(r) + epsilon;
                d1 = np.multiply(gradient, 1.0 / r)
                w = w - alpha * d1;

            cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            epoch = epoch + 1
            if (epoch % 10 == 0):
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                + ", Cost : " + str(cost))
        self.w = w
        return w


###################### ADAM ############################################
    def adam_sgd_train(self, X, y, X_testing, y_testing, alpha=0.01, epochs=1000, w_init=[], beta1=0.93, beta2=0.999):
        self.x_testing = X_testing
        self.y_testing = y_testing
        w = w_init
        v = np.zeros(w.shape)
        beta = 0.90
        beta1 = beta1
        beta2 = beta2
        epsilon = 0.00000001
        r = np.zeros(w.shape)
        num_range = len(X)
        cost = 100000000
        epoch = 1
        indices = np.random.choice(num_range, num_range, replace=False)
        while(cost > 0.001 and epoch < epochs):
            gradient = 0
            alpha = 1.0 / (1.0 + float(epoch))
            coefficient = ((1/float(epoch)))
            for index in indices:
                condition = y[index] * np.dot(X[index], w)
                if (condition < 1):
                    gradient = alpha * (-(X[index] * y[index]) + (coefficient * w))
                else:
                    gradient = alpha * (coefficient * w)

                v = beta1 * v + (1-beta1) * gradient
                v_hat = v / (1-beta1**epoch)
                r = beta2 * r + (1-beta2) * (np.multiply(gradient,gradient))
                r_hat = r / (1-beta2**epoch)
                w = w - alpha * np.multiply((v_hat),1.0/(np.sqrt(r_hat) + epsilon))
            cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            epoch = epoch + 1
            if (epoch % 10 == 0):
                online_acc = self.online_accuracy(w)
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                + ", Cost : " + str(cost) + ", Test Accuracy : " + str(online_acc) +" %" )
        self.w = w
        return w


###### Online Accuracy Testing ########

    def online_accuracy(self, wt):
        acc = 0

        if self.bulk:
            testing_filepath = self.testing_file
            print("Loading Bulk Testing Files")
            files = os.listdir(testing_filepath)
            print("File Path : " + testing_filepath)
            print(files)
            for file in files:
                print("Loading Testing Bulk File : " + file)
                testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath + "/" + file,
                                                       n_features=self.n_features)
                x_testing, y_testing = testing_loader.load_all_data()
                y_pred = self.custom_predict(x_testing, w=wt)
                acc += self.get_accuracy(y_test=y_testing, y_pred=y_pred)
            acc = acc / len(files)
        if self.split:
            # Print.Print.result2("Data Splitting ...")
            # training_loader = LoadLibsvm.LoadLibSVM(filename=self.trainPath, n_features=self.n_features)
            # x_all, y_all = training_loader.load_all_data()
            # ratio = 0.9
            # world_size = len(x_all)
            # split_index = int(world_size * ratio)
            # self.x_training = x_all[:split_index]
            # self.x_testing = x_all[split_index:]
            # self.y_training = y_all[:split_index]
            # self.y_testing = y_all[split_index:]
            y_pred = self.custom_predict(X=self.x_testing, w=wt)
            acc = self.get_accuracy(y_test=self.y_testing, y_pred=y_pred)

        else:
            testing_loader = LoadLibsvm.LoadLibSVM(filename=self.testPath, n_features=self.n_features)
            self.x_testing, self.y_testing = testing_loader.load_all_data()
            y_pred = self.custom_predict(X=self.x_testing, w=wt)
            acc = self.get_accuracy(y_test=self.y_testing, y_pred=y_pred)

        return acc

    def online_accuracy_light(self,wt, x_testing, y_testing):
        y_pred = self.custom_predict(X=x_testing, w=wt)
        acc = self.get_accuracy(y_test=y_testing, y_pred=y_pred)
        return acc

##########################################
###### OVERALL BENCHMARK SUITE ###########
##########################################

###### SGD Auto and Manual ############

    def train_sgd_light(self, X, y, X_test, y_test, alpha=0.01, epochs=1000, w_init=[], log_file='', log_frequency=1 ,tolerance=0.01, indices_init=[]):
        #self.w = np.zeros(self.n_features)
        #self.w = np.random.uniform(0,1, self.n_features)
        w = w_init
        epoch = 1
        cost = 1000000
        io_time = 0
        while(cost > tolerance and epoch < epochs):
            alpha = 1.0 / (1.0 + float(epoch)) # alpha update rule
            coefficient = 1.0 / float(1.0 + epoch)
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], w)
                    if (condition < 1):
                        w = w - alpha * (-(X[i] * y[i]) + (coefficient * w))
                    else:
                        w = w - alpha * (coefficient * w)

            if(self.randomize == True):
                num_range = len(X)
                indices = indices_init#np.random.choice(n, n, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], w)
                    if (condition < 1):
                        w = w - alpha * (-1*(X[index] * y[index]) + (coefficient * w))
                    else:
                        w = w - alpha * (coefficient * w)

            cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            if epoch ==1:
                initial_cost = cost

            epoch = epoch + 1
            if (epoch % log_frequency == 0):
                start_io_time = 0
                start_io_time -= time.time()
                acc = self.online_accuracy_light(wt=w, x_testing=X_test, y_testing=y_test)
                self.write_epoch_log(alpha=alpha, epoch=epoch, cost=cost, acc=acc, log_file=log_file)
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                    + ", Cost : " + str(cost) + ", Test Accuracy : " + str(acc) +" %")
                start_io_time += time.time()
                io_time += start_io_time
        return w, epoch, cost, io_time, initial_cost


    def train_sgd_light_no_coff(self, X, y, X_test, y_test, alpha=0.01, epochs=1000, w_init=[], log_file='', log_frequency=1 ,tolerance=0.01, indices_init=[]):
        #self.w = np.zeros(self.n_features)
        #self.w = np.random.uniform(0,1, self.n_features)
        w = w_init
        epoch = 1
        cost = 1000000
        io_time = 0
        initial_cost = 0
        while(cost > tolerance and epoch < epochs):
            coefficient = 1.0 / (1.0 + float(epoch))
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], w)
                    if (condition < 1):
                        w = w - alpha * (-(X[i] * y[i]) + (coefficient * w))
                    else:
                        w = w - alpha * (coefficient * w)

            if(self.randomize == True):
                num_range = len(X)
                indices = indices_init#np.random.choice(n, n, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], w)
                    if (condition < 1):
                        w = w - alpha * (-1*(X[index] * y[index]) + (coefficient* w))
                    else:
                        w = w - alpha * (coefficient* w)

            cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            if epoch ==1:
                initial_cost = cost

            epoch = epoch + 1
            if (epoch % log_frequency == 0):
                start_io_time = 0
                start_io_time -= time.time()
                acc = self.online_accuracy_light(wt=w, x_testing=X_test, y_testing=y_test)
                self.write_epoch_log(alpha=alpha, epoch=epoch, cost=cost, acc=acc, log_file=log_file)
                # Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                #                     + ", Cost : " + str(cost) + ", Test Accuracy : " + str(acc) +" %")
                start_io_time += time.time()
                io_time += start_io_time
        return w, epoch, cost, io_time, initial_cost


    def train_sgd_manual_light(self, X, y, X_test, y_test, alpha=0.01, epochs=1000, w_init=[], log_file='', log_frequency=1, tolerance=0.01, indices_init=[]):
        w = w_init
        epoch = 1
        cost = 1000000
        io_time = 0
        initial_cost = 0
        while(cost > tolerance and epoch < epochs):
            if (self.randomize == False):
                for i, x in enumerate(X):
                    condition = y[i] * np.dot(X[i], w)
                    if (condition < 1):
                        w = w + alpha * ((X[i] * y[i]) + (-2 * (1 / epoch) * w))
                    else:
                        w = w + alpha * (-2 * (1 / epoch) * w)

            if(self.randomize == True):
                num_range = len(X)
                indices = indices_init#np.random.choice(n, n, replace=False)
                for index in indices:
                    condition = y[index] * np.dot(X[index], w)
                    if (condition < 1):
                        w = w + alpha * ((X[index] * y[index]) + (-1 * (1 / epoch) * w))
                    else:
                        w = w + alpha * (-1 * (1 / epoch) * w)
            cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            if epoch ==1:
                initial_cost = cost
            epoch = epoch + 1
            if (epoch % log_frequency == 0):
                start_io_time = 0
                start_io_time -= time.time()
                acc = self.online_accuracy_light(wt=w, x_testing=X_test, y_testing=y_test)
                self.write_epoch_log(alpha=alpha, epoch=epoch, cost=cost, acc=acc, log_file=log_file)
                # Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                #                     + ", Cost : " + str(cost) + ", Test Accuracy : " + str(acc) + " %")
                start_io_time += time.time()
                io_time += start_io_time

        return w, epoch, cost, io_time, initial_cost

    def write_epoch_log(self,alpha=0, acc=0, cost=0, epoch=0, log_file=''):
        fp = open(log_file, "a")
        # fp.write("alpha : " + str(self.alpha) + ", epochs : " + str(self.epochs) + ", accuracy : " + str(self.acc) + "%" + ", time : " + str(self.training_time) + " s\n")
        fp.write(str(epoch) + "," + str(alpha) + "," + str(cost) + "," + str(acc) + "\n")
        fp.close()

########## SGD Momemtum #############

    def train_sgd_momentum(self, X, y, X_test, y_test, alpha=0.01, epochs=1000, w_init=[], log_file='', log_frequency=1 ,tolerance=0.01, indices_init=[], gamma=0.98):
        C = self.C
        w = w_init
        # momentum portion
        v = np.array(np.zeros(w.shape))
        epoch = 1
        cost = 100000
        io_time = 0
        coefficient = 1
        initial_cost = 0
        indices = indices_init
        while (cost > tolerance and epoch < epochs):
            num_range = len(X)
            coefficient = 1.0/(1.0 + float(epoch))
            #indices = np.random.choice(n, n, replace=False)
            for index in indices:
                condition = y[index] * np.dot(X[index], w)
                if (condition < 1):
                    v = gamma * v + (1 - gamma) * (coefficient * w - (self.C * X[index] * y[index]))
                    w = w - alpha * v
                else:
                    v = gamma * v + (1 - gamma) * (coefficient * w)
                    w = w - alpha * v


            cost = abs(0.5 * np.dot(w, w.T) + C * condition)
            if epoch ==1:
                initial_cost = cost
            epoch = epoch + 1
            if (epoch % log_frequency == 0):
                start_io_time = 0
                start_io_time -= time.time()
                acc = self.online_accuracy_light(wt=w, x_testing=X_test, y_testing=y_test)
                self.write_epoch_log(alpha=alpha, epoch=epoch, cost=cost, acc=acc, log_file=log_file)
                # Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                #                     + ", Cost : " + str(cost) + ", Test Accuracy : " + str(acc) + " %")
                start_io_time += time.time()
                io_time += start_io_time

        return w, epoch, cost, io_time, initial_cost

##################### Train SGD Ada Light #########################
    def ada_sgd_train_light(self, X, y, X_test, y_test, alpha=0.01, epochs=1000, w_init=[], log_file='', log_frequency=1 ,tolerance=0.01, indices_init=[]):
        w = w_init
        epsilon = 0.00000001
        r = np.zeros(w.shape)
        num_range = len(X)
        epoch = 1
        cost = 100000
        initial_cost = 0
        io_time = 0
        indices = indices_init
        while(cost > tolerance and epoch < epochs):
            gradient = 0
            #coefficient = 1.0 / (1.0 + float(epoch))
            coefficient = 1
            for index in indices:
                condition = y[index] * np.dot(X[index], w)
                if (condition < 1):
                    gradient = alpha * (-(X[index] * y[index]) + (coefficient * w))
                else:
                    gradient = alpha * (coefficient * w)
                r = r + np.multiply(gradient,gradient)
                r = r + epsilon;
                d1 = np.multiply(gradient,1.0/(np.sqrt(r)))
                w = w - (alpha * d1)

            cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            if epoch ==1:
                initial_cost = cost
            epoch = epoch + 1
            if (epoch % log_frequency == 0):
                start_io_time = 0
                start_io_time -= time.time()
                acc = self.online_accuracy_light(wt=w, x_testing=X_test, y_testing=y_test)
                self.write_epoch_log(alpha=alpha, epoch=epoch, cost=cost, acc=acc, log_file=log_file)
                # Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                #                     + ", Cost : " + str(cost) + ", Test Accuracy : " + str(acc) + " %")
                start_io_time += time.time()
                io_time += start_io_time

        self.w = w
        return w, epoch, cost, io_time, initial_cost

    ##################### Train SGD Rmsprop Light #########################
    def train_rmsprop_sgd(self, X, y, X_test, y_test, alpha=0.01, beta=0.90, epochs=1000, w_init=[], log_file='',
                            log_frequency=1, tolerance=0.01, indices_init=[]):
        w = w_init
        beta = beta
        epsilon = 0.00000001
        r = np.zeros(w.shape)
        num_range = len(X)
        cost = 100000000
        epoch = 1
        initial_cost = 0
        io_time = 0
        indices = indices_init
        while (cost > 0.0000001 and epoch < epochs):
            gradient = 0
            coefficient =  (1.0 / (1 + float(epoch)))
            for index in indices:
                condition = y[index] * np.dot(X[index], w)
                if (condition < 1):
                    gradient = alpha * (-(X[index] * y[index]) + (coefficient * w))
                else:
                    gradient = alpha * (coefficient * w)
                r = (beta ** epoch) * r + (1 - (beta ** epoch)) * np.multiply(gradient, gradient)
                r = np.sqrt(r) + epsilon;
                d1 = np.multiply(gradient, 1.0 / r)
                w = w - alpha * d1;

            cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            if epoch ==1:
                initial_cost = cost
            epoch = epoch + 1
            if (epoch % log_frequency == 0):
                start_io_time = 0
                start_io_time -= time.time()
                acc = self.online_accuracy_light(wt=w, x_testing=X_test, y_testing=y_test)
                self.write_epoch_log(alpha=alpha, epoch=epoch, cost=cost, acc=acc, log_file=log_file)
                # Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                #                     + ", Cost : " + str(cost) + ", Test Accuracy : " + str(acc) + " %")
                start_io_time += time.time()
                io_time += start_io_time
        self.w = w
        return w, epoch, cost, io_time, initial_cost

##################### Train SGD Adam Light #########################
    def adam_sgd_train_light(self, X, y, X_test, y_test, alpha=0.01, beta1=0.93, beta2=0.999, epochs=1000, w_init=[], log_file='', log_frequency=1 ,tolerance=0.01, indices_init=[]):
        w = w_init
        v = np.zeros(w.shape)
        beta1 = beta1
        beta2 = beta2
        epsilon = 0.00000001
        r = np.zeros(w.shape)
        num_range = len(X)
        cost = 100000000
        epoch = 1
        initial_cost = 0
        io_time = 0
        indices = indices_init
        while (cost > 0.0000001 and epoch < epochs):
            gradient = 0
            coefficient = 1#((1 / float(epoch)))
            for index in indices:
                condition = y[index] * np.dot(X[index], w)
                if (condition < 1):
                    gradient = alpha * (-(X[index] * y[index]) + (coefficient * w))
                else:
                    gradient = alpha * (coefficient * w)

                v = beta1 * v + (1 - beta1) * gradient
                v_hat = v / (1 - beta1 ** epoch)
                r = beta2 * r + (1 - beta2) * (np.multiply(gradient, gradient))
                r_hat = r / (1 - beta2 ** epoch)
                w = w - alpha * np.multiply((v_hat), 1.0 / (np.sqrt(r_hat) + epsilon))
                #print(w)
            cost = abs(0.5 * np.dot(w, w.T) + self.C * condition)
            if epoch ==1:
                initial_cost = cost
            epoch = epoch + 1
            if (epoch % log_frequency == 0):
                start_io_time = 0
                start_io_time -= time.time()
                acc = self.online_accuracy_light(wt=w, x_testing=X_test, y_testing=y_test)
                self.write_epoch_log(alpha=alpha, epoch=epoch, cost=cost, acc=acc, log_file=log_file)
                Print.Print.result1("Epoch " + str(epoch) + "/" + str(epochs)
                                     + ", Cost : " + str(cost) + ", Test Accuracy : " + str(acc) + " %")
                start_io_time += time.time()
                io_time += start_io_time
        self.w = w
        return w, epoch, cost, io_time, initial_cost
