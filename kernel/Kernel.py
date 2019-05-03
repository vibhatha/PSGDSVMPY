import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
import numpy as np
import scipy as sc
from scipy.spatial.distance import pdist, squareform
import sklearn.metrics.pairwise

class Kernel:

    def __init__(self, Xi, Xj, kernel, gamma=0.01, degree=2):
        self.Xi = Xi
        self.Xj = Xj
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

    def compute(self):
        Xk = 0
        if self.kernel == 'linear':
            Xk = self.linear(self.Xi, self.Xj)

        if self.kernel == 'rbf':
            assert(self.gamma > 0)
            Xk = self.radial(self.Xi, self.Xj, self.gamma)

        if self.kernel == 'poly':
            assert(self.degree >= 1)
            Xk = self.polynomial(self.Xi, self.Xj, self.degree)

        return Xk

    def polynomial(self,xi, xj, degree):
        return np.dot(xi,xj.T)**degree

    def linear(self,xi, xj):
        return np.dot(xi, xj.T)

    def radial(self,xi, xj):
        pairwise_dists = squareform(pdist((xi-xj), 'euclidean'))
        K = sc.exp(-pairwise_dists ** 2 / self.gamma ** 2)
        return K

    @staticmethod
    def self_polynomial(x, degree):
        assert (degree >= 1)
        return np.dot(x,x.T)**degree

    @staticmethod
    def self_linear(x):
        return np.dot(x,x.T)

    @staticmethod
    def self_radial(x, gamma):
        pairwise_dists = squareform(pdist((x), 'euclidean'))
        K = sc.exp(-pairwise_dists ** 2 / gamma ** 2)
        return K

    @staticmethod
    def generate_kerenelized_feature_matrix(X, kernel, gamma=2, degree=2):
        Xk = []
        if (kernel == 'linear'):
            Xk = sklearn.metrics.pairwise.linear_kernel(X=X)

        if (kernel == 'rbf'):
            Xk = sklearn.metrics.pairwise.rbf_kernel(X=X, gamma=gamma)

        if (kernel == 'poly'):
            Xk = sklearn.metrics.pairwise.polynomial_kernel(X=X, degree=degree)

        return Xk
