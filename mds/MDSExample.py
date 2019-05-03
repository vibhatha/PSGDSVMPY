import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

data_home='/home/vibhatha/data/svm/'
dataset ='webspam'
filename = 'training.csv'
datafile = data_home + dataset + '/' + filename
print(datafile)
Xtrue = np.genfromtxt(datafile, delimiter=',')

print(np.array_str(Xtrue, precision=6))

euc_dis = euclidean_distances(Xtrue)

print(euc_dis)

(X, stress, it) = manifold.mds._smacof_single(euc_dis, init=Xtrue, max_iter=100)

# dis = euclidean_distances(X)
#
# print(dis)
#
# stress = ((dis.ravel() - euc_dis.ravel()) ** 2).sum() / 2
# print(stress)
#
# dis[dis == 0] = 1e-5
# ratio = euc_dis / dis
# B = - ratio
# B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
#
# print(B)
#
# n_samples = dis.shape[0]
# Xtemp = 1. / n_samples * np.dot(B, X)
#
# print(Xtemp)
pass
