import sys
import os
import socket
HOME = os.environ['HOME']
sys.path.insert(1, HOME + '/github/StreamingSVM')
from distributed import DistributedDataLoader
from api import Constant
import numpy as np

M = 4
i = 0
dis = DistributedDataLoader.DistributedDataLoader(source_file=Constant.Constant.SOURCE_REAL_SIM, n_features=Constant.Constant.REAL_SLIM_F, n_samples=Constant.Constant.REAL_SLIM_S, world_size=4, rank=i)
x_all, y_all = dis.load_training_data_batch_per_core()
X_test, y_test = dis.load_testing_data()
print(len(x_all),len(y_all),x_all[0].shape, y_all[0].shape)
rank = 0
for i in range(len(x_all)):
    X = x_all[i]
    y = y_all[i]
    m = len(X)
    C = int(m/M)
    m_real = C * M
    range = np.arange(0, m_real-1, M)
    for i in range:
        xi = X[i+rank]
        yi = y[i+rank]


