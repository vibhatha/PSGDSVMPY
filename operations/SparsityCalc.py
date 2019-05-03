import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from operations import LoadLibsvm
import numpy as np

training_filepath = sys.argv[1]
n_features = int(sys.argv[2])

training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=n_features)
x_training, y_training = training_loader.load_all_data()
totalpoints = len(x_training) * n_features
zeropoints = totalpoints - np.count_nonzero(x_training)
print(float(zeropoints)/float(totalpoints)*100.0)
