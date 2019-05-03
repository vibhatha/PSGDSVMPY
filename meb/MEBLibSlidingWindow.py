import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from operations import LoadLibsvm
import numpy as np
from matplotlib import pyplot as plt

X = np.array([[1,1],[2,1],[3,1],[4,1],[1,5],[2,6],[3,7],[4,5]])
y = np.array([1,1,1,1,-1,-1,-1,-1])
X_test = np.array([[1,1.25],[2.1,1.15],[3.1,1.45],[4.23,1.21],[1.3,5.25],[2.11,6.24],[3.3,7.24],[4.212,5.78]])
#plt.scatter(X[:,0],X[:,1])
#plt.show()
dataset = 'ijcnn1'
training_filepath = '/home/vibhatha/data/svm/'+dataset+'/training.csv'
testing_filepath = '/home/vibhatha/data/svm/'+dataset+'/testing.csv'
n_features = 300
split = False
training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=n_features)

x_training = []
y_training = []
x_testing = []
y_testing = []

if split == True:
    x_all, y_all = training_loader.load_all_data()
    ratio = 0.8
    size = len(x_all)
    split_index = int(size * ratio)
    x_training = x_all[:split_index]
    x_testing = x_all[split_index:]
    y_training = y_all[:split_index]
    y_testing = y_all[split_index:]

else :
    training_loader = LoadLibsvm.LoadLibSVM(filename=training_filepath, n_features=n_features)
    testing_loader = LoadLibsvm.LoadLibSVM(filename=testing_filepath, n_features=n_features)
    x_training, y_training = training_loader.load_all_data()
    x_testing, y_testing = testing_loader.load_all_data()

print(x_training.shape)
X = x_training
y = y_training


M = 1
R = 0
e2 = 1
w = y[0] * X[0]
C = 1
i=0
L=200
while (i < len(X)-L):
    for j in range(i, i + L):
        d = np.sqrt(np.linalg.norm(w-y[j]*X[j]) + e2 + 1/C)
        if d >= R:
            w = w + 0.5 * (1 - R/d) * (y[j] * X[j] - w)
            R = R + 0.5 * (d-R)
            e2 = e2 * (1- (0.5) * (1- (R/d)))**2 + (0.5 * (1 - (R/d)))**2
            M = M + 1
        i = i + 1
print("R : ", R)
print("W : ", w)
print("M : ", M)

#plt.scatter(X[:,0],X[:,1])
#plt.show()

labels = []
for x in x_testing:
    label = np.sign(np.dot(w.T, x))
    labels.append(label)

y_pred = np.array(labels)
print(labels)
print(y_testing)
correct = (y_pred == y_testing).sum()
total = len(y_pred)
acc = float(correct) / float(total) * 100.0
print("Acc : ", acc)
