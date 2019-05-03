import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
import numpy as np
from matplotlib import pyplot as plt

X = np.array([[1,1],[2,1],[3,1],[4,1],[1,5],[2,6],[3,7],[4,5]])
y = np.array([1,1,1,1,-1,-1,-1,-1])
X_test = np.array([[1,1.25],[2.1,1.15],[3.1,1.45],[4.23,1.21],[1.3,5.25],[2.11,6.24],[3.3,7.24],[4.212,5.78]])
y_test = np.array([1,1,1,1,-1,-1,-1,-1])
M = 1
R = 0
e2 = 1
w = y[0] * X[0]
C = 1.0
O = X[0]
mebs = []
#for i in range(1,len(X)):
i=0
L=4
while (i < len(X)-L):
    for j in range(i, i + L):
        d = np.sqrt(np.linalg.norm(w-y[j]*X[j]) + e2 + 1/C)
        if d >= R:
            w = w + 0.5 * (1 - R/d) * (y[j] * X[j] - w)
            R = R + 0.5 * (d-R)
            e2 = e2 * (1- (0.5) * (1- (R/d)))**2 + (0.5 * (1 - (R/d)))**2
            M = M + 1
            mebs.append(X[j])
    i = i + 1

    print("R : ", i,  R)
print("R : ", R)
print("W : ", w)
print("M : ", M)
meb_array = np.array(mebs)
O = np.sum(meb_array, axis=0)/len(mebs)
print("O : ", O)

#plt.scatter(X[:,0],X[:,1])
#plt.show()

labels = []
for x in X_test:
    label = np.sign(np.dot(w.T, x))
    labels.append(label)

y_pred = np.array(labels)
print(labels)
print(y_test)
correct = (y_pred == y_test).sum()
total = len(y_pred)
acc = float(correct) / float(total) * 100.0
print("Acc : ", acc)

# graph = plt.scatter(X[:,0],X[:,1])
# circle1 = plt.Circle((O[0],O[1]), R)
# fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
# (or if you have an existing figure)
# fig = plt.gcf()
# ax = fig.gca()

# ax.add_artist(circle1)
# ax.add_artist(graph)
#
#
# fig.savefig('figures/1.png')
