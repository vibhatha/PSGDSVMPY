import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#Training Data
x = np.array([[1,2],[3,2],[3,4],[7,2],[10,1],[7,3],[11,4],[13,3]])
y = np.array([1,1,1,-1,-1,-1,-1,-1])

#Testing Data
x_test=np.array([[1,2],[3,2],[3,4],[7,2],[10,1],[7,3],[11,4],[13,3]])

#Adding 1st column as bias
x = np.c_[np.ones((x.shape[0])), x]

#For Testing
x_test = np.c_[np.ones((x.shape[0])), x_test]

#Initialize weights, epochs, learning rate
w = np.random.uniform(size=(x.shape[1],))
epochs=1000
learning_rate=0.01

#Stochastic Gradient Descent with hinge loss
for epoch in range (0,epochs):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x=x[randomize]
    y=y[randomize]
    loss=0
    for xi,yi in zip(x,y):
        loss+=max(0,1-yi*np.dot(xi,w))
        if(yi*np.dot(xi,w)<1):
            #grad+=-yi*xi
            w=w-learning_rate*(-yi*xi)
        else:
            #grad+=0
            w=w
    print(w,loss)

#Predicting with new values
def predict(x,w):
    return(np.sign(np.dot(x,w)))

#Plotting
def plot(x,w):
    Y=(-w[0] - (w[1] * x)) / w[2]
    psy=(1-w[0] - (w[1] * x)) / w[2]
    nsy=(-1-w[0] - (w[1] * x)) / w[2]
    plt.figure()
    plt.scatter(x[:,1],x[:,2],c=y,edgecolor='black', cmap=plt.cm.Paired,s=20)
    plt.plot(x,Y,'r-')
    plt.plot(x,psy,'b-')
    plt.plot(x,nsy,'b-')
    margin=2/np.sqrt(w[1]**2+w[2]**2)
    plt.show()

ans=predict(x_test,w)
print(ans)

#Finding values for slack variables
for xi,yi in zip(x,y):
    slack=max(0,1-yi*np.dot(xi,w))
    print(xi[1:],' Slack:',slack)

#Finding Support Vectors
for xi in x:
    if(round(np.linalg.norm((np.dot(xi,w))*(w/np.linalg.norm(w))),1)==1):
        print("Support Vector:",xi[1:])
plot(x,w)
