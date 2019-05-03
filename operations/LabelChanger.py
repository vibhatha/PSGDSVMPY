import sys
import numpy as np
from numpy import genfromtxt


def load_csv(inputfile, outputfile):
    #print(inputfile)
    data = genfromtxt(inputfile, delimiter=',')
    #print(data)
    y = data[:, 0]
    y[y==2]=-1
    y = y.reshape((len(y),1))
    #print(y)
    x = np.delete(data, 0, axis=1)
    #print(x)
    all_data = np.concatenate((y,x),1)
    #print(all_data)
    np.savetxt(outputfile, all_data, delimiter=",", fmt='%1.7e')


inputfile = sys.argv[1]
outputfile = sys.argv[2]
load_csv(inputfile=inputfile, outputfile=outputfile)
