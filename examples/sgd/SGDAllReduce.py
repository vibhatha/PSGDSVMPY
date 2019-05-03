import sys
import os

HOME = os.environ['HOME']
sys.path.insert(1, HOME + '/github/StreamingSVM')
from comms import Communication
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

comms = Communication.Communication()

rank = comms.comm.Get_rank()
size = comms.comm.Get_size()

input = np.array([1,2,3,4,5,6,7,8,9,10], dtype='f')

print("Simple: Input :" + str(input) + " From Rank : " + str(rank))

# initialize the numpy arrays that store the results from reduce operation
output_max = np.empty(len(input), 'f')
output_sum = np.empty(len(input), 'f')

# perform reduction based on sum and maximum
comms.allreduce(input=input, output=output_max, op=comms.mpi.MAX, dtype=comms.mpi.FLOAT)
comms.allreduce(input=input, output=output_sum, op=comms.mpi.SUM, dtype=comms.mpi.FLOAT)

if (rank == 0):
    print("Simple: Output Max : " + str(output_max) + ", from Rank " + str(rank) + "\n")
    print("Simple: Output Sum : " + str(output_sum) + ", from Rank " + str(rank) + "\n")
