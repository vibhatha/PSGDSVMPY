import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from comms import Communication
import numpy as np


class ScatterExample:
    comms = Communication.Communication()

    def example(self):
        rank = self.comms.comm.Get_rank()
        size = self.comms.comm.Get_size()
        num_of_data_per_rank = 8
        input = np.array([[1, 2, 3, 4, 5, 6, 7, 8],[1, 3, 5, 7, 9, 11, 13, 15]], np.int32)
        recvbuf = np.empty(num_of_data_per_rank, dtype='i')
        self.comms.scatter(input=input, recvbuf=recvbuf, dtype=self.comms.mpi.INT, root=0)
        if (rank == 0):
            print("Scattering Data : " + str(input) + ", from Rank " + str(rank) + "\n")


        print("Receiving Data : " + str(recvbuf) + ", from Rank " + str(rank) + "\n")


ex = ScatterExample()
ex.example()
