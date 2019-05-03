import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from comms import Communication
import numpy as np


class BcastRecvExample:
    comms = Communication.Communication()

    def example(self):
        rank = self.comms.comm.Get_rank()

        if(rank==0):
            input = np.array([0, 1, 2, 3, 4], dtype='i')
        else:
            input = np.empty(5, dtype='i')
        # if (rank == 0):
        #     print("Broadcasting Data : " + str(input) + ", from Rank " + str(rank) + "\n")
        self.comms.bcast(input=input, dtype=self.comms.mpi.INT, root=0)

        print("Receiving Data : " + str(input) + ", from Rank " + str(rank) + "\n")


ex = BcastRecvExample()
ex.example()
