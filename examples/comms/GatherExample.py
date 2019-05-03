import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from comms import Communication
import numpy as np


class GatherExample:
    comms = Communication.Communication()

    def example(self):
        rank = self.comms.comm.Get_rank()
        size = self.comms.comm.Get_size()
        num_of_data_per_rank = 2
        start = rank * num_of_data_per_rank + 1
        end = (rank+1) * num_of_data_per_rank
        sendbuf = np.linspace(start, end , num_of_data_per_rank, dtype='i')

        print("Sendbuf :" + str(sendbuf) + " From Rank : " + str(rank))
        recvbuf = None
        if (rank == 0):
            recvbuf = np.empty(num_of_data_per_rank * size, dtype='i')

        self.comms.gather(sendbuf=sendbuf, recvbuf=recvbuf, dtype=self.comms.mpi.INT, root=0)

        if (rank == 0):
            print("Receiving Data : " + str(recvbuf) + ", from Rank " + str(rank) + "\n")


ex = GatherExample()
ex.example()
