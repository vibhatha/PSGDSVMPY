import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from comms import Communication
import numpy as np

class ISendIRecvExample:
    comms = Communication.Communication()

    def example(self):
        rank = self.comms.comm.Get_rank()

        if (rank == 0):
            input = np.array([0,1,2,3,4])
            self.comms.isend(input=[input, self.comms.mpi.INT], dest=1, tag=11)
            print("Sending Data : " + str(input) + ", from Rank " + str(rank) + "\n")

        elif (rank == 1):
            data = self.comms.irecv(source=0, tag=11)
            print(type(data))
            print("Receiving Data : " + str(data) + ", from Rank " + str(rank) + "\n")


ex = ISendIRecvExample()
ex.example()
