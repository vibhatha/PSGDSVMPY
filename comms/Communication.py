import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from mpi4py import MPI
import numpy as np
from api import Constant

class Communication:
    mpi = MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    constant = Constant.Constant()


    def __init__(self):
        rank = self.comm.Get_rank()
        if (rank == 0):
            print("Initial Configurations : World Size = " + str(self.size))

    def isend(self, input=[], dest=1, tag=1):
        self.comm.isend(input, dest=dest, tag=tag)

    def irecv(self, source=1, tag=1):
        data = self.comm.irecv(source=source, tag=tag)
        return data

    def send(self, input= np.empty(1, dtype='i'), dtype=mpi.Datatype, dest=1, tag=1):
        self.comm.Send([input, dtype], dest=dest, tag=tag)


    def recv(self, dtype=mpi.Datatype, source=0, tag=1, size=1):
        data = np.empty(size, dtype=self.constant.get_type(dtype))
        self.comm.Recv([data, dtype], source=source, tag=tag)
        return data

    def bcast(self, input=np.empty(1, dtype='i'), dtype=mpi.Datatype, root=0):
        self.comm.Bcast([input, dtype], root=root)

    def scatter(self, input=np.empty(1, dtype='i'), recvbuf=np.empty(1, dtype='i') , dtype=mpi.Datatype, root=0):
        self.comm.Scatter([input, dtype], recvbuf=recvbuf, root=root)

    def gather(self, sendbuf=np.empty(1, dtype='i'), recvbuf=np.empty(1, dtype='i'), dtype=mpi.Datatype ,root=0):
        self.comm.Gather([sendbuf, dtype], [recvbuf, dtype], root=root)

    def reduce(self, input=np.empty(1, dtype='i'), output=np.empty(1, dtype='i'), dtype=mpi.Datatype ,op=mpi.SUM, root=0 ):
        self.comm.Reduce([input,dtype], [output, dtype], op=op, root=root)

    def allreduce(self, input=np.empty(1, dtype='i'), output=np.empty(1, dtype='i'), dtype=mpi.Datatype, op=mpi.SUM):
        self.comm.Allreduce([input,dtype], [output, dtype], op=op)
