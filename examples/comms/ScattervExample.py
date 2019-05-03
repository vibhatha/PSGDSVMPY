import numpy
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size=1

a_size = 4
recvdata = numpy.empty(a_size,dtype=numpy.float64)
senddata = None
if rank == 0:
   senddata = numpy.arange(size*a_size,dtype=numpy.float64)
comm.Scatter(senddata,recvdata,root=0)
print 'on task',rank,'after Scatter:    data = ',recvdata

recvdata = numpy.empty(a_size,dtype=numpy.float64)
counts = None
dspls = None
if rank == 0:
   senddata = numpy.arange(16,dtype=numpy.float64)
   counts=(1,2,3,4)
   dspls=(0,1,2,7)
   print("Sendata, ", senddata)
comm.Scatterv([senddata,counts,dspls,MPI.DOUBLE],recvdata,root=0)
print 'on task',rank,'after Scatterv:    data = ',recvdata
