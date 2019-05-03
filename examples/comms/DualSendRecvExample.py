import sys
import os

HOME = os.environ['HOME']
sys.path.insert(1, HOME + '/github/StreamingSVM')
from comms import Communication
import numpy as np
import time
from threading import Thread
from dask.distributed import Client


class SendRecvExample:
    comms = Communication.Communication()

    def get_data(self, rank):
        if (rank == 0):
            return np.array([1, 1], dtype='i')
        if (rank == 1):
            return np.array([2, 1], dtype='i')
        if (rank == 2):
            return np.array([3, 1], dtype='i')
        if (rank == 3):
            return np.array([4, 2], dtype='i')

    def example(self):
        rank = self.comms.comm.Get_rank()
        dsize = 2

        if (rank == 0):
            input = self.get_data(rank)
            self.comms.send(input=input, dtype=self.comms.mpi.INT, dest=1, tag=11)
            print("Sending Data : " + str(input) + ", from Rank " + str(rank) + "\n")

        elif (rank == 1):
            data = self.comms.recv(source=0, dtype=self.comms.mpi.INT, tag=11, size=dsize)
            print("Receiving Data : " + str(data) + ", from Rank " + str(rank) + "\n")

        if (rank == 1):
            input = self.get_data(rank)
            self.comms.send(input=input, dtype=self.comms.mpi.INT, dest=0, tag=11)
            print("Sending Data : " + str(input) + ", from Rank " + str(rank) + "\n")

        elif (rank == 0):
            data = self.comms.recv(source=1, dtype=self.comms.mpi.INT, tag=11, size=dsize)
            print("Receiving Data : " + str(data) + ", from Rank " + str(rank) + "\n")

    def threadExample(self):
        world_rank = self.comms.comm.Get_rank()
        world_size = self.comms.comm.Get_size()
        client = Client()
        exec_time = 0
        exec_time -= time.time()
        a = client.submit(self.sendToRank, 1, world_size)  # calls inc(10) in background thread or process
        b = client.submit(self.recvFromRank, 0, world_size )  # calls inc(20) in background thread or process
        print(a.result(), b.result())
        exec_time += time.time()
        print("Dask Time Taken : " + str(exec_time))

    def myfunc(self, i):
        print "sleeping 5 sec from thread %d" % i
        time.sleep(5)
        print "finished sleeping from thread %d" % i

    def callmyfunc(self):

        t = Thread(target=self.myfunc, args=(1,))
        t.start()

    def sendToRank(self, world_rank, world_size):
        send_data = self.get_data(world_rank)
        dest = world_rank
        self.comms.send(input=send_data, dtype=self.comms.mpi.INT, dest=dest, tag=11)
        print(world_rank, "Send Data ", send_data)

    def recvFromRank(self, world_rank, world_size):
        data = self.comms.recv(source=world_rank, dtype=self.comms.mpi.INT, tag=11, size=2)
        print(world_rank, "Data ", data)

    def do_ring(self):
        world_rank = self.comms.comm.Get_rank()
        world_size = self.comms.comm.Get_size()
        t1 = Thread(target=self.sendToRank, args=(world_rank, world_size,))
        t2 = Thread(target=self.recvFromRank, args=(world_rank, world_size,))
        t1.start()
        t2.start()


ex = SendRecvExample()
ex.threadExample()
