import sys
import os

HOME = os.environ['HOME']
sys.path.insert(1, HOME + '/github/StreamingSVM')
from comms import Communication
import numpy as np
import time

class SendRecvExample:
    comms = Communication.Communication()

    def get_data(self,rank):
        if(rank==0):
            return np.array([1,1], dtype='i')
        if (rank == 1):
            return np.array([2, 1], dtype='i')
        if (rank == 2):
            return np.array([3, 1], dtype='i')
        if (rank == 3):
            return np.array([4, 2], dtype='i')


    def example(self):
        world_rank = self.comms.comm.Get_rank()
        world_size = self.comms.comm.Get_size()
        dsize = 2
        source = 0
        input = np.array([0, 0], dtype='i')
        master_input = np.array([0, 0], dtype='i')
        count = 0
        partner_rank = 1
        dest = -1
        p = np.random.randint(4)
        exec_time = 0
        exec_time -= time.time()
        max_itr = 10000
        for i in range(0,max_itr):
            if (world_rank == 0):

                if (world_size == 1):
                    dest = world_rank
                else:
                    dest = world_rank + 1
                master_input = self.get_data(world_rank)
                # np.asarray(master_input, dtype='i')
                self.comms.send(input=master_input, dtype=self.comms.mpi.INT, dest=dest, tag=0)

                if (world_size == 1):
                    source = world_rank
                else:
                    source = world_size - 1

                data = self.comms.recv(source=source, dtype=self.comms.mpi.INT, tag=0, size=dsize)
                #print("I am Master " + str(world_rank) + ", I received from " +str(source) +" : ", data)

            else:
                source = world_rank - 1
                data = self.comms.recv(source=source, dtype=self.comms.mpi.INT, tag=0, size=dsize)
                #print("I am slave " + str(world_rank) + ", I received from " +str(source) +" : ", data)
                data = self.get_data(world_rank)
                #print("I am slave " + str(world_rank) + ", I sent ", data)
                dest = (world_rank + 1) % world_size;
                # np.asarray(data, dtype='i')
                self.comms.send(input=data, dtype=self.comms.mpi.INT, dest=dest, tag=0)

        exec_time += time.time()
        if(i==(max_itr-1)):
            if(world_rank==0):
                exec_time = exec_time / 10
                print("Execution Time : ", exec_time)


    def example1(self):
        world_rank = self.comms.comm.Get_rank()
        world_size = self.comms.comm.Get_size()
        dsize = 2
        source = 0
        input = np.array([0, 1, 2, 3, 4], dtype='i')
        master_input = np.array([0, 1, 2, 3, 4], dtype='i')
        count = 0
        partner_rank = 1
        dest = 0

        if(world_rank == 0) :
            if(source == 0):
                print('Starting the programme ...', master_input)

            if(world_size == 1):
                dest = world_rank
            else:
                dest = world_rank + 1
            master_input = master_input * (world_rank + 1)
            #np.asarray(master_input, dtype='i')
            self.comms.send(input=master_input, dtype=self.comms.mpi.INT, dest=dest, tag=0)

            if(world_size == 1):
                source = world_rank
            else:
                source = world_size - 1

            data = self.comms.recv(source=source, dtype=self.comms.mpi.INT, tag=0, size=dsize)
            print("I am Master " + str(world_rank) + ", I received : ", data)

        else:
            source = world_rank - 1
            data = self.comms.recv(source=source, dtype=self.comms.mpi.INT, tag=0, size=5)
            print("I am slave " + str(world_rank) + ", I received : ", data)
            data = data * world_rank
            print("I am slave " + str(world_rank) + ", I sent ", data)
            dest = (world_rank + 1) % world_size;
            #np.asarray(data, dtype='i')
            self.comms.send(input=data, dtype=self.comms.mpi.INT, dest=dest, tag=0)


    def example2(self):
        comms = Communication.Communication()
        max_itr = 10000
        rank = self.comms.comm.Get_rank()
        size = self.comms.comm.Get_size()

        input = np.array(rank, dtype='i')

        print("Simple: Input :" + str(input) + " From Rank : " + str(rank))

        # initialize the numpy arrays that store the results from reduce operation
        # output_max = np.array(0, 'i')
        exec_time = 0
        exec_time -= time.time()
        for i in range(0, max_itr):
            output_sum = np.array(0, 'i')

            # perform reduction based on sum and maximum
            # self.comms.allreduce(input=input, output=output_max, op=self.comms.mpi.MAX, dtype=self.comms.mpi.INT)
            self.comms.allreduce(input=input, output=output_sum, op=self.comms.mpi.SUM, dtype=self.comms.mpi.INT)

            if (rank == 0):
                k=1
                # print("Simple: Output Max : " + str(output_max) + ", from Rank " + str(rank) + "\n")
                #print("Simple: Output Sum : " + str(output_sum) + ", from Rank " + str(rank) + "\n")
        exec_time += time.time()
        if (i == (max_itr - 1)):
            if (rank == 0):
                exec_time = exec_time / 10
                print("Execution Time : ", exec_time)





ex = SendRecvExample()
ex.example()
