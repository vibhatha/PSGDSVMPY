import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')
from comms import Communication
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor


class ReduceExample:
    comms = Communication.Communication()

    def execute(self):
        rank = self.comms.comm.Get_rank()
        size = self.comms.comm.Get_size()

        input = np.array(rank, dtype='i')

        print("Simple: Input :" + str(input) + " From Rank : " + str(rank))

        # initialize the numpy arrays that store the results from reduce operation
        output_max = np.array(0, 'i')
        output_sum = np.array(0, 'i')

        #perform reduction based on sum and maximum
        self.comms.reduce(input=input, output=output_max, op=self.comms.mpi.MAX, dtype=self.comms.mpi.INT, root=0)
        self.comms.reduce(input=input, output=output_sum, op=self.comms.mpi.SUM, dtype=self.comms.mpi.INT, root=0)

        if (rank == 0):
            print("Simple: Output Max : " + str(output_max) + ", from Rank " + str(rank) + "\n")
            print("Simple: Output Sum : " + str(output_sum) + ", from Rank " + str(rank) + "\n")


class ReduceAdvancedExample:
    comms = Communication.Communication()

    def execute(self):
        rank = self.comms.comm.Get_rank()
        size = self.comms.comm.Get_size()

        num_of_data_per_rank = 2
        start = rank * num_of_data_per_rank + 1
        end = (rank + 1) * num_of_data_per_rank
        input = np.linspace(start, end, num_of_data_per_rank, dtype='i')

        print("Advanced: Input :" + str(input) + " From Rank : " + str(rank))

        # initialize the numpy arrays that store the results from reduce operation
        output_max = np.empty(num_of_data_per_rank, 'i')
        output_sum = np.empty(num_of_data_per_rank, 'i')

        #perform reduction based on sum and maximum
        self.comms.reduce(input=input, output=output_max, op=self.comms.mpi.MAX, dtype=self.comms.mpi.INT, root=0)
        self.comms.reduce(input=input, output=output_sum, op=self.comms.mpi.SUM, dtype=self.comms.mpi.INT, root=0)

        if (rank == 0):
            print("Advanced: Output Max : " + str(output_max) + ", from Rank " + str(rank) + "\n")
            print("Advanced: Output Sum : " + str(output_sum) + ", from Rank " + str(rank) + "\n")



#t1 = threading.Thread(target=ReduceExample().execute)
#t1.start()


#t2 = threading.Thread(target=ReduceAdvancedExample().execute)
#t2.start()

#t1.join()
#t2.join()

pool = ThreadPoolExecutor(4)
ex1 = ReduceExample().execute
ex2 = ReduceAdvancedExample().execute
future1 = pool.submit(ex2)

exs = [ex1,ex2]

if(future1.done()):
    print("Results",future1.result())

threading._sleep(1)

future2 = pool.submit(ex1)

# with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exeuctor:
#     furture_to_execute = {exeuctor.submit(ex): ex for ex in exs}
#     for future in concurrent.futures.as_completed(furture_to_execute):
#         execute = furture_to_execute[future]
#         try:
#             data = future.result()
#         except Exception as exc:
#             print(execute, exc)
#         else:
#             print(execute, data)
