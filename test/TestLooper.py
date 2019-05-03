import numpy as np
import time
r1 = np.arange(0,10000,1)
# sum = 0
# exec_zip_time = 0
# exec_zip_time -= time.time()
# for i,j in zip(r1,r1):
#     print(i,j)
# exec_zip_time += time.time()
# print("sum : " + str(sum))
# print("zip exec time : " + str(exec_zip_time))
# print("--------------------------------")
# print("--------------------------------")
# print("--------------------------------")
sum = 0
exec_zip_time = 0
exec_zip_time -= time.time()
for i in r1:
    for j in r1:
        sum += i * j
exec_zip_time += time.time()
print("sum : " + str(sum))
print("normal exec time : " + str(exec_zip_time))
print("--------------------------------")
import itertools

sum=0
exec_zip_time = 0
exec_zip_time -= time.time()
for i, j in itertools.product(r1, r1):
    sum += i * j
exec_zip_time += time.time()
print("sum : " + str(sum))
print("iter tools exec time : " + str(exec_zip_time))
