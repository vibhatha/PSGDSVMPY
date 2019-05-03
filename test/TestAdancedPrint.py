import time

for n in range(1, 11):
    print('I am on number {} of 10'.format(n), end='\r')
    time.sleep(.5)
