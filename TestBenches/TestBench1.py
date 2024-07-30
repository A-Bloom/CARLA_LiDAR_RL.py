import numpy as np
import timeit
import math


def nphypot():
    x = np.array([-2, -2, -2, -3, -3, -3, -3, -3, -2, -2, -1, 0, 0, 1, 2, 3, 4, 4, 4, 4, 3, 3, 3, 3], dtype=int)
    y = np.array([-4, -3, -2, -2, -1, 0, 1, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 1, 0, -1, -1, -2, -3, -4], dtype=int)



def pyhypot():
    x = [-2, -2, -2, -3, -3, -3, -3, -3, -2, -2, -1, 0, 0, 1, 2, 3, 4, 4, 4, 4, 3, 3, 3, 3]
    y = [-4, -3, -2, -2, -1, 0, 1, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 1, 0, -1, -1, -2, -3, -4]

    mid = [0, 0]

    z = []

    for i in range(len(x)):
        z.append(math.hypot((y[i]-mid[1]), (x[i]-mid[0])))


nptime = timeit.Timer(nphypot)

print(nptime.timeit(number=1))

pytime = timeit.Timer(pyhypot)

print(pytime.timeit(number=1))

