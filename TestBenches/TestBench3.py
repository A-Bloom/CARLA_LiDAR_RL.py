# This TestBench is a simplified version of how create_lidar_plane() takes a variable number of points coming in from
# the listener and adds them to the incoming LiDAR observation.

import numpy as np

A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
B = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])

e = np.zeros((2, 10, 2))

e[0] = np.dstack((A, B))

z = np.dstack((a, b))

e[1][:4] = z

print(z)
print(e[1])
z = z.squeeze()*2
print(z)

e[1][:2] = z[:2]
print(e[1])


