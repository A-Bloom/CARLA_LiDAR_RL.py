import numpy as np
a = np.zeros((2, 2))
b = np.ones((2, 2))
c = np.array([[2, 2], [2, 2]])

print(np.dstack((a, b, c)))
print(np.stack((a, b, c)))

