import numpy as np

a = np.zeros((2, 2))
b = a
print(b)

a[1][1] = 1

print(b)
