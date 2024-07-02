import numpy as np

words = np.array(["one","two","three","four"])
mask = np.array([1,2,4])-1
print(words[mask])
