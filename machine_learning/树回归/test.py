import numpy as np



a = [[1, 2],
     [3, 4],
     [5, 6]]
a = np.mat(a)
print(np.mean(a[:, -1]))
print(np.mean(a))