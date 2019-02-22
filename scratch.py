

import numpy as np
import cupy as cp
import time

na = np.random.rand(1000, 1000)
nb = np.random.rand(1000, 1000)
num_iter = 100

start = time.time()
for i in range(0, num_iter):
    np.matmul(na, nb)
end = time.time()

print('numpy time: {}'.format(end-start))


ca = cp.random.rand(1000, 1000)
cb = cp.random.rand(1000, 1000)

start2 = time.time()
for i in range(0, num_iter):
    cp.matmul(ca, cb)
end2 = time.time()

print('cupy time: {}'.format(end2 - start2))
