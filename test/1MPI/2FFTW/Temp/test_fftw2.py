import os, sys
import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

#N = np.array([10, 10, 10], dtype=int)
N = np.array([128, 128, 128], dtype=int)
# axes = (0, 1, 2)
axes = (1, 0, 2)
comm = MPI.COMM_WORLD
fft = PFFT(comm, N, axes=axes, dtype="complex128")
# print(fft.grid)
#fft = PFFT(comm, N, axes=(2, 0, 1), dtype="complex128")
u = newDistArray(fft, False)
print(u.shape)
u[:] = np.random.random(u.shape).astype(u.dtype)

u_hat = fft.forward(u)
uj = np.zeros_like(u)
uj = fft.backward(u_hat, uj)

assert np.allclose(uj, u)

print(comm.Get_rank(), u.shape)
