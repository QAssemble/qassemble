#!/usr/bin/env python3

import os, sys
import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

#N = np.array([10, 10, 10], dtype=int)
N = np.array([128, 128, 1], dtype=int)
axes = (0, 1, 2)
comm = MPI.COMM_WORLD
fft = PFFT(comm, N, axes=axes, dtype="complex128",grid=(8, 1, 1))
# fft = PFFT(comm, N, axes=(2, 0, 1), dtype=np.float64)
u = newDistArray(fft, False)
u[:] = np.random.random(u.shape).astype(u.dtype)

u_hat = fft.forward(u)
uj = np.zeros_like(u)
uj = fft.backward(u_hat, uj)

assert np.allclose(uj, u)

print(comm.Get_rank(), u.shape)
