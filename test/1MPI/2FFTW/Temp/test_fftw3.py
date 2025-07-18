import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

comm = MPI.COMM_WORLD  # say mpi size = 3
# “Global shape” now (10,1), but we still only FFT over axis 0
# fft = PFFT(comm, global_shape=(10, 1), axes=(0,), dtype="complex128")
size = comm.Get_size()
N = 10
fft = PFFT(comm, global_shape=(N, 1), axes=(0, 1), dtype="complex128", grid=(size, 1))
local = newDistArray(fft, real=True)

print(f"rank {comm.rank} sees local.shape = {local.shape}")
# → rank 0: (4,1), rank 1: (3,1), rank 2: (3,1)
# you can then just ignore the second dim in your math:
local = local[:, 0]
fft_data = fft.forward(local.reshape(local.shape), normalize=True)
