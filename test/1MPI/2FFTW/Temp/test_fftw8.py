#!/usr/bin/env python3

import os
import sys

import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

N = np.array([10, 10, 10], dtype=int)
axes = (0, 1, 2)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create PFFT object - adjust grid based on number of processes
nprocs = comm.Get_size()
if nprocs == 4:
    fft = PFFT(comm, N, axes=axes, dtype="complex128", grid=(2, 2, 1))
elif nprocs == 8:
    fft = PFFT(comm, N, axes=axes, dtype="complex128", grid=(4, 2, 1))
else:
    fft = PFFT(comm, N, axes=axes, dtype="complex128")  # Auto grid

# IMPORTANT: Get local slices for PHYSICAL space (False), not spectral space (True)
local_physical = fft.local_slice(False)  # For physical space
local_spectral = fft.local_slice(True)  # For spectral space

print(f"Rank {rank}: Physical space local slices - {local_physical}")
print(f"Rank {rank}: Spectral space local slices - {local_spectral}")

# Create the full data array on all ranks
full_data = np.arange(np.prod(N), dtype=np.complex128).reshape(N)
full_data = full_data + 1j * full_data  # Make it complex

# Create distributed array in PHYSICAL space
u = newDistArray(fft, False)  # False = physical space
print(f"Rank {rank}: Local shape of u (physical): {u.shape}")

# Fill with local portion of full_data using PHYSICAL space slices
u[:] = full_data[local_physical[0], local_physical[1], local_physical[2]]

print(f"Rank {rank}: First few values of local u: {u.flat[:5]}")

# Perform forward FFT
u_hat = fft.forward(u)
print(f"Rank {rank}: Local shape of u_hat (spectral): {u_hat.shape}")

# Perform backward FFT
uj = np.zeros_like(u)
uj = fft.backward(u_hat, uj)

# Verify the transform is correct
assert np.allclose(uj, u)
print(f"Rank {rank}: Forward/backward transform verified!")

# GATHERING - Reconstruct the full array on rank 0
gathered_u = u.get((slice(None), slice(None), slice(None)))

if rank == 0:
    print(f"\n{'='*50}")
    print(f"Rank 0: Gathered u shape: {gathered_u.shape}")
    print(f"Rank 0: First 10 values of gathered u: {gathered_u.flat[:10]}")
    print(f"Rank 0: First 10 values of original: {full_data.flat[:10]}")
    print(f"Rank 0: Arrays match: {np.allclose(gathered_u, full_data)}")
    print(f"{'='*50}\n")

# Also gather the spectral array to see the transformed data
gathered_u_hat = u_hat.get((slice(None), slice(None), slice(None)))
if rank == 0:
    print(f"Rank 0: Gathered u_hat shape: {gathered_u_hat.shape}")

# Alternative shorter syntax for gathering
gathered_short = u.get((...,))  # Same as (slice(None), slice(None), slice(None))
if rank == 0:
    print(f"Rank 0: Short syntax also works: {np.allclose(gathered_short, full_data)}")

# Memory-efficient approach (only rank 0 creates full_data)
if rank == 0:
    print(f"\n{'='*30} Memory Efficient Method {'='*30}")

u_efficient = newDistArray(fft, False)

# Method 1: Using scatter (more efficient for large arrays)
if rank == 0:
    # Only rank 0 creates the full array
    full_data_efficient = np.arange(np.prod(N), dtype=np.complex128).reshape(N)
    full_data_efficient = full_data_efficient + 1j * full_data_efficient

    # Prepare data for each rank
    for i in range(comm.Get_size()):
        # Get the local slice for process i in physical space
        local_i = fft.local_slice(False, rank=i)
        local_data = full_data_efficient[local_i[0], local_i[1], local_i[2]]
        if i == 0:
            u_efficient[:] = local_data
        else:
            comm.Send(local_data, dest=i, tag=11)
elif rank > 0:
    # Other ranks receive their portion
    comm.Recv(u_efficient, source=0, tag=11)

# Verify the efficient method gives the same result
gathered_efficient = u_efficient.get((slice(None), slice(None), slice(None)))
if rank == 0:
    print(
        f"Memory-efficient method also works: {np.allclose(gathered_efficient, full_data_efficient)}"
    )

comm.Barrier()
print(f"\nRank {rank}: All operations completed successfully!")
