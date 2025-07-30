#!/usr/bin/env python3
"""
Simple debug script to show PFFT slicing
"""

from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import numpy as np


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print(f"Process {rank} of {size} starting...")

    # Create your 10x10 matrix
    matrix = np.zeros((10, 10), dtype=np.complex128)
    for i in range(10):
        for j in range(10):
            matrix[i, j] = (j - i) % 10

    if rank == 0:
        print("Original matrix:")
        print(matrix.real.astype(int))
        print()

    # Create PFFT
    fft = PFFT(comm, shape=(10, 10), dtype=np.complex128)

    # Create local array
    u = newDistArray(fft, forward_output=False)

    # Check what this process got
    my_slice = fft.local_slice(forward_output=False)

    print(f"Process {rank}:")
    print(f"  My slice: {my_slice}")
    print(f"  My array shape: {u.shape}")

    if u.size > 0:
        # Get the slice boundaries
        row_slice = my_slice[0]
        col_slice = my_slice[1]

        # Extract start and end indices
        if isinstance(row_slice, slice):
            r_start, r_end = row_slice.start, row_slice.stop
        else:
            r_start, r_end = row_slice, row_slice + 1

        if isinstance(col_slice, slice):
            c_start, c_end = col_slice.start, col_slice.stop
        else:
            c_start, c_end = col_slice, col_slice + 1

        print(f"  I get rows {r_start} to {r_end-1}")
        print(f"  I get cols {c_start} to {c_end-1}")

        # Fill my local array
        u[:, :] = matrix[r_start:r_end, c_start:c_end]

        print(f"  My local data:")
        for i in range(u.shape[0]):
            row_data = u[i, :].real.astype(int)
            global_row = r_start + i
            print(f"    Row {i} (global {global_row}): {row_data}")
    else:
        print(f"  I have no data!")

    print()


if __name__ == "__main__":
    main()
