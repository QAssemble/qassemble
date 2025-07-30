#!/usr/bin/env python3

import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # Create your specific 10x10 matrix
    a = np.zeros((10, 10, 10), dtype=np.complex128)
    for i in range(10):
        for j in range(10):
            a[i, j] = (j - i) % 10

    if rank == 0:
        print("Original 10x10 matrix:")
        for i in range(10):
            print(f"Row {i}: {a[i].real.astype(int)}")
        print()

    # Create PFFT object
    fft = PFFT(comm, shape=(10, 10, 10), dtype=np.complex128)

    # Create distributed array
    u = newDistArray(fft, forward_output=False)

    # Get slice information
    my_slice = fft.local_slice(forward_output=False)

    # Extract slice details
    row_slice = my_slice[0]
    col_slice = my_slice[1]

    if isinstance(row_slice, slice):
        row_start = row_slice.start
        row_end = row_slice.stop
    else:
        row_start = row_slice
        row_end = row_slice + 1

    if isinstance(col_slice, slice):
        col_start = col_slice.start
        col_end = col_slice.stop
    else:
        col_start = col_slice
        col_end = col_slice + 1

    print(f"Process {rank}:")
    print(f"  Gets rows {row_start} to {row_end-1} (slice [{row_start}:{row_end}])")
    print(f"  Gets cols {col_start} to {col_end-1} (slice [{col_start}:{col_end}])")
    print(f"  Local array shape: {u.shape}")

    # Fill local array with corresponding data
    if u.size > 0:
        u[:, :, :] = a[row_start:row_end, col_start:col_end, :]

        print(f"  Local data:")
        for i in range(u.shape[0]):
            local_row = u[i, :].real.astype(int)
            global_row_idx = row_start + i
            print(f"    Local row {i} (global row {global_row_idx}): {local_row}")
    else:
        print(f"  No data assigned to this process")

    print()


if __name__ == "__main__":
    main()
