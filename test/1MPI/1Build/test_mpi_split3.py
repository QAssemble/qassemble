#!/usr/bin/env python
import sys
from mpi4py import MPI


def parse_args():
    """
    Parse command-line arguments:
      sys.argv[1] -> nproc_k (number of processes along nk)
      sys.argv[2] -> nproc_w (number of processes along nw)
      sys.argv[3] -> nk (global size in k-dimension)
      sys.argv[4] -> nw (global size in w-dimension)
    """
    if len(sys.argv) != 5:
        prog = sys.argv[0]
        print(f"Usage: {prog} <nproc_k> <nproc_w> <nk> <nw>")
        sys.exit(1)
    try:
        nproc_k = int(sys.argv[1])
        nproc_w = int(sys.argv[2])
        nk = int(sys.argv[3])
        nw = int(sys.argv[4])
    except ValueError:
        print("All arguments must be integers: nproc_k, nproc_w, nk, nw.")
        sys.exit(1)
    return nproc_k, nproc_w, nk, nw


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # parse layout and global sizes
    nproc_k, nproc_w, nk, nw = parse_args()

    # validate process grid
    if nproc_k * nproc_w != size:
        if rank == 0:
            print(
                f"Error: nproc_k*nproc_w = {nproc_k*nproc_w}, but MPI world size = {size}"
            )
        sys.exit(1)

    # ensure divisibility of work
    #   if nk % nproc_k != 0 or nw % nproc_w != 0:
    #       if rank == 0:
    #           print("Error: nk must be divisible by nproc_k and nw must be divisible by nproc_w.")
    #       sys.exit(1)

    # compute each rank's 2D coordinates
    k_idx = rank // nproc_w
    w_idx = rank % nproc_w

    # compute each rank's local index ranges
    k_per = nk // nproc_k
    w_per = nw // nproc_w
    k0, k1 = k_idx * k_per, (k_idx + 1) * k_per
    w0, w1 = w_idx * w_per, (w_idx + 1) * w_per

    # build sub-communicators
    comm_k = comm.Split(color=k_idx, key=w_idx)
    comm_w = comm.Split(color=w_idx, key=k_idx)

    # report status
    print(
        f"[rank {rank:>2}] coords=(k={k_idx},w={w_idx}), "
        f"k_range=[{k0}:{k1}), w_range=[{w0}:{w1})  "
        f"sizes: comm_k={comm_k.Get_size()}, comm_w={comm_w.Get_size()}"
    )


if __name__ == "__main__":
    main()
