#!/usr/bin/env python
import sys

from mpi4py import MPI


def parse_args():
    if len(sys.argv) != 3:
        prog = sys.argv[0]
        print(f"Usage: {prog} <nproc_k> <nproc_w>")
        sys.exit(1)
    try:
        nproc_k = int(sys.argv[1])
        nproc_w = int(sys.argv[2])
    except ValueError:
        print("Both nproc_k and nproc_w must be integers.")
        sys.exit(1)
    return nproc_k, nproc_w


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 0) parse the desired 2D layout from argv
    nproc_k, nproc_w = parse_args()
    if nproc_k * nproc_w != size:
        if rank == 0:
            print(
                f"Error: nproc_k*nproc_w = {nproc_k*nproc_w}, but MPI world size = {size}"
            )
        sys.exit(1)

    # 1) compute each rank’s grid coordinates
    k_idx = rank // nproc_k
    w_idx = rank % nproc_w

    # 2) define the global problem size
    nk, nw = 10, 10
    #   if nk % nproc_k or nw % nproc_w:
    #       if rank == 0:
    #           print("Error: nk must be divisible by nproc_k and nw by nproc_w.")
    #       sys.exit(1)

    # 3) local sub‐ranges
    k_per = nk // nproc_k
    w_per = nw // nproc_w
    k0, k1 = k_idx * k_per, (k_idx + 1) * k_per
    w0, w1 = w_idx * w_per, (w_idx + 1) * w_per

    # 4) build sub‐communicators
    comm_k = comm.Split(color=k_idx, key=w_idx)
    comm_w = comm.Split(color=w_idx, key=k_idx)

    # 5) echo what each rank is doing
    print(
        f"[rank {rank:>2}] coords=(k={k_idx},w={w_idx}), "
        f"k_range=[{k0}:{k1}), w_range=[{w0}:{w1})  "
        f"sizes: comm_k={comm_k.Get_size()}, comm_w={comm_w.Get_size()}"
    )


if __name__ == "__main__":
    main()
