import numpy as np
import mpi4py
from mpi4py import MPI

def main(n : int = 10):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split the communicator into two groups
    # Processes with even ranks will have color 0, odd ranks will have color 1

    color = rank // n
    key = rank

    new_comm = comm.Split(color=color, key=key)
    new_rank = new_comm.Get_rank()
    new_size = new_comm.Get_size()

    print(f"Original Rank: {rank}, Original Size : {size}, New Rank: {new_rank}, Color: {color}, Key: {key}, New Size: {new_size}")
    MPI.Finalize()  # Manually finalize MPI
    return None

if __name__ == "__main__":
    mpi4py.rc.initialize = False  # Disable automatic initialization
    mpi4py.rc.finalize = False  # Disable automatic finalization
    main()
    