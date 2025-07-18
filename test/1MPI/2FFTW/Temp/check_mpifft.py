import numpy as np
import matplotlib.pyplot as plt
import os, sys
qapath = os.environ.get("QAssemble")
sys.path.append(qapath+'/src')
# from qacore.FPathDyn import *
# from qacore.CorrelationFunction import CorrelationFunction
# from qacore.Crystal import Crystal
from mpi4py import MPI
from qacore.MPIManager import *
N = (10, 10, 10)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpifft = MPIFFT(comm=comm, shape=N)

if (rank == 0):
    print(f'local shapef : {mpifft.localshapef}')
    print(mpifft.slicef)
    print(f'local shapeb : {mpifft.localshapeb}')
    print(mpifft.sliceb)