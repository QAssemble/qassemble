import os
import pprint
import sys

import numpy as np
from mpi4py import MPI

qapath = os.environ.get("QAssemble")
sys.path.append(qapath + "/src")

from qacore.MPIManager import FLatDynMPI, MPIManager

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprock = 2
nprocw = 2

nk1 = 4
nk2 = 10
nw1 = 10
nw2 = 20

mpimanager = MPIManager(comm=comm)

flatdynmpi = FLatDynMPI(
    crystal=None,
    ftgrid=None,
    nk=nk1,
    nw=nw1,
    nprock=nprock,
    nprocw=nprocw,
    mpimanager=mpimanager,
)

flatdynmpi2 = FLatDynMPI(
    crystal=None,
    ftgrid=None,
    nk=nk2,
    nw=nw2,
    nprock=nprock,
    nprocw=nprocw,
    mpimanager=mpimanager,
)

if rank == 0:
    pprint.pprint(mpimanager.mpidict)
    pprint.pprint(flatdynmpi.nodedict)
    pprint.pprint(flatdynmpi2.nodedict)
