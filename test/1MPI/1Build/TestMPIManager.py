import os, sys
import pprint
import numpy as np
from mpi4py import MPI
qapath = os.environ.get("QAssemble")
sys.path.append(qapath + "/src")
from qacore.MPIManager import FLatDynMPI, MPIManager

comm = MPI.COMM_WORLD
mpimanager = MPIManager(comm)

nprock = 3
nprocw = 2
nk = 10
nw = 10
ntau = 10

flatdynmpi = FLatDynMPI(
    crystal=None,
    ftgrid=None,
    nk=nk,
    nw=nw,
    ntau=ntau,
    nprock=nprock,
    nprocw=nprocw,
    mpimanager=mpimanager,
)

pprint.pprint(mpimanager.mpidict)

print(flatdynmpi.nodedict['submatrixk'])

for key in flatdynmpi.nodedict['submatrixk']:
    a = list(range(key[0], key[1]))
    print(f"submatrix size : {len(a)}")
    print(f"test : {key[1]- key[0]}")
