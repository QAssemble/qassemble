#!/usr/bin/env python3
import os
import sys

import numpy as np
from mpi4py import MPI

QApath = os.environ.get("QAssemble")
sys.path.append(QApath + "/src")
from qacore.MPIManager import FLatDynMPI, MPIManager

norb = 2
ns = 2
nk = 20
nw = 20
A = np.zeros((norb, norb, ns, nk, nw), dtype=np.complex128)

comm = MPI.COMM_WORLD

nprock = int(sys.argv[1])
nprocw = int(sys.argv[2])
# mgm = MPIManager(comm=comm)
flatdynmpi = FLatDynMPI(comm=comm)

commk, commw = flatdynmpi.Split(nprock=nprock, nprocw=nprocw, A=A)
# dimension = [4, 5, 6, 7]

# (a, b) = mgm.Split(dimension, A)

# commk, commw = mgm.Split(nprock=2, nprocw=2, nk=nk, nw=nw)


# print(f"Rank {flatdynmpi.rank} has commk: {commk}, commw: {commw}")
print(
    f"Rank {flatdynmpi.rank} has submatrixk: {flatdynmpi.submatrixk}, submatrixw: {flatdynmpi.submatrixw}"
)

