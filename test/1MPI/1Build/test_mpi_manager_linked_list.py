#!/usr/bin/env python3
import os, sys
QApath = os.environ.get("QAssemble")
sys.path.append(QApath + "/src")
from qacore.MPIManager import MPIManager, FLatDynMPI
# import MPIManager
from mpi4py import MPI

# Use MPI.COMM_WORLD by default
a = FLatDynMPI()
b = FLatDynMPI()
print("head:", FLatDynMPI.head)
print("tail:", FLatDynMPI.tail)
print("a.next is b:", a.next is b)
print("b.next is None:", getattr(b, "next", None) is None)
print("comm same:", a.comm == b.comm)
print("rank:", a.rank, b.rank, a.size, b.size)
