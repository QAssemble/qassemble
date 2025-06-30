import os
import sys

import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

qapath = os.environ.get("QAssemble")
sys.path.append(qapath + "/src")
sys.path.append(qapath + "/src/qacore/modules")
import QAFort

norb = 3
ns = 2
nk = 125
nomega = 10

fr = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order="F")
fk = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order="F")
tempmat = np.zeros((norb, norb, ns, 5, 5, 5, nomega), dtype=np.complex128, order="F")
fr2 = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order="F")
ind = np.zeros([1, 1, 1], dtype=int, order="F")
irk = 0
divisionarray = np.array([5, 5, 5], dtype=int, order="F")

for iomega in range(nomega):
    for kk in range(5):
        for jk in range(5):
            for ik in range(5):
                ind = [ik + 1, jk + 1, kk + 1]
                ind = np.array(ind, order="F")
                QAFort.common.indexing(nk, divisionarray, 1, irk, ind)

                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            fk[iorb, jorb, js, irk, iomega] = (
                                ((iorb + 1) - (jorb + 1)) / 2.0
                                + (js + 1) * 0.1
                                + (ik + jk + kk) / 2.0
                                + (iomega + 1) * 0.001
                            )

fr = QAFort.fourier.flatdyn_k2r(divisionarray, fk)

comm = MPI.COMM_WORLD

tempmat2 = fk.reshape((norb, norb, ns, 5, 5, 5, nomega), order="F")

fft = PFFT(comm=comm, shape=divisionarray, axes=(0, 1, 2), dtype="complex")

for iomega in range(nomega):
    for js in range(ns):
        for jorb in range(norb):
            for iorb in range(norb):
                tempmat[iorb, jorb, js, ..., iomega] = fft.backward(
                    tempmat2[iorb, jorb, js, ..., iomega], normalization=True
                )

fr2 = tempmat.reshape((norb, norb, ns, nk, nomega), order="F")

assert np.allclose(fr, fr2, rtol=1e-6), "Error occurs"
