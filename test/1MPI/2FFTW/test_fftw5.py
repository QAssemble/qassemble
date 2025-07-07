#!/usr/bin/env python3
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

# Original arrays for comparison
fr = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order="F")
fk = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order="F")
fr2 = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order="F")
ind = np.zeros([1, 1, 1], dtype=int, order="F")
irk = 0
divisionarray = np.array([5, 5, 5], dtype=int, order="F")

# Fill the k-space data
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

# Get reference result from your Fortran routine
fr = QAFort.fourier.flatdyn_k2r(divisionarray, fk)

comm = MPI.COMM_WORLD

# Create PFFT object for the 3D grid only
fft = PFFT(comm=comm, shape=divisionarray, axes=(0, 1, 2), dtype="complex128")
print(f"FFT created for shape: {fft.shape()}")

# Create distributed arrays using newDistArray
# False = input array (k-space), True = output array (r-space)
k_array = newDistArray(fft, forward_output=False)  # k-space distributed array
r_array = newDistArray(fft, forward_output=True)   # r-space distributed array

print(f"Local k-space array shape: {k_array.shape}")
print(f"Local r-space array shape: {r_array.shape}")

# Now we need to handle the orbital/spin/frequency dimensions separately
# since PFFT only handles the spatial (5x5x5) grid

# Reshape fk to separate the spatial grid from other dimensions
tempmat2 = fk.reshape((norb, norb, ns, 5, 5, 5, nomega), order="F")
tempmat = np.zeros((norb, norb, ns, 5, 5, 5, nomega), dtype=np.complex128, order="F")

# Perform FFT for each orbital/spin/frequency combination
for iomega in range(nomega):
    for js in range(ns):
        for jorb in range(norb):
            for iorb in range(norb):
                # Copy the 3D spatial data to the distributed k-array
                k_array[:] = tempmat2[iorb, jorb, js, :, :, :, iomega]

                # Perform the backward FFT (k->r)
                r_array = fft.backward(k_array, normalize=True)

                # Store the result
                tempmat[iorb, jorb, js, :, :, :, iomega] = r_array[:]

# Reshape back to the original format
fr2 = tempmat.reshape((norb, norb, ns, nk, nomega), order="F")

# Test if they match
try:
    assert np.allclose(fr, fr2, rtol=1e-6), "Error occurs"
    print("SUCCESS: PFFT results match Fortran routine!")
except AssertionError:
    print("Mismatch detected. Checking differences...")
    diff = np.abs(fr - fr2)
    print(f"Max absolute difference: {np.max(diff)}")
    print(f"Max relative difference: {np.max(diff / (np.abs(fr) + 1e-16))}")

    # Check if it's just a scaling or phase issue
    ratios = fr.flat / fr2.flat
    valid_ratios = ratios[np.abs(fr.flat) > 1e-10]
    if len(valid_ratios) > 0:
        unique_ratios = np.unique(np.round(valid_ratios, 6))
        print(f"Unique ratios (fr/fr2): {unique_ratios[:10]}")  # First 10 unique ratios
