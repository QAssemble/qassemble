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
# False = k-space input array for backward FFT, True = r-space output array
k_array = newDistArray(fft, forward_output=False)  # k-space distributed array
r_array = newDistArray(fft, forward_output=True)   # r-space distributed array

print(f"Local k-space array shape: {k_array.shape}")
print(f"Local r-space array shape: {r_array.shape}")
# print(f"Local k-space slice: {k_array.local_slice()}")
# print(f"Local r-space slice: {r_array.local_slice()}")

# Get the local slices for the current MPI process
# Use PFFT.local_slice(forward) to obtain the distributed slice for k- and r-space
k_slice = fft.local_slice(False)
r_slice = fft.local_slice(True)

print(f"Local k-space slice: {k_slice}")
print(f"Local r-space slice: {r_slice}")
# Reshape fk to separate the spatial grid from other dimensions
# tempmat2 = fk.reshape((norb, norb, ns, 5, 5, 5, nomega), order="F")
# tempmat = np.zeros((norb, norb, ns, 5, 5, 5, nomega), dtype=np.complex128, order="F")

# # Perform FFT for each orbital/spin/frequency combination
# for iomega in range(nomega):
#     for js in range(ns):
#         for jorb in range(norb):
#             for iorb in range(norb):
#                 # Extract the LOCAL portion of the 3D spatial data for this process
#                 local_k_data = tempmat2[iorb, jorb, js, k_slice[0], k_slice[1], k_slice[2], iomega]

#                 # Copy to the distributed k-array (this should now match dimensions)
#                 k_array[:] = local_k_data

#                 # Perform the backward FFT (k->r)
#                 r_array = fft.backward(k_array, normalize=True)
#                 # print(r_array.shape)
#                 # Store the LOCAL result back to the corresponding slice
#                 tempmat[iorb, jorb, js, r_slice[0], r_slice[1], r_slice[2], iomega] = r_array[:]

# # To compare results, we need to gather the distributed data back together
# # For now, let's just check if the FFT ran without errors on this process
# print(f"Process {comm.Get_rank()}: FFT computation completed successfully!")

# # If you're running on a single process or want to gather results, you can do:
# if comm.Get_size() == 1:
#     # Single process - can compare directly
#     fr2 = tempmat.reshape((norb, norb, ns, nk, nomega), order="F")

#     try:
#         assert np.allclose(fr, fr2, rtol=1e-6), "Error occurs"
#         print("SUCCESS: PFFT results match Fortran routine!")
#     except AssertionError:
#         print("Mismatch detected. Checking differences...")
#         diff = np.abs(fr - fr2)
#         print(f"Max absolute difference: {np.max(diff)}")
#         print(f"Max relative difference: {np.max(diff / (np.abs(fr) + 1e-16))}")
# else:
#     print(f"Multi-process run - use MPI gather operations to collect full results for comparison")
#     # For multi-process, you'd need to use comm.gather() or similar to collect all pieces
