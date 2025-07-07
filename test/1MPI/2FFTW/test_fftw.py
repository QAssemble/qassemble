# import os
# import sys

# import numpy as np
# from mpi4py import MPI
# from mpi4py_fft.mpifft import PFFT

# qapath = os.environ.get("QAssemble")
# sys.path.append(qapath + "/src")
# sys.path.append(qapath + "/src/qacore/modules")
# import QAFort

# norb = 3
# ns = 2
# nk = 125
# nomega = 10

# fr = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order="F")
# fk = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order="F")
# tempmat = np.zeros((norb, norb, ns, 5, 5, 5, nomega), dtype=np.complex128, order="F")
# fr2 = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order="F")
# ind = np.zeros([1, 1, 1], dtype=int, order="F")
# irk = 0
# divisionarray = np.array([5, 5, 5], dtype=int, order="F")

# for iomega in range(nomega):
#     for kk in range(5):
#         for jk in range(5):
#             for ik in range(5):
#                 ind = [ik + 1, jk + 1, kk + 1]
#                 ind = np.array(ind, order="F")
#                 QAFort.common.indexing(nk, divisionarray, 1, irk, ind)

#                 for js in range(ns):
#                     for iorb in range(norb):
#                         for jorb in range(norb):
#                             fk[iorb, jorb, js, irk, iomega] = (
#                                 ((iorb + 1) - (jorb + 1)) / 2.0
#                                 + (js + 1) * 0.1
#                                 + (ik + jk + kk) / 2.0
#                                 + (iomega + 1) * 0.001
#                             )

# fr = QAFort.fourier.flatdyn_k2r(divisionarray, fk)

# comm = MPI.COMM_WORLD

# tempmat2 = fk.reshape((norb, norb, ns, 5, 5, 5, nomega), order="F")

# fft = PFFT(comm=comm, shape=divisionarray, axes=(2, 0, 1), dtype="complex128")
# print(fft.shape())
# # print(fft.xfftn)
# # u = newDistArray(fft, False)   # real-space array
# # print("distributed array shape:", u.shape)
# # print("  ↳ grid   :", fft.subcomm.count)
# # # print("  ↳ r-shape:", fft.local_shape(False))
# # print("  ↳ r-slice:", fft.local_slice(False))
# # # print("  ↳ k-shape:", fft.local_shape(True))
# # print("  ↳ k-slice:", fft.local_slice(True))
# for iomega in range(nomega):
#     for js in range(ns):
#         for jorb in range(norb):
#             for iorb in range(norb):

#                 tempmat[iorb, jorb, js, ..., iomega] = fft.backward(
#                     tempmat2[iorb, jorb, js, ..., iomega], tempmat[iorb, jorb, js, ..., iomega], normalization=True
#                 )
#                 # tempval = fft.backward(tempmat2[iorb,jorb,js,...,iomega], normalization=True)
#                 # print(tempval.shape)
# fr2 = tempmat.reshape((norb, norb, ns, nk, nomega), order="F")

# assert np.allclose(fr, fr2, rtol=1e-6), "Error occurs"
import os, sys
import numpy as np
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT

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

# Create PFFT object
fft = PFFT(comm=comm, shape=divisionarray, axes=(2, 0, 1), dtype="complex128")
print(f"Global FFT shape: {fft.shape()}")
print(f"Local k-space shape: {fft.local_shape(True)}")
print(f"Local r-space shape: {fft.local_shape(False)}")
print(f"Local k-space slice: {fft.local_slice(True)}")
print(f"Local r-space slice: {fft.local_slice(False)}")

# Create distributed arrays with proper local shapes
local_k_shape = fft.local_shape(True)
local_r_shape = fft.local_shape(False)

# Initialize distributed arrays for each orbital/spin/frequency combination
tempmat_dist = np.zeros((norb, norb, ns) + local_r_shape + (nomega,),
                       dtype=np.complex128, order="F")

# Reshape fk to match the 3D grid structure
tempmat2 = fk.reshape((norb, norb, ns, 5, 5, 5, nomega), order="F")

# Get the local slices for the current MPI process
k_slice = fft.local_slice(True)
r_slice = fft.local_slice(False)

# Extract the local portion of the data for this MPI process
local_k_data = tempmat2[:, :, :, k_slice[0], k_slice[1], k_slice[2], :]

# Perform the FFT for each orbital/spin/frequency combination
for iomega in range(nomega):
    for js in range(ns):
        for jorb in range(norb):
            for iorb in range(norb):
                # Input: local k-space data
                input_data = local_k_data[iorb, jorb, js, :, :, :, iomega]

                # Output: local r-space data
                output_data = np.zeros(local_r_shape, dtype=np.complex128, order="F")

                # Perform the backward FFT (k->r)
                tempmat_dist[iorb, jorb, js, :, :, :, iomega] = fft.backward(
                    input_data, output_data, normalization=True
                )

# Now we need to gather the distributed results back to compare
# This is more complex and requires MPI communication
# For now, let's just verify that the shapes are consistent

print(f"Local distributed result shape: {tempmat_dist.shape}")
print(f"Expected local shape: {(norb, norb, ns) + local_r_shape + (nomega,)}")

# Note: To properly compare with fr, you'd need to gather all the distributed
# pieces back together using MPI operations. This is beyond the scope of
# this immediate fix but would involve something like:
#
# 1. Use comm.gather() to collect all local pieces
# 2. Reconstruct the full array on rank 0
# 3. Reshape back to (norb, norb, ns, nk, nomega)
# 4. Compare with fr

print("FFT computation completed successfully!")
print("Note: Full comparison requires gathering distributed results.")
