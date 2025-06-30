from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import numpy as np
import sys, os
qapath = os.environ.get('QAssemble')
sys.path.append(qapath+'/src/qacore/modules')
import QAFort

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
norb = 3
ns = 2
nk = 125  # This will be your FFT dimension
nomega = 10

# Original data shape
shape_fk = (norb, norb, ns, nk, nomega)

# Generate fk data exactly as your original test
fk = np.zeros(shape_fk, dtype=complex, order='F')
irk = 0
ind = np.zeros([1,1,1],dtype=int,order='F')
divisionarray = np.array([5,5,5],dtype=int,order='F')

for iomega in range(nomega):
    for kk in range(5):
        for jk in range(5):
            for ik in range(5):
                ind = [ik + 1, jk + 1, kk + 1]
                ind = np.array(ind, order='F')
                QAFort.common.indexing(nk, divisionarray, 1, irk, ind)
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            fk[iorb, jorb, js, irk, iomega] = (
                                ((iorb + 1) - (jorb + 1)) / 2.0
                                + (js + 1) * 0.1
                                + (ik + jk + kk) / 2.0
                                + (iomega + 1) * 0.001
                            )
# Define FFT along axis=3 only
# norb, ns, nk, nomega = 3, 2, 125, 10
fk = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order='F')

# Set up FFT along nk (4th axis)
fft = PFFT(comm, (nk,), axes=(0,), dtype=np.complex128, grid=(size,))

# Get local shape and slice
local_slice = fft.forward.input_array.local_slice()
local_nk = fft.forward.input_array.shape[0]
offset = local_slice[0].start

# Allocate local distributed arrays
fk_local = np.zeros((norb, norb, ns, local_nk, nomega), dtype=np.complex128)

# Distribute original data to local arrays
for iomega in range(nomega):
    for js in range(ns):
        for jorb in range(norb):
            for iorb in range(norb):
                fk_local[iorb, jorb, js, :, iomega] = fk[iorb, jorb, js, offset:offset + local_nk, iomega]

# Perform FFT
fr_local = np.zeros_like(fk_local)
fft.backward(fk_local, fr_local, normalize=True)

# Gather if needed
fr_full = None
if comm.rank == 0:
    fr_full = np.zeros((norb, norb, ns, nk, nomega), dtype='complex128')

# Use MPI gather explicitly to gather fr_local back to rank 0
comm.Gather(fr_local, fr_full, root=0)

if comm.rank == 0:
    print("Gathered FFT result ready for comparison")

