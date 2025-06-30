# ------------------ test_mpi_fftw2.py ------------------

from mpi4py import MPI
from mpi4py_fft import PFFT
import numpy as np
import sys, os

# Make sure QAFort is on the PYTHONPATH:
qapath = os.environ.get('QAssemble')
sys.path.append(qapath + '/src/qacore/modules')
import QAFort

comm = MPI.COMM_WORLD
size = comm.Get_size()

# -------------------------------------------
# 1) Problem parameters (the same as your original code)
norb   = 3
ns     = 2
nk     = 125    # the global FFT length along that axis (4th axis)
nomega = 10
ind = np.zeros([1,1,1],dtype=int,order='F')
divisionarray = np.array([5,5,5],dtype=int,order='F')
# We assume 'divisionarray' and 'irk' are already defined by your environment,
# since QAFort.common.indexing(nk, divisionarray, 1, irk, ind) uses them.
# (Adjust these two lines as needed. E.g.:
#    divisionarray = ...
#    irk = np.empty(1, dtype=np.int32)
# )

# ------------------------------------------------
# 2) Allocate the full fk array on rank 0 so we can fill it exactly as you did
if comm.rank == 0:
    fk_full = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order='F')
else:
    fk_full = None
irk = 0
# On rank 0, fill fk_full with your nested loop + QAFort.common.indexing:
if comm.rank == 0:
    for iomega in range(nomega):
        for kk in range(5):
            for jk in range(5):
                for ik in range(5):
                    ind = np.array([ik+1, jk+1, kk+1], order='F')

                    # Call QAFort.common.indexing to get irk (0 ≤ irk < nk)
                    # irk_buf = np.empty(1, dtype=np.int32)
                    QAFort.common.indexing(nk, divisionarray, 1, irk, ind)
                    # irk = int(irk_buf[0])

                    for js in range(ns):
                        for jorb in range(norb):
                            for iorb in range(norb):
                                fk_full[iorb, jorb, js, irk, iomega] = (
                                    ((iorb + 1) - (jorb + 1)) / 2.0
                                    + (js + 1) * 0.1
                                    + (ik + jk + kk) / 2.0
                                    + (iomega + 1) * 0.001
                                )

# ------------------------------------------------
# 3) Build a 1D PFFT that decomposes exactly across 'nk' with 'size' ranks:
fft = PFFT(comm,
           shape = [nk],      # the global FFT length (nk)
           axes  = (0,),      # do FFT only along that single axis
           dtype = np.complex128,
           grid  = (size,)    # “all size ranks along this 1D dimension”
          )

# 4) Figure out each rank’s “slab” in the global nk-range:
local_slice = fft.forward.input_array.local_slice()
local_nk    = fft.forward.input_array.shape[0]  # how many k-points this rank owns
offset      = local_slice[0].start              # the global index (in [0..nk-1]) where this slab begins

# 5) Allocate local buffers on each rank:
#    fk_local has shape (norb, norb, ns, local_nk, nomega)
fk_local  = np.zeros((norb, norb, ns, local_nk, nomega), dtype=np.complex128)
fr2_local = np.zeros_like(fk_local)

# 6) Scatter slabs of fk_full from rank 0 → each rank’s fk_local
if comm.rank == 0:
    # On rank 0, slice off each rank’s chunk and Send it
    for r in range(size):
        sl = fft.forward.input_array.local_slice(r)  # a tuple like (slice(start_r, stop_r, None),)
        start_r = sl[0].start
        stop_r  = sl[0].stop
        slab = fk_full[:, :, :, start_r:stop_r, :]    # shape = (norb, norb, ns, local_nk, nomega)

        if r == 0:
            fk_local[...] = slab.copy()
        else:
            comm.Send(slab, dest=r, tag=77)
else:
    # On all other ranks, receive into fk_local
    comm.Recv(fk_local, source=0, tag=77)

# 7) mpi4py_fft expects the “distributed axis” to be axis=0 of the local array,
#    so we must reorder dimensions. Currently fk_local.shape = (norb,norb,ns,local_nk,nomega).
#    We want “local_nk” up front:
fk_work = np.transpose(fk_local, (3, 0, 1, 2, 4))
# Now fk_work.shape = (local_nk, norb, norb, ns, nomega).

fr2_work = np.zeros_like(fk_work)

# 8) Run the inverse FFT (k → r) along axis=0 of fk_work:
fft.backward(fk_work, fr2_work, normalize=True)

# 9) Put the “real‐space” data back into the original layout:
fr2_local[:] = np.transpose(fr2_work, (1, 2, 3, 0, 4))
# Now fr2_local.shape = (norb, norb, ns, local_nk, nomega)

# 10) Gather all the slabs of fr2_local back onto rank 0 into fr2_full:
if comm.rank == 0:
    fr2_full = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128)
else:
    fr2_full = None

if comm.rank == 0:
    # Rank 0 takes its own slab first:
    fr2_full[:, :, :, offset:offset+local_nk, :] = fr2_local.copy()
    # Then receive from each other rank
    for r in range(1, size):
        sl = fft.forward.input_array.local_slice(r)
        start_r = sl[0].start
        stop_r  = sl[0].stop
        buffer = np.empty((norb, norb, ns, stop_r - start_r, nomega), dtype=np.complex128)
        comm.Recv(buffer, source=r, tag=99)
        fr2_full[:, :, :, start_r:stop_r, :] = buffer
else:
    # Other ranks just send their slab
    comm.Send(fr2_local, dest=0, tag=99)

# ------------------------------------------------
# 11) On rank 0, compare the result with QAFort.fourier.flatdyn_k2r
if comm.rank == 0:
    # Serial FT for comparison
    fr_qafort = QAFort.fourier.flatdyn_k2r(divisionarray, fk_full)

    # fr2_full is the MPI‐FFTW result
    diff = np.linalg.norm(fr_qafort - fr2_full)
    print(f"[rank 0] ‖QAFort – MPI‐FFTW‖ = {diff:.3e}")
    if diff < 1e-12:
        print("✅ The results match to machine precision.")
    else:
        print("⚠️ Warning: numerical difference is larger than expected.")

# End of test_mpi_fftw2.py
