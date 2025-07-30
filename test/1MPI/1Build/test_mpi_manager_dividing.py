#!/usr/bin/env python3
import os
import sys
import numpy as np
from mpi4py import MPI
QApath = os.environ.get("QAssemble")
sys.path.append(QApath + "/src")
from qacore.MPIManager import FLatDynMPI, MPIManager

def main():

    nprock = int(sys.argv[1])
    nprocw = int(sys.argv[2])

    norb = 2
    ns = 2
    nk = 20
    nw = 20
    A = np.zeros((norb, norb, ns, nk, nw), dtype=np.complex128)
    for iw in range(nw):
        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        A[iorb, jorb, js, ik, iw] = ((iorb + 1) - (jorb + 1))/2.0 + (js + 1) * 0.1 + ik/2 + (iw + 1) * 0.01j
    
    comm = MPI.COMM_WORLD
    flatdynmpi = FLatDynMPI(comm=comm)
    commk, commw = flatdynmpi.Split(nprock=nprock, nprocw=nprocw, A=A)

    k0, k1 = flatdynmpi.klocalrange
    w0, w1 = flatdynmpi.wlocalrange

    local_bloc = A[:, :, :, k0:k1, w0:w1]

    expected_shape = (norb, norb, ns, k1 - k0, w1 - w0)
    assert local_bloc.shape == expected_shape, (
        f"Rank {flatdynmpi.rank}: local block shape {local_bloc.shape} != expected {expected_shape}"
    )
    print(f"Rank {flatdynmpi.rank}: local block shape {local_bloc.shape} 7= expected {expected_shape}")
    subk = flatdynmpi.submatrixk
    subw = flatdynmpi.submatrixw
    all_subk = comm.gather(subk, root=0)
    all_subw = comm.gather(subw, root=0)
    all_localk = comm.gather((k0, k1), root=0)
    all_localw = comm.gather((w0, w1), root=0)

    if flatdynmpi.rank == 0:
        # Verify consistency of global lists
        for idx, sk in enumerate(all_subk):
            assert sk == subk, f"Rank {idx}: inconsistent submatrixk"
        for idx, sw in enumerate(all_subw):
            assert sw == subw, f"Rank {idx}: inconsistent submatrixw"

        # Check full coverage without overlap
        covered = set()
        for ik, k_range in enumerate(subk):
            for iw, w_range in enumerate(subw):
                k0g, k1g = k_range
                w0g, w1g = w_range
                for kk in range(k0g, k1g):
                    for ww in range(w0g, w1g):
                        covered.add((kk, ww))
        expected = set((kk, ww) for kk in range(nk) for ww in range(nw))
        assert covered == expected, "Global coverage mismatch"

        print(f"[Test passed] nprock={nprock}, nprocw={nprocw}, nk={nk}, nw={nw}")

        

if __name__ == "__main__":
    main()