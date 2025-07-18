#!/usr/bin/env python3
import os, sys
import numpy as np
from mpi4py import MPI
qapath = os.environ.get("QAssemble")
sys.path.append(qapath + "/src")
# sys.path.append(qapath + "/src/qacore/modules")
from qacore.MPIManager import FLatDynMPI, MPIManager
from qacore.Crystal import Crystal
from qacore.FTGrid import FTGrid
from qacore.FLatDyn import FLatDyn

def parse_args():
    if len(sys.argv) != 3:
        prog = sys.argv[0]
        print(f"Usage: {prog} <nproc_k> <nproc_w>")
        sys.exit(1)
    try:
        nproc_k = int(sys.argv[1])
        nproc_w = int(sys.argv[2])
    except ValueError:
        print("Both nproc_k and nproc_w must be integers.")
        sys.exit(1)
    return nproc_k, nproc_w

def main():
    comm = MPI.COMM_WORLD
    mpimanager = MPIManager(comm=comm)

    nprock, nprocw = parse_args()
    RVec = [[10,0,0],[0,10,0],[0,0,10]]
    Basis = [[[1/2,1/2,1/2],1]]
    NSpin = 1
    SOC = False
    KGrid = [9, 9, 9]
    NElec = 1
    T = 2000
    cutoff = 100
    cry = {
        'RVec' : RVec,
        'Basis': Basis,
        'CorF' : 'F',
        'SOC' : SOC,
        'NSpin' : NSpin,
        'NElec' : NElec,
        'KGrid' : KGrid
    }
    ft = {
        'T' : T,
        'cutoff' : cutoff
    }
    crystal = Crystal(cry=cry)
    ftgrid = FTGrid(ft=ft)
    flatdyn = FLatDyn(crystal=crystal, ft=ftgrid)
    nk = len(crystal.kpoint)
    nw = len(ftgrid.omega)
    ntau = len(ftgrid.tau)
    norb = len(crystal.find)
    ns = crystal.ns
    flatdynmpi = FLatDynMPI(crystal=crystal, ftgrid=ftgrid,nprock=nprock, nprocw=nprocw, nk = len(crystal.kpoint), nw = len(ftgrid.omega), ntau = len(ftgrid.tau), mpimanager=mpimanager)

    glatt0 = np.zeros((norb, norb, ns, nk, nw), dtype=np.complex128, order='F')
    sig = np.zeros((norb, norb, ns, nk, nw), dtype=np.complex128, order='F')

    for iw in range(nw):
        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        if iorb == jorb:
                            glatt0[iorb, jorb, js, ik, iw] = 1.0/(1j*ftgrid.omega[iw] - 1 + ik + js + 0.1 + (iorb + jorb) *2)

    for iw in range(nw):
        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        sig[iorb, jorb, js, ik, iw] = 5*ik + iorb + 0.1+ iw + jorb*2+jorb
    
    # if (comm.Get_rank ==)
    tempmat = flatdyn.Dyson(glatt0, sig)
    glatt0 = glatt0.reshape((norb, norb, ns, 9, 9, 9, nw), order='F')
    sig = sig.reshape((norb, norb, ns, 9, 9, 9, nw), order='F')
    tempmat2 = flatdynmpi.Dyson(glatt0, sig)
    tempmat3 = np.zeros_like(tempmat2, dtype=np.complex128, order='F')
    tempmat4 = np.zeros_like(tempmat2, dtype=np.complex128, order='F')

    flatdynmpi.commw.Allreduce(tempmat2, tempmat3, op=MPI.SUM)
    flatdynmpi.commk.Allreduce(tempmat3, tempmat4, op=MPI.SUM)
    tempmat4 = tempmat4.reshape((norb, norb, ns, nk, nw),order='F')

    if (comm.Get_rank() == 0):
        for iw in range(nw):
            for ik in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            err = tempmat[iorb, jorb, js, ik, iw] - tempmat4[iorb, jorb, js, ik, iw]
                            if (abs(err) > 1.0e-6):
                                print(iorb, jorb, js, ik, iw, abs(err), tempmat[iorb, jorb, js, ik, iw], tempmat4[iorb, jorb, js, ik, iw])


if __name__ == '__main__':
    main()
