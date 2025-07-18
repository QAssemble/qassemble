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

    tempmat = np.zeros((norb, norb, ns, nk, nw), dtype=np.complex128, order='F')

    for iw in range(nw):
        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        tempmat[iorb, jorb, js, ik, iw] = (((iorb + 1) - (jorb + 1)) / 2.0 + (js + 1)*0.1 + (crystal.kpoint[ik][0]+crystal.kpoint[ik][1]+crystal.kpoint[ik][2]) / 2.0 + (iw + 1) * 0.001)

    tempmat2 = flatdyn.Inverse(tempmat)
    tempmat = tempmat.reshape((norb, norb, ns, 9, 9, 9, nw), order='F')

    tempmat3 = flatdynmpi.Inverse(tempmat)

    tempmat4 = np.zeros_like(tempmat, dtype=np.complex128, order='F')
    tempmat5 = np.zeros_like(tempmat, dtype=np.complex128, order='F')

    flatdynmpi.commw.Allreduce(tempmat3, tempmat4, op=MPI.SUM)
    flatdynmpi.commk.Allreduce(tempmat4, tempmat5, op=MPI.SUM)
    tempmat5 = tempmat5.reshape((norb, norb, ns, nk, nw),order='F')

    # if (comm.Get_rank() == 0):
    #     print(tempmat3[0, 0, 0, 0, 0])
    if (comm.Get_rank() == 0):
        for iw in range(nw):
            for ik in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            err = tempmat2[iorb, jorb, js, ik, iw] - tempmat5[iorb, jorb, js, ik, iw]
                            if (abs(err) > 1.0e-6).all():
                                print(iorb, jorb, js, ik, iw, abs(err), tempmat2[iorb, jorb, js, ik, iw], tempmat5[iorb, jorb, js, ik, iw])


if __name__ == '__main__':
    main()
