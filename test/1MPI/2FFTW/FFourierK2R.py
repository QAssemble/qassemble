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

    glatk = np.zeros((norb, norb, ns, nk, nw), dtype=np.complex128, order='F')

    for iw in range(nw):
        for ik in range(nk):
            for js in range(NSpin):
                for jorb in range(norb):
                    for iorb in range(norb):
                        glatk[iorb, jorb, js, ik, iw] = np.sin(np.linalg.norm(crystal.kpoint[ik]))*1j + np.cos(np.linalg.norm(crystal.kpoint[ik]))
    
    glatr = flatdyn.K2R(glatk)

    glatk = glatk.reshape((norb, norb, ns, KGrid[0], KGrid[1], KGrid[2], nw), order='F')

    tempmat = flatdynmpi.K2R(glatk)
    glatr2 = np.zeros_like(glatk,dtype=np.complex128, order='F')
    # glatk2 = np.zeros_like(glatk,dtype=np.complex128, order='F')
    flatdynmpi.commw.Allreduce(tempmat, glatr2, op=MPI.SUM)

    glatr_check = glatr2.reshape((norb, norb, ns, nk, nw), order='F')
    # tempmat2 = flatdynmpi.R2K(glatr)

    # flatdynmpi.commw.Allreduce(tempmat2, glatk2, op=MPI.SUM)
    # glatk_check = glatk2.reshape((norb, norb, ns, nk, nw), order='F')

    if (comm.Get_rank() == 0):
        print('Checking the results of K2R')

        for iw in range(nw):
            for ik in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            err = glatr[iorb, jorb, js, ik, iw] - glatr_check[iorb, jorb, js, ik, iw]
                            if (abs(err) > 1.0e-6):
                                print(iorb, jorb, js, ik, iw, abs(err), glatr[iorb, jorb, js, ik, iw], glatr_check[iorb, jorb, js, ik, iw])
        
        # print('Checking the results of R2K')
        # for iw in range(nw):
        #     for ik in range(nk):
        #         for js in range(ns):
        #             for jorb in range(norb):
        #                 for iorb in range(norb):
        #                     err = glatk[iorb, jorb, js, ik, iw] - glatk_check[iorb, jorb, js, ik, iw]
        #                     if (abs(err) > 1.0e-6):
        #                         print(iorb, jorb, js, ik, iw, abs(err), glatk[iorb, jorb, js, ik, iw], glatk_check[iorb, jorb, js, ik, iw])
if __name__ == '__main__':
    main()