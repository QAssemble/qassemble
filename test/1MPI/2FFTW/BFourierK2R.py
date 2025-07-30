#!/usr/bin/env python3
import os, sys
import numpy as np
from mpi4py import MPI
qapath = os.environ.get("QAssemble")
sys.path.append(qapath + "/src")
# sys.path.append(qapath + "/src/qacore/modules")
from QAssemble.Src.MPIManager import BLatDynMPI, MPIManager
from QAssemble.Src.Crystal import Crystal
from QAssemble.Src.FTGrid import FTGrid
from QAssemble.Src.BLatDyn import BLatDyn

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
    Basis = [[[0, 0, 0],2],[[1/2,1/2,1/2],2]]
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
    blatdyn = BLatDyn(crystal=crystal, ft=ftgrid)
    nk = len(crystal.kpoint)
    nw = len(ftgrid.omega)
    ntau = len(ftgrid.tau)
    norb = len(crystal.bind)
    ns = crystal.ns
    blatdynmpi = BLatDynMPI(crystal=crystal, ftgrid=ftgrid,nprock=nprock, nprocw=nprocw, nk = len(crystal.kpoint), nw = len(ftgrid.omega), ntau = len(ftgrid.tau), mpimanager=mpimanager)

    glatk = np.zeros((norb, norb, ns, ns, nk, nw), dtype=np.complex128, order='F')

    for iw in range(nw):
        for ik in range(nk):
            for ks in range(ns):
                for js in range(NSpin):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            glatk[iorb, jorb, js, ks, ik, iw] = np.sin(np.linalg.norm(crystal.kpoint[ik]))*1j + np.cos(np.linalg.norm(crystal.kpoint[ik]))
    
    glatr = blatdyn.K2R(glatk)
    nkloc = len(blatdynmpi.mpimanager.klocal[blatdynmpi.commk.Get_rank()])
    submatrixf = blatdynmpi.submatrixw[blatdynmpi.commw.Get_rank()]
    nwloc = submatrixf[1] - submatrixf[0]
    tempmat = np.zeros((norb, norb, ns, ns, nkloc, nwloc), dtype=np.complex128, order='F')
    for iw in range(nwloc):
        for ik in range(nkloc):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            kidx = blatdynmpi.mpimanager.KLocal2Global([blatdynmpi.commk.Get_rank(), ik])
                            fidx = blatdynmpi.mpimanager.FLocal2Global([blatdynmpi.commw.Get_rank(), iw])
                            tempmat[iorb, jorb, js, ks, ik, iw] = glatk[iorb, jorb, js, ks, kidx, fidx]
    tempmat2 = blatdynmpi.K2R(tempmat)

    nrloc = tempmat2.shape[4]

    if (comm.Get_rank() == 0):
        print('Checking the results of K2R')
    nrloc = len(blatdynmpi.mpimanager.rlocal[blatdynmpi.commk.Get_rank()])
    for iw in range(nwloc):
        for ir in range(nrloc):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            ridx = blatdynmpi.mpimanager.RLocal2Global([blatdynmpi.commk.Get_rank(), ir])
                            fidx = blatdynmpi.mpimanager.FLocal2Global([blatdynmpi.commw.Get_rank(), iw])
                            err = glatr[iorb, jorb, js, ks, ridx, fidx] - tempmat2[iorb, jorb, js, ks, ir, iw]
                            if (abs(err) > 1.0e-6):
                                print(iorb, jorb, js, ks, ir, iw, abs(err), glatr[iorb, jorb, js, ks, ridx, fidx], tempmat2[iorb, jorb, js, ks, ir, iw])
        # for iw in range(nw):
        #     for ik in range(nk):
        #         for js in range(ns):
        #             for jorb in range(norb):
        #                 for iorb in range(norb):
        #                     err = glatr[iorb, jorb, js, ik, iw] - matr2[iorb, jorb, js, ik, iw]
        #                     if (abs(err) > 1.0e-6):
        #                         print(iorb, jorb, js, ik, iw, abs(err), glatr[iorb, jorb, js, ik, iw], matr2[iorb, jorb, js, ik, iw])
        
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