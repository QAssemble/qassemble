#!/usr/bin/env python3
import os, sys
import numpy as np
from mpi4py import MPI
qapath = os.environ.get("QAssemble")
sys.path.append(qapath + "/src")
# sys.path.append(qapath + "/src/qacore/modules")
from QAssemble.MPIManager import BLatDynMPI, MPIManager
from QAssemble.Crystal import Crystal
from QAssemble.FTGrid import FTGrid
from QAssemble.BLatDyn import BLatDyn

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
            for ks in range(NSpin):
                for js in range(NSpin):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            glatk[iorb, jorb, js, ks, ik, iw] = np.sin(np.linalg.norm(crystal.kpoint[ik]))*1j + np.cos(np.linalg.norm(crystal.kpoint[ik]))
    
    glatr = blatdyn.K2R(glatk)

    # glatr = glatr.reshape((norb, norb, ns, KGrid[0], KGrid[1], KGrid[2], nw), order='F')

    nrloc = len(blatdynmpi.mpimanager.rlocal[blatdynmpi.commk.Get_rank()])
    # tempmat = np.zeros((norb, norb, ns, nrloc, nw), dtype=np.complex128, order='F')
    
    submatrixf = blatdynmpi.submatrixw[blatdynmpi.commw.Get_rank()]
    nwloc = submatrixf[1] - submatrixf[0]

    tempmat = np.zeros((norb, norb, ns, ns, nrloc, nwloc), dtype=np.complex128, order='F')
    for iw in range(nwloc):
        for ir in range(nrloc):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            ridx = blatdynmpi.mpimanager.RLocal2Global([blatdynmpi.commk.Get_rank(), ir])
                            widx = blatdynmpi.mpimanager.FLocal2Global([blatdynmpi.commw.Get_rank(), iw])
                            tempmat[iorb, jorb, js, ks, ir, iw] = glatr[iorb, jorb, js, ks, ridx, widx]
    

    tempmat2 = blatdynmpi.R2K(tempmat)
    # glatk2 = np.zeros_like(glatr,dtype=np.complex128, order='F')

    # flatdynmpi.commw.Allreduce(tempmat, tempmat2, op=MPI.SUM)
    # flatdynmpi.commk.Allreduce(tempmat2, glatk2, op=MPI.SUM)
    # glatk_check = glatk2.reshape((norb, norb, ns, nk, nw), order='F')

    if (comm.Get_rank() == 0):
        print('Checking the results of R2K')
    nkloc = len(blatdynmpi.mpimanager.klocal[blatdynmpi.commk.Get_rank()])
    for iw in range(nwloc):
        for ik in range(nkloc):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            kidx = blatdynmpi.mpimanager.KLocal2Global([blatdynmpi.commk.Get_rank(), ik])
                            fidx = blatdynmpi.mpimanager.FLocal2Global([blatdynmpi.commw.Get_rank(), iw])
                            err = glatk[iorb, jorb, js, ks, kidx, fidx] - tempmat2[iorb, jorb, js, ks, ik, iw]
                            if (abs(err) > 1.0e-6):
                                print(iorb, jorb, js, ks, ik, iw, abs(err), glatk[iorb, jorb, js, ks, kidx, fidx], tempmat2[iorb, jorb, js, ks, ik, iw])
if __name__ == '__main__':
    main()