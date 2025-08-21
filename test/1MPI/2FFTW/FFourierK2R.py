#!/usr/bin/env python3
import os, sys
import numpy as np
from mpi4py import MPI
qapath = os.environ.get("QAssemble")
sys.path.append(qapath + "/src")
# sys.path.append(qapath + "/src/qacore/modules")
from QAssemble.Src_mpi.MPIManager import MPIManager
from QAssemble.Src_mpi.Crystal import Crystal
from QAssemble.Src_mpi.FTGrid import FTGrid
from QAssemble.Src_mpi.FLatDyn import FLatDyn as FLatDynMPI
from QAssemble.Src.FLatDyn import FLatDyn

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
    # mpimanager = MPIManager(comm=comm)

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
    mpimanager = MPIManager(comm=comm, crystal=crystal)
    flatdynmpi = FLatDynMPI(crystal=crystal, ftgrid=ftgrid,nprock=nprock, nprocw=nprocw, nk = len(crystal.kpoint), nw = len(ftgrid.omega), ntau = len(ftgrid.tau), mpimanager=mpimanager)

    glatk = np.zeros((norb, norb, ns, nk, nw), dtype=np.complex128, order='F')

    for iw in range(nw):
        for ik in range(nk):
            for js in range(NSpin):
                for jorb in range(norb):
                    for iorb in range(norb):
                        glatk[iorb, jorb, js, ik, iw] = np.sin(np.linalg.norm(crystal.kpoint[ik]))*1j + np.cos(np.linalg.norm(crystal.kpoint[ik]))
    
    glatr = flatdyn.K2R(glatk)
    nkloc = len(flatdynmpi.mpimanager.klocal[flatdynmpi.commk.Get_rank()])
    # glatk = glatk.reshape((norb, norb, ns, KGrid[0], KGrid[1], KGrid[2], nw), order='F')
    
    # tempmat = np.zeros((norb, norb, ns, nkloc, nw), dtype=np.complex128, order='F')
    # submatrixk = flatdynmpi.submatrixkb[flatdynmpi.commk.Get_rank()]
    submatrixf = flatdynmpi.submatrixw[flatdynmpi.commw.Get_rank()]
    nwloc = submatrixf[1] - submatrixf[0]
    tempmat = np.zeros((norb, norb, ns, nkloc, nwloc), dtype=np.complex128, order='F')
    for iw in range(nwloc):
        for ik in range(nkloc):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        # print([flatdynmpi.commk.Get_rank(), ik])
                        kidx = flatdynmpi.mpimanager.KLocal2Global([flatdynmpi.commk.Get_rank(), ik])
                        fidx = flatdynmpi.mpimanager.FLocal2Global([flatdynmpi.commw.Get_rank(), iw])
                        tempmat[iorb, jorb, js, ik, iw] = glatk[iorb, jorb, js, kidx, fidx]
    tempmat2 = flatdynmpi.K2R(tempmat)

    nrloc = tempmat2.shape[3]

    if (comm.Get_rank() == 0):
        print('Fourier Transform K2R test Start')
    nrloc = len(flatdynmpi.mpimanager.rlocal[flatdynmpi.commk.Get_rank()])
    for iw in range(nwloc):
        for ir in range(nrloc):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        ridx = flatdynmpi.mpimanager.RLocal2Global([flatdynmpi.commk.Get_rank(), ir])
                        fidx = flatdynmpi.mpimanager.FLocal2Global([flatdynmpi.commw.Get_rank(), iw])
                        err = glatr[iorb, jorb, js, ridx, fidx] - tempmat2[iorb, jorb, js, ir, iw]
                        if (abs(err) > 1.0e-6):
                            print(iorb, jorb, js, ir, iw, abs(err), glatr[iorb, jorb, js, ridx, fidx], tempmat2[iorb, jorb, js, ir, iw])
    if (comm.Get_rank() == 0):
        print('Fourier Transform K2R test Finish')
    
if __name__ == '__main__':
    main()