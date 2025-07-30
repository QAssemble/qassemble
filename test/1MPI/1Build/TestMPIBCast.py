#!/usr/bin/env python3
import os
import sys

import numpy as np
from mpi4py import MPI

qapath = os.environ.get("QAssemble")
sys.path.append(qapath + "/src")
from QAssemble.Src_mpi.Crystal import Crystal
from QAssemble.Src_mpi.FLatDyn import FLatDyn
from QAssemble.Src_mpi.FTGrid import FTGrid

# sys.path.append(qapath + "/src/qacore/modules")
from QAssemble.Src_mpi.MPIManager import FLatDynMPI, MPIManager


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
    RVec = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    Basis = [[[1 / 2, 1 / 2, 1 / 2], 1]]
    NSpin = 1
    SOC = False
    KGrid = [9, 9, 9]
    NElec = 1
    T = 300
    cutoff = 100
    cry = {
        "RVec": RVec,
        "Basis": Basis,
        "CorF": "F",
        "SOC": SOC,
        "NSpin": NSpin,
        "NElec": NElec,
        "KGrid": KGrid,
    }
    ft = {"T": T, "cutoff": cutoff}
    crystal = Crystal(cry=cry)
    ftgrid = FTGrid(ft=ft)
    flatdyn = FLatDyn(crystal=crystal, ft=ftgrid)
    nk = len(crystal.kpoint)
    nw = len(ftgrid.omega)
    ntau = len(ftgrid.tau)
    norb = len(crystal.find)
    ns = crystal.ns
    flatdynmpi = FLatDynMPI(
        crystal=crystal,
        ftgrid=ftgrid,
        nprock=nprock,
        nprocw=nprocw,
        nk=len(crystal.kpoint),
        nw=len(ftgrid.omega),
        ntau=len(ftgrid.tau),
        mpimanager=mpimanager,
    )

    glatt0 = np.zeros((norb, norb, ns, nk, nw), dtype=np.complex128, order="F")
    hmat = np.zeros((norb, norb, ns, nk), dtype=np.complex128, order="F")
    identity = np.identity(norb, np.complex128)

    for ik in range(nk):
        for js in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    hmat[iorb, jorb, js, ik] = ((iorb + 1) + (jorb + 1)) * 0.5 + (
                        js + 1
                    ) * 0.3 * (ik + 1)
                    if iorb == jorb:
                        hmat[iorb, jorb, js, ik] = hmat[iorb, jorb, js, ik] - 6.0

    for iw in range(nw):
        for ik in range(nk):
            for js in range(ns):
                tempmat = identity * ftgrid.omega[iw] * 1j - hmat[:, :, js, ik]
                glatt0[:, :, js, ik, iw] = np.linalg.inv(tempmat)
    
    submatrixf = flatdynmpi.submatrixw[flatdynmpi.commw.Get_rank()]
    nwloc = submatrixf[1] - submatrixf[0]
    nkloc = len(flatdynmpi.mpimanager.klocal[flatdynmpi.commk.Get_rank()])

    tempmat2 = np.zeros((norb, norb, ns, nkloc, nwloc), dtype=np.complex128, order="F")
    # tempmat3 = np.zeros((norb, norb, ns, nkloc, nwloc), dtype=np.complex128, order='F')
    for iw in range(nwloc):
        for ik in range(nkloc):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        kidx = flatdynmpi.mpimanager.KLocal2Global(
                            [flatdynmpi.commk.Get_rank(), ik]
                        )
                        fidx = flatdynmpi.mpimanager.FLocal2Global(
                            [flatdynmpi.commw.Get_rank(), iw]
                        )
                        tempmat2[iorb, jorb, js, ik, iw] = glatt0[
                            iorb, jorb, js, kidx, fidx
                        ]


    fflast = flatdynmpi.mpimanager.FMPIBCast(flatdynmpi.commw, tempmat2, len(ftgrid.omega)-1)

    print(glatt0.shape)
    print(fflast.shape)
    for ik in range(nkloc):
        for js in range(ns):
            for jorb in range(norb):
                for iorb in range(norb):
                    kidx = flatdynmpi.mpimanager.KLocal2Global([flatdynmpi.commk.Get_rank(), ik])
                    err = glatt0[iorb, jorb, js, kidx, -1] - fflast[iorb, jorb, js, ik]
                    if (abs(err) > 1.0e-6):
                        print(comm.Get_rank(), iorb, jorb, js, ik, abs(err), glatt0[iorb, jorb, js, kidx, -1], fflast[iorb, jorb, js, ik])

if __name__ == "__main__":
    main()
