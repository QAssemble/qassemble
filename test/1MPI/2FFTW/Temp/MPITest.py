#!/usr/bin/env python3
import os, sys
import numpy as np
from mpi4py import MPI
qapath = os.environ.get("QAssemble")
sys.path.append(qapath + "/src")
sys.path.append(qapath + "/src/QAssemble/modules")
from QAssemble.Src_mpi.MPIManager import FLatDynMPI, MPIManager
from QAssemble.Src_mpi.Crystal import Crystal
from QAssemble.Src_mpi.FTGrid import FTGrid
from QAssemble.Src_mpi.FLatDyn import FLatDyn
import QAFort
# from mpi4py_fft import 

def block(N, size, rank):
    
    q,r = divmod(N, size)
    n = q + (1 if r > rank else 0)
    s = rank * q + min(rank, r)
    return n, s

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

def create_rank_composite_index(global_shape, rank_slices):
    """
    Creates a local-to-global composite index mapping for each MPI rank.

    Args:
        global_shape: (Nz, Ny, Nx) shape of the global array (mpi4py-fft ordering: z,y,x)
        rank_slices: dict {rank: [(z_start, z_stop), (y_start, y_stop), (x_start, x_stop)]}

    Returns:
        dict of dicts: {rank: {local_linear_index: [x,y,z]}}
    """
    rank_composite_indices = {}

    Nz, Ny, Nx = global_shape

    for rank, slices in rank_slices.items():
        (x0, x1), (y0, y1), (z0, z1) = slices

        local_index = 0
        local_dict = {}
        for z in range(z0, z1):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    local_dict[local_index] = [x, y, z]
                    local_index += 1
        rank_composite_indices[rank] = local_dict

    return rank_composite_indices


def main():
    comm = MPI.COMM_WORLD
    mpimanager = MPIManager(comm=comm)

    nprock, nprocw = parse_args()
    RVec = [[10,0,0],[0,10,0],[0,0,10]]
    Basis = [[[1/2,1/2,1/2],1]]
    NSpin = 1
    SOC = False
    KGrid = [10, 10, 10]
    NElec = 1
    T = 300
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
    norb = len(crystal.find)
    ns = crystal.ns
    flatdynmpi = FLatDynMPI(crystal=crystal, ftgrid=ftgrid,nprock=nprock, nprocw=nprocw, nk = len(crystal.kpoint), nw = len(ftgrid.omega), ntau = len(ftgrid.tau), mpimanager=mpimanager)
    nk = len(flatdynmpi.mpimanager.klocal)
    submatrixf = flatdynmpi.submatrixw[flatdynmpi.commw.Get_rank()]
    nw = submatrixf[1] - submatrixf[0]

    hmat = np.zeros((norb, norb, ns, nk), dtype=np.complex128, order='F')
    glatt0 = np.zeros((norb, norb, ns, nk, nw), dtype=np.complex128, order='F')
    tempmat = np.zeros((norb, norb), dtype=np.complex128, order='F')

    for ik in range(nk):
        for js in range(ns):
            for jorb in range(norb):
                for iorb in range(norb):
                    hmat[iorb, jorb, js, ik] = ((iorb+1)+(jorb+1))*0.5+(js+1)*0.3*(ik+1)
                    if (iorb == jorb):
                        hmat[iorb, jorb, js, ik] += -6.0

    for ik in range(nk):
        for js in range(ns):
            for iw in range(nw):
                fidx = flatdynmpi.mpimanager.FLocal2Global([flatdynmpi.commw.Get_rank(), iw])
                tempmat = np.identity(norb)*ftgrid.omega[fidx]*1j - hmat[...,js,ik]
                tempmat2 = QAFort.common.dcmplx_matinv(tempmat, norb)
                glatt0[:, :, js, ik, iw] = tempmat2

    nwglob = len(ftgrid.omega)

    (rank1, local1) = flatdynmpi.mpimanager.FGlobal2Local(nwglob-1)
    (rank2, local2) = flatdynmpi.mpimanager.FGlobal2Local(nwglob-2)

    
    if flatdynmpi.commw.Get_rank() == rank1:
        print(rank1, local1, nwglob-1, glatt0[0, 0, 0, 0, local1])
        tempval1 = glatt0[0, 0, 0, 0, local1]
    else:
        tempval1 = None
    tempval1 = flatdynmpi.commw.bcast(tempval1, root=rank1)

    if flatdynmpi.commw.Get_rank() == rank2:
        print(rank2, local2, nwglob-2, glatt0[0, 0, 0, 0, local2])
        tempval2 = glatt0[0, 0, 0, 0, local2]
    else:
        tempval2 = None
    tempval2 = flatdynmpi.commw.bcast(tempval2, root=rank2)

    print(comm.Get_rank(), tempval1, tempval2)


    

        
if __name__ == '__main__':
    main()