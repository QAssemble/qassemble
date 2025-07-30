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
    KGrid = [4, 5, 6]
    NElec = 1
    T = 2000
    cutoff = 11
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
    
    # print(f"submatrixkf : {flatdynmpi.submatrixkf[flatdynmpi.commk.Get_rank()]}")
    # print(f"local shape : {flatdynmpi.mpimanager.localshapef[flatdynmpi.commk.Get_rank()]}")
    # print(f"submatrixf : {flatdynmpi.submatrixw[flatdynmpi.commw.Get_rank()]}")

    if (flatdynmpi.commw.Get_rank() == 0):
        print(flatdynmpi.submatrixw)
        floc = flatdynmpi.mpimanager.FTLocalGlobal(flatdynmpi.submatrixw)
        print(floc)
        
if __name__ == '__main__':
    main()