from numba.core.types import none
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import os, sys
import numba
import scipy.linalg
import numpy as np
import scipy
import h5py
from Common import Common
from Fourier import Fourier

def CheckMPI():

    # for OpenMPI:
    if os.environ.get('OMPI_COMM_WORLD_RANK'):
        ismpi = True
    # for MPICH and intel based MPI:
    elif os.environ.get('PMI_RANK'):
        ismpi = True
    # for PMIx (used by srun/Slurm with PMIx support):
    elif os.environ.get('PMIX_RANK'):
        ismpi = True
    elif os.environ.get('CRAY_MPICH_VERSION'):
        ismpi = True
    # to force the MPI init manually
    elif os.environ.get('TRIQS_FORCE_MPI_INIT'):
        ismpi = True
    else:
        print('Warning: could not identify MPI environment!')

    return ismpi


class NoMPI():

    def SliceArrayK(self, A):

        return A

    def SliceArrayW(self, A):

        return A

    def SliceArrayTau(self, A):

        return A


class IsMPI():


    def __init__(self):

        self.floc = {}
        self.tloc = {}
        self.kloc = {}
        self.rloc = {}


    def FTLocalGlobal(self, submatrix):
        
        floc = {}

        for irank in range(len(submatrix)):
            floc[irank] = {}
            i = 0
            f = submatrix[irank]
            for ifreq in range(f[0], f[1]):
                floc[irank][i] = ifreq
                i += 1
        
        return floc

    def FTGlobal2Local(self, idx : int, loc_dict : dict) -> list:

        for key, val in loc_dict.items():
            for key2, val2 in val.items():
                if (val2 == idx):
                    return [key, key2]

    def FTLocal2Global(self, loc_list : list, loc_dict : dict) -> int:

        rank, loc_idx = loc_list

        return loc_dict[rank][loc_idx]

    def KRCompositeIndex(self, local_slice):

        idx = {}

        for rank, slices in local_slice.items():
            (x0, x1), (y0, y1), (z0, z1) = slices
            
            loc_idx = 0
            loc_dict = {}

            for z in range(z0, z1):
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        loc_dict[loc_idx] = [x, y, z]
                        loc_idx += 1

            idx[rank] = loc_dict

        return idx

    def KRLocalGlobal(self, comm : MPI.COMM_WORLD, local_slice, shape):

        map = {}
        
        kidx = Common.KIdx2KVec(shape)

        for irank in range(comm.Get_size()):
            map[irank] = {}
            for key, val in local_slice[irank].items():
                idx = Common.KVec2KIdx(val, kidx)
                map[irank][key] = idx

        return map

    def KRGlobal2Local(self, kidx : int, klocal2global : dict):

        for key, val in klocal2global.items():
            for key2, val2 in val.items():
                if (kidx == val2):
                    return [key, key2]

    def KRLocal2Global(self, klist : list, kloc : dict ):

        rank, local_index = klist

        return [rank, kloc[rank][local_index]]

    def KRList2Local(self, klist : list, kloc : dict) -> list:

        rank, k3d = klist

        for key, val in kloc[rank].items():
            if (k3d == val):
                return [rank, key]

    def KRLocal2List(self, klist : list, kloc : dict) -> list:

        rank, loc_idx = klist

        return [rank, kloc[rank][loc_idx]]

    def KRIdx2KVec(self, rank, matin : np.ndarray, kloc : dict, localshapef : dict) -> np.ndarray:

        (nkx, nky, nkz) = localshapef[rank]

        if (len(matin) != nkx*nky*nkz):
            print(self.KIdx2KVec.__name__)
            print("Input array is wrong. Check the array dimension")
            print(matin.shape, nkx, nky, nkz)
            sys.exit()

        matout = np.zeros((nkx, nky, nkz), dtype=np.complex128)

        for ik in range(nkx*nky*nkz):
            _, [ikx, iky, ikz] = self.KRLocal2List([rank, ik], kloc)
            matout[ikx, iky, ikz] = matin[ik]

        return matout

    def KRVec2KIdx(self, rank, matin : np.ndarray, kloc : dict) -> np.ndarray:

        (nkx, nky, nkz) = matin.shape

        if (nkx*nky*nkz != len(kloc[rank])):
            print(self.KVec2KIdx.__name__)
            print("Input array is wrong. Check the array dimension")
            print(matin.shape, nkx, nky, nkz)
            sys.exit()

        matout = np.zeros((nkx*nky*nkz), dtype=np.complex128)

        for ik in range(nkx*nky*nkz):
            _, [ikx, iky, ikz] = self.KRLocal2List([rank, ik], kloc)
            matout[ik] = matin[ikx, iky, ikz]

        return matout

    def FTAllReduce(self, comm : MPI.COMM_WORLD, matin : np.ndarray, ndim : int, loc_dict : dict) -> np.ndarray:

        nloc = matin.shape[0]

        tempmat = np.zeros((ndim), dtype=np.complex128)
        matout = np.zeros((ndim), dtype=np.complex128)

        for i in range(nloc):
            idx = self.FTLocal2Global([comm.Get_rank(), i], loc_dict)
            tempmat[idx] = matin[i]

        comm.Allreduce(tempmat, matout, op=MPI.SUM)

        return matout

    def KRAllReduce(self, comm : MPI.COMM_WORLD, matin : np.ndarray, ndim : int, loc_dict : dict) -> np.ndarray:


        nloc = matin.shape[0]

        tempmat = np.zeros((ndim), dtype=np.complex128)
        matout = np.zeros((ndim), dtype=np.complex128)

        for i in range(nloc):
            idx = self.KRLocal2Global([comm.Get_rank(), i], loc_dict)
            tempmat[idx] = matin[i]

        comm.Allreduce(tempmat, matout, op=MPI.SUM)

        return matout

    def SliceArrayK(self, A : np.ndarray, rank : int) -> np.ndarray:

        pass

