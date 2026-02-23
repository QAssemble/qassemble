from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import os, sys
import numba
import scipy.linalg
import numpy as np
import scipy
import h5py
# from MPI import IsMPI as MPIFunction
from Common import Common
from FFT import FFT
from Fourier import Fourier

class MPIManager(object):

    def __init__(self, comm):


        self.mf = MPIFunction()
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if (self.rank == 0):
            print("Parallelization with MPI Start")

        required = MPI.THREAD_MULTIPLE
        provided = MPI.Query_thread()
        if provided < required:
            if self.rank == 0:
                print("MPI does not support THREAD_MULTIPLE")
            sys.exit(1)

        self.mpidict = {}

    def Query(self, **kwargs):

        nprock = kwargs.get('nprock', 1)
        nprocf = kwargs.get('nprocf', 1)
        
        nk = kwargs['nk']
        nf = kwargs['nf']
        ntau = kwargs['ntau']
        shape = kwargs['shape']

        if (nk, nf, ntau, nprock, nprocf) in self.mpidict:
            return self.mpidict[(nk, nf, ntau, nprock, nprocf)]
        else:
            nodedict = {}

            if (nprock * nprocf != self.size):
                if self.rank == 0:
                    errmsg = f"Error: nprock*nprocf = {nprock*nprocf}, but world size = {self.size}"
                    print(errmsg)
                    raise ValueError("nprock*nprocf must equal MPI world size")
            ftemp = np.arange(nf)
            fchunk = np.array_split(ftemp, nprocf)
            submatrixf = [(chunk[0], chunk[-1]+1) for chunk in fchunk]
            nodedict['submatrixf'] = submatrixf
            floc = self.mf.FTLocalGlobal(submatrixf)
            self.mf.floc = floc
            
            
            tautemp = np.arange(ntau)
            tauchunk = np.array_split(tautemp, nprocf)
            submatrixtau = [(chunk[0], chunk[-1]+1) for chunk in tauchunk]
            nodedict['submatrixtau'] = submatrixtau
            tloc = self.mf.FTLocalGlobal(submatrixtau)
            self.mf.tloc = tloc

            kidx = self.rank // nprock
            fidx = self.rank % nprock

            commk = self.comm.Split(color=kidx, key = fidx)
            commf = self.comm.Split(color=fidx, key=kidx)
            commtau = self.comm.Split(color=fidx, key=kidx)

            # print(commk.Get_size(), commf.Get_size(), commtau.Get_size())

            self.fft = FFT(commk, shape)

            klocal = self.mf.KRCompositeIndex(self.fft.slicef)
            rlocal = self.mf.KRCompositeIndex(self.fft.sliceb)
            self.mf.kloc = klocal
            self.mf.rloc = rlocal
            # print(klocal)
            kloc2glob = self.mf.KRLocalGlobal(commk, klocal, shape)
            rloc2glob = self.mf.KRLocalGlobal(commk, rlocal, shape)
            self.mf.kloc2glob = kloc2glob
            self.mf.rloc2glob = rloc2glob

            # Store communicators
            nodedict['commk'] = commk
            nodedict['commf'] = commf
            nodedict['commtau'] = commtau

            # Store submatrix
            nodedict['submatrixk'] = self.fft.slicef
            nodedict['localshapek'] = self.fft.localshapef
            nodedict['submatrixr'] = self.fft.sliceb
            nodedict['localshaper'] = self.fft.localshapeb

            # Store FFT variables
            nodedict['fft'] = self.fft

            # Store local indices
            nodedict['kloc'] = klocal
            nodedict['rloc'] = rlocal
            nodedict['floc'] = floc
            nodedict['tloc'] = tloc

            # Store local to global indices
            nodedict['kloc2glob'] = kloc2glob
            nodedict['rloc2glob'] = rloc2glob
            nodedict['grid'] = shape

            self.mpidict[(nk, nf, ntau, nprock, nprocf)] = nodedict

            return nodedict

class MPIFunction():


    def __init__(self):

        self.floc = {}
        self.tloc = {}
        self.kloc = {}
        self.rloc = {}
        self.kloc2glob = {}
        self.rloc2glob = {}


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
            print(self.KRIdx2KVec.__name__)
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
            print(self.KRVec2KIdx.__name__)
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

        loc_idx = self.kloc2glob[rank].values()

        return A[loc_idx]

    def SliceArrayR(self, A : np.ndarray, rank : int) -> np.ndarray:

        loc_idx = self.rloc2glob[rank].values()

        return A[loc_idx]

    def SliceArrayTau(self, A : np.ndarray, rank : int) -> np.ndarray:

        loc_idx = self.tloc[rank].values()

        return A[loc_idx]

