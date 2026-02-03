from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import os, sys
import numba
import scipy.linalg
import numpy as np
import scipy
import h5py
from MPI import CheckMPI

from Fourier import Fourier

class MPIManager(object):

    def __init__(self):

        self.ismpi = CheckMPI()
        if self.ismpi:
            from MPI import IsMPI as MPIFunction
            self.mf = MPIFunction()
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            
        
            # for member in members: setattr(self, member, getattr(mf, member))
        else:
            from MPI import NoMPI as MPIFunction
            self.mf = MPIFunction()
            self.comm = None
            self.rank = 0
            self.size = 1        

        if (self.rank == 0):
            print("Parallelization with MPI Start")

        required = MPI.THREAD_MULTIPLE
        provided = MPI.Query_thread()
        if provided < required:
            if self.rank == 0:
                print("MPI does not support THREAD_MULTIPLE")
            sys.exit(1)

        self.mpidict = {}

    def Quary(self, **kwargs):

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

            if self.ismpi:

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

                fft, arr, arrT, slicef, sliceb, localshapef, localshapeb = Fourier.FFT(comm=commk, shape=shape)

                klocal = self.mf.KRCompositeIndex(slicef)
                rlocal = self.mf.KRCompositeIndex(sliceb)
                self.mf.kloc = klocal
                self.mf.rloc = rlocal
                # print(klocal)
                kloc2glob = self.mf.KRLocalGlobal(commk, klocal, shape)
                rloc2glob = self.mf.KRLocalGlobal(commk, rlocal, shape)

                # Store communicators
                nodedict['commk'] = commk
                nodedict['commf'] = commf
                nodedict['commtau'] = commtau

                # Store submatrix
                nodedict['submatrixk'] = slicef
                nodedict['localshapek'] = localshapef
                nodedict['submatrixr'] = sliceb
                nodedict['localshaper'] = localshapeb

                # Store FFT variables
                nodedict['fft'] = fft
                nodedict['arr'] = arr
                nodedict['arrT'] = arrT

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

