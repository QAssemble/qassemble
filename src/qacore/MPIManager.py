from mpi4py import MPI
import sys
from .Crystal import Crystal
from .FTGrid import FTGrid
import numpy as np


class MPIManager(object):

    mpidict = {}
    def __init__(self, comm : MPI.COMM_WORLD):

        print("Parallelization with MPI Start")
        self.comm = comm 
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def Quary(self, nk : int, nw : int, nprock : int, nprocw : int):

        if (nk, nw, nprock, nprocw) in MPIManager.mpidict:
            #return the node dict for nk, nw
            return MPIManager.mpidict[(nk, nw, nprock, nprocw)]
        else:
            nodedict = {}
            # nodedict['nk'] = nk
            # nodedict['nw'] = nw
            if nprock * nprocw != self.size:
                if self.rank == 0:
                    print(
                        f"Error: nprock*nprocw = {nprock*nprocw}, but world size = {self.size}"
                    )
                raise ValueError("nprock*nprocw must equal MPI world size")
            
            ktemp = np.arange(nk)
            kchunk = np.array_split(ktemp, nprock)
            submatrixk = [(chunk[0], chunk[-1]+1) for chunk in kchunk]
            nodedict['submatrixk'] = submatrixk

            wtemp = np.arange(nw)
            wchunk = np.array_split(wtemp, nprocw)
            submatrixw = [(chunk[0], chunk[-1]+1) for chunk in wchunk]
            nodedict['submatrixw'] = submatrixw

            kidx = self.rank // nprocw
            widx = self.rank % nprocw

            commk = self.comm.Split(color=kidx, key=widx)
            commw = self.comm.Split(color=widx, key=kidx)

            nodedict['commk'] = commk
            nodedict['commw'] = commw
            nodedict['comkrank'] = commk.Get_rank()
            nodedict['commwrank'] = commw.Get_rank()
            nodedict['commksize'] = commk.Get_size()
            nodedict['commwsize'] = commw.Get_size()

            MPIManager.mpidict[(nk, nw, nprock, nprocw)] = nodedict

            # commk, commw, submatrixk, submatrixw, commk.rank, commk.size, commw.rank, commw.size, 
            return nodedict

    # 
    
class FLatDynMPI(object):

    def __init__(self, crystal : Crystal, ftgrid : FTGrid, nk : int, nw : int, nprock : int, nprocw : int, mpimanager : MPIManager):

        self.crystal = crystal
        self.ftgrid = ftgrid
        self.nk = nk
        self.nw = nw
        self.nprock = nprock
        self.nprocw = nprocw
        # self.mpimanager = mpimanager

        
        self.nodedict = mpimanager.Quary(nk, nw, nprock, nprocw)

        self.commk = self.nodedict['commk']
        self.commw = self.nodedict['commw']
        self.submatrixk = self.nodedict['submatrixk']
        self.submatrixw = self.nodedict['submatrixw']

        
        


class FLatDynIrrCoh(FLatDynMPI):

    head = None
    tail = None


class FLatDynFullCoh(FLatDynMPI):

    head = None
    tail = None

class FLatDynFineFine(FLatDynMPI):

    head = None
    tail = None
    