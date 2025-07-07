from mpi4py import MPI
from mpi4py_fft import PFFT
import os, sys

import scipy.linalg
from .Crystal import Crystal
from .FTGrid import FTGrid
import numpy as np
import scipy
import h5py
qapath = os.environ.get('QAssemble','')
sys.path.append(qapath+'/src/qacore/modules')
import QAFort



class MPIManager(object):


    def __init__(self, comm : MPI.COMM_WORLD):

        print("Parallelization with MPI Start")
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.mpidict = {}

    def Quary(self, nk : int, nf : int, ntau : int, nprock : int, nprocf : int):

        if (nk, nf, ntau, nprock, nprocf) in self.mpidict:
            #return the node dict for nk, nw
            return self.mpidict[(nk, nf, ntau, nprock, nprocf)]
        else:
            nodedict = {}
            # nodedict['nk'] = nk
            # nodedict['nw'] = nw
            if nprock * nprocf != self.size:
                if self.rank == 0:
                    print(
                        f"Error: nprock*nprocf = {nprock*nprocf}, but world size = {self.size}"
                    )
                raise ValueError("nprock*nprocf must equal MPI world size")

            ktemp = np.arange(nk)
            kchunk = np.array_split(ktemp, nprock)
            submatrixk = [(chunk[0], chunk[-1]+1) for chunk in kchunk]
            nodedict['submatrixk'] = submatrixk

            wtemp = np.arange(nf)
            wchunk = np.array_split(wtemp, nprocf)
            submatrixw = [(chunk[0], chunk[-1]+1) for chunk in wchunk]
            nodedict['submatrixw'] = submatrixw

            tautemp = np.arange(ntau)
            tauchunk = np.array_split(tautemp, nprocf)
            submatrixtau = [(chunk[0], chunk[-1]+1) for chunk in tauchunk]
            nodedict['submatrixtau'] = submatrixtau

            kidx = self.rank // nprock
            widx = self.rank % nprock

            commk = self.comm.Split(color=kidx, key=widx)
            commf = self.comm.Split(color=widx, key=kidx)
            commtau = self.comm.Split(color=widx, key=kidx)

            nodedict['commk'] = commk
            nodedict['commf'] = commf
            nodedict['commtau']  = commtau
            nodedict['commkrank'] = commk.Get_rank()
            nodedict['commfrank'] = commf.Get_rank()
            nodedict['commtaurank'] = commtau.Get_rank()
            nodedict['commksize'] = commk.Get_size()
            nodedict['commfsize'] = commf.Get_size()
            nodedict['commtausize'] = commtau.Get_size()

            self.mpidict[(nk, nf, ntau, nprock, nprocf)] = nodedict

            # commk, commw, submatrixk, submatrixw, commk.rank, commk.size, commw.rank, commw.size,
            return nodedict

    #

class FLatDynMPI(object):

    def __init__(self, crystal : Crystal, ftgrid : FTGrid, nk : int, nw : int, ntau : int, nprock : int, nprocw : int, mpimanager : MPIManager):

        self.crystal = crystal
        self.ftgrid = ftgrid
        self.nk = nk
        self.nw = nw
        self.nprock = nprock
        self.nprocw = nprocw
        self.mpimanager = mpimanager


        self.nodedict = mpimanager.Quary(nk, nw, ntau, nprock, nprocw)

        self.commk = self.nodedict['commk']
        self.commw = self.nodedict['commf']
        self.submatrixk = self.nodedict['submatrixk']
        self.submatrixw = self.nodedict['submatrixw']

        self.commtau = self.nodedict['commtau']

        self.submatrixtau = self.nodedict['submatrixtau']

    def CheckGroup(self, filepath : str, group : str):

        with h5py.File(filepath, 'r') as file:
            return group in file

    def Save(self, hdf5file : str = None, group : str = None, subgroup : str = None, data : np.ndarray = None, dataname : str = None):

        with h5py.File(hdf5file, 'a') as file:
            if (self.CheckGroup(hdf5file, group)):
                g =  file[group]
                if subgroup in g:
                    subg = g[subgroup]
                else:
                    subg = g.create_group(subgroup)
            else:
                g = file.create_group(group)
                subg = g.create_group(subgroup)

            subg.create_dataset(dataname, data=data, dtype=np.complex128, driver='mpio', comm = self.mpimanager.comm)

            return None

    def Load(self, hdf5file : str = None, group : str = None, subgroup : str = None, data : np.ndarray = None, dataname : str = None):

        with h5py.File(hdf5file, 'r') as file:
            if (self.CheckGroup(hdf5file, group)):
                g =  file[group]
                if subgroup in g:
                    subg = g[subgroup]
                    if dataname in subg:
                        data = subg[dataname][:]
                        return data
                    else:
                        raise KeyError(f"{dataname} not found in {subgroup}")
                else:
                    raise KeyError(f"{subgroup} not found in {group}")
            else:
                raise KeyError(f"{group} not found in {hdf5file}")

    def Inverse(self, matin : np.ndarray) -> np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[3]
        nft = matin.shape[4]

        matout = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')

        submatrixk = self.submatrixk[self.nodedict['comkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]

        for ift in range(submatrixw[0], submatrixw[1]):
            for irk in range(submatrixk[0], submatrixk[1]):
                for js in range(ns):
                    matout[:, :, js, irk, ift] = np.linalg.inv(matin[:, :, js, irk, ift])

        return matout

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:


        matout = np.zeros_like(mat1, dtype=np.complex128, order='F')

        submatrixk = self.submatrixk[self.nodedict['comkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]

        for ift in range(submatrixw[0], submatrixw[1]):
            for irk in range(submatrixk[0], submatrixk[1]):
                matout[:, :, :, irk, ift] = QAFort.dyson.flocstc(mat1[:, :, :, irk, ift], mat2[:, :, :, irk, ift])

        return matout

    # @numba.jit
    def K2R(self, matk : np.ndarray) -> np.ndarray:

        rkvec = self.crystal.kpoint
        rkgrid = self.crystal.rkgrid

        norb = matk.shape[0]
        ns = matk.shape[2]

        commk = self.commk
        # submatrixk = self.submatrixk[self.nodedict['comkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]

        # subk0, subk1 = submatrixk
        subw0, subw1 = submatrixw
        # local_nrk = subk1 - subk0
        local_nft = subw1 - subw0

        # nrk = local_nrk
        nrk = len(rkvec)
        nft = local_nft

        matr = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
        tempmat = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
        tempmat2 = np.zeros((norb, norb, ns, rkgrid[0], rkgrid[1], rkgrid[2], nft), dtype=np.complex128, order='F')

        for loc_ift, ift in enumerate(range(subw0, subw1)):
            # for loc_irk, global_irk in enumerate(range(subk0, subk1)):
            for irk in range(nrk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            a, m1 = self.crystal.FAtomOrb(iorb)
                            b, m2 = self.crystal.FAtomOrb(jorb)
                            delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
                            phase = np.exp(2.0j*np.pi*np.dot(rkvec[irk], delta))
                            tempmat[iorb, jorb, js, irk, loc_ift] = matk[iorb, jorb, js, irk, loc_ift] * phase

        tempmat = tempmat.reshape((norb, norb, ns, rkgrid[0], rkgrid[1], rkgrid[2], nft), order='F')
        fft = PFFT(comm=commk, shape=rkgrid, axes=(0, 1, 2), dtype=np.complex128)

        # @numba.jit
        for loc_ift in range(nft):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        tempmat2[iorb, jorb, js, :, :, :, loc_ift] = fft.backward(
                            tempmat[iorb, jorb, js, :, :, :, loc_ift],
                            normalization=True
                        )

        matr = tempmat2.reshape((norb, norb, ns, nrk, nft), order='F')

        return matr

    def R2K(self, matr : np.ndarray) -> np.ndarray:

        rkvec = self.crystal.kpoint
        rkgrid = self.crystal.rkgrid

        norb = matr.shape[0]
        ns = matr.shape[2]

        commk = self.commk
        # submatrixk = self.submatrixk[self.nodedict['comkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]

        # subk0, subk1 = submatrixk
        subw0, subw1 = submatrixw
        # local_nrk = subk1 - subk0
        local_nft = subw1 - subw0

        # nrk = local_nrk
        nrk = len(rkvec)
        nft = local_nft

        matk = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
        # tempmat = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
        tempmat = np.zeros((norb, norb, ns, rkgrid[0], rkgrid[1], rkgrid[2], nft), dtype=np.complex128, order='F')
        # tempmat2 = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')

        tempmat2 = matr.reshape((norb, norb, ns, rkgrid[0], rkgrid[1], rkgrid[2], nft), order='F')
        fft = PFFT(comm=commk, shape=rkgrid, axes=(0, 1, 2), dtype=np.complex128)
        for loc_ift in range(nft):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        tempmat[iorb, jorb, js, :, :, :, loc_ift] = fft.forward(
                            tempmat2[iorb, jorb, js, :, :, :, loc_ift]
                        )

        # matr = tempmat2.reshape((norb, norb, ns, nrk, nft), order='F')
        tempmat = tempmat.reshape((norb, norb, ns, nrk, nft), order='F')
        for loc_ift, ift in enumerate(range(subw0, subw1)):
            # for loc_irk, global_irk in enumerate(range(subk0, subk1)):
            for irk in range(nrk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            a, m1 = self.crystal.FAtomOrb(iorb)
                            b, m2 = self.crystal.FAtomOrb(jorb)
                            delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
                            phase = np.exp(2.0j*np.pi*np.dot(rkvec[irk], delta))
                            matk[iorb, jorb, js, irk, loc_ift] = tempmat[iorb, jorb, js, irk, loc_ift] * phase

        return matk






class FLatDynIrrCoh(FLatDynMPI):

    head = None
    tail = None


class FLatDynFullCoh(FLatDynMPI):

    head = None
    tail = None

class FLatDynFineFine(FLatDynMPI):

    head = None
    tail = None
