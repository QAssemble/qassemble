from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
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


        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        if (self.rank == 0):
            print("Parallelization with MPI Start")
        self.mpidict = {}
        self.fft = None
        self.arr = None
        self.arrT = None
        self.slicef = None
        self.sliceb = None
        self.localshapef = None
        self.localshapeb = None

    def Quary(self, nk : int, nf : int, ntau : int, nprock : int, nprocf : int, crystal : Crystal):

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

            # ktemp = np.arange(nk)
            # kchunk = np.array_split(ktemp, nprock)
            # submatrixk = [(chunk[0], chunk[-1]+1) for chunk in kchunk]
            # nodedict['submatrixk'] = submatrixk

            wtemp = np.arange(nf)
            wchunk = np.array_split(wtemp, nprocf)
            submatrixw = [(chunk[0], chunk[-1]+1) for chunk in wchunk]
            nodedict['submatrixw'] = submatrixw
            self.crystal = crystal

            tautemp = np.arange(ntau)
            tauchunk = np.array_split(tautemp, nprocf)
            submatrixtau = [(chunk[0], chunk[-1]+1) for chunk in tauchunk]
            nodedict['submatrixtau'] = submatrixtau

            kidx = self.rank // nprock
            widx = self.rank % nprock

            commk = self.comm.Split(color=kidx, key=widx)
            commf = self.comm.Split(color=widx, key=kidx)
            commtau = self.comm.Split(color=widx, key=kidx)

            # mpifft = MPIFFT(commk,kgrid)
            self.fft = self.FFT(commk,crystal.rkgrid)
            self.klocal = self.CreateMPICompositeIndex(self.sliceb)
            self.rlocal = self.CreateMPICompositeIndex(self.slicef)
            
            self.klocal2global = self.MappingGlobal2Local(commk, self.klocal)
            self.rlocal2global = self.MappingGlobal2Local(commk, self.rlocal)
            nodedict['submatrixkf'] = self.slicef
            nodedict['localshapef'] = self.localshapef
            nodedict['submatrixkb'] = self.sliceb
            nodedict['localshapeb'] = self.localshapeb
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

            # del mpifft
            # commk, commw, submatrixk, submatrixw, commk.rank, commk.size, commw.rank, commw.size,
            return nodedict
        
    def FFT(self, comm = MPI.COMM_WORLD, shape = None, decomposition = 'pencil'):

        if (decomposition == 'slab'):
            fft = PFFT(comm = comm, shape=shape, axes=(0, 1, 2), dtype=np.complex128, grid=(-1))
        elif (decomposition == 'pencil'):
            fft = PFFT(comm = comm, shape=shape, axes=(0, 1, 2), dtype=np.complex128)

        self.arr = newDistArray(fft, forward_output=True)
        self.arrT = newDistArray(fft, forward_output=False)
        localslicef = fft.local_slice(forward_output=True)
        localsliceb = fft.local_slice(forward_output=False)

        # self.slicef = {comm.Get_rank(): [(s.start, s.stop) for s in localslicef]}
        # self.sliceb = {comm.Get_rank(): [(s.start, s.stop) for s in localsliceb]}
        # self.localshapef = {comm.Get_rank(): tuple(s.stop - s.start for s in localslicef)}
        # self.localshapeb = {comm.Get_rank(): tuple(s.stop - s.start for s in localsliceb)}
        localslicef = [(s.start, s.stop) for s in localslicef]
        localsliceb = [(s.start, s.stop) for s in localsliceb]
        localshapef = tuple(s[1] - s[0] for s in localslicef)
        localshapeb = tuple(s[1] - s[0] for s in localsliceb)

        # Gather data from all ranks
        all_slicef = comm.allgather(localslicef)
        all_sliceb = comm.allgather(localsliceb)
        all_localshapef = comm.allgather(localshapef)
        all_localshapeb = comm.allgather(localshapeb)

        # Store the data as a single dictionary accessible from all ranks
        self.slicef = {rank: all_slicef[rank] for rank in range(comm.Get_size())}
        self.sliceb = {rank: all_sliceb[rank] for rank in range(comm.Get_size())}
        self.localshapef = {rank: all_localshapef[rank] for rank in range(comm.Get_size())}
        self.localshapeb = {rank: all_localshapeb[rank] for rank in range(comm.Get_size())}
        # kb = self.CreateMPICompositeIndex(shape, self.sliceb)
        # kf = self.CreateMPICompositeIndex(shape, self.slicef)

        return fft
    
    def Forward(self, matin):

        val = self.arrT
        result = self.arr
        val[:] = matin
        result = self.fft.forward(val, result, normalize=False)

        return result
    
    def Backward(self, matin):

        val = self.arr
        result = self.arrT
        val[:] = matin

        result = self.fft.backward(val, result, normalize=False)

        return result

    def CreateMPICompositeIndex(self, rank_slices):
        """
        Creates a local-to-global composite index mapping for each MPI rank.

        Args:
            global_shape: (Nz, Ny, Nx) shape of the global array (mpi4py-fft ordering: z,y,x)
            rank_slices: dict {rank: [(z_start, z_stop), (y_start, y_stop), (x_start, x_stop)]}

        Returns:
            dict of dicts: {rank: {local_linear_index: [x,y,z]}}
        """
        rank_composite_indices = {}

        # Nz, Ny, Nx = global_shape

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
    
    def MappingGlobal2Local(self, commk, localdict : dict) -> dict:
        mapping = {}
        for irank in range(commk.Get_size()):
            mapping[irank] = {}
            for key, value in localdict[irank].items():
                kidx = self.crystal.MergeKind(value)
                mapping[irank][key] = kidx

        return mapping
    
    def KGlobal2Local(self, kidx : int) -> list:
        """
        Convert a global k-index to its corresponding local rank and index.

        Args :
            kidx (int): Global k-index to convert.
        Returns:
            (rank, local_index) list: A list containing the rank and local index corresponding to the global k-index.

        """

        for key, val in self.klocal2global.items():
            for key2, val2 in val.items():
                if (kidx == val2):
                    return [key, key2]
                
    def KLocal2Global(self, klocal : list) -> int:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klocal (list): Local k-index in the form [rank, local_index].
        Returns:
            int: Global k-index corresponding to the local k-index.
        """
        rank, local_index = klocal
        return self.klocal2global[rank][local_index]
                



    

class FLatDynMPI(object):

    def __init__(self, crystal : Crystal, ftgrid : FTGrid, nk : int, nw : int, ntau : int, nprock : int, nprocw : int, mpimanager : MPIManager):

        self.crystal = crystal
        self.ftgrid = ftgrid
        self.nk = nk
        self.nw = nw
        self.nprock = nprock
        self.nprocw = nprocw
        self.mpimanager = mpimanager
        self.nodedict = mpimanager.Quary(nk, nw, ntau, nprock, nprocw, self.crystal)

        self.commk = self.nodedict['commk']
        self.commw = self.nodedict['commf']
        self.submatrixkf = self.nodedict['submatrixkf']
        self.submatrixkb = self.nodedict['submatrixkb']
        self.submatrixw = self.nodedict['submatrixw']

        self.commtau = self.nodedict['commtau']

        self.submatrixtau = self.nodedict['submatrixtau']

        # self.kb = mpimanager.CreateMPICompositeIndex(crystal.rkgrid, self.submatrixkb)
        # self.kf = mpimanager.CreateMPICompositeIndex(crystal.rkgrid, self.submatrixkf)

        # self.kb2global = self.MappingGlobal2Local(self.kb)
        # self.kf2global = self.MappingGlobal2Local(self.kf)

        # self.fft = self.mpimanager.FFT()
        

        # self.mpifft = MPIFFT(comm=self.commk)

    def K2K3D(self, matin : np.ndarray) -> np.ndarray:

        rkgrid = self.crystal.rkgrid
        if (len(matin) != rkgrid[0]*rkgrid[1]*rkgrid[2]):
            print("Input array is wrong. Check the array dimension")
            sys.exit()
        matout = np.zeros((rkgrid[0], rkgrid[1], rkgrid[2]),dtype=np.complex128, order='F')

        for key, val in self.crystal.kind.items():
            matout[val[0], val[1], val[2]] = matin[key]

        return matout
    
    def MappingGlobal2Local(self, localdict : dict) -> dict:
        mapping = {}
        for irank in range(self.commk.Get_size()):
            mapping[irank] = {}
            for key, value in localdict[irank].items():
                kidx = self.crystal.MergeKind(value)
                mapping[irank][key] = kidx

        return mapping

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

        
        norb, _, ns, nkx, nky, nkz, nft = matin.shape

        matout = np.zeros((norb, norb, ns, nkx, nky, nkz, nft), dtype=np.complex128, order='F')
        

        submatrixk = self.submatrixkb[self.nodedict['commkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]
    

        for ift in range(submatrixw[0], submatrixw[1]):
            for ikz in range(submatrixk[2][0], submatrixk[2][1]):
                for iky in range(submatrixk[1][0], submatrixk[1][1]):
                    for ikx in range(submatrixk[0][0], submatrixk[0][1]):
                        for js in range(ns):
                            matout[:, :, js, ikx, iky, ikz, ift] = np.linalg.inv(matin[:, :, js ,ikx, iky, ikz, ift])

        return matout

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        submatrixk = self.submatrixkb[self.nodedict['commkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]

        matout = np.zeros_like(mat1, dtype=np.complex128, order='F')
        
        for ift in range(submatrixw[0], submatrixw[1]):
            for ikz in range(submatrixk[2][0], submatrixk[2][1]):
                for iky in range(submatrixk[1][0], submatrixk[1][1]):
                    for ikx in range(submatrixk[0][0], submatrixk[0][1]):
                        matout[:, :, :, ikx, iky, ikz, ift] = QAFort.dyson.flocstc(mat1[:, :, :, ikx, iky, ikz, ift], mat2[:, :, :, ikx, iky, ikz, ift])      


        return matout

    def K2R(self, matk : np.ndarray) -> np.ndarray:

        
        rkvec = self.crystal.kpoint.reshape((self.crystal.rkgrid[0], self.crystal.rkgrid[1], self.crystal.rkgrid[2], 3), order='F')

        norb, _, ns, nkx, nky, nkz, nf = matk.shape
        tempmat = np.zeros_like(matk, dtype=np.complex128, order='F')
        tempmat2 = np.zeros((nkx, nky, nkz), order='F', dtype=np.complex128)
        matr = np.zeros_like(matk, dtype=np.complex128, order='F')

        submatrixk = self.submatrixkf[self.nodedict['commkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]

        for iff in range(submatrixw[0], submatrixw[1]):
            for ikz in range(submatrixk[2][0], submatrixk[2][1]):
                for iky in range(submatrixk[1][0], submatrixk[1][1]):
                    for ikx in range(submatrixk[0][0], submatrixk[0][1]):
                        for js in range(ns):
                            for jorb in range(norb):
                                for iorb in range(norb):
                                    a, m1 = self.crystal.FAtomOrb(iorb)
                                    b, m2 = self.crystal.FAtomOrb(jorb)
                                    delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]

                                    phase = np.exp(2.0j * np.pi * np.dot(rkvec[ikx, iky, ikz], delta))
                                    tempmat[iorb, jorb, js, ikx, iky, ikz, iff] = matk[iorb, jorb, js, ikx, iky, ikz, iff] * phase

        
        
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        tempval = tempmat[iorb, jorb, js, ..., iff]
                        local_slices = self.mpimanager.fft.local_slice(True)
                        tempval2 = self.mpimanager.Backward(tempval[local_slices])
                        local_slice = self.mpimanager.fft.local_slice(forward_output=False)
                        tempmat2[local_slice] = tempval2*1/(nkx*nky*nkz)

                        self.commk.Allreduce(tempmat2, matr[iorb, jorb, js, :, :, :, iff])

        return matr
    

    def R2K(self, matr : np.ndarray) -> np.ndarray:

        rkvec = self.crystal.kpoint.reshape((self.crystal.rkgrid[0], self.crystal.rkgrid[1], self.crystal.rkgrid[2], 3), order='F')

        norb, _, ns, nkx, nky, nkz, nf = matr.shape
        tempmat = np.zeros_like(matr, dtype=np.complex128, order='F')
        tempmat2 = np.zeros((nkx, nky, nkz), order='F', dtype=np.complex128)
        matk = np.zeros_like(matr, dtype=np.complex128, order='F')


        submatrixk = self.submatrixkb[self.nodedict['commkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]

        
        for iff in range(submatrixw[0], submatrixw[1]):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        
                        tempval = matr[iorb, jorb, js, ..., iff]
                        local_slices = self.mpimanager.fft.local_slice(False)
                        tempval2 = self.mpimanager.Forward(tempval[local_slices])
                        local_slice = self.mpimanager.fft.local_slice(forward_output=True)
                        tempmat2[local_slice] = tempval2
                        # if (self.commk.Get_rank() == 0):
                        #     print(f"Data : {tempmat2[local_slice]}")
                        self.commk.Allreduce(tempmat2, tempmat[iorb, jorb, js, :, :, :, iff])
                        # if (self.commk.Get_rank() == 0):
                        #     print(f"tempmat : {tempmat[iorb, jorb, js, :, :, :, iff]}")
        
        # for iff in range(submatrixw[0], submatrixw[1]):
            for ikz in range(submatrixk[2][0], submatrixk[2][1]):
                for iky in range(submatrixk[1][0], submatrixk[1][1]):
                    for ikx in range(submatrixk[0][0], submatrixk[0][1]):
                        for js in range(ns):
                            for jorb in range(norb):
                                for iorb in range(norb):
                                    a, m1 = self.crystal.FAtomOrb(iorb)
                                    b, m2 = self.crystal.FAtomOrb(jorb)
                                    delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]

                                    phase = np.exp(-2.0j * np.pi * np.dot(rkvec[ikx, iky, ikz], delta))
                                    matk[iorb, jorb, js, ikx, iky, ikz, iff] = tempmat[iorb, jorb, js, ikx, iky, ikz, iff] * phase

        return matk
    
    # def F2T(self, matf : np.ndarray) -> np.ndarray:

        




class BLatDynMPI(object):

    def __init__(self, crystal : Crystal, ftgrid : FTGrid, nk : int, nw : int, ntau : int, nprock : int, nprocw : int, mpimanager : MPIManager):

        self.crystal = crystal
        self.ftgrid = ftgrid
        self.nk = nk
        self.nw = nw
        self.nprock = nprock
        self.nprocw = nprocw
        self.mpimanager = mpimanager
        self.nodedict = mpimanager.Quary(nk, nw, ntau, nprock, nprocw, self.crystal.rkgrid)

        self.commk = self.nodedict['commk']
        self.commw = self.nodedict['commf']
        self.submatrixkf = self.nodedict['submatrixkf']
        self.submatrixkb = self.nodedict['submatrixkb']
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



    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        submatrixk = self.submatrixkb[self.nodedict['commkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]

        matout = np.zeros_like(mat1, dtype=np.complex128, order='F')
        
        for ift in range(submatrixw[0], submatrixw[1]):
            for ikz in range(submatrixk[2][0], submatrixk[2][1]):
                for iky in range(submatrixk[1][0], submatrixk[1][1]):
                    for ikx in range(submatrixk[0][0], submatrixk[0][1]):
                        matout[:, :, :, :, ikx, iky, ikz, ift] = QAFort.dyson.blocstc(mat1[:, :, :, :, ikx, iky, ikz, ift], mat2[:, :, :, :, ikx, iky, ikz, ift])      


        return matout

    
    def Inverse(self, matin : np.ndarray) -> np.ndarray:

        
        norb, _, ns, _, nkx, nky, nkz, nft = matin.shape

        matout = np.zeros((norb, norb, ns, ns, nkx, nky, nkz, nft), dtype=np.complex128, order='F')
        tempmat = np.zeros((norb*ns, norb*ns, nkx, nky, nkz, nft), dtype=np.complex128, order='F')
        tempmat2 = np.zeros((norb*ns, norb*ns, nkx, nky, nkz, nft), dtype=np.complex128, order='F')

        submatrixk = self.submatrixkb[self.nodedict['commkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]

        # self.crystal.OrbSpin2Composite()

        for ift in range(submatrixw[0], submatrixw[1]):
            for ikz in range(submatrixk[2][0], submatrixk[2][1]):
                for iky in range(submatrixk[1][0], submatrixk[1][1]):
                    for ikx in range(submatrixk[0][0], submatrixk[0][1]):
                        tempmat[:, :, ikx, iky, ikz, ift] = self.crystal.OrbSpin2Composite(matin[:, :, :, :, ikx, iky, ikz, ift])

                        tempmat2[:, :, ikx, iky, ikz, ift] = np.linalg.inv(tempmat[:, :, ikx, iky, ikz, ift])

                        matout[:, :, :, :, ikx, iky, ikz, ift] = self.crystal.Composite2OrbSpin(tempmat2[:, :, ikx, iky, ikz, ift])

        return matout

    def K2R(self, matk : np.ndarray) -> np.ndarray:

        
        rkvec = self.crystal.kpoint.reshape((self.crystal.rkgrid[0], self.crystal.rkgrid[1], self.crystal.rkgrid[2], 3), order='F')

        norb, _, ns, _, nkx, nky, nkz, nf = matk.shape
        tempmat = np.zeros_like(matk, dtype=np.complex128, order='F')
        tempmat2 = np.zeros((nkx, nky, nkz), order='F', dtype=np.complex128)
        matr = np.zeros_like(matk, dtype=np.complex128, order='F')

        submatrixk = self.submatrixkf[self.nodedict['commkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]

        for iff in range(submatrixw[0], submatrixw[1]):
            for ikz in range(submatrixk[2][0], submatrixk[2][1]):
                for iky in range(submatrixk[1][0], submatrixk[1][1]):
                    for ikx in range(submatrixk[0][0], submatrixk[0][1]):
                        for ks in range(ns):
                            for js in range(ns):
                                for jorb in range(norb):
                                    for iorb in range(norb):
                                        a, _ = self.crystal.FAtomOrb(iorb)
                                        b, _ = self.crystal.FAtomOrb(jorb)
                                        delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]

                                        phase = np.exp(2.0j * np.pi * np.dot(rkvec[ikx, iky, ikz], delta))
                                        tempmat[iorb, jorb, js, ks, ikx, iky, ikz, iff] = matk[iorb, jorb, js, ks, ikx, iky, ikz, iff] * phase

        
        for iff in range(submatrixw[0], submatrixw[1]):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            tempval = tempmat[iorb, jorb, js, ks, ..., iff]
                            local_slices = self.mpimanager.fft.local_slice(True)
                            tempval2 = self.mpimanager.Backward(tempval[local_slices])
                            local_slice = self.mpimanager.fft.local_slice(forward_output=False)
                            tempmat2[local_slice] = tempval2*1/(nkx*nky*nkz)

                            self.commk.Allreduce(tempmat2, matr[iorb, jorb, js, ks, :, :, :, iff])

        return matr

    def R2K(self, matr : np.ndarray) -> np.ndarray:

        rkvec = self.crystal.kpoint.reshape((self.crystal.rkgrid[0], self.crystal.rkgrid[1], self.crystal.rkgrid[2], 3), order='F')

        norb, _, ns, _, nkx, nky, nkz, nf = matr.shape
        tempmat = np.zeros_like(matr, dtype=np.complex128, order='F')
        tempmat2 = np.zeros((nkx, nky, nkz), order='F', dtype=np.complex128)
        matk = np.zeros_like(matr, dtype=np.complex128, order='F')


        submatrixk = self.submatrixkb[self.nodedict['commkrank']]
        submatrixw = self.submatrixw[self.nodedict['commfrank']]

        
        for iff in range(submatrixw[0], submatrixw[1]):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            
                            tempval = matr[iorb, jorb, js, ks, ..., iff]
                            local_slices = self.mpimanager.fft.local_slice(False)
                            tempval2 = self.mpimanager.Forward(tempval[local_slices])
                            local_slice = self.mpimanager.fft.local_slice(forward_output=True)
                            tempmat2[local_slice] = tempval2
                            # if (self.commk.Get_rank() == 0):
                            #     print(f"Data : {tempmat2[local_slice]}")
                            self.commk.Allreduce(tempmat2, tempmat[iorb, jorb, js, ks, :, :, :, iff])
                            # if (self.commk.Get_rank() == 0):
                            #     print(f"tempmat : {tempmat[iorb, jorb, js, :, :, :, iff]}")
        
        for iff in range(submatrixw[0], submatrixw[1]):
            for ikz in range(submatrixk[2][0], submatrixk[2][1]):
                for iky in range(submatrixk[1][0], submatrixk[1][1]):
                    for ikx in range(submatrixk[0][0], submatrixk[0][1]):
                        for ks in range(ns):
                            for js in range(ns):
                                for jorb in range(norb):
                                    for iorb in range(norb):
                                        a, _ = self.crystal.FAtomOrb(iorb)
                                        b, _ = self.crystal.FAtomOrb(jorb)
                                        delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]

                                        phase = np.exp(-2.0j * np.pi * np.dot(rkvec[ikx, iky, ikz], delta))
                                        matk[iorb, jorb, js, ks, ikx, iky, ikz, iff] = tempmat[iorb, jorb, js, ks, ikx, iky, ikz, iff] * phase

        return matk




        
    # @numba.jit
    # def K2R(self, matk : np.ndarray) -> np.ndarray:

    #     rkvec = self.crystal.kpoint
    #     rkgrid = self.crystal.rkgrid


    #     commk = self.commk
    #     submatrixk = self.submatrixkb[self.nodedict['comkrank']]
    #     submatrixw = self.submatrixw[self.nodedict['commfrank']]
        
    #     # subk0, subk1 = submatrixk
    #     subw0, subw1 = submatrixw
    #     # local_nrk = subk1 - subk0
    #     local_nft = subw1 - subw0

    #     # nrk = local_nrk
    #     nrk = len(rkvec)
    #     nft = local_nft

    #     matr = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
    #     tempmat = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
    #     tempmat2 = np.zeros((norb, norb, ns, rkgrid[0], rkgrid[1], rkgrid[2], nft), dtype=np.complex128, order='F')

    #     for loc_ift, ift in enumerate(range(subw0, subw1)):
    #         # for loc_irk, global_irk in enumerate(range(subk0, subk1)):
    #         for irk in range(nrk):
    #             for js in range(ns):
    #                 for jorb in range(norb):
    #                     for iorb in range(norb):
    #                         a, m1 = self.crystal.FAtomOrb(iorb)
    #                         b, m2 = self.crystal.FAtomOrb(jorb)
    #                         delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
    #                         phase = np.exp(2.0j*np.pi*np.dot(rkvec[irk], delta))
    #                         tempmat[iorb, jorb, js, irk, loc_ift] = matk[iorb, jorb, js, irk, loc_ift] * phase

    #     tempmat = self.K2K3D(tempmat)
    #     # fft = PFFT(comm=commk, shape=rkgrid, axes=(0, 1, 2), dtype=np.complex128)

    #     # @numba.jit
    #     # for loc_ift in range(nft):
    #     #     for js in range(ns):
    #     #         for jorb in range(norb):
    #     #             for iorb in range(norb):
    #     #                 tempmat2[iorb, jorb, js, :, :, :, loc_ift] = fft.backward(
    #     #                     tempmat[iorb, jorb, js, :, :, :, loc_ift],
    #     #                     normalization=True
    #     #                 )
    #     tempmat2 = mpifft(tempmat, False)

    #     matr = tempmat2.reshape((norb, norb, ns, nrk, nft), order='F')

    #     return matr

    # def R2K(self, matr : np.ndarray) -> np.ndarray:

    #     rkvec = self.crystal.kpoint
    #     rkgrid = self.crystal.rkgrid

    #     norb = matr.shape[0]
    #     ns = matr.shape[2]

    #     commk = self.commk
    #     # submatrixk = self.submatrixk[self.nodedict['comkrank']]
    #     submatrixw = self.submatrixw[self.nodedict['commfrank']]

    #     # subk0, subk1 = submatrixk
    #     subw0, subw1 = submatrixw
    #     # local_nrk = subk1 - subk0
    #     local_nft = subw1 - subw0

    #     # nrk = local_nrk
    #     nrk = len(rkvec)
    #     nft = local_nft

    #     matk = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
    #     # tempmat = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
    #     tempmat = np.zeros((norb, norb, ns, rkgrid[0], rkgrid[1], rkgrid[2], nft), dtype=np.complex128, order='F')
    #     # tempmat2 = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')

    #     tempmat2 = matr.reshape((norb, norb, ns, rkgrid[0], rkgrid[1], rkgrid[2], nft), order='F')
    #     fft = PFFT(comm=commk, shape=rkgrid, axes=(0, 1, 2), dtype=np.complex128)
    #     for loc_ift in range(nft):
    #         for js in range(ns):
    #             for jorb in range(norb):
    #                 for iorb in range(norb):
    #                     tempmat[iorb, jorb, js, :, :, :, loc_ift] = fft.forward(
    #                         tempmat2[iorb, jorb, js, :, :, :, loc_ift]
    #                     )

    #     # matr = tempmat2.reshape((norb, norb, ns, nrk, nft), order='F')
    #     tempmat = tempmat.reshape((norb, norb, ns, nrk, nft), order='F')
    #     for loc_ift, ift in enumerate(range(subw0, subw1)):
    #         # for loc_irk, global_irk in enumerate(range(subk0, subk1)):
    #         for irk in range(nrk):
    #             for js in range(ns):
    #                 for jorb in range(norb):
    #                     for iorb in range(norb):
    #                         a, m1 = self.crystal.FAtomOrb(iorb)
    #                         b, m2 = self.crystal.FAtomOrb(jorb)
    #                         delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
    #                         phase = np.exp(2.0j*np.pi*np.dot(rkvec[irk], delta))
    #                         matk[iorb, jorb, js, irk, loc_ift] = tempmat[iorb, jorb, js, irk, loc_ift] * phase

    #     return matk
    
    
        



# class FLatDynIrrCoh(FLatDynMPI):

#     head = None
#     tail = None


# class FLatDynFullCoh(FLatDynMPI):

#     head = None
#     tail = None

# class FLatDynFineFine(FLatDynMPI):

#     head = None
#     tail = None
