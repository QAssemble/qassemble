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
            self.crystal = crystal
            
            wtemp = np.arange(nf)
            wchunk = np.array_split(wtemp, nprocf)
            submatrixw = [(chunk[0], chunk[-1]+1) for chunk in wchunk]
            nodedict['submatrixw'] = submatrixw
            self.floc = self.FTLocalGlobal(submatrixw)

            tautemp = np.arange(ntau)
            tauchunk = np.array_split(tautemp, nprocf)
            submatrixtau = [(chunk[0], chunk[-1]+1) for chunk in tauchunk]
            nodedict['submatrixtau'] = submatrixtau
            self.tloc = self.FTLocalGlobal(submatrixtau)

            kidx = self.rank // nprock
            widx = self.rank % nprock

            commk = self.comm.Split(color=kidx, key=widx)
            commf = self.comm.Split(color=widx, key=kidx)
            commtau = self.comm.Split(color=widx, key=kidx)

            # mpifft = MPIFFT(commk,kgrid)
            self.fft = self.FFT(commk,crystal.rkgrid)
            self.klocal = self.CreateMPICompositeIndex(self.slicef)
            self.rlocal = self.CreateMPICompositeIndex(self.sliceb)

            self.klocal2 = self.CreateMPICompositeIndex2(self.localshapef)
            self.rlocal2 = self.CreateMPICompositeIndex2(self.localshapeb)
            
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
    
    def FTLocalGlobal(self, submatrixf : list):

        floc = {}
        
        for irank in range(len(submatrixf)):
            floc[irank] = {}
            i = 0
            f = submatrixf[irank]
            for ifreq in range(f[0], f[1]):
                floc[irank][i] = ifreq
                i += 1

        return floc
    
    def FGlobal2Local(self, fidx : int) -> list:

        for key, val in self.floc.items():
            for key2, val2 in val.items():
                if (val2 == fidx):
                    return (key, key2)
    
    def FLocal2Global(self, flist : list) -> int:

        rank, floc = flist
        return self.floc[rank][floc]

    def TGlobal2Local(self, tidx : int) -> list:

        for key, val in self.tloc.items():
            for key2, val2 in val.items():
                if (val2 == tidx):
                    return (key, key2)
                
    def TLocal2Global(self, tlist : list) -> int:

        rank, tloc = tlist

        return self.tloc[rank][tloc]

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
    
    def CreateMPICompositeIndex2(self, rank_slices):
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
            nx, ny, nz = slices

            local_index = 0
            local_dict = {}
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
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
            kidx (int) : Global k-index corresponding to the local k-index.
        """
        rank, local_index = klocal
        return self.klocal2global[rank][local_index]
    
    def KLocalList(self, klocal : list) -> list:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klocal (list): Local k-index in the form [rank, local_index].
        Returns:
            klist (list): Local 3D k-index corresponding to the local k-index.
        """

        rank, local_index = klocal

        return [rank, self.klocal[rank][local_index]]
    
    def KListLocal(self, klist : list) -> list:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klist (list): Local 3D k-index corresponding to the local k-index.
        Returns:
            klocal (list): Local k-index in the form [rank, local_index].
        """

        rank, k3d = klist

        for key, val in self.klocal[rank].items():
            if (k3d == val):
                return [rank ,key]
            
    def KLocalList2(self, klocal : list) -> list:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klocal (list): Local k-index in the form [rank, local_index].
        Returns:
            klist (list): Local 3D k-index corresponding to the local k-index.
        """

        rank, local_index = klocal

        return [rank, self.klocal2[rank][local_index]]
    
    def KListLocal2(self, klist : list) -> list:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klist (list): Local 3D k-index corresponding to the local k-index.
        Returns:
            klocal (list): Local k-index in the form [rank, local_index].
        """

        rank, k3d = klist

        for key, val in self.klocal2[rank].items():
            if (k3d == val):
                return [rank ,key]

    
    def RGlobal2Local(self, ridx : int) -> list:
        """
        Convert a global r-index to its corresponding local rank and index.

        Args :
            ridx (int): Global k-index to convert.
        Returns:
            (rank, local_index) list: A list containing the rank and local index corresponding to the global k-index.

        """

        for key, val in self.rlocal2global.items():
            for key2, val2 in val.items():
                if (ridx == val2):
                    return [key, key2]
                
    def RLocal2Global(self, rlocal : list) -> int:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klocal (list): Local k-index in the form [rank, local_index].
        Returns:
            int: Global k-index corresponding to the local k-index.
        """
        rank, local_index = rlocal
        return self.rlocal2global[rank][local_index]
    
    def RLocalList(self, rlocal : list) -> list:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klocal (list): Local k-index in the form [rank, local_index].
        Returns:
            klist (list): Local 3D k-index corresponding to the local k-index.
        """

        rank, local_index = rlocal

        return [rank, self.rlocal[rank][local_index]]
    
    def RListLocal(self, rlist : list) -> list:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klist (list): Local 3D k-index corresponding to the local k-index.
        Returns:
            klocal (list): Local k-index in the form [rank, local_index].
        """

        rank, k3d = rlist

        for key, val in self.rlocal[rank].items():
            if (k3d == val):
                return [rank ,key]
            
    def RLocalList2(self, rlocal : list) -> list:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klocal (list): Local k-index in the form [rank, local_index].
        Returns:
            klist (list): Local 3D k-index corresponding to the local k-index.
        """

        rank, local_index = rlocal

        return [rank, self.rlocal2[rank][local_index]]
    
    def RListLocal2(self, rlist : list) -> list:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klist (list): Local 3D k-index corresponding to the local k-index.
        Returns:
            klocal (list): Local k-index in the form [rank, local_index].
        """

        rank, k3d = rlist

        for key, val in self.rlocal2[rank].items():
            if (k3d == val):
                return [rank ,key]
            
    def K2K3D(self, commk : MPI.COMM_WORLD, matin : np.ndarray) -> np.ndarray:

        
        rank = commk.Get_rank()
        (nkx, nky, nkz) = self.localshapef[rank]
        if (len(matin) != nkx*nky*nkz):
            print(self.K2K3D.__name__)
            print("Input array is wrong. Check the array dimension")
            print(matin.shape, nkx, nky, nkz)
            sys.exit()

        
        matout = np.zeros((nkx, nky, nkz), dtype=np.complex128, order='F')

        for ik in range(len(matin)):
            _, [ikx, iky, ikz] = self.KLocalList2([rank, ik])
            # print(rank, ik, ikx, iky, ikz)
            # print(nkx, nky, nkz)
            matout[ikx, iky, ikz] = matin[ik]

        return matout
    
    def K3D2K(self, commk : MPI.COMM_WORLD,  matin : np.ndarray) -> np.ndarray:


        rank = commk.Get_rank()
        (nkx, nky, nkz) = matin.shape
        if (nkx*nky*nkz != len(self.klocal[rank])):
            print(self.K3D2K.__name__)
            print("Input array is wrong. Check the array dimension")
            sys.exit()

        nk =len(self.klocal[rank]) 
        matout = np.zeros((nk), dtype=np.complex128, order='F')

        for ik in range(nk):
            _, [ix, iy, iz] = self.KLocalList2([rank, ik])
            matout[ik] = matin[ix, iy, iz]

        return matout
    
    def R2R3D(self, commk : MPI.COMM_WORLD, matin : np.ndarray) -> np.ndarray:

        
        rank = commk.Get_rank()
        (nx, ny, nz) = self.localshapeb[rank]
        if (len(matin) != nx*ny*nz):
            print(self.R2R3D.__name__)
            print("Input array is wrong. Check the array dimension")
            print(matin.shape, nx, ny, nz)
            sys.exit()


        matout = np.zeros((nx, ny, nz), dtype=np.complex128, order='F')

        for ir in range(len(matin)):
            _, [ix, iy, iz] = self.RLocalList2([rank, ir])
            # print(rank, ik, ikx, iky, ikz)
            # print(nkx, nky, nkz)
            matout[ix, iy, iz] = matin[ir]

        return matout
    
    def R3D2R(self, commk : MPI.COMM_WORLD, matin : np.ndarray) -> np.ndarray:

        
        rank = commk.Get_rank()
        (nx, ny, nz) = matin.shape
        if (nx*ny*nz != len(self.rlocal[rank])):
            print(self.R3D2R.__name__)
            print("Input array is wrong. Check the array dimension")
            sys.exit()

        nr =len(self.rlocal[rank]) 
        matout = np.zeros((nr), dtype=np.complex128, order='F')

        for ir in range(nr):
            _, [ix, iy, iz] = self.RLocalList2([rank, ir])
            matout[ir] = matin[ix, iy, iz]

        return matout
                



    

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

        with h5py.File(hdf5file, 'a', driver='mpio', comm = self.mpimanager.comm) as file:
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

        
        norb, _, ns, nk, nft = matin.shape

        matout = np.zeros((norb, norb, ns, nk, nft), dtype=np.complex128, order='F')
        

        for ift in range(nft):
            for ik in range(nk):
                for js in range(ns):
                    matout[:, :, js, ik, ift] = np.linalg.inv(matin[:, :, js ,ik, ift])

        return matout

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        # norb, _, ns, nk, nft = mat1.shape
        nk = mat1.shape[3]
        nft = mat1.shape[4]
        matout = np.zeros_like(mat1, dtype=np.complex128, order='F')
        
        for ift in range(nft):
            for ik in range(nk):
                matout[:, :, :, ik, ift] = QAFort.dyson.flocstc(mat1[:, :, :, ik, ift], mat2[:, :, :, ik, ift])      


        return matout

    def K2R(self, matk : np.ndarray) -> np.ndarray:

        
        # rkvec = self.crystal.kpoint.reshape((self.crystal.rkgrid[0], self.crystal.rkgrid[1], self.crystal.rkgrid[2], 3), order='F')

        
        # norb, _, ns, nkx, nky, nkz, nf = matk.shape
        norb, _, ns, nk, nf = matk.shape
        rkvec = self.crystal.kpoint
        rank = self.nodedict['commkrank']
        (nkx, nky, nkz) = self.mpimanager.localshapef[self.nodedict['commkrank']]
        nkglobal = self.crystal.rkgrid[0] * self.crystal.rkgrid[1] * self.crystal.rkgrid[2]
        if (nk != nkx * nky * nkz):
            print(f"Error: nk ({nk}) does not match local shape ({nkx}, {nky}, {nkz})")
            sys.exit()
        (nx, ny, nz) = self.mpimanager.localshapeb[self.nodedict['commkrank']]
        tempmat = np.zeros((norb, norb, ns, nk, nf), dtype=np.complex128, order='F')
        tempmat2 = np.zeros((nx, ny, nz), order='F', dtype=np.complex128)
        
        nr = len(self.mpimanager.rlocal[rank])
        matr = np.zeros((norb, norb, ns, nr, nf), dtype=np.complex128, order='F')

        for iff in range(nf):            
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        for ik in range(nk):
                            a, _ = self.crystal.FAtomOrb(iorb)
                            b, _ = self.crystal.FAtomOrb(jorb)
                            delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                            kidx = self.mpimanager.KLocal2Global([rank, ik])
                            phase = np.exp(2.0j * np.pi * np.dot(rkvec[kidx], delta))
                            tempmat[iorb, jorb, js, ik, iff] = matk[iorb, jorb, js, ik, iff] * phase
                        # --------------------------------------------------------------------------- #
                        tempval = self.mpimanager.K2K3D(self.commk, tempmat[iorb, jorb, js, :, iff])
                        tempval2 = self.mpimanager.Backward(tempval)
                        tempmat2 = tempval2*1/(nkglobal)

                        matr[iorb, jorb, js,:,iff] = self.mpimanager.R3D2R(self.commk, tempmat2)

        return matr
    

    def R2K(self, matr : np.ndarray) -> np.ndarray:

        rkvec = self.crystal.kpoint

        norb, _, ns, nr, nf = matr.shape
        rank = self.nodedict['commkrank']
        (nx, ny, nz) = self.mpimanager.localshapeb[rank]
        (nkx, nky, nkz) = self.mpimanager.localshapef[rank]
        if (nr != nx * ny * nz):
            print(f"Error: nk ({nr}) does not match local shape ({nx}, {ny}, {nz})")
            sys.exit()
        tempmat = np.zeros_like(matr, dtype=np.complex128, order='F')
        tempmat2 = np.zeros((nkx, nky, nkz), order='F', dtype=np.complex128)
        matk = np.zeros((norb, norb, ns, nkx*nky*nkz, nf), dtype=np.complex128, order='F')

        
        for iff in range(nf):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        
                        tempval = self.mpimanager.R2R3D(self.commk, matr[iorb, jorb, js, :, iff])
                        tempval2 = self.mpimanager.Forward(tempval)
                        tempmat2 = tempval2
                        tempmat[iorb, jorb, js, :, iff] = self.mpimanager.K3D2K(self.commk, tempmat2)
                        
        
                        for ik in range(nkx*nky*nkz):
                            a, _ = self.crystal.FAtomOrb(iorb)
                            b, _ = self.crystal.FAtomOrb(jorb)
                            delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                            kidx = self.mpimanager.KLocal2Global([rank, ik])
                            phase = np.exp(-2.0j * np.pi * np.dot(rkvec[kidx], delta))
                            matk[iorb, jorb, js, ik, iff] = tempmat[iorb, jorb, js, ik, iff] * phase

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
        self.nodedict = mpimanager.Quary(nk, nw, ntau, nprock, nprocw, self.crystal)

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

        nk = mat1.shape[4]
        nft = mat1.shape[5]
        matout = np.zeros_like(mat1, dtype=np.complex128, order='F')
        
        for ift in range(nft):
            for ik in range(nk):
                matout[:, :, :, :, ik, ift] = QAFort.dyson.blocstc(mat1[:, :, :, :, ik, ift], mat2[:, :, :, :, ik, ift])      

        return matout

    
    def Inverse(self, matin : np.ndarray) -> np.ndarray:

        
        norb, _, ns, _, nk, nft = matin.shape

        matout = np.zeros((norb, norb, ns, ns, nk, nft), dtype=np.complex128, order='F')
        tempmat = np.zeros((norb*ns, norb*ns, nk, nft), dtype=np.complex128, order='F')
        tempmat2 = np.zeros((norb*ns, norb*ns, nk, nft), dtype=np.complex128, order='F')


        for ift in range(nft):
            for ik in range(nk):
                tempmat[:, :, ik, ift] = self.crystal.OrbSpin2Composite(matin[:, :, :, :, ik, ift])

                tempmat2[:, :, ik, ift] = np.linalg.inv(tempmat[:, :, ik, ift])

                matout[:, :, :, :, ik, ift] = self.crystal.Composite2OrbSpin(tempmat2[:, :, ik, ift])

        return matout

    def K2R(self, matk : np.ndarray) -> np.ndarray:


        norb, _, ns, _, nk, nf = matk.shape
        rkvec = self.crystal.kpoint
        rank = self.nodedict['commkrank']
        (nkx, nky, nkz) = self.mpimanager.localshapef[rank]
        (nx, ny, nz) = self.mpimanager.localshapeb[rank]
        nkglobal = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        if (nk != nkx*nky*nkz):
            print(f"Error: nk ({nk}) does not match local shape ({nkx}, {nky}, {nkz})")
            sys.exit()
        tempmat = np.zeros((norb, norb, ns, ns, nk, nf), dtype=np.complex128, order='F')
        tempmat2 = np.zeros((nx, ny, nz), order='F', dtype=np.complex128)
        nr = len(self.mpimanager.rlocal[rank])
        matr = np.zeros((norb, norb, ns, ns, nr, nf), dtype=np.complex128, order='F')


        for iff in range(nf):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            for ik in range(nk):
                                a, _ = self.crystal.BAtomOrb(iorb)
                                b, _ = self.crystal.BAtomOrb(jorb)
                                delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                                kidx = self.mpimanager.KLocal2Global([rank, ik])
                                phase = np.exp(2.0j * np.pi * np.dot(rkvec[kidx], delta))
                                tempmat[iorb, jorb, js, ks, ik, iff] = matk[iorb, jorb, js, ks, ik, iff] * phase
                            # ----------------------------------------------------------------------------------- #
                            tempval = self.mpimanager.K2K3D(self.commk, tempmat[iorb, jorb, js, ks, :, iff])
                            tempval2 = self.mpimanager.Backward(tempval)
                            tempmat2 = tempval2 * 1/(nkglobal)

                            matr[iorb, jorb, js, ks, :, iff] = self.mpimanager.R3D2R(self.commk, tempmat2)

        return matr

            

        
        

    def R2K(self, matr : np.ndarray) -> np.ndarray:

        

        norb, _, ns, _, nr, nf = matr.shape
        rkvec = self.crystal.kpoint
        rank = self.nodedict['commkrank']
        (nx, ny, nz) = self.mpimanager.localshapeb[rank]
        (nkx, nky, nkz) = self.mpimanager.localshapef[rank]

        if (nr != nx * ny * nz):
            print(f"Error: nk ({nr}) does not match local shape ({nx}, {ny}, {nz})")
            sys.exit()

        tempmat = np.zeros_like(matr, dtype=np.complex128, order='F')
        tempmat2 = np.zeros((nx, ny, nz), dtype=np.complex128, order='F')
        matk = np.zeros((norb, norb, ns, ns, nkx*nky*nkz, nf), dtype=np.complex128, order='F')

        for iff in range(nf):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            tempval = self.mpimanager.R2R3D(self.commk, matr[iorb, jorb, js, ks, :, iff])
                            tempval2 = self.mpimanager.Forward(tempval)
                            tempmat2 = tempval2
                            tempmat[iorb, jorb, js, ks, :, iff] = self.mpimanager.K3D2K(self.commk, tempmat2)

                            for ik in range(nkx*nky*nkz):
                                a, _ = self.crystal.BAtomOrb(iorb)
                                b, _ = self.crystal.BAtomOrb(jorb)

                                delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
                                kidx = self.mpimanager.KLocal2Global([rank, ik])
                                phase = np.exp(-2.0j * np.pi * np.dot(rkvec[kidx], delta))
                                matk[iorb, jorb, js, ks, ik, iff] = tempmat[iorb, jorb, js, ks, ik, iff] * phase

        return matk


