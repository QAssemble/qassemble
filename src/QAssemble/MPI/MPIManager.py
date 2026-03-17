from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import os, sys
import numba
import scipy.linalg
from .Crystal import Crystal
from .FTGrid import FTGrid
import numpy as np
import scipy
import h5py
import finufft
qapath = os.environ.get('QAssemble','')
sys.path.append(qapath+'/src/QAssemble/modules')
import QAFort



class MPIManager(object):


    def __init__(self, comm : MPI.COMM_WORLD, crystal : Crystal):


        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.crystal = crystal
        if (self.rank == 0):
            print("Parallelization with MPI Start")
        required = MPI.THREAD_MULTIPLE
        provided = MPI.Query_thread()
        if provided < required:
            if self.rank == 0:
                print("MPI does not support THREAD_MULTIPLE")
            sys.exit(1)
        self.mpidict = {}
        self.fft = None
        self.arr = None
        self.arrT = None
        self.slicef = None
        self.sliceb = None
        self.localshapef = None
        self.localshapeb = None

    def Quary(self, nk : int, nf : int, ntau : int, nprock : int, nprocf : int, shape : list):
        
        self.nprock = nprock
        self.nprocf = nprocf
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
            self.fft = self.FFT(commk,shape)
            self.klocal = self.CreateMPICompositeIndex(self.slicef)
            self.rlocal = self.CreateMPICompositeIndex(self.sliceb)

            self.klocal2 = self.CreateMPICompositeIndex2(self.localshapef)
            self.rlocal2 = self.CreateMPICompositeIndex2(self.localshapeb)
            
            self.klocal2global = self.MappingGlobal2Local(commk, self.klocal, shape)
            self.rlocal2global = self.MappingGlobal2Local(commk, self.rlocal, shape)

            nodedict['nfreq'] = nf
            nodedict['ntau'] = ntau
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
            nodedict['fft'] = self.fft
            nodedict['arr'] = self.arr
            nodedict['arrT'] = self.arrT
            nodedict['klocal'] = self.klocal
            nodedict['rlocal'] = self.rlocal
            nodedict['floc'] = self.floc
            nodedict['tloc'] = self.tloc
            nodedict['klocal2global'] = self.klocal2global
            nodedict['rlocal2global'] = self.rlocal2global
            # nodedict['RLocal2Global'] = self.RLocal2Global
            # nodedict['RGlobal2Local'] = self.RGlobal2Local
            # nodedict['KLocal2Global'] = self.KLocal2Global
            # nodedict['KGlobal2Local'] = self.KGlobal2Local
            # nodedict['FLocal2Global'] = self.FLocal2Global
            # nodedict['FGlobal2Local'] = self.FGlobal2Local
            # nodedict['TLocal2Global'] = self.TLocal2Global
            # nodedict['TGlobal2Local'] = self.TGlobal2Local
            nodedict['grid'] = shape

            self.mpidict[(nk, nf, ntau, nprock, nprocf)] = nodedict

            # del mpifft
            # commk, commw, submatrixk, submatrixw, commk.rank, commk.size, commw.rank, commw.size,
            return nodedict
        
    def FFT(self, comm = MPI.COMM_WORLD, shape : list = None):

        if (shape[2] == 1):
            fft = PFFT(comm = comm, shape=shape, axes=(0, 1, 2), dtype=np.complex128, grid=(-1,))
        else:
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
    
    def FGlobal2Local(self, fidx : int, nodedict : dict) -> list:

        floc = nodedict['floc']
        for key, val in floc.items():
            for key2, val2 in val.items():
                if (val2 == fidx):
                    return (key, key2)
    
    def FLocal2Global(self, flist : list, nodedict : dict) -> int:

        rank, floc = flist
        flocdict = nodedict['floc']
        return flocdict[rank][floc]

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
    
    def MappingGlobal2Local(self, commk, localdict : dict, grid : list = None) -> dict:
        mapping = {}
        kind = self.crystal.K2K3D(grid)
        for irank in range(commk.Get_size()):
            mapping[irank] = {}
            for key, value in localdict[irank].items():
                kidx = self.crystal.MergeKind(value, kind)
                mapping[irank][key] = kidx

        return mapping
    
    def KGlobal2Local(self, kidx : int, klocal2global : dict = None) -> list:
        """
        Convert a global k-index to its corresponding local rank and index.

        Args :
            kidx (int): Global k-index to convert.
        Returns:
            (rank, local_index) list: A list containing the rank and local index corresponding to the global k-index.

        """
        if (klocal2global is None):
            klocal2global = self.klocal2global
        for key, val in klocal2global.items():
            for key2, val2 in val.items():
                if (kidx == val2):
                    return [key, key2]
                
    def KLocal2Global(self, klocal : list, klocal2global : dict = None) -> int:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klocal (list): Local k-index in the form [rank, local_index].
        Returns:
            kidx (int) : Global k-index corresponding to the local k-index.
        """
        if (klocal2global is None):
            klocal2global = self.klocal2global
        rank, local_index = klocal
        return klocal2global[rank][local_index]
    
    def KLocalList(self, klist : list, klocal : dict = None) -> list:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klocal (list): Local k-index in the form [rank, local_index].
        Returns:
            klist (list): Local 3D k-index corresponding to the local k-index.
        """
        if (klocal is None):
            klocal = self.klocal
        rank, local_index = klist

        return [rank, klocal[rank][local_index]]

    def KListLocal(self, klist : list, klocal : dict = None) -> list:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klist (list): Local 3D k-index corresponding to the local k-index.
        Returns:
            klocal (list): Local k-index in the form [rank, local_index].
        """
        
        if (klocal is None):
            klocal = self.klocal
        rank, k3d = klist

        for key, val in klocal[rank].items():
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

    
    def RGlobal2Local(self, ridx : int, rlocal2global : dict = None) -> list:
        """
        Convert a global r-index to its corresponding local rank and index.

        Args :
            ridx (int): Global k-index to convert.
        Returns:
            (rank, local_index) list: A list containing the rank and local index corresponding to the global k-index.

        """

        if (rlocal2global is None):
            rlocal2global = self.rlocal2global

        for key, val in rlocal2global.items():
            for key2, val2 in val.items():
                if (ridx == val2):
                    return [key, key2]
                
    def RLocal2Global(self, rlocal : list, rlocal2global : dict = None) -> int:
        """
        Convert a local k-index to its corresponding global index.

        Args :
            klocal (list): Local k-index in the form [rank, local_index].
        Returns:
            int: Global k-index corresponding to the local k-index.
        """
        if (rlocal2global is None):
            rlocal2global = self.rlocal2global

        rank, local_index = rlocal
        return rlocal2global[rank][local_index]
    
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
    
    def FMPIAllreduce(self, nodedict : dict, matin : np.ndarray) -> np.ndarray:

        nfloc = matin.shape[0]
        commw = nodedict['commf']

        nf = nodedict['nfreq']
        tempmat = np.zeros((nf), dtype=np.complex128, order='F')
        matout = np.zeros((nf), dtype=np.complex128, order='F')

        for iff in range(nfloc):            
            fidx = self.FLocal2Global([commw.Get_rank(), iff], nodedict)
            tempmat[fidx] = matin[iff]

        commw.Allreduce(tempmat, matout, op=MPI.SUM)

        return matout
    
    def TMPIAllreduce(self, commtau : MPI.COMM_WORLD, matin : np.ndarray, ntau : int) -> np.ndarray:

        # if (commtau.Get_rank() == 0):
        #     print("MPI Allreduce Start")

        ntauloc = matin.shape[0]

        tempmat = np.zeros((ntau), dtype=np.complex128, order='F')
        matout = np.zeros((ntau), dtype=np.complex128, order='F')

        for iff in range(ntauloc):
            fidx = self.TLocal2Global([commtau.Get_rank(), iff])
            tempmat[fidx] = matin[iff]

        commtau.Allreduce(tempmat, matout, op=MPI.SUM)
        
        # if (commtau.Get_rank() == 0):
        #     print("MPI Allreduce Finish")

        return matout

    def KMPIAllreduce(self, nodedict : dict, matin : np.ndarray) -> np.ndarray:

        nk = nodedict['grid'][0] * nodedict['grid'][1] * nodedict['grid'][2]
        nkloc = matin.shape[0]

        commk = nodedict['commk']
        tempmat = np.zeros((nk), dtype=np.complex128, order='F')
        matout = np.zeros((nk), dtype=np.complex128, order='F')

        for ik in range(nkloc):            
            kidx = self.KLocal2Global([commk.Get_rank(), ik], nodedict['klocal2global'])
            tempmat[kidx] = matin[ik]

        commk.Allreduce(tempmat, matout, op=MPI.SUM)

        return matout
    
    def RMPIAllreduce(self, nodedict : dict, matin : np.ndarray) -> np.ndarray:

        nr = nodedict['grid'][0] * nodedict['grid'][1] * nodedict['grid'][2]
        nrloc = matin.shape[0]

        commk = nodedict['commk']
        tempmat = np.zeros((nr), dtype=np.complex128, order='F')
        matout = np.zeros((nr), dtype=np.complex128, order='F')

        for ir in range(nrloc):
            ridx = self.RLocal2Global([commk.Get_rank(), ir], nodedict['rlocal2global'])
            tempmat[ridx] = matin[ir]

        commk.Allreduce(tempmat, matout, op=MPI.SUM)
        

        return matout
    
    def FMPIBCast(self, comm : MPI.COMM_WORLD, matin : np.ndarray, idx : int, nodedict) -> np.ndarray:

        (rank, localidx) = self.FGlobal2Local(idx, nodedict)
        if (comm.Get_rank() == rank):
            val = matin[...,localidx]
        else:
            val = None
        matout = comm.bcast(val, root=rank)

        return matout
    
    def TMPIBCast(self, comm : MPI.COMM_WORLD, matin : np.ndarray, idx : int) -> np.ndarray:

        (rank, localidx) = self.TGlobal2Local(idx)
        if (comm.Get_rank() == rank):
            val = matin[...,localidx]
        else:
            val = None
        matout = comm.bcast(val, root=rank)

        return matout


    
    
    

# class FLatDynMPI(object):

#     def __init__(self, crystal : Crystal, ftgrid : FTGrid, nk : int, nw : int, ntau : int, nprock : int, nprocw : int, mpimanager : MPIManager):

#         self.crystal = crystal
#         self.ftgrid = ftgrid
#         self.nk = nk
#         self.nw = nw
#         self.nprock = nprock
#         self.nprocw = nprocw
#         self.mpimanager = mpimanager
#         self.nodedict = mpimanager.Quary(nk, nw, ntau, nprock, nprocw, self.crystal)

#         self.commk = self.nodedict['commk']
#         self.commw = self.nodedict['commf']
#         self.submatrixkf = self.nodedict['submatrixkf']
#         self.submatrixkb = self.nodedict['submatrixkb']
#         self.submatrixw = self.nodedict['submatrixw']

#         self.commtau = self.nodedict['commtau']

#         self.submatrixtau = self.nodedict['submatrixtau']

    
    
#     def MappingGlobal2Local(self, localdict : dict) -> dict:
#         mapping = {}
#         for irank in range(self.commk.Get_size()):
#             mapping[irank] = {}
#             for key, value in localdict[irank].items():
#                 kidx = self.crystal.MergeKind(value)
#                 mapping[irank][key] = kidx

#         return mapping

#     def CheckGroup(self, filepath : str, group : str):

#         with h5py.File(filepath, 'r') as file:
#             return group in file

#     def Save(self, hdf5file : str = None, group : str = None, subgroup : str = None, data : np.ndarray = None, dataname : str = None):

#         with h5py.File(hdf5file, 'a', driver='mpio', comm = self.mpimanager.comm) as file:
#             if (self.CheckGroup(hdf5file, group)):
#                 g =  file[group]
#                 if subgroup in g:
#                     subg = g[subgroup]
#                 else:
#                     subg = g.create_group(subgroup)
#             else:
#                 g = file.create_group(group)
#                 subg = g.create_group(subgroup)

#             subg.create_dataset(dataname, data=data, dtype=np.complex128)

#             return None

#     def Load(self, hdf5file : str = None, group : str = None, subgroup : str = None, data : np.ndarray = None, dataname : str = None):

#         with h5py.File(hdf5file, 'r', driver='mpio', comm=self.mpimanager.comm) as file:
#             if (self.CheckGroup(hdf5file, group)):
#                 g =  file[group]
#                 if subgroup in g:
#                     subg = g[subgroup]
#                     if dataname in subg:
#                         data = subg[dataname][:]
#                         return data
#                     else:
#                         raise KeyError(f"{dataname} not found in {subgroup}")
#                 else:
#                     raise KeyError(f"{subgroup} not found in {group}")
#             else:
#                 raise KeyError(f"{group} not found in {hdf5file}")

#     def Inverse(self, matin : np.ndarray) -> np.ndarray:

        
#         norb, _, ns, nk, nft = matin.shape

#         matout = np.zeros((norb, norb, ns, nk, nft), dtype=np.complex128, order='F')
        

#         for ift in range(nft):
#             for ik in range(nk):
#                 for js in range(ns):
#                     matout[:, :, js, ik, ift] = np.linalg.inv(matin[:, :, js ,ik, ift])

#         return matout

#     def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

#         # norb, _, ns, nk, nft = mat1.shape
#         # nk = mat1.shape[3]
#         # nft = mat1.shape[4]
#         matout = np.zeros_like(mat1, dtype=np.complex128, order='F')
        
#         # for ift in range(nft):
#         #     for ik in range(nk):
#         #         matout[:, :, :, ik, ift] = QAFort.dyson.flocstc(mat1[:, :, :, ik, ift], mat2[:, :, :, ik, ift])      
#         matout = QAFort.dyson.flatdyn(mat1, mat2)


#         return matout

#     def K2R(self, matk : np.ndarray) -> np.ndarray:

        
#         # rkvec = self.crystal.kpoint.reshape((self.crystal.rkgrid[0], self.crystal.rkgrid[1], self.crystal.rkgrid[2], 3), order='F')

        
#         # norb, _, ns, nkx, nky, nkz, nf = matk.shape
#         norb, _, ns, nk, nf = matk.shape
#         rkvec = self.crystal.kpoint
#         rank = self.nodedict['commkrank']
#         (nkx, nky, nkz) = self.mpimanager.localshapef[self.nodedict['commkrank']]
#         nkglobal = self.crystal.rkgrid[0] * self.crystal.rkgrid[1] * self.crystal.rkgrid[2]
#         if (nk != nkx * nky * nkz):
#             raise ValueError(f"Error: nk ({nk}) does not match local shape ({nkx}, {nky}, {nkz})")
#         (nx, ny, nz) = self.mpimanager.localshapeb[self.nodedict['commkrank']]
#         tempmat = np.zeros((norb, norb, ns, nk, nf), dtype=np.complex128, order='F')
#         tempmat2 = np.zeros((nx, ny, nz), order='F', dtype=np.complex128)
        
#         nr = len(self.mpimanager.rlocal[rank])
#         matr = np.zeros((norb, norb, ns, nr, nf), dtype=np.complex128, order='F')

#         for iff in range(nf):            
#             for js in range(ns):
#                 for jorb in range(norb):
#                     for iorb in range(norb):
#                         for ik in range(nk):
#                             a, _ = self.crystal.FAtomOrb(iorb)
#                             b, _ = self.crystal.FAtomOrb(jorb)
#                             delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
#                             kidx = self.mpimanager.KLocal2Global([rank, ik])
#                             phase = np.exp(2.0j * np.pi * np.dot(rkvec[kidx], delta))
#                             tempmat[iorb, jorb, js, ik, iff] = matk[iorb, jorb, js, ik, iff] * phase
#                         # --------------------------------------------------------------------------- #
#                         tempval = self.mpimanager.K2K3D(self.commk, tempmat[iorb, jorb, js, :, iff])
#                         tempval2 = self.mpimanager.Backward(tempval)
#                         tempmat2 = tempval2*1/(nkglobal)

#                         matr[iorb, jorb, js,:,iff] = self.mpimanager.R3D2R(self.commk, tempmat2)

#         return matr
    

#     def R2K(self, matr : np.ndarray) -> np.ndarray:

#         rkvec = self.crystal.kpoint

#         norb, _, ns, nr, nf = matr.shape
#         rank = self.nodedict['commkrank']
#         (nx, ny, nz) = self.mpimanager.localshapeb[rank]
#         (nkx, nky, nkz) = self.mpimanager.localshapef[rank]
#         if (nr != nx * ny * nz):
#             print(f"Error: nk ({nr}) does not match local shape ({nx}, {ny}, {nz})")
#             sys.exit()
#         tempmat = np.zeros_like(matr, dtype=np.complex128, order='F')
#         tempmat2 = np.zeros((nkx, nky, nkz), order='F', dtype=np.complex128)
#         matk = np.zeros((norb, norb, ns, nkx*nky*nkz, nf), dtype=np.complex128, order='F')

        
#         for iff in range(nf):
#             for js in range(ns):
#                 for jorb in range(norb):
#                     for iorb in range(norb):
                        
#                         tempval = self.mpimanager.R2R3D(self.commk, matr[iorb, jorb, js, :, iff])
#                         tempval2 = self.mpimanager.Forward(tempval)
#                         tempmat2 = tempval2
#                         tempmat[iorb, jorb, js, :, iff] = self.mpimanager.K3D2K(self.commk, tempmat2)
                        
        
#                         for ik in range(nkx*nky*nkz):
#                             a, _ = self.crystal.FAtomOrb(iorb)
#                             b, _ = self.crystal.FAtomOrb(jorb)
#                             delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
#                             kidx = self.mpimanager.KLocal2Global([rank, ik])
#                             phase = np.exp(-2.0j * np.pi * np.dot(rkvec[kidx], delta))
#                             matk[iorb, jorb, js, ik, iff] = tempmat[iorb, jorb, js, ik, iff] * phase

#         return matk
    
#     def Moment(self, ff : np.ndarray, isgreen : bool, highzero : bool) -> np.ndarray:



#         norb, _, ns, nkloc, nfloc = ff.shape 
#         omega = self.ftgrid.omega*1j
#         moment = np.zeros((norb, norb, ns, nkloc, 3), dtype=np.complex128, order='F')
#         high = np.zeros((norb, norb, ns, nkloc), dtype=np.complex128, order='F')
        

#         fflast = self.mpimanager.FMPIBCast(self.commw, ff, len(omega)-1)
#         fflast2 = self.mpimanager.FMPIBCast(self.commw, ff, len(omega)-2)

#         if (isgreen):
#             for ik in range(nkloc):
#                 for js in range(ns):
#                     for jorb in range(norb):
#                         for iorb in range(norb):
                            
#                             if (iorb == jorb):
#                                 moment[iorb, jorb, js, ik, 0] = 1.0
#                             else:
#                                 moment[iorb, jorb, js, ik, 0] = 0.0

#                             moment[iorb, jorb, js, ik, 1] += (fflast[iorb, jorb, js, ik] + np.conjugate(fflast[jorb, iorb, js, ik])) \
#                                 / 2.0 * (omega[-1])**2

#                             moment[iorb, jorb, js, ik, 2] += (fflast[iorb, jorb, js, ik] - np.conjugate(fflast[jorb, iorb, js, ik]) 
#                                                               - moment[iorb, jorb, js, ik, 0]* 2.0/(omega[-1])) \
#                                                                   /2.0 * (omega[-1])**3

#         else:
#             if (highzero):
#                 for ik in range(nkloc):
#                     for js in range(ns):
#                         for jorb in range(norb):
#                             for iorb in range(norb):

#                                 moment[iorb, jorb, js, ik, 0] += (fflast[iorb, jorb, js, ik] \
#                                                                   - np.conjugate(fflast[jorb, iorb, js, ik]))/2.0 * (omega[-1])
#                                 moment[iorb, jorb, js, ik, 1] += (fflast[iorb, jorb, js, ik] \
#                                                                   + np.conjugate(ff[jorb, iorb, js, ik]))/2.0 * (omega[-1])**2
#             else:
#                 amat = np.zeros((4, 4), dtype=np.complex128, order='F')
#                 bmat = np.zeros((4, 1), dtype=np.complex128, order='F')
#                 # ipiv = np.zeros((4), dtype=int, order='F')
#                 for ik in range(nkloc):
#                     for js in range(ns):
#                         for jorb in range(norb):
#                             for iorb in range(norb):
#                                 w1 = omega[-1]
#                                 w2 = omega[-2]

#                                 amat[0, :] = [1.0, 1.0/(w1), 1.0/(w1)**2, 1.0/(w1)**3]
#                                 amat[1, :] = [1.0, -1.0/(w1), 1.0/(w1)**2, -1.0/(w1)**3]
#                                 amat[2, :] = [1.0, 1.0 / (w2), 1.0 / (w2) ** 2, 1.0 / (w2) ** 3]
#                                 amat[3, :] = [1.0, -1.0 / (w2), 1.0 / (w2) ** 2, -1.0 / (w2) ** 3]

#                                 bmat[0, 0] = fflast[iorb, jorb, js, ik]
#                                 bmat[1, 0] = np.conjugate(fflast[jorb, iorb, js, ik])
#                                 bmat[2, 0] = fflast2[iorb ,jorb, js, ik]
#                                 bmat[3, 0] = np.conjugate(fflast2[jorb, iorb, js, ik])

#                                 x = scipy.linalg.solve(amat, bmat)

#                                 high[iorb, jorb, js, ik] = x[0, 0]
#                                 moment[iorb, jorb, js, ik ,0] = x[1, 0]
#                                 moment[iorb, jorb, js, ik, 1] = x[2, 0]
#                                 moment[iorb, jorb, js, ik, 2] = x[3, 0]

#         for ik in range(nkloc):
#             for js in range(ns):
#                 high[:, :, js, ik] = (np.conjugate(high[:, :, js, ik]).T + high[:, :, js, ik])/2.0
#                 for i in range(3):
#                     moment[:, :, js, ik, i] = (np.conjugate(moment[:, :, js, ik, i]).T + moment[:, :, js, ik, i]) / 2.0

#         return moment, high
    
#     def F2T(self, ff : np.ndarray, isgreen : bool, highzero : bool):

#         rank = self.nodedict['commtaurank']
#         ntauloc = self.submatrixtau[rank][1] - self.submatrixtau[rank][0]
#         tau = np.zeros((ntauloc), dtype=np.float64, order='F')
#         for itau in range(ntauloc):
#             tauidx = self.mpimanager.TLocal2Global([rank, itau])
#             tau[itau] = self.ftgrid.tau[tauidx]
#         norb = ff.shape[0]
#         ns = ff.shape[2]
#         nk = ff.shape[3]
#         nomega = len(self.ftgrid.omega)
#         ftau = np.zeros((norb, norb, ns, nk, ntauloc), dtype=np.complex128, order='F')
        
#         moment, high = self.Moment(ff, isgreen, highzero)

#         ffglob = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order='F')

#         for ik in range(nk):
#             for js in range(ns):
#                 for jorb in range(norb):
#                     for iorb in range(norb):
#                         ffglob[iorb, jorb, js, ik] = self.mpimanager.FMPIAllreduce(self.commw, ff[iorb, jorb, js, ik], nomega)
        
#         ftau = QAFort.fourier.flatdyn_f2t(self.ftgrid.omega, ffglob, moment, tau)

#         return ftau




#     # def F2T(self, ff : np.ndarray, isgreen : bool, highzero : bool) -> np.ndarray:

#     #     if (self.mpimanager.rank == 0):
#     #         print("Compute Fourier transform F2T Start")
#     #     moment, high = self.Moment(ff, isgreen, highzero)

#     #     rank = self.nodedict['commtaurank']
#     #     ntauloc = self.submatrixtau[rank][1]-self.submatrixtau[rank][0]
#     #     beta = self.ftgrid.beta
#     #     omega = self.ftgrid.omega*1j
#     #     nomega = len(omega)
#     #     norb, _, ns, nrk, _ = ff.shape

#     #     tau = np.zeros((ntauloc), dtype=np.float64, order='F')
#     #     ftau = np.zeros((norb, norb, ns, nrk, ntauloc), dtype=np.complex128, order='F')
#     #     ffglob = np.zeros((norb, norb, ns, nrk, nomega), dtype=np.complex128, order='F')

#     #     for itau in range(ntauloc):
#     #         tidx = self.mpimanager.TLocal2Global([rank, itau])
#     #         tau[itau] = self.ftgrid.tau[tidx]

#     #     for irk in range(nrk):
#     #         for js in range(ns):
#     #             for jorb in range(norb):
#     #                 for iorb in range(norb):
#     #                     ffglob[iorb, jorb, js, irk] = self.mpimanager.FMPIAllreduce(self.commw, ff[iorb, jorb, js, irk], nomega)
        
        
#     #             # for jorb in range(norb):
#     #             #     for iorb in range(norb):
#     #             #         for itau in range(ntauloc):
#     #             #             for iomega in range(nomega):
#     #             #                 ftau[iorb, jorb, js, irk, itau] += 1/beta * np.exp(-tau[itau]*omega[iomega]) \
#     #             #                 *(
#     #             #                     ffglob[iorb, jorb, js, irk, iomega] 
#     #             #                     - moment[iorb, jorb, js, irk, 0]/omega[iomega] 
#     #             #                     - moment[iorb, jorb, js, irk, 1]/(omega[iomega])**2 
#     #             #                     - moment[iorb, jorb, js, irk, 2]/(omega[iomega])**3
#     #             #                 ) \
#     #             #                   +1/beta * np.exp(tau[itau]*omega[iomega]) \
#     #             #                   *(
#     #             #                       np.conjugate(ffglob[jorb, iorb, js, irk, iomega])
#     #             #                       + moment[iorb, jorb, js, irk, 0]/omega[iomega]
#     #             #                       - moment[iorb, jorb, js, irk, 1]/(omega[iomega])**2
#     #             #                       + moment[iorb, jorb, js, irk, 2]/(omega[iomega])**3
#     #             #                   )
#     #             #             ftau[iorb, jorb, js, irk, itau] += \
#     #             #             - moment[iorb, jorb, js, irk, 0]/2.0 \
#     #             #             + moment[iorb, jorb, js, irk, 1]*beta/2.0 * \
#     #             #                 (tau[itau]/beta - 1/2) \
#     #             #             - moment[iorb, jorb, js, irk, 2]*beta**2/4.0 * \
#     #             #                 ((tau[itau]/beta)**2 - (tau[itau]/beta))
        
#     #     ftau = QAFort.fourier.dyn_f2t(omega, ffglob, moment, tau)
#     #     if (self.mpimanager.rank == 0):
#     #         print("Compute Fourier transform F2T Finish")
                            
#     #     return ftau

    
#     # def T2F(self, ftau: np.ndarray) -> np.ndarray:
#     #     """
#     #     Performs the T2F transform using FINUFFT's built-in MPI support.
#     #     """
#     #     # 1. Get communicators and local shape info
#     #     comm_f = self.nodedict['commf']
#     #     comm_tau = self.nodedict['commtau']
#     #     rank_f = self.nodedict['commfrank']

#     #     nfloc = self.submatrixw[rank_f][1] - self.submatrixw[rank_f][0]
#     #     norb, _, ns, nk, ntauloc = ftau.shape

#     #     # 2. Prepare the source (tau) and target (omega) points
#     #     # Each process only needs its local slice of tau points
#     #     rank_tau = self.nodedict['commtaurank']
#     #     tau_start = self.submatrixtau[rank_tau][0]
#     #     tau_end = self.submatrixtau[rank_tau][1]
#     #     local_tau = self.ftgrid.tau[tau_start:tau_end]

#     #     # For fermionic transform, rescale tau to [0, 2*pi]
#     #     # This corresponds to the integer modes 'n' in omega_n = (2n+1)pi/beta
#     #     # The kernel is exp(i*n*x_j) * exp(i*pi*tau/beta)
#     #     local_x_j = 2 * np.pi * local_tau / self.ftgrid.beta
#     #     phase_factor = np.exp(1j * np.pi * local_tau / self.ftgrid.beta)
        
#     #     # Total number of output omega modes
#     #     nomega_total = len(self.ftgrid.omega)

#     #     # 3. Allocate output array
#     #     ff = np.zeros((norb, norb, ns, nk, nfloc), dtype=np.complex128, order='F')

#     #     # 4. Loop over the data and perform the parallel NUFFT
#     #     # The loops are still needed, but the core operation is now a parallel NUFFT call.
#     #     for iorb in range(norb):
#     #         for jorb in range(norb):
#     #             for js in range(ns):
#     #                 for ik in range(nk):
#     #                     # Get the local slice of ftau for this matrix element
#     #                     # phase_factor = np.sqrt(local_tau )
#     #                     local_strengths = ftau[iorb, jorb, js, ik, :]

#     #                     # Apply the phase factor for fermionic frequencies
#     #                     modified_strengths = local_strengths * phase_factor

#     #                     # This is the key call:
#     #                     # - Each process provides its LOCAL tau points and strengths.
#     #                     # - FINUFFT handles the MPI_Alltoallv communication internally.
#     #                     # - The output is distributed across the same communicator.
#     #                     local_ff_slice = finufft.nufft1d1(
#     #                         local_x_j,
#     #                         modified_strengths,
#     #                         n_modes=nomega_total,
#     #                         isign=1,  # For exp(+i*omega*tau)
#     #                         eps=1e-12,
#     #                         nthreads=1  # Use the tau communicator
#     #                     )

#     #                     # Since comm_f and comm_tau have the same processes, the output
#     #                     # `local_ff_slice` is already the correct local frequency slice.
#     #                     for iomega in range(nfloc):
#     #                         fidx = self.mpimanager.FLocal2Global([comm_f.Get_rank(), iomega])
#     #                         ff[iorb, jorb, js, ik, iomega] = local_ff_slice[fidx]

#     #     return ff

#     def T2F(self, ftau : np.ndarray) -> np.ndarray:

#         rank = self.nodedict['commfrank']
#         nfloc = self.submatrixw[rank][1]-self.submatrixw[rank][0]
#         norb = ftau.shape[0]
#         ns = ftau.shape[2]
#         nk = ftau.shape[3]
#         # ntauloc = ftau.shape[4]
#         # tau = self.ftgrid.tau
#         ntau = len(self.ftgrid.tau)
        
#         # print(f"commfrank : {rank}, total comm rank : {self.mpimanager.rank}")

#         ff = np.zeros((norb, norb, ns, nk, nfloc), dtype=np.complex128, order='F')
#         omega = np.zeros((nfloc), dtype=np.float64, order='F')

#         ftauglob = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')

#         for ik in range(nk):
#             for js in range(ns):
#                 for jorb in range(norb):
#                     for iorb in range(norb):
#                         ftauglob[iorb, jorb, js, ik] = self.mpimanager.TMPIAllreduce(self.commtau, ftau[iorb, jorb, js, ik], ntau)

#         for ifloc in range(nfloc):
#             fidx = self.mpimanager.FLocal2Global([rank, ifloc])
#             omega[ifloc] = self.ftgrid.omega[fidx]

#         # ff = QAFort.fourier.flatdyn_t2f(self.ftgrid.tau, self.ftgrid.beta, ftauglob, omega)
#         ffglob = QAFort.fourier.flatdyn_t2f(self.ftgrid.tau, self.ftgrid.beta, ftauglob, self.ftgrid.omega)

#         for ifreq in range(nfloc):
#             fidx = self.mpimanager.FLocal2Global([rank, ifreq])
#             ff[...,ifreq] = ffglob[...,fidx]

#         return ff


        

#     # def T2F_v0(self, ftau : np.ndarray) -> np.ndarray:

#     #     rank = self.nodedict['commfrank']
#     #     nfloc = self.submatrixw[rank][1]-self.submatrixw[rank][0]
#     #     beta = self.ftgrid.beta
#     #     tau = self.ftgrid.tau
#     #     omega = self.ftgrid.omega
#     #     ntau = len(tau)

#     #     norb, _, ns, nk, ntauloc = ftau.shape
#     #     ntaufinu = 2*ntau
#     #     nffinu = 4*len(omega) -1
#     #     tauradfinu = np.zeros((ntaufinu), dtype=np.float64, order='F')
#     #     ftaufinu = np.zeros((ntaufinu), dtype=np.complex128, order='F')
#     #     ff = np.zeros((norb, norb, ns, nk, nfloc), dtype=np.complex128, order='F')

#     #     for itau in range(ntau):
#     #         # itehta = QAFort.common.ttind(itau, ntau)
#     #         tauradfinu[itau] = tau[itau]/beta*np.pi
#     #         tauradfinu[-itau -1] = -tauradfinu[itau]

#     #     for ik in range(nk):
#     #         for js in range(ns):
#     #             for jorb in range(norb):
#     #                 for iorb in range(norb):
#     #                     ftauglob = self.mpimanager.TMPIAllreduce(self.commtau, ftau[iorb, jorb, js, ik], ntau = ntau)
#     #                     for itau in range(ntau):
#     #                         ftaufinu[itau] = ftauglob[itau] * np.sqrt(tau[itau]*(beta-tau[itau]))*np.pi/ntau
#     #                         ftaufinu[itau-ntau] = -ftaufinu[itau]
                        
#     #                     fffinu = finufft.nufft1d1(tauradfinu, ftaufinu, nffinu, isign=1, eps=1e-12, nthreads=1)
#     #                     for ifreq in range(0, 2*len(omega)):
#     #                         if ((ifreq % 2) == 1):
#     #                             irank, iomega = self.mpimanager.FGlobal2Local(int((ifreq-1)/2))
#     #                             if (irank == rank):
#     #                                 ff[iorb, jorb, js, iomega] = fffinu[ifreq]/2.0

#     #     return ff
                        
#     # def T2F(self, ftau : np.ndarray) -> np.ndarray:

#     #     if (self.mpimanager.rank == 0):
#     #         print("Compute Fourier transform T2F Start")
        
#     #     rank = self.nodedict['commfrank']
#     #     nfloc = self.submatrixw[rank][1]-self.submatrixw[rank][0]
#     #     beta = self.ftgrid.beta
#     #     # nomega = len(self.ftgrid.omega)
#     #     ntau = len(self.ftgrid.tau)
#     #     norb = ftau.shape[0]
#     #     ns = ftau.shape[2]
#     #     nrk = ftau.shape[3]

#     #     # tempmat = np.zeros((ntau), dtype=np.complex128, order='F')
#     #     ftauglob = np.zeros((ntau), dtype=np.complex128, order='F')
#     #     omega = np.zeros((nfloc), dtype=np.float64, order='F')
#     #     ff = np.zeros((norb, norb, ns, nrk, nfloc), dtype=np.complex128, order='F')

#     #     for ifreq in range(nfloc):
#     #         fidx = self.mpimanager.FLocal2Global([rank, ifreq])
#     #         omega[ifreq] = self.ftgrid.omega[fidx]

#     #     for irk in range(nrk):
#     #         for js in range(ns):
#     #             for jorb in range(norb):
#     #                 for iorb in range(norb):
#     #                     ftauglob = self.mpimanager.TMPIAllreduce(self.commtau, ftau[iorb, jorb, js, irk], ntau)
#     #                     ff[iorb, jorb, js, irk] = QAFort.fourier.dyn_t2f(self.ftgrid.tau, ftauglob, omega)
        
#     #     if (self.mpimanager.rank == 0):
#     #         print("Compute Fourier transform T2F Finish")
        
#     #     return ff


# class BLatDynMPI(object):

#     def __init__(self, crystal : Crystal, ftgrid : FTGrid, nk : int, nw : int, ntau : int, nprock : int, nprocw : int, mpimanager : MPIManager):

#         self.crystal = crystal
#         self.ftgrid = ftgrid
#         self.nk = nk
#         self.nw = nw
#         self.nprock = nprock
#         self.nprocw = nprocw
#         self.mpimanager = mpimanager
#         self.nodedict = mpimanager.Quary(nk, nw, ntau, nprock, nprocw, self.crystal)

#         self.commk = self.nodedict['commk']
#         self.commw = self.nodedict['commf']
#         self.submatrixkf = self.nodedict['submatrixkf']
#         self.submatrixkb = self.nodedict['submatrixkb']
#         self.submatrixw = self.nodedict['submatrixw']

#         self.commtau = self.nodedict['commtau']

#         self.submatrixtau = self.nodedict['submatrixtau']


#     def CheckGroup(self, filepath : str, group : str):

#         with h5py.File(filepath, 'r') as file:
#             return group in file
        
    
#     def Save(self, hdf5file : str = None, group : str = None, subgroup : str = None, data : np.ndarray = None, dataname : str = None):

#         with h5py.File(hdf5file, 'a') as file:
#             if (self.CheckGroup(hdf5file, group)):
#                 g =  file[group]
#                 if subgroup in g:
#                     subg = g[subgroup]
#                 else:
#                     subg = g.create_group(subgroup)
#             else:
#                 g = file.create_group(group)
#                 subg = g.create_group(subgroup)

#             subg.create_dataset(dataname, data=data, dtype=np.complex128, driver='mpio', comm = self.mpimanager.comm)

#             return None

#     def Load(self, hdf5file : str = None, group : str = None, subgroup : str = None, data : np.ndarray = None, dataname : str = None):

#         with h5py.File(hdf5file, 'r') as file:
#             if (self.CheckGroup(hdf5file, group)):
#                 g =  file[group]
#                 if subgroup in g:
#                     subg = g[subgroup]
#                     if dataname in subg:
#                         data = subg[dataname][:]
#                         return data
#                     else:
#                         raise KeyError(f"{dataname} not found in {subgroup}")
#                 else:
#                     raise KeyError(f"{subgroup} not found in {group}")
#             else:
#                 raise KeyError(f"{group} not found in {hdf5file}")



#     def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

#         nk = mat1.shape[4]
#         nft = mat1.shape[5]
#         matout = np.zeros_like(mat1, dtype=np.complex128, order='F')
        
#         for ift in range(nft):
#             for ik in range(nk):
#                 matout[:, :, :, :, ik, ift] = QAFort.dyson.blocstc(mat1[:, :, :, :, ik, ift], mat2[:, :, :, :, ik, ift])      

#         return matout

    
#     def Inverse(self, matin : np.ndarray) -> np.ndarray:

        
#         norb, _, ns, _, nk, nft = matin.shape

#         matout = np.zeros((norb, norb, ns, ns, nk, nft), dtype=np.complex128, order='F')
#         tempmat = np.zeros((norb*ns, norb*ns, nk, nft), dtype=np.complex128, order='F')
#         tempmat2 = np.zeros((norb*ns, norb*ns, nk, nft), dtype=np.complex128, order='F')


#         for ift in range(nft):
#             for ik in range(nk):
#                 tempmat[:, :, ik, ift] = self.crystal.OrbSpin2Composite(matin[:, :, :, :, ik, ift])

#                 tempmat2[:, :, ik, ift] = np.linalg.inv(tempmat[:, :, ik, ift])

#                 matout[:, :, :, :, ik, ift] = self.crystal.Composite2OrbSpin(tempmat2[:, :, ik, ift])

#         return matout

#     def K2R(self, matk : np.ndarray) -> np.ndarray:


#         norb, _, ns, _, nk, nf = matk.shape
#         rkvec = self.crystal.kpoint
#         rank = self.nodedict['commkrank']
#         (nkx, nky, nkz) = self.mpimanager.localshapef[rank]
#         (nx, ny, nz) = self.mpimanager.localshapeb[rank]
#         nkglobal = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
#         if (nk != nkx*nky*nkz):
#             print(f"Error: nk ({nk}) does not match local shape ({nkx}, {nky}, {nkz})")
#             sys.exit()
#         tempmat = np.zeros((norb, norb, ns, ns, nk, nf), dtype=np.complex128, order='F')
#         tempmat2 = np.zeros((nx, ny, nz), order='F', dtype=np.complex128)
#         nr = len(self.mpimanager.rlocal[rank])
#         matr = np.zeros((norb, norb, ns, ns, nr, nf), dtype=np.complex128, order='F')


#         for iff in range(nf):
#             for ks in range(ns):
#                 for js in range(ns):
#                     for jorb in range(norb):
#                         for iorb in range(norb):
#                             for ik in range(nk):
#                                 a, _ = self.crystal.BAtomOrb(iorb)
#                                 b, _ = self.crystal.BAtomOrb(jorb)
#                                 delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
#                                 kidx = self.mpimanager.KLocal2Global([rank, ik])
#                                 phase = np.exp(2.0j * np.pi * np.dot(rkvec[kidx], delta))
#                                 tempmat[iorb, jorb, js, ks, ik, iff] = matk[iorb, jorb, js, ks, ik, iff] * phase
#                             # ----------------------------------------------------------------------------------- #
#                             tempval = self.mpimanager.K2K3D(self.commk, tempmat[iorb, jorb, js, ks, :, iff])
#                             tempval2 = self.mpimanager.Backward(tempval)
#                             tempmat2 = tempval2 * 1/(nkglobal)

#                             matr[iorb, jorb, js, ks, :, iff] = self.mpimanager.R3D2R(self.commk, tempmat2)

#         return matr

            

        
        

#     def R2K(self, matr : np.ndarray) -> np.ndarray:

        

#         norb, _, ns, _, nr, nf = matr.shape
#         rkvec = self.crystal.kpoint
#         rank = self.nodedict['commkrank']
#         (nx, ny, nz) = self.mpimanager.localshapeb[rank]
#         (nkx, nky, nkz) = self.mpimanager.localshapef[rank]

#         if (nr != nx * ny * nz):
#             print(f"Error: nk ({nr}) does not match local shape ({nx}, {ny}, {nz})")
#             sys.exit()

#         tempmat = np.zeros_like(matr, dtype=np.complex128, order='F')
#         tempmat2 = np.zeros((nx, ny, nz), dtype=np.complex128, order='F')
#         matk = np.zeros((norb, norb, ns, ns, nkx*nky*nkz, nf), dtype=np.complex128, order='F')

#         for iff in range(nf):
#             for ks in range(ns):
#                 for js in range(ns):
#                     for jorb in range(norb):
#                         for iorb in range(norb):
#                             tempval = self.mpimanager.R2R3D(self.commk, matr[iorb, jorb, js, ks, :, iff])
#                             tempval2 = self.mpimanager.Forward(tempval)
#                             tempmat2 = tempval2
#                             tempmat[iorb, jorb, js, ks, :, iff] = self.mpimanager.K3D2K(self.commk, tempmat2)

#                             for ik in range(nkx*nky*nkz):
#                                 a, _ = self.crystal.BAtomOrb(iorb)
#                                 b, _ = self.crystal.BAtomOrb(jorb)

#                                 delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
#                                 kidx = self.mpimanager.KLocal2Global([rank, ik])
#                                 phase = np.exp(-2.0j * np.pi * np.dot(rkvec[kidx], delta))
#                                 matk[iorb, jorb, js, ks, ik, iff] = tempmat[iorb, jorb, js, ks, ik, iff] * phase

#         return matk


