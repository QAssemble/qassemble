from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import os, sys
import numba
import scipy.linalg
import numpy as np
import scipy
import h5py
import finufft
qapath = os.environ.get('QAssemble','')
sys.path.append(qapath+'/src/QAssemble/Src_mpi')
from Crystal import Crystal



class MPIManager(object):


    def __init__(self, comm : MPI.COMM_WORLD):


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
    
    def FMPIAllreduce(self, commw : MPI.COMM_WORLD, matin : np.ndarray, nf : int) -> np.ndarray:

        nfloc = matin.shape[0]


        tempmat = np.zeros((nf), dtype=np.complex128, order='F')
        matout = np.zeros((nf), dtype=np.complex128, order='F')

        for iff in range(nfloc):            
            fidx = self.FLocal2Global([commw.Get_rank(), iff])
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
    
    def KMPIAllreduce(self, commk : MPI.COMM_WORLD, matin : np.ndarray) -> np.ndarray:

        nk = len(self.crystal.kpoint)
        nkloc = matin.shape[0]


        tempmat = np.zeros((nk), dtype=np.complex128, order='F')
        matout = np.zeros((nk), dtype=np.complex128, order='F')

        for ik in range(nkloc):            
            kidx = self.KLocal2Global([commk.Get_rank(), ik])
            tempmat[kidx] = matin[ik]

        commk.Allreduce(tempmat, matout, op=MPI.SUM)

        return matout
    
    def RMPIAllreduce(self, commk : MPI.COMM_WORLD, matin : np.ndarray) -> np.ndarray:

        nr = len(self.crystal.kpoint)
        nrloc = matin.shape[0]

        tempmat = np.zeros((nr), dtype=np.complex128, order='F')
        matout = np.zeros((nr), dtype=np.complex128, order='F')

        for ir in range(nrloc):
            ridx = self.RLocal2Global([commk.Get_rank(), ir])
            tempmat[ridx] = matin[ir]

        commk.Allreduce(tempmat, matout, op=MPI.SUM)
        

        return matout
    
    def FMPIBCast(self, comm : MPI.COMM_WORLD, matin : np.ndarray, idx : int) -> np.ndarray:

        (rank, localidx) = self.FGlobal2Local(idx)
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
