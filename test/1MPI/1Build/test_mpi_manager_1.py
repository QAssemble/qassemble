import os
import sys

qapath = os.environ.get("QAssemble")
sys.path.append(qapath + "/src")
import numpy as np
from mpi4py import MPI
from qacore.MPIManager import MPIManager

# class Another:
#    def __init__(self, manager: MPIManager, nprock: int, nprocw: int, nk: int, nw: int):
#        self.manager = manager
#        self.nprock, self.nprocw = nprock, nprocw
#        self.nk, self.nw = nk, nw
#        self.comm_k = self.comm_w = None
#        self.k0 = self.k1 = None
#        self.w0 = self.w1 = None
#        self._setup()
#
#    def _setup(self):
#        commk, commw, (k0, k1), (w0, w1) = self.manager.Split(
#            self.nprock, self.nprocw, self.nk, self.nw
#        )
#        self.comm_k, self.comm_w = commk, commw
#        self.k0, self.k1 = k0, k1
#        self.w0, self.w1 = w0, w1
#
#    def get_submatrix(self, A: np.ndarray) -> np.ndarray:
#        return A[self.k0 : self.k1, self.w0 : self.w1]
#
#    def save_submatrix_to_hdf5(self,
#                               filename: str,
#                               dataset: str,
#                               data: np.ndarray):
#        """
#        Each rank writes its local block `data` into the shared
#        file `filename`, dataset `dataset`, which spans the full (nk,nw).
#        """
#        # 1) Open file in parallel
#        with h5py.File(filename,
#                       'w',
#                       driver='mpio',
#                       comm=self.manager.comm) as f:
#            # 2) create the full-size dataset (only on rank 0,
#            #    but with mpio all ranks see it)
#            if self.manager.rank == 0:
#                f.create_dataset(dataset,
#                                 shape=(self.nk, self.nw),
#                                 dtype=data.dtype)
#            # barrier to ensure dataset exists before writing
#            self.manager.comm.Barrier()
#
#            # 3) all ranks open the dataset and write their slab
#            dset = f[dataset]
#            dset[self.k0:self.k1, self.w0:self.w1] = data
#            # flush to disk
#            f.flush()

#    def info(self):
#        return (
#            f"Rank {self.manager.rank:2d} "
#            f"k-range=[{self.k0},{self.k1}) "
#            f"w-range=[{self.w0},{self.w1})"
#        )


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    mgr = MPIManager(comm)
    nprock, nprocw = 2, 2
    nk, nw = 8, 12
    commk, commw = mgr.Split(nprock=nprock, nprocw=nprocw, nk=nk, nw=nw)

    # 각 프로세스의 행/열 그룹 정보 출력
    print(
        f"[Global rank {comm.Get_rank()}] "
        f"commk rank/size: {commk.Get_rank()}/{commk.Get_size()}, "
        f"commw rank/size: {commw.Get_rank()}/{commw.Get_size()}"
    )
