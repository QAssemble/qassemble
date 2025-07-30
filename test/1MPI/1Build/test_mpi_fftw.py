import numpy as np
import mpi4py
import sys, os
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

def main():
    qapath = os.environ.get('QAssemble')
    sys.path.append(qapath+'/src/qacore/modules')
    import QAFort
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    norb = 3
    ns = 2
    nk = 125
    nomega = 10
    fr = np.zeros((norb,norb, ns, nk,nomega), dtype=np.complex128, order='F')
    fk = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
    fr2 = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
    bk = np.zeros([norb,norb,ns,ns,nk,nomega],dtype=complex,order='F')
    br = np.zeros([norb,norb,ns,ns,nk,nomega],dtype=complex,order='F')
    br2 = np.zeros([norb,norb,ns,ns,nk,nomega],dtype=complex,order='F')
    irk = 0
    ind = np.zeros([1,1,1],dtype=int,order='F')
    divisionarray = np.array([5,5,5],dtype=int,order='F')

    for iomega in range(nomega):
        for kk in range(5):
            for jk in range(5):
                for ik in range(5):
                    ind = [ik + 1, jk + 1, kk + 1]
                    ind = np.array(ind, order='F')
                    QAFort.common.indexing(nk, divisionarray,1, irk, ind)
                    for js in range(ns):
                        for jorb in range(norb):
                            for iorb in range(norb):
                                fk[iorb, jorb, js, irk, iomega]  = ((iorb+1)-(jorb+1))/2.0 +(js+1)*0.1 + (ik+jk+kk)/2.0 + (iomega+1)*0.001

    fr = QAFort.fourier.flatdyn_k2r(divisionarray, fk)

    plan = PFFT(comm=comm, shape=(5, 5, 5), dtype=np.complex128, axes=(0,1,2))

    # fk_loc = newDistArray(plan, True)
    # fr2_loc = newDistArray(plan, True)

    # local_sl = plan.local_slice(True)

    # fk_loc[:] = fk[local_sl]
    # plan.forward(fk_loc, fr2_loc)
    for iomega in range(nomega):
        for js in range(ns):
            for jorb in range(norb):
                for iorb in range(norb):
                    tempmat = fk[iorb, jorb, js, :, iomega].reshape((5, 5, 5), order='F')
                    tempmat2 = np.zeros_like(tempmat, order='F')
                    plan.forward(tempmat, tempmat2)
                    fr2[iorb, jorb, js, :, iomega] = tempmat2.reshape((nk), order='F')

    if rank == 0:
        print("Performed FFT. 5D global shape:", fk.shape)
    print(f"Rank {rank:02d}/{size:02d} local mpifft input‐shape = {fk.shape}, "
          f"output‐shape = {fr2.shape}")
    
    # loc_norm = np.linalg.norm(fr2_loc)
    # total_norm = comm.allreduce(loc_norm)
    # if rank == 0:
    #     print("Sum of ||fr2_local|| over all ranks =", total_norm)

    # ikr = 0

    # for iomega in range(nomega):
    #     for kk in range(5):
    #         for jk in range(5):
    #             for ik in range(5):
    #                 ind = [ik+1,jk+1,kk+1]
    #                 ind = np.array(ind,order='F')
    #                 QAFort.common.indexing(nk, divisionarray,1, irk, ind)
    #                 for ks in range(ns):
    #                     for js in range(ns):
    #                         for jorb in range(norb):
    #                             for iorb in range(norb):
    #                                 bk[iorb, jorb, js, ks, ikr, iomega] = ((iorb+1)-(jorb+1))/2.0 + ((js+1)-(ks+1))*0.1+(ik+jk+kk)/2.0 + iomega*0.001

    # br = QAFort.fourier.blatdyn_k2r(divisionarray, bk)

    return None

if __name__ == "__main__":
    mpi4py.rc.initialize = False  # Disable automatic initialization
    mpi4py.rc.finalize = False  # Disable automatic finalization
    main()
    MPI.Finalize()  # Manually finalize MPI