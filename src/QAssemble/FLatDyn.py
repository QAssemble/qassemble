import numpy as np
import sys
import scipy.optimize
import scipy.linalg.lapack
import copy
import h5py
import time, datetime
import logging
from .Crystal import Crystal
from .FLatStc import FLatStc
from .utility.DLR import DLR
from .utility.Common import Common
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
from .utility.Mixing import Mixing
from .utility.StagedHDF5 import save_distributed_dataset


logger = logging.getLogger("QAssemble")

class FLatDyn(object):
    def __init__(self,crystal : Crystal, dlr : DLR, nodedict : dict = None, mixing_method: str = "pulay", npulay: int = 5) -> object:
        self.crystal = crystal
        self.dlr = dlr
        self.nodedict = nodedict
        self._mixer = Mixing(method=mixing_method, npulay=npulay)
        self.mappingidx = None
        self._fermion_phase_cache_k2r = self._get_fermion_phaseK2R()
        self._fermion_phase_cache_r2k = self._get_fermion_phaseR2K()

    def _kr_global_indices(self, loc2glob_key: str, nloc: int) -> list:

        if self.nodedict is None:
            return list(range(nloc))

        from .utility.MPIManager import MPIFunction

        mf = MPIFunction()
        rank = self.nodedict["commk"].Get_rank()
        loc2glob = self.nodedict[loc2glob_key]

        idx = []
        for iloc in range(nloc):
            _, iglob = mf.KRLocal2Global([rank, iloc], loc2glob)
            idx.append(iglob)

        return idx

    def _get_local_kfrac(self, nk: int = None) -> np.ndarray:

        nk_global = len(self.crystal.kpoint)

        if self.nodedict is None:
            if nk is None:
                nk = nk_global
            if nk != nk_global:
                raise ValueError(
                    f"Serial FLatDyn expected k-axis length {nk_global}, got {nk}."
                )
            return np.array(self.crystal.kpoint[:nk], dtype=float, copy=True)

        rank = self.nodedict["commk"].Get_rank()
        nk_local = len(self.nodedict["kloc2glob"][rank])

        if nk is None:
            nk = nk_local
        if nk != nk_local:
            raise ValueError(
                f"Parallel FLatDyn expected local k-axis length {nk_local}, got {nk}."
            )

        kglob = self._kr_global_indices("kloc2glob", nk)
        return np.array(self.crystal.kpoint[kglob], dtype=float, copy=True)

    def _get_fermion_phaseK2R(self, nk: int = None) -> np.ndarray:

        basis_orb = self.crystal.basisf[self.crystal.forb2atom]

        kfrac = self._get_local_kfrac(nk)

        kv = kfrac @ basis_orb.T

        kv_delta = kv[:, :, None] - kv[:, None, :]

        phases = np.exp(2.0j * np.pi * kv_delta)

        phases_T = np.transpose(phases, (1, 2, 0))
        return phases_T

    def _get_fermion_phaseR2K(self, nk: int = None) -> np.ndarray:

        basis_orb = self.crystal.basisf[self.crystal.forb2atom]

        kfrac = self._get_local_kfrac(nk)

        kv = kfrac @ basis_orb.T

        kv_delta = kv[:, :, None] - kv[:, None, :]

        phases = np.exp(-2.0j * np.pi * kv_delta)

        phases_T = np.transpose(phases, (1, 2, 0))
        return phases_T

    def _phase_k2r(self, nk: int) -> np.ndarray:

        if self._fermion_phase_cache_k2r.shape[2] == nk:
            return self._fermion_phase_cache_k2r

        return self._get_fermion_phaseK2R(nk)

    def _phase_r2k(self, nk: int) -> np.ndarray:

        if self._fermion_phase_cache_r2k.shape[2] == nk:
            return self._fermion_phase_cache_r2k

        return self._get_fermion_phaseR2K(nk)

    def _nk_global(self) -> int:

        return len(self.crystal.kpoint)

    def _nk_local(self, matin: np.ndarray = None) -> int:

        if matin is not None:
            return matin.shape[3]

        if self.nodedict is None:
            return self._nk_global()

        rank = self.nodedict["commk"].Get_rank()
        return len(self.nodedict["kloc2glob"][rank])

    def _allreduce_scalar(self, value: float, op=None) -> float:

        if self.nodedict is None:
            return value

        return self.nodedict["commk"].allreduce(value, op=op)

    def _allreduce_array(self, matin: np.ndarray) -> np.ndarray:

        if self.nodedict is None:
            return np.array(matin, dtype=np.complex128, copy=True)

        from mpi4py import MPI

        commk = self.nodedict["commk"]
        matloc = np.ascontiguousarray(matin, dtype=np.complex128)
        matout = np.zeros_like(matloc)
        commk.Allreduce(matloc, matout, op=MPI.SUM)

        return matout

    def _slice_local_k(self, matin: np.ndarray) -> np.ndarray:

        mat = np.array(matin, dtype=np.complex128, copy=True)

        if self.nodedict is None:
            return mat

        nk_local = self._nk_local()
        nk_global = self._nk_global()

        if mat.shape[3] == nk_local:
            return mat

        if mat.shape[3] != nk_global:
            raise ValueError(
                f"Expected k-axis length {nk_local} (local) or {nk_global} (global), got {mat.shape[3]}."
            )

        kglob = self._kr_global_indices("kloc2glob", nk_local)
        return np.asfortranarray(np.take(mat, kglob, axis=3))

    def _gather_global_k(self, matin: np.ndarray) -> np.ndarray:

        mat = np.array(matin, dtype=np.complex128, copy=True)

        if self.nodedict is None:
            return mat

        commk = self.nodedict["commk"]
        rank = commk.Get_rank()
        idx_local = self._kr_global_indices("kloc2glob", mat.shape[3])

        gathered_idx = commk.gather(idx_local, root=0)
        gathered_mat = commk.gather(mat, root=0)

        if rank != 0:
            return None

        shape_global = list(mat.shape)
        shape_global[3] = self._nk_global()
        matout = np.zeros(shape_global, dtype=np.complex128)

        for idx_rank, mat_rank in zip(gathered_idx, gathered_mat):
            matout[:, :, :, idx_rank, ...] = mat_rank

        return matout

    def _gather_ft_on_fermion_root(
        self, matin: np.ndarray, ndim: int, loc_dict_key: str
    ) -> np.ndarray:

        mat = np.asfortranarray(matin, dtype=np.complex128)

        if self.nodedict is None:
            return mat

        if "commfermion" not in self.nodedict or loc_dict_key not in self.nodedict:
            return mat

        commf = self.nodedict["commfermion"]
        rankf = commf.Get_rank()
        loc_dict = self.nodedict[loc_dict_key]
        nloc = len(loc_dict[rankf])

        if mat.shape[-1] == ndim:
            if rankf == 0:
                return mat
            return None

        if mat.shape[-1] != nloc:
            raise ValueError(
                f"Expected FT-axis length {nloc} (local) or {ndim} (global), got {mat.shape[-1]}."
            )

        idx_local = [loc_dict[rankf][iloc] for iloc in range(nloc)]
        gathered_idx = commf.gather(idx_local, root=0)
        gathered_mat = commf.gather(mat, root=0)

        if rankf != 0:
            return None

        shape_global = list(mat.shape)
        shape_global[-1] = ndim
        matout = np.zeros(shape_global, dtype=np.complex128)

        for idx_rank, mat_rank in zip(gathered_idx, gathered_mat):
            matout[..., idx_rank] = mat_rank

        return np.asfortranarray(matout)

    def _gather_global_r_on_k_root(self, matin: np.ndarray) -> np.ndarray:

        mat = np.asfortranarray(matin, dtype=np.complex128)

        if self.nodedict is None:
            return mat

        commk = self.nodedict["commk"]
        rankk = commk.Get_rank()
        idx_local = self._kr_global_indices("rloc2glob", mat.shape[-2])

        gathered_idx = commk.gather(idx_local, root=0)
        gathered_mat = commk.gather(mat, root=0)

        if rankk != 0:
            return None

        shape_global = list(mat.shape)
        shape_global[-2] = len(self.crystal.kpoint)
        matout = np.zeros(shape_global, dtype=np.complex128)

        for idx_rank, mat_rank in zip(gathered_idx, gathered_mat):
            matout[..., idx_rank, :] = mat_rank

        return np.asfortranarray(matout)

    def _scatter_global_r_from_k_root(self, matin: np.ndarray) -> np.ndarray:

        if self.nodedict is None:
            return np.asfortranarray(matin, dtype=np.complex128)

        commk = self.nodedict["commk"]
        rankk = commk.Get_rank()
        payload = None

        if rankk == 0:
            payload = []
            for irank in range(commk.Get_size()):
                idx = [
                    self.nodedict["rloc2glob"][irank][iloc]
                    for iloc in range(len(self.nodedict["rloc2glob"][irank]))
                ]
                payload.append(np.asfortranarray(np.take(matin, idx, axis=-2)))

        matloc = commk.scatter(payload, root=0)

        return np.asfortranarray(matloc)

    def _t2f_serial(self, ftau: np.ndarray) -> np.ndarray:

        norb = ftau.shape[0]
        ns = ftau.shape[2]
        nk = ftau.shape[3]
        ntau = ftau.shape[4]
        nfreq = len(self.dlr.omega)

        ff = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128)
        batch = norb * norb * ns

        for ik in range(nk):
            ftau_block = np.moveaxis(ftau[:, :, :, ik, :], -1, 0)
            ftau_2d = np.ascontiguousarray(ftau_block).reshape(ntau, batch)

            fxx = self.dlr.dF.dlr_from_tau(ftau_2d)
            ff_2d = self.dlr.dF.matsubara_from_dlr(fxx, beta=self.dlr.beta, xi=-1)
            ff[:, :, :, ik, :] = np.moveaxis(
                ff_2d.reshape(nfreq, norb, norb, ns),
                0,
                -1,
            )

        return np.asfortranarray(ff)

    def _f2t_serial(self, ff: np.ndarray) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]
        nfreq = ff.shape[4]
        ntau = len(self.dlr.tauF)

        ftau = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128)
        batch = norb * norb * ns

        for ik in range(nk):
            ff_block = np.moveaxis(ff[:, :, :, ik, :], -1, 0)
            ff_2d = np.ascontiguousarray(ff_block).reshape(nfreq, batch)

            fxx = self.dlr.dF.dlr_from_matsubara(ff_2d, beta=self.dlr.beta, xi=-1)
            ftau_2d = self.dlr.dF.tau_from_dlr(fxx)
            ftau[:, :, :, ik, :] = np.moveaxis(
                ftau_2d.reshape(ntau, norb, norb, ns),
                0,
                -1,
            )

        return np.asfortranarray(ftau)
        
    def Inverse(self, mat : np.ndarray) -> np.ndarray:

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]
        nft = mat.shape[4]

        matinv = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128)

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    matinv[:,:,js,irk,ift] = Common.MatInv(mat[:,:,js,irk,ift])
        # for js, irk, ift in itertools.product(list(range(ns)),list(range(nrk),list(range(nft)))):
        #     matinv[:,:,js,irk,ift] = np.linalg.inv(mat[:,:,js,irk,ift])
        
        return matinv

    
    def T2F(self,ftau : np.ndarray) -> np.ndarray:

        return self._t2f_serial(np.asfortranarray(ftau))

    def F2T(self,ff : np.ndarray) -> np.ndarray:

        return self._f2t_serial(np.asfortranarray(ff))

    
    def Moment(self,ff : np.ndarray, isgreen : bool, highzero : bool) -> tuple:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]

        moment = np.zeros((norb,norb,ns,nk,3),dtype=np.complex128)
        high = np.zeros((norb,norb,ns,nk),dtype=np.complex128)

        if ff.shape[4] < 2:
            raise ValueError("Need at least two frequency points to build high-frequency moments.")

        high_freq_slice = ff[..., -1]
        prev_freq_slice = ff[..., -2]

        # moment, high = QAFort.fourier.flatdyn_m(self.dlr.omega,tempmat,isgreen,highzero)
        moment, high = Fourier.FLatDynM(self.dlr.omega, high_freq_slice, prev_freq_slice, isgreen, highzero)

        return moment, high
    
    
    def K2R(self,matk : np.ndarray) -> np.ndarray:

        rkvec = self.crystal.kpoint
        rkgrid = self.crystal.rkgrid

        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[3]
        nft = matk.shape[4]

        matr = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128)
        tempmat = copy.deepcopy(matk)
        kglob = self._kr_global_indices("kloc2glob", nrk)

        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        [a, m1] = self.crystal.FAtomOrb(iorb)
                        [b, m2] = self.crystal.FAtomOrb(jorb)

                        delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                        phase = np.exp(2.0j * np.pi * np.dot(rkvec[kglob[irk]], delta))
                        tempmat[iorb, jorb, js, irk, :] *= phase

        if self.nodedict is not None:
            from .utility.Fourier import FourierMPI as Fourier
            matr = Fourier.FLatDynK2R(tempmat, self.nodedict)
        else:
            from .utility.Fourier import Fourier
            matr = Fourier.FLatDynK2R(tempmat, rkgrid)

        return matr
    
    def R2K(self, matr : np.ndarray) -> np.ndarray:

        rkvec = self.crystal.kpoint
        rkgrid = self.crystal.rkgrid

        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[3]
        nft = matr.shape[4]

        
        
        matk = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128)

        if self.nodedict is not None:
            from .utility.Fourier import FourierMPI as Fourier
            tempmat = Fourier.FLatDynR2K(matr, self.nodedict)
        else:
            from .utility.Fourier import Fourier
            tempmat = Fourier.FLatDynR2K(matr, rkgrid)

        nkout = tempmat.shape[3]
        kglob = self._kr_global_indices("kloc2glob", nkout)

        for irk in range(nkout):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        [a, m1] = self.crystal.FAtomOrb(iorb)
                        [b, m2] = self.crystal.FAtomOrb(jorb)

                        delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                        phase = np.exp(-2.0j * np.pi * np.dot(rkvec[kglob[irk]], delta))
                        matk[iorb, jorb, js, irk, :] = tempmat[iorb, jorb, js, irk, :] * phase

        return matk
    
    
    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nrk = y.shape[3]
        nft = y.shape[4]

        ynew = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128)

        w0 = (1.0 - 3.0*w1)*np.pi*temperature
        widtharray = w0+w1*x
        cnt = 0
        for irk in range(nrk):
            for x0 in x:
                if (x0>cutoff+(w0+w1*cutoff)*3.0):
                    ynew[...,irk,cnt] = y[...,irk,cnt]
                else : 
                    if ((x0>3*widtharray[cnt])and((x[-1]-x0)>3*widtharray[cnt])):
                        dist = 1.0/np.sqrt(2*np.pi)/widtharray[cnt]*np.exp(-(x-x0)**2/2.0/widtharray[cnt]**2)
                        for js in range(ns):
                            for iorb in range(norb):
                                for jorb in range(norb):
                                    ynew[iorb,jorb,js,irk,cnt] = sum(dist*y[iorb,jorb,js,irk])/sum(dist)
                    else:
                        ynew[...,irk,cnt] = y[...,irk,cnt]
                cnt += 1

        return ynew
    
    def Mixing(self, iter : int, mix : float, Fb : np.ndarray, Fm : np.ndarray) -> np.ndarray:
        if iter == 1:
            Fm = np.zeros_like(Fb)
        return self._mixer(iter=iter, mix=mix, Fnew=Fb, Fold=Fm)
    
    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        # matout = QAFort.dyson.flatdyn(mat1,mat2)
        return Dyson.FLatDyn(mat1, mat2)
    
    def ChemEmbedding(self,mu : np.float64) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        if hasattr(self, "gkfmu0") and self.gkfmu0 is not None:
            nrk = self.gkfmu0.shape[3]
        elif hasattr(self, "gbare") and self.gbare is not None:
            nrk = self.gbare.shape[3]
        else:
            nrk = self._nk_local()
        nft = len(self.dlr.omega)#self.ft.size

        chem = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128)
        diag = np.arange(norb)
        chem[diag, diag, :, :, :] = mu

        return chem
    
    def StcEmbedding(self, matin : np.ndarray) -> np.ndarray:

        matloc = self._slice_local_k(matin)
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = matloc.shape[3]
        nft = len(self.dlr.omega)#self.ft.size

        matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128)

        for ift in range(nft):
            matout[...,ift] = matloc

        return matout
    
    
    def CheckGroup(self, filepath :str, group : str):
        
        with h5py.File(filepath,'r') as file:
            return group in file
        
    
    def Spectral(self, green : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        nfreq = len(self.dlr.omega)

        akf = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,oder='F')

        akf = -1/np.pi*green.imag

        return akf
    
    def R2KArb(self,matr : np.ndarray = None,kpoint : np.ndarray = None): # R2KAny

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nr = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        nk = len(kpoint)
        nft = matr.shape[4]

        self.crystal.RVec()
        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb,norb,ns,nk,nft),dtype=complex)

        for ift in range(nft):
            for ik in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            temp = 0
                            for ir in range(nr):
                                temp += tempmat[iorb,jorb,js,ir,ift]*np.exp(-2.0j*np.pi*(kpoint[ik]@self.crystal.rvec[ir]))
                            [a,m1] = self.crystal.FAtomOrb(iorb)
                            [b,m2] = self.crystal.FAtomOrb(jorb)
                            delta = self.crystal.basisf[a,:]-self.crystal.basisf[b,:]
                            phase = np.exp(-2.0j*np.pi*(kpoint[ik]@delta))
                            matk[iorb,jorb,js,ik,ift] = temp*phase
        
        return matk

    def KArb(self, matr : np.ndarray = None, kpoint : np.ndarray = None):

        norb = matr.shape[0]
        ns = matr.shape[2]
        nr = matr.shape[3]
        nfreq = matr.shape[4]
        nk = len(kpoint)

        tempmat = np.zeros((norb,norb,ns,nr,nfreq),dtype=complex)
        matkinv = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex)

        matrinv = self.Inverse(matr)
        omega = self.dlr.omega

        for ifreq in range(nfreq):
            for ir in range(nr):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            if iorb==jorb:
                                tempmat[iorb,jorb,js,ir,ifreq] = 1j*omega[ifreq]-matrinv[iorb,jorb,js,ir,ifreq]
                            else:
                                tempmat[iorb,jorb,js,ir,ifreq] = -matrinv[iorb,jorb,js,ir,ifreq]

        tempmat2 = self.R2KArb(tempmat,kpoint)

        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            if iorb==jorb:
                                matkinv[iorb,jorb,js,ik,ifreq] = 1j*omega[ifreq]-tempmat2[iorb,jorb,js,ik,ifreq]
                            else:
                                matkinv[iorb,jorb,js,ik,ifreq] = -tempmat2[iorb,jorb,js,ik,ifreq]
        
        matk = self.Inverse(matkinv)

        return matk
    
    def R2mR(self, matin : np.ndarray) -> np.ndarray:

        return Common.R2mR(matin, self.crystal.kpoint)
    
    def T2mT(self, ftau : np.ndarray) -> np.ndarray:

        taum = self.dlr.beta - self.dlr.tauF

        norb, _, ns, nrk, ntau = ftau.shape

        fout = np.zeros((norb, norb, ns, nrk, ntau), dtype=np.complex128)

        for irk in range(nrk):
            for js in range(ns):
                block = ftau[:, :, js, irk, :].T  # (ntau, norb, norb)
                ntau_b = block.shape[0]
                block_2d = block.reshape(ntau_b, -1)

                fxx = self.dlr.dF.dlr_from_tau(block_2d)
                fout_3d = self.dlr.dF.eval_dlr_tau(fxx[:, :, None], taum, self.dlr.beta)
                fout[:, :, js, irk, :] = fout_3d[:, :, 0].reshape(-1, norb, norb).T

        return fout

    def TauB2TauF(self, ftau : np.ndarray) -> np.ndarray:

        ntauB = len(self.dlr.tauB)
        ntauF = len(self.dlr.tauF)
        ftau_full = np.asfortranarray(ftau, dtype=np.complex128)

        norb, _, ns, ns2, nk, _ = ftau_full.shape
        fout = np.zeros((norb, norb, ns, ns2, nk, ntauF), dtype=np.complex128)
        batch = norb * norb * ns * ns2

        for ik in range(nk):
            ftau_block = np.moveaxis(ftau_full[:, :, :, :, ik, :], -1, 0)
            ftau_2d = np.ascontiguousarray(ftau_block).reshape(ntauB, batch)

            fxx = self.dlr.dB.dlr_from_tau(ftau_2d)
            fout_2d = self.dlr.dB.eval_dlr_tau(
                fxx[:, :, None], self.dlr.tauF, self.dlr.beta
            )[:, :, 0]
            fout[:, :, :, :, ik, :] = np.moveaxis(
                fout_2d.reshape(ntauF, norb, norb, ns, ns2),
                0,
                -1,
            )

        return np.asfortranarray(fout)
    
    def Diagonalize(self, matk : np.ndarray):

        norb, _, ns, nk, nfreq = matk.shape

        eigval = np.zeros((norb, norb, ns, nk, nfreq), dtype=float)
        eigvec = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128)

        for ifreq in range(nfreq):
                for ik in range(nk):
                    for js in range(ns):
                        e, v, info = scipy.linalg.lapack.zheev(matk[:, :, js, ik, ifreq])
                        eigval[:, :, js, ik, ifreq] = np.diag(e)
                        eigvec[:, :, js, ik, ifreq] = v

        return eigval, eigvec

    
class GreenBare(FLatDyn):

    def __init__(self, crystal: Crystal, dlr : DLR, nodedict : dict = None, hamtb : np.ndarray = None, hdf5file : str = None, group : str = None) -> object:
        
        super().__init__(crystal, dlr, nodedict)
        # print(self.niham.hamtb[...,0,0])
        self.hamtb = hamtb
        self.kt = None
        self.kf = None
        self.rt = None
        self.rf = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        logger.info("Bare Green's function Calculation Start")
        start = time.time()
        self.Cal()
        if hdf5file != None:
            self.Save()
        end = time.time()    
        logger.info("Bare Green's function Calculation Finish")
        logger.info(f"Calculation Time : {str(datetime.timedelta(seconds=end-start))}")
        

    def Cal(self): # freq, tau combine
        
        from .utility.Bare import Bare
        # print(self.hamtb[:,:,0,0])
        # gnotkf = QAFort.bare.flatfreq(self.hamtb,self.dlr.omega)
        gnotkf = Bare.FLatFreq(self.dlr.omega, self.hamtb)
        gnotrf = self.K2R(gnotkf)#######
        
        self.kf = gnotkf
        self.rf = gnotrf

        # gnotkt = QAFort.bare.flattau(self.hamtb,self.dlr.tau)
        gnotkt = Bare.FLatTau(tau=self.dlr.tauF, beta=self.dlr.beta, hlatt=self.hamtb)
        gnotrt = self.K2R(gnotkt)

        self.kt = gnotkt
        self.rt = gnotrt

        return None
    
    def Save(self):

        save_distributed_dataset(
            hdf5file=self.hdf5file,
            group=self.group,
            subgroup=self.subgroup,
            dataset_name="g0kf",
            data=self.kf,
            nodedict=self.nodedict,
            distributed_axes=[(3, "kloc2glob"), (4, "floc")],
        )

        return None
    
class GreenInt(FLatDyn):

    def __init__(self, crystal: Crystal, dlr : DLR, nodedict : dict = None, greenbare : np.ndarray = None, sigmah : np.ndarray = None, sigmaf : np.ndarray = None, sigmagwc : np.ndarray = None, hdf5file : str = 'glob.h5', group : str = None) -> object:
        
        if greenbare is None:
            logger.error("Bare Green's function doesn't exist")
            sys.exit()
        super().__init__(crystal, dlr, nodedict)
        self.flatstc = FLatStc(crystal=crystal, nodedict=nodedict)
        self.gbare = self._slice_local_k(greenbare)
        self.sigmah = None if sigmah is None else self._slice_local_k(sigmah)
        self.sigmaf = None if sigmaf is None else self._slice_local_k(sigmaf)
        self.sigmac = None if sigmagwc is None else self._slice_local_k(sigmagwc)
        norb, _, ns, nk, nfreq = self.gbare.shape
        ntau = len(self.dlr.tauF)
        self.kf = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128)
        self.kt = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128)
        self.rf = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128)
        self.rt = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128)
        self.gkfmu0 = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128)
        self.gktmu0 = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128)
        self.grfmu0 = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128)
        self.grtmu0 = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128)
        self.occ = None
        self.occk = None
        self.occr = None
        self.mu = np.float64(0.0)
        self.c = np.float64(0.0)
        # tau_uniform = self.dlr.TauUniform()
        # self._tau_beta = tau_uniform[-1]
        self._tau_beta = self.dlr.beta
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        
        logger.info("Interacting Green's function Calculation Start")
        start = time.time()
        self.CalMu0()
        
        self.SearchMu()
        end = time.time()
        logger.info("Interacting Green's function Calculation Finish")
        logger.info(f"Calculation Time : {str(datetime.timedelta(seconds=end-start))}")

    def CalMu0(self):

        norb, _, ns, nrk, nomega = self.gbare.shape
        sigma = np.zeros((norb,norb,ns,nrk,nomega),dtype=np.complex128)
        logger.info("Initialization start")
        if (self.sigmah is None)and(self.sigmaf is None)and(self.sigmac is None):
            self.gkfmu0 = np.array(self.gbare, dtype=np.complex128, copy=True)
        else:
            if (self.sigmah is not None):
                # print(sigma[:,:,0,0,0])
                diag = np.diagonal(self.sigmah[:,:,0,0])
                const = np.mean(diag, dtype=np.float64)
                self.c = np.real(const)
                # print(const)
                sigma += self.StcEmbedding(self.sigmah)
                sigma += self.ChemEmbedding(-const)
                # logger.info('Hartree')
                # logger.debug(sigma[:,:,0,0,0])
            if (self.sigmaf is not None):
                # print(sigma[:,:,0,0,0])
                sigma += self.StcEmbedding(self.sigmaf)
                # logger.info('Fock')
                # logger.debug(sigma[:,:,0,0,0])
            if (self.sigmac is not None):
                # print(sigma[:,:,0,0,0])
                sigma += self.sigmac
                # logger.info('GWC')
                # logger.debug(sigma[:,:,0,0,0])
            self.gkfmu0 = self.Dyson(self.gbare,sigma) 
        

        self.gktmu0 = self.F2T(self.gkfmu0)
        self.grfmu0 = self.K2R(self.gkfmu0)
        self.grtmu0 = self.K2R(self.gktmu0)
        logger.info("Initialization finish")
        return None
    
    def Occ(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk_local = self.kt.shape[3]
        nk_global = self._nk_global()
        
        
        occk = np.zeros((norb,norb,ns,nk_local),dtype=np.complex128)
        
        logger.info("Density matrixy calculation start")
        # kt = np.copy(self.kt)
        # ntau = 5000
        tau_beta = np.array([self._tau_beta], dtype=np.float64)

        for irk in range(nk_local):
            for js in range(ns):

                block = self.kt[:, :, js, irk, :].T  # (ntau, norb, norb)
                ntau_b = block.shape[0]
                block_2d = block.reshape(ntau_b, -1)

                fxx = self.dlr.dF.dlr_from_tau(block_2d)
                fout = self.dlr.dF.eval_dlr_tau(fxx[:, :, None], tau_beta, beta=self.dlr.beta)

                occk[:, :, js, irk] = -fout[0, :, 0].reshape(norb, norb)


        
        occ_local = occk.sum(axis=3)
        occ = self._allreduce_array(occ_local) / nk_global
        self.occ = occ
        self.occk = occk
        
        self.occr = self.flatstc.K2R(occk)
        logger.info("Density matrixy calculation finish")
        return None
    
    def UpdateMu(self) -> np.ndarray:

        logger.info("Chemical potential shift start")
        chem = self.ChemEmbedding(self.mu)
    
    
        gkfnew = self.Dyson(self.gkfmu0,-chem)
    
        
        self.kf = gkfnew
        self.kt = self.F2T(gkfnew)
        # self.grf = self.K2R(self.Dyson(self.gkfmu0,-chem))
        # self.grt = self.K2R(self.F2T(self.Dyson(self.gkfmu0,-chem),1,1))
        self.rf = self.K2R(self.kf)
        self.rt = self.K2R(self.kt)
        logger.info("Chemical potential shift finish")
        self.Occ()

        return None
    
    def _num_of_e_from_g0inv(self, g0inv_cache: np.ndarray, mu: np.float64) -> np.float64:

        norb = len(self.crystal.find)
        nk_global = self._nk_global()
        nfreq = len(self.dlr.omega)

        # Use cached G0inv: G(mu) = (G0inv + mu*I)^{-1}
        mat = np.array(g0inv_cache, dtype=np.complex128, copy=True)
        diag = np.arange(norb)
        mat[diag, diag, :, :, :] += mu

        # Batch invert: reshape (norb,norb,ns,nk,nfreq) -> (ns*nk*nfreq, norb,norb)
        mat_batch = np.moveaxis(mat, (0, 1), (-2, -1))  # (ns,nk,nfreq,norb,norb)
        orig_shape = mat_batch.shape[:-2]
        mat_flat = mat_batch.reshape(-1, norb, norb)
        gcalf_flat = np.linalg.inv(mat_flat)
        gcalf_batch = gcalf_flat.reshape(orig_shape + (norb, norb))
        gcalf = np.moveaxis(gcalf_batch, (-2, -1), (0, 1))  # (norb,norb,ns,nk,nfreq)

        # Extract diagonal elements only for DLR: shape (norb, ns, nk, nfreq)
        gdiag = gcalf[diag, diag, :, :, :]  # (norb, ns, nk, nfreq)

        # Batch DLR: Matsubara -> single tau point
        gdiag_perm = np.ascontiguousarray(np.moveaxis(gdiag, -1, 0))  # (nfreq, norb, ns, nk)
        batch_shape = gdiag_perm.shape[1:]
        gdiag_flat = gdiag_perm.reshape(nfreq, -1)
        fxx = self.dlr.dF.dlr_from_matsubara(gdiag_flat, beta=self.dlr.beta, xi=-1)
        fout = self.dlr.dF.eval_dlr_tau(fxx[:, :, None], self._tau_beta_cache, beta=self.dlr.beta)
        gtau_beta = fout[0, :, 0].reshape(batch_shape)  # (norb, ns, nk)

        Ne = -np.real(gtau_beta.sum()) / nk_global

        return (self.crystal.nume - Ne)

    def NumOfE(self, mu : np.float64):

        if self.nodedict is None:
            return self._num_of_e_from_g0inv(self._g0inv_cache, mu)

        commk = self.nodedict["commk"]
        rank = commk.Get_rank()
        diff = None

        if rank == 0:
            diff = self._num_of_e_from_g0inv(self._g0inv_cache_root, mu)

        return commk.bcast(diff, root=0)

    def SearchMu(self):

        logger.info("Finding chemical potential start")
        mumin = self.dlr.omega[0]
        mumax = self.dlr.omega[-1]
        logger.info(f"minimum : {mumin}, maximum : {mumax}")

        # Precompute G0^{-1} for vectorized NumOfE
        norb = len(self.crystal.find)
        g0 = self.gkfmu0  # (norb, norb, ns, nk, nfreq)
        g0_batch = np.moveaxis(g0, (0, 1), (-2, -1))  # (..., norb, norb)
        orig_shape = g0_batch.shape[:-2]
        g0_flat = g0_batch.reshape(-1, norb, norb)
        g0inv_flat = np.linalg.inv(g0_flat)
        g0inv_batch = g0inv_flat.reshape(orig_shape + (norb, norb))
        self._g0inv_cache = np.moveaxis(g0inv_batch, (-2, -1), (0, 1))  # (norb, norb, ns, nk, nfreq)
        self._g0inv_cache_root = self._gather_global_k(self._g0inv_cache)
        self._tau_beta_cache = np.array([self._tau_beta], dtype=np.float64)

        nmin = self.NumOfE(mumin)
        nmax = self.NumOfE(mumax)
        if (nmin < 0) or (nmax>0):
            logger.error("Chemical potential is out of the bisection range")
            logger.error(f"nmin : {nmin}, nmax : {nmax}")
            sys.exit()
        sol = scipy.optimize.brentq(self.NumOfE,mumin,mumax,xtol=1.0e-10)
        self.mu = sol
        logger.info("Finding chemical potential finish")

        # Clean up caches
        del self._g0inv_cache
        if hasattr(self, "_g0inv_cache_root"):
            del self._g0inv_cache_root
        del self._tau_beta_cache

        self.UpdateMu()
        return None
    
    def Save(self, fn: str, chem : bool = False):

        scalar_datasets = None
        if chem:
            scalar_datasets = {"mu": np.real(self.mu + self.c)}

        save_distributed_dataset(
            hdf5file=self.hdf5file,
            group=self.group,
            subgroup=self.subgroup,
            dataset_name=fn,
            data=self.kf,
            nodedict=self.nodedict,
            distributed_axes=[(3, "kloc2glob"), (4, "floc")],
            scalar_datasets=scalar_datasets,
        )

        return None

    
class SigmaGWC(FLatDyn):

    def __init__(self, crystal: Crystal, dlr : DLR, nodedict : dict = None, green : np.ndarray = None, wlat : np.ndarray = None, hdf5file : str = 'glob.h5',group : str = None) -> object:
        super().__init__(crystal, dlr, nodedict)
        self.flatstc = FLatStc(crystal=crystal, nodedict=nodedict)

        if green is None:
            logger.error("Error, green doesn't exist")
            sys.exit()

        if wlat is None:
            logger.error("Error, wlat doesn't exist")
            sys.exit()

        norb, _, ns, nr_local, nfreq = green.shape
        nk_local = self._nk_local()
        ntau = len(self.dlr.tauF)
        self.rt = np.zeros((norb, norb, ns, nr_local, ntau), dtype=np.complex128)
        self.rf = np.zeros((norb, norb, ns, nr_local, nfreq), dtype=np.complex128)
        self.kt = np.zeros((norb, norb, ns, nk_local, ntau), dtype=np.complex128)
        self.kf = np.zeros((norb, norb, ns, nk_local, nfreq), dtype=np.complex128)
        self.stck = np.zeros((norb, norb, ns, nk_local), dtype=np.complex128)
        self.z = np.zeros((norb, norb, ns, nk_local), dtype=np.complex128)
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        if wlat.shape[4] != nr_local:
            raise ValueError(
                f"SigmaGWC expected wlat local r-axis length {nr_local}, got {wlat.shape[4]}."
            )
        if wlat.shape[2] != ns or wlat.shape[3] != ns:
            raise ValueError(
                f"SigmaGWC spin axes mismatch: green ns={ns}, wlat ns=({wlat.shape[2]}, {wlat.shape[3]})."
            )
        if self.nodedict is not None:
            rank = self.nodedict["commk"].Get_rank()
            nr_expected = len(self.nodedict["rloc2glob"][rank])
            if nr_local != nr_expected:
                raise ValueError(
                    f"SigmaGWC expected local r-axis length {nr_expected} on rank {rank}, got {nr_local}."
                )
        self.green = green
        self.wlat = wlat

        logger.info("GWC self-energy Calculation Start")
        start = time.time()
        self.Cal()
        end = time.time()
        logger.info("GWC self-energy Calculation Finish")
        logger.info(f"Calculation Time : {str(datetime.timedelta(seconds=end-start))}")

    def Cal(self)->np.ndarray: #SigmaGWC
        '''
        Generate correlated self-energy
        input : Wc(R,t), G(R,t)

        Sigma[k,p,S] = -sum_{i,j} Wc[bb[k,i], bb[j,p], S] * G[i,j,S]

        where bb = bbasis maps fermion orbital pairs to boson indices.
        The inner loops over unique boson indices are replaced by
        direct fancy-indexing lookup of Wc via bbasis.

        return : crtau, crfreq, cktau, ckfreq
        '''

        norbc = self.green.shape[0]
        ns = self.green.shape[2]
        nr = self.green.shape[3]
        ntau = len(self.dlr.tauF)
        norb = self.wlat.shape[0]

        G = self.green

        if self.nodedict is None:
            Wc = self.TauB2TauF(self.wlat)
        else:
            commf = self.nodedict["commfermion"]
            rankf = commf.Get_rank()
            wc_local = None

            if rankf == 0:
                wlat_full_tau = self._gather_ft_on_fermion_root(
                    self.wlat, len(self.dlr.tauB), "bloc"
                )
                wlat_global = self._gather_global_r_on_k_root(wlat_full_tau)

                if self.nodedict["commk"].Get_rank() == 0:
                    wc_global = self.TauB2TauF(wlat_global)
                else:
                    wc_global = None

                wc_local = self._scatter_global_r_from_k_root(wc_global)

            Wc = np.asfortranarray(commf.bcast(wc_local, root=0))

        bbasis = self.crystal.bbasis - 1  # bbasis is 1-based, convert to 0-based
        s_idx = np.arange(ns)
        Wc_diag = Wc[:, :, s_idx, s_idx, :, :]  # (norb, norb, ns, nr, ntau)

        # Flatten (ns, nr, ntau) into single batch dim S for efficient BLAS dispatch
        S = ns * nr * ntau
        Wc_flat = Wc_diag.reshape(norb, norb, S)
        G_flat = np.ascontiguousarray(G).reshape(norbc, norbc, S)
        out_flat = np.zeros((norbc, norbc, S), dtype=np.complex128)

        # Group fermion orbitals by atom
        atom_groups = {}
        for i in range(norbc):
            a = int(self.crystal.forb2atom[i])
            atom_groups.setdefault(a, []).append(i)

        for orbs_a in atom_groups.values():
            oa = np.array(orbs_a)
            na = len(oa)
            bb_a = bbasis[np.ix_(oa, oa)]  # (na, na) — valid 0-based boson indices

            for orbs_b in atom_groups.values():
                ob = np.array(orbs_b)
                nb = len(ob)
                bb_b = bbasis[np.ix_(ob, ob)]  # (nb, nb)

                G_block = G_flat[np.ix_(oa, ob)]  # (na, nb, S)

                # Fancy-index Wc to get W4[k,i,j,p,S] = Wc[bb_a[k,i], bb_b[j,p], S]
                # bb_a[k,i] -> (na, na), bb_b[j,p] -> (nb, nb)
                # Broadcast to (na, na, nb, nb, S)
                W4 = Wc_flat[bb_a[:, :, None, None],
                             bb_b[None, None, :, :]]  # (na, na, nb, nb, S)

                # Sigma[k,p,S] = -sum_{i,j} W4[k,i,j,p,S] * G[i,j,S]
                result = np.einsum('kijpS,ijS->kpS', W4, G_block)

                out_flat[np.ix_(oa, ob)] -= result

        crtau = np.asfortranarray(out_flat.reshape(norbc, norbc, ns, nr, ntau))
        cktau = self.R2K(crtau)
        crfreq = self.T2F(crtau)
        ckfreq = self.R2K(crfreq)

        self.rt = crtau
        self.kt = cktau
        self.rf = crfreq
        self.kf = ckfreq

        return None
    
    def SigmaStc(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = self.kf.shape[3]

        sigma0 = self.kf[..., 0]
        sigma0_dag = np.transpose(np.conjugate(sigma0), (1, 0, 2, 3))
        sigmastc = 0.5 * (sigma0 + sigma0_dag)

        self.stck = np.asfortranarray(sigmastc, dtype=np.complex128)
        # self.Save('sigmastc',obj=sigmastc)

        return None
    
    def Zfactor(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = self.kf.shape[3]
        beta = self.dlr.beta

        sigma0 = self.kf[..., 0]
        sigma0_dag = np.transpose(np.conjugate(sigma0), (1, 0, 2, 3))
        iw = 1j * beta / (2.0 * np.pi)
        tempmat = np.asfortranarray(iw * (sigma0 - sigma0_dag), dtype=np.complex128)

        diag_idx = np.arange(norb)
        tempmat[diag_idx, diag_idx, :, :] += 1.0

        z = self.flatstc.Inverse(tempmat)

        self.z = z
        # self.Save('zfactor',obj=z)
        return None
    
    def Save(self, fn: str, obj : np.ndarray = None):
        data = self.kf if obj is None else obj

        distributed_axes = [(3, "kloc2glob")]
        replicated_comm_keys = ["commfermion"]

        if data.ndim == 5:
            distributed_axes.append((4, "floc"))
            replicated_comm_keys = None
        elif data.ndim != 4:
            raise ValueError(
                f"SigmaGWC.Save expected rank-4 or rank-5 data, got shape {data.shape}."
            )

        save_distributed_dataset(
            hdf5file=self.hdf5file,
            group=self.group,
            subgroup=self.subgroup,
            dataset_name=fn,
            data=data,
            nodedict=self.nodedict,
            distributed_axes=distributed_axes,
            replicated_comm_keys=replicated_comm_keys,
        )

        return None

class GreenAB(FLatDyn):

    def __init__(self, crystal: Crystal, dlr : DLR) -> object:
        super().__init__(crystal, dlr)

        glob = h5py.File('../../glob_dat/global.dat', 'r')
        self.i_kerf = glob['full_space']['gw']['i_kref'][:]
        self.kpt_latt = glob['combasis_fermion']['kpt_latt'][:]
        self.nbndf = glob['full_space']['gw']['nbndf'][:]
        self.n_omega = glob['full_space']['gw']['n_omega'][:]
        self.n3 = glob['full_space']['Gfull_n3'][:]
        glob.close()

    def KI2KF(self):

        tempmat = np.zeros((self.nbndf[0], self.nbndf[0], self.n3[0], len(self.kpt_latt), self.crystal.ns), dtype=np.complex128)

        glob = h5py.File('../../glob_dat/global.dat', 'r')

        for js in range(self.crystal.ns):
            for iw in range(self.n3[0]):
                for ik in range(len(self.kpt_latt)):
                    kidx = self.i_kerf[ik]
                    name = 'Gfull_w_'+str(iw+1)+'_k_'+str(kidx)
                    tempmat[...,iw,ik, js] = glob['full_space'][name][:]
        glob.close()
        # kpt_latt != kpoints

        self.kf = np.copy(tempmat)

        self.kt = self.F2T(tempmat)
        self.rf = self.K2R(tempmat)
        self.rt = self.K2R(self.kt)

        return None
    
