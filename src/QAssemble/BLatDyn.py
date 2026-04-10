import numpy as np
import sys, os
import itertools
import copy, gc, time, datetime
import logging
import h5py
from .Crystal import Crystal
from .BLatStc import VBare
from .utility.Common import Common
from .utility.DLR import DLR
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
from .utility.Mixing import Mixing
from .utility.StagedHDF5 import save_distributed_dataset

logger = logging.getLogger("QAssemble")


class BLatDyn(object):
    def __init__(self, crystal: Crystal, dlr: DLR, nodedict : dict = None, mixing_method: str = "pulay", npulay: int = 5):
        self.crystal = crystal
        self.dlr = dlr
        self.nodedict = nodedict
        self.mixer = Mixing(method=mixing_method, npulay=npulay)
        # self.flatdyn = flatdyn
        self._boson_phase_cache_k2r = self._get_boson_phaseK2R()
        self._boson_phase_cache_r2k = self._get_boson_phaseR2K()

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
                    f"Serial BLatDyn expected k-axis length {nk_global}, got {nk}."
                )
            return np.array(self.crystal.kpoint[:nk], dtype=float, copy=True)

        rank = self.nodedict["commk"].Get_rank()
        nk_local = len(self.nodedict["kloc2glob"][rank])

        if nk is None:
            nk = nk_local
        if nk != nk_local:
            raise ValueError(
                f"Parallel BLatDyn expected local k-axis length {nk_local}, got {nk}."
            )

        kglob = self._kr_global_indices("kloc2glob", nk)
        return np.array(self.crystal.kpoint[kglob], dtype=float, copy=True)

    def _get_boson_phaseK2R(self, nk: int = None) -> np.ndarray:

        basis_orb = self.crystal.basisf[self.crystal.borb2atom]
        kfrac = self._get_local_kfrac(nk)
        kv = kfrac @ basis_orb.T

        kv_delta = kv[:, :, None] - kv[:, None, :]
        phases = np.exp(2.0j * np.pi * kv_delta)
        phases_T = np.transpose(phases, (1, 2, 0))

        return phases_T
    
    def _get_boson_phaseR2K(self, nk: int = None) -> np.ndarray:

        basis_orb = self.crystal.basisf[self.crystal.borb2atom]
        kfrac = self._get_local_kfrac(nk)
        kv = kfrac @ basis_orb.T

        kv_delta = kv[:, :, None] - kv[:, None, :]
        phases = np.exp(-2.0j * np.pi * kv_delta)
        phases_T = np.transpose(phases, (1, 2, 0))

        return phases_T

    def _phase_k2r(self, nk: int) -> np.ndarray:

        if self._boson_phase_cache_k2r.shape[2] == nk:
            return self._boson_phase_cache_k2r

        return self._get_boson_phaseK2R(nk)

    def _phase_r2k(self, nk: int) -> np.ndarray:

        if self._boson_phase_cache_r2k.shape[2] == nk:
            return self._boson_phase_cache_r2k

        return self._get_boson_phaseR2K(nk)

    def _allreduce_boson_ft(self, matin: np.ndarray, ndim: int) -> np.ndarray:

        mat = np.asfortranarray(matin, dtype=np.complex128)

        if self.nodedict is None:
            return mat

        if "commboson" not in self.nodedict or "bloc" not in self.nodedict:
            return mat

        commb = self.nodedict["commboson"]
        rank = commb.Get_rank()
        bloc = self.nodedict["bloc"]
        nloc = len(bloc[rank])

        if mat.shape[-1] == ndim:
            return mat

        if mat.shape[-1] != nloc:
            raise ValueError(
                f"Expected bosonic FT-axis length {nloc} (local) or {ndim} (global), got {mat.shape[-1]}."
            )

        from mpi4py import MPI

        batch_shape = mat.shape[:-1]
        mat2d = np.ascontiguousarray(np.moveaxis(mat, -1, 0).reshape(nloc, -1))
        temp = np.zeros((ndim, mat2d.shape[1]), dtype=np.complex128)

        for iloc in range(nloc):
            iglob = bloc[rank][iloc]
            temp[iglob, :] = mat2d[iloc, :]

        out = np.zeros_like(temp)
        commb.Allreduce(temp, out, op=MPI.SUM)

        matout = out.reshape((ndim,) + batch_shape)
        matout = np.moveaxis(matout, 0, -1)

        return np.asfortranarray(matout)

    def _slice_local_axis(self, matin: np.ndarray, axis: int, loc2glob_key: str) -> np.ndarray:

        mat = np.asfortranarray(matin, dtype=np.complex128)

        if self.nodedict is None:
            return mat

        rank = self.nodedict["commk"].Get_rank()
        nloc = len(self.nodedict[loc2glob_key][rank])
        nglob = len(self.crystal.kpoint)

        if mat.shape[axis] == nloc:
            return mat

        if mat.shape[axis] != nglob:
            raise ValueError(
                f"Expected axis length {nloc} (local) or {nglob} (global), got {mat.shape[axis]}."
            )

        idx = self._kr_global_indices(loc2glob_key, nloc)
        return np.asfortranarray(np.take(mat, idx, axis=axis))

    def _gather_ft_on_boson_root(self, matin: np.ndarray, ndim: int, loc_dict_key: str) -> np.ndarray:

        mat = np.asfortranarray(matin, dtype=np.complex128)

        if self.nodedict is None:
            return mat

        commb = self.nodedict["commboson"]
        rankb = commb.Get_rank()
        loc_dict = self.nodedict[loc_dict_key]
        nloc = len(loc_dict[rankb])

        if mat.shape[-1] == ndim:
            if rankb == 0:
                return mat
            return None

        if mat.shape[-1] != nloc:
            raise ValueError(
                f"Expected FT-axis length {nloc} (local) or {ndim} (global), got {mat.shape[-1]}."
            )

        idx_local = [loc_dict[rankb][iloc] for iloc in range(nloc)]
        gathered_idx = commb.gather(idx_local, root=0)
        gathered_mat = commb.gather(mat, root=0)

        if rankb != 0:
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

    def _gather_global_k_on_k_root(self, matin: np.ndarray) -> np.ndarray:

        mat = np.asfortranarray(matin, dtype=np.complex128)

        if self.nodedict is None:
            return mat

        commk = self.nodedict["commk"]
        rankk = commk.Get_rank()
        idx_local = self._kr_global_indices("kloc2glob", mat.shape[-2])

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

    def _scatter_global_k_from_k_root(self, matin: np.ndarray) -> np.ndarray:

        if self.nodedict is None:
            return np.asfortranarray(matin, dtype=np.complex128)

        commk = self.nodedict["commk"]
        rankk = commk.Get_rank()
        payload = None

        if rankk == 0:
            payload = []
            for irank in range(commk.Get_size()):
                idx = [
                    self.nodedict["kloc2glob"][irank][iloc]
                    for iloc in range(len(self.nodedict["kloc2glob"][irank]))
                ]
                payload.append(np.asfortranarray(np.take(matin, idx, axis=-2)))

        matloc = commk.scatter(payload, root=0)

        return np.asfortranarray(matloc)

    def _nk_local(self) -> int:

        if self.nodedict is None:
            return len(self.crystal.kpoint)

        rank = self.nodedict["commk"].Get_rank()
        return len(self.nodedict["kloc2glob"][rank])

    def _nr_local(self) -> int:

        if self.nodedict is None:
            return len(self.crystal.kpoint)

        rank = self.nodedict["commk"].Get_rank()
        return len(self.nodedict["rloc2glob"][rank])

    def _f2t_serial(self, bf: np.ndarray) -> np.ndarray:

        norb = bf.shape[0]
        ns = bf.shape[2]
        nrk = bf.shape[4]
        nfreq = bf.shape[5]
        ntau = len(self.dlr.tauB)

        btau = np.zeros((norb, norb, ns, ns, nrk, ntau), dtype=np.complex128)
        batch = norb * norb * ns * ns

        from scipy.linalg import lu_solve

        for ik in range(nrk):
            bf_block = np.moveaxis(bf[:, :, :, :, ik, :], -1, 0)
            bf_2d = np.ascontiguousarray(bf_block).reshape(nfreq, batch)

            G_xaa = lu_solve(
                (self.dlr.dB.dlrmf2cf, self.dlr.dB.mf2cfpiv),
                bf_2d / self.dlr.beta,
            )
            G_xaa /= self.dlr.dB.bosonic_corr_x[:, None]

            btau_2d = np.tensordot(self.dlr.dB.T_lx, G_xaa, axes=(1, 0))
            btau[:, :, :, :, ik, :] = np.moveaxis(
                btau_2d.reshape(ntau, norb, norb, ns, ns),
                0,
                -1,
            )

        return np.asfortranarray(btau)

    def _t2f_serial(self, btau: np.ndarray) -> np.ndarray:

        norb = btau.shape[0]
        ns = btau.shape[2]
        nrk = btau.shape[4]
        ntau = btau.shape[5]
        nfreq = len(self.dlr.nu)

        bf = np.zeros((norb, norb, ns, ns, nrk, nfreq), dtype=np.complex128)
        batch = norb * norb * ns * ns

        from scipy.linalg import lu_solve

        for ik in range(nrk):
            btau_block = np.moveaxis(btau[:, :, :, :, ik, :], -1, 0)
            btau_2d = np.ascontiguousarray(btau_block).reshape(ntau, batch)

            fxx = lu_solve((self.dlr.dB.dlrit2cf, self.dlr.dB.it2cfpiv), btau_2d)
            bf_2d = self.dlr.beta * np.tensordot(
                self.dlr.dB.T_qx * self.dlr.dB.bosonic_corr_x[None, :],
                fxx,
                axes=(1, 0),
            )
            bf[:, :, :, :, ik, :] = np.moveaxis(
                bf_2d.reshape(nfreq, norb, norb, ns, ns),
                0,
                -1,
            )

        return np.asfortranarray(bf)

    def Inverse(self, matin: np.ndarray) -> np.ndarray:
        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]
        nft = matin.shape[5]

        matout = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=np.complex128)
        tempmat = np.zeros((norb * ns, norb * ns), dtype=np.complex128)
        tempmat2 = np.zeros((norb * ns, norb * ns), dtype=np.complex128)

        # Make composite matrix #
        for ift in range(nft):
            for irk in range(nrk):
                tempmat = self.crystal.OrbSpin2Composite(matin[:, :, :, :, irk, ift])
                tempmat2 = np.linalg.inv(tempmat)
                matout[:, :, :, :, irk, ift] = self.crystal.Composite2OrbSpin(tempmat2)

        return matout

    def Moment(self, bf: np.ndarray, oddzero: bool, highzero: bool) -> tuple:
        norb = bf.shape[0]
        ns = bf.shape[2]
        nrk = bf.shape[4]

        moment = np.zeros((norb, norb, ns, ns, nrk, 3), dtype=np.complex128)
        high = np.zeros((norb, norb, ns, nrk), dtype=np.complex128)

        # moment, high = QAFort.fourier.blatdyn_m(self.dlr.nu, bf, oddzero, highzero)
        moment, high = Fourier.BLatDynM(self.dlr.nu, bf, oddzero, highzero)

        return moment, high

    def F2T(self, bf: np.ndarray) -> np.ndarray:
        bf_full = self._allreduce_boson_ft(bf, len(self.dlr.nu))

        return self._f2t_serial(bf_full)

    def T2F(self, btau: np.ndarray) -> np.ndarray:
        btau_full = self._allreduce_boson_ft(btau, len(self.dlr.tauB))

        return self._t2f_serial(btau_full)

    def K2R(self, matk: np.ndarray) -> np.ndarray:
        
        rkgrid = self.crystal.rkgrid
        nrk = matk.shape[4]

        phase_view = self._phase_k2r(nrk)[:, :, np.newaxis, np.newaxis, :, np.newaxis]
        tempmat = np.empty_like(matk)
        np.multiply(matk, phase_view, out=tempmat)

        if self.nodedict is not None:
            from .utility.Fourier import FourierMPI as Fourier
            matr = Fourier.BLatDynK2R(tempmat, self.nodedict)
        else:
            from .utility.Fourier import Fourier
            matr = Fourier.BLatDynK2R(tempmat, rkgrid)

        return matr

    def R2K(self, matr: np.ndarray) -> np.ndarray:

        rkgrid = self.crystal.rkgrid

        if self.nodedict is not None:
            from .utility.Fourier import FourierMPI as Fourier
            tempmat = Fourier.BLatDynR2K(matr, self.nodedict)
        else:
            from .utility.Fourier import Fourier
            tempmat = Fourier.BLatDynR2K(matr, rkgrid)

        nkout = tempmat.shape[4]
        phase_view = self._phase_r2k(nkout)[:, :, np.newaxis, np.newaxis, :, np.newaxis]
        matk = np.empty_like(tempmat)
        np.multiply(tempmat, phase_view, out=matk)

        return matk

    def GaussianLinearBroad(self, x, y, w1, temperature, cutoff):
        norb = y.shape[0]
        ns = y.shape[2]
        nrk = y.shape[3]
        nft = y.shape[4]

        ynew = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=np.complex128)

        w0 = (1.0 - 3.0 * w1) * np.pi * temperature
        widtharray = w0 + w1 * x
        cnt = 0
        for irk in range(nrk):
            for x0 in x:
                if x0 > cutoff + (w0 + w1 * cutoff) * 3.0:
                    ynew[..., irk, cnt] = y[..., irk, cnt]
                else:
                    if (x0 > 3 * widtharray[cnt]) and (
                        (x[-1] - x0) > 3 * widtharray[cnt]
                    ):
                        dist = (
                            1.0
                            / np.sqrt(2 * np.pi)
                            / widtharray[cnt]
                            * np.exp(-((x - x0) ** 2) / 2.0 / widtharray[cnt] ** 2)
                        )
                        for js in range(ns):
                            for ks in range(ns):
                                for iorb in range(norb):
                                    for jorb in range(norb):
                                        ynew[iorb, jorb, js, ks, irk, cnt] = sum(
                                            dist * y[iorb, jorb, js, ks, irk]
                                        ) / sum(dist)
                    else:
                        ynew[..., irk, cnt] = y[..., irk, cnt]
                cnt += 1

        return ynew

    def Mixing(self, iter: int, mix: float, Bb: np.ndarray, Bold: np.ndarray) -> np.ndarray:
        if iter == 1:
            Bold = np.zeros_like(Bb)
        return self.mixer(iter=iter, mix=mix, Fnew=Bb, Fold=Bold)

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
        # matout = QAFort.dyson.blatdyn(mat1, mat2)
        return Dyson.BLatDyn(mat1, mat2)

    # def Projection(self, matin: np.ndarray):
    #     norbc = self.crystal.bprojector.shape[1]
    #     ns = self.crystal.ns
    #     nft = len(self.dlr.nu)  # self.ft.size
    #     nspace = self.crystal.bprojector.shape[3]

    #     matout = np.zeros(
    #         (norbc, norbc, ns, ns, nft, nspace), dtype=np.complex128
    #     )

    #     for ispace in range(nspace):
    #         matout[..., ispace] = QAFort.projection.blatdyn(
    #             matin, self.crystal.bprojector[..., ispace]
    #         )

    #     return matout

    def Quad2Double(self, matin: np.ndarray) -> np.ndarray:
        # norb = len(self.crystal.bind)
        # ns = self.crystal.ns
        # nrk = len(self.crystal.kpoint)
        # nft = len(self.dlr.nu)#self.ft.size
        _, _, _, _, ns, _, nrk, nft = matin.shape
        norb = len(self.crystal.bind)

        matout = np.zeros(
            (norb, norb, ns, ns, nrk, nft), dtype=np.complex128
        )

        # for ift in range(nft):
        #     for irk in range(nrk):
        for irk, ift in itertools.product(list(range(nrk)), list(range(nft))):
            for ks, js in itertools.product(range(ns), repeat=2):
                matout[:, :, js, ks, irk, ift] = self.crystal.Quad2Double(
                    matin[:, :, :, :, js, ks, irk, ift]
                )

        return matout

    def Double2Quad(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.find)
        # ns = self.crystal.ns
        # nrk = len(self.crystal.kpoint)
        # nft = len(self.dlr.nu)#self.ft.size
        _, _, ns, _, nrk, nft = matin.shape

        matout = np.zeros(
            (norb, norb, norb, norb, ns, ns, nrk, nft), dtype=np.complex128
        )

        for ift in range(nft):
            for irk in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    matout[:, :, :, :, js, ks, irk, ift] = self.crystal.Double2Quad(
                        matin[:, :, js, ks, irk, ift]
                    )

        return matout

    def Double2Full(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.find)
        _, _, ns, _, nrk, nft = matin.shape
        nind = norb * norb
        c2b = np.asarray(self.crystal.c2b, dtype=np.int64)

        matout = np.zeros(
            (nind, nind, ns, ns, nrk, nft), dtype=np.complex128
        )
        matout[np.ix_(c2b, c2b)] = matin

        del matin
        gc.collect()
        return matout

    def Full2Double(self, matin: np.ndarray) -> np.ndarray:
        c2b = np.asarray(self.crystal.c2b, dtype=np.int64)

        matout = np.asarray(matin[np.ix_(c2b, c2b)], dtype=np.complex128)

        return matout

    def Quad2Full(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.dlr.nu)  # self.ft.size

        matout = np.zeros(
            (norb * norb, norb * norb, ns, ns, nrk, nft), dtype=np.complex128
        )

        for ift in range(nft):
            for irk in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    matout[:, :, js, ks, irk, ift] = self.crystal.Quad2Full(
                        matin[:, :, :, :, js, ks, irk, ift]
                    )

        return matout

    def Full2Quad(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.dlr.nu)  # self.ft.size

        matout = np.zeros(
            (norb, norb, norb, norb, ns, ns, nrk, nft), dtype=np.complex128
        )

        for ift in range(nft):
            for irk in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    matout[:, :, :, :, js, ks, irk, ift] = self.crystal.Full2Quad(
                        matin[:, :, js, ks, irk, ift]
                    )

        return matout

    def StcEmbedding(self, matin: np.ndarray) -> np.ndarray:
        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]
        nft = len(self.dlr.nu)  # self.ft.size

        matout = np.zeros(
            (norb, norb, ns, ns, nrk, nft), dtype=np.complex128
        )

        for ift in range(nft):
            matout[..., ift] += matin
        del matin
        gc.collect()
        return matout

    def Save(self, matin: np.ndarray, fn: str):
        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]
        nft = matin.shape[5]

        if os.path.exists("blatdyn"):
            pass
        else:
            os.mkdir("blatdyn")
        os.chdir("blatdyn")

        with open(fn + ".txt", "w") as f:
            f.write("iorb, jorb, is, js, irk, ift, Re(B(k,w)), Im(B(k,w))\n")
            for ift in range(nft):
                for irk in range(nrk):
                    for ks, js in itertools.product(range(ns), repeat=2):
                        for jorb, iorb in itertools.product(range(norb), repeat=2):
                            f.write(
                                f"{iorb} {jorb} {js} {ks} {irk} {ift} {matin[iorb, jorb, js, ks, irk, ift].real} {matin[iorb, jorb, js, ks, irk, ift].imag}\n"
                            )

        os.chdir("..")

        return None

    def R2KArb(self, matr: np.ndarray = None, kpoint: np.ndarray = None):  # R2KAny
        # if self.crystal.kpath == None:
        #     print("Error, kpath doesn't generate")
        #     sys.exit()
        # kpoint = self.crystal.kpath
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nr = self.crystal.rkgrid[0] * self.crystal.rkgrid[1] * self.crystal.rkgrid[2]
        nk = len(kpoint)
        nft = matr.shape[4]

        self.crystal.RVec()
        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb, norb, ns, ns, nk, nft), dtype=complex)

        for ift in range(nft):
            for ik in range(nk):
                for ks in range(ns):
                    for js in range(ns):
                        for jorb in range(norb):
                            for iorb in range(norb):
                                temp = 0
                                for ir in range(nr):
                                    temp += tempmat[
                                        iorb, jorb, js, ks, ir, ift
                                    ] * np.exp(
                                        -2.0j
                                        * np.pi
                                        * (kpoint[ik] @ self.crystal.rvec[ir])
                                    )
                                [a, m1] = self.crystal.FAtomOrb(iorb)
                                [b, m2] = self.crystal.FAtomOrb(jorb)
                                delta = (
                                    self.crystal.basisf[a, :]
                                    - self.crystal.basisf[b, :]
                                )
                                phase = np.exp(-2.0j * np.pi * (kpoint[ik] @ delta))
                                matk[iorb, jorb, js, ks, ik, ift] = temp * phase

        return matk

    def CheckGroup(self, filepath: str, group: str):
        with h5py.File(filepath, "r") as file:
            return group in file

    def RT2mRmT(self, ftau: np.ndarray):
        ftau_mr = Common.R2mR(ftau, self.crystal.kpoint)
        norb, _, ns, nr, ntau = ftau_mr.shape
        fmtau_mr = np.zeros((norb, norb, ns, nr, ntau), dtype=np.complex128)

        for ir in range(nr):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        fmtau_mr[iorb, jorb, js, ir] = self.dlr.T2mT(
                            ftau_mr[iorb, jorb, js, ir]
                        )
        # fmtau_mr = self.dlr.T2mT(ftau_mr)

        return fmtau_mr
    
    def TauF2TauB(self, ftau : np.ndarray) -> np.ndarray:

        norb, _, ns, nk, _ = ftau.shape
        ntau = len(self.dlr.tauB)
        fout = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128)

        for ik in range(nk):
            for js in range(ns):
                for jorb, iorb in itertools.product(range(norb), repeat=2):
                    tempmat = ftau[iorb, jorb, js, ik]
                    fout[iorb, jorb, js, ik] = self.dlr.TauF2TauB(tempmat)

        return fout


class PolLat(BLatDyn):
    def __init__(self,crystal: Crystal,dlr: DLR,nodedict : dict = None,green: np.ndarray = None,hdf5file: str = "glob.h5",group: str = None,):
        super().__init__(crystal, dlr, nodedict)
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = self._nr_local()
        nkk = self._nk_local()
        nfreq = len(self.dlr.nu)
        ntau = len(self.dlr.tauB)
        self.rt = np.zeros(
            (norb, norb, ns, ns, nrk, ntau), dtype=np.complex128
        )
        self.kt = np.zeros(
            (norb, norb, ns, ns, nkk, ntau), dtype=np.complex128
        )
        self.rf = np.zeros(
            (norb, norb, ns, ns, nrk, nfreq), dtype=np.complex128
        )
        self.kf = np.zeros(
            (norb, norb, ns, ns, nkk, nfreq), dtype=np.complex128
        )
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        if green is None:
            logger.error("Error, There is no Green's function.")
            sys.exit()
        self.green = self._slice_local_axis(green, 3, "rloc2glob")

        logger.info("Polarizability Calculation Start")
        start = time.time()
        self.Cal()
        self.kt = self.R2K(self.rt)

        self.rf = self.T2F(self.rt)
        self.kf = self.T2F(self.kt)
        end = time.time()
        logger.info("Polarizability Calculation Done")
        logger.info(f"Calculation Time : {str(datetime.timedelta(seconds=end-start))}")

    def _build_polrt(self, grt: np.ndarray, gmrt: np.ndarray) -> np.ndarray:

        ns = self.crystal.ns
        nrk = grt.shape[3]
        ntau = len(self.dlr.tauB)
        norb = len(self.crystal.bind)

        polrt = np.zeros(
            (norb, norb, ns, ns, nrk, ntau), dtype=np.complex128
        )

        if ns == 2:
            map0 = np.array([self.crystal.MappingBosonFermion(i)[0] for i in range(norb)])
            map1 = np.array([self.crystal.MappingBosonFermion(i)[1] for i in range(norb)])

            term1_tensor = gmrt[map1[np.newaxis, :], map0[:, np.newaxis], :, :, :]
            term2_tensor = grt[map1[:, np.newaxis], map0[np.newaxis, :], :, :, :]
            diagonal_product = term1_tensor * term2_tensor
            s_indices = np.arange(ns)

            polrt[:, :, s_indices, s_indices, :, :] = diagonal_product

        else:
            if self.crystal.soc == True:
                coeff = 1
            else:
                coeff = 2

            map0 = np.array([self.crystal.MappingBosonFermion(i)[0] for i in range(norb)])
            map1 = np.array([self.crystal.MappingBosonFermion(i)[1] for i in range(norb)])

            term1_slice = gmrt[map1[np.newaxis, :], map0[:, np.newaxis], 0, :, :]
            term2_slice = grt[map1[:, np.newaxis], map0[np.newaxis, :], 0, :, :]
            result_slice = term1_slice * term2_slice * coeff
            polrt[:, :, 0, 0, :, :] = result_slice

        return np.asfortranarray(polrt)
        
    def Cal(self):
        
        if self.nodedict is None:
            grt = self.TauF2TauB(self.green)
            gmrt = self.RT2mRmT(grt)
            self.rt = self._build_polrt(grt, gmrt)
            return None

        commb = self.nodedict["commboson"]
        rankb = commb.Get_rank()

        green_full_tau = self._gather_ft_on_boson_root(
            self.green, len(self.dlr.tauF), "floc"
        )

        polrt_local = None

        if rankb == 0:
            green_global = self._gather_global_r_on_k_root(green_full_tau)

            if self.nodedict["commk"].Get_rank() == 0:
                grt_global = self.TauF2TauB(green_global)
                gmrt_global = self.RT2mRmT(grt_global)
                polrt_global = self._build_polrt(grt_global, gmrt_global)
            else:
                polrt_global = None

            polrt_local = self._scatter_global_r_from_k_root(polrt_global)

        self.rt = np.asfortranarray(commb.bcast(polrt_local, root=0))


        return None

    def Save(self, fn: str):
        save_distributed_dataset(
            hdf5file=self.hdf5file,
            group=self.group,
            subgroup=self.subgroup,
            dataset_name=fn,
            data=self.kf,
            nodedict=self.nodedict,
            distributed_axes=[(4, "kloc2glob"), (5, "bloc")],
        )

        return None


class WLat(BLatDyn):
    def __init__(self,crystal: Crystal,dlr: DLR,nodedict : dict = None,pol: np.ndarray = None,vbare: np.ndarray = None,c: float = 1.0,hdf5file: str = "glob.h5", group: str = None,):
        super().__init__(crystal, dlr, nodedict)
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nr_local = self._nr_local()
        nk_local = self._nk_local()
        nk_expected = nk_local if self.nodedict is not None else len(self.crystal.kpoint)
        nfreq = len(self.dlr.nu)
        ntau = len(self.dlr.tauB)

        # W quantity
        self.rt = np.zeros(
            (norb, norb, ns, ns, nr_local, ntau), dtype=np.complex128
        )
        self.kt = np.zeros(
            (norb, norb, ns, ns, nk_local, ntau), dtype=np.complex128
        )
        self.rf = np.zeros(
            (norb, norb, ns, ns, nr_local, nfreq), dtype=np.complex128
        )
        self.kf = np.zeros(
            (norb, norb, ns, ns, nk_local, nfreq), dtype=np.complex128
        )

        # Wc quantity
        self.crt = np.zeros(
            (norb, norb, ns, ns, nr_local, ntau), dtype=np.complex128
        )  # rt to kf
        self.ckt = np.zeros(
            (norb, norb, ns, ns, nk_local, ntau), dtype=np.complex128
        )
        self.crf = np.zeros(
            (norb, norb, ns, ns, nr_local, nfreq), dtype=np.complex128
        )
        self.ckf = np.zeros(
            (norb, norb, ns, ns, nk_local, nfreq), dtype=np.complex128
        )

        self.c = c
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        if pol is None:
            logger.error("Error, polarizability doesn't exist")
            sys.exit()
        if vbare is None:
            logger.error("Error, bare coulomb interaction doesn't exist")
            sys.exit()
        self.pol = np.asfortranarray(pol, dtype=np.complex128)
        self.vbare_k = np.asfortranarray(vbare, dtype=np.complex128)

        if self.pol.ndim != 6:
            raise ValueError(
                f"Polarizability must be rank-6, got shape {self.pol.shape}."
            )
        if self.vbare_k.ndim != 5:
            raise ValueError(
                f"Bare Coulomb interaction must be rank-5, got shape {self.vbare_k.shape}."
            )
        if self.pol.shape[4] != nk_expected:
            raise ValueError(
                f"Polarizability k-axis length {self.pol.shape[4]} does not match expected nk {nk_expected}."
            )
        if self.vbare_k.shape[4] != nk_expected:
            raise ValueError(
                f"VBare k-axis length {self.vbare_k.shape[4]} does not match expected nk {nk_expected}."
            )
        if self.pol.shape[5] != nfreq:
            raise ValueError(
                f"Polarizability frequency-axis length {self.pol.shape[5]} does not match DLR size {nfreq}."
            )

        logger.info("Screened Coulomb Interaction Calculation Start")
        start = time.time()
        self.Cal()

        # self.wkt = self.F2T(self.wkf,1,1)
        # self.wrf = self.K2R(self.wkf)
        # self.wrt = self.K2R(self.wkt)

        logger.info(f"Fourier transform in {self.__class__.__name__} start")
        self.ckt = self.F2T(self.ckf)
        self.crf = self.K2R(self.ckf)
        self.crt = self.K2R(self.ckt)
        end= time.time()
        logger.info(f"Fourier transform in {self.__class__.__name__} finish")
        logger.info("Screened Coulomb Interaction Calculation Finish")
        logger.info(f"Screened Coulomb interaction use time : {datetime.timedelta(seconds=end - start)} s")

    def Cal(self):  # calculate W and Wc
        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns
        nk = self.pol.shape[4]
        nfreq = self.pol.shape[5]
        ####### Initialization #######
        wkf = np.zeros((norb, norb, ns, ns, nk, nfreq), dtype=np.complex128)

        # for ifreq in range(nfreq):
        #     vdyn[...,ifreq] = self.vbare.k
        logger.info("Make dynamic bare Coulomb interaction start")
        vdyn = self.StcEmbedding(self.vbare_k)
        logger.info("Make dynamic bare Coulomb interaction finish")
        ####### Initialization #######
        polcomp = self.Double2Full(self.pol) * self.c
        # del self.pol
        vcomp = self.Double2Full(vdyn)

        logger.info("Dyson equation solving start")
        start = time.time()

        if self.nodedict is None:
            tempmat = self.Dyson(vcomp, polcomp)
            wkf = self.Full2Double(tempmat)
            wckf = wkf - vdyn
        else:
            vdyn_global = self._gather_global_k_on_k_root(vdyn)
            polcomp_global = self._gather_global_k_on_k_root(polcomp)
            vcomp_global = self._gather_global_k_on_k_root(vcomp)

            wkf_global = None
            wckf_global = None

            if self.nodedict["commk"].Get_rank() == 0:
                tempmat = self.Dyson(vcomp_global, polcomp_global)
                wkf_global = self.Full2Double(tempmat)
                wckf_global = wkf_global - vdyn_global

            wkf = self._scatter_global_k_from_k_root(wkf_global)
            wckf = self._scatter_global_k_from_k_root(wckf_global)

        end = time.time()
        logger.info("Dyson equation solving finish")
        logger.info(f"Dyson equation solving use time : {datetime.timedelta(seconds=end - start)} s")

        self.kf = wkf
        self.ckf = wckf

        return None

    def Save(self, fn: str):
        save_distributed_dataset(
            hdf5file=self.hdf5file,
            group=self.group,
            subgroup=self.subgroup,
            dataset_name=fn,
            data=self.kf,
            nodedict=self.nodedict,
            distributed_axes=[(4, "kloc2glob"), (5, "bloc")],
        )

        return None
