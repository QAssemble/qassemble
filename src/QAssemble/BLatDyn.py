import string as string
from typing import Any
import matplotlib as mat
import re as re
import numpy as np
import matplotlib.font_manager as fm
import sys, os
import itertools
from sympy.physics.wigner import gaunt, wigner_3j
from scipy.fftpack import fftn, ifftn
import copy, gc, time, datetime
import h5py
import time,datetime
# import Crystal, FTGrid
from .Crystal import Crystal
from .BLatStc import VBare
from .utility.DLR import DLR
from .utility.Common import Common
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
# qapath = os.environ.get("QAssemble", "")
# sys.path.append(qapath + "/src/QAssemble/modules")
# import QAFort


class BLatDyn(Crystal, DLR):
    def __init__(self, control : dict) -> object:
        Crystal.__init__(self, control['crystal'])
        DLR.__init__(self, control['dlr'])
        self._boson_phase_cache = None

    def _get_boson_phase(self) -> np.ndarray:
        if self._boson_phase_cache is not None:
            return self._boson_phase_cache

        norb = len(self.bind)
        nk = len(self.kpoint)
        phase = np.empty((norb, norb, nk), dtype=np.complex128)

        for irk, kvec in enumerate(self.kpoint):
            for iorb in range(norb):
                a, _ = self.BAtomOrb(iorb)
                for jorb in range(norb):
                    b, _ = self.BAtomOrb(jorb)
                    delta = self.basisf[a, :] - self.basisf[b, :]
                    phase[iorb, jorb, irk] = np.exp(2.0j * np.pi * np.dot(kvec, delta))

        self._boson_phase_cache = phase
        return phase

    def Inverse(self, matin: np.ndarray) -> np.ndarray:
        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]
        nft = matin.shape[5]

        matout = np.zeros(
            (norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
        )
        tempmat = np.zeros((norb * ns, norb * ns), dtype=np.complex128)
        tempmat2 = np.zeros((norb * ns, norb * ns), dtype=np.complex128)

        # Make composite matrix #
        for ift in range(nft):
            for irk in range(nrk):
                tempmat = self.OrbSpin2Composite(matin[:, :, :, :, irk, ift])
                tempmat2 = np.linalg.inv(tempmat)
                matout[:, :, :, :, irk, ift] = self.Composite2OrbSpin(tempmat2)

        return matout

    def Moment(self, bf: np.ndarray, oddzero: bool, highzero: bool) -> tuple:
        norb = bf.shape[0]
        ns = bf.shape[2]
        nrk = bf.shape[4]

        moment = np.zeros((norb, norb, ns, ns, nrk, 3), dtype=np.complex128, order="F")
        high = np.zeros((norb, norb, ns, nrk), dtype=np.complex128, order="F")

        # moment, high = QAFort.fourier.blatdyn_m(self.dlr.nu, bf, oddzero, highzero)
        moment, high = Fourier.BLatDynM(self.nu, bf, oddzero, highzero)

        return moment, high

    def F2T(self, bf: np.ndarray, nodedict: dict = None) -> np.ndarray:
        norb = bf.shape[0]
        ns = bf.shape[2]
        nrk = bf.shape[4]
        nfreq = bf.shape[5]
        ntau = len(self.tauB)
        btau = np.zeros((norb, norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F")

        if nodedict is not None:
            from mpi4py import MPI
            commf = nodedict['commf']
            floc  = nodedict['floc']
            tloc  = nodedict['tloc']
            rank  = commf.Get_rank()

            # 1) Gathering \nu slices from all ranks to form the complete \nu array
            bf_local  = np.zeros((norb, norb, ns, ns, nrk, nfreq), dtype=np.complex128, order="F")
            bf_global = np.zeros((norb, norb, ns, ns, nrk, nfreq), dtype=np.complex128, order="F")
            for loc_idx, glob_idx in floc[rank].items():
                bf_local[..., glob_idx] = bf[..., loc_idx]
            commf.Allreduce(bf_local, bf_global, op=MPI.SUM)

            # 2) Compute the Fourier transform using DLR
            btau_global = np.zeros((norb, norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F")
            for ik in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    for jorb, iorb in itertools.product(range(norb), repeat=2):
                        btau_global[iorb, jorb, js, ks, ik] = self.BF2T(bf_global[iorb, jorb, js, ks, ik])

            # 3) Separate the complete \tau array back into local slices for each rank
            for loc_idx, glob_idx in tloc[rank].items():
                btau[..., loc_idx] = btau_global[..., glob_idx]

        else:
            for ik in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    for jorb, iorb in itertools.product(range(norb), repeat=2):
                        btau[iorb, jorb, js, ks, ik] = self.BF2T(bf[iorb, jorb, js, ks, ik])

        return btau

    def T2F(self, btau: np.ndarray, nodedict: dict = None) -> np.ndarray:
        norb = btau.shape[0]
        ns = btau.shape[2]
        nrk = btau.shape[4]
        ntau = btau.shape[5]
        nfreq = len(self.nu)
        bf = np.zeros((norb, norb, ns, ns, nrk, nfreq), dtype=np.complex128, order="F")

        if nodedict is not None:
            from mpi4py import MPI
            commtau = nodedict['commtau']
            floc    = nodedict['floc']
            tloc    = nodedict['tloc']
            rank    = commtau.Get_rank()

            # 1) Gathering \tau slices from all ranks to form the complete \tau array
            btau_local  = np.zeros((norb, norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F")
            btau_global = np.zeros((norb, norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F")
            for loc_idx, glob_idx in tloc[rank].items():
                btau_local[..., glob_idx] = btau[..., loc_idx]
            commtau.Allreduce(btau_local, btau_global, op=MPI.SUM)

            # 2) Compute the Fourier transform using DLR
            bf_global = np.zeros((norb, norb, ns, ns, nrk, nfreq), dtype=np.complex128, order="F")
            for ik in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    for jorb, iorb in itertools.product(range(norb), repeat=2):
                        bf_global[iorb, jorb, js, ks, ik] = self.BT2F(btau_global[iorb, jorb, js, ks, ik])

            # 3) Separate the complete \nu array back into local slices for each rank
            for loc_idx, glob_idx in floc[rank].items():
                bf[..., loc_idx] = bf_global[..., glob_idx]

        else:
            for ik in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    for jorb, iorb in itertools.product(range(norb), repeat=2):
                        bf[iorb, jorb, js, ks, ik] = self.BT2F(btau[iorb, jorb, js, ks, ik])

        return bf

    def K2R(self, matk: np.ndarray, nodedict: dict = None) -> np.ndarray:
        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[4]
        nft = matk.shape[5]
        rkgrid = self.rkgrid

        rkvec = self.kpoint
        tempmat = matk.copy()

        if nodedict is not None:
            from .utility.Fourier import FourierMPI as Fourier
        else:
            from .utility.Fourier import Fourier

        orb2atom = np.empty(norb, dtype=np.int64)
        for iorb in range(norb):
            a, _ = self.BAtomOrb(iorb)
            orb2atom[iorb] = a
        
        basis_orb = self.basisf[orb2atom]
        kv = rkvec[:nrk]@basis_orb.T

        kv_delta = kv[:, :, None] - kv[:, None, :]

        phases = np.exp(2.0j * np.pi * kv_delta)
        phases_T = np.transpose(phases, (1, 2, 0))

        matr = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F")
        
        tempmat *= phases_T[:, :, None, None, :, None]

        matr = Fourier.BLatDynK2R(tempmat, rkgrid)

        return matr

    def R2K(self, matr: np.ndarray, nodedict: dict = None) -> np.ndarray:
        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[4]
        nft = matr.shape[5]
        rkgrid = self.rkgrid

        # phases = self._get_boson_phase()
        # phase_conj = np.conjugate(phases)[:, :, np.newaxis, np.newaxis, :]

        rkvec = self.kpoint

        if nodedict is not None:
            from .utility.Fourier import FourierMPI as Fourier
        else:
            from .utility.Fourier import Fourier

        orb2atom = np.empty(norb, dtype=np.int64)
        for iorb in range(norb):
            a, _ = self.BAtomOrb(iorb)
            orb2atom[iorb] = a

        basis_orb = self.basisf[orb2atom]

        kv = rkvec[:nrk]@basis_orb.T
        kv_delta = kv[:, :, None] - kv[:, None, :]

        phases = np.exp(-2.0j * np.pi * kv_delta)
        phases_T = np.transpose(phases, (1, 2, 0))


        matk = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F")
        tempmat = np.empty((norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F")

        tempmat = Fourier.BLatDynR2K(matr, rkgrid)

        matk = tempmat * phases_T[:, :, None, None, :, None]

        return matk

    def GaussianLinearBroad(self, x, y, w1, temperature, cutoff):
        norb = y.shape[0]
        ns = y.shape[2]
        nrk = y.shape[3]
        nft = y.shape[4]

        ynew = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F")

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

    def Mixing(
        self, iter: int, mix: float, Bb: np.ndarray, Bold: np.ndarray
    ) -> np.ndarray:
        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nrk = Bb.shape[4]
        nft = Bb.shape[5]

        Bnew = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F")

        if iter == 1:
            mix = 1.0
            Bold = np.zeros(
                (norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
            )

        Bnew = mix * Bb + (1 - mix) * Bold

        return Bnew

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
        # matout = QAFort.dyson.blatdyn(mat1, mat2)
        return Dyson.BLatDyn(mat1, mat2)

    # def Projection(self, matin: np.ndarray):
    #     norbc = self.crystal.bprojector.shape[1]
    #     ns = self.crystal.ns
    #     nft = len(self.dlr.nu)  # self.ft.size
    #     nspace = self.crystal.bprojector.shape[3]

    #     matout = np.zeros(
    #         (norbc, norbc, ns, ns, nft, nspace), dtype=np.complex128, order="F"
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
        norb = len(self.bind)

        matout = np.zeros(
            (norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
        )

        # for ift in range(nft):
        #     for irk in range(nrk):
        for irk, ift in itertools.product(list(range(nrk)), list(range(nft))):
            for ks, js in itertools.product(range(ns), repeat=2):
                matout[:, :, js, ks, irk, ift] = self.Quad2Double(
                    matin[:, :, :, :, js, ks, irk, ift]
                )

        return matout

    def Double2Quad(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.find)
        # ns = self.crystal.ns
        # nrk = len(self.crystal.kpoint)
        # nft = len(self.dlr.nu)#self.ft.size
        _, _, ns, _, nrk, nft = matin.shape

        matout = np.zeros(
            (norb, norb, norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
        )

        for ift in range(nft):
            for irk in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    matout[:, :, :, :, js, ks, irk, ift] = self.Double2Quad(
                        matin[:, :, js, ks, irk, ift]
                    )

        return matout

    def Double2Full(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.find)
        # ns = self.ns
        # nrk = len(self.kpoint)
        # nft = len(self.nu)
        _, _, ns, _, nrk, nft = matin.shape

        matout = np.zeros((norb * norb, norb * norb, ns, ns, nrk, nft), dtype=np.complex128, order="F")

        
        for ift in range(nft):
            for irk in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    matout[:, :, js, ks, irk, ift] = Crystal.Double2Full(self,
                        matin[:, :, js, ks, irk, ift]
                    )
        del matin
        gc.collect()
        return matout

    def Full2Double(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.bind)
        # ns = self.ns
        # nrk = len(self.kpoint)
        # nft = len(self.dlr.nu)#self.ft.size
        _, _, ns, _, nrk, nft = matin.shape

        matout = np.zeros(
            (norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
        )

        for ift in range(nft):
            for irk in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    matout[:, :, js, ks, irk, ift] = Crystal.Full2Double(self,
                        matin[:, :, js, ks, irk, ift]
                    )

        return matout

    def Quad2Full(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.find)
        ns = self.ns
        nrk = len(self.kpoint)
        nft = len(self.nu)  # self.ft.size

        matout = np.zeros(
            (norb * norb, norb * norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
        )

        for ift in range(nft):
            for irk in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    matout[:, :, js, ks, irk, ift] = Crystal.Quad2Full(self,
                        matin[:, :, :, :, js, ks, irk, ift]
                    )

        return matout

    def Full2Quad(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.find)
        ns = self.ns
        nrk = len(self.kpoint)
        nft = len(self.nu)  # self.ft.size

        matout = np.zeros(
            (norb, norb, norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
        )

        for ift in range(nft):
            for irk in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    matout[:, :, :, :, js, ks, irk, ift] = Crystal.Full2Quad(self,
                        matin[:, :, js, ks, irk, ift]
                    )

        return matout

    def StcEmbedding(self, matin: np.ndarray) -> np.ndarray:
        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]
        nft = len(self.nu)  # self.ft.size

        matout = np.zeros(
            (norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
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
        norb = len(self.find)
        ns = self.ns
        nr = self.rkgrid[0] * self.rkgrid[1] * self.rkgrid[2]
        nk = len(kpoint)
        nft = matr.shape[4]

        self.RVec()
        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb, norb, ns, ns, nk, nft), dtype=complex, order="F")

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
                                        * (kpoint[ik] @ self.rvec[ir])
                                    )
                                [a, m1] = self.FAtomOrb(iorb)
                                [b, m2] = self.FAtomOrb(jorb)
                                delta = (
                                    self.basisf[a, :]
                                    - self.basisf[b, :]
                                )
                                phase = np.exp(-2.0j * np.pi * (kpoint[ik] @ delta))
                                matk[iorb, jorb, js, ks, ik, ift] = temp * phase

        return matk

    def CheckGroup(self, filepath: str, group: str):
        with h5py.File(filepath, "r") as file:
            return group in file

    def RT2mRmTDLR(self, ftau: np.ndarray):
        ftau_mr = self.R2mR(ftau)
        norb, _, ns, nr, ntau = ftau_mr.shape
        fmtau_mr = np.zeros((norb, norb, ns, nr, ntau), dtype=np.complex128, order="F")

        for ir in range(nr):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        fmtau_mr[iorb, jorb, js, ir] = DLR.T2mT(self,
                            ftau_mr[iorb, jorb, js, ir]
                        )
        # fmtau_mr = self.dlr.T2mT(ftau_mr)

        return fmtau_mr
    
    def TauF2TauB(self, ftau : np.ndarray) -> np.ndarray:

        norb, _, ns, nk, _ = ftau.shape
        ntau = len(self.tauB)
        fout = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')

        for ik in range(nk):
            for js in range(ns):
                for jorb, iorb in itertools.product(range(norb), repeat=2):
                    tempmat = ftau[iorb, jorb, js, ik]
                    fout[iorb, jorb, js, ik] = DLR.TauF2TauB(self,tempmat)

        return fout
    
    def OrbSpin2Composite(self, matin: np.ndarray):
        
        norb, _, ns, _, nrk, nft = matin.shape

        matout = np.zeros((norb*ns, norb*ns, nrk, nft), dtype=np.complex128, order="F")

        for ift in range(nft):
            for irk in range(nrk):
                matout[..., irk, ift] = Crystal.OrbSpin2Composite(self, matin[:, :, :, :, irk, ift])
        
        return matout
    
    def Composite2OrbSpin(self, matin: np.ndarray):

        _, _,  nrk, nft = matin.shape

        norb = len(self.full)
        ndim = matin.shape[0]
        ns = ndim // norb

        matout = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F")

        for ift in range(nft):
            for irk in range(nrk):
                matout[..., irk, ift] = Crystal.Composite2OrbSpin(self, matin[..., irk, ift])

        return matout
        


class PolLat(BLatDyn):
    def __init__(self, control : dict, green: np.ndarray = None, hdf5file: str = "glob.h5", group: str = None):
        super().__init__(control)
        norb = len(self.find)
        ns = self.ns
        nrk = self.nk
        nfreq = len(self.nu)
        ntau = len(self.tauB)
        self.rt = np.zeros(
            (norb*norb, norb*norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F"
        )
        self.kt = np.zeros(
            (norb*norb, norb*norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F"
        )
        self.rf = np.zeros(
            (norb*norb, norb*norb, ns, ns, nrk, nfreq), dtype=np.complex128, order="F"
        )
        self.kf = np.zeros(
            (norb*norb, norb*norb, ns, ns, nrk, nfreq), dtype=np.complex128, order="F"
        )
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        if green is None:
            print("Error, There is no Green's function.")
            sys.exit()
        self.green = green

        print("Polarizability Calculation Start")
        start = time.time()
        self.Cal()
        self.kt = self.R2K(self.rt)

        self.rf = self.T2F(self.rt)
        self.kf = self.T2F(self.kt)
        end = time.time()
        print("Polarizability Calculation Done")
        print(f"Calculation Time : {str(datetime.timedelta(seconds=end-start))}")
    def Cal(self):
        # norbc = len(self.crystal.find)
        ns = self.ns
        nrk = len(self.kpoint)
        # ntau = 5000
        ntau = len(self.tauB)
        # grt = self.green
        grt = self.TauF2TauB(self.green)
        # norbf = self.green.shape[0]
        # grt = np.zeros((norbf, norbf, ns, nrk, ntau), dtype=np.complex128, order='F')
        # for irk in range(nrk):
        #     for js in range(ns):
        #         grt[:, :, js, irk, :] = self.dlr.TauDLR2Uniform(ftau=self.green[:, :, js, irk, :], ntau=ntau)
        norb = len(self.bind)

        polrt = np.zeros(
            (norb, norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F"
        )

        # gmrt = self.crystal.RT2mRmT(grt)
        gmrt = self.RT2mRmTDLR(grt)

        if ns == 2:
            map0 = np.array([self.MappingBosonFermion(i)[0] for i in range(norb)])
            map1 = np.array([self.MappingBosonFermion(i)[1] for i in range(norb)])

            term1 = gmrt[map1[np.newaxis, :], map0[:, np.newaxis], :, :, :]
            term2 = grt[map1[:, np.newaxis], map0[np.newaxis, :], :, :, :]

            diagonal_product = term1 * term2

            spin = np.arange(ns)

            polrt[:, :, spin, spin, :, :] = diagonal_product

        else:
            if self.soc == True:
                C = 1
                map0 = np.array([self.MappingBosonFermion(i)[0] for i in range(norb)])
                map1 = np.array([self.MappingBosonFermion(i)[1] for i in range(norb)])

                term1 = gmrt[map1[np.newaxis, :], map0[:, np.newaxis], :, :, :]
                term2 = grt[map1[np.newaxis, :], map0[:, np.newaxis], :, :, :]

                diagonal_product = term1 * term2

                spin = np.arange(ns)

                polrt[:, :, spin, spin, :, :] = diagonal_product * C

            else:
                C = 2
                map0 = np.array([self.MappingBosonFermion(i)[0] for i in range(norb)])
                map1 = np.array([self.MappingBosonFermion(i)[1] for i in range(norb)])

                term1 = gmrt[map1[np.newaxis, :], map0[:, np.newaxis], :, :, :]
                term2 = grt[map1[np.newaxis, :], map0[:, np.newaxis], :, :, :]

                diagonal_product = term1 * term2

                spin = np.arange(ns)

                polrt[:, :, spin, spin, :, :] = diagonal_product * C

        self.rt = polrt
        # for irk in range(nrk):
        #     for ks in range(ns):
        #         for js in range(ns):
        #             self.rt[:, :, js, ks, irk] = self.dlr.TauUniform2DLR(polrt[:, :, js, ks, irk])

        return None

    def Save(self, fn: str):
        with h5py.File(self.hdf5file, "a") as file:
            if self.CheckGroup(self.hdf5file, self.group):
                group = file[self.group]
                if self.subgroup in group:
                    pol = group[self.subgroup]
                else:
                    pol = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                pol = group.create_group(self.subgroup)
            pol.create_dataset(fn, dtype=complex, data=self.kf)

        return None


class WLat(BLatDyn):
    def __init__(self, control : dict, pol: np.ndarray = None, vbare: VBare = None, c: float = 1.0, hdf5file: str = "glob.h5", group: str = None):
        super().__init__(control)
        norb = len(self.bind)
        ns = self.ns
        nrk = self.nk
        nfreq = len(self.nu)
        ntau = len(self.tauB)

        # W quantity
        self.rt = np.zeros(
            (norb, norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F"
        )
        self.kt = np.zeros(
            (norb, norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F"
        )
        self.rf = np.zeros(
            (norb, norb, ns, ns, nrk, nfreq), dtype=np.complex128, order="F"
        )
        self.kf = np.zeros(
            (norb, norb, ns, ns, nrk, nfreq), dtype=np.complex128, order="F"
        )

        # Wc quantity
        self.crt = np.zeros(
            (norb, norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F"
        )  # rt to kf
        self.ckt = np.zeros(
            (norb, norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F"
        )
        self.crf = np.zeros(
            (norb, norb, ns, ns, nrk, nfreq), dtype=np.complex128, order="F"
        )
        self.ckf = np.zeros(
            (norb, norb, ns, ns, nrk, nfreq), dtype=np.complex128, order="F"
        )

        self.c = c
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        if pol is None:
            print("Error, polarizability doesn't exist")
            sys.exit()
        if vbare is None:
            print("Error, bare coulomb interaction doesn't exist")
            sys.exit()
        self.pol = pol
        self.vbare = vbare

        print("Screened Coulomb Interaction Calculation Start")
        start = time.time()
        self.Cal()

        # self.wkt = self.F2T(self.wkf,1,1)
        # self.wrf = self.K2R(self.wkf)
        # self.wrt = self.K2R(self.wkt)

        print(f"Fourier transform in {self.__class__.__name__} start")
        self.ckt = self.F2T(self.ckf)
        self.crf = self.K2R(self.ckf)
        self.crt = self.K2R(self.ckt)
        end= time.time()
        print(f"Fourier transform in {self.__class__.__name__} finish")
        print("Screened Coulomb Interaction Calculation Finish")
        print(f"Screened Coulomb interaction use time : {datetime.timedelta(seconds=end - start)} s")

    def Cal(self):  # calculate W and Wc
        norb = len(self.bind)
        norbc = len(self.find)
        ns = self.ns
        nk = len(self.kpoint)
        nfreq = len(self.nu)
        ####### Initialization #######
        tempmat = np.zeros(
            (norbc * norbc, norbc * norbc, ns, ns, nk, nfreq),
            dtype=np.complex128,
            order="F",
        )
        wkf = np.zeros((norb, norb, ns, ns, nk, nfreq), dtype=np.complex128, order="F")
        wckf = np.zeros((norb, norb, ns, ns, nk, nfreq), dtype=np.complex128, order="F")
        vdyn = np.zeros((norb, norb, ns, ns, nk, nfreq), dtype=np.complex128, order="F")

        # for ifreq in range(nfreq):
        #     vdyn[...,ifreq] = self.vbare.k
        print("Make dynamic bare Coulomb interaction start")
        vdyn = self.StcEmbedding(self.vbare.k)
        print("Make dynamic bare Coulomb interaction finish")
        polcomp = np.zeros(
            (norbc * norbc, norbc * norbc, ns, ns, nk, nfreq),
            dtype=np.complex128,
            order="F",
        )
        vcomp = np.zeros(
            (norbc * norbc, norbc * norbc, ns, ns, nk, nfreq),
            dtype=np.complex128,
            order="F",
        )
        ####### Initialization #######
        polcomp = self.Double2Full(self.pol) * self.c
        # del self.pol
        vcomp = self.Double2Full(vdyn)

        print("Dyson equation solving start")
        start = time.time()
        vcom2 = self.OrbSpin2Composite(vcomp)
        polcom2 = self.OrbSpin2Composite(polcomp)
        tempmat = self.Dyson(vcom2, polcom2)
        tempmat2 = self.Composite2OrbSpin(tempmat)
        wkf = self.Full2Double(tempmat2)
        end = time.time()
        # print(f"Dyson equation solving use time: {end - start} s")
        print("Dyson equation solving finish")
        print(f"Dyson equation solving use time : {datetime.timedelta(seconds=end - start)} s")

        self.kf = wkf

        wckf = wkf - vdyn

        self.ckf = wckf

        return None

    def Save(self, fn: str):
        with h5py.File(self.hdf5file, "a") as file:
            if self.CheckGroup(self.hdf5file, self.group):
                group = file[self.group]
                if self.subgroup in group:
                    w = group[self.subgroup]
                else:
                    w = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                w = group.create_group(self.subgroup)

            w.create_dataset(fn, dtype=complex, data=self.kf)

        return None


