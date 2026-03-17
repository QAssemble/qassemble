import numpy as np
import sys, os
import itertools
import copy, gc, time, datetime
import h5py
from .Crystal import Crystal
from .BLatStc import VBare
from .utility.DLR import DLR
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
# qapath = os.environ.get("QAssemble", "")
# sys.path.append(qapath + "/src/QAssemble/modules")
# import QAFort


class BLatDyn(object):
    def __init__(self, crystal: Crystal, dlr: DLR):
        self.crystal = crystal
        self.dlr = dlr
        # self.flatdyn = flatdyn
        self._boson_phase_cache_k2r = self._get_boson_phaseK2R()
        self._boson_phase_cache_r2k = self._get_boson_phaseR2K()

    def _get_boson_phaseK2R(self) -> np.ndarray:
        

        nrk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]

        basis_orb = self.crystal.basisf[self.crystal.borb2atom]

        kv = self.crystal.kpoint[:nrk] @ basis_orb.T

        kv_delta = kv[:, :, None] - kv[:, None, :]
        phases = np.exp(2.0j * np.pi * kv_delta)
        phases_T = np.transpose(phases, (1, 2, 0))

        return phases_T
    
    def _get_boson_phaseR2K(self) -> np.ndarray:

        nrk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]

        basis_orb = self.crystal.basisf[self.crystal.borb2atom]

        kv = self.crystal.kpoint[:nrk] @ basis_orb.T

        kv_delta = kv[:, :, None] - kv[:, None, :]
        phases = np.exp(-2.0j * np.pi * kv_delta)
        phases_T = np.transpose(phases, (1, 2, 0))

        return phases_T

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
                tempmat = self.crystal.OrbSpin2Composite(matin[:, :, :, :, irk, ift])
                tempmat2 = np.linalg.inv(tempmat)
                matout[:, :, :, :, irk, ift] = self.crystal.Composite2OrbSpin(tempmat2)

        return matout

    def Moment(self, bf: np.ndarray, oddzero: bool, highzero: bool) -> tuple:
        norb = bf.shape[0]
        ns = bf.shape[2]
        nrk = bf.shape[4]

        moment = np.zeros((norb, norb, ns, ns, nrk, 3), dtype=np.complex128, order="F")
        high = np.zeros((norb, norb, ns, nrk), dtype=np.complex128, order="F")

        # moment, high = QAFort.fourier.blatdyn_m(self.dlr.nu, bf, oddzero, highzero)
        moment, high = Fourier.BLatDynM(self.dlr.nu, bf, oddzero, highzero)

        return moment, high

    def F2T(self, bf: np.ndarray) -> np.ndarray:
        norb = bf.shape[0]
        ns = bf.shape[2]
        nrk = bf.shape[4]
        nfreq = bf.shape[5]
        ntau = len(self.dlr.tauB)

        btau = np.zeros((norb, norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F")
        tempmat = np.zeros((nfreq), dtype=np.complex128, order="F")

        for ik in range(nrk):
            for ks, js in itertools.product(range(ns), repeat=2):
                # tempmat = np.transpose(bf[:, :, js, ks, ik], (2, 0, 1))
                # tempmat2 = self.dlr.BF2T(tempmat)
                for jorb, iorb in itertools.product(range(norb), repeat=2):
                    tempmat = bf[iorb, jorb, js, ks, ik]
                    tempmat2 = self.dlr.BF2T(tempmat)
                    # btau[:, :, js, ks, ik] = np.transpose(tempmat2, (1, 2, 0))
                    # for jorb, iorb in itertools.product(range(norb), repeat=2):
                    btau[iorb, jorb, js, ks, ik] = tempmat2

        return btau

    def T2F(self, btau: np.ndarray) -> np.ndarray:
        norb = btau.shape[0]
        ns = btau.shape[2]
        nrk = btau.shape[4]
        ntau = btau.shape[5]
        nfreq = len(self.dlr.nu)

        bf = np.zeros((norb, norb, ns, ns, nrk, nfreq), dtype=np.complex128, order="F")
        tempmat = np.zeros((ntau), dtype=np.complex128, order="F")

        for ik in range(nrk):
            for ks, js in itertools.product(range(ns), repeat=2):
                # tempmat = np.transpose(btau[:, :, js, ks, ik], (2, 0, 1))
                # tempmat2 = self.dlr.BT2F(tempmat)
                # bf[:, :, js, ks, ik] = np.transpose(tempmat2, (1, 2, 0))
                # bf[:, :, js, ks, ik, :] = self.dlr.BT2F(btau[:, :, js, ks, ik, :])
                for jorb, iorb in itertools.product(range(norb), repeat=2):
                    tempmat = btau[iorb, jorb, js, ks, ik]
                    tempmat2 = self.dlr.BT2F(tempmat)
                    # for jorb, iorb in itertools.product(range(norb), repeat=2):
                    bf[iorb, jorb, js, ks, ik] = tempmat2

        return bf

    def K2R(self, matk: np.ndarray) -> np.ndarray:
        
        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[4]
        nft = matk.shape[5]
        
        matr = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F")

        tempmat *= self._boson_phase_cache_k2r[:, :, None, None, :, None]

        matr = Fourier.BLatDynK2R(matk, self.crystal.rkgrid)


        return matr

    def R2K(self, matr: np.ndarray) -> np.ndarray:
        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[4]
        nft = matr.shape[5]
        rkgrid = self.crystal.rkgrid
        
        matk = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F")
        tempmat = np.empty((norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F")

        tempmat = Fourier.BLatDynR2K(matr, rkgrid)

        matk = tempmat * self._boson_phase_cache_r2k[:, :, None, None, :, None]
        

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
        norb = len(self.crystal.bind)

        matout = np.zeros(
            (norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
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
            (norb, norb, norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
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
        # ns = self.crystal.ns
        # nrk = len(self.crystal.kpoint)
        # nft = len(self.dlr.nu)#self.ft.size
        _, _, ns, _, nrk, nft = matin.shape

        matout = np.zeros(
            (norb * norb, norb * norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
        )

        for ift in range(nft):
            for irk in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    matout[:, :, js, ks, irk, ift] = self.crystal.Double2Full(
                        matin[:, :, js, ks, irk, ift]
                    )
        del matin
        gc.collect()
        return matout

    def Full2Double(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.bind)
        # ns = self.crystal.ns
        # nrk = len(self.crystal.kpoint)
        # nft = len(self.dlr.nu)#self.ft.size
        _, _, ns, _, nrk, nft = matin.shape

        matout = np.zeros(
            (norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
        )

        for ift in range(nft):
            for irk in range(nrk):
                for ks, js in itertools.product(range(ns), repeat=2):
                    matout[:, :, js, ks, irk, ift] = self.crystal.Full2Double(
                        matin[:, :, js, ks, irk, ift]
                    )

        return matout

    def Quad2Full(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.dlr.nu)  # self.ft.size

        matout = np.zeros(
            (norb * norb, norb * norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
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
            (norb, norb, norb, norb, ns, ns, nrk, nft), dtype=np.complex128, order="F"
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
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nr = self.crystal.rkgrid[0] * self.crystal.rkgrid[1] * self.crystal.rkgrid[2]
        nk = len(kpoint)
        nft = matr.shape[4]

        self.crystal.RVec()
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
        ftau_mr = self.crystal.R2mR(ftau)
        norb, _, ns, nr, ntau = ftau_mr.shape
        fmtau_mr = np.zeros((norb, norb, ns, nr, ntau), dtype=np.complex128, order="F")

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
        fout = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')

        for ik in range(nk):
            for js in range(ns):
                for jorb, iorb in itertools.product(range(norb), repeat=2):
                    tempmat = ftau[iorb, jorb, js, ik]
                    fout[iorb, jorb, js, ik] = self.dlr.TauF2TauB(tempmat)

        return fout


class PolLat(BLatDyn):
    def __init__(self,crystal: Crystal,dlr: DLR,green: np.ndarray = None,hdf5file: str = "glob.h5",group: str = None,):
        super().__init__(crystal, dlr)
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = self.crystal.nk
        nfreq = len(self.dlr.nu)
        ntau = len(self.dlr.tauB)
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
        
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        
        ntau = len(self.dlr.tauB)
        
        grt = self.TauF2TauB(self.green)
    
        norb = len(self.crystal.bind)

        polrt = np.zeros(
            (norb, norb, ns, ns, nrk, ntau), dtype=np.complex128, order="F"
        )

        # gmrt = self.crystal.RT2mRmT(grt)
        gmrt = self.RT2mRmT(grt)

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
                C = 1
                map0 = np.array([self.crystal.MappingBosonFermion(i)[0] for i in range(norb)])
                map1 = np.array([self.crystal.MappingBosonFermion(i)[1] for i in range(norb)])

                term1_slice = gmrt[map1[np.newaxis, :], map0[:, np.newaxis], 0, :, :]
                term2_slice = grt[map1[:, np.newaxis], map0[np.newaxis, :], 0, :, :]
                result_slice = term1_slice * term2_slice * C
                polrt[:, :, 0, 0, :, :] = result_slice

            else:
                C = 2
                map0 = np.array([self.crystal.MappingBosonFermion(i)[0] for i in range(norb)])
                map1 = np.array([self.crystal.MappingBosonFermion(i)[1] for i in range(norb)])

                term1_slice = gmrt[map1[np.newaxis, :], map0[:, np.newaxis], 0, :, :]
                term2_slice = grt[map1[:, np.newaxis], map0[np.newaxis, :], 0, :, :]
                result_slice = term1_slice * term2_slice * C
                polrt[:, :, 0, 0, :, :] = result_slice

        self.rt = polrt
        

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
    def __init__(self,crystal: Crystal,dlr: DLR,pol: np.ndarray = None,vbare: VBare = None,c: float = 1.0,hdf5file: str = "glob.h5", group: str = None,):
        super().__init__(crystal, dlr)
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = self.crystal.nk
        nfreq = len(self.dlr.nu)
        ntau = len(self.dlr.tauB)

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
        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nfreq = len(self.dlr.nu)
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
        tempmat = self.Dyson(vcomp, polcomp)
        wkf = self.Full2Double(tempmat)
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


