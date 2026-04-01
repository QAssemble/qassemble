import copy
import gc
import itertools
import sys

import h5py
import numpy as np

from .BLocStc import VLoc
from .Crystal import Crystal
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson



class BLatStc(object):

    def __init__(self, crystal: Crystal):
        self.crystal = crystal
        self._phase_cache = None

    def _get_phase(self) -> np.ndarray:
        if self._phase_cache is not None:
            return self._phase_cache

        norb = len(self.crystal.bind)
        nk = len(self.crystal.kpoint)
        phase = np.empty((norb, norb, nk), dtype=np.complex128)

        for irk, kvec in enumerate(self.crystal.kpoint):
            for iorb in range(norb):
                a, _ = self.crystal.BAtomOrb(iorb)
                for jorb in range(norb):
                    b, _ = self.crystal.BAtomOrb(jorb)
                    delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                    phase[iorb, jorb, irk] = np.exp(2.0j * np.pi * np.dot(kvec, delta))

        self._phase_cache = phase
        return phase

    def Inverse(self, matin: np.ndarray) -> np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]

        matout = np.zeros((norb, norb, ns, ns, nrk), dtype=np.complex128, order="F")
        tempmat = np.zeros((norb * ns, norb * ns), dtype=np.complex128)
        tempmat2 = np.zeros((norb * ns, norb * ns), dtype=np.complex128)

        for irk in range(nrk):
            tempmat = self.crystal.OrbSpin2Composite(matin[..., irk])
            tempmat2 = np.linalg.inv(tempmat)
            matout[..., irk] = self.crystal.Composite2OrbSpin(tempmat2)

        return matout

    def K2R(self, matk: np.ndarray) -> np.ndarray:

        rkgrid = self.crystal.rkgrid
        norb = matk.shape[0]
        ns = self.crystal.ns
        nrk = len(rkvec)

        phases = self._get_phase()
        matr = np.zeros((norb, norb, ns, ns, nrk), dtype=np.complex128, order="F")
        tempmat = np.empty((norb, norb, ns, ns, nrk), dtype=np.complex128, order="F")
        phase_view = phases[:, :, np.newaxis, np.newaxis, :]
        np.multiply(matk, phase_view, out=tempmat)

        # matr = QAFort.fourier.blatstc_k2r(rkgrid, tempmat)
        matr = Fourier.BLatStcK2R(tempmat, rkgrid)

        return matr

    def R2K(self, matr: np.ndarray) -> np.ndarray:

        rkgrid = self.crystal.rkgrid
        norb = matr.shape[0]
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        phases = self._get_phase()
        phase_conj = np.conjugate(phases)[:, :, np.newaxis, np.newaxis, :]
        tempmat = np.empty((norb, norb, ns, ns, nrk), dtype=np.complex128, order="F")

        # matk = QAFort.fourier.blatstc_r2k(rkgrid, matr)
        tempk = Fourier.BLatStcR2K(matr, rkgrid)
        np.multiply(tempk, phase_conj, out=tempmat)
        matk = tempmat

        return matk

    def Mixing(
        self, iter: int, mix: float, Bb: np.ndarray, Bold: np.ndarray
    ) -> np.ndarray:

        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nrk = Bb.shape[4]

        Bnew = np.zeros((norb, norb, ns, ns, nrk), dtype=np.complex128, order="F")

        if iter == 1:
            mix = 1.0

        Bnew = mix * Bb + (1.0 - mix) * Bold

        return Bnew

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray):
        # matout = QAFort.dyson.blatstc(mat1, mat2)
        return Dyson.BLatDyn(mat1, mat2)

    # def Projection(self, matin: np.ndarray):

    #     norbc = self.crystal.bprojector.shape[1]
    #     nspace = self.crystal.bprojector.shape[3]
    #     ns = self.crystal.ns

    #     matout = np.zeros(
    #         (norbc, norbc, ns, ns, nspace), dtype=np.complex128, order="F"
    #     )

    #     for ispace in range(nspace):
    #         matout[..., ispace] = QAFort.projection.blatstc(
    #             matin, self.crystal.bprojector[..., ispace]
    #         )

    #     return matout

    def Quad2Double(self, matin: np.ndarray) -> np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros((norb, norb, ns, ns, nrk), dtype=np.complex128, order="F")

        for irk in range(nrk):
            for ks in range(ns):
                for js in range(ns):
                    matout[:, :, js, ks, irk] = self.crystal.Quad2Double(
                        matin[:, :, :, :, js, ks, irk]
                    )

        return matout

    def Double2Quad(self, matin: np.ndarray) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros(
            (norb, norb, norb, norb, ns, ns, nrk), dtype=np.complex128, order="F"
        )

        for irk in range(nrk):
            for ks in range(ns):
                for js in range(ns):
                    matout[:, :, :, :, js, ks, irk] = self.crystal.Double2Quad(
                        matin[:, :, js, ks, irk]
                    )

        return matout

    def Double2Full(self, matin: np.ndarray) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros(
            (norb * norb, norb * norb, ns, ns, nrk), dtype=np.complex128, order="F"
        )

        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    matout[:, :, js, ks, irk] = self.crystal.Double2Full(
                        matin[:, :, js, ks, irk]
                    )

        return matout

    def Full2Double(self, matin: np.ndarray) -> np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros((norb, norb, ns, ns, nrk), dtype=np.complex128, order="F")

        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    matout[:, :, js, ks, irk] = self.crystal.Full2Double(
                        matin[:, :, js, ks, irk]
                    )

        return matout

    def Quad2Full(self, matin: np.ndarray) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros(
            (norb * norb, norb * norb, ns, ns, nrk), dtype=np.complex128, order="F"
        )

        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    matout[:, :, js, ks, irk] = self.crystal.Quad2Full(
                        matin[:, :, :, :, js, ks, irk]
                    )

        return matout

    def Full2Quad(self, matin: np.ndarray) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros(
            (norb, norb, norb, norb, ns, ns, nrk), dtype=np.complex128, order="F"
        )

        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    matout[:, :, :, :, js, ks, irk] = self.crystal.Full2Quad(
                        matin[:, :, js, ks, irk]
                    )

        return matout

    def Save(self, matin: np.ndarray, fn: str):

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]

        # if os.path.exists('blatstc'):
        #     pass
        # else:
        #     os.mkdir("blatstc")
        # os.chdir('blatstc')

        with open(fn + ".txt", "w") as f:
            f.write("#iorb, jorb, is, js, irk, Re(B(k)), Im(B(k))\n")
            for irk in range(nrk):
                for ks in range(ns):
                    for js in range(ns):
                        for jorb in range(norb):
                            for iorb in range(norb):
                                f.write(
                                    f"{iorb} {jorb} {js} {ks} {irk} {matin[iorb,jorb,js,ks,irk].real} {matin[iorb,jorb,js,ks,irk].imag}\n"
                                )

        # os.chdir('..')

        return None

    def HermitianCheck(self, matin: np.ndarray):

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nk = self.crystal.rkgrid[0] * self.crystal.rkgrid[1] * self.crystal.rkgrid[2]

        errmessage = "The matrix is not hermitian. Check the input file again"
        for ik in range(nk):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            err = matin[iorb, jorb, js, ks, ik] - np.conjugate(
                                matin[jorb, iorb, js, ks, ik]
                            )
                            if abs(err) > 1.0e-6:
                                print(errmessage)
                                sys.exit()
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

        self.crystal.RVec()
        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb, norb, ns, ns, nk), dtype=complex, order="F")

        for ik in range(nk):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            temp = 0
                            for ir in range(nr):
                                temp += tempmat[iorb, jorb, js, ks, ir] * np.exp(
                                    -2.0j * np.pi * (kpoint[ik] @ self.crystal.rvec[ir])
                                )
                            [a, m1] = self.crystal.FAtomOrb(iorb)
                            [b, m2] = self.crystal.FAtomOrb(jorb)
                            delta = (
                                self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                            )
                            phase = np.exp(-2.0j * np.pi * (kpoint[ik] @ delta))
                            matk[iorb, jorb, js, ks, ik] = temp * phase

        return matk

    def CheckGroup(self, filepath: str, group: str):

        with h5py.File(filepath, "r") as file:
            return group in file


class VBare(BLatStc):

    def __init__(
        self,
        crystal: Crystal,
        vloc: VLoc = None,
        orboption: dict = None,
        intamp: dict = None,
        ohno: bool = False,
        jth: bool = False,
        ohnoyuka: bool = False,
        hdf5file: str = None,
        group: str = None,
    ):
        super().__init__(crystal)
        self.k = None
        self.r = None
        self.intamp = None

        if intamp != None:
            intamplist = []
            for orb, val in intamp.items():
                for v, lat in val.items():
                    for r in lat:
                        intamplist.append([v, list(orb[0]), list(orb[1]), r])
            self.intamp = intamplist
        self.locoption = orboption
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = self.crystal.rkgrid[0] * self.crystal.rkgrid[1] * self.crystal.rkgrid[2]
        self.nonlock = np.zeros((norb, norb, ns, ns, nrk), dtype=complex, order="F")
        self.nonlocr = np.zeros((norb, norb, ns, ns, nrk), dtype=complex, order="F")
        self.sigmaonsiter = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        print("Bare Coulomb Interaction Calculation Start")
        if (ohno == False) and (intamp == None) and (jth == False):
            print("Only calculate the local coulomb interaction")
        if vloc == None:
            if orboption != None:
                self.vloc = VLoc(crystal, orboption)
            else:
                print("Error, orboption is not exsist. v local can't generate in here")
        else:
            self.vloc = vloc

        if ohno:
            self.OhnoParameter()
            # self.Cal()
        elif jth:
            print("JTH Potential calculation start")
            self.JTHPotential()
            print("JTH Potential calculation finish")
        elif ohnoyuka:
            print("Ohno-Yukawa calculation start")
            self.OhnoYukawa()
            print("Ohno-Yukawa calculation finish")
        else:
            if intamp != None:
                # self.InteractingAmplitue(intamp)
                self.Cal()
        self.LocPlusNonLoc()
        if hdf5file != None:
            self.Save()
        # self.GetOnsiteEnergy()
        print("Bare Coulomb Interaction Calculation Finish")

    def Cal(self):

        errmessage = "Wrong value entered, please check the input.ini file"
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nk = len(rkvec)
        vnlk = np.zeros((norb, norb, ns, ns, nk), dtype=np.complex128, order="F")
        tempmat = np.zeros(
            (norb, norb, ns, ns, rkgrid[0], rkgrid[1], rkgrid[2]),
            dtype=np.complex128,
            order="F",
        )

        for js in range(ns):
            for ks in range(ns):
                for ind in self.intamp:
                    vij = ind[0]
                    (a, m) = ind[1]
                    (b, mp) = ind[2]
                    iorb = self.crystal.BIndex([a, [m, m]])
                    jorb = self.crystal.BIndex([b, [mp, mp]])
                    R = ind[3]

                    # tempmat[iorb,jorb,js,ks,R[0],R[1],R[2]] += vij

                    if (iorb == jorb) and (R == [0, 0, 0]):
                        # tempmat[iorb,jorb,js,ks,R[0],R[1],R[2]] += vij
                        print(errmessage)
                        sys.exit()

                    # else:
                    tempmat[iorb, jorb, js, ks, R[0], R[1], R[2]] = vij
                    tempmat[jorb, iorb, js, ks, -R[0], -R[1], -R[2]] = vij

        vnlr = tempmat.reshape((norb, norb, ns, ns, nk), order="F")
        vnlk = self.R2K(vnlr)
        self.HermitianCheck(vnlk)

        self.nonlocr = vnlr
        self.nonlock = vnlk

        return None

    # def InteractingAmplitue(self,intamp : list)-> list:

    #     pass

    def LocPlusNonLoc(self):

        vloc = self.vloc.vloc
        # print(vloc[:, :, 0, 0])
        #       vnlk = self.nonlock

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        vbare = np.zeros((norb, norb, ns, ns, nrk), dtype=np.complex128, order="F")
        # if (self.nonlock == None):
        #     for ik in range(nrk):
        #         vbare[...,ik] = vloc
        # else:
        #     for ik in range(nrk):
        #         vbare[...,ik] = vloc + vnlk[...,ik]
        # for ik in range(nrk):
        #     vbare[...,ik] = vloc + vnlk[...,ik]
        vbare = copy.deepcopy(self.nonlocr)
        vbare[..., 0] += vloc
        #       self.k = vbare
        #       self.r = self.K2R(vbare)
        self.r = vbare
        self.k = self.R2K(vbare)

        return None

    def Save(self):

        with h5py.File(self.hdf5file, "a") as file:
            if self.CheckGroup(self.hdf5file, self.group):
                group = file[self.group]
                if self.subgroup in group:
                    vbare = group[self.subgroup]
                else:
                    vbare = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                vbare = group.create_group(self.subgroup)
            vbare.create_dataset("vk", dtype=complex, data=self.k)

        return None

    def OhnoParameter(self):

        ns = self.crystal.ns
        norb = len(self.crystal.bind)
        natom = len(self.crystal.basisf)
        R = copy.deepcopy(self.crystal.rkgrid)

        Rij = 0
        nr = R[0] * R[1] * R[2]
        vr = np.zeros((norb, norb, ns, ns, nr), dtype=complex, order="F")
        tempmat = np.zeros(
            (norb, norb, ns, ns, R[0], R[1], R[2]), dtype=complex, order="F"
        )
        a0 = 0.592
        au = 27.2114
        vloc = copy.deepcopy(self.vloc.vloc)
        # Rtemp = []
        if self.intamp != None:
            for ks in range(ns):
                for js in range(ns):
                    for ind in self.intamp:
                        vij = ind[0]
                        (a, m) = ind[1]
                        (b, mp) = ind[2]
                        # Rtemp.append(ind[3])
                        iorb = self.crystal.BIndex([a, [m, m]])
                        jorb = self.crystal.BIndex([b, [mp, mp]])
                        vloc[iorb, jorb, js, ks] = vij
                        vloc[jorb, iorb, js, ks] = vij

        self.vloc.vloc = copy.deepcopy(vloc)
        print("Ohno loop start")
        for ks in range(ns):
            for js in range(ns):
                for jatom in range(natom):
                    jj = self.crystal.orboption[jatom]
                    j_orb_list = list(range(jj))
                    for m3, m2 in itertools.product(j_orb_list, j_orb_list):
                        jorb = self.crystal.BIndex([jatom, [m2, m3]])
                        for iatom in range(natom):
                            ii = self.crystal.orboption[iatom]
                            i_orb_list = list(range(ii))
                            for m4, m1 in itertools.product(i_orb_list, i_orb_list):
                                iorb = self.crystal.BIndex([iatom, [m1, m4]])
                                Uij = self.vloc.vloc[iorb, jorb, js, ks]
                                U = Uij / au
                                # print(f'Uij : {Uij}, iorb : {iorb}, jorb : {jorb}')
                                for iz in range(R[2]):
                                    for iy in range(R[1]):
                                        for ix in range(R[0]):
                                            # print(f'ix : {ix}, iy : {iy}, iz : {iz}')
                                            if [ix, iy, iz] == [0, 0, 0]:
                                                continue
                                            else:
                                                rvec = np.array(
                                                    [ix, iy, iz], dtype=float
                                                )
                                                ind = np.where(
                                                    (self.crystal.rind == rvec).all(
                                                        axis=1
                                                    )
                                                )[0][0]
                                                rvec = self.crystal.rvec[ind]
                                                delta = self.crystal.basisc[
                                                    iatom, :
                                                ] - (
                                                    self.crystal.basisc[jatom, :]
                                                    + rvec @ self.crystal.avec
                                                )
                                                # rij = self.RMin(
                                                #     delta, iatom, jatom, rvec
                                                # )
                                                epsilon = 1e-10
                                                rij = self.RMin2(delta)
                                                Rij = rij / a0
                                                vij = (
                                                    1
                                                    / (Rij**2 + 1 / (U + epsilon) ** 2)
                                                    ** (0.5)
                                                    # * np.exp(-Rij)
                                                    * au
                                                )
                                                tempmat[
                                                    iorb, jorb, js, ks, ix, iy, iz
                                                ] = vij

        # for ks in range(ns):
        #     for js in range(ns):
        #         for jatom in range(natom):
        #             jj = self.crystal.orboption[jatom]
        #             j_orb_list = list(range(jj))
        #             for m3, m2 in itertools.product(j_orb_list,j_orb_list):
        #                 jorb = self.crystal.BIndex([jatom,[m2,m3]])
        #                 for iatom in range(natom):
        #                     ii = self.crystal.orboption[iatom]
        #                     i_orb_list = list(range(ii))
        #                     for m4, m1 in itertools.product(i_orb_list,i_orb_list):
        #                         iorb = self.crystal.BIndex([iatom,[m1,m4]])
        #                         Uij = self.vloc.vloc[iorb,jorb,js,ks]
        #                         U = Uij/au
        #                         if (iorb <= jorb):
        #                             for iz in range(R[2]):
        #                                 for iy in range(R[1]):
        #                                     for ix in range(R[0]):
        #                                         if ([ix,iy,iz]==[0,0,0]):
        #                                             continue
        #                                         else:
        #                                             rvec = np.array([ix,iy,iz],dtype=float)
        #                                             ind = np.where((self.crystal.rind==rvec).all(axis=1))[0][0]
        #                                             rvec = self.crystal.rvec[ind]
        #                                             delta = self.crystal.basisc[iatom,:] - (self.crystal.basisc[jatom,:]+rvec@self.crystal.avec)
        #                                             rij = self.RMin(delta,iatom,jatom,rvec)
        #                                             Rij = rij/a0
        #                                             vij = 1/(Rij**2+1/U**2)**(0.5)*au
        #                                             tempmat[iorb,jorb,js,ks,ix,iy,iz] = vij
        #                                             tempmat[jorb,iorb,js,ks,-ix,-iy,-iz] = vij

        print("Ohno loop end")
        if self.intamp != None:
            for ks in range(ns):
                for js in range(ns):
                    for ind in self.intamp:
                        vij = ind[0]
                        (a, m) = ind[1]
                        (b, mp) = ind[2]
                        [ix, iy, iz] = ind[3]
                        if [ix, iy, iz] == [0, 0, 0]:
                            continue
                        else:
                            iorb = self.crystal.BIndex([a, [m, m]])
                            jorb = self.crystal.BIndex([b, [mp, mp]])
                            tempmat[iorb, jorb, js, ks, ix, iy, iz] = vij
                            tempmat[jorb, iorb, js, ks, -ix, -iy, -iz] = vij

        vr = tempmat.reshape((norb, norb, ns, ns, nr), order="F")
        # self.intamp = V

        self.nonlocr = copy.deepcopy(vr)
        self.nonlock = self.R2K(vr)

        return None

    def JTHPotential(self):

        ns = self.crystal.ns
        norb = len(self.crystal.bind)
        natom = len(self.crystal.basisf)
        R = copy.deepcopy(self.crystal.rkgrid)
        nr = R[0] * R[1] * R[2]
        vr = np.zeros((norb, norb, ns, ns, nr), dtype=complex, order="F")
        tempmat = np.zeros(
            (norb, norb, ns, ns, R[0], R[1], R[2]), dtype=complex, order="F"
        )
        # vr = np.zeros((norbc,norbc,norbc,norbc,ns,ns,nr),dtype=complex,order='F')
        a0 = 0.592
        au = 27.2114

        for ks in range(ns):
            for js in range(ns):
                for jatom in range(natom):
                    jj = self.crystal.orboption[jatom]
                    j_orb_list = list(range(jj))
                    for m3, m2 in itertools.product(j_orb_list, j_orb_list):
                        jorb = self.crystal.BIndex([jatom, [m2, m3]])
                        for iatom in range(natom):
                            ii = self.crystal.orboption[iatom]
                            i_orb_list = list(range(ii))
                            for m4, m1 in itertools.product(i_orb_list, i_orb_list):
                                iorb = self.crystal.BIndex([iatom, [m1, m4]])
                                # if iorb <= jorb:
                                Ui = self.vloc.vloc[iorb, iorb, js, ks]
                                Uj = self.vloc.vloc[jorb, jorb, js, ks]
                                U = (Ui + Uj) / 2.0 / au
                                for iz, iy, ix in itertools.product(
                                    list(range(self.crystal.rkgrid[2])),
                                    list(range(self.crystal.rkgrid[1])),
                                    list(range(self.crystal.rkgrid[0])),
                                ):
                                    if ([ix, iy, iz] == [0, 0, 0]) and (iorb == jorb):
                                        continue
                                    # ind = np.where(self.crystal.rind == np.array([ix,iy,iz],dtype=float).all( axis = 1))[0][0]
                                    rvec = np.array([ix, iy, iz], dtype=float)
                                    ind = np.where(
                                        (self.crystal.rind == rvec).all(axis=1)
                                    )[0][0]
                                    rvec = self.crystal.rvec[ind]
                                    delta = self.crystal.basisc[iatom, :] - (
                                        self.crystal.basisc[jatom, :]
                                        + rvec[0] * self.crystal.avec[0]
                                        + rvec[1] * self.crystal.avec[1]
                                        + rvec[2] * self.crystal.avec[2]
                                    )
                                    rij = self.RMin2(delta)
                                    Rij = rij / a0
                                    # vij = (1/(Rij**2 + 1/(U**2))**0.5 * np.exp(-Rij)) * au
                                    vij = (1 / (Rij**2 + 1 / (U**2)) ** 0.5) * au
                                    tempmat[iorb, jorb, js, ks, ix, iy, iz] = vij
                                    # tempmat[jorb, iorb, js, ks, -ix, -iy, -iz] = vij

        vr = tempmat.reshape((norb, norb, ns, ns, nr), order="F")
        # self.intamp = V

        self.nonlocr = copy.deepcopy(vr)
        self.nonlock = self.R2K(vr)

        del (
            vr,
            tempmat,
            Rij,
            rij,
            vij,
            U,
            # delta,
            rvec,
            R,
            iorb,
            jorb,
            ix,
            iy,
            iz,
            js,
            ks,
            au,
            a0,
        )
        gc.collect()

        return None

    def OhnoYukawa(self):

        ns = self.crystal.ns
        norb = len(self.crystal.bind)
        natom = len(self.crystal.basisf)
        R = copy.deepcopy(self.crystal.rkgrid)
        nr = R[0] * R[1] * R[2]

        vr = np.zeros((norb, norb, ns, ns, nr), dtype=np.complex128, order="F")
        tempmat = np.zeros(
            (norb, norb, ns, ns, R[0], R[1], R[2]), dtype=np.complex128, order="F"
        )

        a0 = 0.592
        au = 27.2114
        q0 = 0.001

        for ks in range(ns):
            for js in range(ns):
                for jatom in range(natom):
                    jj = self.crystal.orboption[jatom]
                    j_orb_list = list(range(jj))
                    for m3, m2 in itertools.product(j_orb_list, j_orb_list):
                        jorb = self.crystal.BIndex([jatom, [m2, m3]])
                        for iatom in range(natom):
                            ii = self.crystal.orboption[iatom]
                            i_orb_list = list(range(ii))
                            for m4, m1 in itertools.product(i_orb_list, i_orb_list):
                                iorb = self.crystal.BIndex([iatom, [m1, m4]])

                                Ui = self.vloc.vloc[iorb, iorb, js, ks]
                                Uj = self.vloc.vloc[jorb, jorb, js, ks]

                                U = (Ui + Uj) / 2.0 / au

                                for iz, iy, ix in itertools.product(
                                    list(range(R[2])),
                                    list(range(R[1])),
                                    list(range(R[0])),
                                ):
                                    if ([ix, iy, iz] == [0, 0, 0]) and (iorb == jorb):
                                        continue

                                    rvec = np.array([ix, iy, iz], dtype=float)
                                    ind = np.where(
                                        (self.crystal.rind == rvec).all(axis=1)
                                    )[0][0]

                                    rvec = self.crystal.rvec[ind]

                                    delta = self.crystal.basisc[iatom, :] - (
                                        self.crystal.basisc[jatom, :]
                                        + rvec[0] * self.crystal.avec[0]
                                        + rvec[1] * self.crystal.avec[1]
                                        + rvec[2] * self.crystal.avec[2]
                                    )

                                    rij = self.RMin2(delta)
                                    Rij = rij / a0

                                    vij = (
                                        (1 / np.sqrt(Rij**2 + 1 / (U**2))) * np.exp(-q0 * a0 * Rij)
                                    ) * au

                                    tempmat[iorb, jorb, js, ks, ix, iy, iz] = vij

        vr = tempmat.reshape((norb, norb, ns, ns, nr), order="F")

        self.nonlocr = np.copy(vr)
        self.nonlock = self.R2K(vr)

        del (
            vr,
            tempmat,
            Rij,
            rij,
            vij,
            U,
            rvec,
            R,
            iorb,
            jorb,
            ix,
            iy,
            iz,
            js,
            ks,
            au,
            a0,
        )
        gc.collect()

        return None

    #   def RMin(self, d: np.ndarray, a: int, b: int, rvec: np.ndarray):

    #       # print("Minimizing the distance between atoms calculation start")
    #       # R = 1000000
    #       # Rtemp = 0
    #       # r = 0
    #       # R1 = 0
    #       # R2 = 0
    #       # dtemp = 0
    #       # svec = self.crystal.svec
    #       # # print(f"initial : {d}, guess : {dtemp}")
    #       # R1 = np.linalg.norm(d)
    #       # for iz in range(-1, 2):
    #       #     for iy in range(-1, 2):
    #       #         for ix in range(-1, 2):
    #       #             r = np.array(
    #       #                 [ix * svec[0], iy * svec[1], iz * svec[2]], dtype=np.float64
    #       #             )
    #       #             dtemp = self.crystal.basisc[a, :] - (
    #       #                 self.crystal.basisc[b, :] + rvec @ self.crystal.avec + r
    #       #             )
    #       #             R2 = np.linalg.norm(dtemp)
    #       #             if R1 > R2:
    #       #                 Rtemp = R2
    #       #             else:
    #       #                 Rtemp = R1

    #       #             if abs(Rtemp) == 0:
    #       #                 pass
    #       #             if Rtemp < R:
    #       #                 R = Rtemp

    #       # # print(f"R min : {R}")
    #       # # print("Minimizing the distance between atoms calculation end")
    #       # del Rtemp, R1, R2, dtemp
    #       # gc.collect()

    #       # return R
    #       comm = MPI.COMM_WORLD
    #       rank = comm.Get_rank()
    #       size = comm.Get_size()

    #       R = 1000000
    #       Rtemp = 0
    #       r = 0
    #       R1 = 0
    #       R2 = 0
    #       dtemp = 0
    #       svec = self.crystal.svec
    #       R1 = np.linalg.norm(d)

    #       # Distribute the work among processes
    #       for iz in range(rank - 1, 2, size):
    #           for iy in range(-1, 2):
    #               for ix in range(-1, 2):
    #                   if (ix==0)and(iy==0)and(iz==0):
    #                       continue
    #                   r = np.array(
    #                       [ix * svec[0], iy * svec[1], iz * svec[2]], dtype=np.float64
    #                   )
    #                   dtemp = self.crystal.basisc[a, :] - (
    #                       self.crystal.basisc[b, :] + rvec @ self.crystal.avec + r
    #                   )
    #                   R2 = np.linalg.norm(dtemp)
    #                   if R1 > R2:
    #                       Rtemp = R2
    #                   else:
    #                       Rtemp = R1

    #                   if abs(Rtemp) == 0:
    #                       pass
    #                   if Rtemp < R:
    #                       R = Rtemp

    #       # Gather the minimum R from all processes
    #       R = comm.allreduce(R, op=MPI.MIN)

    #       # ... existing code ...

    #       return R

    def RMin2(self, d: np.ndarray):
        from .utility.Common import Common
        svec = self.crystal.svec


        R = Common.MinDistance(svec, d)

        return R

    # def BoundaryCheck(self,a : int, b : int, rvec : np.ndarray):

    #     taua = copy.deepcopy(self.crystal.basisf[a,:])
    #     taub = copy.deepcopy(self.crystal.basisf[b,:])
    #     r = rvec@self.crystal.avec
    #     delta = self.crystal.basisc[a,:] - (self.crystal.basisc[b,:] + r)
    #     rmin1 = self.RMin2(delta)
    #     # rmin = 0.0
    #     taua_p = taua.copy()
    #     taub_p = taub.copy()
    #     if (a == b):
    #         return rmin1
    #     else:
    #         rmin = 1.0e6
    #         rtemp = 0
    #         if (taua[0] < 1.0e-6):
    #             taua_p[0] += 1
    #         if (taua[1] < 1.0e-6):
    #             taua_p[1] += 1
    #         if (taub[0] < 1.0e-6):
    #             taub_p[0] += 1
    #         if (taub[1] < 1.0e-6):
    #             taub_p[1] += 1
    #         delta2 = taua_p@self.crystal.avec - (taub_p@self.crystal.avec +r)
    #         rmin2 = self.RMin2(delta2)

    #         if (rmin1 <= rmin2):
    #             rtemp = rmin1
    #         else:
    #             rtemp = rmin2
    #         if (rtemp < 1.0e-6):
    #             pass
    #         if (rtemp < rmin):
    #             rmin = rtemp
    #         return rmin
    #     # if (np.array_equal(rvec,[0,0,0])):#(rvec == [0,0,0]):
    #     #     rmin = 0.0
    #     #     if (taub[0]< 1.0e-6):
    #     #         taub_p[0] = taub_p[0]+1
    #     #     if (taub[1] < 1.0e-6):
    #     #         taub_p[1] = taub_p[1]+1
    #     #     # if (taua[2] < 1.0e-6):
    #     #     #     taub[2] = taub[2]+1
    #     #     delta2 = taua@self.crystal.avec - (taub@self.crystal.avec + r)
    #     #     rmin2 = self.RMin2(delta2)

    #     #     if (rmin1 < rmin2):
    #     #         rmin = rmin1
    #     #     else:
    #     #         rmin = rmin2

    #     #     return rmin
    #     # else:
    #     #     rmin = 1.0e6
    #     #     rtemp = 0
    #     #     # temp = [0,1]
    #     #     # for iy in temp:
    #     #     #     for ix in temp:
    #     #     # taub_p = taub.copy()
    #     #     if (taub[0] < 1.0e-6):
    #     #         taub_p[0] += 1

    #     #     if (taub[1] < 1.0e-6):
    #     #         taub_p[1] += 1

    #     #     delta2 = taua@self.crystal.avec - (taub_p@self.crystal.avec + r)
    #     #     rmin2 = self.RMin2(delta2)

    #     #     if (rmin1 <= rmin2):
    #     #         rtemp = rmin1
    #     #     else:
    #     #         rtemp = rmin2
    #     #     if (rtemp < 1.0e-6):
    #     #         pass
    #     #     if (rtemp < rmin):
    #     #         rmin = rtemp
    #     #     # rmin = rmin1
    #     #     return rmin
