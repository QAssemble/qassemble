"""
"""

import DiagE
import itertools
import sys

# import re
import json
import os

# import shutil
# from collections import OrderedDict
# import string
# from typing import Any
import subprocess
import matplotlib.pyplot as plt

# import matplotlib.font_manager as fm
# import matplotlib as mpl
import numpy as np
import scipy.linalg
import scipy.optimize

# from scipy.fftpack import ifftn, fftn
from sympy.physics.wigner import gaunt  # , wigner_3j
from pymatgen.core import Lattice, Structure

# from pymatgen.transformations.standard_transformations import SupercellTransformation

diage_path = os.environ.get("DIAGE", "")
path = diage_path + "/modules"
sys.path.append(path)


class Crystal(object):  # chemical potential object, num of electron
    def __init__(
        self,
        latt: list,
        basisposition: dict,
        ns: int,
        soc: bool,
        rkgrid: list,
        orboption: list,
        N: float,
        supercell: list = [1, 1, 1],
        impdict: dict = None,
    ):
        latt = np.array(latt, dtype=float)
        # basisposition = np.array(basisposition,dtype=float)
        # tempmat = np.zeros((basisposition.shape[0],basisposition.shape[1]),dtype=float)
        # for jj in range(basisposition.shape[1]):
        #     for ii in range(basisposition.shape[0]):
        #         if 0<=basisposition[ii,jj]<=1:
        #             tempmat[ii,jj] = basisposition[ii,jj]
        #         if basisposition[ii,jj] < 0 :
        #             tempmat[ii,jj] = 1 + basisposition[ii,jj]
        #         if basisposition[ii,jj] > 1 :
        #             tempmat[ii,jj] = basisposition[ii,jj] - 1
        self.avec = latt
        a = latt[0]
        b = latt[1]
        c = latt[2]
        alpha = np.degrees(
            np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        )
        beta = np.degrees(
            np.arccos(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c)))
        )
        gamma = np.degrees(
            np.arccos(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c)))
        )
        if basisposition["CorF"] == "C":
            pos = np.array(basisposition["pos"])
            lat = Lattice.from_parameters(
                np.linalg.norm(a),
                np.linalg.norm(b),
                np.linalg.norm(c),
                alpha,
                beta,
                gamma,
            )
            structure = Structure(lat, ["X"] * len(pos), pos, coords_are_cartesian=True)
            structurebasisc = []
            structurebasisf = []
            for site in structure.sites:
                structurebasisc.append(site.coords.tolist())
                structurebasisf.append(site.frac_coords.tolist())
            print(structure)
        if basisposition["CorF"] == "F":
            pos = np.array(basisposition["pos"])
            lat = Lattice.from_parameters(
                np.linalg.norm(a),
                np.linalg.norm(b),
                np.linalg.norm(c),
                alpha,
                beta,
                gamma,
            )
            structure = Structure(lat, ["X"] * len(pos), pos)
            structurebasisc = []
            structurebasisf = []
            for site in structure.sites:
                structurebasisc.append(site.coords.tolist())
                structurebasisf.append(site.frac_coords.tolist())
            print(structure)
        structurebasisc = np.array(structurebasisc)
        structurebasisf = np.array(structurebasisf)
        # for jj in range(structurebasisf.shape[1]):
        #     for ii in range(structurebasisf.shape[0]):
        #         if (structurebasisf[ii,jj] >= 1):
        #             structurebasisf[ii,jj] -= 1
        #         if (structurebasisf[ii,jj] < 0):
        #             structurebasisf[ii,jj] += 1

        self.basisf = structurebasisf
        self.basisc = np.dot(self.basisf, self.avec)
        self.ns = ns
        self.soc = soc
        self.nume = N
        # self.basisf = tempmat
        # self.basisc = np.dot(self.basisf,self.avec)
        self.bvec = np.zeros((3, 3))
        self.vol = np.dot(np.cross(latt[:, 0], latt[:, 1]), latt[:, 2])
        self.bvec[:, 0] = 2 * np.pi * np.cross(latt[:, 1], latt[:, 2]) / self.vol
        self.bvec[:, 1] = 2 * np.pi * np.cross(latt[:, 2], latt[:, 0]) / self.vol
        self.bvec[:, 2] = 2 * np.pi * np.cross(latt[:, 0], latt[:, 1]) / self.vol

        self.kpath = None
        self.rkgrid = rkgrid
        nk = rkgrid[0] * rkgrid[1] * rkgrid[2]
        kpoint_temp = np.array(
            list(
                itertools.product(
                    np.linspace(0, 1, num=rkgrid[2], endpoint=False),
                    np.linspace(0, 1, num=rkgrid[1], endpoint=False),
                    np.linspace(0, 1, num=rkgrid[0], endpoint=False),
                )
            )
        )
        kpoint = np.fliplr(kpoint_temp)
        self.kpoint = kpoint

        self.nk = nk
        self.find = {}
        self.bind = {}
        self.b2f = []
        self.c2f = []
        self.c2b = []
        self.probspace = {}
        self.fimpdict = {}
        self.bimpdict = {}
        self.fprojector = None
        self.bprojector = None

        self.SetBasisIndex(orboption)
        if impdict != None:
            self.Projector(impdict)

    def Kpath(self, kpath: list, nk: int) -> np.ndarray:
        kpath = np.array(kpath, dtype=float)
        nnod = kpath.shape[0]
        kmat = np.linalg.inv(np.dot(self.avec, self.avec.T))
        knode = np.zeros(nnod, dtype=float)
        for n in range(1, nnod):
            dk = kpath[n] - kpath[n - 1]
            l = np.sqrt(np.dot(dk, np.dot(kmat, dk)))
            knode[n] = knode[n - 1] + l

        nk = nk * (nnod - 1)

        indnod = [0]
        for n in range(1, nnod - 1):
            frac = knode[n] / knode[-1]
            indnod.append(int(round(frac * (nk - 1))))
        indnod.append(nk - 1)

        kvec = np.zeros((nk, kpath.shape[1]))
        kvec[0] = kpath[0]
        cnt = 0
        for i in range(1, nnod):
            ni = indnod[i - 1]
            nf = indnod[i]
            ki = kpath[i - 1]
            kf = kpath[i]
            for j in range(ni, nf + 1):
                frac = float(j - ni) / float(nf - ni)
                kvec[j] = ki + frac * (kf - ki)

        self.kpath = kvec

    def SetBasisIndex(self, orboption: list) -> dict:
        """
        Modify orbital option for each atom basis
        """
        for option in orboption:
            find = []
            bind = []
            orblist = list(range(option[1]))

            for m1 in range(option[1]):
                find.append([option[0], m1])
            for m1, m2 in itertools.product(orblist, orblist):
                bind.append([option[0], [m1, m2]])

            forb = len(self.find)
            borb = len(self.bind)
            ii = 0
            for iorb in range(forb, forb + option[1]):
                self.find[iorb] = find[ii]
                ii += 1
            ii = 0
            for iorb in range(borb, borb + option[1] ** 2):
                self.bind[iorb] = bind[ii]
                ii += 1
                self.Boson2Fermion(iorb)
            self.Composite2Boson()
            self.Composite2Fermion()

    def FAtomOrb(self, key: int) -> list:
        """
        input : composite index for fermion
        output : atom and orbital index in fermion case

        e.g.
        0 -> [0,0]
        """
        return self.find[key]

    def FIndex(self, val: list) -> int:
        """
        input : atom and orbital index with list
        output : composite index for fermion

        e.g.
        [0,0] -> 0
        """

        for key, value in self.find.items():
            if value == val:
                return key

    def BAtomOrb(self, key: int) -> list:
        """
        input : composite index for fermion
        output : atom and orbital index in boson case

        e.g.
        0 -> [0,[0,0]]
        """
        return self.bind[key]

    def BIndex(self, val: list) -> int:
        """
        input : atom and orbital index with list
        output : composite index for boson

        e.g.
        [0,[0,0]] -> 0
        """
        for key, value in self.bind.items():
            if val == value:
                return key

    def Boson2Fermion(self, ind: int):
        """
        Mapping with boson index to fermion index
        """
        [a, [m1, m2]] = self.BAtomOrb(ind)
        iorbc1 = self.FIndex([a, m1])
        iorbc2 = self.FIndex([a, m2])
        self.b2f.append([iorbc1, iorbc2])

    def Composite2Fermion(self):
        """
        Mapping with fermion index to composite index
        """
        norbc = len(self.find)
        norb = norbc * norbc
        c2f = []

        for iorbc in range(norbc):
            for jorbc in range(norbc):
                nn1 = [iorbc, jorbc]
                iorb, nn1 = self.indexing(norb, 2, [norbc, norbc], 1, 0, nn1)
                c2f.append([iorbc, jorbc])
        self.c2f = c2f

    def Composite2Boson(self):
        norbc = len(self.find)
        ndim = norbc * norbc
        c2b = []

        for ind in range(ndim):
            nn1 = [0] * 2
            ind, [iorbc, jorbc] = self.indexing(ndim, 2, [norbc, norbc], 0, ind, nn1)
            [a, m1] = self.FAtomOrb(iorbc)
            [a_p, m2] = self.FAtomOrb(jorbc)
            if a == a_p:
                borb = self.BIndex([a, [m1, m2]])
                if borb is not None:
                    c2b.append([borb, ind])
        self.c2b = c2b

    def Composite2OrbSpin(self, mat: np.ndarray):
        norb = len(self.bind)
        ns = self.ns
        matout = np.zeros((norb, norb, ns, ns), dtype=complex, order="F")
        ndim = mat.shape[0]

        for ind1 in range(ndim):
            nn1 = [0] * 2
            ind1, [iorb, js] = self.indexing(ndim, 2, [norb, ns], 0, ind1, nn1)
            for ind2 in range(ndim):
                nn2 = [0] * 2
                ind2, [jorb, ks] = self.indexing(ndim, 2, [norb, ns], 0, ind2, nn2)
                matout[iorb, jorb, js, ks] = mat[ind1, ind2]

        return matout

    def OrbSpin2Composite(self, mat: np.ndarray):
        norb = mat.shape[0]
        ns = mat.shape[2]
        matout = np.zeros((norb * ns, norb * ns), dtype=complex, order="F")

        for js in range(ns):
            for iorb in range(norb):
                nn1 = [iorb, js]
                ind1, nn1 = self.indexing(norb * ns, 2, [norb, ns], 1, 0, nn1)
                for ks in range(ns):
                    for jorb in range(norb):
                        nn2 = [jorb, ks]
                        ind2, nn2 = self.indexing(norb * ns, 2, [norb, ns], 1, 0, nn2)
                        matout[ind1, ind2] = mat[iorb, jorb, js, ks]
        return matout

    def Quad2Double(self, mat: np.ndarray) -> np.ndarray:  # 4 index <-> 2 index
        norb = len(self.bind)

        matret = np.zeros((norb, norb), dtype=complex)

        for iorb, [iorbc, lorbc] in enumerate(self.b2f):
            for jorb, [jorbc, korbc] in enumerate(self.b2f):
                matret[iorb, jorb] = mat[iorbc, jorbc, korbc, lorbc]

        return matret

    def Double2Quad(self, mat: np.ndarray) -> np.ndarray:
        norbc = len(self.find)

        matret = np.zeros((norbc, norbc, norbc, norbc), dtype=complex, order="F")

        for iorb, [iorbc, lorbc] in enumerate(self.b2f):
            for jorb, [jorbc, korbc] in enumerate(self.b2f):
                matret[iorbc, jorbc, korbc, lorbc] = mat[iorb, jorb]

        return matret

    def Full2Quad(self, mat: np.ndarray) -> np.ndarray:
        norbc = len(self.find)

        matret = np.zeros((norbc, norbc, norbc, norbc), dtype=complex, order="F")

        for iorb, [iorbc, lorbc] in enumerate(self.c2f):
            for jorb, [jorbc, korbc] in enumerate(self.c2f):
                matret[iorbc, jorbc, korbc, lorbc] = mat[iorb, jorb]

        return matret

    def Quad2Full(self, mat: np.ndarray) -> np.ndarray:
        norb = len(self.find) ** 2

        matret = np.zeros((norb, norb))

        for iorb, [iorbc, lorbc] in enumerate(self.c2f):
            for jorb, [jorbc, korbc] in enumerate(self.c2f):
                matret[iorb, jorb] = mat[iorbc, jorbc, korbc, lorbc]

        return matret

    def Full2Double(self, mat: np.ndarray) -> np.ndarray:
        norb = len(self.bind)

        matret = np.zeros((norb, norb), dtype=complex, order="F")

        for iorb, ind1 in self.c2b:
            for jorb, ind2 in self.c2b:
                matret[iorb, jorb] = mat[ind1, ind2]

        return matret

    def Double2Full(self, mat: np.ndarray) -> np.ndarray:
        nind = len(self.find) ** 2

        matret = np.zeros((nind, nind), dtype=complex, order="F")

        for iorb, ind1 in self.c2b:
            for jorb, ind2 in self.c2b:
                matret[ind1, ind2] = mat[iorb, jorb]

        return matret  ## construct

    def Projector(self, impdict: dict):
        """
        Generate the projector for impurity quantity

        e.g.
        input : {"1" : [[0,0],[1,0]]}
        output : fprojector, bprojector
        """

        nspace = 0
        forbc = 0
        borbc = 0
        ns = self.ns
        probspace = {}
        fimpdict = {}
        bimpdict = {}

        for key, val in impdict.items():
            # probspace[key] = []
            for orblist in val:
                atom = 0
                for orb in orblist:
                    if orb == orblist[0]:
                        atom = orb[0]
                    if atom != orb[0]:
                        print("Different atoms are involved in the same space")
                        sys.exit()
            probspace[key] = [nspace + i for i in range(len(val))]
            nspace += len(val)

        self.probspace = probspace

        for key, val in impdict.items():
            fimpdict[key] = []
            for orblist in val:
                templist = []
                for orb in orblist:
                    find = self.FIndex(orb)
                    templist.append(find)
                fimpdict[key].append(templist)
        self.fimpdict = fimpdict
        for val in fimpdict.values():
            for orb in val:
                if len(orb) > forbc:
                    forbc = len(orb)
        for key, val in fimpdict.items():
            bimpdict[key] = []
            for orb in val:
                templist = []
                for iorb in orb:
                    for jorb in orb:
                        [a, m1] = self.FAtomOrb(iorb)
                        [b, m2] = self.FAtomOrb(jorb)
                        if a == b:
                            bind = self.b2f.index([iorb, jorb])
                            templist.append(bind)
                bimpdict[key].append(templist)
        for val in bimpdict.values():
            for orb in val:
                if len(orb) > borbc:
                    borbc = len(orb)
        self.bimpdict = bimpdict
        fprojector = np.zeros(
            (len(self.find), forbc, ns, nspace), dtype=float, order="F"
        )
        bprojector = np.zeros(
            (len(self.bind), borbc, ns, nspace), dtype=float, order="F"
        )

        for js in range(ns):
            for key, val in probspace.items():
                for ii, ispace in enumerate(val):
                    for ind in self.fimpdict[key][ii]:
                        fprojector[
                            ind, self.fimpdict[key][ii].index(ind), js, ispace
                        ] = 1.0

        for js in range(ns):
            for key, val in probspace.items():
                for ii, ispace in enumerate(val):
                    for ind in self.bimpdict[key][ii]:
                        bprojector[
                            ind, self.bimpdict[key][ii].index(ind), js, ispace
                        ] = 1.0

        self.fprojector = fprojector
        self.bprojector = bprojector

        return None

    def indexing(self, ntot, ndivision, divisionarray, flag, n1, n2):
        tmpsize = 1
        for size in divisionarray:
            tmpsize *= size

        if tmpsize != ntot:
            print("array_division wrong")
            return

        if flag == 1:
            n1 = n2[0]
            for ii in range(1, ndivision):
                tempcnt = 1
                for jj in range(ii):
                    tempcnt *= divisionarray[jj]
                n1 += (n2[ii]) * tempcnt
        else:
            n2_array = [0] * ndivision
            tempcnt = n1
            for ii in range(ndivision - 1):
                n2_array[ii] = (
                    tempcnt - ((tempcnt) // divisionarray[ii]) * divisionarray[ii]
                )
                tempcnt = (tempcnt - n2_array[ii]) // divisionarray[ii]
            n2_array[ndivision - 1] = tempcnt

            # Copy the values from the temporary array to the n2 output array
            for i in range(ndivision):
                n2[i] = n2_array[i]

        return n1, n2

    def FindPositions(self, array, value):
        positions = []
        for row_index, row in enumerate(array):
            for col_index, col_value in enumerate(row):
                if col_value == value:
                    positions.append([row_index, col_index])
        return positions


class FT_grid(object):
    def __init__(self, T: float = 300, size: int = 1000) -> object:
        self.T = T
        self.beta = 1 / (T * 8.6173303 * 10**-5)
        self.size = size
        self.omega = np.zeros((size), dtype=float, order="F")
        self.nu = np.zeros((size), dtype=float, order="F")
        self.tau = np.zeros((size), dtype=float, order="F")

        self.Omega()
        self.Tau()
        self.Nu()

    def Omega(self) -> np.ndarray:
        nomega = self.size
        for iomega in range(nomega):
            self.omega[iomega] = np.pi / self.beta * (2 * iomega + 1)

    def Tau(self) -> np.ndarray:
        ntau = self.size
        for itau in range(ntau):
            itheta = DiagE.common.ttind(itau, ntau)
            self.tau[itau] = (
                self.beta / 2.0 * (np.cos(np.pi * (itheta + 0.5) / ntau) + 1.0)
            )

    def Nu(self) -> np.ndarray:
        nnu = self.size
        for inu in range(nnu):
            self.nu[inu] = np.pi / self.beta * (2 * inu)


class FLatDyn(object):
    def __init__(self, crystal: Crystal, ft: FT_grid) -> object:
        self.crystal = crystal
        self.ft = ft
        self.mappingidx = None

    def Inverse(self, mat: np.ndarray) -> np.ndarray:
        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]
        nft = mat.shape[4]

        matinv = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    matinv[:, :, js, irk, ift] = np.linalg.inv(mat[:, :, js, irk, ift])

        return matinv

    def T2F(self, ftau: np.ndarray) -> np.ndarray:
        norb = ftau.shape[0]
        ns = ftau.shape[2]
        nk = ftau.shape[3]
        nfreq = len(self.ft.omega)
        ff = np.zeros((norb, norb, ns, nk, nfreq), dtype=complex, order="F")

        ff = DiagE.fourier.flatdyn_t2f(self.ft.tau, ftau, self.ft.omega)

        return ff

    def F2T(self, ff: np.ndarray, isgreen: int, highzero: int) -> np.ndarray:
        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]
        ntau = len(self.ft.tau)

        ftau = np.zeros((norb, norb, ns, nk, ntau), dtype=complex, order="F")

        moment, high = self.Moment(ff, isgreen, highzero)

        ftau = DiagE.fourier.flatdyn_f2t(self.ft.omega, ff, moment, self.ft.tau)

        return ftau

    def Moment(self, ff: np.ndarray, isgreen: int, highzero: int) -> np.ndarray:
        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]

        moment = np.zeros((norb, norb, ns, nk, 3), dtype=complex, order="F")
        high = np.zeros((norb, norb, ns, nk), dtype=complex, order="F")

        moment, high = DiagE.fourier.flatdyn_m(self.ft.omega, ff, isgreen, highzero)

        return moment, high

    def K2R(self, matk: np.ndarray) -> np.ndarray:
        rkvec = self.crystal.kpoint
        rkgrid = self.crystal.rkgrid

        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[3]
        nft = matk.shape[4]

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            [a, m1] = self.crystal.FAtomOrb(iorb)
                            [b, m2] = self.crystal.FAtomOrb(jorb)

                            delta = (
                                self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                            )
                            phase = np.exp(2.0j * np.pi * np.dot(rkvec[irk], delta))

                            matk[iorb, jorb, js, irk, ift] *= phase

        matr = DiagE.fourier.flatdyn_k2r(rkgrid, matk)

        return matr

    def R2K(self, matr: np.ndarray) -> np.ndarray:
        rkvec = self.crystal.kpoint
        rkgrid = self.crystal.rkgrid

        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[3]
        nft = matr.shape[4]

        matk = DiagE.fourier.flatdyn_r2k(rkgrid, matr)

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            [a, m1] = self.crystal.FAtomOrb(iorb)
                            [b, m2] = self.crystal.FAtomOrb(jorb)

                            delta = (
                                self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                            )

                            phase = np.exp(-2.0j * np.pi * np.dot(rkvec[irk], delta))
                            matk[iorb, jorb, js, irk, ift] *= phase
        return matk

    def R2mR(self) -> list:  # move to crystal
        rkvec = self.crystal.kpoint

        mrkvec = np.array(1.0 - rkvec, dtype=float)

        for ii in range(mrkvec.shape[0]):
            for jj in range(mrkvec.shape[1]):
                if mrkvec[ii, jj] == 1.0:
                    mrkvec[ii, jj] = 0.0

        mappingidx = []

        for ii in range(rkvec.shape[0]):
            for jj in range(mrkvec.shape[1]):
                if (
                    (abs(rkvec[ii, 0] - mrkvec[jj, 0]) <= 1.0e-6)
                    and (abs(rkvec[ii, 1] - mrkvec[jj, 1]) <= 1.0e-6)
                    and (abs(rkvec[ii, 2] - mrkvec[jj, 2]) <= 1.0e-6)
                ):
                    mappingidx.append([ii, jj])

        self.mappingidx = mappingidx
        return None

    def RT2mRmT(self, G: np.ndarray) -> np.ndarray:  # move to crystal
        self.R2mR()

        norb = G.shape[0]
        ns = G.shape[2]
        nr = G.shape[3]
        ntau = G.shape[4]

        GmRmT = np.zeros((norb, norb, ns, nr, ntau), dtype=complex, order="F")

        for itau in range(ntau):
            for rp in self.mappingidx:
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            GmRmT[iorb, jorb, js, rp[0], itau] = -G[
                                iorb, jorb, js, rp[1], ntau - itau - 1
                            ]

        return GmRmT

    def GaussianLinearBroad(self, x, y, w1, temperature, cutoff):
        norb = y.shape[0]
        ns = y.shape[2]
        nrk = y.shape[3]
        nft = y.shape[4]

        ynew = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

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
                            for iorb in range(norb):
                                for jorb in range(norb):
                                    ynew[iorb, jorb, js, irk, cnt] = sum(
                                        dist * y[iorb, jorb, js, irk]
                                    ) / sum(dist)
                    else:
                        ynew[..., irk, cnt] = y[..., irk, cnt]
                cnt += 1

        return ynew

    def Mixing(
        self, iter: int, mix: float, Fb: np.ndarray, Fm: np.ndarray
    ) -> np.ndarray:
        norb = Fb.shape[0]
        ns = Fb.shape[2]
        nrk = Fb.shape[3]
        nft = Fb.shape[4]

        Fnew = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

        if iter == 1:
            mix = 1.0
            Fm = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

        Fnew = mix * Fb + (1.0 - mix) * Fm

        return Fnew

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray):
        norb = mat1.shape[0]
        ns = mat1.shape[2]
        nrk = mat1.shape[3]
        nft = mat1.shape[4]

        matout = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

        matout = DiagE.dyson.flatdyn(mat1, mat2)

        return matout

    def Projection(self, matin: np.ndarray):
        ns = matin.shape[2]
        nft = matin.shape[4]
        norbc = self.crystal.fprojector.shape[1]
        nspace = self.crystal.fprojector.shape[3]

        matout = np.zeros((norbc, norbc, ns, nft, nspace), dtype=complex, order="F")

        for ispace in range(nspace):
            matout[..., ispace] = DiagE.projection.flatdyn(
                matin, self.crystal.fprojector[..., ispace]
            )

        return matout

    def ChemEmbedding(self, mu: float) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        chem = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    chem[:, :, js, irk, ift] = np.eye(norb, norb, dtype=complex) * mu

        return chem

    def StcEmbedding(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

        for ift in range(nft):
            matout[..., ift] = matin

        return matout

    # def Embedding(self, matin : np.ndarray): # -> local quantity

    #     pass


class GreenBare(FLatDyn):
    def __init__(self, crystal: Crystal, ft: FT_grid, niham) -> object:
        super().__init__(crystal, ft)
        self.niham = niham
        self.g0kt = None
        self.g0kf = None
        self.g0rt = None
        self.g0rf = None

        # self.GBareLatFreq() # Frequency Tau both generate
        # self.GBareLatTau()
        self.Cal()

    # def __init__(self,niham : object, ft : FT_grid):

    #     self.niham = niham
    #     self.ft = ft
    #     self.gnotkt = None
    #     self.gnotkf = None
    #     self.gnotrt = None
    #     self.gnotrf = None
    #     # super().__init__(crystal,ft)

    def Cal(self):  # freq, tau combine
        gnotkf = DiagE.bare.flatfreq(self.niham.hamtb, self.ft.omega)

        gnotrf = self.K2R(gnotkf)

        self.g0kf = gnotkf
        self.g0rf = gnotrf

        gnotkt = DiagE.bare.flattau(self.niham.hamtb, self.ft.tau)

        gnotrt = self.K2R(gnotkt)

        self.g0kt = gnotkt
        self.g0rt = gnotrt

        return None


class GreenInt(FLatDyn):
    def __init__(
        self,
        crystal: Crystal,
        ft: FT_grid,
        greenbare: GreenBare,
        sigmah: object = None,
        sigmaf: object = None,
        sigmagwc: object = None,
    ) -> object:
        super().__init__(crystal, ft)
        self.flatstc = FLatStc(crystal=crystal)
        self.gkf = None
        self.gkt = None
        self.grf = None
        self.grt = None
        self.gkfmu0 = None
        self.gbare = greenbare
        self.sigmah = sigmah
        self.sigmaf = sigmaf
        self.sigmac = sigmagwc
        self.occ = None
        self.occk = None
        self.occr = None
        self.mu = 0

        self.CalMu0()
        # self.Occ()

    def CalMu0(self):
        norb = self.gbare.g0kf.shape[0]
        ns = self.gbare.g0kf.shape[2]
        nrk = self.gbare.g0kf.shape[3]
        nft = self.gbare.g0kf.shape[4]
        print(self.gbare.g0kf[..., 0, :])
        if (self.sigmah == None) and (self.sigmaf == None) and (self.sigmac == None):
            # self.gkf = self.gbare.g0kf
            # self.gkt = self.gbare.g0kt
            # self.grf = self.gbare.g0rf
            # self.grt = self.gbare.g0rt
            self.gkfmu0 = self.gbare.g0kf

        if (self.sigmah is not None) and (self.sigmaf is not None) and (self.sigmac is None):
            tempmat = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")
            tempmat2 = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")
            tempmat = self.StcEmbedding(self.sigmah.hk)
            tempmat2 = self.StcEmbedding(self.sigmaf.fk)
            sigma = tempmat + tempmat2
            self.gkfmu0 = self.Dyson(self.gbare.g0kf, sigma)

            # self.gkf = tempmat
            # self.gkt = self.F2T(self.gkf,1,1)
            # self.grf = self.K2R(tempmat)
            # self.grt = self.F2T(self.grf,1,1)

        if (self.sigmah is not None) and (self.sigmaf is not None) and (self.sigmac is not None):
            tempmat = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")
            tempmat2 = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")
            tempmat = self.StcEmbedding(self.sigmah.hk)
            tempmat2 = self.StcEmbedding(self.sigmaf.fk)
            sigma = tempmat + tempmat2
            self.gkfmu0 = self.Dyson(self.gbare.g0kf, sigma)

            # self.gkf = tempmat
            # self.gkt = self.F2T(self.gkf,1,1)
            # self.grf = self.K2R(tempmat)
            # self.grt = self.F2T(self.grf,1,1)

        return None

    def Occ(self):
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        occk = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")
        occ = np.zeros((norb, norb, ns), dtype=complex, order="F")
        tempmat = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

        # for irk in range(nrk):
        #     for js in range(ns):
        #         for iorb in range(norb):
        #             for jorb in range(norb):
        #                 occk[iorb,jorb,js,irk] = -self.gkt[iorb,jorb,js,irk,-1]
        occk = -self.gkt[..., -1]
        for irk in range(nrk):
            occ += occk[..., irk]
        for ift in range(nft):
            tempmat[..., ift] = occk
        occ /= nrk
        self.occ = occ
        self.occk = occk
        # tempmat2 = self.K2R(tempmat)
        # self.occr = tempmat2[...,0]
        self.occr = self.flatstc.K2R(occk)

        return None

    def UpdateMu(self) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        gkfnew = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")
        chem = self.ChemEmbedding(self.mu)

        gkfnew = self.Dyson(self.gkfmu0, -chem)

        self.gkf = gkfnew
        self.gkt = self.F2T(gkfnew, 1, 1)
        self.grf = self.K2R(gkfnew)
        self.grt = self.K2R(self.gkt)

        self.Occ()

        return None

    def NumOfE(self, mu: float):
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        chem = self.ChemEmbedding(mu)
        gcalf = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")
        gcalt = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

        # gcalf = self.Dyson(self.gkf,-chem)
        gcalf = self.Dyson(self.gkfmu0, -chem)
        gcalt = self.F2T(gcalf, 1, 1)

        Ne = 0

        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    Ne += -np.real(gcalt[iorb, iorb, js, irk, -1])
        Ne /= nrk

        N = self.crystal.nume

        return N - Ne

    def SearchMu(self):
        mumin = -1000
        mumax = 1000
        nmin = self.NumOfE(mumin)
        nmax = self.NumOfE(mumax)
        if (nmin < 0) or (nmax > 0):
            print("Chemical potential is out of the bisection range")
            sys.exit()
        sol = scipy.optimize.brentq(self.NumOfE, mumin, mumax)
        self.mu = sol

        self.UpdateMu()
        return None


class SigmaGWC(FLatDyn):
    def __init__(
        self, crystal: Crystal, ft: FT_grid, green: GreenInt = None, wlat: object = None
    ) -> object:
        super().__init__(crystal, ft)
        self.rt = None
        self.rf = None
        self.kt = None
        self.kf = None
        if green == None:
            print("Error, green doesn't exist")
            sys.exit()

        if wlat == None:
            print("Error, wlat doesn't exist")
            sys.exit()
        self.green = green
        self.wlat = wlat
        self.Cal()

    def Cal(self) -> np.ndarray:  # SigmaGWC
        """
        Generate correlated self-energy
        input : Wc(R,t), G(R,t)

        return : crtau, crfreq, cktau, ckfreq
        """

        G = self.green.grt
        Wc = self.wlat.wcrt
        norbc = G.shape[0]
        ns = G.shape[2]
        nr = G.shape[3]
        ntau = G.shape[4]
        norb = Wc.shape[0]

        crtau = np.zeros((norbc, norbc, ns, nr, ntau), dtype=complex, order="F")

        for itau in range(ntau):
            for ir in range(nr):
                for js in range(ns):
                    for iorb in range(norb):
                        iorbc1, iorbc2 = self.crystal.b2f[iorb]
                        for jorb in range(norb):
                            jorbc1, jorbc2 = self.crystal.b2f[jorb]
                            crtau[iorbc1, jorbc1, js, ir, itau] += (
                                G[iorbc2, jorbc2, js, ir, itau]
                                * Wc[iorb, jorb, js, js, ir, itau]
                            )

        cktau = self.R2K(crtau)
        crfreq = self.T2F(crtau)
        ckfreq = self.T2F(cktau)

        self.rt = crtau
        self.kt = cktau
        self.rf = crfreq
        self.kf = ckfreq

        return None


class FLatStc(object):
    def __init__(self, crystal: Crystal):
        self.crystal = crystal

    def Inverse(self, mat: np.ndarray):
        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]

        matinv = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")

        for irk in range(nrk):
            for js in range(ns):
                matinv[:, :, js, irk] = np.linalg.inv(mat[:, :, js, irk])

        return matinv

    def K2R(self, matk: np.ndarray) -> np.ndarray:
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[3]

        tempmat = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")
        print(matk[:, :, 0, int(nrk / 2)])
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        [a, m1] = self.crystal.FAtomOrb(iorb)
                        [b, m2] = self.crystal.FAtomOrb(jorb)

                        delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]

                        phase = np.exp(2.0j * np.pi * np.dot(rkvec[irk], delta))

                        # matk[iorb,jorb,js,irk] *= phase
                        tempmat[iorb, jorb, js, irk] = matk[iorb, jorb, js, irk] * phase

        matk = tempmat
        print(matk[:, :, 0, int(nrk / 2)])
        matr = DiagE.fourier.flatstc_k2r(rkgrid, matk)

        return matr

    def R2K(self, matr: np.ndarray) -> np.ndarray:
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[3]

        matk = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")
        tempmat = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")
        matk = DiagE.fourier.flatstc_r2k(rkgrid, matr)

        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        [a, m1] = self.crystal.FAtomOrb(iorb)
                        [b, m2] = self.crystal.FAtomOrb(jorb)

                        delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                        phase = np.exp(-2.0j * np.pi * np.dot(rkvec[irk], delta))

                        tempmat[iorb, jorb, js, irk] = matk[iorb, jorb, js, irk] * phase

        matk = tempmat
        return matk

    def Band(self, energy: np.ndarray, fn: str = None):
        norb = energy.shape[0]
        ns = energy.shape[2]
        nk = energy.shape[3]

        energyplot = np.zeros((norb, ns, nk), dtype=float)

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    energyplot[iorb, js, ik] = energy[iorb, iorb, js, ik]
        if self.crystal.ns == 1:
            plt.plot(energyplot.T[:, 0, :])
            if fn == None:
                plt.show()
            else:
                plt.savefig(fn)
        else:
            up = energyplot[:, 0, :]
            down = energyplot[:, 1, :]
            plt.plot(up, "k-")
            plt.plot(down, "r-")
            if fn == None:
                plt.show()
            else:
                plt.savefig(fn)

        return None

    def Diagonalize(self, matk: np.ndarray, eigvec: bool = False):
        nk = matk.shape[3]
        norb = matk.shape[0]
        ns = matk.shape[2]

        energy = np.zeros((norb, norb, ns, nk), dtype=float)
        evec = np.zeros((norb, norb, ns, nk), dtype=complex)

        # if eigvec == False:
        #     for ik in range(nk):
        #         for js in range(ns):
        #             e = np.linalg.eigvalsh(matk[:,:,js,ik])
        #             energy[:,:,js,ik] = np.diag(e)
        #     return energy
        # else:
        #     for ik in range(nk):
        #         for js in range(ns):
        #             (e,v) = np.linalg.eigh(matk[:,:,js,ik])
        #             energy[:,:,js,ik] = np.diag(e)
        #             evec[:,:,js,ik] = v

        #     return energy, evec
        if eigvec == False:
            for ik in range(nk):
                for js in range(ns):
                    e, v, info = scipy.linalg.lapack.zheev(matk[:, :, js, ik])
                    energy[:, :, js, ik] = np.diag(e)
            return energy
        else:
            for ik in range(nk):
                for js in range(ns):
                    e, v, info = scipy.linalg.lapack.zheev(matk[:, :, js, ik])
                    energy[:, :, js, ik] = np.diag(e)
                    evec[:, :, js, ik] = v

            return energy, evec

    def Visualization(self, energy: np.ndarray, fn: str = None):
        if self.crystal.rkgrid[2] != 1:
            print("Energy surface for only 2D case")
            sys.exit()
        else:
            norb = energy.shape[0]
            ns = energy.shape[2]
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            kx = self.crystal.kpoint[:, 0].reshape(
                self.crystal.rkgrid[0], self.crystal.rkgrid[1], self.crystal.rkgrid[2]
            )
            ky = self.crystal.kpoint[:, 1].reshape(
                self.crystal.rkgrid[0], self.crystal.rkgrid[1], self.crystal.rkgrid[2]
            )
            energy = energy.T
            energy = energy.reshape(
                self.crystal.rkgrid[0],
                self.crystal.rkgrid[1],
                self.crystal.rkgrid[2],
                ns,
                norb,
                norb,
            )

            for js in range(ns):
                for iorb in range(norb):
                    ax.plot_surface(
                        kx[:, :, 0], ky[:, :, 0], energy[:, :, 0, js, iorb, iorb]
                    )

            ax.view_init(azim=-120, elev=0)
            ax.set_xlabel("kx")
            ax.set_ylabel("ky")
            ax.set_zlabel("Energy eV")
            if fn is None:
                plt.show()
            elif fn is not None:
                fig.savefig(fn)

        return None

    def Mixing(
        self, iter: int, mix: float, Fb: np.ndarray, Fm: np.ndarray
    ) -> np.ndarray:
        # norb = Fb.shape[0]
        # ns = Fb.shape[2]
        # nrk = Fb.shape[3]
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        Fnew = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")
        # print(Fnew.shape)
        if iter == 1:
            mix = 1.0
            Fm = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        Fnew[iorb, jorb, js, irk] = (
                            mix * Fb[iorb, jorb, js, irk]
                            + (1.0 - mix) * Fm[iorb, jorb, js, irk]
                        )

        return Fnew

    def ChemEmbedding(self, mu: float) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        chem = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")

        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    chem[iorb, iorb, js, irk] = mu

        return chem

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray):
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")

        matout = DiagE.dyson.flatstc(mat1, mat2)

        return matout

    def Projection(self, matin: np.ndarray):
        norb = len(self.crystal.fin)
        ns = self.crystal.ns
        norbc = self.crystal.fprojector.shape[1]
        nspace = self.crystal.fprojector.shape[3]

        matout = np.zeros((norbc, norbc, ns, nspace), dtype=complex, order="F")

        for ispace in range(nspace):
            matout[..., ispace] = DiagE.projection.flatstc(
                matin, self.crystal.fprojector[..., ispace]
            )

        return matout


class Hamiltonian(FLatStc):
    def __init__(
        self,
        crystal: Crystal,
        ham: np.ndarray,
        beta: float = None,
        sigmah=None,
        sigmaf=None,
        sigmac=None,
    ):
        super().__init__(crystal)

        self.occ = None
        self.occk = None
        self.occr = None
        self.ham = ham
        self.sigmah = sigmah
        self.sigmaf = sigmaf
        self.sigmac = sigmac
        self.beta = beta
        self.hk = None
        self.hkmu0 = None
        self.mu = 0
        # self.muold = mu
        self.CalMu0()

    def CalMu0(self) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        tempmat = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")

        tempmat = self.ham
        print(tempmat[:, :, 0, 0])

        if self.sigmah != None:
            tempmat += self.sigmah.hk
            print(self.sigmah.hk[:, :, 0, 0])
            print(tempmat[:, :, 0, 0])
        if self.sigmaf != None:
            tempmat += self.sigmaf.fk
            print(self.sigmaf.fk[:, :, 0, 0])
            print(tempmat[:, :, 0, 0])
        if self.sigmac != None:
            z = self.sigma.zfactor.z
            sigma = self.sigma.sigmastc.sigmastc
            # chem = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
            tempmat2 = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")
            tempmat3 = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")
            tempmat4 = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")
            tempmat4 = tempmat
            eigval, eigvec = self.Diagonalize(z, True)
            for ik in range(nrk):
                for js in range(ns):
                    for iorb in range(norb):
                        # chem[iorb,iorb,js,ik] = -self.mu
                        if 0 <= (eigval[iorb, iorb, js, ik]) <= 1:
                            continue
                        else:
                            print(
                                "Error : The z-factor was calculated incorrectly. Please rerun the code."
                            )
                            print(eigval[iorb, iorb, js, ik])
                            sys.exit()
                    tempmat2[:, :, js, ik] = np.dot(
                        np.dot(eigvec[:, :, js, ik], np.sqrt(eigval[:, :, js, ik])),
                        np.linalg.inv(eigvec[:, :, js, ik]),
                    )

            tempmat4 = tempmat4 + sigma

            for ik in range(nrk):
                for js in range(ns):
                    tempmat3[:, :, js, ik] = np.dot(
                        np.dot(tempmat2[:, :, js, ik], tempmat4[:, :, js, ik]),
                        tempmat2[:, :, js, ik],
                    )

            tempmat = tempmat3
            del tempmat2, tempmat3, tempmat4

        self.hkmu0 = tempmat
        del tempmat
        return None

    def NumOfE(self, mu: float) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        tempmat = np.zeros((norb, norb, ns, nk), dtype=complex, order="F")
        chem = self.ChemEmbedding(mu)
        # tempmat = self.hkmu0 - chem
        # np.save('hkmu0',tempmat)
        # tempmat2 = np.zeros((norb,norb,ns,nk),dtype=float)

        energy = self.Diagonalize(self.hkmu0)

        Ne = 0

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    Ne += 1 / (
                        1 + np.exp((energy[iorb, iorb, js, ik] - mu) * self.beta)
                    )

        Ne /= nk
        N = self.crystal.nume

        return N - Ne

    def SearchMu(self):
        energy = self.Diagonalize(self.hkmu0)
        mumin = energy[0, 0].min() - 1000
        mumax = energy[1, 1].max() + 1000

        nmin = self.NumOfE(mumin)
        nmax = self.NumOfE(mumax)
        if (nmin < 0) or (nmax > 0):
            print("Chemical potential is out of the bisection range")
            sys.exit()
        sol = scipy.optimize.brentq(self.NumOfE, mumin, mumax)
        # try:
        #     sol = scipy.optimize.brentq(self.NumOfE,mumin,mumax)
        # except:
        #     sol = scipy.optimize.newton(self.NumOfE,0,tol=10**(-10))
        self.mu = sol
        print(self.mu)
        self.UpdateMu()
        return None

    def Occ(self) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        # energy = self.Diagonalize(self.hk)

        occk = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")
        occ = np.zeros((norb, norb, ns), dtype=complex, order="F")
        tempmat = np.zeros((norb, norb), dtype=float, order="F")

        energy, eigvec = self.Diagonalize(self.hk, True)
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    tempmat[iorb, iorb] = 1 / (
                        np.exp(energy[iorb, iorb, js, irk] * self.beta) + 1
                    )
                # occk[:,:,js,irk] = np.dot(eigvec[:,:,js,irk],np.dot(tempmat,np.linalg.inv(eigvec[:,:,js,irk])))
                occk[:, :, js, irk] = np.dot(
                    eigvec[:, :, js, irk],
                    np.dot(tempmat, scipy.linalg.inv(eigvec[:, :, js, irk])),
                )

            occ += occk[..., irk]

        occ /= nrk

        self.occ = occ
        self.occk = occk
        self.occr = self.K2R(occk)

        return None

    def UpdateMu(self) -> np.ndarray:
        chem = self.ChemEmbedding(self.mu)

        ham = self.hkmu0 - chem

        self.hk = ham

        self.Occ()

        return None


class NIHamiltonian(FLatStc):
    def __init__(self, crystal: Crystal, hoppinglist: list, onsitelist: list):
        super().__init__(crystal)
        self.hoppinglist = hoppinglist
        self.onsitelist = onsitelist
        self.hamtb = None
        # self.Hopping()
        # self.Onsite()

        self.Cal()

    def Cal(self):  # GenHam
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        kvec = self.crystal.kpoint

        hamtb = np.zeros((norb, norb, ns, nk), dtype=complex, order="F")
        tempmat = np.zeros(
            (
                norb,
                norb,
                ns,
                self.crystal.rkgrid[0],
                self.crystal.rkgrid[1],
                self.crystal.rkgrid[2],
            ),
            dtype=complex,
            order="F",
        )

        # for ik in range(nk):
        #     for js in range(ns):
        #         for iorb in range(norb):
        #             hamtb[iorb,iorb,js,ik] = self.onsitelist[iorb]
        # for iorb in range(norb):
        #     tempmat[iorb,iorb] = self.onsitelist[iorb]

        # for ik in range(nk):
        #     for js in range(ns):
        #         for hopp in self.hoppinglist:
        #             t = hopp[0]
        #             iorb = hopp[1]
        #             jorb = hopp[2]
        #             R = hopp[3]

        #             [a,m1] = self.crystal.FAtomOrb(iorb)
        #             [b,m2] = self.crystal.FAtomOrb(jorb)

        #             rvec = self.crystal.basisf[a,:]-self.crystal.basisf[b,:] + R
        #             phase = np.exp(-2.0j*np.pi*np.dot(kvec[ik],rvec))
        #             hamtb[iorb,jorb,js,ik] += t*phase
        #             hamtb[jorb,iorb,js,ik] += t*np.conjugate(phase)
        for js in range(ns):
            for hopp in self.hoppinglist:
                tij = hopp[0]
                iorb = hopp[1]
                jorb = hopp[2]
                R = hopp[3]

                tempmat[iorb, jorb, js, R[0], R[1], R[2]] += tij
                tempmat[
                    jorb,
                    iorb,
                    js,
                    (self.crystal.rkgrid[0] - R[0] - 1) % self.crystal.rkgrid[0],
                    (self.crystal.rkgrid[1] - R[1] - 1) % self.crystal.rkgrid[1],
                    (self.crystal.rkgrid[2] - R[2] - 1) % self.crystal.rkgrid[2],
                ] += tij

        for iorb in range(norb):
            tempmat[iorb, iorb, 0, 0, 0, 0] = self.onsitelist[iorb]
        tempmat = tempmat.reshape(norb, norb, ns, nk)
        hamtb = self.R2K(tempmat)

        self.hamtb = hamtb

        return None

    # def Hopping(self):
    #     pass

    # def Onsite(self):
    #     pass


class QPHamiltonian(FLatStc):
    def __init__(
        self, crystal: Crystal, niham: NIHamiltonian, sigma: object, mu: float
    ):  # object input
        super().__init__(crystal)
        self.niham = niham
        self.sigma = sigma
        self.mu = mu
        self.hamqp = None

        self.Cal()

    def Cal(self):
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)

        hamtb = self.niham.hamtb
        z = self.sigma.zfactor.z
        sigma = self.sigma.sigmastc.sigmastc
        hamqp = np.zeros((norb, norb, ns, nk), dtype=complex, order="F")
        # chem = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
        tempmat = np.zeros((norb, norb, ns, nk), dtype=complex, order="F")
        tempmat2 = np.zeros((norb, norb, ns, nk), dtype=complex, order="F")

        eigval, eigvec = self.Diagonalize(z, True)
        chem = self.ChemEmbedding(-self.mu)
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    # chem[iorb,iorb,js,ik] = -self.mu
                    if 0 <= (eigval[iorb, iorb, js, ik]) <= 1:
                        continue
                    else:
                        print(
                            "Error : The z-factor was calculated incorrectly. Please rerun the code."
                        )
                        print(eigval[iorb, iorb, js, ik])
                        sys.exit()
                tempmat[:, :, js, ik] = np.dot(
                    np.dot(eigvec[:, :, js, ik], np.sqrt(eigval[:, :, js, ik])),
                    np.linalg.inv(eigvec[:, :, js, ik]),
                )

        tempmat2 = hamtb + sigma + chem

        for ik in range(nk):
            for js in range(ns):
                hamqp[:, :, js, ik] = np.dot(
                    np.dot(tempmat[:, :, js, ik], tempmat2[:, :, js, ik]),
                    tempmat[:, :, js, ik],
                )

        self.hamqp = hamqp

        return None


# class HFHamiltonian(FLatStc):

#     def __init__(self, crystal: Crystal, niham : NIHamiltonian, sigma : object,mu : float):
#         super().__init__(crystal)

#         self.niham = niham
#         self.sigma = sigma
#         self.mu = mu
#         self.hamhfk = None
#         self.hamhfr = None
#         self.Cal()

#     def Cal(self):
#         # print(self.mu)
#         hamtb = self.niham.hamtb
#         hk = self.sigma.sigmahartree.hk
#         fk = self.sigma.sigmafock.fk

#         norb = hamtb.shape[0]
#         ns = hamtb.shape[2]
#         nk = hamtb.shape[3]

#         tempmat = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
#         chem = np.zeros((norb,norb,ns,nk),dtype=complex,order='F') # embedding method in FLatDyn to FLocStc

#         for ik in range(nk):
#             for js in range(ns):
#                 for iorb in range(norb):
#                     chem[iorb,iorb,js,ik] -= self.mu


#         tempmat = hamtb+hk+fk+chem
#         self.hamhfk = tempmat
#         self.hamhfr = self.K2R(tempmat)

#         return None


class SigmaHartree(FLatStc):
    def __init__(self, crystal: Crystal, occ=None, vbare=None):  # green -> occ
        super().__init__(crystal)
        self.hr = None
        self.hk = None
        self.hdyn = None
        self.vbare = vbare
        self.occ = occ

        self.Cal()
        # self.MakeDyn()

    def Cal(self):
        # vbare = self.vbare.k
        occ = self.occ
        vk = self.vbare.Double2Quad(self.vbare.k)
        norbc = len(self.crystal.find)  # occk.shape[0]
        ns = self.crystal.ns  # occk.shape[2]
        nk = len(self.crystal.kpoint)  # occk.shape[3]
        norb = len(self.crystal.bind)  # vbare.shape[0]

        # tempmat = np.zeros((norb*ns,norb*ns,nk),dtype=complex,order='F')
        h = np.zeros((norbc, norbc, ns, nk), dtype=complex, order="F")

        if self.crystal.ns != 1:
            #     for ik in range(nk):
            #         tempmat[...,ik] = self.crystal.OrbSpin2Composite(vbare[...,ik])

            # for ik in range(nk):
            #     for ind1 in range(norb*ns):
            #         nn1 = [0]*2
            #         ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
            #         [iorbc1,iorbc2] = self.crystal.b2f[iorb]

            #         for ind2 in range(norb*ns):
            #             nn2 = [0]*2
            #             ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
            #             [iorbc3,iorbc4] = self.crystal.b2f[jorb]
            #             h[iorbc1,iorbc2,js,ik] += tempmat[ind1,ind2,0]*occ[iorbc4,iorbc3,ks]
            # for jk in range(nk):
            #     h[iorbc1,iorbc2,js,ik] += tempmat[ind1,ind2,0]*occ[iorbc4,iorbc3,ks,jk]/nk
            for ik in range(nk):
                for ind1 in range(norb * ns):
                    nn1 = [0] * 2
                    ind1, [iorb, js] = self.crystal.indexing(
                        norb * ns, 2, [norb, ns], 0, ind1, nn1
                    )
                    [a, [m1, m2]] = self.crystal.BAtomOrb(iorb)
                    iorbc1 = self.crystal.FIndex([a, m1])
                    iorbc2 = self.crystal.FIndex([a, m2])
                    for ind2 in range(norb * ns):
                        nn2 = [0] * 2
                        ind2, [jorb, ks] = self.crystal.indexing(
                            norb * ns, 2, [norb, ns], 0, ind2, nn2
                        )
                        [b, [m3, m4]] = self.crystal.BAtomOrb(jorb)
                        iorbc3 = self.crystal.FIndex([b, m3])
                        iorbc4 = self.crystal.FIndex([b, m4])
                        h[iorbc1, iorbc2, js, ik] += (
                            vk[iorbc1, iorbc3, iorbc4, iorbc2, js, ks, 0]
                            * occ[iorbc4, iorbc3, ks]
                        )
                        # h[iorbc1,iorbc2,js,ik] += vk[iorbc1,iorbc3,iorbc4,iorbc2,js,ks,0]*occ[iorbc3,iorbc4,ks]

        else:
            if self.crystal.soc == True:
                C = 1
                # for ik in range(nk):
                #     for iorb in range(norb):
                #         iorbc1,iorbc2 = self.crystal.b2f[iorb]
                #         for jorb in range(norb):
                #             iorbc3, iorbc4 = self.crystal.b2f[jorb]
                #             # gtemp = np.zeros((norbc,norbc,1),dtype=complex)
                #             # for jk in range(nk):
                #             #     gtemp[iorbc4,iorbc3,0] += g0kt[iorbc4,iorbc3,0,0,-1]
                #             h[iorbc1,iorbc2,0,ik] += vbare[iorb,jorb,0,0,0]*occ[iorbc4,iorbc3,0]*C #1/nk*gtemp[iorbc4,iorbc3,0]*C
                for ik in range(nk):
                    for ind1 in range(norb * ns):
                        nn1 = [0] * 2
                        ind1, [iorb, js] = self.crystal.indexing(
                            norb * ns, 2, [norb, ns], 0, ind1, nn1
                        )
                        [a, [m1, m2]] = self.crystal.BAtomOrb(iorb)
                        iorbc1 = self.crystal.FIndex([a, m1])
                        iorbc2 = self.crystal.FIndex([a, m2])
                        for ind2 in range(norb * ns):
                            nn2 = [0] * 2
                            ind2, [jorb, ks] = self.crystal.indexing(
                                norb * ns, 2, [norb, ns], 0, ind2, nn2
                            )
                            [b, [m3, m4]] = self.crystal.BAtomOrb(jorb)
                            iorbc3 = self.crystal.FIndex([b, m3])
                            iorbc4 = self.crystal.FIndex([b, m4])
                            h[iorbc1, iorbc2, js, ik] += (
                                vk[iorbc1, iorbc3, iorbc4, iorbc2, js, ks, 0]
                                * occ[iorbc4, iorbc3, ks]
                                * C
                            )

            else:
                C = 2
                # for ik in range(nk):
                #     for iorb in range(norb):
                #         iorbc1,iorbc2 = self.crystal.b2f[iorb]
                #         for jorb in range(norb):
                #             iorbc3, iorbc4 = self.crystal.b2f[jorb]
                #             h[iorbc1,iorbc2,0,ik] += vbare[iorb,jorb,0,0,0]*occ[iorbc4,iorbc3,0]*C
                #             # for jk in range(nk):
                #             #     h[iorbc1,iorbc2,0,ik] += vbare[iorb,jorb,0,0,0]*occ[iorbc4,iorbc3,0,jk]/nk*C
                for ik in range(nk):
                    for ind1 in range(norb * ns):
                        nn1 = [0] * 2
                        ind1, [iorb, js] = self.crystal.indexing(
                            norb * ns, 2, [norb, ns], 0, ind1, nn1
                        )
                        [a, [m1, m2]] = self.crystal.BAtomOrb(iorb)
                        iorbc1 = self.crystal.FIndex([a, m1])
                        iorbc2 = self.crystal.FIndex([a, m2])
                        for ind2 in range(norb * ns):
                            nn2 = [0] * 2
                            ind2, [jorb, ks] = self.crystal.indexing(
                                norb * ns, 2, [norb, ns], 0, ind2, nn2
                            )
                            [b, [m3, m4]] = self.crystal.BAtomOrb(jorb)
                            iorbc3 = self.crystal.FIndex([b, m3])
                            iorbc4 = self.crystal.FIndex([b, m4])
                            h[iorbc1, iorbc2, js, ik] += (
                                vk[iorbc1, iorbc3, iorbc4, iorbc2, js, ks, 0]
                                * occ[iorbc4, iorbc3, ks]
                                * C
                            )
                            # h[iorbc1,iorbc2,js,ik] += vk[iorbc1,iorbc3,iorbc4,iorbc2,js,ks,0]*occ[iorbc3,iorbc4,ks]*C

        self.hk = h
        self.hr = self.K2R(h)

        return None

    def MakeDyn(self):
        norb = self.green.gkf.shape[0]
        ns = self.green.gkf.shape[2]
        nrk = self.green.gkf.shape[3]
        nft = self.green.gkf.shape[4]

        tempmat = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

        for ift in range(nft):
            tempmat[..., ift] = self.hk
        self.hdyn = tempmat

        return


class SigmaFock(FLatStc):
    def __init__(
        self, crystal: Crystal, occr=None, vbare: object = None
    ):  # green -> occ
        super().__init__(crystal)
        self.fr = None
        self.fk = None
        self.fdyn = None
        # self.green = green
        self.occr = occr
        self.vbare = vbare

        self.Cal()
        # self.MakeDyn()

    def Cal(self):
        # g0rt = self.green.glatrt
        occr = self.occr
        vr = self.vbare.Double2Quad(self.vbare.r)

        norbc = occr.shape[0]
        ns = occr.shape[2]
        nr = occr.shape[3]
        norb = vr.shape[0]

        fr = np.zeros((norbc, norbc, ns, nr), dtype=complex, order="F")

        # for ir in range(nr):
        #     for js in range(ns):
        #         for iorb in range(norb):
        #             [iorbc1,iorbc4] = self.crystal.b2f[iorb]
        #             for jorb in range(norb):
        #                 [iorbc2,iorbc3] = self.crystal.b2f[jorb]
        #                 fr[iorbc1,iorbc3,js,ir] = -occr[iorbc4,iorbc2,js,ir]*vr[iorb,jorb,js,js,ir]
        for ir in range(nr):
            for ind1 in range(norb * ns):
                nn1 = [0] * 2
                ind1, [iorb, js] = self.crystal.indexing(
                    norb * ns, 2, [norb, ns], 0, ind1, nn1
                )
                [a, [m1, m4]] = self.crystal.BAtomOrb(iorb)
                iorbc1 = self.crystal.FIndex([a, m1])
                iorbc4 = self.crystal.FIndex([a, m4])
                for ind2 in range(norb * ns):
                    nn2 = [0] * 2
                    ind2, [jorb, ks] = self.crystal.indexing(
                        norb * ns, 2, [norb, ns], 0, ind2, nn2
                    )
                    [b, [m3, m2]] = self.crystal.BAtomOrb(jorb)
                    iorbc3 = self.crystal.FIndex([b, m3])
                    iorbc2 = self.crystal.FIndex([b, m2])
                    if js == ks:
                        fr[iorbc1, iorbc2, js, ir] += (
                            -occr[iorbc4, iorbc3, js, ir]
                            * vr[iorbc1, iorbc3, iorbc2, iorbc4, js, ks, ir]
                        )
                        # fr[iorbc1,iorbc2,js,ir] += -occr[iorbc3,iorbc4,js,ir]*vr[iorbc1,iorbc3,iorbc2,iorbc4,js,ks,ir]

        fk = self.R2K(fr)

        self.fr = fr
        self.fk = fk

        return None

    def MakeDyn(self):  # move to LatStc
        norb = self.green.gkf.shape[0]
        ns = self.green.gkf.shape[2]
        nrk = self.green.gkf.shape[3]
        nft = self.green.gkf.shape[4]

        tempmat = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

        for ift in range(nft):
            tempmat[..., ift] = self.fk

        self.fdyn = tempmat

        return None


class SigmaStc(FLatStc):
    def __init__(self, crystal: Crystal, sigma: object):
        super().__init__(crystal)
        self.sigma = sigma

        self.sigmastc = None
        self.Cal()

    def Cal(self):
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nfreq = self.sigma.cmkfreq.shape[4]

        sigmagwc = self.sigma.cmkfreq
        sigmastc = np.zeros((norb, norb, ns, nk), dtype=complex, order="F")
        tempmat = np.zeros((norb, norb, ns, nk, nfreq), dtype=complex, order="F")

        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    tempmat[:, :, js, ik, ifreq] = np.transpose(
                        np.conjugate(sigmagwc[:, :, js, ik, ifreq])
                    )

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        sigmastc[iorb, jorb, js, ik] = (
                            sigmagwc[iorb, jorb, js, ik, 0]
                            + tempmat[iorb, jorb, js, ik, 0]
                        ) / 2

        self.sigmastc = sigmastc

        return None


class ZFactor(FLatStc):
    def __init__(self, crystal: Crystal, sigma: object):
        super().__init__(crystal)
        self.sigma = sigma
        self.z = None
        self.Cal()

    def Cal(self):
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nfreq = self.sigma.cmkfreq.shape[4]
        beta = self.sigma.sigmagwc.ft.beta

        sigma = self.sigma.cmkfreq
        # sigma = self.sigma.sigmagwc.gwckf
        z = np.zeros((norb, norb, ns, nk), dtype=complex, order="F")
        identity = np.zeros((norb, norb, ns, nk, nfreq), dtype=complex, order="F")
        tempmat = np.zeros((norb, norb, ns, nk, nfreq), dtype=complex, order="F")
        tempmat2 = np.zeros((norb, norb, ns, nk), dtype=complex, order="F")

        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    identity[:, :, js, ik, ifreq] = np.eye(
                        norb, norb, dtype=complex, order="F"
                    )
                    tempmat[:, :, js, ik, ifreq] = np.transpose(
                        np.conjugate(sigma[:, :, js, ik, ifreq])
                    )

        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            tempmat2[iorb, jorb, js, ik] = identity[
                                iorb, jorb, js, ik, ifreq
                            ] - beta * (
                                sigma[iorb, jorb, js, ik, ifreq]
                                - tempmat[iorb, jorb, js, ik, ifreq]
                            ) / (
                                2 * np.pi
                            )
        print(tempmat2.shape)
        for ik in range(nk):
            for js in range(ns):
                z[:, :, js, ik] = np.linalg.inv(tempmat2[:, :, js, ik])

        self.z = z

        return None


class Occ(FLatStc):
    def __init__(self, crystal: Crystal, green: object):
        super().__init__(crystal)
        self.green = green
        self.occ = None
        self.occk = None
        self.occr = None
        self.Cal()

    def Cal(self):
        gkt = self.green.glatkt
        norb = gkt.shape[0]
        ns = gkt.shape[2]
        nk = gkt.shape[3]

        occ = np.zeros((norb, norb, ns, nk), dtype=complex)
        tempmat = np.zeros((norb, norb, ns), dtype=complex)
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        occ[iorb, jorb, js, ik] += -gkt[iorb, jorb, js, ik, -1]

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        tempmat[iorb, jorb, js] += occ[iorb, jorb, js, ik]
        tempmat /= nk
        self.occ = tempmat
        self.occk = occ
        occr = self.K2R(occ)
        self.occr = occr

        return None


class FLocDyn(object):
    def __init__(self, crystal: Crystal, ft: FT_grid):
        self.crystal = crystal
        self.ft = ft

    def Inverse(self, mat: np.ndarray):
        norb = mat.shape[0]
        ns = mat.shape[2]
        nft = mat.shape[3]

        matinv = np.zeros((norb, norb, ns, nft), dtype=complex, order="F")

        for ift in range(nft):
            for js in range(ns):
                matinv[:, :, js, ift] = np.linalg.inv(mat[:, :, js, ift])

        return matinv

    def Moment(self, ff: np.ndarray, isgreen: int, highzero: int) -> np.ndarray:
        norb = ff.shape[0]
        ns = ff.shape[2]

        moment = np.zeros((norb, norb, ns, 3), dtype=complex, order="F")
        high = np.zeros((norb, norb, ns), dtype=complex, order="F")

        moment, high = DiagE.fourier.flocdyn_m(self.ft.omega, ff, isgreen, highzero)

        return moment, high

    def F2T(self, ff: np.ndarray, isgreen: int, highzero: int) -> np.ndarray:
        norb = ff.shape[0]
        ns = ff.shape[2]
        ntau = len(self.ft.tau)

        ftau = np.zeros((norb, norb, ns, ntau), dtype=complex, order="F")

        moment, high = self.Moment(ff, isgreen, highzero)

        ftau = DiagE.fourier.flocdyn_f2t(self.ft.omega, ff, moment, self.ft.tau)

        return ftau

    def T2F(self, ftau: np.ndarray) -> np.ndarray:
        norb = ftau.shape[0]
        ns = ftau.shape[2]
        nfreq = len(self.ft.omega)

        ff = np.zeros((norb, norb, ns, nfreq), dtype=complex, order="F")

        ff = DiagE.fourier.flocdyn_t2f(self.ft.tau, ftau, self.ft.omega)

        return ff

    def GaussianLinearBroad(self, x, y, w1, temperature, cutoff):
        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]

        ynew = np.zeros((norb, norb, ns, nft), dtype=complex, order="F")
        w0 = (1.0 - 3.0 * w1) * np.pi * temperature
        widtharray = w0 + w1 * x
        cnt = 0

        for x0 in x:
            if x0 > cutoff + (w0 + w1 * cutoff) * 3.0:
                ynew[..., cnt] = y[..., cnt]
            else:
                if (x0 > 3 * widtharray[cnt]) and ((x[-1] - x0) > 3 * widtharray[cnt]):
                    dist = (
                        1.0
                        / np.sqrt(2 * np.pi)
                        / widtharray[cnt]
                        * np.exp(-((x - x0) ** 2) / 2.0 / widtharray[cnt] ** 2)
                    )
                    for js in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                ynew[iorb, jorb, js, cnt] = sum(
                                    dist * y[iorb, jorb, js]
                                ) / sum(dist)
                else:
                    ynew[..., cnt] = y[..., cnt]
            cnt += 1

        return ynew

    def Mixing(self, iter: int, mix: float, Fb: np.ndarray, Fold: np.ndarray):
        norb = Fb.shape[0]
        ns = Fb.shape[2]
        nft = Fb.shape[3]

        Fnew = np.zeros((norb, norb, ns, nft), dtype=complex, order="F")

        if iter == 1:
            mix = 1.0
            Fold = np.zeros((norb, norb, ns, nft), dtype=complex, order="F")

        Fnew = mix * Fb + (1.0 - mix) * Fold

        return Fnew

    def Imp2Loc(self, matimp: np.ndarray) -> np.ndarray:
        norb = matimp.shape[0]
        ns = matimp.shape[2]
        nft = matimp.shape[3]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb, norb, ns, nft, nspace), dtype=complex, order="F")

        for key, val in self.crystal.probspace.items():
            iprob = int(key) - 1
            for ispace in val:
                matloc[..., ispace] = matimp[..., iprob]

        return matloc

    def Loc2Imp(self, matloc: np.ndarray) -> np.ndarray:
        nprob = len(self.crystal.probspace)
        norb = matloc.shape[0]
        ns = matloc.shape[2]
        nft = matloc.shape[3]

        matimp = np.zeros((norb, norb, ns, nft, nprob), dtype=complex, order="F")

        for key, val in self.crystal.probspace.items():
            iprob = int(key) - 1
            tempmat = np.zeros((norb, norb, ns), dtype=complex)
            for ispace in val:
                tempmat += matloc[..., ispace]
            tempmat /= len(val)
            matimp[..., iprob] = tempmat

        return matimp

    def Arr2Dict(self, equiv: np.ndarray, matin: np.ndarray) -> dict:
        ns = matin.shape[2]
        nind = np.amax(equiv)
        matdict = {}

        for ind in range(nind):
            matdict[ind + 1] = []
            pos = self.crystal.FindPositions(equiv, ind + 1)
            for js in range(ns):
                e = 0
                for ii, jj in pos:
                    e += matin[ii, jj, js]
                e /= len(pos)
                matdict[ind + 1].append(e.tolist())

        return matdict

    def Dict2Arr(self, equiv: np.ndarray, matdict: np.ndarray) -> np.ndarray:
        norb = len(equiv)
        ns = self.crystal.ns
        nfreq = len(matdict["1"])

        matout = np.zeros((norb, norb, ns, nfreq), dtype=complex, order="F")
        nind = np.amax(equiv)

        for js in range(ns):
            for ind in range(nind):
                pos = self.crystal.FindPositions(equiv, ind + 1)
                for ii, jj in pos:
                    matout[ii, jj, js] = matdict[str(ind + 1)]

        return matout

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray):
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nft = self.ft.size

        matout = np.zeros((norb, norb, ns, nft), dtype=complex, order="F")

        matout = DiagE.dyson.flocdyn(mat1, mat2)

        return matout

    def Embedding(self, matin: np.ndarray):
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size
        nspace = self.crystal.fprojector.shape[3]

        matout = np.zeros((norb, norb, ns, nrk, nft), dtype=complex, order="F")

        for ispace in range(nspace):
            matout += DiagE.embedding.flocdyn(
                nrk, matin[..., ispace], self.crystal.fprojector[..., ispace]
            )

        return matout


class GreenLoc(FLocDyn):
    def __init__(self, crystal: Crystal, ft: FT_grid, green: GreenInt):
        super().__init__(crystal, ft)
        self.green = green
        self.gf = None
        self.gt = None

        self.Cal()

    def Cal(self):  # projection
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nft = self.ft.size
        nspace = self.crystal.fprojector.shape[3]

        gf = np.zeros((norbc, norbc, ns, nft, nspace), dtype=complex)

        for ispace in range(nspace):
            gf[..., ispace] = DiagE.projection.flatdyn(
                self.green.gkf, self.crystal.fprojector[..., ispace]
            )

        self.gf = gf
        self.gt = self.F2T(gf, 1, 1)

        return None


class GreenImp(FLocDyn):  # read CTQMC output
    def __init__(self, crystal: Crystal, ft: FT_grid, ctqmc_dict):
        super().__init__(crystal, ft)
        self.Cal(ctqmc_dict)

    def Cal(self, ctqmc_dict):
        # read obs.json, see ClassDiagE.py for info on how.
        # Get a dict, then transform
        self.imp = super().Dict2Arr(ctqmc_dict)
        return None


class SigmaLoc(FLocDyn):
    def __init__(self, crystal: Crystal, ft: FT_grid, sigma: object):
        super().__init__(crystal, ft)

        self.sigma = sigma
        self.f = None
        self.Cal()

    def Cal(self):  # projection
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nft = self.ft.size
        nspace = self.crystal.fprojector.shape[3]

        sigmalocf = np.zeros((norbc, norbc, ns, nft, nspace), dtype=complex, order="F")

        for isapce in range(nspace):
            sigmalocf[..., isapce] = DiagE.projection.flatdyn(
                self.sigma, self.crystal.fprojector[..., isapce]
            )

        self.f = sigmalocf
        self.t = self.F2T(sigmalocf, 0, 1)

        return None


class SigmaImp(FLocDyn):  # read CTQMC output
    def __init__(self, crystal: Crystal, ft: FT_grid, ctqmc_dict):
        # read obs.json, see ClassDiagE.py for info on how.
        # Get a dict, then transform
        super().__init__(crystal, ft)
        self.Cal(ctqmc_dict)

    def Cal(self, ctqmc_dict):
        sigma_imp = super().Dict2Arr(ctqmc_dict)
        self.imp = sigma_imp
        return None


class SigmaLGWC(FLocDyn):
    def __init__(self, crystal: Crystal, ft: FT_grid):
        super().__init__(crystal, ft)
        pass


class Hybridisation(FLocDyn):
    def __init__(
        self,
        crystal: Crystal,
        ft: FT_grid,
        implev: np.ndarray,
        gimp: np.ndarray,
        sigmaimp: np.ndarray,
    ):
        super().__init__(crystal, ft)
        self.Cal(implev, gimp, sigmaimp)

    def Cal(self, E_imp: np.ndarray, G_imp: np.ndarray, Sigma: np.ndarray):
        norb = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nspace = self.crystal.fprojector.shape[3]
        nfreq = len(self.ft.omega)
        omega = self.ft.omega

        hyb = np.zeros((norb, norb, ns, nfreq, nspace), dtype=complex, order="F")
        G_imp_inv = self.Inverse(G_imp)
        Omega = np.zeros((norb, norb, ns, nfreq, nspace), dtype=complex, order="F")
        for ifreq in range(nfreq):
            for js in range(ns):
                for jspace in range(nspace):
                    Omega[:, :, js, ifreq, jspace] = (
                        np.eye(norb, norb) * 1j * omega[ifreq]
                    )

        for jspace in range(nspace):
            for ifreq in range(nfreq):
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            hyb[iorb, jorb, js, ifreq, jspace] = (
                                Omega[iorb, jorb, js, ifreq, jspace]
                                - E_imp[iorb, jorb, js, jspace]
                                - G_imp_inv[iorb, jorb, js, ifreq, jspace]
                                - Sigma[iorb, jorb, js, ifreq, jspace]
                            )

        print(f"hybridisation Sigma high frequncy limit : {Sigma[:,:,:,-1,:]}")
        print(f"hybridisation high frequency limit : {hyb[:,:,:,-1,:]}")
        print(
            -G_imp_inv[..., -1, :] + Omega[..., -1, :],
            -G_imp_inv[..., -1, :],
            Omega[..., -1, :],
        )
        print(E_imp)

        self.hyb = hyb
        return None


class FLocStc(object):
    def __init__(self, crystal: Crystal):
        self.crystal = crystal

    def Inverse(self, mat: np.ndarray):
        norb = mat.shape[0]
        ns = mat.shape[2]

        matinv = np.zeros((norb, norb, ns), dtype=complex, order="F")

        for js in range(ns):
            matinv[:, :, js] = np.linalg.inv(mat[:, :, js])

        return matinv

    def Mixing(
        self, iter: int, mix: float, Fb: np.ndarray, Fold: np.ndarray
    ) -> np.ndarray:
        norb = Fb.shape[0]
        ns = Fb.shape[2]

        Fnew = np.zeros((norb, norb, ns), dtype=complex, order="F")

        if iter == 1:
            mix = 1.0
            Fold = np.zeros((norb, norb, ns), dtype=complex, order="F")

        Fnew = mix * Fb + (1.0 - mix) * Fold

        return Fnew

    def Imp2Loc(self, matimp: np.ndarray) -> np.ndarray:
        norb = matimp.shape[0]
        ns = matimp.shape[2]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb, norb, ns, nspace), dtype=complex, order="F")

        for key, val in self.crystal.probspace.items():
            iprob = int(key) - 1
            for ispace in val:
                matloc[..., ispace] = matimp[..., iprob]

        return matloc

    def Loc2Imp(self, matimp: np.ndarray) -> np.ndarray:
        norb = matimp.shape[0]
        ns = matimp.shape[2]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb, norb, ns, nspace), dtype=complex, order="F")

        for key, val in self.crystal.probspace.items():
            iprob = int(key) - 1
            for ispace in val:
                matloc[..., ispace] = matimp[..., iprob]

        return matloc

    def Arr2Dict(self, equiv: np.ndarray, matin: np.ndarray) -> dict:
        ns = matin.shape[2]
        nind = np.amax(equiv)
        matdict = {}

        for ind in range(nind):
            matdict[ind + 1] = []
            pos = self.crystal.FindPositions(equiv, ind + 1)
            for js in range(ns):
                e = 0
                for ii, jj in pos:
                    e += matin[ii, jj, js]
                e /= len(pos)
                matdict[ind + 1].append(e)

        return matdict

    def Dict2Arr(self, equiv: np.ndarray, matdict: dict) -> np.ndarray:
        norb = len(equiv)
        ns = self.crystal.ns
        matout = np.zeros((norb, norb, ns), dtype=complex, order="F")
        nind = np.amax(equiv)

        for js in range(ns):
            for ind in range(nind):
                pos = self.crystal.FindPositions(equiv, ind + 1)
                for ii, jj in pos:
                    matout[ii, jj, js] = matdict[str(ind + 1)]

        return matout

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray):
        norb = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norb, norb, ns), dtype=complex, order="F")

        matout = DiagE.dyson.flocstc(mat1, mat2)

        return matout

    def Embedding(self, matin: np.ndarray):
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nspace = self.crystal.fprojector.shape[3]

        matout = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")

        for ispace in range(nspace):
            matout += DiagE.embedding.flocstc(
                nrk, matin[..., ispace], self.crystal.fprojector[..., ispace]
            )

        return matout


class ImpurityLevel(FLocStc):
    def __init__(self, crystal: Crystal, niham: NIHamiltonian, mu: float):
        super().__init__(crystal)

        self.niham = niham
        self.mu = mu
        self.loc = None
        self.imp = None
        self.Cal()

    def Cal(self):
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nspace = self.crystal.fprojector.shape[3]

        ham = self.niham.UpdateMu(self.niham.hamtb, self.mu)

        eimp = np.zeros((norbc, norbc, ns, nspace), dtype=complex, order="F")

        for ispace in range(nspace):
            eimp[..., ispace] = DiagE.projection.flatstc(
                ham, self.crystal.fprojector[..., ispace]
            )

        self.loc = eimp
        self.imp = self.Loc2Imp(eimp)

        return None


class SigmaHLoc(FLocStc):
    def __init__(self, crystal: Crystal, gloc: GreenLoc, vbare: object):
        super().__init__(crystal)

        self.gloc = gloc
        self.vbare = vbare
        self.hloc = None
        self.himp = None
        self.hdyn = None
        self.Cal()
        self.MakeDyn()

    def Cal(self):
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        norb = self.crystal.bprojector.shape[1]
        nspace = self.crystal.bprojector.shape[3]

        U = np.zeros((norb, norb, ns, ns, nspace), dtype=complex, order="F")
        hloc = np.zeros((norbc, norbc, ns, nspace), dtype=complex, order="F")
        tempmat = np.zeros((norb * ns, norb * ns), dtype=complex, order="F")

        for ispace in range(nspace):
            U[..., ispace] = DiagE.projection.blatstc(
                self.vbare.k, self.crystal.bprojector[..., ispace]
            )

            if ns == 2:
                tempmat = self.crystal.OrbSpin2Composite(U[..., ispace])
                for ind1 in range(norb * ns):
                    nn1 = [0] * 2
                    ind1, [iorb, js] = self.crystal.indexing(
                        norb * ns, 2, [norb, ns], 0, ind1, nn1
                    )
                    iorbc1, iorbc2 = self.crystal.b2f[iorb]
                    for ind2 in range(norb * ns):
                        nn2 = [0] * 2
                        ind2, [jorb, ks] = self.crystal.indexing(
                            norb * ns, 2, [norb, ns], ind2, nn2
                        )
                        iorbc3, iorbc4 = self.crystal.b2f[jorb]
                        hloc[iorbc1, iorbc2, js, ispace] += (
                            -tempmat[ind1, ind2]
                            * self.gloc.gf[iorbc4, iorbc3, ks, -1, ispace]
                        )
            else:
                if self.crystal.soc == False:
                    C = 2
                    for iorb in range(norb):
                        iorbc1, iorbc2 = self.crystal.b2f[iorb]
                        for jorb in range(norb):
                            iorbc3, iorbc4 = self.crystal.b2f[jorb]
                            hloc[iorbc1, iorbc2, 0, ispace] += (
                                -U[iorb, jorb, 0, 0, ispace]
                                * self.gloc.gf[iorbc4, iorbc3, 0, -1, ispace]
                            )
                else:
                    C = 1
                    for iorb in range(norb):
                        iorbc1, iorbc2 = self.crystal.b2f[iorb]
                        for jorb in range(norb):
                            iorbc3, iorbc4 = self.crystal.b2f[jorb]
                            hloc[iorbc1, iorbc2, 0, ispace] += (
                                -U[iorb, jorb, 0, 0, ispace]
                                * self.gloc.gf[iorbc4, iorbc3, 0, -1, ispace]
                            )

        self.hloc = hloc
        self.himp = self.Loc2Imp(hloc)

        return None

    def MakeDyn(self):
        norb = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nft = self.gloc.gf.shape[3]
        nspace = self.crystal.fprojector.shape[3]

        hdyn = np.zeros((norb, norb, ns, nft, nspace), dtype=complex, order="F")

        for ift in range(nft):
            hdyn[..., ift, :] = self.hloc

        self.hdyn = hdyn

        return None


class SigmaHImp(FLocStc):
    def __init__(self, crystal: Crystal):
        super().__init__(crystal)
        self.Cal()

    def Cal(self):
        pass


class SigmaFLoc(FLocStc):
    def __init__(self, crystal: Crystal, gloc: GreenLoc, vbare: object):
        super().__init__(crystal)

        self.gloc = gloc
        self.vbare = vbare
        self.floc = None
        self.fimp = None
        self.fdyn = None

        self.Cal()
        self.MakeDyn()

    def Cal(self):
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        norb = self.crystal.bprojector.shape[1]
        nspace = self.crystal.fprojector.shape[3]

        U = np.zeros((norb, norb, ns, ns, nspace), dtype=complex, order="F")
        floc = np.zeros((norbc, norbc, ns, nspace), dtype=complex, order="F")

        for ispace in range(nspace):
            U[..., ispace] = DiagE.projection.blatstc(
                self.vbare.k, self.crystal.bprojector[..., ispace]
            )

            for js in range(ns):
                for iorb in range(norb):
                    iorbc1, iorbc4 = self.crystal.b2f[iorb]
                    for jorb in range(norb):
                        iorbc3, iorbc2 = self.crystal.b2f[jorb]
                        floc[iorbc1, iorbc2, js, ispace] += (
                            self.gloc.gf[iorbc4, iorbc3, js, -1, ispace]
                            * U[iorb, jorb, js, js, ispace]
                        )

        self.floc = floc
        self.fimp = self.Loc2Imp(floc)

        return None

    def MakeDyn(self):
        norb = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nft = self.gloc.gf.shape[3]
        nspace = self.crystal.fprojector.shape[3]

        fdyn = np.zeros((norb, norb, ns, nft, nspace), dtype=complex, order="F")

        for ift in range(nft):
            fdyn[..., ift, :] = self.floc

        return None


class SigmaFImp(FLocStc):
    def __init__(self, crystal: Crystal):
        super().__init__(crystal)
        self.Cal()

    def Cal(self):
        pass


class BLatDyn(object):
    def __init__(self, crystal: Crystal, ft: FT_grid):
        self.crystal = crystal
        self.ft = ft
        # self.flatdyn = flatdyn

    def Inverse(self, matin: np.ndarray) -> np.ndarray:
        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]
        nft = matin.shape[5]

        matout = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=complex, order="F")
        tempmat = np.zeros((norb * ns, norb * ns), dtype=complex)
        tempmat2 = np.zeros((norb * ns, norb * ns), dtype=complex)

        # Make composite matrix#
        for ift in range(nft):
            for irk in range(nrk):
                tempmat = self.crystal.OrbSpin2Composite(matin[:, :, :, :, irk, ift])
                tempmat2 = np.linalg.inv(tempmat)
                matout[:, :, :, :, irk, ift] = self.crystal.Composite2OrbSpin(tempmat2)

        return matout

    def Moment(self, bf: np.ndarray, oddzero: int, highzero: int) -> np.ndarray:
        norb = bf.shape[0]
        ns = bf.shape[2]
        nrk = bf.shape[4]

        moment = np.zeros((norb, norb, ns, ns, nrk, 3), dtype=complex, order="F")
        high = np.zeros((norb, norb, ns, nrk), dtype=complex, order="F")

        moment, high = DiagE.fourier.blatdyn_m(self.ft.nu, bf, oddzero, highzero)

        return moment, high

    def F2T(self, bf: np.ndarray, oddzero: int, highzero: int) -> np.ndarray:
        norb = bf.shape[0]
        ns = bf.shape[2]
        nrk = bf.shape[4]
        ntau = len(self.ft.tau)

        btau = np.zeros((norb, norb, ns, ns, nrk, ntau), dtype=complex, order="F")

        moment, high = self.Moment(bf, oddzero, highzero)

        btau = DiagE.fourier.blatdyn_f2t(self.ft.nu, bf, moment, self.ft.tau)

        return btau

    def T2F(self, btau: np.ndarray) -> np.ndarray:
        norb = btau.shape[0]
        ns = btau.shape[2]
        nrk = btau.shape[4]
        nfreq = len(self.ft.nu)

        bf = np.zeros((norb, norb, ns, ns, nrk, nfreq), dtype=complex, order="F")

        bf = DiagE.fourier.blatdyn_t2f(self.ft.tau, btau, self.ft.nu)

        return bf

    def K2R(self, matk: np.ndarray) -> np.ndarray:
        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[4]
        nft = matk.shape[5]
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        matr = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=complex, order="F")

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                [a, [m1, m4]] = self.crystal.BAtomOrb(iorb)
                                [b, [m2, m3]] = self.crystal.BAtomOrb(jorb)

                                delta = (
                                    self.crystal.basisf[a, :]
                                    - self.crystal.basisf[b, :]
                                )

                                phase = np.exp(2.0j * np.pi * np.dot(rkvec[irk], delta))
                                matk[iorb, jorb, js, ks, irk, ift] *= phase

        matr = DiagE.fourier.blatdyn_k2r(rkgrid, matk)

        return matr

    def R2K(self, matr: np.ndarray) -> np.ndarray:
        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[4]
        nft = matr.shape[5]
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        matk = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=complex, order="F")

        matk = DiagE.fourier.blatdyn_r2k(rkgrid, matr)

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                [a, [m1, m4]] = self.crystal.BAtomOrb(iorb)
                                [b, [m2, m3]] = self.crystal.BAtomOrb(jorb)

                                delta = (
                                    self.crystal.basisf[a, :]
                                    - self.crystal.basisf[b, :]
                                )
                                phase = np.exp(
                                    -2.0j * np.pi * np.dot(rkvec[irk], delta)
                                )

                                matk[iorb, jorb, js, ks, irk, ift] *= phase

        return matk

    def GaussianLinearBroad(self, x, y, w1, temperature, cutoff):
        norb = y.shape[0]
        ns = y.shape[2]
        nrk = y.shape[3]
        nft = y.shape[4]

        ynew = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=complex, order="F")

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

        Bnew = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=complex, order="F")

        if iter == 1:
            mix = 1.0
            Bold = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=complex, order="F")

        Bnew = mix * Bb + (1 - mix) * Bold

        return Bnew

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
        norb = mat1.shape[0]
        ns = mat1.shape[2]
        nrk = mat1.shape[3]
        nft = mat1.shape[4]

        matout = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=complex, order="F")

        matout = DiagE.dyson.blatdyn(mat1, mat2)

        return matout

    def Projection(self, matin: np.ndarray):
        norbc = self.crystal.bprojector.shape[1]
        ns = self.crystal.ns
        nft = self.ft.size
        nspace = self.crystal.bprojector.shape[3]

        matout = np.zeros((norbc, norbc, ns, ns, nft, nspace), dtype=complex, order="F")

        for ispace in range(nspace):
            matout[..., ispace] = DiagE.projection.blatdyn(
                matin, self.crystal.bprojector[..., ispace]
            )

        return matout

    def Quad2Double(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=complex, order="F")

        for ift in range(nft):
            for irk in range(nrk):
                for ks in range(ns):
                    for js in range(ns):
                        matout[:, :, js, irk, ift] = self.crystal.Quad2Double(
                            matin[:, :, :, :, js, ks, irk, ift]
                        )

        return matout

    def Double2Quad(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros(
            (norb, norb, norb, norb, ns, ns, nrk, nft), dtype=complex, order="F"
        )

        for ift in range(nft):
            for irk in range(nrk):
                for ks in range(ns):
                    for js in range(ns):
                        matout[:, :, :, :, js, ks, irk, ift] = self.crystal.Double2Quad(
                            matin[:, :, js, ks, irk, ift]
                        )

        return matout

    def Double2Full(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros(
            (norb * norb, norb * norb, ns, ns, nrk, nft), dtype=complex, order="F"
        )

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        matout[:, :, js, ks, irk, ift] = self.crystal.Double2Full(
                            matin[:, :, js, ks, irk, ift]
                        )

        return matout

    def Full2Double(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=complex, order="F")

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        matout[:, :, js, ks, irk, ift] = self.crystal.Full2Double(
                            matin[:, :, js, ks, irk, ift]
                        )

        return matout

    def Quad2Full(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros(
            (norb * norb, norb * norb, ns, ns, nrk, nft), dtype=complex, order="F"
        )

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        matout[:, :, js, ks, irk, ift] = self.crystal.Quad2Full(
                            matin[:, :, :, :, js, ks, irk, ift]
                        )

        return matout

    def Full2Quad(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros(
            (norb, norb, norb, norb, ns, ns, nrk, nft), dtype=complex, order="F"
        )

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        matout[:, :, :, :, js, ks, irk, ift] = self.crystal.Full2Quad(
                            matin[:, :, js, ks, irk, ift]
                        )

        return matout


class PolLat(BLatDyn):
    def __init__(self, crystal: Crystal, ft: FT_grid, green=None):
        super().__init__(crystal, ft)
        self.polrt = None  # rt to kf
        self.polrf = None
        self.polkt = None
        self.polkf = None
        if green == None:
            print("Error, There is no Green's function.")
            sys.exit()
        self.green = green

        self.Cal()
        self.polkt = self.K2R(self.polrt)
        self.polrf = self.T2F(self.polrt)
        self.polkf = self.T2F(self.polkt)

    def Cal(self):
        grt = self.green.glatrt
        norbc = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        ntau = len(self.ft.tau)
        nfreq = len(self.ft.nu)
        norb = len(self.crystal.bind)

        tempmat = np.zeros(
            (norbc, norbc, norbc, norbc, ns, ns, nrk, ntau), dtype=complex, order="F"
        )
        polrt = np.zeros((norb, norb, ns, ns, nrk, ntau), dtype=complex, order="F")

        gmrt = self.green.greenbare.RT2mRmT(grt)

        if ns == 2:
            for itau in range(ntau):
                for irk in range(nrk):
                    for js in range(ns):
                        for ks in range(ns):
                            for iorbc, jorbc, korbc, lorbc in itertools.product(
                                list(range(norbc)),
                                list(range(norbc)),
                                list(range(norbc)),
                                list(range(norbc)),
                            ):
                                if js == ks:
                                    tempmat[
                                        iorbc, lorbc, jorbc, korbc, js, ks, irk, itau
                                    ] = (
                                        gmrt[jorbc, iorbc, js, irk, itau]
                                        * grt[korbc, lorbc, ks, irk, itau]
                                    )
        else:
            if self.crystal.soc == True:
                C = 1
                for itau in range(ntau):
                    for irk in range(nrk):
                        for iorbc, jorbc, korbc, lorbc in itertools.product(
                            list(range(norbc)),
                            list(range(norbc)),
                            list(range(norbc)),
                            list(range(norbc)),
                        ):
                            tempmat[iorbc, lorbc, jorbc, korbc, 0, 0, irk, itau] = (
                                gmrt[jorbc, iorbc, 0, irk, itau]
                                * grt[korbc, lorbc, 0, irk, itau]
                                * C
                            )
            else:
                C = 2
                for itau in range(ntau):
                    for irk in range(nrk):
                        for iorbc, jorbc, korbc, lorbc in itertools.product(
                            list(range(norbc)),
                            list(range(norbc)),
                            list(range(norbc)),
                            list(range(norbc)),
                        ):
                            tempmat[iorbc, lorbc, jorbc, korbc, 0, 0, irk, itau] = (
                                gmrt[jorbc, iorbc, 0, irk, itau]
                                * grt[korbc, lorbc, 0, irk, itau]
                                * C
                            )

        for itau in range(ntau):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        polrt[:, :, js, ks, irk, itau] = self.crystal.Quad2Double(
                            tempmat[:, :, :, :, js, ks, irk, itau]
                        )
        self.polrt = polrt

        return None


class WLat(BLatDyn):
    def __init__(self, crystal: Crystal, ft: FT_grid, pol: object = None, vbare=None):
        super().__init__(crystal, ft)
        self.wrt = None  # rt to kf
        self.wrf = None
        self.wkt = None
        self.wkf = None
        self.wcrt = None  # rt to kf
        self.wcrf = None
        self.wckt = None
        self.wckf = None
        if pol == None:
            print("Error, polarizability doesn't exist")
            sys.exit()
        if vbare == None:
            print("Error, bare coulomb interaction doesn't exist")
            sys.exit()
        self.pol = pol
        self.vbare = vbare

        self.Cal()

        self.wkt = self.F2T(self.wkf, 1, 1)
        self.wrf = self.K2R(self.wkf)
        self.wrt = self.K2R(self.wkt)

        self.wckt = self.F2T(self.wckf, 1, 1)
        self.wcrf = self.K2R(self.wckf)
        self.wcrt = self.K2R(self.wckt)

    def Cal(self):  # calculate W and Wc
        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nfreq = len(self.ft.nu)
        ####### Initialization #######
        tempmat = np.zeros(
            (norbc * norbc, norbc * norbc, ns, ns, nk, nfreq), dtype=complex, order="F"
        )
        wkf = np.zeros((norb, norb, ns, ns, nk, nfreq), dtype=complex, order="F")
        wckf = np.zeros((norb, norb, ns, ns, nk, nfreq), dtype=complex, order="F")
        vdyn = np.zeros((norb, norb, ns, ns, nk, nfreq), dtype=complex, order="F")

        for ifreq in range(nfreq):
            vdyn[..., ifreq] = self.vbare.vbarek
        polcomp = np.zeros(
            (norbc * norbc, norbc * norbc, ns, ns, nk, nfreq), dtype=complex, order="F"
        )
        vcomp = np.zeros(
            (norbc * norbc, norbc * norbc, ns, ns, nk, nfreq), dtype=complex, order="F"
        )
        ####### Initialization #######
        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    for ks in range(ns):
                        polcomp[:, :, js, ks, ik, ifreq] = self.crystal.Double2Full(
                            self.pol.polkf[:, :, js, ks, ik, ifreq]
                        )
                        vcomp[:, :, js, ks, ik, ifreq] = self.crystal.Double2Full(
                            vdyn[:, :, js, ks, ik, ifreq]
                        )
        tempmat = self.Dyson(vcomp, polcomp)

        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    for ks in range(ns):
                        wkf[:, :, js, ks, ik, ifreq] = self.crystal.Full2Double(
                            tempmat[:, :, js, ks, ik, ifreq]
                        )
        self.wkf = wkf

        wckf = wkf - vdyn

        self.wckf = wckf

        return None


# class WcLat(BLatDyn):

#     def __init__(self, crystal: Crystal, ft: FT_grid, w ):
#         super().__init__(crystal, ft, flatdyn)

#         pass


class BLatStc(object):
    def __init__(self, crystal: Crystal):
        self.crystal = crystal

    def Inverse(self, matin: np.ndarray) -> np.ndarray:
        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]

        matout = np.zeros((norb, norb, ns, ns, nrk), dtype=complex, order="F")
        tempmat = np.zeros((norb * ns, norb * ns), dtype=complex)
        tempmat2 = np.zeros((norb * ns, norb * ns), dtype=complex)

        for irk in range(nrk):
            tempmat = self.crystal.OrbSpin2Composite(matin[..., irk])
            tempmat2 = np.linalg.inv(tempmat)
            matout[..., irk] = self.crystal.Composite2OrbSpin(tempmat2)

        return matout

    def K2R(self, matk: np.ndarray) -> np.ndarray:
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matk.shape[0]
        ns = self.crystal.ns
        nrk = len(rkvec)

        matr = np.zeros((norb, norb, ns, ns, nrk), dtype=complex, order="F")

        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            [a, [m1, m4]] = self.crystal.BAtomOrb(iorb)
                            [b, [m2, m3]] = self.crystal.BAtomOrb(jorb)

                            delta = (
                                self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                            )
                            phase = np.exp(2.0j * np.pi * np.dot(rkvec[irk], delta))

                            matk[iorb, jorb, js, ks, irk] *= phase

        matr = DiagE.fourier.blatstc_k2r(rkgrid, matk)

        return matr

    def R2K(self, matr: np.ndarray) -> np.ndarray:
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matr.shape[0]
        ns = self.crystal.ns
        nrk = len(rkvec)

        matk = np.zeros((norb, norb, ns, ns, nrk), dtype=complex, order="F")

        matk = DiagE.fourier.blatstc_r2k(rkgrid, matr)

        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            [a, [m1, m4]] = self.crystal.BAtomOrb(iorb)
                            [b, [m2, m3]] = self.crystal.BAtomOrb(jorb)

                            delta = (
                                self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                            )
                            phase = np.exp(-2.0j * np.pi * np.dot(rkvec[irk], delta))

                            matk[iorb, jorb, js, ks, irk] *= phase

        return matk

    def Mixing(
        self, iter: int, mix: float, Bb: np.ndarray, Bold: np.ndarray
    ) -> np.ndarray:
        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nrk = Bb.shape[4]

        Bnew = np.zeros((norb, norb, ns, ns, nrk), dtype=complex, order="F")

        if iter == 1:
            mix = 1.0

        Bnew = mix * Bb + (1.0 - mix) * Bold

        return Bnew

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray):
        norb = mat1.shape[0]
        ns = mat1.shape[2]
        nrk = mat1.shape[4]

        matout = np.zeros((norb, norb, ns, ns, nrk), dtype=complex, order="F")

        matout = DiagE.dyson.blatstc(mat1, mat2)

        return matout

    def Projection(self, matin: np.ndarray):
        norbc = self.crystal.bprojector.shape[1]
        nspace = self.crystal.bprojector.shape[3]
        ns = self.crystal.ns

        matout = np.zeros((norbc, norbc, ns, ns, nspace), dtype=complex, order="F")

        for ispace in range(nspace):
            matout[..., ispace] = DiagE.projection.blatstc(
                matin, self.crystal.bprojector[..., ispace]
            )

        return matout

    def Quad2Double(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros((norb, norb, ns, ns, nrk), dtype=complex, order="F")

        for irk in range(nrk):
            for ks in range(ns):
                for js in range(ns):
                    matout[:, :, js, irk] = self.crystal.Quad2Double(
                        matin[:, :, :, :, js, ks, irk]
                    )

        return matout

    def Double2Quad(self, matin: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros(
            (norb, norb, norb, norb, ns, ns, nrk), dtype=complex, order="F"
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
            (norb * norb, norb * norb, ns, ns, nrk), dtype=complex, order="F"
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

        matout = np.zeros((norb, norb, ns, ns, nrk), dtype=complex, order="F")

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
            (norb * norb, norb * norb, ns, ns, nrk), dtype=complex, order="F"
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
            (norb, norb, norb, norb, ns, ns, nrk), dtype=complex, order="F"
        )

        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    matout[:, :, :, :, js, ks, irk] = self.crystal.Full2Quad(
                        matin[:, :, js, ks, irk]
                    )

        return matout


class VBare(BLatStc):
    def __init__(
        self, crystal: Crystal, vloc=None, orboption: dict = None, intamp: list = None
    ):
        super().__init__(crystal)
        # self.vbarek = None
        # self.vbarer = None
        # self.vnonloc = None
        self.k = None
        self.r = None
        self.intamp = intamp
        self.nonlock = None
        self.nonlocr = None
        if vloc == None:
            if orboption != None:
                self.vloc = VLoc(crystal, orboption)
            else:
                print("Error, orboption is not exsist. v local can't generate in here")
        else:
            self.vloc = vloc
        if intamp != None:
            # self.InteractingAmplitue(intamp)
            self.Cal()
        self.LocPlusNonLoc()

    def Cal(self):
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint
        rtest = np.multiply(rkvec, rkgrid)  # for test
        rtest = np.around(rtest.tolist()).tolist()

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nk = len(rkvec)
        vnlk = np.zeros((norb, norb, ns, ns, nk), dtype=complex, order="F")
        tempmat = np.zeros(
            (norb, norb, ns, ns, rkgrid[0], rkgrid[1], rkgrid[2]),
            dtype=complex,
            order="F",
        )

        # for ik in range(nk):
        #     for js in range(ns):
        #         for ks in range(ns):
        #             for ind in self.intamp:
        #                 vij = ind[0]
        #                 iorb = ind[1]
        #                 jorb = ind[2]
        #                 R = np.array(ind[3])
        #                 [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
        #                 [b,[m2,m3]] = self.crystal.BAtomOrb(jorb)

        #                 rvec = self.crystal.basisf[a,:] - self.crystal.basisf[b,:] + R
        #                 phase = np.exp(-2.0j*np.pi*np.dot(rkvec[ik],rvec))

        #                 vnlk[iorb,jorb,js,ks,ik] += vij*phase
        #                 vnlk[jorb,iorb,js,ks,ik] += vij*np.conjugate(phase)

        for js in range(ns):
            for ks in range(ns):
                for ind in self.intamp:
                    vij = ind[0]
                    iorb = ind[1]
                    jorb = ind[2]
                    R = ind[3]

                    tempmat[iorb, jorb, js, ks, R[0], R[1], R[2]] += vij
                    tempmat[jorb, iorb, js, ks, R[0], R[1], R[2]] += vij

        vnlk = tempmat.reshape(norb, norb, ns, ns, nk)
        self.nonlocr = vnlk
        self.nonlock = self.R2K(vnlk)
        # self.nonlock = vnlk
        # self.nonlocr = self.K2R(vnlk)

        return None

    # def InteractingAmplitue(self,intamp : list)-> list:

    #     pass

    def LocPlusNonLoc(self):
        vloc = self.vloc.vloc
        vnlk = self.nonlock

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)

        vbare = np.zeros((norb, norb, ns, ns, nk), dtype=complex, order="F")
        if self.intamp == None:
            for ik in range(nk):
                vbare[..., ik] = vloc
        else:
            for ik in range(nk):
                vbare[..., ik] = vloc + vnlk[..., ik]

        self.k = vbare
        self.r = self.K2R(vbare)

        return None


class BLocDyn(object):
    def __init__(self, crystal: Crystal, ft: FT_grid):
        self.crystal = crystal
        self.ft = ft

    def Inverse(self, matin: np.ndarray) -> np.ndarray:
        norb = matin.shape[0]
        ns = matin.shape[2]
        nft = self.ft.size

        matout = np.zeros((norb, norb, ns, ns, nft), dtype=complex, order="F")
        tempmat = np.zeros((norb * ns, norb * ns), dtype=complex)
        tempmat2 = np.zeros((norb * ns, norb * ns), dtype=complex)

        for ift in range(nft):
            tempmat = self.crystal.OrbSpin2Composite(matin[..., ift])
            tempmat2 = np.linalg.inv(tempmat)
            matout[..., ift] = self.crystal.Composite2OrbSpin(tempmat2)

        return matout

    def Moment(self, bf: np.ndarray, oddzero: int, highzero: int) -> np.ndarray:
        norb = len(self.crystal.bind)
        ns = self.crystal.ns

        moment = np.zeros((norb, norb, ns, ns, 3), dtype=complex, order="F")
        high = np.zeros((norb, norb, ns, ns), dtype=complex, order="F")
        moment, high = DiagE.fourier.blocdyn_m(self.ft.nu, bf, oddzero, highzero)

        return moment, high

    def F2T(self, bf: np.ndarray, oddzero: int, highzero: int) -> np.ndarray:
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nft = self.ft.size

        btau = np.zeros((norb, norb, ns, ns, nft), dtype=complex, order="F")

        moment, high = self.Moment(bf, oddzero, highzero)

        btau = DiagE.fourier.blocdyn_f2t(self.ft.nu, bf, moment, self.ft.tau)

        return btau

    def T2F(self, btau: np.ndarray) -> np.ndarray:
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nft = self.ft.size

        bf = np.zeros((norb, norb, ns, ns, nft), dtype=complex, order="F")

        bf = DiagE.fourier.blocdyn_t2f(self.ft.tau, btau, self.ft.nu)

        return bf

    def GaussianLinearBroad(self, x, y, w1, temperature, cutoff):
        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]

        ynew = np.zeros((norb, norb, ns, ns, nft), dtype=complex, order="F")
        w0 = (1.0 - 3.0 * w1) * np.pi * temperature
        widtharray = w0 + w1 * x
        cnt = 0

        for x0 in x:
            if x0 > cutoff + (w0 + w1 * cutoff) * 3.0:
                ynew[..., cnt] = y[..., cnt]
            else:
                if (x0 > 3 * widtharray[cnt]) and ((x[-1] - x0) > 3 * widtharray[cnt]):
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
                                    ynew[iorb, jorb, js, ks, cnt] = sum(
                                        dist * y[iorb, jorb, js, ks]
                                    ) / sum(dist)
                else:
                    ynew[..., cnt] = y[..., cnt]
            cnt += 1

        return ynew

    def Mixing(self, iter: int, mix: float, Bb: np.ndarray, Bold: np.ndarray):
        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nft = Bb.shape[4]

        Bnew = np.zeros((norb, norb, ns, ns, nft), dtype=complex, order="F")

        if iter == 1:
            mix = 1.0
            Bold = np.zeros((norb, norb, ns, ns, nft), dtype=complex, order="F")

        Bnew = mix * Bb + (1 - mix) * Bold

        return Bnew

    def Imp2Loc(self, matimp: np.ndarray) -> np.ndarray:
        norb = matimp.shape[0]
        ns = matimp.shape[2]
        nft = matimp.shape[3]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb, norb, ns, ns, nft, nspace), dtype=complex, order="F")

        for key, val in self.crystal.probspace.items():
            iprob = int(key) - 1
            for ispace in val:
                matloc[..., ispace] = matimp[..., iprob]

        return matloc

    def Loc2Imp(self, matloc: np.ndarray) -> np.ndarray:
        nprob = len(self.crystal.probspace)
        norb = matloc.shape[0]
        ns = matloc.shape[2]
        nft = matloc.shape[3]

        matimp = np.zeros((norb, norb, ns, ns, nft, nprob), dtype=complex, order="F")

        for key, val in self.crystal.probspace.items():
            iprob = int(key) - 1
            tempmat = np.zeros((norb, norb, ns), dtype=complex)
            for ispace in val:
                tempmat += matloc[..., ispace]
            tempmat /= len(val)
            matimp[..., iprob] = tempmat

        return matimp

    def Arr2Dict(self, equiv: np.ndarray, matin: np.ndarray) -> dict:
        ns = matin.shape[2]
        nind = np.amax(equiv)
        matdict = {}

        for ind in range(nind):
            matdict[ind + 1] = []
            pos = self.crystal.FindPositions(equiv, ind + 1)
            for js in range(ns):
                for ks in range(ns):
                    e = 0
                    for ii, jj in pos:
                        e += matin[ii, jj, js, ks]
                    e /= len(pos)
                    matdict[ind + 1].append(e.tolist())

        return matdict

    def Dict2Arr(self, equiv: np.ndarray, matdict: np.ndarray) -> np.ndarray:
        norb = len(equiv)
        ns = self.crystal.ns
        nfreq = len(matdict["1"])

        matout = np.zeros((norb, norb, ns, ns, nfreq), dtype=complex, order="F")
        nind = np.amax(equiv)

        for js in range(ns):
            for ks in range(ns):
                for ind in range(nind):
                    pos = self.crystal.FindPositions(equiv, ind + 1)
                    for ii, jj in pos:
                        matout[ii, jj, js, ks] = matdict[str(ind + 1)]

        return matout

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray):
        norb = mat1.shape[0]
        ns = self.crystal.ns
        nft = self.ft.size

        matout = np.zeros((norb, norb, ns, ns, nft), dtype=complex, order="F")

        matout = DiagE.dyson.blocdyn(mat1, mat2)

        return matout

    def Embedding(self, matin: np.ndarray):
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size
        nspace = self.crystal.bprojector.shape[3]

        matout = np.zeros((norb, norb, ns, ns, nrk, nft), dtype=complex, order="F")

        for ispace in range(nspace):
            matout += DiagE.embedding.blocdyn(
                nrk, matin[..., ispace], self.crystal.bprojector[..., ispace]
            )

        return matout


class PolLoc(BLocDyn):
    def __init__(self, crystal: Crystal, ft: FT_grid, green, pol: object):
        super().__init__(crystal, ft)
        self.Cal()

    def Cal(self):
        pass


class PolImp(BLocDyn):  # read Polarizability from CTQMC
    def __init__(self, crystal: Crystal, ft: FT_grid):
        super().__init__(crystal, ft)

        pass


class WLoc(BLocDyn):
    def __init__(self, crystal: Crystal, ft: FT_grid, flocdyn: FLocDyn):
        super().__init__(crystal, ft, flocdyn)

        pass


class WImp(BLocDyn):
    def __init__(self, crystal: Crystal, ft: FT_grid, flocdyn: FLocDyn):
        super().__init__(crystal, ft, flocdyn)

        pass


class WcLoc(BLocDyn):
    def __init__(self, crystal: Crystal, ft: FT_grid, flocdyn: FLocDyn):
        super().__init__(crystal, ft, flocdyn)

        pass


class WcImp(BLocDyn):
    def __init__(self, crystal: Crystal, ft: FT_grid, flocdyn: FLocDyn):
        super().__init__(crystal, ft, flocdyn)

        pass


class BLocStc(object):
    def __init__(self, crystal: Crystal):
        self.crystal = crystal

    def Inverse(self, matin: np.ndarray) -> np.ndarray:
        norb = matin.shape[0]
        ns = matin.shape[2]

        matout = np.zeros((norb, norb, ns, ns), dtype=complex, order="F")
        tempmat = np.zeros((norb * ns, norb * ns), dtype=complex)
        tempmat2 = np.zeros((norb * ns, norb * ns), dtype=complex)

        tempmat = self.crystal.OrbSpin2Composite(matin)
        tempmat2 = np.linalg.inv(tempmat)
        matout = self.crystal.Composite2OrbSpin(tempmat2)

        return matout

    def Mixing(
        self, iter: int, mix: float, Bb: np.ndarray, Bold: np.ndarray
    ) -> np.ndarray:
        norb = Bb.shape[0]
        ns = Bb.shape[2]

        Bnew = np.zeros((norb, norb, ns, ns), dtype=complex, order="F")

        if iter == 1:
            mix = 1.0
            Bold = np.zeros((norb, norb, ns, ns), dtype=complex, order="F")

        Bnew = mix * Bb + (1 - mix) * Bold

        return Bnew

    def Imp2Loc(self, matimp: np.ndarray) -> np.ndarray:
        norb = matimp.shape[0]
        ns = matimp.shape[2]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb, norb, ns, ns, nspace), dtype=complex, order="F")

        for key, val in self.crystal.probspace.items():
            iprob = int(key) - 1
            for ispace in val:
                matloc[..., ispace] = matimp[..., iprob]

        return matloc

    def Loc2Imp(self, matimp: np.ndarray) -> np.ndarray:
        norb = matimp.shape[0]
        ns = matimp.shape[2]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb, norb, ns, ns, nspace), dtype=complex, order="F")

        for key, val in self.crystal.probspace.items():
            iprob = int(key) - 1
            for ispace in val:
                matloc[..., ispace] = matimp[..., iprob]

        return matloc

    def Arr2Dict(self, equiv: np.ndarray, matin: np.ndarray) -> dict:
        ns = matin.shape[2]
        nind = np.amax(equiv)
        matdict = {}

        for ind in range(nind):
            matdict[ind + 1] = []
            pos = self.crystal.FindPositions(equiv, ind + 1)
            for js in range(ns):
                for ks in range(ns):
                    e = 0
                    for ii, jj in pos:
                        e += matin[ii, jj, js, ks]
                    e /= len(pos)
                    matdict[ind + 1].append(e)

        return matdict

    def Dict2Arr(self, equiv: np.ndarray, matdict: dict) -> np.ndarray:
        norb = len(equiv)
        ns = self.crystal.ns
        matout = np.zeros((norb, norb, ns), dtype=complex, order="F")
        nind = np.amax(equiv)

        for js in range(ns):
            for ks in range(ns):
                for ind in range(nind):
                    pos = self.crystal.FindPositions(equiv, ind + 1)
                    for ii, jj in pos:
                        matout[ii, jj, js, ks] = matdict[str(ind + 1)]

        return matout

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray):
        norb = mat1.shape[0]
        ns = mat1.shape[2]

        matout = np.zeros((norb, norb, ns, ns), dtype=complex, order="F")

        matout = DiagE.dyson.blocstc(mat1, mat2)

        return matout

    def Embedding(self, matin: np.ndarray):
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nspace = self.crystal.bprojector.shape[3]

        matout = np.zeros((norb, norb, ns, ns, nrk), dtype=complex, order="F")

        for ispace in range(nspace):
            matout += DiagE.embedding.blocstc(
                nrk, matin[..., ispace], self.crystal.bprojector.shape[..., ispace]
            )

        return matout


class VLoc(BLocStc):
    def __init__(self, crystal: Crystal, orboption: dict):
        super().__init__(crystal)
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        self.vloc = np.zeros((norb, norb, ns, ns), dtype=float, order="F")

        self.SetLocalInteracting(orboption)

    def SetLocalInteracting(self, orboption: dict):
        ns = self.crystal.ns
        for val in orboption.values():
            norbc = len(val["orbitals"])

            if val["KorS"] == "K":
                tempmat = self.KanamoriParameter(norbc, val["value"])
                for js in range(ns):
                    for ks in range(ns):
                        for iorbc in val["orbitals"]:
                            for jorbc in val["orbitals"]:
                                for korbc in val["orbitals"]:
                                    for lorbc in val["orbitals"]:
                                        [a, m1] = self.crystal.FAtomOrb(iorbc)
                                        [b, m2] = self.crystal.FAtomOrb(jorbc)
                                        [bp, m3] = self.crystal.FAtomOrb(korbc)
                                        [ap, m4] = self.crystal.FAtomOrb(lorbc)
                                        if (a == ap) and (b == bp):
                                            iorb = self.crystal.BIndex([a, [m1, m4]])
                                            jorb = self.crystal.BIndex([b, [m2, m3]])
                                            self.vloc[iorb, jorb, js, ks] = tempmat[
                                                m1, m2, m3, m4, js, ks
                                            ]
            elif val["KorS"] == "S":
                tempmat = self.SlaterParameter(norbc, val["value"])
                for js in range(ns):
                    for ks in range(ns):
                        for iorbc in val["orbitals"]:
                            for jorbc in val["orbitals"]:
                                for korbc in val["orbitals"]:
                                    for lorbc in val["orbitals"]:
                                        [a, m1] = self.crystal.FAtomOrb(iorbc)
                                        [b, m2] = self.crystal.FAtomOrb(jorbc)
                                        [bp, m3] = self.crystal.FAtomOrb(korbc)
                                        [ap, m4] = self.crystal.FAtomOrb(lorbc)
                                        if (a == ap) and (b == bp):
                                            iorb = self.crystal.BIndex([a, [m1, m4]])
                                            jorb = self.crystal.BIndex([b, [m2, m3]])
                                            self.vloc[iorb, jorb, js, ks] = tempmat[
                                                m1, m2, m3, m4, js, ks
                                            ]

        return None

    def KanamoriParameter(self, norb: int, val: list) -> np.ndarray:
        print("Warning : In kanamori interaction, self interaction term has been added")
        ns = self.crystal.ns
        v = np.zeros((norb, norb, norb, norb, ns, ns), dtype=float, order="F")
        U = val[0]
        Up = val[1]
        J = val[2]

        for js in range(ns):
            for ks in range(ns):
                for m1 in range(norb):
                    for m2 in range(norb):
                        for m3 in range(norb):
                            for m4 in range(norb):
                                if (m1 == m2 == m3 == m4) and (js != ks):
                                    v[m1, m2, m3, m4, js, ks] = U
                                elif (
                                    (m1 == m4)
                                    and (m2 == m3)
                                    and (m1 != m2)
                                    and (js != ks)
                                ):
                                    v[m1, m2, m3, m4, js, ks] = Up
                                elif (
                                    (m1 == m4)
                                    and (m2 == m3)
                                    and (m1 != m2)
                                    and (js == ks)
                                ):
                                    v[m1, m2, m3, m4, js, ks] = Up - J
                                elif (
                                    (m1 == m3)
                                    and (m2 == m4)
                                    and (m1 != m2)
                                    and (js != ks)
                                ):
                                    v[m1, m2, m3, m4, js, ks] = J
                                elif (
                                    (m1 == m2)
                                    and (m3 == m4)
                                    and (m1 != m3)
                                    and (js != ks)
                                ):
                                    v[m1, m2, m3, m4, js, ks] = J
        v *= 0.5
        return v

    def SlaterParameter(self, norb: int, val: list, sc: str = "c") -> np.ndarray:
        ns = self.crystal.ns
        v = np.zeros((norb, norb, norb, norb, ns, ns), dtype=float, order="F")

        l = int((norb - 1) / 2)
        m = list(range(-l, l + 1))

        for n, f in enumerate(val):
            k = 2 * n

            for js in range(ns):
                for ks in range(ns):
                    for m1 in m:
                        for m2 in m:
                            for m3 in m:
                                for m4 in m:
                                    v[
                                        m1 + l, m2 + l, m3 + l, m4 + l, js, ks
                                    ] += f * self.AngularIntegral(l, k, m1, m2, m3, m4)
        if sc == "c":
            for js in range(ns):
                for ks in range(ns):
                    tempmat = v[:, :, :, :, js, ks]
                    tempmat2 = self.Spherical2Cubic(tempmat, l)
                    v[:, :, :, :, js, ks] = tempmat2
            return v
        else:
            return v

    def AngularIntegral(self, l, k, m1, m2, m3, m4):
        ang_int = 0
        pi = np.pi

        for q in range(-k, k + 1):
            ang_int += (
                gaunt(l, k, l, -m1, q, m3)
                * np.conjugate(gaunt(l, k, l, m4, -q, -m2))
                * ((-1.0 if (m1 + q + m2) % 2 == 1 else 1.0))
            )

        ang_int *= 4 * pi / (2 * k + 1)

        return ang_int

    def RotationMatrix(self, l: int):
        mrange = int(2 * l + 1)
        R = np.zeros((mrange, mrange), dtype=complex)

        if l == 0:
            R = np.eye(mrange, mrange, dtype=complex)
        elif l == 1:
            """/n
            py, pz, px
            """
            R[0, 0] = 1j / np.sqrt(2)
            R[2, 0] = 1j / np.sqrt(2)

            R[1, 1] = 1

            R[0, 2] = 1 / np.sqrt(2)
            R[2, 2] = -1 / np.sqrt(2)

        elif l == 2:
            """/n
            xy, yz, z^2, xz, x^2-y^2
            """

            R[0, 0] = 1j / np.sqrt(2)
            R[4, 0] = -1j / np.sqrt(2)

            R[1, 1] = 1j / np.sqrt(2)
            R[3, 1] = 1j / np.sqrt(2)

            R[2, 2] = 1

            R[1, 3] = 1 / np.sqrt(2)
            R[3, 3] = -1 / np.sqrt(2)

            R[0, 4] = 1 / np.sqrt(2)
            R[4, 4] = 1 / np.sqrt(2)

        elif l == 3:
            """/n
            3x^2-y^2 xyz yz^2 xz^2 z(x^2-y^2) x(x^2-3y^2)
            """

            R[0, 0] = 1j / np.sqrt(2)
            R[6, 0] = 1j / np.sqrt(2)

            R[1, 1] = 1j / np.sqrt(2)
            R[5, 1] = -1j / np.sqrt(2)

            R[2, 2] = 1j / np.sqrt(2)
            R[4, 2] = 1j / np.sqrt(2)

            R[3, 3] = 1

            R[2, 4] = 1 / np.sqrt(2)
            R[4, 4] = -1 / np.sqrt(2)

            R[1, 5] = 1 / np.sqrt(2)
            R[5, 5] = 1 / np.sqrt(2)

            R[0, 6] = 1 / np.sqrt(2)
            R[6, 6] = -1 / np.sqrt(2)

        return R

    def Spherical2Cubic(self, v: np.ndarray, l: int):
        R = self.RotationMatrix(l)
        Rdag = np.conjugate(np.transpose(R))

        tempmat = np.einsum("ab,cd,bdeg,ef,gh", Rdag, Rdag, v, R, R)
        tempmat = np.real(tempmat)

        V = np.array(tempmat, dtype=float, order="F")

        return V

    def GetUijklComCTQMC(self):
        pass


# class Green(object):

#     def __init__(self,greenbare : GreenBare = None, greenint : GreenInt = None, greenloc : GreenLoc = None, greenimp : GreenImp = None):

#         self.greenbare = None
#         self.greenimp = None
#         self.greenint = None
#         self.greenloc = None
#         self.chem = None
#         self.glatkt = None
#         self.glatkf = None
#         self.glatrt = None
#         self.glatrf = None
#         self.glockt = None
#         self.glockf = None
#         self.glocrt = None
#         self.glocrf = None
#         self.gimpkt = None
#         self.gimpkf = None
#         self.gimprt = None
#         self.gimprf = None


#         if greenbare != None:
#             self.greenbare = greenbare
#             self.glatkf = self.greenbare.g0kf
#             self.glatkt = self.greenbare.g0kt
#             self.glatrt = self.greenbare.g0rt
#             self.glatrf = self.greenbare.g0rf
#         if greenint != None:
#             self.greenint = greenint
#             self.glatkf = self.greenint.gkf
#             self.glatkt = self.greenint.gkt
#             self.glatrt = self.greenint.grt
#             self.glatrf = self.greenint.grf
#         if greenloc != None:
#             self.greenloc = greenloc
#             self.glockf = self.greenloc.gkf
#             self.glockt = self.greenloc.gkt
#             self.glocrt = self.greenloc.grt
#             self.glocrf = self.greenloc.grf
#         if greenimp != None:
#             self.greenimp = greenimp
#             self.gimpkf = self.greenimp.gkf
#             self.gimpkt = self.greenimp.gkt
#             self.gimprt = self.greenimp.grt
#             self.gimprf = self.greenimp.grf

#     def SearchMu(self):

#         mumin = -40
#         mumax = 40

#         sol = scipy.optimize.brentq(self.NumOfE,mumin,mumax)

#         return sol

#     def NumOfE(self,mu : float):

#         self.MakeMu(mu)
#         if self.greenint != None:
#             green = self.greenint
#             gkt = green.gkt
#             gkf = green.gkf
#         else:
#             if self.greenbare != None:
#                 green = self.greenbare
#                 gkt = green.g0kt
#                 gkf = green.g0kf
#             else:
#                 print("Error, This class does not have FLatDyn Green'function quantity.")
#                 sys.exit()

#         norb = len(green.crystal.find)
#         ns = green.crystal.ns
#         nrk = len(green.crystal.kpoint)
#         nft = green.ft.size
#         gcalf = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         gcalt = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         chem = -self.chem
#         gcalf = DiagE.dyson.flatdyn(gkf,chem)
#         gcalt = green.F2T(gcalf,1,1)

#         Ne = 0

#         for irk in range(nrk):
#             for js in range(ns):
#                 for iorb in range(norb):
#                     Ne += -np.real(gcalt[iorb,iorb,js,irk,-1])

#         Ne /= nrk
#         N = green.crystal.nume


#         return N - Ne

#     def MakeMu(self,mu : float):

#         norb = self.greenbare.g0kf.shape[0]
#         ns = self.greenbare.g0kf.shape[2]
#         nrk = self.greenbare.g0kf.shape[3]
#         nft = self.greenbare.g0kf.shape[4]

#         chem = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')

#         for ift in range(nft):
#             for irk in range(nrk):
#                 for js in range(ns):
#                     for iorb in range(norb):
#                         chem[iorb,iorb,js,irk,ift] = mu
#         self.chem = chem

#         return None

#     def UpdateMu(self,mu : float):# Embedding chemical potential for using dyson

#         norb = self.glatkf.shape[0]
#         ns = self.glatkf.shape[2]
#         nk = self.glatkf.shape[3]
#         nfreq = self.glatkf.shape[4]

#         self.MakeMu(mu)

#         tempmat = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,order='F')

#         tempmat = self.greenbare.Dyson(self.glatkf,-self.chem)

#         self.glatkf = tempmat
#         self.glatkt = self.greenbare.F2T(tempmat,1,1)

#         return None

# class SigmaC(object): # Various quantity name : SigmaC -> Sigma

#     def __init__(self,sigmahartree : SigmaHartree = None, sigmafock : SigmaFock = None, sigmagwc : SigmaGWC = None, sigmac : object= None, sigmastc : SigmaStc = None, zfactor : ZFactor = None) -> object:

#         self.sigmahartree = None
#         self.sigmafock = None
#         self.sigmagwc = None
#         self.sigmastc = None
#         self.zfactor = None
#         if sigmahartree != None:
#             self.sigmahartree = sigmahartree
#         if sigmafock != None:
#             self.sigmafock = sigmafock
#         if sigmagwc != None:
#             self.sigmagwc = sigmagwc
#         if sigmastc != None:
#             self.sigmastc = sigmastc
#         if zfactor != None:
#             self.zfactor = zfactor

#         self.sigmahfdyn = self.sigmahartree.hdyn + self.sigmafock.fdyn

#     # def __init__(self,ft : FT_grid):
#     #     self.ft = ft
#         norb = self.sigmahfdyn.shape[0]
#         ns = self.sigmahfdyn.shape[2]
#         nrk = self.sigmahfdyn.shape[3]
#         nft = self.sigmahfdyn.shape[4]
#         self.crtau = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         self.cktau = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         self.ckfreq = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         self.crfreq = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         if sigmac == None:
#             self.cmkfreq = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#             self.cmktau = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#             self.cmrfreq = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#             self.cmrtau = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         else:
#             self.cmkfreq = sigmac.cmkfreq
#             self.cmktau = sigmac.cmktau
#             self.cmrfreq = sigmac.cmrfreq
#             self.cmrtau = sigmac.cmrtau


#         self.Cal()

#     def Cal(self):

#         if self.sigmagwc == None:
#             self.ckfreq = self.sigmahfdyn
#             self.cktau = self.sigmahfdyn
#             self.crfreq = self.sigmahartree.K2R(self.ckfreq)
#             self.crtau = self.crfreq
#         if self.sigmagwc != None:
#             self.ckfreq = self.sigmahfdyn + self.sigmagwc.gwckf
#             self.cktau = self.sigmagwc.F2T(self.ckfreq,0,1)
#             self.crfreq = self.sigmagwc.K2R(self.ckfreq)
#             self.crtau = self.sigmagwc.K2R(self.cktau)

#         return None

#     def Mixing(self,iter : int,mix : float):

#         if self.sigmagwc == None:
#             norb = len(self.sigmahartree.crystal.find)
#             ns = self.sigmahartree.crystal.ns
#             nrk = len(self.sigmahartree.crystal.kpoint)
#             nft = self.ckfreq.shape[4]
#             sigmamix = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#             for ift in range(nft):
#                 sigmamix[...,ift] = self.sigmahartree.Mixing(iter,mix,self.ckfreq[...,ift],self.cmkfreq[...,ift])

#             self.cmkfreq = sigmamix
#             self.cmrfreq = self.sigmahartree.K2R(sigmamix)
#         else:
#             sigmamix = self.sigmagwc.Mixing(iter,mix,self.ckfreq,self.cmkfreq)
#             # print(sigmamix)
#             self.cmkfreq = sigmamix
#             self.cmktau = self.sigmagwc.F2T(sigmamix,0,1)
#             self.cmrfreq = self.sigmagwc.K2R(sigmamix)
#             self.cmrtau = self.sigmagwc.K2R(self.cmktau)

#         return None

# class SigmaC(FLatDyn): # Various quantity name : SigmaC -> Sigma

#     def __init__(self,sigmagwc : SigmaGWC = None, sigmac : object= None, sigmastc : SigmaStc = None, zfactor : ZFactor = None) -> object:

#         self.sigmahartree = None
#         self.sigmafock = None
#         self.sigmagwc = None
#         self.sigmastc = None
#         self.zfactor = None
#         if sigmahartree != None:
#             self.sigmahartree = sigmahartree
#         if sigmafock != None:
#             self.sigmafock = sigmafock
#         if sigmagwc != None:
#             self.sigmagwc = sigmagwc
#         if sigmastc != None:
#             self.sigmastc = sigmastc
#         if zfactor != None:
#             self.zfactor = zfactor

#         norb = self.sigmahfdyn.shape[0]
#         ns = self.sigmahfdyn.shape[2]
#         nrk = self.sigmahfdyn.shape[3]
#         nft = self.sigmahfdyn.shape[4]
#         self.crtau = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         self.cktau = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         self.ckfreq = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         self.crfreq = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         if sigmac == None:
#             self.cmkfreq = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#             self.cmktau = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#             self.cmrfreq = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#             self.cmrtau = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#         else:
#             self.cmkfreq = sigmac.cmkfreq
#             self.cmktau = sigmac.cmktau
#             self.cmrfreq = sigmac.cmrfreq
#             self.cmrtau = sigmac.cmrtau


#         self.Cal()

#     def Cal(self):

#         if self.sigmagwc == None:
#             self.ckfreq = self.sigmahfdyn
#             self.cktau = self.sigmahfdyn
#             self.crfreq = self.sigmahartree.K2R(self.ckfreq)
#             self.crtau = self.crfreq
#         if self.sigmagwc != None:
#             self.ckfreq = self.sigmahfdyn + self.sigmagwc.gwckf
#             self.cktau = self.sigmagwc.F2T(self.ckfreq,0,1)
#             self.crfreq = self.sigmagwc.K2R(self.ckfreq)
#             self.crtau = self.sigmagwc.K2R(self.cktau)

#         return None

#     def Mixing(self,iter : int,mix : float):

#         if self.sigmagwc == None:
#             norb = len(self.sigmahartree.crystal.find)
#             ns = self.sigmahartree.crystal.ns
#             nrk = len(self.sigmahartree.crystal.kpoint)
#             nft = self.ckfreq.shape[4]
#             sigmamix = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')
#             for ift in range(nft):
#                 sigmamix[...,ift] = self.sigmahartree.Mixing(iter,mix,self.ckfreq[...,ift],self.cmkfreq[...,ift])

#             self.cmkfreq = sigmamix
#             self.cmrfreq = self.sigmahartree.K2R(sigmamix)
#         else:
#             sigmamix = self.sigmagwc.Mixing(iter,mix,self.ckfreq,self.cmkfreq)
#             # print(sigmamix)
#             self.cmkfreq = sigmamix
#             self.cmktau = self.sigmagwc.F2T(sigmamix,0,1)
#             self.cmrfreq = self.sigmagwc.K2R(sigmamix)
#             self.cmrtau = self.sigmagwc.K2R(self.cmktau)

#         return None


class CorrelationFunction(object):
    def __init__(
        self, latt, basisposition, ns, soc, rkgrid, orboption, N, impdict=None
    ):
        self.green = None
        self.sigmah = None
        self.sigmaf = None
        self.sigmagwc = None
        self.ham = None
        self.hamtb = None
        self.hamhf = None
        self.hamqp = None
        self.occ = None
        self.vbare = None
        cry = Crystal(
            latt=latt,
            basisposition=basisposition,
            ns=ns,
            soc=soc,
            rkgrid=rkgrid,
            orboption=orboption,
            N=N,
        )
        self.cry = cry

    def TighBinding(self, hoppinglist: list, onsitelist: list):
        niham = NIHamiltonian(self.cry, hoppinglist, onsitelist)
        self.hamtb = niham.hamtb
        return niham.hamtb

    def HartreeFockH(
        self,
        itermax: int,
        mix: float,
        T: float,
        size: int,
        hoppinglist: list,
        onsitelist: list,
        option: dict = None,
        intamp: list = None,
    ):
        cry = self.cry
        niham = NIHamiltonian(
            crystal=cry, hoppinglist=hoppinglist, onsitelist=onsitelist
        )
        self.hamtb = niham.hamtb
        ft = FT_grid(T=T, size=size)
        vbare = VBare(crystal=cry, orboption=option, intamp=intamp)
        self.ft = ft
        self.vbare = vbare
        self.hamtb = niham.hamtb
        self.vbare = vbare

        for iter in range(1, itermax + 1):
            if iter == 1:
                hold = Hamiltonian(crystal=cry, ham=niham.hamtb, beta=ft.beta)
                hold.SearchMu()
                hkold = None
                fkold = None
            print(hold.occ)
            sigmah = SigmaHartree(crystal=cry, occ=hold.occ, vbare=vbare)
            sigmaf = SigmaFock(crystal=cry, occr=hold.occr, vbare=vbare)
            sigmah.hk = sigmah.Mixing(iter, mix, sigmah.hk, hkold)
            sigmaf.fk = sigmaf.Mixing(iter, mix, sigmaf.fk, fkold)
            # print(sigmah.hk[:,:,0,0])
            # print(sigmaf.fk[:,:,0,0])
            hnew = Hamiltonian(
                crystal=cry,
                ham=self.TighBinding(hoppinglist=hoppinglist, onsitelist=onsitelist),
                beta=ft.beta,
                sigmah=sigmah,
                sigmaf=sigmaf,
            )
            print(self.hamtb[:, :, 0, 0])
            hnew.SearchMu()

            fcheck = self.FermionSCF(hnew.occk, hold.occk)
            mucheck = abs(hnew.mu - hold.mu)
            print(
                f" iteration : {iter} \n criteria : {fcheck} \n chemicalpotential : {hnew.mu}"
            )
            if (fcheck <= 1.0e-4) and (mucheck <= 0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.ham = hnew
                self.sigmah = sigmah
                self.sigmaf = sigmaf
                return hnew.hk, sigmah, sigmaf
            elif iter == itermax:
                print(
                    f"Notice: Broadening schemes will be turned off from the {iter}-th iteration."
                )
                self.ham = hnew
                self.sigmah = sigmah
                self.sigmaf = sigmaf
                return hnew.hk, sigmah, sigmaf
            else:
                hold = hnew
                hkold = sigmah.hk
                fkold = sigmaf.fk
                del sigmah, sigmaf, hnew

    def HartreeFock(
        self,
        itermax: int,
        mix: float,
        T: float,
        size: int,
        hoppinglist: list,
        onsitelist: list,
        option: dict = None,
        intamp: list = None,
    ):
        cry = self.cry
        niham = NIHamiltonian(
            crystal=cry, hoppinglist=hoppinglist, onsitelist=onsitelist
        )
        ft = FT_grid(T=T, size=size)
        gbare = GreenBare(crystal=cry, ft=ft, niham=niham)
        vbare = VBare(crystal=cry, orboption=option, intamp=intamp)
        self.hamtb = niham.hamtb
        self.vbare = vbare

        for iter in range(1, itermax + 1):
            if iter == 1:
                gold = GreenInt(crystal=cry, ft=ft, greenbare=gbare)
                gold.SearchMu()
                # sigmahold = SigmaHartree(crystal=cry,occ=gold.occ,vbare=vbare)
                # sigmafold = SigmaFock(crystal=cry,occr=gold.occr,vbare=vbare)
                hkold = None
                fkold = None
            print(gold.occ)
            sigmah = SigmaHartree(crystal=cry, occ=gold.occ, vbare=vbare)
            sigmaf = SigmaFock(crystal=cry, occr=gold.occr, vbare=vbare)
            hk = sigmah.Mixing(iter, mix, sigmah.hk, hkold)
            fk = sigmaf.Mixing(iter, mix, sigmaf.fk, fkold)
            print(sigmah.hk[:, :, 0, 0])
            print(hk[:, :, 0, 0])
            print(sigmaf.fk[:, :, 0, 0])
            print(fk[:, :, 0, 0])
            sigmah.hk = hk
            sigmaf.fk = fk
            print(sigmah.hk[:, :, 0, 0])
            print(sigmaf.fk[:, :, 0, 0])
            gnew = GreenInt(
                crystal=cry, ft=ft, greenbare=gbare, sigmah=sigmah, sigmaf=sigmaf
            )
            gnew.SearchMu()

            fcheck = self.FermionSCF(gnew.occk, gold.occk)
            mucheck = abs(gnew.mu - gold.mu)
            print(
                f" iteration : {iter} \n criteria : {fcheck} \n chemicalpotential : {gnew.mu}"
            )
            if (fcheck <= 1.0e-3) and (mucheck <= 0.001):
                print(f"Self-consistency is achived with {iter}-th")
                self.green = gnew
                self.sigmah = sigmah
                self.sigmaf = sigmaf
                chem = niham.ChemEmbedding(gnew.mu)
                self.hamhf = niham.hamtb + sigmah.hk + sigmaf.fk - chem
                break
            elif iter == itermax:
                print(
                    f"Notice: Broadening schemes will be turned off from the {iter}-th iteration."
                )
                self.green = gnew
                self.sigmah = sigmah
                self.sigmaf = sigmaf
                chem = niham.ChemEmbedding(gnew.mu)
                self.hamhf = niham.hamtb + sigmah.hk + sigmaf.fk  # - chem
            else:
                gold = gnew
                sigmahold = sigmah
                sigmafold = sigmaf
                hkold = sigmah.hk
                fkold = sigmaf.fk
                del sigmah, sigmaf, gnew

    def GWApproximation(
        self,
        itermax: int,
        mix: float,
        T: float,
        size: int,
        hoppinglist: list,
        onsitelist: list,
        option: dict,
        intamp: list,
    ):
        cry = self.cry
        ft = FT_grid(T, size)
        niham = NIHamiltonian(cry, hoppinglist, onsitelist)
        gbare = GreenBare(crystal=cry, ft=ft, niham=niham)
        vbare = VBare(crystal=cry, orboption=option, intamp=intamp)

        for iter in range(1, itermax + 1):
            if iter == 1:
                greenold = Green(greenbare=gbare)
                muold = 0
                sigmaold = None
                occold = Occ(cry, greenold)
            sigmah = SigmaHartree(cry, greenold, vbare)
            sigmaf = SigmaFock(cry, greenold, vbare)
            pol = PolLat(cry, ft, greenold)
            w = WLat(cry, ft, pol, vbare)
            sigmagwc = SigmaGWC(cry, ft, greenold, w)
            sigma = SigmaC(
                sigmahartree=sigmah,
                sigmafock=sigmaf,
                sigmagwc=sigmagwc,
                sigmac=sigmaold,
            )
            sigma.Mixing(iter, mix)
            gint = GreenInt(cry, ft, gbare, sigma)
            green = Green(greenbare=gbare, greenint=gint)
            mu = green.SearchMu()
            green.UpdateMu(mu)
            occ = Occ(cry, green)

            # check = self.FermionSCF(green.glatkf,greenold.glatkf)
            check = self.FermionSCF(occ.occk, occold.occk)

            print(
                f"iteration : {iter} \n criteria : {check} \n chemicalpotential : {mu}"
            )

            if (check <= 1.0e-3) and (abs(mu - muold) <= 0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.green = green
                sigmaold = sigma
                self.occ = occ
                sigmastc = SigmaStc(cry, sigma)
                zfactor = ZFactor(cry, sigma)
                sigma = SigmaC(
                    sigmahartree=sigmah,
                    sigmafock=sigmaf,
                    sigmagwc=sigmagwc,
                    sigmac=sigmaold,
                    sigmastc=sigmastc,
                    zfactor=zfactor,
                )
                self.fock = sigma.sigmafock
                self.hartree = sigma.sigmahartree
                self.sigmac = sigma
                hamqp = QPHamiltonian(cry, niham, sigma, mu)
                self.hamqp = hamqp
            elif iter == itermax:
                print(
                    f"Notice: Broadening schemes will be turned off from the {iter}-th iteration."
                )
                self.green = green
                sigmaold = sigma
                self.occ = occ
                sigmastc = SigmaStc(cry, sigma)
                zfactor = ZFactor(cry, sigma)
                sigma = SigmaC(
                    sigmahartree=sigmah,
                    sigmafock=sigmaf,
                    sigmagwc=sigmagwc,
                    sigmac=sigmaold,
                    sigmastc=sigmastc,
                    zfactor=zfactor,
                )
                self.fock = sigma.sigmafock
                self.hartree = sigma.sigmahartree
                self.sigmac = sigma
                hamqp = QPHamiltonian(cry, niham, sigma, mu)
                self.hamqp = hamqp
            else:
                greenold = green
                sigmaold = sigma
                occold = occ
                muold = mu

    def DMFT(
        self,
        itermax: int,
        hmat: np.ndarray,
        U4_index: np.ndarray,
        N: float,
        mix: float,
        T: float,
        size: int,
        equiv: list,
    ):
        """
        :param int itermax:          Max number of DMFT solver calls
        :param np.ndarray hmat:      Noninteracting, local, Hamiltonian
        :param  np.ndarray U4_index: Coulomb U matrix, 4-index
        :param float N:              Number of electrons?
        :param float mix:            Linear mixing parameter for self-energy mixing
        :param float T:              Electronic temperature
        :param int size:             Number of grid points for Matsubara mesh and imaginary time mesh
        :param list equiv:           List equivalent orbitals in impurity problem
        """
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        ft = FT_grid(T=T, size=size)
        nimp = len(self.cry.probspace)

        G_bare = GreenBare(self.cry, ft, hmat)
        G_int = GreenInt(self.cry, ft, G_bare)
        G_loc = GreenLoc(self.cry, ft, G_int)
        G_int.Occ()
        E_imp = ImpurityLevel(self.cry, hmat, G_int.mu)
        # Actually, only pick inequivalent sites, not all sites in G_loc...
        G_imp_mat = G_loc.Loc2Imp(G_loc.gf)
        E_imp_mat = E_imp.Loc2Imp(E_imp.loc)
        Sig_imp_mat = np.zeros_like(G_imp_mat)
        Sig_loc = SigmaLoc(self.cry, ft, np.zeros_like(G_int.gbare.g0kf))
        G_imp = np.empty_like(norb, norb, ns, nf, nimp, dtype=complex)
        Sig_imp = np.empty_like(norb, norb, ns, nf, nimp, dtype=complex)
        for it in range(itermax):
            hybridisation = Hybridisation(
                self.cry, self.ft, E_imp_mat, G_loc.Loc2Imp(G_loc.gf), Sig_imp_mat
            )
            mu_old = G_int.mu
            occ_old = G_int.occ
            
            for imp in self.cry.probspace:
                impdir = f"impurity/{imp}"
                if not os.path_exists(impdir):
                    os.mkdir(impdir)
                os.chdir(impdir)
                G_imp[..., imp], Sig_imp[..., imp] = self.RunImpurityAction(
                    self.ft.beta, E_imp, hybridisation, U4_index, equiv
                )
                cwd = os.getcwd()

            G_loc.gf = G_loc.Imp2Loc(G_imp.imp)
            Sig_loc.f = Sig_loc.Mixing(
                iter=it, mix=mix, Fb=Sig_loc.Imp2Loc(Sig_imp.imp), Fold=Sig_loc.f
            )
            Sig_loc.t = Sig_loc.Mixing(
                iter=it, mix=mix, Fb=Sig_loc.Imp2Loc(Sig_imp.imp), Fold=Sig_loc.t
            )
            G_int = GreenInt(G_bare, np.zeros(norb, norb, ns, nspace, dtype=complex), np.zeros(norb, norb, ns, nspace, dtype=complex), Sig_loc.Embedding(Sig_loc.f))
            G_int.SearchMu()
            G_int.Occ()
            occ = G_int.occ
            if abs(G_int.mu - mu_old) < 1e-2 or self.FermionSCF(occ_old, occ) < 1e-3:
                break
            G_loc = GreenLoc(self.cry, ft, G_int)

        return None

    def FermionSCF(self, mat1: np.ndarray, mat2: np.ndarray):
        check = 0
        tempmat = abs(mat1 - mat2)
        check = tempmat.max()

        return check

    def RunImpurityAction(
        self,
        beta: float,
        E_imp: ImpurityLevel,
        hybridisation: Hybridisation,
        U4: np.ndarray,
        equiv: list,
    ):
        self.WriteParamsJson(beta, equiv, E_imp.ndarray, U4, hybridisation.ndarray)
        self.RunCTQMC()
        return self.MeasureCTQMC()

    def RunCTQMC(self):
        # run_cmd = 'mpirun -np 1 '+diage_path+'/ComCTQMC/bin/CTQMC params'
        run_cmd = "mpirun -np 4 " + diage_path + "/ComCTQMC/bin/CTQMC params"
        print(run_cmd)

        with open("./ctqmc.out", "w") as logfile, open("./ctqmc.err", "w") as errfile:
            ret = subprocess.call(run_cmd, shell=True, stdout=logfile, stderr=errfile)
            if ret != 0:
                print("Error in CTQMC. Check ctqmc.err for error message.")
                sys.exit()

        return None

    def MeasureCTQMC(self):
        run_cmd = "mpirun -np 4 " + diage_path + "/ComCTQMC/bin/EVALSIM params"

        print(run_cmd)
        with open("./evalsim.out", "w") as logfile, open(
            "./evalsim.err", "w"
        ) as errfile:
            ret = subprocess.call(run_cmd, shell=True, stdout=logfile, stderr=errfile)
            if ret != 0:
                print("Error in EVALSIM. Check evalsim.err for error message.")
                sys.exit()
        print("measure self-energy done")

        obs_json = "params.obs.json"
        observables = json.load(open(obs_json))
        observables = observables["partition"]
        green_dict = {}
        for key, val in observables["green"].items():
            green_dict[key] = []
            for real, imag in zip(val["function"]["real"], val["function"]["imag"]):
                green_dict[key].append(real + 1j * imag)
        sig_bare_dict = {}
        sig_hf_dict = {}
        for key, val in observables["self-energy"].items():
            sig_hf_dict[key] = complex(val["moments"][0])
            sig_bare_dict["key"] = []
            for real, imag in zip(val["function"]["real"], val["function"]["imag"]):
                sig_bare_dict[key].append(real + 1j * imag)

        return GreenImp(self.cry, self.ft, green_dict), SigmaImp(
            self.cry, self.ft, sig_bare_dict
        )

    def WriteParamsJson(self, beta, equiv, E_imp, U4, hyb, SOC=False):
        """
        Write parameters for running ComDMFT to a json archive
        :paramn float beta: Inverse temperature
        :param list equiv:  List of equivalent impurity orbitals
        :E_imp np.ndarray: Impurity levels
        :param np.ndarray U4: 4-index Coulomb matrix
        :param np.ndarray hyb: Hybridisation function, ndarray with dimensions (norb, norb, ns, nfreq, nimp)
        :param bool SOC: Include Spin orbit coupling? (default: False)
        """
        norb, _, ns, _ = hyb.shape
            self.WriteHyb(hyb[..., imp], beta, SOC)
            if not SOC:
                if ns == 1:
                    params = {}
                    params["hloc"] = {}
                    mu_ctqmc = -np.real(E_imp[0, 0, 0])
                    # print(mu_ctqmc,type(mu_ctqmc))
                    E_imp = E_imp[:, :, 0] + mu_ctqmc * np.eye(
                        E_imp.shape[0], E_imp.shape[0]
                    )
                    E_imp = np.array(np.real(E_imp), dtype=float)
                    tempmat = np.kron(E_imp, np.eye(2, 2))
                    params["hloc"]["one body"] = tempmat.tolist()
                    # self.boson.get_Uijkl_comctqmc(key)
                    params["hloc"]["two body"] = U4  # self.boson.U_ctqmc.tolist()
                    # params["hloc"]["two body"] = {}
                    # params["hloc"]["two body"]["parametrisataion"] = "slater-condon"
                    # params["hloc"]["two body"]["F0"]=5.0
                    # params["hloc"]["two body"]["F2"]=0.0
                    # params["hloc"]["two body"]["F4"]=0.0
                    # params["hloc"]["two body"]["approximation"] = "none"

                    params["partition"] = {}

                    params["partition"]["green basis"] = "matsubara"
                    params["partition"]["green bulla"] = True
                    params["partition"]["green matsubara cutoff"] = 50
                    params["partition"]["occupation susceptibility bulla"] = True
                    params["partition"]["occupation susceptibility direct"] = False
                    params["partition"]["quantum number susceptibility"] = True
                    params["partition"]["susceptibility cutoff"] = 50
                    params["partition"]["susceptibility tail"] = 200
                    params["partition"]["quantum numbers"] = {}
                    tempmat = np.ones(E_imp.shape[0] * 2)
                    params["partition"]["quantum numbers"]["N"] = tempmat.tolist()
                    tempmat[:norb] *= 0.5
                    tempmat[norb:] *= -0.5
                    # [1,1,1,1,1,1,1,1,1,1]
                    # for ii in range(len(tempmat)):
                    #     if ii < E_imp.shape[0]:
                    #         tempmat[ii] *= 0.5
                    #     elif ii >= E_imp.shape[0]:
                    #         tempmat[ii] *= -0.5
                    params["partition"]["quantum numbers"][
                        "Sz"
                    ] = tempmat.tolist()  # make
                    # [0.5,0.5,0.5,0.5,0.5,-0.5,-0.5,-0.5,-0.5,-0.5]
                    # params["partition"]["observables"]={}
                    # params["partition"]["observables"]["S2"] = {}
                    params["partition"]["probabilities"] = {}
                    params["partition"]["probabilities"] = [
                        "N",
                        "energy",
                        "Sz",
                    ]  # ["N","energy","S2","Sz"]
                    params["partition"]["density matrix precise"] = True
                    params["partition"]["print eigenstates"] = True
                    params["partition"]["print density matrix"] = True

                    # params["dyn"]={}
                    # params["dyn"]["quantum numbers"] = np.ones(E_imp.shape[0]*2).tolist()
                    # # [[1,1,1,1,1,1,1,1,1,1]]
                    # params["dyn"]["functions"] = "dyn.json"
                    # params["dyn"]["matrix"] = [["F0"]]
                    params["beta"] = beta
                    params["complex"] = False
                    params["mu"] = mu_ctqmc
                    params["hybridisation"] = {}
                    # tempmat2 = np.kron(equiv,np.ones((2,2)))
                    tempmat2 = np.kron(equiv, np.eye(2, 2))
                    tempmat2 = tempmat2.tolist()
                    for ii in range(len(tempmat2)):
                        for jj in range(len(tempmat2)):
                            if tempmat2[ii][jj] == 0.0:
                                tempmat2[ii][jj] = ""
                            else:
                                tempmat2[ii][jj] = str(int(tempmat2[ii][jj]))

                    params["hybridisation"]["matrix"] = tempmat2
                    params["hybridisation"]["functions"] = "hyb.json"
                    params["thermalisation time"] = 1  # imp['thermalization_time']
                    params["quantum number susceptibility"] = True
                    params["occupation susceptibility bulla"] = True
                    params["green bulla"] = True
                    params["density matrix precise"] = False  # True
                    params["measurement time"] = 3  # imp['measurement_time']

                    # with open(f"params.{it}.{imp}.json", "w") as outfile:
                    #     json.dump(
                    #         params,
                    #         outfile,
                    #         sort_keys=True,
                    #         indent=4,
                    #         separators=(",", ": "),
                    #     )
                    with open("params.json", "w") as outfile:
                        json.dump(
                            params,
                            outfile,
                            sort_keys=True,
                            indent=4,
                            separators=(",", ": "),
                        )
                    # print("params.json written", file=self.m_ini.control['h_log'])
                elif ns == 2:
                    raise NotImplementedError
                    print("Nspin is not 1")
                    sys.exit()
            elif SOC:
                raise NotImplementedError
                print("SOC is not  False, please change SOC")
                sys.exit()

        return None

    def WriteHyb(self, hyb, beta, SOC=False):
        """
        Write hybridisation function to json file, so that CTQMC can read it.
        :param np.ndarray hyb: Hybridisation function, ndarray with dimensions (norb, norb, ns, nfreq, nimp)
        :paramn beta float: Inverse temperature
        :param bool SOC: Include Spin orbit coupling? (default: False)
        """
        assert len(hyb.shape) == 5
        _, _, ns, _ = hyb.shape
        if not SOC:
            if ns == 1:
                json_dict = {}
                for key, val in hyb.items():
                    json_dict[key] = {}
                    for imp in range(nimp):
                        json_dict[str(imp)]["beta"] = beta
                        json_dict[str(imp)]["real"] = np.real(hyb[..., imp]).tolist()
                        json_dict[str(imp)]["imag"] = np.imag(hyb[..., imp]).tolist()

                # with open(f"hyb.{it}.{key}.json", "w") as outfile:
                #     json.dump(
                #         json_dict[imp],
                #         outfile,
                #         sort_keys=True,
                #         indent=4,
                #         separators=(",", ": "),
                #     )
                with open("hyb.json", "w") as outfile:
                    json.dump(
                        json_dict,
                        outfile,
                        sort_keys=True,
                        indent=4,
                        separators=(",", ": "),
                    )

            elif ns == 2:
                raise NotImplementedError
                print("Nspin is not 1")
                sys.exit()
        elif SOC is True:
            raise NotImplementedError
            print("SOC must be False")
            sys.exit()
        return None
