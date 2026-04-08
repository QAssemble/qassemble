import copy
import itertools
import logging
import sys

import h5py
import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.optimize

from .Crystal import Crystal
from .utility.Common import Common
from .utility.Dyson import Dyson
from .utility.Fourier import Fourier
from .utility.Mixing import Mixing

logger = logging.getLogger("QAssemble")



class FLatStc(object):

    def __init__(self, crystal: Crystal, nodedict : dict = None, mixing_method: str = "pulay", npulay: int = 5):

        self.crystal = crystal
        self.nodedict = nodedict
        self._mixer = Mixing(method=mixing_method, npulay=npulay)

    def Inverse(self, mat: np.ndarray):

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]

        matinv = np.zeros((norb, norb, ns, nrk), dtype=np.complex128)

        for irk in range(nrk):
            for js in range(ns):
                matinv[:, :, js, irk] = Common.MatInv(mat[:, :, js, irk])

        return matinv

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

    def _slice_local_kr(self, mat: np.ndarray, loc2glob_key: str) -> np.ndarray:

        if self.nodedict is None:
            return mat

        rank = self.nodedict["commk"].Get_rank()
        nloc = len(self.nodedict[loc2glob_key][rank])
        idx = self._kr_global_indices(loc2glob_key, nloc)

        return np.asfortranarray(np.take(mat, idx, axis=-1))

    def K2R(self, matk: np.ndarray = None) -> np.ndarray:

        
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[3]

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

                        # matk[iorb,jorb,js,irk] *= phase
                        tempmat[iorb, jorb, js, irk] *= phase

        # matr = QAFort.fourier.flatstc_k2r(rkgrid, tempmat)
        if self.nodedict is not None:
            from .utility.Fourier import FourierMPI as Fourier
            matr = Fourier.FLatStcK2R(tempmat, self.nodedict)
        else:
            from .utility.Fourier import Fourier
            matr = Fourier.FLatStcK2R(tempmat, rkgrid)

        return matr

    def R2K(self, matr: np.ndarray = None) -> np.ndarray:

        
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[3]

        matk = np.zeros((norb, norb, ns, nrk), dtype=np.complex128)
        tempmat = copy.deepcopy(matr)
        # matk = QAFort.fourier.flatstc_r2k(rkgrid, tempmat)

        if self.nodedict is not None:
            from .utility.Fourier import FourierMPI as Fourier
            matk = Fourier.FLatStcR2K(tempmat, self.nodedict)
        else:
            from .utility.Fourier import Fourier
            matk = Fourier.FLatStcR2K(tempmat, rkgrid)

        nkout = matk.shape[3]
        kglob = self._kr_global_indices("kloc2glob", nkout)

        for irk in range(nkout):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        [a, m1] = self.crystal.FAtomOrb(iorb)
                        [b, m2] = self.crystal.FAtomOrb(jorb)

                        delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                        phase = np.exp(-2.0j * np.pi * np.dot(rkvec[kglob[irk]], delta))

                        matk[iorb, jorb, js, irk] = matk[iorb, jorb, js, irk] * phase

        return matk

    def Band(
        self,
        energy: np.ndarray,
        fn: str = None,
        plotoption: bool = False,
        label: list = None,
    ):

        norb = energy.shape[0]
        ns = energy.shape[2]
        nk = energy.shape[3]

        energyplot = np.zeros((norb, ns, nk), dtype=float)

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    energyplot[iorb, js, ik] = energy[iorb, iorb, js, ik]
        if plotoption:
            if self.crystal.ns == 1:
                fig, ax = plt.subplots()
                ax.set_xlim(self.crystal.knode[0], self.crystal.knode[-1])
                ax.set_xticks(self.crystal.knode)
                if label == None:
                    pass
                else:
                    ax.set_xticklabels(label)
                for i in range(len(self.crystal.knode)):
                    ax.axvline(
                        x=self.crystal.knode[i],
                        linewidth=0.5,
                        color="r",
                        linestyle="--",
                    )
                for iorb in range(norb):
                    ax.plot(self.crystal.kdist, energyplot[iorb, 0, :].T, "k-")
                ax.set_ylabel("E (eV)")
                ax.set_title("Band")
                # plt.plot(energyplot.T[:,0,:])
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
        else:
            with open("band.dat", "w") as f:
                for js in range(ns):
                    for ik in range(nk):
                        linedata = [self.crystal.kdist[ik]] + energyplot[
                            :, js, ik
                        ].tolist()
                        line = " ".join(map(str, linedata))
                        f.write(line + "\n")

        return None

    def Diagonalize(self, matk: np.ndarray, eigvec: bool = False):

        nk = matk.shape[3]
        norb = matk.shape[0]
        ns = matk.shape[2]

        energy = np.zeros((norb, norb, ns, nk), dtype=float)
        evec = np.zeros((norb, norb, ns, nk), dtype=np.complex128)

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

    def Gaussian(self, x, mu, sigma=0.1):

        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    def DOS(
        self,
        hamr: np.ndarray = None,
        sigma: float = 0.1,
        kgrid: list = [20, 20, 20],
        plotoption: bool = False,
        emax: float = 10,
        emin: float = -10,
    ):

        logger.info("***** DOS Calculation Start *****")
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        if type(kgrid) == list:
            nk = kgrid[0] * kgrid[1] * kgrid[2]
            kpointtemp = np.array(
                list(
                    itertools.product(
                        np.linspace(0, 1, num=kgrid[2], endpoint=False),
                        np.linspace(0, 1, num=kgrid[1], endpoint=False),
                        np.linspace(0, 1, num=kgrid[0], endpoint=False),
                    )
                )
            )
            kpoint = np.fliplr(kpointtemp)
        elif type(kgrid) == np.ndarray:
            nk = len(kgrid)
            kpoint = kgrid

        logger.info("***** Fourier transfrom R2K Start")
        hamk = self.R2KArb(hamr, kpoint)
        logger.info("***** Fourier transfrom R2K Finish")
        logger.info("***** Hamiltonian Diagonalization Start *****")
        (energy, eigvec) = self.Diagonalize(matk=hamk, eigvec=True)
        logger.info("***** Hamiltonian Diagonalization Finish *****")
        emin = emin  # energy[0,0,0].min()
        emax = emax  # energy[-1,-1,0].max()
        energyrange = np.linspace(emin, emax, nk)
        # dos = np.zeros_like(energyrange)
        dos = np.zeros((norb, ns, nk), dtype=float)
        tempmat = np.zeros((norb, ns, nk), dtype=float)

        logger.info("***** Gaussian Approach Start *****")
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    e = energy[iorb, iorb, js, ik]
                    tempmat[iorb, js] += self.Gaussian(energyrange, e, sigma) / nk
        logger.info("***** Gaussian Approach Finish *****")

        for ik in range(nk):
            for js in range(ns):
                tempmat2 = np.linalg.inv(eigvec[:, :, js, ik])
                # tempmat3 = np.array(np.dot(tempmat2,eigvec[:,:,js,ik]),dtype=float)
                D = np.diag(tempmat[:, js, ik])
                tempmat3 = eigvec[:, :, js, ik] @ (D @ tempmat2)
                for iorb in range(norb):
                    dos[iorb, js, ik] = tempmat3[iorb, iorb]
                # for jorb in range(norb):
                #     for iorb in range(norb):
                #         # dos[iorb,js,ik] = tempmat2[iorb,jorb]*tempmat[jorb,js,ik]*eigvec[jorb,iorb,js,ik]
                #         # dos[iorb,js,ik] = tempmat3[iorb,jorb]*tempmat[jorb,js,ik]
                #         dos[iorb,js,ik] = eigvec[jorb,iorb,js,ik]*tempmat[jorb,js,ik]*tempmat2[jorb,iorb]

        logger.debug(
            f"Integration gaussian : {np.trapz(self.Gaussian(energyrange,0),energyrange)}"
        )
        temp = 0
        for js in range(ns):
            for iorb in range(norb):
                temp += np.trapz(dos[iorb, js], energyrange)

        logger.debug(f"Integration dos : {temp}")
        if plotoption:
            fig, ax = plt.subplots()
            ax.set_xlim(energyrange[0], energyrange[-1])
            legend = []
            for js in range(ns):
                for iorb in range(norb):
                    ax.plot(energyrange, dos[iorb, js])
                    legend.append(iorb + 1)
            ax.legend(legend)
            ax.set_xlabel("E (eV)")
            ax.set_ylabel("DOS")
            plt.show()
        else:
            with open("dos.dat", "w") as f:
                for i in range(len(energyrange)):
                    f.write(f"{energyrange[i]}  {dos[i]}")
        logger.info("***** DOS Calculation Finish *****")
        return None

    def Visualization(self, energy: np.ndarray, grid : list = None, fn: str = None):

        if grid is None:
            grid = self.crystal.rkgrid
            kpoint = self.crystal.kpoint
        else:
            kpoint = self.crystal.KPoint(grid)

        # if grid[2] != 1:
        #     print("Energy surface for only 2D case")
        #     sys.exit()
        # else:
            
        norb = energy.shape[0]
        ns = energy.shape[2]
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        kx = kpoint[:, 0].reshape(
            grid[0], grid[1], grid[2]
        )
        ky = kpoint[:, 1].reshape(
            grid[0], grid[1], grid[2]
        )
        energy = energy.T
        energy = energy.reshape(
            grid[0],
            grid[1],
            grid[2],
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
        if iter == 1:
            Fm = np.zeros_like(Fb)
        return self._mixer(iter=iter, mix=mix, Fnew=Fb, Fold=Fm)

    def ChemEmbedding(self, mu: float) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        if hasattr(self, "hkmu0") and self.hkmu0 is not None:
            nrk = self.hkmu0.shape[3]
        else:
            nrk = self._nk_local()

        chem = np.zeros((norb, norb, ns, nrk), dtype=np.complex128)

        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    chem[iorb, iorb, js, irk] = mu

        return chem

    def Dyson(self, mat1: np.ndarray, mat2: np.ndarray):

        # matout = QAFort.dyson.flatstc(mat1, mat2)
        return Dyson.FLatStc(mat1, mat2)

    # def Projection(self, matin: np.ndarray):

    #     norb = len(self.crystal.find)
    #     ns = self.crystal.ns
    #     norbc = self.crystal.fprojector.shape[1]
    #     nspace = self.crystal.fprojector.shape[3]

    #     matout = np.zeros((norbc, norbc, ns, nspace), dtype=np.complex128)

    #     for ispace in range(nspace):
    #         matout[..., ispace] = QAFort.projection.flatstc(
    #             matin, self.crystal.fprojector[..., ispace]
    #         )

    #     return matout

    # def Save(self, matin: np.ndarray, fn: str):

    #     norb = matin.shape[0]
    #     ns = matin.shape[2]
    #     nrk = matin.shape[3]

    #     # if os.path.exists('flatstc'):
    #     #     pass
    #     # else:
    #     #     os.mkdir("flatstc")
    #     # os.chdir("flatstc")
    #     with open(fn + ".txt", "w") as f:
    #         f.write("#iorb, jorb, is, ik, Re(F(k)), Im(F(k))\n")
    #         for irk in range(nrk):
    #             for js in range(ns):
    #                 for jorb in range(norb):
    #                     for iorb in range(norb):
    #                         f.write(
    #                             f"{iorb} {jorb} {js} {irk} {matin[iorb,jorb,js,irk].real} {matin[iorb,jorb,js,irk].imag}\n"
    #                         )
    #     # os.chdir("..")
    #     return None

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
        matk = np.zeros((norb, norb, ns, nk), dtype=complex)

        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        temp = 0
                        for ir in range(nr):
                            temp += tempmat[iorb, jorb, js, ir] * np.exp(
                                -2.0j * np.pi * (kpoint[ik] @ self.crystal.rvec[ir])
                            )
                        [a, m1] = self.crystal.FAtomOrb(iorb)
                        [b, m2] = self.crystal.FAtomOrb(jorb)
                        delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                        phase = np.exp(-2.0j * np.pi * (kpoint[ik] @ delta))
                        matk[iorb, jorb, js, ik] = temp * phase

        return matk

    def HermitianCheck(self, matin: np.ndarray):

        norb, _, ns, nk = matin.shape

        errmessage = "The matrix is not hermitian. Check the input file again"
        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        err = matin[iorb, jorb, js, ik] - np.conjugate(
                            matin[jorb, iorb, js, ik]
                        )
                        if abs(err) > 1.0e-6:
                            logger.error(errmessage)
                            sys.exit()
        return None

    def CheckGroup(self, filepath: str, group: str):

        with h5py.File(filepath, "r") as file:
            return group in file
        
    def SortKpoint(self, kp : np.ndarray, p1, p2):

        kx, ky = kp
        kx1, ky1 = p1
        kx2, ky2 = p2

        return (ky-ky1)*(kx2-kx1) - (kx-kx1)*(ky2-ky1) >= 0
    
    def KValley(self, kgrid : list = None):

        # grid = self.crystal.rkgrid
        if (kgrid is None):
            kgrid = self.crystal.rkgrid
        kplus = []
        kminus = []
        kpoint_temp=np.array(list(itertools.product(np.linspace(0,1,num=kgrid[2],endpoint=False),np.linspace(0,1,num=kgrid[1],endpoint=False),np.linspace(0,1,num=kgrid[0],endpoint=False))))
        kpoint=np.fliplr(kpoint_temp)
        bk = kpoint@self.crystal.bvec
        bk2 = bk.reshape((kgrid[0],kgrid[1],kgrid[2],3))

        for ik in range(len(bk)):
            kp = bk[ik]
            if self.SortKpoint(kp[0:2], (bk2[0, 0, 0, 0], bk2[0, 0, 0, 1]), (bk2[kgrid[0]-1, kgrid[1]-1, 0, 0],bk2[kgrid[0]-1, kgrid[1]-1, 0, 1])):
                kminus.append(ik)
            else:
                kplus.append(ik)

        return (kplus, kminus)


class NIHamiltonian(FLatStc):

    def __init__(
        self, crystal: Crystal = None, nodedict : dict = None, hopping: dict = None, onsite: dict = None, spin : bool = False, 
        ferro : bool = False, aferro : bool = False, valley: bool = False, avalley : bool = False, site : bool = False, 
        asite : bool = False, hdf5file: h5py.File = None, group: str = None,
    ):

        super().__init__(crystal, nodedict=nodedict)

        logger.info("Non-interacting Hamiltonian Calculation Start")
        hopplist = []
        for orb, val in hopping.items():
            for t, lat in val.items():
                for r in lat:
                    hopplist.append([t, list(orb[0]), list(orb[1]), r])

        # print(hopplist)
        self.hopping = hopplist
        self.onsite = onsite
        self.hdf5file = hdf5file
        self.spin = spin
        self.asite = asite
        self.site = site
        self.ferro = ferro
        self.aferro = aferro
        self.group = group
        self.subgroup = self.__class__.__name__
        # print(self.onsite)
        self.k = None
        self.r = None
        # self.Hopping()
        # self.Onsite()

        self.Cal()
        if valley:
            self.Valley()
        
        if avalley:
            self.AntiValley()

        if hdf5file != None:
            self.Save()

        logger.info("Non-interacting Hamiltonian Calculation Finish")

    def Cal(self):  # GenHam

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)

        tempmat = np.zeros(
            (norb, norb, ns, self.crystal.rkgrid[0], self.crystal.rkgrid[1], self.crystal.rkgrid[2]),
            dtype=np.complex128,
        )

        for js in range(ns):
            for hopp in self.hopping:
                tij = hopp[0]
                # iorb = hopp[1]
                # jorb = hopp[2]

                (a, m) = hopp[1]
                (b, mp) = hopp[2]
                iorb = self.crystal.FIndex([a, m])
                jorb = self.crystal.FIndex([b, mp])
                R = hopp[3]

                # tempmat[iorb,jorb,js,R[0],R[1],R[2]] += -tij
                if (iorb == jorb) and (R == [0, 0, 0]):
                    logger.error("Wrong value entered, please check the input.ini file")
                    sys.exit()
                else:
                    tempmat[iorb, jorb, js, R[0], R[1], R[2]] += -tij
                    tempmat[jorb, iorb, js, -R[0], -R[1], -R[2]] += -tij.conjugate()
                    # print(tij,iorb,jorb,R,tempmat[iorb,jorb,js,R[0],R[1],R[2]],tempmat[jorb,iorb,js,-R[0],-R[1],-R[2]])

                # 0 == -0

        if (self.spin): 
                tempmat[0,0,0,0,0,0] += 1
                tempmat[0,0,1,0,0,0] += -1
                tempmat[norb - 1,norb-1,0,0,0,0] += 1
                tempmat[norb - 1,norb-1,1,0,0,0] += -1

        if (self.aferro):
            tempmat[0, 0, 0, 0, 0, 0] += 1
            tempmat[0, 0, 1, 0, 0, 0] += -1
            tempmat[norb-1, norb-1, 0, 0, 0, 0] += -1
            tempmat[norb-1, norb-1, 1, 0, 0, 0] += 1

        if (self.site):
            for js in range(ns):
                tempmat[0, 0, js, 0, 0, 0] += 1     
                # tempmat[1, 1, js, 0, 0, 0] += 1
                # tempmat[norb, -2, norb - 2, js, 0, 0, 0] += 1
                tempmat[norb - 1, norb - 1, js, 0, 0, 0] += 1
                # for iorb in range(norb):
                #     if (iorb % 2 == 0):
                #         tempmat[iorb, iorb, js, 0, 0, 0] += 1
                #     else:
                #         tempmat[iorb, iorb, js, 0, 0, 0] += -1
        if (self.asite):
            for js in range(ns):
                tempmat[0, 0, js, 0, 0, 0] += 1     
                # tempmat[1, 1, js, 0, 0, 0] += 1
                # tempmat[norb, -2, norb - 2, js, 0, 0, 0] += -1
                tempmat[norb - 1, norb - 1, js, 0, 0, 0] += -1
        
        if self.onsite != None:
            for js, value in self.onsite.items():
                for orb, val in value.items():
                    iorb = self.crystal.FIndex(list(orb))
                    tempmat[iorb, iorb, js, 0, 0, 0] += val
        #           for js in range(ns):
        #               for orb, val in self.onsite.items():
        #                   iorb = self.crystal.FIndex(list(orb))
        #                   # jorb = self.crystal.FIndex(list(orb[1]))
        #                   tempmat[iorb,iorb,js,0,0,0] += val
        # print(tempmat[iorb,iorb,js,0,0,0],val)
        # for iorb in range(norb):
        #     tempmat[iorb,iorb,js,0,0,0] = +self.onsitelist[iorb]
        # Hermitian check
        tempmat = np.asfortranarray(tempmat.reshape((norb, norb, ns, nk)))

        self.r = self._slice_local_kr(tempmat, "rloc2glob")
        hamtb = self.R2K(self.r)

        self.HermitianCheck(hamtb)
        self.k = hamtb

        return None

    def Save(self):

        # if os.path.exists('niham'):
        #     pass
        # else:
        #     os.mkdir('niham')
        # os.chdir('niham')
        # os.chdir('work')

        # filepath = 'flatstc.h5'
        # groupname = 'niham'
        # with h5py.File(filepath,'a') as file:
        #     if self.CheckGroup(filepath,groupname):
        #         group = file[groupname]
        #     else:
        #         group=file.create_group(groupname)

        #     group.create_dataset('h0k',dtype=complex,data=self.k)
        # os.chdir('..')
        with h5py.File(self.hdf5file, "a") as file:
            if self.CheckGroup(self.hdf5file, self.group):
                tb = file[self.group]
                if self.subgroup in tb:
                    niham = tb[self.subgroup]
                else:
                    niham = tb.create_group(self.subgroup)
            else:
                tb = file.create_group(self.group)
                niham = tb.create_group(self.subgroup)
            niham.create_dataset("h0k", dtype=complex, data=self.k)
        # self.hdf5file.create_dataset('h0k',dtype=float,data=self.k)

        return None

    def Valley(self):

        # kpoint = self.crystal.kpoint

        h0k = np.copy(self.k)
        norb = h0k.shape[0]
        ns = h0k.shape[2]
        nk = h0k.shape[3]
        # (norb, norb, ns, nk) = h0k.shape
        (kplus, kminus) = self.KValley()

        # for ik in range(nk):
        #     for js in range(ns):
        #         for iorb in range(norb):
        #             if (np.isclose(kpoint[ik], [2 / 3, 1 / 3, 0])).all():
        #                 h0k[iorb, iorb, js, ik] += 1.0
        #             if (np.isclose(kpoint[ik], [1 / 3, 2 / 3, 0])).all():
        #                 h0k[iorb, iorb, js, ik] += -1.0

        for ik in range(nk):
            for js in range(ns):
                if (ik in kplus):
                    h0k[0, 0, js, ik] += 1
                    h0k[1, 1, js, ik] += 1
                    h0k[norb-2, norb-2, js, ik] += 1
                    h0k[norb-1, norb-1, js, ik] += 1
                if (ik in kminus):
                    h0k[0, 0, js, ik] += -1
                    h0k[1, 1, js, ik] += -1
                    h0k[norb-2, norb-2, js, ik] += -1
                    h0k[norb-1, norb-1, js, ik] += -1
                        

        self.k = h0k
        self.r = self.K2R(h0k)

        return None
    
    def AntiValley(self):

        # kpoint = self.crystal.kpoint

        h0k = np.copy(self.k)
        norb = h0k.shape[0]
        ns = h0k.shape[2]
        nk = h0k.shape[3]
        # (norb, norb, ns, nk) = h0k.shape
        (kplus, kminus) = self.KValley()

        # for ik in range(nk):
        #     for js in range(ns):
        #         for iorb in range(norb):
        #             if (np.isclose(kpoint[ik], [2 / 3, 1 / 3, 0])).all():
        #                 h0k[iorb, iorb, js, ik] += 1.0
        #             if (np.isclose(kpoint[ik], [1 / 3, 2 / 3, 0])).all():
        #                 h0k[iorb, iorb, js, ik] += -1.0

        for ik in range(nk):
            for js in range(ns):
                if (ik in kplus):
                    h0k[0, 0, js, ik] += 1
                    h0k[1, 1, js, ik] += 1
                    h0k[norb-2, norb-2, js, ik] += -1
                    h0k[norb-1, norb-1, js, ik] += -1
                if (ik in kminus):
                    h0k[0, 0, js, ik] += -1
                    h0k[1, 1, js, ik] += -1
                    h0k[norb-2, norb-2, js, ik] += 1
                    h0k[norb-1, norb-1, js, ik] += 1
                        

        self.k = h0k
        self.r = self.K2R(h0k)

        return None




    # def Hopping(self):
    #     pass

    # def Onsite(self):
    #     pass


class SigmaHartree(FLatStc):

    def __init__(self,crystal: Crystal,nodedict : dict = None, occ:np.ndarray=None,vbare: np.ndarray = None,hdf5file: str = "glob.h5",group: str = None,):  # green -> occ
        super().__init__(crystal, nodedict=nodedict)
        self.r = None
        self.k = None
        self.vbare = vbare
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.occ = occ

        logger.info("Hartree Self-energy Calculation Start")
        self.Cal()
        # self.MakeDyn()
        logger.info("Hartree Self-energy Calculation Finish")

    def _get_vbare_k0(self) -> np.ndarray:

        if self.nodedict is None:
            return np.array(self.vbare[..., 0], dtype=np.complex128, copy=True)

        from .utility.MPIManager import MPIFunction

        commk = self.nodedict["commk"]
        mf = MPIFunction()
        owner = mf.KRGlobal2Local(0, self.nodedict["kloc2glob"])

        if owner is None:
            raise ValueError("Global k-index 0 was not found in kloc2glob.")

        owner_rank, local_idx = owner
        vk0 = None

        if commk.Get_rank() == owner_rank:
            vk0 = np.array(self.vbare[..., local_idx], dtype=np.complex128, copy=True)

        vk0 = commk.bcast(vk0, root=owner_rank)

        return vk0

    def Cal(self):
        # vbare = self.vbare.k
        occ = self.occ
        # vk = self.vbare.Double2Quad(self.vbare.k)
        norbc = len(self.crystal.find)  # occk.shape[0]
        ns = self.crystal.ns  # occk.shape[2]
        if self.nodedict is not None:
            rank = self.nodedict["commk"].Get_rank()
            nk = len(self.nodedict["kloc2glob"][rank])
        else:
            nk = len(self.crystal.kpoint)  # occk.shape[3]
        norb = len(self.crystal.bind)  # vbare.shape[0]
        vk0 = self._get_vbare_k0()

        # onsite = self.R2K(self.onsiter)
        h0 = np.zeros((norbc, norbc, ns), dtype=np.complex128)

        if self.crystal.ns != 1:
            #     for ik in range(nk):
            #         tempmat[...,ik] = self.crystal.OrbSpin2Composite(vbare[...,ik])

            # for ik in range(nk):
            #     for ind1 in range(norb*ns):
            #         nn1 = [0]*2
            #         ind1, [iorb,js] = Common.Indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
            #         [iorbc1,iorbc2] = self.crystal.b2f[iorb]

            #         for ind2 in range(norb*ns):
            #             nn2 = [0]*2
            #             ind2, [jorb,ks] = Common.Indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
            #             [iorbc3,iorbc4] = self.crystal.b2f[jorb]
            #             h[iorbc1,iorbc2,js,ik] += tempmat[ind1,ind2,0]*occ[iorbc4,iorbc3,ks]
            # for jk in range(nk):
            #     h[iorbc1,iorbc2,js,ik] += tempmat[ind1,ind2,0]*occ[iorbc4,iorbc3,ks,jk]/nk
            for ind1 in range(norb * ns):
                nn1 = [0] * 2
                ind1, [iorb, js] = Common.Indexing(
                    norb * ns, 2, [norb, ns], 0, ind1, nn1
                )
                [a, [m1, m2]] = self.crystal.BAtomOrb(iorb)
                iorbc1 = self.crystal.FIndex([a, m1])
                iorbc2 = self.crystal.FIndex([a, m2])
                for ind2 in range(norb * ns):
                    nn2 = [0] * 2
                    ind2, [jorb, ks] = Common.Indexing(
                        norb * ns, 2, [norb, ns], 0, ind2, nn2
                    )
                    [b, [m3, m4]] = self.crystal.BAtomOrb(jorb)
                    iorbc3 = self.crystal.FIndex([b, m3])
                    iorbc4 = self.crystal.FIndex([b, m4])
                    # h[iorbc1,iorbc2,js,ik] += vk[iorbc1,iorbc3,iorbc4,iorbc2,js,ks,0]*occ[iorbc4,iorbc3,ks]
                    h0[iorbc1, iorbc2, js] += (
                        vk0[iorb, jorb, js, ks] * occ[iorbc4, iorbc3, ks]
                    )

        else:
            if self.crystal.soc == True:
                C = 1
                # for ik in range(nk):
                #     for iorb in range(norb):
                #         iorbc1,iorbc2 = self.crystal.b2f[iorb]
                #         for jorb in range(norb):
                #             iorbc3, iorbc4 = self.crystal.b2f[jorb]
                #             # gtemp = np.zeros((norbc,norbc,1),dtype=np.complex64)
                #             # for jk in range(nk):
                #             #     gtemp[iorbc4,iorbc3,0] += g0kt[iorbc4,iorbc3,0,0,-1]
                #             h[iorbc1,iorbc2,0,ik] += vbare[iorb,jorb,0,0,0]*occ[iorbc4,iorbc3,0]*C #1/nk*gtemp[iorbc4,iorbc3,0]*C
                for ind1 in range(norb * ns):
                    nn1 = [0] * 2
                    ind1, [iorb, js] = Common.Indexing(
                        norb * ns, 2, [norb, ns], 0, ind1, nn1
                    )
                    [a, [m1, m2]] = self.crystal.BAtomOrb(iorb)
                    iorbc1 = self.crystal.FIndex([a, m1])
                    iorbc2 = self.crystal.FIndex([a, m2])
                    for ind2 in range(norb * ns):
                        nn2 = [0] * 2
                        ind2, [jorb, ks] = Common.Indexing(
                            norb * ns, 2, [norb, ns], 0, ind2, nn2
                        )
                        [b, [m3, m4]] = self.crystal.BAtomOrb(jorb)
                        iorbc3 = self.crystal.FIndex([b, m3])
                        iorbc4 = self.crystal.FIndex([b, m4])
                        h0[iorbc1, iorbc2, js] = (
                            vk0[iorb, jorb, js, ks]
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
                for ind1 in range(norb * ns):
                    nn1 = [0] * 2
                    ind1, [iorb, js] = Common.Indexing(
                        norb * ns, 2, [norb, ns], 0, ind1, nn1
                    )
                    [a, [m1, m2]] = self.crystal.BAtomOrb(iorb)
                    iorbc1 = self.crystal.FIndex([a, m1])
                    iorbc2 = self.crystal.FIndex([a, m2])
                    for ind2 in range(norb * ns):
                        nn2 = [0] * 2
                        ind2, [jorb, ks] = Common.Indexing(
                            norb * ns, 2, [norb, ns], 0, ind2, nn2
                        )
                        [b, [m3, m4]] = self.crystal.BAtomOrb(jorb)
                        iorbc3 = self.crystal.FIndex([b, m3])
                        iorbc4 = self.crystal.FIndex([b, m4])
                        # h[iorbc1,iorbc2,js,ik] += vk[iorbc1,iorbc3,iorbc4,iorbc2,js,ks,0]*occ[iorbc4,iorbc3,ks]*C
                        h0[iorbc1, iorbc2, js] += (
                            vk0[iorb, jorb, js, ks]
                            * occ[iorbc4, iorbc3, ks]
                            * C
                        )

        self.k = np.repeat(h0[..., np.newaxis], nk, axis=3)  # +onsite
        self.r = self.K2R(self.k)

        return None

    def Save(self, fn: str):

        # os.chdir('work')

        # filepath = 'flatstc.h5'
        # groupname = 'sigmah'
        # with h5py.File(filepath,'a') as file:
        #     if self.CheckGroup(filepath,groupname):
        #         group = file[groupname]
        #     else:
        #         group=file.create_group(groupname)

        #     group.create_dataset(fn,dtype=complex,data=self.k)
        # os.chdir('..')
        with h5py.File(self.hdf5file, "a") as file:
            if self.CheckGroup(self.hdf5file, self.group):
                group = file[self.group]
                if self.subgroup in group:
                    sigmah = group[self.subgroup]
                else:
                    sigmah = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                sigmah = group.create_group(self.subgroup)
            sigmah.create_dataset(fn, dtype=complex, data=self.k)

        return None


class SigmaFock(FLatStc):

    def __init__(self,crystal: Crystal,nodedict : dict = None, occr:np.ndarray=None,vbare: np.ndarray = None,hdf5file: str = "glob.h5",group: str = None,):  # green -> occ
        super().__init__(crystal, nodedict=nodedict)
        self.r = None
        self.k = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        # self.green = green
        self.occr = occr
        self.vbare = vbare

        logger.info("Fock Self-energy Calculation Start")
        self.Cal()
        logger.info("Fock Self-energy Calculation Finish")
        # self.MakeDyn()

    def _ensure_local_r_axis(self, matin: np.ndarray) -> np.ndarray:

        if self.nodedict is None:
            return np.array(matin, dtype=np.complex128, copy=True)

        rank = self.nodedict["commk"].Get_rank()
        nr_local = len(self.nodedict["rloc2glob"][rank])
        nr_global = len(self.crystal.kpoint)
        nr_in = matin.shape[-1]

        if nr_in == nr_local:
            return np.array(matin, dtype=np.complex128, copy=True)

        if nr_in == nr_global:
            return self._slice_local_kr(np.array(matin, dtype=np.complex128, copy=True), "rloc2glob")

        raise ValueError(
            f"Unexpected real-space axis length {nr_in}; expected local {nr_local} "
            f"or global {nr_global}."
        )

    def Cal(self):

        # g0rt = self.green.glatrt
        occr = self._ensure_local_r_axis(self.occr)
        vbare = self._ensure_local_r_axis(self.vbare)
        # vr = self.vbare.Double2Quad(self.vbare.r)

        norbc = len(self.crystal.find)
        ns = occr.shape[2]
        nr = occr.shape[3]
        norb = len(self.crystal.bind)

        fr = np.zeros((norbc, norbc, ns, nr), dtype=np.complex128)

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
                ind1, [iorb, js] = Common.Indexing(
                    norb * ns, 2, [norb, ns], 0, ind1, nn1
                )
                [a, [m1, m4]] = self.crystal.BAtomOrb(iorb)
                iorbc1 = self.crystal.FIndex([a, m1])
                iorbc4 = self.crystal.FIndex([a, m4])
                for ind2 in range(norb * ns):
                    nn2 = [0] * 2
                    ind2, [jorb, ks] = Common.Indexing(
                        norb * ns, 2, [norb, ns], 0, ind2, nn2
                    )
                    [b, [m3, m2]] = self.crystal.BAtomOrb(jorb)
                    iorbc3 = self.crystal.FIndex([b, m3])
                    iorbc2 = self.crystal.FIndex([b, m2])
                    if js == ks:
                        # fr[iorbc1,iorbc2,js,ir] += -occr[iorbc4,iorbc3,js,ir]*vr[iorbc1,iorbc3,iorbc2,iorbc4,js,ks,ir]
                        fr[iorbc1, iorbc2, js, ir] += (
                            -occr[iorbc4, iorbc3, js, ir]
                            * vbare[iorb, jorb, js, ks, ir]
                        )

                        # fr[iorbc1,iorbc2,js,ir] += -occr[iorbc3,iorbc4,js,ir]*vr[iorbc1,iorbc3,iorbc2,iorbc4,js,ks,ir]

        fk = self.R2K(fr)

        self.r = fr
        self.k = fk
        del fr, occr, vbare
        return None

    def Save(self, fn: str):

        # os.chdir('work')

        # filepath = 'flatstc.h5'
        # groupname = 'sigmaf'
        # with h5py.File(filepath,'a') as file:
        #     if self.CheckGroup(filepath,groupname):
        #         group = file[groupname]
        #     else:
        #         group=file.create_group(groupname)

        #     group.create_dataset(fn,dtype=complex,data=self.k)
        # os.chdir('..')
        with h5py.File(self.hdf5file, "a") as file:
            if self.CheckGroup(self.hdf5file, self.group):
                group = file[self.group]
                if self.subgroup in group:
                    sigmaf = group[self.subgroup]
                else:
                    sigmaf = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                sigmaf = group.create_group(self.subgroup)
            sigmaf.create_dataset(fn, dtype=complex, data=self.k)

        return None


class Hamiltonian(FLatStc):

    def __init__(self, crystal: Crystal, nodedict : dict = None, ham: np.ndarray = None, beta: float = None, sigmah: np.ndarray = None, sigmaf: np.ndarray = None, sigmac: np.ndarray = None, z : np.ndarray = None, hdf5file: str = "glob.h5", group: str = None,):
        super().__init__(crystal, nodedict)

        self.occ = None
        self.occk = None
        self.occr = None
        self.ham = np.array(ham, dtype=np.complex128, copy=True)
        self.sigmah = sigmah
        self.sigmaf = sigmaf
        self.sigmac = sigmac
        self.z = z
        self.beta = beta
        self.k = None
        self.r = None
        self.kmu0 = None
        self.hkmu0eig = None
        self.hkmu0eigdiag = None
        self.mu = 0
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        # self.muold = mu
        logger.info("Hamiltonian with Self-energy Calculation Start")
        self.CalMu0()
        self.SearchMu()
        logger.info("Hamiltonian with Self-energy Calculation Finish")

    def _nk_global(self) -> int:

        return len(self.crystal.kpoint)

    def _nk_local(self, matk: np.ndarray = None) -> int:

        if matk is not None:
            return matk.shape[3]

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

    def _gather_hkmu0_energy_diag(self) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk_global = self._nk_global()

        diag_local = np.zeros((norb, ns, self.hkmu0eig.shape[3]), dtype=float)

        for ik in range(self.hkmu0eig.shape[3]):
            for js in range(ns):
                diag_local[:, js, ik] = np.diag(self.hkmu0eig[:, :, js, ik])

        if self.nodedict is None:
            return diag_local

        commk = self.nodedict["commk"]
        rank = commk.Get_rank()
        idx_local = self._kr_global_indices("kloc2glob", diag_local.shape[2])

        gathered_idx = commk.gather(idx_local, root=0)
        gathered_diag = commk.gather(diag_local, root=0)

        if rank != 0:
            return None

        diag_global = np.zeros((norb, ns, nk_global), dtype=float)
        for idx_rank, diag_rank in zip(gathered_idx, gathered_diag):
            diag_global[:, :, idx_rank] = diag_rank

        return diag_global

    def CalMu0(self) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns

        tempmat = np.array(self.ham, dtype=np.complex128, copy=True)
        nrk = tempmat.shape[3]

        if self.sigmah is not None:
            tempmat += self.sigmah

        if self.sigmaf is not None:
            tempmat += self.sigmaf

        if self.sigmac is not None:
            if self.z is None:
                raise ValueError("Dynamic self-energy provided without corresponding z-factor.")

            sigma = self.sigmac
            eigval, eigvec = self.Diagonalize(self.z, True)
            tempmat += sigma

            for ik in range(nrk):
                for js in range(ns):
                    diag_vals = np.diag(eigval[:, :, js, ik])
                    if np.any((diag_vals < 0.0) | (diag_vals > 1.0)):
                        logger.error("Error : The z-factor was calculated incorrectly. Please rerun the code.")
                        logger.error(diag_vals)
                        sys.exit()

                    sqrt_diag = np.sqrt(diag_vals)
                    transform = eigvec[:, :, js, ik]
                    inv_transform = np.linalg.inv(transform)
                    dressing = transform @ (np.diag(sqrt_diag) @ inv_transform)
                    block = tempmat[:, :, js, ik]
                    tempmat[:, :, js, ik] = dressing @ block @ dressing

        self.hkmu0 = np.array(tempmat, dtype=np.complex128, copy=True)
        self.hkmu0eig = self.Diagonalize(self.hkmu0)
        self.hkmu0eigdiag = self._gather_hkmu0_energy_diag()
        return None

    def NumOfE(self, mu: float) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk_global = self._nk_global()
        N = self.crystal.nume

        if self.nodedict is None:
            energy = self.hkmu0eigdiag
            Ne = 0.0
            for ik in range(nk_global):
                for js in range(ns):
                    for iorb in range(norb):
                        Ne += 1 / (1 + np.exp((energy[iorb, js, ik] - mu) * self.beta))

            Ne /= nk_global
            return N - Ne

        commk = self.nodedict["commk"]
        rank = commk.Get_rank()
        diff = None

        if rank == 0:
            energy = self.hkmu0eigdiag
            Ne = 0.0
            for ik in range(nk_global):
                for js in range(ns):
                    for iorb in range(norb):
                        Ne += 1 / (1 + np.exp((energy[iorb, js, ik] - mu) * self.beta))

            Ne /= nk_global
            diff = N - Ne

        return commk.bcast(diff, root=0)

    def SearchMu(self):

        if self.nodedict is None:
            energy = self.hkmu0eigdiag
            mumin = float(energy.min()) - 1000
            mumax = float(energy.max()) + 1000
        else:
            commk = self.nodedict["commk"]
            rank = commk.Get_rank()
            mumin = None
            mumax = None

            if rank == 0:
                energy = self.hkmu0eigdiag
                mumin = float(energy.min()) - 1000
                mumax = float(energy.max()) + 1000

            mumin = commk.bcast(mumin, root=0)
            mumax = commk.bcast(mumax, root=0)

        nmin = self.NumOfE(mumin)
        nmax = self.NumOfE(mumax)
        if (nmin < 0) or (nmax > 0):
            logger.error("Chemical potential is out of the bisection range")
            sys.exit()
        self.mu = scipy.optimize.brentq(self.NumOfE, mumin, mumax, xtol=1.0e-10)

        self.UpdateMu()
        return None

    def Occ(self) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = self.k.shape[3]
        nk_global = self._nk_global()

        # energy = self.Diagonalize(self.hk)

        occk = np.zeros((norb, norb, ns, nrk), dtype=np.complex128)
        occ_local = np.zeros((norb, norb, ns), dtype=np.complex128)
        tempmat = np.zeros((norb, norb), dtype=float)

        energy, eigvec = self.Diagonalize(self.k, True)
        for irk in range(nrk):
            for js in range(ns):
                tempmat.fill(0.0)
                for iorb in range(norb):
                    tempmat[iorb, iorb] = 1 / (
                        np.exp(energy[iorb, iorb, js, irk] * self.beta) + 1
                    )
                # occk[:,:,js,irk] = np.dot(eigvec[:,:,js,irk],np.dot(tempmat,np.linalg.inv(eigvec[:,:,js,irk])))
                occk[:, :, js, irk] = np.dot(
                    eigvec[:, :, js, irk],
                    np.dot(tempmat, scipy.linalg.inv(eigvec[:, :, js, irk])),
                )

            occ_local += occk[..., irk]

        occ = self._allreduce_array(occ_local)
        occ /= nk_global

        self.occ = occ
        self.occk = occk
        self.occr = self.K2R(occk)

        return None

    def UpdateMu(self) -> np.ndarray:

        chem = self.ChemEmbedding(self.mu)

        ham = self.hkmu0 - chem
        hamr = self.K2R(ham)
        self.k = ham
        self.r = hamr
        self.Occ()

        return None

    def Save(self, fn: str, chem: bool = False):
        # os.chdir('work')

        # filepath = 'flatstc.h5'
        # groupname = 'sigmah'
        # with h5py.File(filepath,'a') as file:
        #     if self.CheckGroup(filepath,groupname):
        #         group = file[groupname]
        #     else:
        #         group=file.create_group(groupname)

        #     group.create_dataset(fn,dtype=complex,data=self.k)
        # os.chdir('..')
        with h5py.File(self.hdf5file, "a") as file:
            if self.CheckGroup(self.hdf5file, self.group):
                group = file[self.group]
                if self.subgroup in group:
                    ham = group[self.subgroup]
                else:
                    ham = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                ham = group.create_group(self.subgroup)
            if chem:
                ham.create_dataset("mu", dtype=float, data=self.mu)
            ham.create_dataset(fn, dtype=complex, data=self.k)

        return None
    
    def OccMixing(self, iter : int = None, mix : float = None, occkb : np.ndarray = None, occkm : np.ndarray = None) -> np.ndarray:

        norb = occkb.shape[0]
        ns = occkb.shape[2]
        nk = occkb.shape[3]
        
        occnew = np.zeros((norb, norb, ns),dtype=np.complex128)
        occknew = self.Mixing(iter=iter, mix=mix, Fb=occkb, Fm=occkm)

        for ik in range(nk):
            occnew += occknew[..., ik]

        occnew = occknew/nk
        occrnew = self.K2R(occknew)

        self.occ = occnew
        self.occk = occknew
        self.occr = occrnew

        return None

class HamiltonianAB(FLatStc):

    def __init__(self, crystal: Crystal):
        super().__init__(crystal)

        glob = h5py.File('../../glob_dat/global.dat', 'r')
        self.i_kerf = glob['full_space']['gw']['i_kref'][:]
        self.kpt_latt = glob['combasis_fermion']['kpt_latt'][:]
        self.nbndf = glob['full_space']['gw']['nbndf'][:]
        self.n_omega = glob['full_space']['gw']['n_omega'][:]
        self.n3 = glob['full_space']['hf_n3'][:]
        glob.close()

    def KI2KF(self):
        
        tempmat = np.zeros((self.nbndf[0], self.nbndf[0], self.n3[0], len(self.kpt_latt), self.crystal.ns), dtype=np.complex128)

        glob = h5py.File('../../glob_dat/global.dat', 'r')

        for js in range(self.crystal.ns):
            for iw in range(self.n3[0]):
                for ik in range(len(self.kpt_latt)):
                    kidx = self.i_kerf[ik]
                    name = 'hf_w_'+str(iw+1)+'_k_'+str(kidx)
                    tempmat[...,iw,ik, js] = glob['full_space'][name][:]
        glob.close()

        self.k = tempmat
        self.r = self.K2R(self.k)

        return None
    
class ZFactor(FLatStc):

    def __init__(self, crystal : Crystal, sigmagwc = None, hdf5file : str = 'glob.h5',group : str = None):

        super().__init__(crystal)

        self.sigmagwc = sigmagwc
        self.beta = sigmagwc.dlr.beta
        self.r = None
        self.k = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        logger.info("Z-factor Calculation Start")
        self.Cal()
        self.Check(self.k)
        logger.info("Z-factor Calculation Finish")

    def Cal(self):

        norb, _, ns, nk, nfreq = self.sigmagwc.kf.shape
        dlr = self.sigmagwc.dlr
        beta = self.beta

        # DLR Matsubara -> evaluate at lowest uniform Matsubara frequency
        sigma0 = np.zeros((norb, norb, ns, nk), dtype=np.complex128)
        iw0 = np.array([np.pi / beta]) * 1j 
        # omega = dlr.MatsubaraFermionUniform(dlr.cutoff, beta=beta) * 1j
        for iorb in range(norb):
            for jorb in range(norb):
                for js in range(ns):
                    for ik in range(nk):
                        ff = self.sigmagwc.kf[iorb, jorb, js, ik, :]
                        fxx = dlr.dF.dlr_from_matsubara(ff, beta=beta, xi=-1)
                        val = dlr.dF.eval_dlr_freq(fxx[:, None, None], iw0, beta=beta, xi=-1)
                        sigma0[iorb, jorb, js, ik] = val[0, 0, 0]

        iw = 1j * beta / (2.0 * np.pi)

        tempmat = np.zeros((norb, norb, ns, nk), dtype=np.complex128)

        # for jorb in range(norb):
        #     for iorb in range(norb):
        #         if (iorb == jorb):
        #             tempmat[iorb, jorb, :, :] = 1.0 - iw * (
        #                 sigma0[iorb, jorb, :, :]
        #                 - np.conjugate(sigma0[jorb, iorb, :, :])
        #             )
        #         else:
        #             tempmat[iorb, jorb, :, :] = - iw * (
        #                 sigma0[iorb, jorb, :, :]
        #                 - np.conjugate(sigma0[jorb, iorb, :, :])
        #             )

        sigma0_dag = np.transpose(np.conjugate(sigma0), (1, 0, 2, 3))

        tempmat = np.asfortranarray(iw * (sigma0 - sigma0_dag), dtype=np.complex128)
        diag_idx = np.arange(norb)
        tempmat[diag_idx, diag_idx, :, :] += 1.0

        z = self.Inverse(tempmat)

        # Validate z-factor: eigenvalues must be in (0, 1]
        

        self.k = z
        self.r = self.K2R(z)
        del z, tempmat

        return None
    
    def Check(self, z : np.ndarray):

        norb, _, ns, nk = z.shape
        eigval, eigvec = self.Diagonalize(z, True)
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    if not (0 < eigval[iorb, iorb, js, ik] <= 1):
                        logger.error("Error : The z-factor was calculated incorrectly. "
                                     f"eigval[{iorb},{iorb},{js},{ik}] = {eigval[iorb, iorb, js, ik]:.6f}")
                        logger.error("Please rerun the code.")
        
        return None
    
    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            if self.CheckGroup(self.hdf5file,self.group):
                group = file[self.group]
                if self.subgroup in group:
                    sigmac = group[self.subgroup]
                else:
                    sigmac = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                sigmac = group.create_group(self.subgroup)
            

            if obj != None:
                sigmac.create_dataset(fn,dtype=complex,data=obj)
            else:
                sigmac.create_dataset(fn,dtype=complex,data=self.k)

        return None


class SigmaStc(FLatStc):

    def __init__(self, crystal : Crystal, sigmagwc = None, hdf5file : str = 'glob.h5',group : str = None):

        super().__init__(crystal)

        self.sigmagwc = sigmagwc
        self.beta = sigmagwc.dlr.beta
        self.r = None
        self.k = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        logger.info("Static Self-energy Calculation Start")
        self.Cal()
        logger.info("Static Self-energy Calculation Finish")

    def Cal(self):

        norb, _, ns, nk, nfreq = self.sigmagwc.kf.shape
        dlr = self.sigmagwc.dlr
        beta = self.beta

        # DLR Matsubara -> evaluate at lowest uniform Matsubara frequency
        sigma0 = np.zeros((norb, norb, ns, nk), dtype=np.complex128)
        iw0 = np.array([np.pi / beta]) * 1j
        for iorb in range(norb):
            for jorb in range(norb):
                for js in range(ns):
                    for ik in range(nk):
                        ff = self.sigmagwc.kf[iorb, jorb, js, ik, :]
                        fxx = dlr.dF.dlr_from_matsubara(ff, beta=beta, xi=-1)
                        val = dlr.dF.eval_dlr_freq(fxx[:, None, None], iw0, beta=beta, xi=-1)
                        sigma0[iorb, jorb, js, ik] = val[0, 0, 0]

        tempmat = np.zeros((norb, norb, ns, nk), dtype=np.complex128)

        for jorb in range(norb):
            for iorb in range(norb):
                tempmat[iorb, jorb] = 1/2 * (
                    sigma0[iorb, jorb, :, :]
                    + np.conjugate(sigma0[jorb, iorb, :, :])
                )

        self.k = tempmat
        self.r = self.K2R(tempmat)

        del tempmat

        return None
    
    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            if self.CheckGroup(self.hdf5file,self.group):
                group = file[self.group]
                if self.subgroup in group:
                    sigmac = group[self.subgroup]
                else:
                    sigmac = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                sigmac = group.create_group(self.subgroup)
            

            if obj != None:
                sigmac.create_dataset(fn,dtype=complex,data=obj)
            else:
                sigmac.create_dataset(fn,dtype=complex,data=self.k)

        return None
