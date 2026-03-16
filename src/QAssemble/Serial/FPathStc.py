import copy
import itertools
import json
import os
import re as re
import shutil
import string as string
import subprocess
import sys
from collections import OrderedDict
from typing import Any

import h5py
import matplotlib as mat
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.optimize
from pylab import cm
from scipy.fftpack import fftn, ifftn
from sympy.physics.wigner import gaunt, wigner_3j

# qapath = os.environ.get("QAssemble", "")
# sys.path.append(qapath + "/src/QAssemble/modules")
# import QAFort

from .Crystal import Crystal
from .FLatStc import FLatStc
from .utility.Fourier import Fourier


class FPathStc(object):

    def __init__(
        self, crystal: Crystal = None, obj: object = None, hdf5file: str = "glob.h5"
    ):

        if (crystal is not None) and (obj is not None):
            pass
        else:
            if os.path.exists(hdf5file):
                glob = h5py.File(hdf5file)
                ini = glob["input"]
                tempcry = ini["Crystal"]
                cry = {}
                for key in tempcry.keys():
                    if type(tempcry[key][()]) == bytes:
                        cry[key] = str(tempcry[key][()], "utf-8")
                    else:
                        cry[key] = tempcry[key][()]
                for key in cry.keys():
                    if key == "Basis":
                        cry[key] = eval(cry[key])
                    elif key == "KGrid":
                        cry[key] = eval(cry[key])
                    elif key == "RVec":
                        cry[key] = eval(cry[key])
                    else:
                        cry[key] = cry[key]

                crystal = Crystal(cry=cry)
                glob.close()
            else:
                print(f"Error : Check the {self.__class__.__name__} input again")
                sys.exit()

        self.crystal = crystal
        self.obj = obj
        self.flatstc = FLatStc(crystal=self.crystal)
        self.hdf5file = hdf5file

    def CheckGroup(self, filepath: str, group: str):

        with h5py.File(filepath, "r") as file:
            return group in file

    def Inverse(self, mat: np.ndarray):

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]

        matinv = np.zeros((norb, norb, ns, nrk), dtype=np.complex64, order="F")

        for irk in range(nrk):
            for js in range(ns):
                matinv[:, :, js, irk] = np.linalg.inv(mat[:, :, js, irk])

        return matinv

    def R2K(self, matr: np.ndarray = None, rvec : np.ndarray = None, kpoint: np.ndarray = None):  # R2KAny

        # if self.crystal.kpath == None:
        #     print("Error, kpath doesn't generate")
        #     sys.exit()
        # kpoint = self.crystal.kpath
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nr = self.crystal.rkgrid[0] * self.crystal.rkgrid[1] * self.crystal.rkgrid[2]
        nk = len(kpoint)

        
        tempmat = copy.deepcopy(matr)
        # matk = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
        if (rvec is None):
            self.crystal.RVec()
            # matk = QAFort.fourier.fpathstc_r2k(self.crystal.rvec, kpoint, tempmat)
            matk = Fourier.FPathStcR2K(tempmat, kpoint, self.crystal.rvec)
        else:
            # matk = QAFort.fourier.fpathstc_r2k(rvec, kpoint, tempmat) 
            matk = Fourier.FPathStcR2K(tempmat, kpoint, rvec)

        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        # temp = 0
                        # for ir in range(nr):
                        #     temp += tempmat[iorb,jorb,js,ir]*np.exp(-2.0j*np.pi*(kpoint[ik]@self.crystal.rvec[ir]))
                        [a, m1] = self.crystal.FAtomOrb(iorb)
                        [b, m2] = self.crystal.FAtomOrb(jorb)
                        delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                        phase = np.exp(-2.0j * np.pi * (kpoint[ik] @ delta))
                        matk[iorb, jorb, js, ik] = matk[iorb, jorb, js, ik] * phase
                        # for ir in range(nr):
                        #     [a,m1] = self.crystal.FAtomOrb(iorb)
                        #     [b,m2] = self.crystal.FAtomOrb(jorb)
                        #     delta = self.crystal.basisf[a,:]-self.crystal.basisf[b,:]
                        #     temp += tempmat[iorb,jorb,js,ir]*np.exp(-2.0j*np.pi*(kpoint[ik]@(delta-self.crystal.rvec[ir])))
                        # matk[iorb,jorb,js,ik] = temp

        return matk
    
    def K2R(self, matk : np.ndarray = None, rvec : np.ndarray = None, kpoint : np.ndarray = None):

        (norb, norb, ns, nk) = matk.shape

        if (nk != len(kpoint)):
            print('Put the wrong kpoint, Please check again')
            sys.exit()

        nr = len(rvec)
        tempmat = copy.deepcopy(matk)

        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        [a, m1] = self.crystal.FAtomOrb(iorb)
                        [b, m2] = self.crystal.FAtomOrb(jorb)

                        delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]

                        phase = np.exp(2.0j * np.pi * kpoint[ik]@delta)

                        tempmat[iorb,jorb,js,ik] *= phase

        matr = QAFort.fourier.fpathstc_k2r(rvec, kpoint, tempmat)

        return matr
    
    def Slab(self, matk : np.ndarray = None):

        (norb, norb, ns, nk) = matk.shape
        kgrid = self.crystal.rkgrid
        if (nk != kgrid[0]*kgrid[1]*kgrid[2]):
            print('Put the wrong kpoint, Please check again')
            sys.exit()
        
        kpoint = self.SlabKpoint()
        tempmat = self.Reshape(matk=matk, kpoint=kpoint)

        for ikz in range(kgrid[2]):
            for iky in range(kgrid[1]):
                for ikx in range(kgrid[0]):
                    for js in range(ns):
                        for jorb in range(norb):
                            [b, m2] = self.crystal.FAtomOrb(jorb)
                            for iorb in range(norb):
                                [a, m1] = self.crystal.FAtomOrb(iorb)
                                delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                                phase = np.exp(2.0j*np.pi*kpoint[0, 0, ikz]@delta)

                                tempmat[iorb, jorb, js, ikx, iky, ikz] *= phase

        Zslab = self.SlabZmat()
        (nz, nz) = Zslab.shape
        tempmat2 = QAFort.fourier.fpathstc_slab(Zslab, kpoint, tempmat)

        matslab = copy.deepcopy(tempmat2)

        for iz1 in range(nz //2 + 1):
            for iz2 in range(nz //2 + 1, nz):
                for iky in range(kgrid[1]):
                    for ikx in range(kgrid[0]):
                        for js in range(ns):
                            for jorb in range(norb):
                                for iorb in range(norb):
                                    matslab[iorb, jorb, js, ikx, iky, iz1, iz2] = 0.0e0
                                    matslab[iorb, jorb, js, ikx, iky, iz2, iz1] = 0.0e0

        del Zslab, matk, kpoint, tempmat, tempmat2
        return matslab
    
    def RVec(self, grid : list = None):

        r = np.zeros((grid[0]*grid[1]*grid[2],3), dtype=np.float64)
        nr = grid[0]*grid[1]*grid[2]

        for iz in range(grid[2]):
            for iy in range(grid[1]):
                for ix in range(grid[0]):
                    nn1 = [ix, iy, iz]
                    ind1, nn1 = self.crystal.indexing(nr, 3, grid, 1, 0, nn1)
                    if (ix > grid[0] // 2):
                        xx = ix - grid[0]
                    else:
                        xx = ix
                    if (iy > grid[1] // 2):
                        yy = iy - grid[1]
                    else:
                        yy = iy
                    if (iz > grid[2] // 2):
                        zz = iz - grid[2]
                    else:
                        zz = iz

                    r[ind1] = [xx, yy, zz]
        
        return r

    def Gaussian(self, x, mu, sigma=0.1):

        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    def Dos(
        self,
        matr: np.ndarray = None,
        kgrid=[20, 20, 20],
        sigma: float = 0.1,
        plotoption: bool = False,
        energyrange: list = None,
    ):

        print("***** DOS Calculation Start *****")
        norb = matr.shape[0]
        ns = matr.shape[2]
        if type(kgrid) is list:
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
        elif type(kgrid) is np.ndarray:
            nk = len(kgrid)
            kpoint = kgrid

        print("***** Fourier transfrom R2K Start *****")
        matk = self.R2K(matr=matr, kpoint=kpoint)
        print("***** Fourier transfrom R2K Finish *****")
        print("***** Hamiltonian Diagonalization Start *****")
        (energy, eigvec) = self.flatstc.Diagonalize(matk=matk, eigvec=True)
        print("***** Hamiltonian Diagonalization Finish *****")

        if energyrange == None:
            emin = -10
            emax = 10
        else:
            emin = energyrange[0]
            emax = energyrange[-1]
        E = np.linspace(emin, emax, nk)
        dos = np.zeros((norb, ns, nk), dtype=complex)
        tempmat = np.zeros((norb, ns, nk), dtype=float)

        print("***** Gaussian Approach Start *****")
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    e = energy[iorb, iorb, js, ik]
                    tempmat[iorb, js] += self.Gaussian(E, e, sigma) / nk
        print("***** Gaussian Approach Finish *****")

        eiginv = self.Inverse(eigvec)

        for ik in range(nk):
            for js in range(ns):
                D = np.diag(tempmat[:, js, ik])
                tempmat2 = eigvec[:, :, js, ik] @ (D @ eiginv[:, :, js, ik])
                for iorb in range(norb):
                    dos[iorb, js, ik] = tempmat2[iorb, iorb]

        print(f"Integration gaussian : {np.trapz(self.Gaussian(E,0),E)}")

        temp = 0
        for js in range(ns):
            for iorb in range(norb):
                temp += np.trapz(dos[iorb, js], E)

        print(f"Integration dos : {temp}")

        if plotoption:
            fig, ax = plt.subplots()
            ax.set_xlim(E[0], E[-1])
            legend = []
            for js in range(ns):
                for iorb in range(norb):
                    ax.plot(E, dos[iorb, js].real)
                    legend.append(iorb + 1)

            ax.legend(legend)
            ax.set_xlabel("E (eV)")
            ax.set_ylabel("DOS")
            plt.show()
        else:
            with open("dos.dat", "w") as f:
                for ie in range(len(E)):
                    for js in range(ns):
                        linedata = [E[ie]] + dos[:, js, ie].real.tolist()
                        line = " ".join(map(str, linedata))
                        f.write(line + "\n")

        return (E, dos)

    def Band(
        self,
        hmat: np.ndarray,
        fn: str = None,
        plotoption: bool = False,
        label: list = None,
    ):
        if (self.crystal.kpath == None).all():
            print("Error: K-path not created, please check your K-path options")
            sys.exit()

        hmatk = self.R2K(matr=hmat, kpoint=self.crystal.kpath)

        energy = self.flatstc.Diagonalize(hmatk)
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
                plt.show()
                # plt.plot(energyplot.T[:,0,:])
            # if fn == None:
            #    plt.show()
            # else:
            #    plt.savefig(fn)
            else:
                up = energyplot[:, 0, :]
                down = energyplot[:, 1, :]
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
                    plt.plot(self.crystal.kdist, up[iorb, :].T, "k-")
                    plt.plot(self.crystal.kdist, down[iorb, :].T, "r-")
                ax.set_ylabel("E (ev)")
                ax.set_title("Band")
                plt.show()
        #               if fn == None:
        #                   plt.show()
        #               else:
        #                   plt.savefig(fn)
        else:
            if fn != None:
                with open(fn, "w") as f:
                    for js in range(ns):
                        for ik in range(nk):
                            linedata = [self.crystal.kdist[ik]] + energyplot[
                                :, js, ik
                            ].tolist()
                            line = " ".join(map(str, linedata))
                            f.write(line + "\n")
            else:
                with open("band.dat", "w") as f:
                    for js in range(ns):
                        for ik in range(nk):
                            linedata = [self.crystal.kdist[ik]] + energyplot[
                                :, js, ik
                            ].tolist()
                            line = " ".join(map(str, linedata))
                            f.write(line + "\n")

        with h5py.File(self.hdf5file,'a') as file:
            if self.CheckGroup(self.hdf5file, 'Post'):
                post = file['Post']                
            else:
                post = file.create_group('Post')
            post.create_dataset('band',dtype=float,data=energyplot)
            post.create_dataset('kdist',dtype=float,data=self.crystal.kdist)
            post.create_dataset('knode',dtype=float,data=self.crystal.knode)
                
        

        return None
    
    def FermiSurface(self, hmat : np.ndarray = None, num : int = 101):

        kp = np.linspace(-1,1,num=num)

        kpoint_temp = np.array(list(itertools.product(np.linspace(0,1,num=1),kp,kp)))
        kpoint = np.fliplr(kpoint_temp)

        hmat_fs = self.R2K(hmat,kpoint)
        (norb,norb,ns,nk) = hmat_fs.shape

        fs = self.flatstc.Diagonalize(hmat_fs)

        fs = fs.reshape((norb,norb,ns,num,num,1),order='F')
        kpoint = kpoint.reshape((num,num,1,3),order='F')

        with h5py.File(self.hdf5file,'a') as file:
            if (self.CheckGroup(self.hdf5file,'Post')):
                post = file['Post']
            else:
                post = file.create_group('Post')
            post.create_dataset('fermi_surface',dtype=float,data=fs)
            post.create_dataset('kpoint',dtype=float,data=kpoint)
        
        return None
    
    def Occ(self, hmat : np.ndarray = None, beta : float = None):

        (norb, norb, ns, nk) = hmat.shape

        occk = np.zeros((norb, norb, ns, nk),dtype=np.complex128,order='F')
        tempmat = np.zeros((norb, norb),dtype=np.complex128, order='F')
        eigval, eigvec = self.flatstc.Diagonalize(hmat, True)

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    tempmat[iorb,iorb] = 1 / (np.exp(eigval[iorb, iorb, js, ik] * beta) + 1)

                occk[:, :, js, ik] = eigvec[:, :, js, ik] @ (tempmat @ np.linalg.inv(eigvec[:, :, js, ik]))

        return occk
    
    def SlabKpoint(self):

        kgrid = self.crystal.rkgrid

        kpoint = np.zeros((kgrid[0], kgrid[1], kgrid[2], 3), order='F')

        for ikz in range(kgrid[2]):
            for iky in range(kgrid[1]):
                for ikx in range(kgrid[0]):
                    kpoint[ikx, iky, ikz] = [ikx/kgrid[0], iky/kgrid[1], ikz/kgrid[2]]

        return kpoint
    
    def Reshape(self, matk : np.ndarray = None, kpoint : np.ndarray = None):

        kgrid = self.crystal.rkgrid

        (norb, norb, ns, nk) = matk.shape
        matkret = np.zeros((norb, norb, ns, kgrid[0], kgrid[1], kgrid[2]), dtype=np.complex128, order='F')

        for ikz in range(kgrid[2]):
            for iky in range(kgrid[1]):
                for ikx in range(kgrid[0]):
                    for ik in range(nk):
                        diff = self.crystal.kpoint[ik] - kpoint[ikx, iky, ikz]
                        if(abs(diff[0])<1.0e-6)and(abs(diff[1])<1.0e-6)and(abs(diff[2])<1.0e-6):
                            matkret[:, :, :, ikx, iky, ikz] = matk[:, :, :, ik]

        return matkret

    def SlabZmat(self):

        Zslab = np.arange(self.crystal.rkgrid[2])
        # row1 = [Zslab[0]] + list(Zslab[1:])[::-1]
        Z = np.zeros((len(Zslab), len(Zslab)))

        for iz2 in range(len(Zslab)):
            for iz1 in range(len(Zslab)):
                Z[iz1, iz2] = Zslab[iz1]-Zslab[iz2]
                if (Z[iz1, iz2] < 0):
                    Z[iz1, iz2] += self.crystal.rkgrid[2]

        return Z
    
    def Moments(self, matk : np.ndarray = None, beta : np.float64 = None, kgrid : list = None):

        (kplus, kminus) = self.flatstc.KValley(kgrid)
        
        if (kgrid is not None):
            kpoint_temp=np.array(list(itertools.product(np.linspace(0,1,num=kgrid[2],endpoint=False),np.linspace(0,1,num=kgrid[1],endpoint=False),np.linspace(0,1,num=kgrid[0],endpoint=False))))
            kpoint=np.fliplr(kpoint_temp)

        norb = matk.shape[0]
        ns = matk.shape[2]
        
        n = np.zeros((norb, ns, 2), dtype=np.complex128, order='F')
        m = np.zeros((norb), dtype=np.complex128)
        v = np.zeros((norb),dtype=np.complex128)
        s = np.zeros((norb//2), dtype=np.complex128)

        if (kgrid is None):
            tempmat = matk
        else:
            matr = self.flatstc.K2R(matk)
            tempmat = self.R2K(matr = matr, kpoint=kpoint)

        occ = self.Occ(hmat=tempmat, beta = beta)
        for js in range(ns):
            for iorb in range(norb):
                for ik in kplus:
                    n[iorb, js, 0] += 1/len(kplus) * occ[iorb, iorb, js, ik]
                for ik2 in kminus:
                    n[iorb, js, 1] += 1/len(kminus) * occ[iorb, iorb, js, ik2]

        for js in range(ns):
            for ik in range(2):
                m += n[:, 0, ik] - n[:, 1, ik]
                v += n[:, js, 0] - n[:, js, 1]
                for iorb in range(0, norb, 2):
                    s[iorb//2] += n[iorb, js, ik] - n[iorb + 1, js, ik]

        return (n, m, v, s)