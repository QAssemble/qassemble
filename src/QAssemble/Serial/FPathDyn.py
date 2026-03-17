import copy
import os
import re as re
import shutil
import string as string
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
# qapath = os.environ.get("QAssemble", "")
# sys.path.append(qapath + "/src/QAssemble/modules")
# import QAFort
import subprocess
from .Crystal import Crystal
from .FLatDyn import FLatDyn
# from .FTGrid import FTGrid
from .utility.DLR import DLR
from .utility.Fourier import Fourier


class FPathDyn(object):

    def __init__(
        self,
        crystal: Crystal = None,
        dlr: DLR = None,
        obj: object = None,
        kpath: list = None,
        nk: int = None,
        hdf5file: str = "glob.h5",
    ):

        if (crystal is not None) and (dlr is not None) and (obj is not None):
            pass
        else:
            if os.path.exists(hdf5file):
                glob = h5py.File(hdf5file)
                ini = glob["input"]
                tempcry = ini["Crystal"]
                control = ini["Control"]
                cry = {}
                kb = 8.6173303 * 10**-5
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
                if ("T" in control) and ("beta" not in control):
                    T = control["T"][()]
                    beta = 1 / (T * kb)
                elif ("T" not in control) and ("beta" in control):
                    beta = control["beta"][()]
                    T = 1 / (beta * kb)
                cutoff = control.get("MatsubaraCutOff", 50)
                ft_input = {}
                ft_input["T"] = T
                ft_input["beta"] = beta
                ft_input["cutoff"] = cutoff[()]
                # print(cutoff[()])
                dlr = DLR(ft_input)
                glob.close()
            elif (crystal is not None) and (dlr is not None):
                pass                
            else:
                print(f"Error : Check the {self.__class__.__name__} input again")
                sys.exit()
        self.crystal = crystal
        self.dlr = dlr
        self.flatdyn = FLatDyn(self.crystal, self.dlr)
        if (kpath is not None) and (nk is not None):
            self.kpath = self.crystal.Kpath(kpath=kpath, nk=nk)
        self.k = None
        self.hdf5file = hdf5file

        if obj is not None:
            if obj.__class__.__name__ == "GreenInt":
                self.k = self.KArb(obj.rf, kpoint=self.kpath)
            elif obj.__class__.__name__ == "GreenBare":
                self.k = self.KArb(obj.g0rf, kpoint=self.kpath)
            elif obj.__class__.__name__ == "SigmaGWC":
                self.k = self.R2K(matr=obj.rf, kpoint=self.kpath)

    def CheckKeyinString(self, key: str, dictionary: dict):

        if key not in dictionary:
            print("missing '" + key + "' in input", flush=True)
            sys.exit()
        return None

    def Inverse(self, mat: np.ndarray) -> np.ndarray:

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]
        nft = mat.shape[4]

        matinv = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex64, order="F")

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    matinv[:, :, js, irk, ift] = np.linalg.inv(mat[:, :, js, irk, ift])
        # for js, irk, ift in itertools.product(list(range(ns)),list(range(nrk),list(range(nft)))):
        #     matinv[:,:,js,irk,ift] = np.linalg.inv(mat[:,:,js,irk,ift])

        return matinv

    def R2K(self, matr: np.ndarray = None, kpoint: np.ndarray = None):  # R2KAny

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nr = self.crystal.rkgrid[0] * self.crystal.rkgrid[1] * self.crystal.rkgrid[2]
        nk = len(kpoint)
        nft = matr.shape[4]

        self.crystal.RVec()
        tempmat = copy.deepcopy(matr)
        matk = Fourier.FPathDynR2K(tempmat, kpoint, self.crystal.rvec)
        # matk = QAFort.fourier.fpathdyn_r2k(self.crystal.rvec, kpoint, tempmat)

        for ift in range(nft):
            for ik in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            # temp = 0
                            # for ir in range(nr):
                            #     temp += tempmat[iorb,jorb,js,ir,ift]*np.exp(-2.0j*np.pi*(kpoint[ik]@self.crystal.rvec[ir]))
                            [a, m1] = self.crystal.FAtomOrb(iorb)
                            [b, m2] = self.crystal.FAtomOrb(jorb)
                            delta = (
                                self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                            )
                            phase = np.exp(-2.0j * np.pi * (kpoint[ik] @ delta))
                            matk[iorb, jorb, js, ik, ift] = (
                                matk[iorb, jorb, js, ik, ift] * phase
                            )

        return matk

    def KArb(self, matk: np.ndarray = None, kpoint: np.ndarray = None, omega : np.ndarray = None):  ## naming

        norb = matk.shape[0]
        ns = matk.shape[2]
        nk = matk.shape[3]
        nfreq = matk.shape[4]
        nkpath = len(kpoint)

        tempmat = np.zeros((norb, norb, ns, nk, nfreq), dtype=complex, order="F")
        # matkinv = np.zeros((norb, norb, ns, nk, nfreq), dtype=complex, order="F")

        matkinv = self.Inverse(matk)

        # moments, high = self.flatdyn.Moment(matk, False, False)
        
        
        if (omega is None):
            omega = self.dlr.omega

        for ifreq in range(nfreq):
            for ir in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            if iorb == jorb:
                                tempmat[iorb, jorb, js, ir, ifreq] = (
                                    1j * omega[ifreq] 
                                    - matkinv[iorb, jorb, js, ir, ifreq]
                                )
                            else:
                                tempmat[iorb, jorb, js, ir, ifreq] = (
                                    -matkinv[iorb, jorb, js, ir, ifreq]
                                )

        # tempmat2 = self.R2K(tempmat, kpoint)
        tempmat2 = self.flatdyn.K2R(tempmat)
        tempmat3 = self.R2K(tempmat2, kpoint)

        matkinv = np.zeros((norb, norb, ns, nkpath, nfreq), dtype=complex, order="F")
        for ifreq in range(nfreq):
            for ik in range(nkpath):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            if iorb == jorb:
                                matkinv[iorb, jorb, js, ik, ifreq] = (
                                    1j * omega[ifreq]
                                    - tempmat3[iorb, jorb, js, ik, ifreq]
                                )
                            else:
                                matkinv[iorb, jorb, js, ik, ifreq] = -tempmat3[
                                    iorb, jorb, js, ik, ifreq
                                ]

        matk = self.Inverse(matkinv)

        return matk

    def  MQEMWrapper(self, option: dict = None):
        '''
        MQEM Wrapper for Analytic Continuation
        Options:
            gauxmode : 'asitis' or 'auxg' (default: 'asitis')
            defaultmodel : 'g_mat' or 'g' (default: 'g_mat')
            smearing : float (default: 0.01)
            interpolation : True or False (default: True)
            gaussian_broadening : float (default: 0.2)
        Returns:
            xnew : real frequency grid
            sig_real : analytically continued self-energy on real frequency grid
        '''

        #       target = option['target']
        # gauxmode = option["gauxmode"]
        gauxmode = option.get("gauxmode", "asitis")
        defaultmodel = option.get("defaultmodel", "g_mat")
        # defaultmodel = option["defaultmodel"]
        blur = float(option.get("smearing", 0.01))
        # blur = float(option["smearing"])
        interpolation = option.get("interpolation", True)
        # interpolation = option["interpolation"]
        gaussian_broadening = float(option.get("gaussian_broadening", 0.2))
        cutoff = option.get("cutoff", self.dlr.cutoff)
        beta = option.get("beta", self.dlr.beta)
        # gaussian_broadening = float(option["gaussian_broadening"])
        # omega = self.dlr.MatsubaraFermionUniform(Emax=cutoff, beta=beta)
        omega = np.loadtxt("omega.dat")

        info = np.loadtxt("info.dat")
        norb = int(info[0])
        ns = int(info[1])
        nomega = int(info[2])

        raw_greem = np.loadtxt("green.dat")

        gmat = np.zeros((norb, norb, ns, 1, nomega), dtype=np.complex128, order="F")
        gauxmat = np.zeros((norb, norb, ns, 1, nomega), dtype=np.complex128, order="F")

        for line in raw_greem:
            iorb = int(line[0])
            jorb = int(line[1])
            js = int(line[2])
            iomega = int(line[3])
            gmat[iorb, jorb, js, 0, iomega] = line[4] + 1j * line[5]

        if gauxmode == "asitis":
            gauxmat = gmat
            # gauxmat, _ = self.flatdyn.Diagonalize(gmat)

            # (moment, high) = self.flatdyn.Moment(gauxmat, True, True)
            (moment, high) = self.flatdyn.Moment(gauxmat, False, False)
        elif gauxmode == "auxg":
            (moment_temp, high_temp) = self.flatdyn.Moment(gmat, False, False)

            for ik in range(1):
                for js in range(ns):
                    for iorb in range(norb):
                        gauxmat[iorb, iorb, js, ik] = 1.0 / (
                            omega * 1j
                            - (gmat[iorb, iorb, js, ik] - high_temp[iorb, iorb, js, ik])
                        )

            # (moment, high) = self.flatdyn.Moment(gauxmat, True, True)
            (moment, high) = self.flatdyn.Moment(gauxmat, False, False)
        #       if (target == 'sig'): #We need gaux moment
        #           (moment, high) = self.flatdyn.Moment(gmat,0,1)
        #       elif (target == 'g'):
        #           (moment,high) = self.flatdyn.Moment(gmat,1,1)

        if os.path.exists("realFreq_Sw.dat_1_1"):
            os.remove("realFreq_Sw.dat_1_1")

        emax = 0

        for ik in range(1):
            for js in range(ns):
                for iorb in range(norb):
                    m1 = moment[iorb, iorb, js, ik, 0]
                    m2 = moment[iorb, iorb, js, ik, 1]
                    m3 = moment[iorb, iorb, js, ik, 2]
                    acenter = m2 / m1
                    awidth = np.sqrt(m3 / m1 - (m2 / m1) ** 2)
                    print(
                        "iorb " + str(iorb + 1) + " center:", acenter, "width:", awidth
                    )
                    emax = max(round(abs(acenter) + awidth * 30), emax)
        # emax = 10
        print("emax:", emax)
        print("\n")

        for ik in range(1):
            for js in range(ns):
                for iorb in range(norb):
                    print("--------------------------")
                    print(
                        "orb:  " + str(iorb + 1) + "  spin:  "+ str(js + 1)+ "  k:  "+ str(ik + 1)
                    )
                    print("--------------------------")
                    f = open("gaux.dat", "w")
                    h = open(
                        "original_"+ str(iorb + 1)+ "_"+ str(js + 1)+ "_"+ str(ik + 1),"w",
                    )
                    for iomega in range(nomega):
                        f.write(
                            "%5s %3s %3s %20.10f %20.10f \n"
                            % (
                                iomega,
                                0,
                                0,
                                np.real(gauxmat[iorb, iorb, js, ik, iomega]),
                                np.imag(gauxmat[iorb, iorb, js, ik, iomega]),
                            )
                        )
                        h.write(
                            "%20.10f %20.10f %20.10f \n"
                            % (
                                omega[iomega],
                                np.real(gauxmat[iorb, iorb, js, ik, iomega]),
                                np.imag(gauxmat[iorb, iorb, js, ik, iomega]),
                            )
                        )
                    f.close()
                    h.close()
                    f = open("mqem.input.toml", "w")
                    f.write('inputFile = "gaux.dat"\n')
                    f.write("NumOrbit = 1\n")
                    f.write("inverse_temp =  " + str(self.dlr.beta) + "\n")
                    f.write("Egrid = 400\n")
                    f.write("EwinOuterRight = " + str(np.real(emax)) + "\n")
                    f.write("EwinOuterLeft = -" + str(np.real(emax)) + "\n")
                    f.write("EwinInnerRight = " + str(np.real(emax) / 10) + "\n")
                    f.write("EwinInnerLeft = -" + str(np.real(emax) / 10) + "\n")
                    f.write("blur = " + str(blur) + "\n")

                    f.write('default_model="' + defaultmodel + '"\n')
                    f.close()
                    print("run MQEM")
                    subprocess.call("julia  $QAssemble/MQEM.jl/src/mem.jl", shell=True)
                    print("run MQEM")
                    shutil.copy("mqem.input.toml", "mqem_input_" + str(iorb + 1) + "_" + str(js + 1)+ "_" + str(ik + 1)+ ".toml")
                    if os.path.exists("realFreq_Sw.dat_1_1"):
                        gaux_real = np.loadtxt("realFreq_Sw.dat_1_1")
                    else:
                        print("maxent for orbital " + str(iorb + 1) + " failed")
                        sys.exit()
                    freal_temp = gaux_real[:, 0]
                    ecutoff = min(15, emax / 4)
                    n1 = np.argmin(abs(freal_temp + ecutoff))
                    n2 = np.argmin(abs(freal_temp - ecutoff))
                    nf = n2 - n1

                    if iorb == 0:
                        sig_real = np.zeros(
                            (norb, norb, ns, 1, nf), dtype=np.complex128, order="F"
                        )
                    if gauxmode == "asitis":
                        omega_real = gaux_real[n1:n2, 0]
                        sig_real[iorb, iorb, js, 0, :] = (
                            gaux_real[n1:n2, 1] + gaux_real[n1:n2, 2] * 1j
                        )
                    elif gauxmode == "auxg":
                        omega_real = gaux_real[n1:n2, 0]
                        sigout = gaux_real[n1:n2, 0] - 1.0 / (
                            gaux_real[n1:n2, 1] + gaux_real[n1:n2, 2] * 1j
                        )
                        sig_real[iorb, iorb, js, ik, :] = sigout + high_temp[iorb, iorb, js, ik]

                    os.remove("realFreq_Sw.dat_1_1")
                    os.remove("Sw_SOLVER.full_fromRetardedSw.dat_0_0")
                    os.remove("gaux.dat")

                    shutil.move(
                        "spectral_function_0_0.dat_model", "gaux_spectra_model_" + str(iorb + 1)+ "_"+ str(js + 1)+ "_"+ str(ik + 1),
                    )
                    shutil.move(
                        "spectral_function_0_0.dat","gaux_spectra_"+ str(iorb + 1)+ "_"+ str(js + 1)+ "_"+ str(ik + 1),
                    )
                    shutil.move(
                        "information.out",
                        "info_" + str(iorb + 1) + "_" + str(js + 1) + "_" + str(ik + 1),
                    )
                    shutil.move(
                        "reproduce_0_0.out",
                        "reproduced_"
                        + str(iorb + 1)
                        + "_"
                        + str(js + 1)
                        + "-"
                        + str(ik + 1),
                    )
                    print("\n")
                    print("\n")
                    print("\n")
                    print("\n")
                    print("\n")

        if interpolation:

            sig_real_bare = sig_real
            deltae = min(blur, 0.01)
            nn = int(min(ecutoff, 12) / deltae)
            sig_real = np.zeros(
                (norb, norb, ns, 1, 2 * nn + 1), dtype=np.complex128, order="F"
            )
            xnew = (np.arange(2 * nn + 1) - nn) * deltae

            for ik in range(1):
                for js in range(ns):
                    for iorb in range(norb):
                        x = omega_real
                        y = sig_real_bare[iorb, iorb, js, ik]
                        f = interpolate.interp1d(x, y, kind="cubic")
                        tempdat = f(xnew)
                        if gaussian_broadening > 0:
                            width = int(gaussian_broadening / (xnew[1] - xnew[0]))
                            sig_real[iorb, iorb, js, ik, :] = gaussian_filter1d(
                                tempdat, width
                            )
                        else:
                            sig_real[iorb, iorb, js, ik, :] = tempdat

        print("Analytic continuation has finished:)")

        return xnew, sig_real

    def Spectral(self, option: dict = None):

        print("Spectral calculation start")
        gauxmode = option["gauxmode"]
        glob = h5py.File(self.hdf5file, "r")
        gw = glob["gw"]
        if gauxmode == "asisit":
            kf = gw["GreenInt"]["gkf"][:]
        if gauxmode == "auxg":
            sigmah = gw["SigmaHartree"]["sigmah"][:]
            sigmaf = gw["SigmaFock"]["sigmaf"][:]
            kf = gw["SigmaGWC"]["sigmagwckf"][:]
            # kf = np.zeros_like(tempmat3,dtype=np.complex128,order='F')
            # for iomega in range(kf.shape[4]):
            #     kf[...,iomega] += tempmat+tempmat2+tempmat3[...,iomega]
        glob.close()
        # kpath = option['kpath']
        # nkpath = option['nkpath']
        # self.crystal.Kpath(kpath=kpath,nk=nkpath)
        # print("Fourier transform K2R start")
        # rf = self.flatdyn.K2R(kf,rkgrid=self.crystal.rkgrid)
        # print("Fourier transform K2R finish")
        # print("Fourier transform R2Kpath start")
        # kpathf = self.R2K(rf, self.crystal.kpath)
        # print("Fourier transform R2Kpath finish")

        print("MQEM calculation start")
        tempmat = self.MQEMWrapper(option=option, gmat=kf)
        print("MQEM calculation finish")
        (norb, norb, ns, nk, nf) = tempmat.shape
        sigout = np.zeros((ns, nk, nf), dtype=np.complex128, order="F")
        gout = np.zeros((ns, nk, nf), dtype=np.complex128, order="F")

        for iorb in range(norb):
            sigout += 1 / norb * tempmat[iorb, iorb]

        if target == "sig":
            gout = sigout

        if target == "g":
            gout = -1 / np.pi * sigout.imag

        self.sigout = sigout
        self.gout = gout

        return None

    def MQEMPrepare(self, gmat : np.ndarray = None, omega : np.ndarray = None):

        norb, _, ns, nk, nomega = gmat.shape

        for ik in range(nk):
            if os.path.isdir("mqem_run_" + str(ik + 1)):
                pass
            else:
                os.mkdir("mqem_run_" + str(ik + 1))

            os.chdir("mqem_run_" + str(ik + 1))
            f = open("green.dat", "w")
            h = open("info.dat", "w")
            h.write("# norb ns nomega\n")
            h.write("%3s %3s %5s\n" % (norb, ns, nomega))
            h.close()            
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):   
                        for iomega in range(nomega):
                            f.write(
                                "%3s %3s %3s %5s %20.10f %20.10f \n"
                                % (
                                    iorb,
                                    jorb,
                                    js,
                                    iomega,
                                    np.real(gmat[iorb, jorb, js, ik, iomega]),
                                    np.imag(gmat[iorb, jorb, js, ik, iomega]),
                                )
                            )
            f.close()
            f = open("omega.dat", "w")
            for iomega in range(nomega):
                f.write("%20.10f\n" % omega[iomega])
            f.close()
            os.chdir("..")
        
        return None