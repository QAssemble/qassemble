import string as string
from typing import Any
import matplotlib as mat
import re as re
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import matplotlib.font_manager as fm
from collections import OrderedDict
import json, os, shutil, sys
import itertools
import scipy.optimize
from sympy.physics.wigner import gaunt, wigner_3j
from scipy.fftpack import fftn, ifftn
import scipy.linalg
import subprocess
import copy
from core.Crystal import Crystal
from core.FTGrid import FTGrid


class FPathDyn(object):

    def __init__(self, crystal : Crystal, ft : FTGrid, obj : object, kpath : list, nk : int):

        self.crystal = crystal
        self.ft = ft
        self.kpath = self.crystal.Kpath(kpath=kpath,nk=nk)
        self.k = None

        if (obj.__class__.__name__ is "GreenInt"):
            self.k = self.KArb(obj.rf,kpoint=self.kpath)
        elif (obj.__class__.__name__ is "GreenBare"):
            self.k = self.KArb(obj.g0rf,kpoint=self.kpath)
        elif (obj.__class__.__name__ is "SigmaGWC"):
            self.k = self.R2K(obj.rf,self.kpath)
        
    
    def Inverse(self, mat : np.ndarray) -> np.ndarray:

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]
        nft = mat.shape[4]

        matinv = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    matinv[:,:,js,irk,ift] = np.linalg.inv(mat[:,:,js,irk,ift])
        # for js, irk, ift in itertools.product(list(range(ns)),list(range(nrk),list(range(nft)))):
        #     matinv[:,:,js,irk,ift] = np.linalg.inv(mat[:,:,js,irk,ift])
        
        return matinv

    def R2K(self,matr : np.ndarray = None,kpoint : np.ndarray = None): # R2KAny

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nr = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        nk = len(kpoint)
        nft = matr.shape[4]

        self.crystal.Rvec()
        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb,norb,ns,nk,nft),dtype=complex,order='F')

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
    
    def KArb(self, matr : np.ndarray = None, kpoint : np.ndarray = None): ## naming

        norb = matr.shape[0]
        ns = matr.shape[2]
        nr = matr.shape[3]
        nfreq = matr.shape[4]
        nk = len(kpoint)

        tempmat = np.zeros((norb,norb,ns,nr,nfreq),dtype=complex,order='F')
        matkinv = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,order='F')

        matrinv = self.Inverse(matr)
        omega = self.ft.omega

        for ifreq in range(nfreq):
            for ir in range(nr):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            if iorb==jorb:
                                tempmat[iorb,jorb,js,ir,ifreq] = 1j*omega[ifreq]-matrinv[iorb,jorb,js,ir,ifreq]
                            else:
                                tempmat[iorb,jorb,js,ir,ifreq] = -matrinv[iorb,jorb,js,ir,ifreq]

        tempmat2 = self.R2K(tempmat,kpoint)

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