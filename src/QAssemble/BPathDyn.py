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
import h5py
from .Crystal import Crystal
from .FTGrid import FTGrid
from .BLatDyn import BLatDyn

class BPathDyn(object):

    def __init__(self, crystal : Crystal = None, ft : FTGrid = None, obj : object = None, hdf5file : str = 'glob.h5') -> object:

        if (crystal is not None)and(ft is not None)and(obj is not None):
            pass
        else:
            if os.path.exists(hdf5file):
                glob = h5py.File(hdf5file)
                ini = glob['input']
                tempcry = ini['Crystal']
                control = ini['Control']
                cry = {}
                kb = 8.6173303*10**-5
                for key in tempcry.keys():
                    if (type(tempcry[key][()])==bytes):
                        cry[key] = str(tempcry[key][()],'utf-8')
                    else:
                        cry[key] = tempcry[key][()]
                for key in cry.keys():
                    if key=='Basis':
                        cry[key] = eval(cry[key])
                    elif key=='KGrid':
                        cry[key] = eval(cry[key])
                    elif key=='RVec':
                        cry[key] = eval(cry[key])
                    else:
                        cry[key] = cry[key]
                
                crystal = Crystal(Rvec=cry['RVec'],CorF=cry['CorF'],Basis=cry['Basis'],Nspin=cry['NSpin'],SOC=cry['SOC'],Nelec=cry['NElec'],KGrid=cry['KGrid'])
                if ('T' in control)and('beta' not in control):
                    T = control['T'][()]
                    beta = 1/(T*kb)
                elif ('T' not in control)and('beta' in control):
                    beta = control['beta'][()]
                    T = 1/(beta*kb)
                cutoff = control.get('MatsubaraCutOff',50)
                ft = FTGrid(T=T,beta=beta,cutoff=cutoff)
                glob.close()
            else:
                print(f"Error : Check the {self.__class__.__name__} input again")
                sys.exit()

        
        self.crystal = crystal
        self.ft = ft
        self.blatdyn = BLatDyn(crystal=self.crystal,ft=self.ft)
    
    def R2K(self, matr : np.ndarray = None, kpoint = None) -> np.ndarray:

        norb = matr.shape[0]
        ns = matr.shape[2]
        nr = matr.shape[4]
        nft = matr.shape[5]
        nk = len(kpoint)

        self.crystal.Rvec()

        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb,norb,ns,ns,nk,nft),dtype=complex,order='F')

        for ift in range(nft):
            for ik in range(nk):
                for ks in range(ns):
                    for js in range(ns):
                        for jorb in range(norb):
                            for iorb in range(norb):
                                temp = 0
                                for ir in range(nr):
                                    temp += tempmat[iorb,jorb,js,ks,ir,ift] * np.exp(-2.0j*np.pi*(kpoint[ik]@self.crystal.rvec[ir]))
                                [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                                [b,[m2,m3]] = self.crystal.BAtomOrb(jorb)
                                delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
                                phase = np.exp(-2.0j*np.pi*(kpoint[ik]@delta))
                                matk[iorb,jorb,js,ks,ik,ift] = temp*phase

        return matk
