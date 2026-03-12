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
from .BLatStc import BLatStc

class BPathStc(object):

    def __init__(self, crystal : Crystal = None, obj : object = None, hdf5file : str = 'glob.h5') -> object:
              
        if (crystal is not None) and (obj is not None):
            pass
        else:
            if os.path.exists(hdf5file):
                glob = h5py.File(hdf5file)
                ini = glob['input']
                tempcry = ini['Crystal']
                cry = {}
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
                
#               crystal = Crystal(Rvec=cry['RVec'],CorF=cry['CorF'],Basis=cry['Basis'],Nspin=cry['NSpin'],SOC=cry['SOC'],Nelec=cry['NElec'],KGrid=cry['KGrid'])
                crystal = Crystal(cry=cry)
                glob.close()
            else:
                print(f"Error : Check the {self.__class__.__name__} input again")
                sys.exit()
        
        self.crystal = crystal
        self.blatstc = BLatStc(crystal=self.crystal)

    def R2K(self, matr : np.ndarray = None, kpoint = None) -> np.ndarray:

        norb = matr.shape[0]
        ns = matr.shape[2]
        nr = matr.shape[4]
        nk = len(kpoint)

        self.crystal.Rvec()

        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb,norb,ns,ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            temp = 0
                            for ir in range(nr):
                                temp += tempmat[iorb,jorb,js,ks,ir] * np.exp(-2.0j*np.pi*(kpoint[ik]@self.crystal.rvec[ir]))
                            [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                            [b,[m2,m3]] = self.crystal.BAtomOrb(jorb)
                            delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
                            phase = np.exp(-2.0j*np.pi*(kpoint[ik]@delta))
                            matk[iorb,jorb,js,ks,ik] = temp*phase

        return matk
