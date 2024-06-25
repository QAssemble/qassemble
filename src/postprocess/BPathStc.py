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

class BPathStc(object):

    def __init__(self, crystal : Crystal = None, obj : object = None) -> object:
              
        if (crystal is None) or (obj is None):
            print(f"Error : Check the {self.__class__.__name__} input again")
            sys.exit()
        
        self.crystal = crystal

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
