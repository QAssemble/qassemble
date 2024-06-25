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
from pymatgen.core import Lattice, Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
import subprocess
import copy
diage_path = os.environ.get('DIAGE','')
path = diage_path+"/modules"
sys.path.append(path)
import DiagE

class FTGrid(object):

    def __init__(self,T : float = 300,beta : float = None,size : int = 1000) -> object:
        
        if beta == None:
            self.T = T
            self.beta = 1/(T*8.6173303*10**-5)
        else:
            self.beta = beta
            self.T = 1/(beta*8.6173303*10**-5)
        self.size = size
        self.omega = np.zeros((size),dtype=float,order='F')
        self.nu = np.zeros((size),dtype=float,order='F')
        self.tau = np.zeros((int(size*2)),dtype=float,order='F')

        self.Omega()
        self.Tau()
        self.Nu()

    def Omega(self) -> np.ndarray:

        nomega = int(self.size)#self.size
        for iomega in range(nomega):
            self.omega[iomega] = np.pi/self.beta*(2*iomega+1)

        return None

    # def Tau(self) -> np.ndarray:

    #     ntau = int(self.size)
        
    #     for itau in range(int(ntau)):
    #         itheta = DiagE.common.ttind(itau,ntau)
    #         self.tau[itau] = self.beta/2.0*(np.cos(np.pi*(itheta+0.5)/ntau)+1.0)

    def Tau(self):

        ntau = len(self.tau)
        # meshscale = (ntau/2)**5
        # prefac = (self.beta/2)/meshscale
        
        # for itau in range(ntau//2):
        #     tauindex = float(itau)**5
        #     if itau == 0:
        #         self.tau[itau] = 1e-16*self.beta
        #     else:
        #         self.tau[itau] = prefac*tauindex
        #     self.tau[ntau-1-itau] = self.beta - self.tau[itau]
        for itau in range(ntau):
            itheta = DiagE.common.ttind(itau,ntau)
            self.tau[itau] = self.beta/2.0*(np.cos(np.pi*(itheta+0.5)/ntau)+1.0)
        
        return None



    def Nu(self) -> np.ndarray:

        nnu = self.size
        for inu in range(nnu):
            self.nu[inu] = np.pi/self.beta*(2*inu)

        return None