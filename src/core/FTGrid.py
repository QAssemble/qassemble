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

    def __init__(self,T : float = 300,beta : float = None,cutoff : int = 50) -> object:
        
        if beta == None:
            self.T = T
            self.beta = 1/(T*8.6173303*10**-5)
        else:
            self.beta = beta
            self.T = 1/(beta*8.6173303*10**-5)
        self.cutoff = cutoff
        self.omega = None
        self.nu = None
        self.tau = None

        self.Omega()
        self.Tau()
        self.Nu()

    def Omega(self) -> np.ndarray:

        # nomega = int(self.size)#self.size
        # for iomega in range(nomega):
        #     self.omega[iomega] = np.pi/self.beta*(2*iomega+1)
        omega = []
        for i in range(1000000):
            w = (2.0*float(i)+1)*np.pi/self.beta
            if (w > self.cutoff):
                break
            omega.append(w)
        self.omega = np.array(omega,dtype=float,order='F')

        return None

    # def Tau(self) -> np.ndarray:

    #     ntau = int(self.size)
        
    #     for itau in range(int(ntau)):
    #         itheta = DiagE.common.ttind(itau,ntau)
    #         self.tau[itau] = self.beta/2.0*(np.cos(np.pi*(itheta+0.5)/ntau)+1.0)

    def Tau(self):

        ntau = int(len(self.omega)*2)
        # meshscale = (ntau/2)**5
        # prefac = (self.beta/2)/meshscale
        
        # for itau in range(ntau//2):
        #     tauindex = float(itau)**5
        #     if itau == 0:
        #         self.tau[itau] = 1e-16*self.beta
        #     else:
        #         self.tau[itau] = prefac*tauindex
        #     self.tau[ntau-1-itau] = self.beta - self.tau[itau]
        tau = np.zeros((ntau),dtype=float,order='F')
        for itau in range(ntau):
            itheta = DiagE.common.ttind(itau,ntau)
            tau[itau] = self.beta/2.0*(np.cos(np.pi*(itheta+0.5)/ntau)+1.0)

        self.tau = tau
        
        return None



    def Nu(self) -> np.ndarray:

        # nnu = self.size
        # for inu in range(nnu):
        #     self.nu[inu] = np.pi/self.beta*(2*inu)
        nu = []
        for i in range(1000000):
            w = (2.0*float(i))*np.pi/self.beta
            if (w > self.cutoff):
                break
            nu.append(w)

        self.nu = np.array(nu, dtype=float,order='F')

        return None