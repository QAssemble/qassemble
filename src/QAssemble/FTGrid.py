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
from .utility.Common import Common
# qapath = os.environ.get('QAssemble','')
# sys.path.append(qapath+'/src/QAssemble/modules')
# import QAFort

class FTGrid(object):

    def __init__(self,ft : dict = None) -> object:

        #self.T = ft.get('T',300) #ft['T']
        #self.beta = ft['beta']
        if ('T' not in ft):
            self.beta = ft['beta']
            self.T = 1/(self.beta*8.6173303*10**-5)

        elif ('beta' not in ft):
            self.T = ft['T']
            self.beta = 1/(self.T*8.6173303*10**-5)
        else:
            self.T = ft['T']
            self.beta = ft['beta']
        self.cutoff = ft['cutoff']

        self.omega = self.Omega()
        self.nu = self.Nu()
        self.tau = self.Tau()       
        
        

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
        # self.omega = np.array(omega,dtype=float,order='F')
        omega = np.array(omega,dtype=float,order='F')

        return omega

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
            itheta = Common.Ttind(itau,ntau)
            tau[itau] = self.beta/2.0*(np.cos(np.pi*(itheta+0.5)/ntau)+1.0)

        # self.tau = tau

        return tau

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

        # self.nu = np.array(nu, dtype=float,order='F')
        nu = np.array(nu, dtype=float,order='F')

        return nu
