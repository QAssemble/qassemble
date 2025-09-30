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
from .Crystal import Crystal
from .FTGrid import FTGrid
from .FLatDyn import GreenInt
qapath = os.environ.get('QAssemble','')
sys.path.append(qapath+'/src/QAssemble/modules')
import QAFort

class FLocDyn(object):

    def __init__(self,crystal : Crystal, ft : FTGrid):
        
        self.crystal = crystal
        self.ft = ft

    def Inverse(self, mat : np.ndarray):
        
        norb = mat.shape[0]
        ns = mat.shape[2]
        nft = mat.shape[3]

        matinv = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            for js in range(ns):
                matinv[:,:,js,ift] = np.linalg.inv(mat[:,:,js,ift])

        return matinv
    
    def Moment(self, ff : np.ndarray, isgreen : int, highzero : int) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]

        moment = np.zeros((norb,norb,ns,3),dtype=np.complex128,order='F')
        high = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        moment, high = QAFort.fourier.flocdyn_m(self.ft.omega,ff,isgreen,highzero)

        return moment, high
    
    def F2T(self, ff : np.ndarray, isgreen : int, highzero : int) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        ntau = len(self.ft.tau)

        ftau = np.zeros((norb,norb,ns,ntau),dtype=np.complex128,order='F')

        moment, high = self.Moment(ff,isgreen,highzero)

        ftau = QAFort.fourier.flocdyn_f2t(self.ft.omega,ff,moment,self.ft.tau)

        return ftau
        
    def T2F(self,ftau : np.ndarray) -> np.ndarray:

        norb = ftau.shape[0]
        ns = ftau.shape[2]
        nfreq = len(self.ft.omega)

        ff = np.zeros((norb,norb,ns,nfreq),dtype=np.complex128,order='F')

        ff = QAFort.fourier.flocdyn_t2f(self.ft.tau,ftau,self.ft.omega)

        return ff
    
    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]

        ynew = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')
        w0 = (1.0 - 3.0*w1)*np.pi*temperature
        widtharray = w0+w1*x
        cnt = 0

        for x0 in x:
            if (x0>cutoff+(w0+w1*cutoff)*3.0):
                ynew[...,cnt]=y[...,cnt]
            else:
                if ((x0>3*widtharray[cnt])and((x[-1]-x0)>3*widtharray[cnt])):
                    dist = 1.0/np.sqrt(2*np.pi)/widtharray[cnt]*np.exp(-(x-x0)**2/2.0/widtharray[cnt]**2)
                    for js in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                ynew[iorb,jorb,js,cnt] = sum(dist*y[iorb,jorb,js])/sum(dist)
                else:
                    ynew[...,cnt] = y[...,cnt]
            cnt += 1

        return ynew
    
    def Mixing(self,iter : int, mix : float, Fb : np.ndarray, Fold : np.ndarray):

        norb = Fb.shape[0]
        ns = Fb.shape[2]
        nft = Fb.shape[3]

        Fnew = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        if iter == 1:
            mix = 1.0
            Fold = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        Fnew = mix*Fb+(1.0-mix)*Fold

        return Fnew
    
    def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]
        nft = matimp.shape[3]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,nft,nspace),dtype=np.complex128,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            for ispace in val:
                matloc[...,ispace] = matimp[...,iprob]

        return matloc
    
    def Loc2Imp(self,matloc : np.ndarray)->np.ndarray:

        nprob = len(self.crystal.probspace)
        norb = matloc.shape[0]
        ns = matloc.shape[2]
        nft = matloc.shape[3]

        matimp = np.zeros((norb,norb,ns,nft,nprob),dtype=np.complex128,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            tempmat = np.zeros((norb,norb,ns, nft),dtype=np.complex128)
            for ispace in val:
                tempmat += matloc[...,ispace]
            tempmat /=len(val)
            matimp[...,iprob] = tempmat

        return matimp
    
    def Arr2Dict(self, equiv : np.ndarray, matin : np.ndarray) -> dict:
        
        ns = matin.shape[2]
        nind = np.amax(equiv)
        matdict = {}

        for ind in range(nind):
            matdict[ind+1] = []
            pos = self.crystal.FindPositions(equiv,ind+1)
            for js in range(ns):
                e = 0
                for ii, jj in pos:
                    e += matin[ii,jj,js]
                e /=len(pos)
                matdict[ind+1].append(e.tolist())
        
        return matdict
    
    def Dict2Arr(self,equiv : np.ndarray, matdict : np.ndarray) -> np.ndarray:

        norb = len(equiv)
        ns = self.crystal.ns
        nfreq = len(matdict["1"])                

        matout = np.zeros((norb,norb,ns,nfreq),dtype=np.complex128,order='F')
        nind = np.amax(equiv)

        for js in range(ns):
            for ind in range(nind):
                pos = self.crystal.FindPositions(equiv,ind+1)
                for ii, jj in pos:
                    matout[ii,jj,js] = matdict[str(ind+1)]

        return matout
    
    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nft = self.ft.size

        matout = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        matout = QAFort.dyson.flocdyn(mat1,mat2)

        return matout

    def Embedding(self, matin : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size
        nspace = self.crystal.fprojector.shape[3]

        matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ispace in range(nspace):
            matout += QAFort.embedding.flocdyn(nrk,matin[...,ispace],self.crystal.fprojector[...,ispace])

        return matout

    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]
        nft = matin.shape[3]

        if os.path.exists('flocdyn'):
            pass
        else:
            os.mkdir("flocdyn")
        os.chdir("flocdyn")
        with open(fn+'.txt','w') as f:
            f.write("iorb, jorb, is, ift, Re(F(w)), Im(F(w))\n")
            for ift in range(nft):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            f.write(f"{iorb} {jorb} {js} {ift} {matin[iorb,jorb,js,ift].real} {matin[iorb,jorb,js,ift].imag}\n")
        os.chdir("..")
        return None
    
class GreenLoc(FLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, green : GreenInt):
        
        super().__init__(crystal, ft)
        self.green = green
        self.gf = None
        self.gt = None
        
        self.Cal()

    def Cal(self): # projection
        
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nft = self.ft.size
        nspace = self.crystal.fprojector.shape[3]

        gf = np.zeros((norbc,norbc,ns,nft,nspace),dtype=np.complex128)

        for ispace in range(nspace):
            gf[...,ispace] = QAFort.projection.flatdyn(self.green.gkf,self.crystal.fprojector[...,ispace])

        self.gf = gf
        self.gt = self.F2T(gf,1,1)

        return None

class GreenImp(FLocDyn): # read CTQMC output

    def __init__(self, crystal: Crystal, ft: FTGrid):
        super().__init__(crystal, ft)
        self.Cal()

    def Cal(self):
        super().Dict2Arr()
        pass

class SigmaLoc(FLocDyn):
    
    def __init__(self, crystal: Crystal, ft: FTGrid, sigma : object):
        super().__init__(crystal, ft)
        
        self.sigma = sigma
        self.f = None
        self.Cal()

    def Cal(self): # projection
        
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nft = self.ft.size
        nspace = self.crystal.fprojector.shape[3]

        sigmalocf = np.zeros((norbc,norbc,ns,nft,nspace),dtype=np.complex128,order='F')

        for isapce in range(nspace):
            sigmalocf[...,isapce] = QAFort.projection.flatdyn(self.sigma,self.crystal.fprojector[...,isapce])

        self.f = sigmalocf
        self.t = self.F2T(sigmalocf,0,1)

        return None


class SigmaImp(FLocDyn): # read CTQMC output

    def __init__(self, crystal: Crystal, ft: FTGrid):
        super().__init__(crystal, ft)
        self.Cal()

    def Cal(self):
        super().Dict2Arr()
        pass

class SigmaLGWC(FLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid):
        super().__init__(crystal, ft)

        pass
    

class Hybridisation(FLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, implev : object, gimp : GreenImp, sigmaimp : SigmaImp):
        super().__init__(crystal, ft)
        self.Cal()
    
    def Cal(self):
        pass
