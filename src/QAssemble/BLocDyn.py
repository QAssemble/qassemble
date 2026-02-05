# import string as string
# from typing import Any
# import matplotlib as mat
# import re as re
# import matplotlib.pyplot as plt
# import numpy as np
# from pylab import cm
# import matplotlib.font_manager as fm
# from collections import OrderedDict
# import json, os, shutil, sys
# import itertools
# import scipy.optimize
# from sympy.physics.wigner import gaunt, wigner_3j
# from scipy.fftpack import fftn, ifftn
# import scipy.linalg
# from pymatgen.core import Lattice, Structure
# from pymatgen.transformations.standard_transformations import SupercellTransformation
# import subprocess
# import copy
# # import Crystal, FTGrid
# from .Crystal import Crystal
# from .FTGrid import FTGrid
# from .FLocDyn import FLocDyn
# from .utility.Common import Common
# from .utility.Fourier import Fourier
# from .utility.Dyson import Dyson

# class BLocDyn(object):

#     def __init__(self, crystal : Crystal, ft : FTGrid):

#         self.crystal = crystal
#         self.ft = ft

#     def Inverse(self, matin : np.ndarray)-> np.ndarray:

#         norb = matin.shape[0]
#         ns = matin.shape[2]
#         nft = matin.shape[4]

#         matout = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')
#         tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex128)
#         tempmat2 = np.zeros((norb*ns,norb*ns),dtype=np.complex128)

#         for ift in range(nft):
#             tempmat = self.crystal.OrbSpin2Composite(matin[...,ift])
#             tempmat2 = np.linalg.inv(tempmat)
#             matout[...,ift] = self.crystal.Composite2OrbSpin(tempmat2)
        
#         return matout

#     def Moment(self, bf : np.ndarray, oddzero : int, highzero : int) -> np.ndarray:

#         norb = len(self.crystal.bind)
#         ns = self.crystal.ns

#         moment = np.zeros((norb,norb,ns,ns,3),dtype=np.complex128,order='F')
#         high = np.zeros((norb,norb,ns,ns),dtype=np.complex128,order='F')
#         moment, high = Fourier.BLocDynM(self.ft.nu,bf,oddzero,highzero)

#         return moment,high
    
#     def F2T(self,bf : np.ndarray, oddzero : int, highzero : int) -> np.ndarray:

#         norb = len(self.crystal.bind)
#         ns = self.crystal.ns
#         # nft = self.ft.size
#         nft = bf.shape[4]

#         btau = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

#         moment, high = self.Moment(bf,oddzero,highzero)

#         btau = Fourier.BLocDynF2T(self.ft.nu,bf,moment,self.ft.tau)

#         return btau

#     def T2F(self, btau : np.ndarray) -> np.ndarray:

#         norb = len(self.crystal.bind)
#         ns = self.crystal.ns
#         nft = self.ft.size 

#         bf = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

#         bf = QAFort.fourier.blocdyn_t2f(self.ft.tau,btau,self.ft.nu)

#         return bf

#     def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

#         norb = y.shape[0]
#         ns = y.shape[2]
#         nft = y.shape[3]

#         ynew = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')
#         w0 = (1.0 - 3.0*w1)*np.pi*temperature
#         widtharray = w0+w1*x
#         cnt = 0

#         for x0 in x:
#             if (x0>cutoff+(w0+w1*cutoff)*3.0):
#                 ynew[...,cnt]=y[...,cnt]
#             else:
#                 if ((x0>3*widtharray[cnt])and((x[-1]-x0)>3*widtharray[cnt])):
#                     dist = 1.0/np.sqrt(2*np.pi)/widtharray[cnt]*np.exp(-(x-x0)**2/2.0/widtharray[cnt]**2)
#                     for js in range(ns):
#                         for ks in range(ns):
#                             for iorb in range(norb):
#                                 for jorb in range(norb):
#                                     ynew[iorb,jorb,js,ks,cnt] = sum(dist*y[iorb,jorb,js,ks])/sum(dist)
#                 else:
#                     ynew[...,cnt] = y[...,cnt]
#             cnt += 1

#         return ynew
    
#     def Mixing(self,iter : int, mix : float, Bb : np.ndarray, Bold : np.ndarray):

#         norb = Bb.shape[0]
#         ns = Bb.shape[2]
#         nft = Bb.shape[4]

#         Bnew = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

#         if iter == 1:
#             mix = 1.0
#             Bold = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

#         Bnew = mix*Bb + (1-mix)*Bold

#         return Bnew
    
#     def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

#         norb = matimp.shape[0]
#         ns = matimp.shape[2]
#         nft = matimp.shape[3]

#         nspace = 0
#         for val in self.crystal.probspace.values():
#             nspace += len(val)

#         matloc = np.zeros((norb,norb,ns,ns,nft,nspace),dtype=np.complex128,order='F')

#         for key, val in self.crystal.probspace.items():
#             iprob = int(key)-1
#             for ispace in val:
#                 matloc[...,ispace] = matimp[...,iprob]

#         return matloc
    
#     def Loc2Imp(self,matloc : np.ndarray)->np.ndarray:

#         nprob = len(self.crystal.probspace)
#         norb = matloc.shape[0]
#         ns = matloc.shape[2]
#         nft = matloc.shape[3]

#         matimp = np.zeros((norb,norb,ns,ns,nft,nprob),dtype=np.complex128,order='F')

#         for key, val in self.crystal.probspace.items():
#             iprob = int(key)-1
#             tempmat = np.zeros((norb,norb,ns),dtype=np.complex128)
#             for ispace in val:
#                 tempmat += matloc[...,ispace]
#             tempmat /=len(val)
#             matimp[...,iprob] = tempmat

#         return matimp
    
#     def Arr2Dict(self, equiv : np.ndarray, matin : np.ndarray) -> dict:
        
#         ns = matin.shape[2]
#         nind = np.amax(equiv)
#         matdict = {}

#         for ind in range(nind):
#             matdict[ind+1] = []
#             pos = self.crystal.FindPositions(equiv,ind+1)
#             for js in range(ns):
#                 for ks in range(ns):
#                     e = 0
#                     for ii, jj in pos:
#                         e += matin[ii,jj,js,ks]
#                     e /=len(pos)
#                     matdict[ind+1].append(e.tolist())
        
#         return matdict
    
#     def Dict2Arr(self,equiv : np.ndarray, matdict : np.ndarray) -> np.ndarray:

#         norb = len(equiv)
#         ns = self.crystal.ns
#         nfreq = len(matdict["1"])                

#         matout = np.zeros((norb,norb,ns,ns,nfreq),dtype=np.complex128,order='F')
#         nind = np.amax(equiv)

#         for js in range(ns):
#             for ks in range(ns):
#                 for ind in range(nind):
#                     pos = self.crystal.FindPositions(equiv,ind+1)
#                     for ii, jj in pos:
#                         matout[ii,jj,js,ks] = matdict[str(ind+1)]

#         return matout

#     def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

#         norb = mat1.shape[0]
#         ns = self.crystal.ns
#         nft = self.ft.size

#         matout = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

#         matout = QAFort.dyson.blocdyn(mat1,mat2)

#         return matout

#     def Embedding(self, matin : np.ndarray):

#         norb = len(self.crystal.bind)
#         ns = self.crystal.ns
#         nrk = len(self.crystal.kpoint)
#         nft = self.ft.size
#         nspace = self.crystal.bprojector.shape[3]

#         matout = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex128,order='F')

#         for ispace in range(nspace):
#             matout += QAFort.embedding.blocdyn(nrk,matin[...,ispace],self.crystal.bprojector[...,ispace])

#         return matout
    
#     def Save(self,matin : np.ndarray, fn : str):

#         norb = matin.shape[0]
#         ns = matin.shape[2]
#         nft = matin.shape[4]

#         if os.path.exists('blocdyn'):
#             pass
#         else:
#             os.mkdir('blocdyn')
#         os.chdir('blocdyn')

#         with open(fn+'txt','w') as f:
#             f.write("iorb, jorb, is, js, ift, Re(B(w)), Im(B(w))\n")
#             for ift in range(nft):
#                 for ks in range(ns):
#                     for js in range(ns):
#                         for jorb in range(norb):
#                             for iorb in range(norb):
#                                 f.write(f"{iorb} {jorb} {js} {ks} {matin[iorb,jorb,js,ks,ift].real} {matin[iorb,jorb,js,ks,ift].imag}\n")
        
#         os.chdir('..')
#         return None

# class PolLoc(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, green, pol : object):
#         super().__init__(crystal, ft)
#         self.Cal()

#     def Cal(self):
#         pass

# class PolImp(BLocDyn): # read Polarizability from CTQMC

#     def __init__(self, crystal: Crystal, ft: FTGrid):
#         super().__init__(crystal, ft)

#         pass

# class WLoc(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass

# class WImp(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass

# class WcLoc(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass

# class WcImp(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass
