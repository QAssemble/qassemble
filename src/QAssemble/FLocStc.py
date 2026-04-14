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
# from .Crystal import Crystal
# from .FLatStc import NIHamiltonian
# from .FLocDyn import GreenLoc
# qapath = os.environ.get('QAssemble','')
# sys.path.append(qapath+'/src/QAssemble/modules')
# import QAFort

# class FLocStc(object):

#     def __init__(self,crystal : Crystal):

#         self.crystal = crystal

#     def Inverse(self,mat : np.ndarray):

#         norb = mat.shape[0]
#         ns = mat.shape[2]

#         matinv = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

#         for js in range(ns):
#             matinv[:,:,js] = np.linalg.inv(mat[:,:,js])

#         return matinv
    
#     def Mixing(self,iter : int, mix : float, Fb : np.ndarray, Fold : np.ndarray) -> np.ndarray:

#         norb = Fb.shape[0]
#         ns = Fb.shape[2]

#         Fnew = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

#         if iter == 1:
#             mix = 1.0
#             Fold = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

#         Fnew = mix*Fb + (1.0-mix)*Fold

#         return Fnew
    
#     def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

#         norb = matimp.shape[0]
#         ns = matimp.shape[2]


#         nspace = 0
#         for val in self.crystal.probspace.values():
#             nspace += len(val)

#         matloc = np.zeros((norb,norb,ns,nspace),dtype=np.complex128,order='F')

#         for key, val in self.crystal.probspace.items():
#             iprob = int(key)-1
#             for ispace in val:
#                 matloc[...,ispace] = matimp[...,iprob]

#         return matloc
        
#     def Loc2Imp(self,matimp : np.ndarray)-> np.ndarray:

#         norb = matimp.shape[0]
#         ns = matimp.shape[2]
    

#         nspace = 0
#         for val in self.crystal.probspace.values():
#             nspace += len(val)

#         matloc = np.zeros((norb,norb,ns,nspace),dtype=np.complex128,order='F')

#         for key, val in self.crystal.probspace.items():
#             iprob = int(key)-1
#             for ispace in val:
#                 matloc[...,ispace] = matimp[...,iprob]

#         return matloc
    
#     def Arr2Dict(self, equiv : np.ndarray, matin : np.ndarray) -> dict:

#         ns = matin.shape[2]
#         nind = np.amax(equiv)
#         matdict = {}

        
#         for ind in range(nind):
#             matdict[ind+1] = []
#             pos = self.crystal.FindPositions(equiv,ind+1)
#             for js in range(ns):
#                 e = 0
#                 for ii, jj in pos:
#                     e += matin[ii,jj,js]
#                 e /= len(pos)
#                 matdict[ind+1].append(e)
        
#         return matdict
    
#     def Dict2Arr(self, equiv : np.ndarray, matdict : dict) -> np.ndarray:

#         norb = len(equiv)
#         ns = self.crystal.ns
#         matout = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')
#         nind = np.amax(equiv)

#         for js in range(ns):
#             for ind in range(nind):
#                 pos = self.crystal.FindPositions(equiv,ind+1)
#                 for ii,jj in pos:
#                     matout[ii,jj,js] = matdict[str(ind+1)]

#         return matout
    
#     def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

       
#         norb = len(self.crystal.find)
#         ns = self.crystal.ns
        
#         matout = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

#         matout = QAFort.dyson.flocstc(mat1,mat2)

#         return matout 

#     def Embedding(self, matin : np.ndarray):

#         norb = len(self.crystal.find)
#         ns = self.crystal.ns
#         nrk = len(self.crystal.kpoint)
#         nspace = self.crystal.fprojector.shape[3]
        
#         matout = np.zeros((norb,norb,ns,nrk),dtype=np.complex128,order='F')
        
#         for ispace in range(nspace):
#             matout += QAFort.embedding.flocstc(nrk,matin[...,ispace],self.crystal.fprojector[...,ispace])

#         return matout
    
#     def Save(self,matin : np.ndarray, fn : str):

#         norb = matin.shape[0]
#         ns = matin.shape[2]

#         if os.path.exists('flocstc'):
#             pass
#         else:
#             os.mkdir("flocstc")
#         os.chdir("flocstc")
#         with open(fn+'.txt','w') as f:
#             f.write("iorb, jorb, is, Re(F), Im(F)\n")
#             for js in range(ns):
#                 for jorb in range(norb):
#                     for iorb in range(norb):
#                         f.write(f"{iorb} {jorb} {js} {matin[iorb,jorb,js].real} {matin[iorb,jorb,js].imag}\n")
#         os.chdir("..")
#         return None
    
    
# class ImpurityLevel(FLocStc):

#     def __init__(self, crystal: Crystal, niham : NIHamiltonian, mu : float):
#         super().__init__(crystal)
        
#         self.niham = niham
#         self.mu = mu
#         self.loc = None
#         self.imp = None
#         self.Cal()

#     def Cal(self):
        
#         norbc = self.crystal.fprojector.shape[1]
#         ns = self.crystal.ns
#         nspace = self.crystal.fprojector.shape[3]

#         ham = self.niham.UpdateMu(self.niham.k,self.mu)

#         eimp = np.zeros((norbc,norbc,ns,nspace),dtype=np.complex128,order='F')

#         for ispace in range(nspace):
#             eimp[...,ispace] = QAFort.projection.flatstc(ham,self.crystal.fprojector[...,ispace])

#         self.loc = eimp
#         self.imp = self.Loc2Imp(eimp)

#         return None
        

# class SigmaHLoc(FLocStc):

#     def __init__(self, crystal: Crystal, gloc : GreenLoc, vbare : object):
#         super().__init__(crystal)
        
#         self.gloc = gloc
#         self.vbare = vbare
#         self.hloc = None
#         self.himp = None
#         self.hdyn = None
#         self.Cal()
#         self.MakeDyn()

#     def Cal(self):
        
#         norbc = self.crystal.fprojector.shape[1]
#         ns = self.crystal.ns
#         norb = self.crystal.bprojector.shape[1]
#         nspace = self.crystal.bprojector.shape[3]

#         U = np.zeros((norb,norb,ns,ns,nspace),dtype=np.complex128,order='F')
#         hloc = np.zeros((norbc,norbc,ns,nspace),dtype=np.complex128,order='F')
#         tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex128,order='F')

#         for ispace in range(nspace):
#             U[...,ispace] = QAFort.projection.blatstc(self.vbare.k,self.crystal.bprojector[...,ispace])
        
#             if ns == 2:
#                 tempmat = self.crystal.OrbSpin2Composite(U[...,ispace])
#                 for ind1 in range(norb*ns):
#                     nn1 = [0]*2
#                     ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
#                     iorbc1, iorbc2 = self.crystal.b2f[iorb]
#                     for ind2 in range(norb*ns):
#                         nn2 = [0]*2
#                         ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],ind2,nn2)
#                         iorbc3,iorbc4 = self.crystal.b2f[jorb]
#                         hloc[iorbc1,iorbc2,js,ispace] += -tempmat[ind1,ind2]*self.gloc.gf[iorbc4,iorbc3,ks,-1,ispace]
#             else:
#                 if self.crystal.soc == False:
#                     C = 2
#                     for iorb in range(norb):
#                         iorbc1, iorbc2 = self.crystal.b2f[iorb]
#                         for jorb in range(norb):
#                             iorbc3,iorbc4 = self.crystal.b2f[jorb]
#                             hloc[iorbc1,iorbc2,0,ispace] += -U[iorb,jorb,0,0,ispace]*self.gloc.gf[iorbc4,iorbc3,0,-1,ispace]
#                 else:
#                     C = 1
#                     for iorb in range(norb):
#                         iorbc1, iorbc2 = self.crystal.b2f[iorb]
#                         for jorb in range(norb):
#                             iorbc3,iorbc4 = self.crystal.b2f[jorb]
#                             hloc[iorbc1,iorbc2,0,ispace] += -U[iorb,jorb,0,0,ispace]*self.gloc.gf[iorbc4,iorbc3,0,-1,ispace]
            
#         self.hloc = hloc
#         self.himp = self.Loc2Imp(hloc)

#         return None
    
#     def MakeDyn(self):

#         norb = self.crystal.fprojector.shape[1]
#         ns = self.crystal.ns
#         nft = self.gloc.gf.shape[3]
#         nspace = self.crystal.fprojector.shape[3]

#         hdyn = np.zeros((norb,norb,ns,nft,nspace),dtype=np.complex128,order='F')
        
#         for ift in range(nft):
#             hdyn[...,ift,:] = self.hloc
        
#         self.hdyn = hdyn

#         return None

# class SigmaHImp(FLocStc):

#     def __init__(self, crystal: Crystal):
#         super().__init__(crystal)
#         self.Cal()

#     def Cal(self):
#         pass

# class SigmaFLoc(FLocStc):

#     def __init__(self, crystal: Crystal, gloc : GreenLoc, vbare : object):
#         super().__init__(crystal)

#         self.gloc = gloc
#         self.vbare = vbare
#         self.floc = None
#         self.fimp = None
#         self.fdyn = None
    
#         self.Cal()
#         self.MakeDyn()

#     def Cal(self):
        
#         norbc = self.crystal.fprojector.shape[1]
#         ns = self.crystal.ns
#         norb = self.crystal.bprojector.shape[1]
#         nspace = self.crystal.fprojector.shape[3]

#         U = np.zeros((norb,norb,ns,ns,nspace),dtype=np.complex128,order='F')
#         floc = np.zeros((norbc,norbc,ns,nspace),dtype=np.complex128,order='F')
        

#         for ispace in range(nspace):
#             U[...,ispace] = QAFort.projection.blatstc(self.vbare.k,self.crystal.bprojector[...,ispace])

#             for js in range(ns):
#                 for iorb in range(norb):
#                     iorbc1, iorbc4 = self.crystal.b2f[iorb]
#                     for jorb in range(norb):
#                         iorbc3, iorbc2 = self.crystal.b2f[jorb]
#                         floc[iorbc1,iorbc2,js,ispace] += self.gloc.gf[iorbc4,iorbc3,js,-1,ispace]*U[iorb,jorb,js,js,ispace]

#         self.floc = floc
#         self.fimp = self.Loc2Imp(floc)
        
#         return None



# class SigmaFImp(FLocStc):

#     def __init__(self, crystal: Crystal):
#         super().__init__(crystal)
#         self.Cal()

#     def Cal(self):
#         pass
