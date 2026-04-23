import numpy as np
import logging
import os, sys
import scipy.optimize
import scipy.linalg.lapack
import copy
import h5py
import time, datetime
from .Crystal import Crystal
from .FLatStc import FLatStc
from .Projector import Projector
from .utility.DLR import DLR
from .utility.Common import Common
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
from .utility.Mixing import Mixing
from .utility.Projection import Projection as PJ

logger = logging.getLogger("QAssemble")

class FLocStc(object):

    def __init__(self,crystal : Crystal, projector : Projector):

        self.crystal = crystal
        self.projector = projector

    def Inverse(self,mat : np.ndarray):

        norb = mat.shape[0]
        ns = mat.shape[2]

        matinv = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        for js in range(ns):
            matinv[:,:,js] = Common.MatInv(mat[:, :, js])

        return matinv
    
    def Mixing(self,iter : int, mix : float, Fb : np.ndarray, Fold : np.ndarray) -> np.ndarray:

        norb = Fb.shape[0]
        ns = Fb.shape[2]

        Fnew = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        if iter == 1:
            mix = 1.0
            Fold = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        Fnew = mix*Fb + (1.0-mix)*Fold

        return Fnew
    
    
    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        return Dyson.FLocStc(mat1, mat2)

    
    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]

        if os.path.exists('flocstc'):
            pass
        else:
            os.mkdir("flocstc")
        os.chdir("flocstc")
        with open(fn+'.txt','w') as f:
            f.write("iorb, jorb, is, Re(F), Im(F)\n")
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        f.write(f"{iorb} {jorb} {js} {matin[iorb,jorb,js].real} {matin[iorb,jorb,js].imag}\n")
        os.chdir("..")
        return None
    
    def Projection(self, matin : np.ndarray):
        if self.projector is None:
            raise ValueError("projector is required for Projection")

        if matin.ndim != 4:
            raise ValueError(f"matin must be 4D, got {matin.ndim}D")

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]

        matdict = {}
        for key, proj in self.projector.fprojector.items():
            norbc = proj.shape[1]
            tempmat = np.zeros((norbc, norbc, ns, nrk), dtype=np.complex128, order='F')

            tempmat = PJ.FLatStc(matin, proj)

            
            matdict[key] = tempmat

        return matdict
    
    
class ImpurityLevel(FLocStc):

    def __init__(self, crystal : Crystal, projector : Projector, hamtb : np.ndarray, mu : float, sigh : np.ndarray = None, sigf : np.ndarray = None, hloc : dict = None, floc : dict = None):

        super().__init__(crystal, projector)

        self.hamtb = hamtb
        self.mu = mu
        self.ham = None
        self.sig = None

        tempmat = np.zeros_like(hamtb, dtype=np.complex128, order='F')

        for ik in range(hamtb.shape[3]):
            for js in range(hamtb.shape[2]):
                tempmat[...,js,ik] = hamtb[...,js,ik] - mu*np.eye(hamtb.shape[0], dtype=np.complex128)
        
        if sigh is not None:
            tempmat += sigh
        
        if sigf is not None:
            tempmat += sigf

        self.ham = tempmat

        if (hloc is not None) and (floc is not None):
            print("Double counting term entered.")
            tempmat2 = {}
            for key in hloc.keys():
                tempmat2[key] = hloc[key] + floc[key]

            self.sig = tempmat2 

        self.e = {}
        self.Cal()

    def Cal(self):
        
        

        e = self.Projection(self.ham)

        if (self.sig is not None):
            for key, mat in e.items():

                mat -= self.sig[key]

        
        self.e = e

        return None
            
            
            
class SigHLoc(FLocStc):

    def __init__(self, crystal : Crystal, projector : Projector, occ : dict = None, vloc : dict = None, hdf5file : str = 'glob.h5', group : str = None):

        super().__init__(crystal, projector)

        self.occ = occ
        self.vloc = vloc
        self.hloc = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

    
    def Cal(self):

        projector = self.projector.fprojector
        h = {}

        for key, proj in projector.items():
            norbc = proj.shape[1]
            ns = proj.shape[2]
            v = self.vloc[key]
            norb = v.shape[0]

            h[key] = np.zeros((norbc, norbc, ns), dtype=np.complex128, order='F')

            if ns != 1:

                for ind1 in range(norb * ns):
                    nn1 = [0] * 2
                    ind1, [iorb, js] = Common.Indexing(norb*ns, 2, [norb, ns], 0, ind1, nn1)

                    iorbc1, iorbc2 = self.projector.ProbBorb2FPair(key, iorb)

                    for ind2 in range(norb * ns):
                        nn2 = [0] * 2
                        ind2, [jorb, ks] = Common.Indexing(norb*ns, 2, [norb, ns], 0, ind2, nn2)

                        iorbc3, iorbc4 = self.projector.ProbBorb2FPair(key, jorb)

                        h[key][iorbc1, iorbc2, js] += (v[iorb, jorb, js, ks] * self.occ[key][iorbc4, iorbc3, ks])
            else:
                if (self.crystal.soc == True):
                    C = 1
                else:
                    C = 2
                
                for ind1 in range(norb * ns):
                    nn1 = [0] * 2
                    ind1, [iorb, js] = Common.Indexing(norb*ns, 2, [norb, ns], 0, ind1, nn1)

                    iorbc1, iorbc2 = self.projector.ProbBorb2FPair(key, iorb)

                    for ind2 in range(norb * ns):
                        nn2 = [0] * 2
                        ind2, [jorb, ks] = Common.Indexing(norb*ns, 2, [norb, ns], 0, ind2, nn2)

                        iorbc3, iorbc4 = self.projector.ProbBorb2FPair(key, jorb)

                        h[key][iorbc1, iorbc2, js] += (v[iorb, jorb, js, ks] * self.occ[key][iorbc4, iorbc3, ks]) * C

        self.hloc = h





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
