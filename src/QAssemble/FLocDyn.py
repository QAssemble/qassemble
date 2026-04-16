import numpy as np
import logging
import sys
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

class FLocDyn(object):

    def __init__(self,crystal : Crystal, ft : DLR, projector : Projector):
        
        self.crystal = crystal
        self.dlr = ft
        self.projector = projector

    def Inverse(self, mat : np.ndarray):
        
        norb = mat.shape[0]
        ns = mat.shape[2]
        nft = mat.shape[3]

        matinv = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            for js in range(ns):
                matinv[:,:,js,ift] = Common.MatInv(mat[:,:,js,ift])

        return matinv
    
    
    def F2T(self, ff : np.ndarray) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nfreq = ff.shape[3]

        ff_t = np.moveaxis(ff, -1, 0)

        batch = norb * norb * ns
        ff_2d = np.ascontiguousarray(ff_t).reshape(nfreq, batch)

        fxx = self.dlr.dF.dlr_from_matsubara(ff_2d, beta=self.dlr.beta, xi = -1)
        ftau_2d = self.dlr.dF.tau_from_dlr(fxx)
        ntau = ftau_2d.shape[0]
        ftau = ftau_2d.reshape(ntau, norb, norb, ns)
        ftau = np.moveaxis(ftau, 0, -1)

        ftau = np.asfortranarray(ftau)

        return ftau
        
    def T2F(self,ftau : np.ndarray) -> np.ndarray:

        norb = ftau.shape[0]
        ns = ftau.shape[2]
        ntau = ftau.shape[3]

        ftau_t = np.moveaxis(ftau, -1, 0)  # (ntau, norb, norb, ns)
        batch = norb * norb * ns 
        ftau_2d = np.ascontiguousarray(ftau_t).reshape(ntau, batch)

        fxx = self.dlr.dF.dlr_from_tau(ftau_2d)
        ff_2d = self.dlr.dF.matsubara_from_dlr(fxx, beta=self.dlr.beta, xi=-1)
        nfreq = ff_2d.shape[0]

        ff = ff_2d.reshape(nfreq, norb, norb, ns)
        ff = np.moveaxis(ff, 0, -1)  # (norb, norb, ns, nfreq)
        ff = np.asfortranarray(ff)

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
    
    # def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

    #     norb = matimp.shape[0]
    #     ns = matimp.shape[2]
    #     nft = matimp.shape[3]

    #     probindex = self.crystal.probindex if self.crystal.probindex else self.crystal.probspace

    #     nspace = 0
    #     for val in probindex.values():
    #         nspace += len(val)

    #     matloc = np.zeros((norb,norb,ns,nft,nspace),dtype=np.complex128,order='F')

    #     for key, val in probindex.items():
    #         iprob = int(key)-1
    #         for ispace in val:
    #             matloc[...,ispace] = matimp[...,iprob]

    #     return matloc
    
    # def Loc2Imp(self,matloc : np.ndarray)->np.ndarray:

    #     probindex = self.crystal.probindex if self.crystal.probindex else self.crystal.probspace

    #     nprob = len(probindex)
    #     norb = matloc.shape[0]
    #     ns = matloc.shape[2]
    #     nft = matloc.shape[3]

    #     matimp = np.zeros((norb,norb,ns,nft,nprob),dtype=np.complex128,order='F')

    #     for key, val in probindex.items():
    #         iprob = int(key)-1
    #         tempmat = np.zeros((norb,norb,ns, nft),dtype=np.complex128)
    #         for ispace in val:
    #             tempmat += matloc[...,ispace]
    #         tempmat /=len(val)
    #         matimp[...,iprob] = tempmat

    #     return matimp
    
    # def Arr2Dict(self, equiv : np.ndarray, matin : np.ndarray) -> dict:
        
    #     ns = matin.shape[2]
    #     nind = np.amax(equiv)
    #     matdict = {}

    #     for ind in range(nind):
    #         matdict[ind+1] = []
    #         pos = self.crystal.FindPositions(equiv,ind+1)
    #         for js in range(ns):
    #             e = 0
    #             for ii, jj in pos:
    #                 e += matin[ii,jj,js]
    #             e /=len(pos)
    #             matdict[ind+1].append(e.tolist())
        
    #     return matdict
    
    # def Dict2Arr(self,equiv : np.ndarray, matdict : np.ndarray) -> np.ndarray:

    #     norb = len(equiv)
    #     ns = self.crystal.ns
    #     nfreq = len(matdict["1"])                

    #     matout = np.zeros((norb,norb,ns,nfreq),dtype=np.complex128,order='F')
    #     nind = np.amax(equiv)

    #     for js in range(ns):
    #         for ind in range(nind):
    #             pos = self.crystal.FindPositions(equiv,ind+1)
    #             for ii, jj in pos:
    #                 matout[ii,jj,js] = matdict[str(ind+1)]

    #     return matout
    
    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        # norb = len(self.crystal.find)
        # ns = self.crystal.ns
        # nft = self.ft.size

        # matout = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        # matout = QAFort.dyson.flocdyn(mat1,mat2)

        return Dyson.FLocDyn(mat1, mat2)

    
    def Projection(self, matin : np.ndarray):

        projector = self.projector.fprojector
        
        
        matdict = {}
        for key in projector.keys():
            proj = projector[key]
            norbc = proj.shape[1]
            ns = proj.shape[2]
            nfreq = matin.shape[4]

            tempmat = np.zeros((norbc, norbc, ns, nfreq), dtype=np.complex128, order='F')

            tempmat = PJ.FLatDyn(matin, proj)

            matdict[key] = tempmat

        return matdict
    
# class GreenLoc(FLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, green : GreenInt):
        
#         super().__init__(crystal, ft)
#         self.green = green
#         self.gf = None
#         self.gt = None
        
#         self.Cal()

#     def Cal(self): # projection
        
#         norbc = self.crystal.fprojector.shape[1]
#         ns = self.crystal.ns
#         nft = self.ft.size
#         nspace = self.crystal.fprojector.shape[3]

#         gf = np.zeros((norbc,norbc,ns,nft,nspace),dtype=np.complex128)

#         for ispace in range(nspace):
#             gf[...,ispace] = QAFort.projection.flatdyn(self.green.gkf,self.crystal.fprojector[...,ispace])

#         self.gf = gf
#         self.gt = self.F2T(gf,1,1)

#         return None

# class GreenImp(FLocDyn): # read CTQMC output

#     def __init__(self, crystal: Crystal, ft: FTGrid):
#         super().__init__(crystal, ft)
#         self.Cal()

#     def Cal(self):
#         super().Dict2Arr()
#         pass

# class SigmaLoc(FLocDyn):
    
#     def __init__(self, crystal: Crystal, ft: FTGrid, sigma : object):
#         super().__init__(crystal, ft)
        
#         self.sigma = sigma
#         self.f = None
#         self.Cal()

#     def Cal(self): # projection
        
#         norbc = self.crystal.fprojector.shape[1]
#         ns = self.crystal.ns
#         nft = self.ft.size
#         nspace = self.crystal.fprojector.shape[3]

#         sigmalocf = np.zeros((norbc,norbc,ns,nft,nspace),dtype=np.complex128,order='F')

#         for isapce in range(nspace):
#             sigmalocf[...,isapce] = QAFort.projection.flatdyn(self.sigma,self.crystal.fprojector[...,isapce])

#         self.f = sigmalocf
#         self.t = self.F2T(sigmalocf,0,1)

#         return None


# class SigmaImp(FLocDyn): # read CTQMC output

#     def __init__(self, crystal: Crystal, ft: FTGrid):
#         super().__init__(crystal, ft)
#         self.Cal()

#     def Cal(self):
#         super().Dict2Arr()
#         pass

# class SigmaLGWC(FLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid):
#         super().__init__(crystal, ft)

#         pass
    

# class Hybridisation(FLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, implev : object, gimp : GreenImp, sigmaimp : SigmaImp):
#         super().__init__(crystal, ft)
#         self.Cal()
    
#     def Cal(self):
#         pass
