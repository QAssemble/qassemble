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
    
    def CheckGroup(self, filepath :str, group : str):
        
        with h5py.File(filepath,'r') as file:
            return group in file
    
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
        if self.projector is None:
            raise ValueError("projector is required for Projection")

        if matin.ndim != 5:
            raise ValueError(f"matin must be 5D, got {matin.ndim}D")

        norb = matin.shape[0]
        ns = matin.shape[2]
        nfreq = matin.shape[4]

        matdict = {}
        for key, proj in self.projector.fprojector.items():
            norbc = proj.shape[1]
            tempmat = np.zeros((norbc, norbc, ns, nfreq), dtype=np.complex128, order='F')

            tempmat = PJ.FLatDyn(matin, proj)

            
            matdict[key] = tempmat

        return matdict
    
class GreenLoc(FLocDyn):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, green : np.ndarray, hdf5file : str = None, group : str = None):

        super().__init__(crystal, dlr, projector)

        
        self.green = green
        self.f = {}
        self.t = {}

        self.occ = None
        self.Cal()
        
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

    def Cal(self):

        self.f = self.Projection(self.green)

        self.t = {}

        for key, mat in self.f.items():
            self.t[key] = self.F2T(mat)

        self.Occ()

        return None
    
    def Occ(self):
        
        self.occ = {}

        tau_beta = np.array([self.dlr.beta], dtype=np.float64)
        for key, mat in self.t.items():
            
            occ = np.zeros_like(mat[...,0], dtype=np.complex128)
            for js in range(mat.shape[2]):
                block = mat[:, :, js, :].T

                ntau_b = block.shape[0]
                block_2d = block.reshape(ntau_b, -1)

                fxx = self.dlr.dF.dlr_from_tau(block_2d)
                fout = self.dlr.dF.eval_dlr_tau(fxx[:, :, None], tau_beta, beta=self.dlr.beta)

                occ[:, :, js] = -fout[0, :, 0].reshape(mat.shape[0], mat.shape[0])

            self.occ[key] = occ

        return None
    
    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            if self.CheckGroup(self.hdf5file,self.group):
                group = file[self.group]
                if self.subgroup in group:
                    gloc = group[self.subgroup]
                else:
                    gloc = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                gloc = group.create_group(self.subgroup)
            

            if obj != None:
                gloc.create_dataset(fn,dtype=complex,data=obj)
            else:
                gloc.create_dataset(fn,dtype=complex,data=self.f)

        return None


class Hybridization(FLocDyn):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, green : dict, eimp : dict, sigh : dict = None, sigf : dict = None, sigc : dict = None, hdf5file : str = None, group : str = None):
        
        super().__init__(crystal, dlr, projector)

        self.green = green
        self.eimp = eimp
        self.sigh = sigh
        self.sigf = sigf
        self.sigc = sigc
        
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        self.f = {}
        self.t = {}

        
    def Cal(self):

        projector = self.projector.fprojector


        for key in projector.keys():
            
            tempmat = np.zeros_like(self.green[key], dtype=np.complex128, order='F')
            sig = np.zeros_like(self.green[key], dtype=np.complex128, order='F')
            if self.sigh is not None:
                sig += self.sigh[key]
            if self.sigf is not None:
                sig += self.sigf[key]
            if self.sigc is not None:
                sig += self.sigc[key]

            g_inv = self.Inverse(self.green[key])
            
            e = self.eimp[key]
            I = np.eye(g_inv.shape[0], dtype=np.complex128)
            omega = self.dlr.omega * 1j

            for iomega in range(len(omega)):
                for js in range(g_inv.shape[2]):
                    tempmat[..., js, iomega] = omega[iomega]*I - e[..., js] - g_inv[..., js, iomega] - sig[..., js, iomega]
            self.f[key] = tempmat
            self.t[key] = self.F2T(tempmat)

        return None
    
    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            if self.CheckGroup(self.hdf5file,self.group):
                group = file[self.group]
                if self.subgroup in group:
                    hyb = group[self.subgroup]
                else:
                    hyb = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                hyb = group.create_group(self.subgroup)
            

            if obj != None:
                hyb.create_dataset(fn,dtype=complex,data=obj)
            else:
                hyb.create_dataset(fn,dtype=complex,data=self.f)

        return None

# class FWeiss(FLocDyn):

#     def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, eim : dict,)