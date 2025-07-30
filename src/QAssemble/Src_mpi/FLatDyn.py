import string as string
from typing import Any
import matplotlib as mat
import re as re
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import matplotlib.font_manager as fm
from collections import OrderedDict
# import json, os, shutil, sys
import os, sys
# import itertools
import scipy.optimize
from sympy.physics.wigner import gaunt, wigner_3j
from scipy.fftpack import fftn, ifftn
# import scipy.linalg
# from pymatgen.core import Lattice, Structure
# from pymatgen.transformations.standard_transformations import SupercellTransformation
# import subprocess
import copy
import h5py
from .Crystal import Crystal
from .FTGrid import FTGrid
from .FLatStc import FLatStc
from .MPIManager import MPIManager
qapath = os.environ.get('QAssemble','')
sys.path.append(qapath+'/src/QAssemble/modules')
import QAFort

class FLatDyn(object):
    def __init__(self,crystal : Crystal, ft : FTGrid) -> object:
        self.crystal = crystal
        self.ft = ft
        self.mappingidx = None
        
    def Inverse(self, mat : np.ndarray) -> np.ndarray:

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]
        nft = mat.shape[4]

        matinv = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    matinv[:,:,js,irk,ift] = np.linalg.inv(mat[:,:,js,irk,ift])
        # for js, irk, ift in itertools.product(list(range(ns)),list(range(nrk),list(range(nft)))):
        #     matinv[:,:,js,irk,ift] = np.linalg.inv(mat[:,:,js,irk,ift])
        
        return matinv
    
    def T2F(self,ftau : np.ndarray) -> np.ndarray:
        

        norb = ftau.shape[0]
        ns = ftau.shape[2]
        nk = ftau.shape[3]
        nfreq = len(self.ft.omega)
        ff = np.zeros((norb,norb,ns,nk,nfreq),dtype=np.complex128,order='F')

        ff = QAFort.fourier.flatdyn_t2f(self.ft.tau,self.ft.beta, ftau,self.ft.omega)

        return ff
    
    def F2T(self,ff : np.ndarray,isgreen : int, highzero : int) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]
        ntau = len(self.ft.tau)

        ftau = np.zeros((norb,norb,ns,nk,ntau),dtype=np.complex128,order='F')
        tempmat = copy.deepcopy(ff)
        moment, high = self.Moment(tempmat,isgreen,highzero)
        
        ftau = QAFort.fourier.flatdyn_f2t(self.ft.omega,tempmat,moment,self.ft.tau)

        return ftau

    
    def Moment(self,ff : np.ndarray, isgreen : int, highzero : int) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]

        moment = np.zeros((norb,norb,ns,nk,3),dtype=np.complex128,order='F')
        high = np.zeros((norb,norb,ns,nk),dtype=np.complex128,order='F')

        tempmat = copy.deepcopy(ff)

        moment, high = QAFort.fourier.flatdyn_m(self.ft.omega,tempmat,isgreen,highzero)

        return moment, high
    
    
    def K2R(self,matk : np.ndarray, rkgrid : list = None) -> np.ndarray:

        rkvec = self.crystal.kpoint
        if rkgrid == None:
            rkgrid = self.crystal.rkgrid
        
        
        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[3]
        nft = matk.shape[4]
        matr = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')
        tempmat = copy.deepcopy(matk)
        

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):

                            [a,m1] = self.crystal.FAtomOrb(iorb)
                            [b,m2] = self.crystal.FAtomOrb(jorb)

                            delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
                            phase = np.exp(2.0j*np.pi*np.dot(rkvec[irk],delta))

                            tempmat[iorb,jorb,js,irk,ift] *= phase

        matr = QAFort.fourier.flatdyn_k2r(rkgrid,tempmat)
        
        return matr
    
    def R2K(self, matr : np.ndarray) -> np.ndarray:

        rkvec = self.crystal.kpoint
        rkgrid = self.crystal.rkgrid

        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[3]
        nft = matr.shape[4]

        matk = QAFort.fourier.flatdyn_r2k(rkgrid,matr)

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):

                            [a,m1] = self.crystal.FAtomOrb(iorb)
                            [b,m2] = self.crystal.FAtomOrb(jorb)

                            delta = self.crystal.basisf[a,:]- self.crystal.basisf[b,:]

                            phase = np.exp(-2.0j*np.pi*np.dot(rkvec[irk],delta))
                            matk[iorb,jorb,js,irk,ift] *= phase
        return matk
    
    def R2mR(self) -> list: # move to crystal

        rkvec = self.crystal.kpoint

        mrkvec = np.array(1.0-rkvec,dtype=float)

        for ii in range(mrkvec.shape[0]):
            for jj in range(mrkvec.shape[1]):
                if mrkvec[ii,jj] == 1.0:
                    mrkvec[ii,jj] = 0.0
        
        mappingidx = []

        for ii in range(rkvec.shape[0]):
            for jj in range(mrkvec.shape[1]):
                if (abs(rkvec[ii,0]-mrkvec[jj,0])<=1.0e-6)and(abs(rkvec[ii,1]-mrkvec[jj,1])<=1.0e-6)and(abs(rkvec[ii,2]-mrkvec[jj,2])<=1.0e-6):
                    mappingidx.append([ii,jj])

        self.mappingidx = mappingidx
        return None
    
    def RT2mRmT(self,G : np.ndarray) -> np.ndarray: # move to crystal

        self.R2mR()

        norb = G.shape[0]
        ns = G.shape[2]
        nr = G.shape[3]
        ntau = G.shape[4]

        GmRmT = np.zeros((norb,norb,ns,nr,ntau),dtype=np.complex128,order='F')

        for itau in range(ntau):
            for rp in self.mappingidx:
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            GmRmT[iorb,jorb,js,rp[0],itau] = -G[iorb,jorb,js,rp[1],ntau-itau-1]

        return GmRmT
    
    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nrk = y.shape[3]
        nft = y.shape[4]

        ynew = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        w0 = (1.0 - 3.0*w1)*np.pi*temperature
        widtharray = w0+w1*x
        cnt = 0
        for irk in range(nrk):
            for x0 in x:
                if (x0>cutoff+(w0+w1*cutoff)*3.0):
                    ynew[...,irk,cnt] = y[...,irk,cnt]
                else : 
                    if ((x0>3*widtharray[cnt])and((x[-1]-x0)>3*widtharray[cnt])):
                        dist = 1.0/np.sqrt(2*np.pi)/widtharray[cnt]*np.exp(-(x-x0)**2/2.0/widtharray[cnt]**2)
                        for js in range(ns):
                            for iorb in range(norb):
                                for jorb in range(norb):
                                    ynew[iorb,jorb,js,irk,cnt] = sum(dist*y[iorb,jorb,js,irk])/sum(dist)
                    else:
                        ynew[...,irk,cnt] = y[...,irk,cnt]
                cnt += 1

        return ynew
    
    def Mixing(self, iter : int, mix : float, Fb : np.ndarray, Fm : np.ndarray) -> np.ndarray:

        norb = Fb.shape[0]
        ns = Fb.shape[2]
        nrk = Fb.shape[3]
        nft = Fb.shape[4]

        Fnew = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        if iter == 1:
            mix = 1.0
            Fm = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')
        
        Fnew = mix*Fb + (1.0-mix)*Fm

        return Fnew
    
    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = mat1.shape[0]
        ns = mat1.shape[2]
        nrk = mat1.shape[3]
        nft = mat1.shape[4]
        
        matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        matout = QAFort.dyson.flatdyn(mat1,mat2)

        return matout

    def Projection(self, matin : np.ndarray):

        
        ns = matin.shape[2]
        nft = matin.shape[4]
        norbc = self.crystal.fprojector.shape[1]
        nspace = self.crystal.fprojector.shape[3]

        matout = np.zeros((norbc,norbc,ns,nft,nspace),dtype=np.complex128,order='F')

        for ispace in range(nspace):
            matout[...,ispace] = QAFort.projection.flatdyn(matin,self.crystal.fprojector[...,ispace])

        return matout
    
    def ChemEmbedding(self,mu : float) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.ft.omega)#self.ft.size

        chem = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            if iorb == jorb:
                                chem[iorb,jorb,js,irk,ift] = mu
                            else:
                                chem[iorb,jorb,js,irk,ift] = 0

        return chem
    
    def StcEmbedding(self, matin : np.ndarray) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.ft.omega)#self.ft.size

        matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            matout[...,ift] = matin

        return matout
    # def Save(self, matin : np.ndarray, fn : str):


    #     os.chdir('work')
    #     # with open(fn+'.txt','w') as f:
    #     #     f.write("#iorb, jorb, is, ik, ift, Re(F(k,w)), Im(F(k,w))\n")
    #     #     for ift in range(nft):
    #     #         for irk in range(nrk):
    #     #             for js in range(ns):
    #     #                 for jorb in range(norb):
    #     #                     for iorb in range(norb):
    #     #                         f.write(f"{iorb} {jorb} {js} {irk} {ift} {matin[iorb,jorb,js,irk,ift].real} {matin[iorb,jorb,js,irk,ift].imag}\n")
    #     with h5py.File('flatdyn.h5','w') as file:
    #         file.create_dataset('name',data='FLatDyn')
    #     os.chdir("..")
    #     return None
    
    def CheckGroup(self, filepath :str, group : str):
        
        with h5py.File(filepath,'r') as file:
            return group in file
        
    
    def Spectral(self, green : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        nfreq = len(self.ft.omega)

        akf = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,oder='F')

        akf = -1/np.pi*green.imag

        return akf
    
    def R2KArb(self,matr : np.ndarray = None,kpoint : np.ndarray = None): # R2KAny

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nr = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        nk = len(kpoint)
        nft = matr.shape[4]

        self.crystal.Rvec()
        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb,norb,ns,nk,nft),dtype=complex,order='F')

        for ift in range(nft):
            for ik in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            temp = 0
                            for ir in range(nr):
                                temp += tempmat[iorb,jorb,js,ir,ift]*np.exp(-2.0j*np.pi*(kpoint[ik]@self.crystal.rvec[ir]))
                            [a,m1] = self.crystal.FAtomOrb(iorb)
                            [b,m2] = self.crystal.FAtomOrb(jorb)
                            delta = self.crystal.basisf[a,:]-self.crystal.basisf[b,:]
                            phase = np.exp(-2.0j*np.pi*(kpoint[ik]@delta))
                            matk[iorb,jorb,js,ik,ift] = temp*phase
        
        return matk

    def KArb(self, matr : np.ndarray = None, kpoint : np.ndarray = None):

        norb = matr.shape[0]
        ns = matr.shape[2]
        nr = matr.shape[3]
        nfreq = matr.shape[4]
        nk = len(kpoint)

        tempmat = np.zeros((norb,norb,ns,nr,nfreq),dtype=complex,order='F')
        matkinv = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,order='F')

        matrinv = self.Inverse(matr)
        omega = self.ft.omega

        for ifreq in range(nfreq):
            for ir in range(nr):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            if iorb==jorb:
                                tempmat[iorb,jorb,js,ir,ifreq] = 1j*omega[ifreq]-matrinv[iorb,jorb,js,ir,ifreq]
                            else:
                                tempmat[iorb,jorb,js,ir,ifreq] = -matrinv[iorb,jorb,js,ir,ifreq]

        tempmat2 = self.R2KArb(tempmat,kpoint)

        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            if iorb==jorb:
                                matkinv[iorb,jorb,js,ik,ifreq] = 1j*omega[ifreq]-tempmat2[iorb,jorb,js,ik,ifreq]
                            else:
                                matkinv[iorb,jorb,js,ik,ifreq] = -tempmat2[iorb,jorb,js,ik,ifreq]
        
        matk = self.Inverse(matkinv)

        return matk
        

    
class GreenBare(FLatDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, hamtb : np.ndarray = None, hdf5file : str = None, group : str = None) -> object:
        
        super().__init__(crystal, ft)
        # print(self.niham.hamtb[...,0,0])
        self.hamtb = hamtb
        self.kt = None
        self.kf = None
        self.rt = None
        self.rf = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.Cal()
        if hdf5file != None:
            self.Save()
        

    def Cal(self): # freq, tau combine
        
        
        print(self.hamtb[:,:,0,0])
        gnotkf = QAFort.bare.flatfreq(self.hamtb,self.ft.omega)
        gnotrf = self.K2R(gnotkf)#######
        
        self.kf = gnotkf
        self.rf = gnotrf

        gnotkt = QAFort.bare.flattau(self.hamtb,self.ft.tau)
        gnotrt = self.K2R(gnotkt)

        self.kt = gnotkt
        self.rt = gnotrt

        return None
    
    def Save(self):

        # if os.path.exists('gbare'):
        #     pass
        # else:
        #     os.mkdir('gbare')

        with h5py.File(self.hdf5file,'a') as file:
            if self.CheckGroup(self.hdf5file,self.group):
                group = file[self.group]
                if self.subgroup in group:
                    gbare = group[self.subgroup]
                else:
                    gbare = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                gbare = group.create_group(self.subgroup)
            gbare.create_dataset('g0kf',dtype=complex,data=self.kf)

        return None
    
    # def Load(self):

    #     os.chdir('work')

    #     filepath = 'flatdyn.h5'
    #     groupname = 'gbare'
    #     errmessage = 'There is no calculation data. Please perform the calculation again.'
    #     with h5py.File(filepath,'r') as file:
    #         if self.CheckGroup(filepath,groupname):
    #             group = file[groupname]
    #         else:
    #             print(errmessage)
    #             sys.exit()
            
    #         g0kf = group['g0kf'][:]

    #     os.chdir('..')

    #     return g0kf
    
class GreenInt(FLatDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, greenbare : np.ndarray = None, sigmah : np.ndarray = None, sigmaf : np.ndarray = None, sigmagwc : np.ndarray = None, hdf5file : str = 'glob.h5', group : str = None) -> object:
        
        if greenbare is None:
            print("Bare Green's function doesn't exist")
            sys.exit()
        super().__init__(crystal, ft)
        self.flatstc = FLatStc(crystal=crystal)
        self.kf = None
        self.kt = None
        self.rf = None
        self.rt = None
        self.gkfmu0 = None
        self.gktmu0 = None
        self.grfmu0 = None
        self.grtmu0 = None
        self.gbare = greenbare
        self.sigmah = sigmah
        self.sigmaf = sigmaf
        self.sigmac = sigmagwc
        self.occ = None
        self.occk = None
        self.occr = None
        self.mu = 0
        self.c = 0
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        print(f"Bare Green's function : \n{self.gbare[:,:,0,0,0]}")
        self.CalMu0()
        self.SearchMu()

    def CalMu0(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nomega = len(self.ft.omega)
        sigma = np.zeros((norb,norb,ns,nrk,nomega),dtype=np.complex128,order='F')
        print("Initialization start")
        if (self.sigmah is None)and(self.sigmaf is None)and(self.sigmac is None):
            self.gkfmu0 = self.gbare
        else:
            if (self.sigmah is not None):
                print(sigma[:,:,0,0,0])
                diag = np.diagonal(self.sigmah[:,:,0,0])
                const = np.mean(diag)
                self.c = np.real(const)
                print(const)
                sigma += self.StcEmbedding(self.sigmah)
                sigma += self.ChemEmbedding(-const)
                print(sigma[:,:,0,0,0])
            if (self.sigmaf is not None):
                print(sigma[:,:,0,0,0])
                sigma += self.StcEmbedding(self.sigmaf)
                print(sigma[:,:,0,0,0])
            if (self.sigmac is not None):
                print(sigma[:,:,0,0,0])
                sigma += self.sigmac
                print(sigma[:,:,0,0,0])
            self.gkfmu0 = self.Dyson(self.gbare,sigma) 
        

        self.gktmu0 = self.F2T(self.gkfmu0,1,1)
        self.grfmu0 = self.K2R(self.gkfmu0)
        self.grtmu0 = self.K2R(self.gktmu0)
        print("Initialization finish")
        return None
    
    def Occ(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        
        
        occk = np.zeros((norb,norb,ns,nrk),dtype=np.complex128,order='F')
        occ = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')
        
        print("Density matrixy calculation start")
        
        occk = -self.kt[...,-1]
    
        for irk in range(nrk):
            occ += occk[...,irk]
            
        occ /= nrk
        self.occ = occ
        self.occk = occk
        
        self.occr = self.flatstc.K2R(occk)
        print("Density matrixy calculation finish")
        return None
    
    def UpdateMu(self) -> np.ndarray:

        print("Chemical potential shift start")
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.ft.omega)

        gkfnew = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')
        chem = self.ChemEmbedding(self.mu)
    
    
        gkfnew = self.Dyson(self.gkfmu0,-chem)
        
        self.kf = gkfnew
        self.kt = self.F2T(gkfnew,1,1)
        # self.grf = self.K2R(self.Dyson(self.gkfmu0,-chem))
        # self.grt = self.K2R(self.F2T(self.Dyson(self.gkfmu0,-chem),1,1))
        self.rf = self.K2R(self.kf)
        self.rt = self.K2R(self.kt)
        print("Chemical potential shift finish")
        self.Occ()

        return None
    
    def NumOfE(self, mu : float):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.ft.omega)#self.ft.size
        tempmat = copy.deepcopy(self.gkfmu0)
        chem = self.ChemEmbedding(mu)
        gcalf = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')
        gcalt = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        
        gcalf = self.Dyson(tempmat,-chem)
        gcalt = self.F2T(gcalf,1,1)
        
        
        
        Ne = 0
        
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    Ne += -np.real(gcalt[iorb,iorb,js,irk,-1])
        Ne /= nrk
        
        N = self.crystal.nume
        # print(N,Ne,N-Ne)
        
        return N - Ne

    def SearchMu(self):
        
        print("Finding chemical potential start")
        mumin = -self.ft.omega[-1]*0.6
        mumax = self.ft.omega[-1]*0.6
        print(f"minimum : {mumin}, maximum : {mumax}")
        nmin = self.NumOfE(mumin)
        nmax = self.NumOfE(mumax)
        if (nmin < 0) or (nmax>0):
            print("Chemical potential is out of the bisection range")
            print(f"nmin : {nmin}, nmax : {nmax}")
            sys.exit()
        sol = scipy.optimize.brentq(self.NumOfE,mumin,mumax,xtol=1.0e-6)
        self.mu = sol
        print("Finding chemical potential finish")

        self.UpdateMu()
        return None
    
    def Save(self, fn: str, chem : bool = False):

        
        with h5py.File(self.hdf5file,'a') as file:
            if self.CheckGroup(self.hdf5file,self.group):
                group = file[self.group]
                if self.subgroup in group:
                    green = group[self.subgroup]
                else:
                    green = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                green = group.create_group(self.subgroup)
            green.create_dataset(fn,dtype=complex,data=self.kf)
            
            if chem:
                mureal = np.real(self.mu+self.c)
                green.create_dataset('mu',dtype=float,data=mureal)

        return None

    
class SigmaGWC(FLatDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, green : np.ndarray = None, wlat : np.ndarray = None, hdf5file : str = 'glob.h5',group : str = None) -> object:
        super().__init__(crystal, ft)
        self.flatstc = FLatStc(crystal=crystal)
        self.rt = None
        self.rf = None
        self.kt = None
        self.kf = None
        self.stck = None
        self.z = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        if green is None:
            print("Error, green doesn't exist")
            sys.exit()

        if wlat is None:
            print("Error, wlat doesn't exist")
            sys.exit()
        self.green = green
        self.wlat = wlat
        self.Cal()

    def Cal(self)->np.ndarray: #SigmaGWC
        '''
        Generate correlated self-energy
        input : Wc(R,t), G(R,t)

        return : crtau, crfreq, cktau, ckfreq
        '''
        
        G = self.green
        Wc = self.wlat
        norbc = G.shape[0]
        ns = G.shape[2]
        nr = G.shape[3]
        ntau = G.shape[4]
        norb = Wc.shape[0]

        crtau = np.zeros((norbc,norbc,ns,nr,ntau),dtype=np.complex128,order='F')
    
        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex128,order='F')
        # for itau in range(ntau):
        #     for ir in range(nr):
        #         tempmat = self.crystal.OrbSpin2Composite(Wc[:,:,:,:,ir,itau])
        #         for ind1 in range(norb*ns):
        #             nn1= [0]*2
        #             ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
        #             [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
        #             iorbc1 = self.crystal.FIndex([a,m1])
        #             iorbc4 = self.crystal.FIndex([a,m4])
        #             for ind2 in range(norb*ns):
        #                 nn2 = [0]*2
        #                 ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
        #                 [b,[m3,m2]] = self.crystal.BAtomOrb(jorb)
        #                 iorbc3 = self.crystal.FIndex([b,m3])
        #                 iorbc2 = self.crystal.FIndex([b,m2])
        #                 if js == ks:
        #                     crtau[iorbc1,iorbc2,js,ir,itau] += -G[iorbc4,iorbc3,js,ir,itau]*tempmat[ind1,ind2]
        
        for itau in range(ntau):
            for ir in range(nr):
                for ind2 in range(norb*ns):
                    nn2 = [0]*2
                    ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
                    [b,[m3,m2]] = self.crystal.BAtomOrb(jorb)
                    iorbc3 = self.crystal.FIndex([b,m3])
                    iorbc2 = self.crystal.FIndex([b,m2])
                    for ind1 in range(norb*ns):
                        nn1 = [0]*2
                        ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                        [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                        iorbc1 = self.crystal.FIndex([a,m1])
                        iorbc4 = self.crystal.FIndex([a,m4])
                        if js==ks:
                            crtau[iorbc1,iorbc2,js,ir,itau] += -G[iorbc4,iorbc3,js,ir,itau]*Wc[iorb,jorb,js,ks,ir,itau]
                
                                        

        cktau = self.R2K(crtau)
        ckfreq = self.T2F(cktau)
        crfreq = self.T2F(crtau)

        self.rt = crtau
        self.kt = cktau
        self.rf = crfreq
        self.kf = ckfreq

        return None
    
    def SigmaStc(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nfreq = len(self.ft.omega)#self.ft.size

        sigmastc = np.zeros((norb,norb,ns,nk),dtype=np.complex128,order="F")
        tempmat = np.zeros((norb,norb,ns,nk,nfreq),dtype=np.complex128,order="F")
        sigma = copy.deepcopy(self.kf)
        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    tempmat[:,:,js,ik,ifreq] = np.transpose(np.conjugate(sigma[:,:,js,ik,ifreq]))

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        sigmastc[iorb,jorb,js,ik] = (self.kf[iorb,jorb,js,ik,0]+tempmat[iorb,jorb,js,ik,0])/2

        self.stck = sigmastc
        # self.Save('sigmastc',obj=sigmastc)

        return None
    
    def Zfactor(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nfreq = len(self.ft.omega)#self.ft.size
        beta = self.ft.beta

        z = np.zeros((norb,norb,ns,nk),dtype=np.complex128,order='F')
        # identity = np.zeros((norb,norb,ns,nk,nfreq),dtype=np.complex128,order='F')
        tempmat = np.zeros((norb,norb,ns,nk),dtype=np.complex128,order='F')
        tempmat2 = np.zeros((norb,norb,ns,nk),dtype=np.complex128,order='F')
        sigma = copy.deepcopy(self.kf)
        # for ifreq in range(nfreq):
        #     for ik in range(nk):
        #         for js in range(ns):
        #             identity[:,:,js,ik,ifreq] = np.eye(norb,norb,dtype=np.complex128,order='F')
        #             tempmat[:,:,js,ik,ifreq] = np.transpose(np.conjugate(self.kf[:,:,js,ik,ifreq]))

        # for ifreq in range(nfreq):
        #     for ik in range(nk):
        #         for js in range(ns):
        #             for iorb in range(norb):
        #                 for jorb in range(norb):
        #                     tempmat2[iorb,jorb,js,ik] = (identity[iorb,jorb,js,ik,ifreq]+beta*(self.kf[iorb,jorb,js,ik,ifreq]-tempmat[iorb,jorb,js,ik,ifreq])/(2*np.pi))
        for ik in range(nk):
            for js in range(ns):
                tempmat2[:,:,js,ik] = np.transpose(np.conjugate(sigma[:,:,js,ik,0]))
        iw = beta/(2.0*np.pi)*1j
        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        if (iorb==jorb):
                            tempmat[iorb,jorb,js,ik] = 1.0+iw*(sigma[iorb,jorb,js,ik,0]-tempmat2[iorb,jorb,js,ik])
                        else:
                            tempmat[iorb,jorb,js,ik] = iw*(sigma[iorb,jorb,js,ik,0]-tempmat2[iorb,jorb,js,ik])
        
        z = self.flatstc.Inverse(tempmat)

        self.z = z
        # self.Save('zfactor',obj=z)
        return None
    
    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            if self.CheckGroup(self.hdf5file,self.group):
                group = file[self.group]
                if self.subgroup in group:
                    sigmac = group[self.subgroup]
                else:
                    sigmac = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                sigmac = group.create_group(self.subgroup)
            

            if obj != None:
                sigmac.create_dataset(fn,dtype=complex,data=obj)
            else:
                sigmac.create_dataset(fn,dtype=complex,data=self.kf)

        return None
