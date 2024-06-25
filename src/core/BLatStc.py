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
import copy, gc
import h5py
from .Crystal import Crystal
from .BLocStc import VLoc
diage_path = os.environ.get('DIAGE','')
path = diage_path+"/modules"
sys.path.append(path)
import DiagE

class BLatStc(object):

    def __init__(self,crystal : Crystal):
        self.crystal = crystal


    def Inverse(self, matin : np.ndarray)-> np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]

        matout = np.zeros((norb,norb,ns,ns,nrk),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex64)
        tempmat2 = np.zeros((norb*ns,norb*ns),dtype=np.complex64)

        for irk in range(nrk):
            tempmat = self.crystal.OrbSpin2Composite(matin[...,irk])
            tempmat2 = np.linalg.inv(tempmat)
            matout[...,irk] = self.crystal.Composite2OrbSpin(tempmat2)
        
        return matout


    def K2R(self, matk : np.ndarray)-> np.ndarray:

        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matk.shape[0]
        ns = self.crystal.ns
        nrk = len(rkvec)

        matr = np.zeros((norb,norb,ns,ns,nrk),dtype=np.complex64,order='F')
        tempmat = copy.deepcopy(matk)

        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                            [b,[m2,m3]] = self.crystal.BAtomOrb(jorb)

                            delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
                            phase = np.exp(2.0j*np.pi*np.dot(rkvec[irk],delta))
                        
                            tempmat[iorb,jorb,js,ks,irk] *= phase
        
        matr = DiagE.fourier.blatstc_k2r(rkgrid,tempmat)

        return matr
    
    def R2K(self,matr : np.ndarray)-> np.ndarray:

        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matr.shape[0]
        ns = self.crystal.ns
        nrk = len(rkvec)

        matk = np.zeros((norb,norb,ns,ns,nrk),dtype=np.complex64,order='F')

        matk = DiagE.fourier.blatstc_r2k(rkgrid,matr)

        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                            [b,[m2,m3]] = self.crystal.BAtomOrb(jorb)

                            delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
                            phase = np.exp(-2.0j*np.pi*np.dot(rkvec[irk],delta))
                            
                            matk[iorb,jorb,js,ks,irk] *= phase
        
        return matk
    
    def Mixing(self,iter : int, mix : float, Bb : np.ndarray, Bold : np.ndarray)-> np.ndarray:

        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nrk = Bb.shape[4]

        Bnew = np.zeros((norb,norb,ns,ns,nrk),dtype=np.complex64,order='F')

        if iter == 1:
            mix = 1.0

        Bnew = mix*Bb+(1.0-mix)*Bold

        return Bnew

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = mat1.shape[0]
        ns = mat1.shape[2]
        nrk = mat1.shape[4]

        matout = np.zeros((norb,norb,ns,ns,nrk),dtype=np.complex64,order='F')

        matout = DiagE.dyson.blatstc(mat1,mat2)

        return matout


    def Projection(self, matin : np.ndarray):

        norbc = self.crystal.bprojector.shape[1]
        nspace = self.crystal.bprojector.shape[3]
        ns = self.crystal.ns

        matout = np.zeros((norbc,norbc,ns,ns,nspace),dtype=np.complex64,order='F')

        for ispace in range(nspace):
            matout[...,ispace] = DiagE.projection.blatstc(matin,self.crystal.bprojector[...,ispace])

        return matout
    
    def Quad2Double(self,matin : np.ndarray)->np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        
        matout = np.zeros((norb,norb,ns,ns,nrk),dtype=np.complex64,order='F')

        for irk in range(nrk):
            for ks in range(ns):
                for js in range(ns):
                    matout[:,:,js,ks,irk] = self.crystal.Quad2Double(matin[:,:,:,:,js,ks,irk])

        return matout
    
    def Double2Quad(self,matin : np.ndarray)->np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros((norb,norb,norb,norb,ns,ns,nrk),dtype=np.complex64,order='F')

        for irk in range(nrk):
            for ks in range(ns):
                for js in range(ns):
                    matout[:,:,:,:,js,ks,irk] = self.crystal.Double2Quad(matin[:,:,js,ks,irk])

        return matout
    
    def Double2Full(self,matin : np.ndarray)->np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros((norb*norb,norb*norb,ns,ns,nrk),dtype=np.complex64,order='F')
        
        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    matout[:,:,js,ks,irk] = self.crystal.Double2Full(matin[:,:,js,ks,irk])

        return matout
    
    def Full2Double(self, matin : np.ndarray) -> np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros((norb,norb,ns,ns,nrk),dtype=np.complex64,order='F')

    
        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    matout[:,:,js,ks,irk] = self.crystal.Full2Double(matin[:,:,js,ks,irk])

        return matout
    
    def Quad2Full(self,matin : np.ndarray) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        
        matout = np.zeros((norb*norb,norb*norb,ns,ns,nrk),dtype=np.complex64,order='F')

        
        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    matout[:,:,js,ks,irk] = self.crystal.Quad2Full(matin[:,:,:,:,js,ks,irk])

        return matout
    
    def Full2Quad(self, matin : np.ndarray)->np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        

        matout = np.zeros((norb,norb,norb,norb,ns,ns,nrk),dtype=np.complex64,order='F')

        
        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    matout[:,:,:,:,js,ks,irk] = self.crystal.Full2Quad(matin[:,:,js,ks,irk])

        return matout

    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]

        # if os.path.exists('blatstc'):
        #     pass
        # else:
        #     os.mkdir("blatstc")
        # os.chdir('blatstc')

        with open(fn+'.txt','w') as f:
            f.write("#iorb, jorb, is, js, irk, Re(B(k)), Im(B(k))\n")
            for irk in range(nrk):
                for ks in range(ns):
                    for js in range(ns):
                        for jorb in range(norb):
                            for iorb in range(norb):
                                f.write(f"{iorb} {jorb} {js} {ks} {irk} {matin[iorb,jorb,js,ks,irk].real} {matin[iorb,jorb,js,ks,irk].imag}\n")
        
        # os.chdir('..')

        return None
    
    def HermitianCheck(self,matin : np.ndarray):

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]


        errmessage = 'The matrix is not hermitian. Check the input file again'
        for ik in range(nk):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            err = matin[iorb,jorb,js,ks,ik]-np.conjugate(matin[jorb,iorb,js,ks,ik])
                            if abs(err)>1.0e-6:
                                print(errmessage)
                                sys.exit()
        return None
    
    def R2KArb(self,matr : np.ndarray = None,kpoint : np.ndarray = None): # R2KAny

        # if self.crystal.kpath == None:
        #     print("Error, kpath doesn't generate")
        #     sys.exit()
        # kpoint = self.crystal.kpath
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nr = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        nk = len(kpoint)

        self.crystal.Rvec()
        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb,norb,ns,ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            temp = 0
                            for ir in range(nr):
                                temp += tempmat[iorb,jorb,js,ks,ir]*np.exp(-2.0j*np.pi*(kpoint[ik]@self.crystal.rvec[ir]))
                            [a,m1] = self.crystal.FAtomOrb(iorb)
                            [b,m2] = self.crystal.FAtomOrb(jorb)
                            delta = self.crystal.basisf[a,:]-self.crystal.basisf[b,:]
                            phase = np.exp(-2.0j*np.pi*(kpoint[ik]@delta))
                            matk[iorb,jorb,js,ks,ik] = temp*phase
        
        return matk
    
    def CheckGroup(self, filepath :str, group : str):
        
        with h5py.File(filepath,'r') as file:
            return group in file

class VBare(BLatStc):

    def __init__(self, crystal: Crystal,vloc : VLoc = None, orboption : dict = None, intamp : list = None, ohno : bool = False):
        super().__init__(crystal)
        self.k = None
        self.r = None
        self.intamp = intamp
        self.locoption = orboption
        self.nonlock = None
        self.nonlocr = None
        self.sigmaonsiter = None
        if (ohno==False)and(intamp==None):
            print("Only calculate the local coulomb interaction")
        if vloc == None:
            if orboption != None:
                self.vloc = VLoc(crystal,orboption)
            else:
                print("Error, orboption is not exsist. v local can't generate in here")
        else:
            self.vloc = vloc
        

        if (ohno)and(intamp!=None):
            print("Choose only one way to generate non-loc bare coulomb interaction")
            sys.exit()

        if ohno:
            if intamp==None:
                self.OhnoParameter()
                self.Cal()
        else:
            if intamp != None:
                # self.InteractingAmplitue(intamp)
                self.Cal()
        self.LocPlusNonLoc()
        # self.GetOnsiteEnergy()
        

    def Cal(self):

        errmessage = "Wrong value entered, please check the input.ini file"
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nk = len(rkvec)
        vnlk = np.zeros((norb,norb,ns,ns,nk),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb,norb,ns,ns,rkgrid[0],rkgrid[1],rkgrid[2]),dtype=np.complex64,order='F')

        # for ik in range(nk):
        #     for js in range(ns):
        #         for ks in range(ns):
        #             for ind in self.intamp:
        #                 vij = ind[0]
        #                 iorb = ind[1]
        #                 jorb = ind[2]
        #                 R = np.array(ind[3])
        #                 [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
        #                 [b,[m2,m3]] = self.crystal.BAtomOrb(jorb)

        #                 rvec = self.crystal.basisf[a,:] - self.crystal.basisf[b,:] + R
        #                 phase = np.exp(-2.0j*np.pi*np.dot(rkvec[ik],rvec))

        #                 vnlk[iorb,jorb,js,ks,ik] += vij*phase
        #                 vnlk[jorb,iorb,js,ks,ik] += vij*np.conjugate(phase)

        for js in range(ns):
            for ks in range(ns):
                for ind in self.intamp:
                    vij = ind[0]
                    (a,m) = ind[1]#self.crystal.FAtomOrb(ind[1])
                    (b,mp) = ind[2]#self.crystal.FAtomOrb(ind[2])
                    iorb = self.crystal.BIndex([a,[m,m]])
                    jorb = self.crystal.BIndex([b,[mp,mp]])
                    R = ind[3]
                    
                    
                    
                    # tempmat[iorb,jorb,js,ks,R[0],R[1],R[2]] += vij
                        
                    if (iorb==jorb)and(R==[0,0,0]):
                        # tempmat[iorb,jorb,js,ks,R[0],R[1],R[2]] += vij
                        print(errmessage)
                        sys.exit()
                
                    # else:
                    tempmat[iorb,jorb,js,ks,R[0],R[1],R[2]] = vij
                    tempmat[jorb,iorb,js,ks,-R[0],-R[1],-R[2]] = vij
                    
        
        vnlr = tempmat.reshape((norb,norb,ns,ns,nk),order='F')
        vnlk = self.R2K(vnlr)
        self.HermitianCheck(vnlk)

        self.nonlocr = vnlr
        self.nonlock = vnlk
        # self.nonlock = vnlk
        # self.nonlocr = self.K2R(vnlk)

        return None
    
    # def InteractingAmplitue(self,intamp : list)-> list:

    #     pass

    def LocPlusNonLoc(self):
        
        vloc = self.vloc.vloc
        vnlk = self.nonlock

        norb = len(self.crystal.bind) 
        ns = self.crystal.ns 
        nrk = len(self.crystal.kpoint) 

        vbare = np.zeros((norb,norb,ns,ns,nrk),dtype=np.complex64,order='F')
        if (self.intamp == None):
            for ik in range(nrk):
                vbare[...,ik] = vloc
        else:
            for ik in range(nrk):
                vbare[...,ik] = vloc + vnlk[...,ik]
        
        self.k = vbare
        self.r = self.K2R(vbare)

        return None
    
    def Save(self):

        os.chdir('work')
        
        filepath = 'blatstc.h5'
        groupname = 'vbare'
        with h5py.File(filepath,'a') as file:
            if self.CheckGroup(filepath,groupname):
                group = file[groupname]
            else:
                group=file.create_group(groupname)
            
            group.create_dataset('vk',dtype=complex,data=self.k)
        os.chdir('..')

        return None
    
    def OhnoParameter(self):

        norbc = len(self.crystal.find)
        U = self.locoption["option"][1]["value"][0]
        V = []
        R = copy.deepcopy(self.crystal.rkgrid)
        a0 = 0.592
        au = 27.2114

        for iz in range(R[2]):
            for iy in range(R[1]):
                for ix in range(R[0]):
                    for jorbc in range(norbc):
                        for iorbc in range(norbc):
                            if (iorbc == jorbc)and([ix,iy,iz]==[0,0,0]):
                                continue
                            if (iorbc<=jorbc):
                                rvec = np.array([ix,iy,iz])
                                a, m1 = self.crystal.FAtomOrb(iorbc)
                                b, m2 = self.crystal.FAtomOrb(jorbc)
                                delta = self.crystal.basisc[a,:]-(self.crystal.basisc[b,:]+rvec@self.crystal.avec)
                                rij = self.RMin(delta)
                                Rij = rij*a0 # convert angstrom to a.u.
                                u = U/au # convert eV to a.u.
                                vij = 1/(Rij**2+1/u**2)**(1/2)
                                Vij = vij*au
                                V.append([Vij,(a,m1),(b,m2),rvec])
#        print(V)
        self.intamp = V

        return None
    
    def RMin(self,d : np.ndarray):

        cell = copy.deepcopy(self.crystal.rkgrid)
        for ii in range(len(cell)):
            if cell[ii] == 1:
                cell[ii] = 0

        [a,b,c] = (np.array(cell)@(self.crystal.avec)).tolist()

        R = 1000000
        Rtemp = 0
        for kk in range(-1,2):
            for jj in range(-1,2):
                for ii in range(-1,2):
                    rvec = np.array([a*ii,b*jj,c*kk])
                    R1 = np.linalg.norm(d)
                    R2 = np.linalg.norm(d+rvec)
                    # print(ii,jj,kk,rvec)
                    # print(f"Rtemp : {Rtemp}, R : {R}")
                    if R1 < R2:
                        Rtemp = R1
                    else:
                        Rtemp = R2
                    if abs(Rtemp) == 0:
                        continue
                    if Rtemp < R:
                        R = Rtemp
                    else:
                        continue
        del Rtemp,R1,R2,a,b,c,cell
        gc.collect()
        
        return R
    
    # def OhnoParameter(self):

    #     norbc = len(self.crystal.find)
    #     kappa = 2.0
    #     U = self.locoption["option"][1]["value"][0]
    #     V = []
    #     R = copy.deepcopy(self.crystal.rkgrid)
        
    #     # Rmin = lambda R1, R2: R1 if R1<R2 else R2

    #     for iz in range(R[2]):
    #         for iy in range(R[1]):
    #             for ix in range(R[0]):
    #                 for jorbc in range(norbc):
    #                     for iorbc in range(norbc):
    #                         if iorbc < jorbc:
    #                             rvec = [ix,iy,iz]
    #                             a, m1 = self.crystal.FAtomOrb(iorbc)
    #                             b, m2 = self.crystal.FAtomOrb(jorbc)
    #                             delta = self.crystal.basisc[a,:]- (self.crystal.basisc[b,:]+np.array(rvec)@self.crystal.avec)
    #                             # R1 = np.linalg.norm(delta)
    #                             # R2 = np.linalg.norm(delta+np.array(R)@self.crystal.avec)

                                
    #                             V.append([U/(kappa*np.sqrt(1+0.6117*Rij**2)),(a,m1),(b,m2),rvec])
    #                         if iorbc==jorbc:
    #                             if [ix,iy,iz]==[0,0,0]:
    #                                 continue
    #                             rvec = [ix,iy,iz]
    #                             a, m1 = self.crystal.FAtomOrb(iorbc)
    #                             b, m2 = self.crystal.FAtomOrb(jorbc)
    #                             delta = self.crystal.basisc[a,:]- (self.crystal.basisc[b,:]+np.array(rvec)@self.crystal.avec)
    #                             R1 = np.linalg.norm(delta)
    #                             R2 = np.linalg.norm(delta+np.array(R)@self.crystal.avec)

    #                             Rij = Rmin(R1,R2)
    #                             V.append([U/(kappa*np.sqrt(1+0.6117*Rij**2)),(a,m1),(b,m2),rvec])
    #                         else:
    #                             continue
    #     self.intamp = V

    #     return None