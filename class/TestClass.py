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

class Crystal(object): # chemical potential object, num of electron 
    def __init__(self,latt : list, basisposition : dict, ns : int, soc : bool, rkgrid : list, orboption : dict, N : float, supercell : list = [1,1,1], impdict : dict = None):
        latt = np.array(latt,dtype=float)
        # basisposition = np.array(basisposition,dtype=float)
        # tempmat = np.zeros((basisposition.shape[0],basisposition.shape[1]),dtype=float)
        # for jj in range(basisposition.shape[1]):
        #     for ii in range(basisposition.shape[0]):
        #         if 0<=basisposition[ii,jj]<=1:
        #             tempmat[ii,jj] = basisposition[ii,jj]
        #         if basisposition[ii,jj] < 0 :
        #             tempmat[ii,jj] = 1 + basisposition[ii,jj]
        #         if basisposition[ii,jj] > 1 :
        #             tempmat[ii,jj] = basisposition[ii,jj] - 1
        self.avec = latt
        a = latt[0]
        b = latt[1]
        c = latt[2]
        alpha = np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        beta = np.degrees(np.arccos(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))))
        gamma = np.degrees(np.arccos(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c))))
        if basisposition["CorF"] == "C":
            pos = np.array(basisposition["pos"])
            lat = Lattice.from_parameters(np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(c),alpha,beta,gamma)
            structure = Structure(lat,["X"]*len(pos),pos,coords_are_cartesian=True)
            structurebasisc = []
            structurebasisf = []
            for site in structure.sites:
                structurebasisc.append(site.coords.tolist())
                structurebasisf.append(site.frac_coords.tolist())
            print(structure)
        if basisposition["CorF"] == "F":
            pos = np.array(basisposition['pos'])
            lat = Lattice.from_parameters(np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(c),alpha,beta,gamma)
            structure = Structure(lat,["X"]*len(pos),pos)
            structurebasisc = []
            structurebasisf = []
            for site in structure.sites:
                structurebasisc.append(site.coords.tolist())
                structurebasisf.append(site.frac_coords.tolist())
            print(structure)
        structurebasisc = np.array(structurebasisc)
        structurebasisf = np.array(structurebasisf)
        # for jj in range(structurebasisf.shape[1]):
        #     for ii in range(structurebasisf.shape[0]):
        #         if (structurebasisf[ii,jj] >= 1):
        #             structurebasisf[ii,jj] -= 1
        #         if (structurebasisf[ii,jj] < 0):
        #             structurebasisf[ii,jj] += 1
        
        self.basisf = structurebasisf
        self.basisc = np.dot(self.basisf,self.avec)
        self.ns = ns
        self.soc = soc
        self.nume = N
        # self.basisf = tempmat
        # self.basisc = np.dot(self.basisf,self.avec)
        self.bvec = np.zeros((3,3))
        self.vol=np.dot(np.cross(latt[:,0], latt[:,1]), latt[:,2])
        self.bvec[:,0]=2*np.pi*np.cross(latt[:,1], latt[:,2])/self.vol
        self.bvec[:,1]=2*np.pi*np.cross(latt[:,2], latt[:,0])/self.vol
        self.bvec[:,2]=2*np.pi*np.cross(latt[:,0], latt[:,1])/self.vol
        
        self.kpath = None
        self.rkgrid = rkgrid
        nk = rkgrid[0]*rkgrid[1]*rkgrid[2]
        kpoint_temp=np.array(list(itertools.product(np.linspace(0,1,num=rkgrid[2],endpoint=False),np.linspace(0,1,num=rkgrid[1],endpoint=False),np.linspace(0,1,num=rkgrid[0],endpoint=False))))
        kpoint=np.fliplr(kpoint_temp)
        self.kpoint = kpoint
        self.nk = nk

        self.rvec = None
        self.kpath = None
        self.kdist = None
        self.knode = None

        self.find = {}
        self.bind = {}
        self.b2f = []
        self.c2f = []
        self.c2b = []
        self.probspace = {}
        self.fimpdict = {}
        self.bimpdict = {}
        self.fprojector = None
        self.bprojector = None

        self.mappingidx = []
        templist = []
        for key, val in orboption.items():
            templist.append([key-1,val])
        self.orboption = orboption


        self.SetBasisIndex(templist)
        if impdict != None:
            self.Projector(impdict)

    def Kpath(self,kpath : list,nk : int) -> np.ndarray:


        kpath = np.array(kpath,dtype=float)
        nnod = kpath.shape[0]
        kmat = np.linalg.inv(np.dot(self.avec,self.avec.T))
        knode = np.zeros(nnod,dtype=float)
        for n in range(1,nnod):
            dk = kpath[n] - kpath[n-1]
            l = np.sqrt(dk@(kmat@dk))
            knode[n] = knode[n-1]+l

        

        indnod = []
        for n in range(1,nnod-1):
            if n == 1:
                indnod.append(0)
            frac = knode[n]/knode[-1]
            indnod.append(int(round(frac*(nk-1))))
        indnod.append(nk-1)

        kdist = np.zeros(nk,dtype=float)
        kvec = np.zeros((nk,kpath.shape[1]),dtype=float)
        kvec[0] = kpath[0]
        
        for i in range(1,nnod):
            n1 = indnod[i-1]
            n2 = indnod[i]
            kd1 = knode[i-1]
            kd2 = knode[i]
            k1 = kpath[i-1]
            k2 = kpath[i]
            # print(n1,n2,kd1,kd2,k1,k2)
            for j in range(n1,n2+1):
                frac = float(j-n1)/float(n2-n1)
                kdist[j] = kd1 + frac*(kd2-kd1)
                kvec[j] = k1 + frac*(k2-k1)

        self.kpath = kvec
        self.kdist = kdist
        self.knode = knode
    
    def SetBasisIndex(self,orboption : list) -> dict:
        '''
        Modify orbital option for each atom basis
        '''
        for option in orboption:
            find = []
            bind = []
            orblist = list(range(option[1]))

            for m1 in range(option[1]):
                find.append([option[0],m1])
            for m2, m1 in itertools.product(orblist,orblist):
                bind.append([option[0],[m1,m2]])
            
            forb = len(self.find)
            borb = len(self.bind)
            ii = 0
            for iorb in range(forb,forb+option[1]):
                self.find[iorb] = find[ii]
                ii +=1
            ii = 0
            for iorb in range(borb,borb+option[1]**2):
                self.bind[iorb] = bind[ii]
                ii +=1
                self.Boson2Fermion(iorb)
            self.Composite2Boson()
            self.Composite2Fermion()
    
    def FAtomOrb(self, key : int) -> list:
        '''
        input : composite index for fermion
        output : atom and orbital index in fermion case

        e.g.
        0 -> [0,0]
        '''
        return self.find[key]
    
    def FIndex(self,val : list) -> int:
        '''
        input : atom and orbital index with list
        output : composite index for fermion

        e.g.
        [0,0] -> 0
        '''
        
        for key, value in self.find.items():
            if value == val:
                return key
    
    def BAtomOrb(self,key : int) -> list:
        '''
        input : composite index for fermion
        output : atom and orbital index in boson case

        e.g.
        0 -> [0,[0,0]]
        '''
        return self.bind[key]
    
    def BIndex(self,val:list) -> int:
        '''
        input : atom and orbital index with list
        output : composite index for boson

        e.g.
        [0,[0,0]] -> 0
        '''
        for key, value in self.bind.items():
            if val==value:
                return key
            
    def Boson2Fermion(self,ind : int):
        '''
        Mapping with boson index to fermion index
        '''
        [a, [m1,m2]] = self.BAtomOrb(ind)
        iorbc1 = self.FIndex([a,m1])
        iorbc2 = self.FIndex([a,m2])
        self.b2f.append([iorbc1,iorbc2])
    
    def Composite2Fermion(self):
        '''
        Mapping with fermion index to composite index
        '''
        norbc = len(self.find)
        norb = norbc*norbc
        c2f = []

        for iorbc in range(norbc):
            for jorbc in range(norbc):
                nn1 = [iorbc,jorbc]
                iorb, nn1 = self.indexing(norb,2,[norbc,norbc],1,0,nn1)
                c2f.append([iorbc,jorbc])
        self.c2f = c2f
    
    def Composite2Boson(self):

        norbc = len(self.find)
        ndim = norbc*norbc
        c2b = []

        for ind in range(ndim):
            nn1 = [0]*2
            ind,[iorbc,jorbc] = self.indexing(ndim,2,[norbc,norbc],0,ind,nn1)
            [a,m1] = self.FAtomOrb(iorbc)
            [a_p,m2] = self.FAtomOrb(jorbc)
            if a==a_p:
                borb = self.BIndex([a,[m1,m2]])
                if borb is not None:
                    c2b.append([borb,ind])
        self.c2b = c2b
    
    def Composite2OrbSpin(self, mat : np.ndarray):
        
        norb = len(self.bind)
        ns = self.ns
        matout = np.zeros((norb,norb,ns,ns),dtype=np.complex64,order='F')
        ndim = mat.shape[0]

        for ind1 in range(ndim):
            nn1 = [0]*2
            ind1, [iorb,js] = self.indexing(ndim,2,[norb,ns],0,ind1,nn1)
            for ind2 in range(ndim):
                nn2 = [0]*2
                ind2, [jorb,ks] = self.indexing(ndim,2,[norb,ns],0,ind2,nn2)
                matout[iorb,jorb,js,ks] = mat[ind1,ind2]

        return matout
    
    def OrbSpin2Composite(self,mat : np.ndarray):
        
        norb = mat.shape[0]
        ns = mat.shape[2]
        matout = np.zeros((norb*ns,norb*ns),dtype=np.complex64,order='F')
        
        for js in range(ns):
            for iorb in range(norb):
                nn1 = [iorb,js]
                ind1, nn1 = self.indexing(norb*ns,2,[norb,ns],1,0,nn1)
                for ks in range(ns):
                    for jorb in range(norb):
                        nn2 = [jorb,ks]
                        ind2, nn2 = self.indexing(norb*ns,2,[norb,ns],1,0,nn2)
                        matout[ind1,ind2] = mat[iorb,jorb,js,ks]
        return matout
    
    def Quad2Double(self,mat : np.ndarray) -> np.ndarray: # 4 index <-> 2 index

        norb = len(self.bind)

        matret = np.zeros((norb,norb),dtype=np.complex64)

        for iorb, [iorbc,lorbc] in enumerate(self.b2f):
            for jorb, [jorbc,korbc] in enumerate(self.b2f):
                matret[iorb,jorb] = mat[iorbc,jorbc,korbc,lorbc]

        return matret
    
    def Double2Quad(self, mat : np.ndarray) -> np.ndarray:

        norbc = len(self.find)

        matret = np.zeros((norbc,norbc,norbc,norbc),dtype=np.complex64,order='F')

        for iorb, [iorbc,lorbc] in enumerate(self.b2f):
            for jorb, [jorbc,korbc] in enumerate(self.b2f):
                matret[iorbc,jorbc,korbc,lorbc] = mat[iorb,jorb]

        return matret

    def Full2Quad(self,mat : np.ndarray) -> np.ndarray:

        norbc = len(self.find)

        matret = np.zeros((norbc,norbc,norbc,norbc),dtype=np.complex64,order='F')
        
        for iorb, [iorbc,lorbc] in enumerate(self.c2f):
            for jorb, [jorbc,korbc] in enumerate(self.c2f):
                matret[iorbc,jorbc,korbc,lorbc] = mat[iorb,jorb]
        
        return matret
    
    def Quad2Full(self, mat : np.ndarray) -> np.ndarray:

        norb = len(self.find)**2

        matret = np.zeros((norb,norb))

        for iorb, [iorbc,lorbc] in enumerate(self.c2f):
            for jorb, [jorbc,korbc] in enumerate(self.c2f):
                matret[iorb,jorb] = mat[iorbc,jorbc,korbc,lorbc]
        
        return matret
    
    def Full2Double(self, mat : np.ndarray) -> np.ndarray:

        norb = len(self.bind)

        matret = np.zeros((norb,norb),dtype=np.complex64,order='F')

        for iorb, ind1 in self.c2b:
            for jorb, ind2 in self.c2b:
                matret[iorb,jorb] = mat[ind1,ind2]
        
        return matret
    
    def Double2Full(self, mat : np.ndarray) -> np.ndarray:

        nind = len(self.find)**2

        matret = np.zeros((nind,nind),dtype=np.complex64,order='F')

        for iorb, ind1 in self.c2b:
            for jorb, ind2 in self.c2b:
                matret[ind1,ind2] = mat[iorb,jorb]
        
        return matret ## construct
    
    def Projector(self,impdict : dict):
        '''
        Generate the projector for impurity quantity
        
        e.g.
        input : {"1" : [[0,0],[1,0]]}
        output : fprojector, bprojector
        '''

        nspace = 0
        forbc = 0
        borbc = 0
        ns = self.ns
        probspace = {}
        fimpdict = {}
        bimpdict = {}

        for key, val in impdict.items():
            # probspace[key] = []
            for orblist in val:
                atom = 0
                for orb in orblist:
                    if orb == orblist[0]:
                        atom = orb[0]
                    if atom != orb[0]:
                        print("Different atoms are involved in the same space")
                        sys.exit()
            probspace[key] = [nspace+i for i in range(len(val))]
            nspace += len(val)
        
        self.probspace = probspace
        
        for key, val in impdict.items():
            fimpdict[key] = []
            for orblist in val:
                templist = []
                for orb in orblist:
                    find = self.FIndex(orb)
                    templist.append(find)
                fimpdict[key].append(templist)
        self.fimpdict = fimpdict
        for val in fimpdict.values():
            for orb in val:
                if len(orb) > forbc:
                    forbc = len(orb)
        for key, val in fimpdict.items():
            bimpdict[key] = []
            for orb in val:
                templist = []
                for iorb in orb:
                    for jorb in orb:
                        [a,m1] = self.FAtomOrb(iorb)
                        [b,m2] = self.FAtomOrb(jorb)
                        if a==b:
                            bind = self.b2f.index([iorb,jorb])
                            templist.append(bind)
                bimpdict[key].append(templist)
        for val in bimpdict.values():
            for orb in val:
                if len(orb)>borbc:
                    borbc = len(orb)
        self.bimpdict = bimpdict
        fprojector = np.zeros((len(self.find),forbc,ns,nspace),dtype=float,order='F')
        bprojector = np.zeros((len(self.bind),borbc,ns,nspace),dtype=float,order='F')

        for js in range(ns):
            for key, val in probspace.items():
                for ii, ispace in enumerate(val):
                    for ind in self.fimpdict[key][ii]:
                        fprojector[ind,self.fimpdict[key][ii].index(ind),js,ispace] = 1.0
        
        for js in range(ns):
            for key, val in probspace.items():
                for ii, ispace in enumerate(val):
                    for ind in self.bimpdict[key][ii]:
                        bprojector[ind,self.bimpdict[key][ii].index(ind),js,ispace] = 1.0

        self.fprojector = fprojector
        self.bprojector = bprojector

        return None
        

    def indexing(self,ntot, ndivision, divisionarray, flag, n1, n2):
        tmpsize = 1
        for size in divisionarray:
            tmpsize *= size

        if tmpsize != ntot:
            print('array_division wrong')
            return

        if flag == 1:
            n1 = n2[0]
            for ii in range(1, ndivision):
                tempcnt = 1
                for jj in range(ii):
                    tempcnt *= divisionarray[jj]
                n1 += (n2[ii] ) * tempcnt
        else:
            n2_array = [0] * ndivision
            tempcnt = n1
            for ii in range(ndivision - 1):
                n2_array[ii] = tempcnt - ((tempcnt) // divisionarray[ii]) * divisionarray[ii]
                tempcnt = (tempcnt - n2_array[ii])//divisionarray[ii]
            n2_array[ndivision - 1] = tempcnt

            # Copy the values from the temporary array to the n2 output array
            for i in range(ndivision):
                n2[i] = n2_array[i]

        return n1, n2
    
    def FindPositions(self,array, value):
        positions = []
        for row_index, row in enumerate(array):
            for col_index, col_value in enumerate(row):
                if col_value == value:
                    positions.append([row_index, col_index])
        return positions
    
    def R2mR(self) -> list: # move to crystal

        rkvec = self.kpoint

        mrkvec = np.array(1.0-rkvec,dtype=float)

        for ii in range(mrkvec.shape[0]):
            for jj in range(mrkvec.shape[1]):
                if mrkvec[ii,jj] == 1.0:
                    mrkvec[ii,jj] = 0.0
        
        mappingidx = []

        for ii in range(rkvec.shape[0]):
            for jj in range(mrkvec.shape[0]):
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

        GmRmT = np.zeros((norb,norb,ns,nr,ntau),dtype=np.complex64,order='F')

        for itau in range(ntau):
            for rp in self.mappingidx:
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            GmRmT[iorb,jorb,js,rp[0],itau] = -G[iorb,jorb,js,rp[1],ntau-itau-1]

        return GmRmT
    
    def Rvec(self):
        
        r = np.zeros((self.rkgrid[0]*self.rkgrid[1]*self.rkgrid[2],3),dtype=float)
        for iz in range(self.rkgrid[2]//2 +1):
            for iy in range(self.rkgrid[1]//2 + 1):
                for ix in range(self.rkgrid[0]//2+1):
                    nn1 = [ix,iy,iz]
                    ind1,nn1 = self.indexing(self.rkgrid[0]*self.rkgrid[1]*self.rkgrid[2],3,self.rkgrid,1,0,nn1)
                    r[ind1] = nn1
                    if (nn1==[0,0,0]):
                        continue
                    ii = (self.rkgrid[0]-ix) % self.rkgrid[0]
                    jj = (self.rkgrid[1]-iy) % self.rkgrid[1]
                    kk = (self.rkgrid[2]-iz) % self.rkgrid[2]
                    nn2 = [ii,jj,kk]
                    ind2,nn2 = self.indexing(self.rkgrid[0]*self.rkgrid[1]*self.rkgrid[2],3,self.rkgrid,1,0,nn2)
                    r[ind2] = [-ix,-iy,-iz]

        self.rvec = r

        return None




class FT_grid(object):

    def __init__(self,T : float = 300,beta : float = None,size : int = 1000) -> object:
        
        if beta == None:
            self.T = T
            self.beta = 1/(T*8.6173303*10**-5)
        else:
            self.beta = beta
            self.T = 1/(beta*8.6173303*10**-5)
        self.size = size
        self.omega = np.zeros((size),dtype=float,order='F')
        self.nu = np.zeros((size),dtype=float,order='F')
        self.tau = np.zeros((int(size)),dtype=float,order='F')

        self.Omega()
        self.Tau()
        self.Nu()

    def Omega(self) -> np.ndarray:

        nomega = int(self.size)#self.size
        for iomega in range(nomega):
            self.omega[iomega] = np.pi/self.beta*(2*iomega+1)

    def Tau(self) -> np.ndarray:

        ntau = int(self.size)
        
        for itau in range(int(ntau)):
            itheta = DiagE.common.ttind(itau,ntau)
            self.tau[itau] = self.beta/2.0*(np.cos(np.pi*(itheta+0.5)/ntau)+1.0)
        

    def Nu(self) -> np.ndarray:

        nnu = self.size
        for inu in range(nnu):
            self.nu[inu] = np.pi/self.beta*(2*inu)

class FLatDyn(object):
    def __init__(self,crystal : Crystal, ft : FT_grid) -> object:
        self.crystal = crystal
        self.ft = ft
        self.mappingidx = None

    def Inverse(self, mat : np.ndarray) -> np.ndarray:

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]
        nft = mat.shape[4]

        matinv = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')

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
        ff = np.zeros((norb,norb,ns,nk,nfreq),dtype=np.complex64,order='F')

        ff = DiagE.fourier.flatdyn_t2f(self.ft.tau,ftau,self.ft.omega)

        return ff
    
    def F2T(self,ff : np.ndarray,isgreen : int, highzero : int) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]
        ntau = len(self.ft.tau)

        ftau = np.zeros((norb,norb,ns,nk,ntau),dtype=np.complex64,order='F')
        tempmat = copy.deepcopy(ff)
        moment, high = self.Moment(tempmat,isgreen,highzero)
        
        ftau = DiagE.fourier.flatdyn_f2t(self.ft.omega,tempmat,moment,self.ft.tau)

        return ftau

    
    def Moment(self,ff : np.ndarray, isgreen : int, highzero : int) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]

        moment = np.zeros((norb,norb,ns,nk,3),dtype=np.complex64,order='F')
        high = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order='F')

        tempmat = copy.deepcopy(ff)

        moment, high = DiagE.fourier.flatdyn_m(self.ft.omega,tempmat,isgreen,highzero)

        return moment, high
    
    
    def K2R(self,matk : np.ndarray, rkgrid : list = None) -> np.ndarray:

        rkvec = self.crystal.kpoint
        if rkgrid == None:
            rkgrid = self.crystal.rkgrid
        
        
        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[3]
        nft = matk.shape[4]
        matr = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')
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

        matr = DiagE.fourier.flatdyn_k2r(rkgrid,tempmat)
        
        return matr
    
    def R2K(self, matr : np.ndarray) -> np.ndarray:

        rkvec = self.crystal.kpoint
        rkgrid = self.crystal.rkgrid

        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[3]
        nft = matr.shape[4]

        matk = DiagE.fourier.flatdyn_r2k(rkgrid,matr)

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

        GmRmT = np.zeros((norb,norb,ns,nr,ntau),dtype=np.complex64,order='F')

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

        ynew = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')

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

        Fnew = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')

        if iter == 1:
            mix = 1.0
            Fm = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')
        
        Fnew = mix*Fb + (1.0-mix)*Fm

        return Fnew
    
    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = mat1.shape[0]
        ns = mat1.shape[2]
        nrk = mat1.shape[3]
        nft = mat1.shape[4]
        
        matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')

        matout = DiagE.dyson.flatdyn(mat1,mat2)

        return matout

    def Projection(self, matin : np.ndarray):

        
        ns = matin.shape[2]
        nft = matin.shape[4]
        norbc = self.crystal.fprojector.shape[1]
        nspace = self.crystal.fprojector.shape[3]

        matout = np.zeros((norbc,norbc,ns,nft,nspace),dtype=np.complex64,order='F')

        for ispace in range(nspace):
            matout[...,ispace] = DiagE.projection.flatdyn(matin,self.crystal.fprojector[...,ispace])

        return matout
    
    def ChemEmbedding(self,mu : float) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        chem = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')

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
        nft = self.ft.size

        matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')

        for ift in range(nft):
            matout[...,ift] = matin

        return matout
    def Save(self, matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[3]
        nft = matin.shape[4]

        # if os.path.exists('flatdyn'):
        #     pass
        # else:
        #     os.mkdir("flatdyn")
        # os.chdir("flatdyn")
        with open(fn+'.txt','w') as f:
            f.write("#iorb, jorb, is, ik, ift, Re(F(k,w)), Im(F(k,w))\n")
            for ift in range(nft):
                for irk in range(nrk):
                    for js in range(ns):
                        for jorb in range(norb):
                            for iorb in range(norb):
                                f.write(f"{iorb} {jorb} {js} {irk} {ift} {matin[iorb,jorb,js,irk,ift].real} {matin[iorb,jorb,js,irk,ift].imag}\n")
        # os.chdir("..")
        return None
    
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

    # def KArb(self, matr : np.ndarray = None, kpoint : np.ndarray = None):

    #     norb = matr.shape[0]
    #     ns = matr.shape[2]
    #     nr = matr.shape[3]
    #     nfreq = matr.shape[4]
    #     nk = len(kpoint)

    #     tempmat = np.zeros((norb,norb,ns,nr,nfreq),dtype=complex,order='F')
    #     matkinv = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,order='F')

    #     matrinv = self.Inverse(matr)
    #     omega = self.ft.omega

    #     for ifreq in range(nfreq):
    #         for ir in range(nr):
    #             for js in range(ns):
    #                 for jorb in range(norb):
    #                     for iorb in range(norb):
    #                         if iorb==jorb:
    #                             tempmat[iorb,jorb,js,ir,ifreq] = 1j*omega[ifreq]-matrinv[iorb,jorb,js,ir,ifreq]
    #                         else:
    #                             tempmat[iorb,jorb,js,ir,ifreq] = -matrinv[iorb,jorb,js,ir,ifreq]

    #     tempmat2 = self.R2KArb(tempmat,kpoint)

    #     for ifreq in range(nfreq):
    #         for ik in range(nk):
    #             for js in range(ns):
    #                 for jorb in range(norb):
    #                     for iorb in range(norb):
    #                         if iorb==jorb:
    #                             matkinv[iorb,jorb,js,ik,ifreq] = 1j*omega[ifreq]-tempmat2[iorb,jorb,js,ik,ifreq]
    #                         else:
    #                             matkinv[iorb,jorb,js,ik,ifreq] = -tempmat2[iorb,jorb,js,ik,ifreq]
        
    #     matk = self.Inverse(matkinv)

    #     return matk
        

    
class GreenBare(FLatDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid, hamtb = None) -> object:
        super(GreenBare,self).__init__(crystal, ft)
        # print(self.niham.hamtb[...,0,0])
        self.hamtb = hamtb
        self.g0kt = None
        self.g0kf = None
        self.g0rt = None
        self.g0rf = None

        self.Cal()
        

    def Cal(self): # freq, tau combine
        
        
        print(self.hamtb[:,:,0,0])
        gnotkf = DiagE.bare.flatfreq(self.hamtb,self.ft.omega)
        gnotrf = self.K2R(gnotkf)#######
        
        self.g0kf = gnotkf
        self.g0rf = gnotrf

        gnotkt = DiagE.bare.flattau(self.hamtb,self.ft.tau)
        gnotrt = self.K2R(gnotkt)

        self.g0kt = gnotkt
        self.g0rt = gnotrt

        return None
    
    def Save(self, fn: str):

        if os.path.exists('gbare'):
            pass
        else:
            os.mkdir('gbare')

        os.chdir('gbare')
        super().Save(self.g0kf, fn)
        os.chdir('..')
        return None
    
class GreenInt(FLatDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid, greenbare : np.ndarray = None, sigmah : np.ndarray = None, sigmaf : np.ndarray = None, sigmagwc : np.ndarray = None) -> object:
        
        if greenbare is None:
            print("Bare Green's function doesn't exist")
            sys.exit()
        super(GreenInt,self).__init__(crystal, ft)
        self.flatstc = FLatStc(crystal=crystal)
        self.gkf = None
        self.gkt = None
        self.grf = None
        self.grt = None
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
        print(f"Bare Green's function : \n{self.gbare[:,:,0,0,0]}")
        self.CalMu0()
        self.SearchMu()

    def CalMu0(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size
        sigma = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')
        print("Initialization start")
        if (self.sigmah is None)and(self.sigmaf is None)and(self.sigmac is None):
            self.gkfmu0 = self.gbare
        else:
            if (self.sigmah is not None):
                print(sigma[:,:,0,0,0])
                sigma += self.StcEmbedding(self.sigmah)
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
        # if (self.sigmah!=None)and(self.sigmaf!=None)and(self.sigmac==None):
        #     tempmat = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')
        #     tempmat2 = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')
        #     tempmat = self.StcEmbedding(self.sigmah.hk)
        #     tempmat2 = self.StcEmbedding(self.sigmaf.fk)
        #     sigma = tempmat+tempmat2
        #     self.gkfmu0 = self.Dyson(self.gbare.g0kf,sigma)
            
        # if (self.sigmah!=None)and(self.sigmaf!=None)and(self.sigmac!=None):
        #     tempmat = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')
        #     tempmat2 = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')
        #     tempmat = self.StcEmbedding(self.sigmah.hk)
        #     tempmat2 = self.StcEmbedding(self.sigmaf.fk)
        #     sigma = tempmat+tempmat2+self.sigmac.kf
        #     self.gkfmu0 = self.Dyson(self.gbare.g0kf,sigma)
        # if (self.sigmah==None)and(self.sigmaf==None)and(self.sigmac==None):
        #     self.gkfmu0 = self.gbare.g0kf
        #     self.gktmu0 = self.gbare.g0kt
        # else:
        #     if self.sigmah != None:
        #         hk = self.StcEmbedding(self.sigmah.hk)
        #         sig += hk
            
        #     if self.sigmaf != None:
        #         fk = self.StcEmbedding(self.sigmaf.fk)
        #         sig += fk
            
        #     if self.sigmac != None:
        #         sig += self.sigmac.kf

        # self.gkfmu0 = self.Dyson(self.gbare.g0kf,sig)

        self.gktmu0 = self.F2T(self.gkfmu0,1,1)
        self.grfmu0 = self.K2R(self.gkfmu0)
        self.grtmu0 = self.K2R(self.gktmu0)
        print("Initialization finish")
        return None
    
    def Occ(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        
        
        occk = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        occ = np.zeros((norb,norb,ns),dtype=np.complex64,order='F')
        
        print("Density matrixy calculation start")
        
        occk = -self.gkt[...,-1]
    
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
        nft = self.ft.size

        gkfnew = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')
        chem = self.ChemEmbedding(self.mu)
    

        gkfnew = self.Dyson(self.gkfmu0,-chem)
        
        self.gkf = gkfnew
        self.gkt = self.F2T(gkfnew,1,1)
        # self.grf = self.K2R(self.Dyson(self.gkfmu0,-chem))
        # self.grt = self.K2R(self.F2T(self.Dyson(self.gkfmu0,-chem),1,1))
        self.grf = self.K2R(self.gkf)
        self.grt = self.K2R(self.gkt)
        print("Chemical potential shift finish")
        self.Occ()

        return None
    
    def NumOfE(self, mu : float):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size
        tempmat = copy.deepcopy(self.gkfmu0)
        chem = self.ChemEmbedding(mu)
        gcalf = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')
        gcalt = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')

        
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
        mumin = -self.ft.omega[-1]*0.4
        mumax = self.ft.omega[-1]*0.4
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
    
    def Save(self, fn: str):

        if os.path.exists('green'):
            pass
        else:
            os.mkdir('green')

        os.chdir('green')
        super().Save(self.gkf, fn)
        os.chdir('..')

        return None

    
class SigmaGWC(FLatDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid,green : np.ndarray = None, wlat : np.ndarray = None) -> object:
        super().__init__(crystal, ft)
        self.flatstc = FLatStc(crystal=crystal)
        self.rt = None
        self.rf = None
        self.kt = None
        self.kf = None
        self.stck = None
        self.z = None

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

        crtau = np.zeros((norbc,norbc,ns,nr,ntau),dtype=np.complex64,order='F')
    
        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex64,order='F')
        for itau in range(ntau):
            for ir in range(nr):
                tempmat = self.crystal.OrbSpin2Composite(Wc[:,:,:,:,ir,itau])
                for ind1 in range(norb*ns):
                    nn1= [0]*2
                    ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                    [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                    iorbc1 = self.crystal.FIndex([a,m1])
                    iorbc4 = self.crystal.FIndex([a,m4])
                    for ind2 in range(norb*ns):
                        nn2 = [0]*2
                        ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
                        [b,[m3,m2]] = self.crystal.BAtomOrb(jorb)
                        iorbc3 = self.crystal.FIndex([b,m3])
                        iorbc2 = self.crystal.FIndex([b,m2])
                        if js == ks:
                            crtau[iorbc1,iorbc2,js,ir,itau] += -G[iorbc4,iorbc3,js,ir,itau]*tempmat[ind1,ind2]
                
                                        

        cktau = self.R2K(crtau)
        crfreq = self.T2F(crtau)
        ckfreq = self.K2R(crfreq)

        self.rt = crtau
        self.kt = cktau
        self.rf = crfreq
        self.kf = ckfreq

        return None
    
    def SigmaStc(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nfreq = self.ft.size

        sigmastc = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order="F")
        tempmat = np.zeros((norb,norb,ns,nk,nfreq),dtype=np.complex64,order="F")

        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    tempmat[:,:,js,ik,ifreq] = np.transpose(np.conjugate(self.kf[:,:,js,ik,ifreq]))

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        sigmastc[iorb,jorb,js,ik] = (self.kf[iorb,jorb,js,ik,0]+tempmat[iorb,jorb,js,ik,0])/2

        self.stck = sigmastc

        return None
    
    def Zfactor(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nfreq = self.ft.size
        beta = self.ft.beta

        z = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order='F')
        identity = np.zeros((norb,norb,ns,nk,nfreq),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb,norb,ns,nk,nfreq),dtype=np.complex64,order='F')
        tempmat2 = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order='F')

        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    identity[:,:,js,ik,ifreq] = np.eye(norb,norb,dtype=np.complex64,order='F')
                    # tempmat[:,:,js,ik,ifreq] = np.transpose(np.conjugate(self.kf[:,:,js,ik,ifreq]))
                    tempmat[:,:,js,ik,ifreq] = np.linalg.inv(self.kf[:,:,js,ik,ifreq])

        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            tempmat2[iorb,jorb,js,ik] = (identity[iorb,jorb,js,ik,ifreq]-beta*(self.kf[iorb,jorb,js,ik,ifreq]-tempmat[iorb,jorb,js,ik,ifreq])/(2*np.pi))
        
        z = self.flatstc.Inverse(tempmat2)

        self.z = z

        return None
    
    def Save(self,fn):

        if os.path.exists('sigmac'):
            pass
        else:
            os.mkdir('sigmac')

        os.chdir('sigmac')
        super().Save(self.kf, fn)
        os.chdir('..')

        return None


class Spectral(FLatDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid, green : np.ndarray) -> object:
        super().__init__(crystal, ft)
        self.green = green
        self.rf = None
        self.kf = None

        self.Cal()

    def Cal(self):
        
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nomega = len(self.ft.omega)

        akf = np.zeros((norb,norb,ns,nk,nomega),dtype=complex,order='F')

        akf = -1/np.pi*self.green.imag

        arf = self.K2R(akf)

        self.kf = akf
        self.rf = arf

        return None


class FLatStc(object):

    def __init__(self,crystal : Crystal):

        self.crystal = crystal

    def Inverse(self,mat : np.ndarray):

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]

        matinv = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')

        for irk in range(nrk):
            for js in range(ns):
                matinv[:,:,js,irk] = np.linalg.inv(mat[:,:,js,irk])
        
        return matinv
    
    def K2R(self, matk : np.ndarray = None, rkgrid : list = None)->np.ndarray:

        if rkgrid == None:
            rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[3]

        tempmat = copy.deepcopy(matk)
        
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        [a,m1] = self.crystal.FAtomOrb(iorb)
                        [b,m2] = self.crystal.FAtomOrb(jorb)

                        delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]

                        phase = np.exp(2.0j*np.pi*np.dot(rkvec[irk],delta))
                        
                        # matk[iorb,jorb,js,irk] *= phase
                        tempmat[iorb,jorb,js,irk] *= phase
                        
        
        
        matr = DiagE.fourier.flatstc_k2r(rkgrid,tempmat)

        return matr
    
    def R2K(self,matr : np.ndarray  = None, rkgrid : list = None)->np.ndarray:

        if rkgrid == None:
            rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[3]

        matk = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        matk = DiagE.fourier.flatstc_r2k(rkgrid,matr)
        
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        [a,m1] = self.crystal.FAtomOrb(iorb)
                        [b,m2] = self.crystal.FAtomOrb(jorb)

                        delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
                        phase = np.exp(-2.0j*np.pi*np.dot(rkvec[irk],delta))

                        matk[iorb,jorb,js,irk] = matk[iorb,jorb,js,irk] * phase
                        
        
        return matk
    
    def Band(self, energy : np.ndarray, fn : str = None, plotoption : bool = False, label : list = None):

        norb = energy.shape[0]
        ns = energy.shape[2]
        nk = energy.shape[3]

        energyplot = np.zeros((norb,ns,nk),dtype=float)

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    energyplot[iorb,js,ik] = energy[iorb,iorb,js,ik]
        if plotoption:
            if self.crystal.ns == 1:
                fig,ax = plt.subplots()
                ax.set_xlim(self.crystal.knode[0],self.crystal.knode[-1])
                ax.set_xticks(self.crystal.knode)
                if label == None:
                    pass
                else:
                    ax.set_xticklabels(label)
                for i in range(len(self.crystal.knode)):
                    ax.axvline(x=self.crystal.knode[i],linewidth=0.5,color='r',linestyle='--')
                for iorb in range(norb):
                    ax.plot(self.crystal.kdist,energyplot[iorb,0,:].T,'k-')
                ax.set_ylabel('E (eV)')
                ax.set_title('Band')
                # plt.plot(energyplot.T[:,0,:])
                if fn == None:
                    plt.show()
                else:
                    plt.savefig(fn)
            else:
                up = energyplot[:,0,:]
                down = energyplot[:,1,:]
                plt.plot(up,'k-')
                plt.plot(down,'r-')
                if fn == None:
                    plt.show()
                else:
                    plt.savefig(fn)
        else:
            with open('band.dat','w') as f:
                for js in range(ns):
                    for ik in range(nk):
                        linedata = [self.crystal.kdist[ik]]+energyplot[:,js,ik].tolist()
                        line = ' '.join(map(str,linedata))
                        f.write(line+'\n')

        
        return None
    
    def Diagonalize(self,matk : np.ndarray, eigvec : bool = False):
        
        nk = matk.shape[3]
        norb = matk.shape[0]
        ns = matk.shape[2]
        
        energy = np.zeros((norb,norb,ns,nk),dtype=float)
        evec = np.zeros((norb,norb,ns,nk),dtype=np.complex64)

        # if eigvec == False:
        #     for ik in range(nk):
        #         for js in range(ns):
        #             e = np.linalg.eigvalsh(matk[:,:,js,ik])
        #             energy[:,:,js,ik] = np.diag(e)
        #     return energy
        # else:
        #     for ik in range(nk):
        #         for js in range(ns):
        #             (e,v) = np.linalg.eigh(matk[:,:,js,ik])
        #             energy[:,:,js,ik] = np.diag(e)
        #             evec[:,:,js,ik] = v

        #     return energy, evec
        if eigvec == False:
            for ik in range(nk):
                for js in range(ns):
                    e,v,info = scipy.linalg.lapack.zheev(matk[:,:,js,ik])
                    energy[:,:,js,ik] = np.diag(e)
            return energy
        else:
            for ik in range(nk):
                for js in range(ns):
                    e,v,info = scipy.linalg.lapack.zheev(matk[:,:,js,ik])
                    energy[:,:,js,ik] = np.diag(e)
                    evec[:,:,js,ik] = v

            return energy, evec
    
    def Gaussian(self, x, mu, sigma = 0.1):

        return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        
    def DOS(self,hamr : np.ndarray = None, sigma : float = 0.1, kgrid : list = [20,20,20], plotoption : bool = False):


        print("***** DOS Calculation Start *****")
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        if type(kgrid)==list:
            nk = kgrid[0]*kgrid[1]*kgrid[2]
            kpointtemp = np.array(list(itertools.product(np.linspace(0,1,num=kgrid[2],endpoint=False),np.linspace(0,1,num=kgrid[1],endpoint=False),np.linspace(0,1,num=kgrid[0],endpoint=False))))
            kpoint = np.fliplr(kpointtemp)
        elif type(kgrid)==np.ndarray:
            nk = len(kgrid)
            kpoint = kgrid

        print("***** Fourier transfrom R2K Start")
        hamk = self.R2KArb(hamr,kpoint)
        print("***** Fourier transfrom R2K Finish")
        print("***** Hamiltonian Diagonalization Start *****")
        (energy,eigvec) = self.Diagonalize(matk=hamk,eigvec=True)
        print("***** Hamiltonian Diagonalization Finish *****")
        emin = -10#energy[0,0,0].min()
        emax = 10#energy[-1,-1,0].max()
        energyrange=np.linspace(emin,emax,nk)
        # dos = np.zeros_like(energyrange)
        dos = np.zeros((norb,ns,nk),dtype=float)
        tempmat = np.zeros((norb,ns,nk),dtype=float)

        print("***** Gaussian Approach Start *****")
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    e = energy[iorb,iorb,js,ik]
                    tempmat[iorb,js] += self.Gaussian(energyrange,e,sigma)/nk
        print("***** Gaussian Approach Finish *****")
        
        for ik in range(nk):
            for js in range(ns):
                tempmat2 = np.linalg.inv(eigvec[:,:,js,ik])
                # tempmat3 = np.array(np.dot(tempmat2,eigvec[:,:,js,ik]),dtype=float)
                D = np.diag(tempmat[:,js,ik])
                tempmat3 = eigvec[:,:,js,ik]@(D@tempmat2)
                for iorb in range(norb):
                    dos[iorb,js,ik] = tempmat3[iorb,iorb]
                # for jorb in range(norb):
                #     for iorb in range(norb):
                #         # dos[iorb,js,ik] = tempmat2[iorb,jorb]*tempmat[jorb,js,ik]*eigvec[jorb,iorb,js,ik]
                #         # dos[iorb,js,ik] = tempmat3[iorb,jorb]*tempmat[jorb,js,ik]
                #         dos[iorb,js,ik] = eigvec[jorb,iorb,js,ik]*tempmat[jorb,js,ik]*tempmat2[jorb,iorb]
        

        print(f"Integration gaussian : {np.trapz(self.Gaussian(energyrange,0),energyrange)}")
        temp = 0
        for js in range(ns):
            for iorb in range(norb):
                temp+= np.trapz(dos[iorb,js],energyrange)

        
        print(f'Integration dos : {temp}')
        if plotoption:
            fig, ax = plt.subplots()
            ax.set_xlim(energyrange[0],energyrange[-1])
            legend = []
            for js in range(ns):
                for iorb in range(norb):
                    ax.plot(energyrange,dos[iorb,js])
                    legend.append(iorb+1)
            ax.legend(legend)
            ax.set_xlabel('E (eV)')
            ax.set_ylabel('DOS')
            plt.show()
        else:
            with open('dos.dat','w') as f:
                for i in range(len(energyrange)):
                    f.write(f'{energyrange[i]}  {dos[i]}')
        print("***** DOS Calculation Finish *****")
        return None
        
        

    def Visualization(self, energy : np.ndarray, fn : str = None):

        if self.crystal.rkgrid[2] != 1:
            print("Energy surface for only 2D case")
            sys.exit()
        else:
            norb = energy.shape[0]
            ns = energy.shape[2]
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            kx = self.crystal.kpoint[:,0].reshape(self.crystal.rkgrid[0],self.crystal.rkgrid[1],self.crystal.rkgrid[2])
            ky = self.crystal.kpoint[:,1].reshape(self.crystal.rkgrid[0],self.crystal.rkgrid[1],self.crystal.rkgrid[2])
            energy = energy.T
            energy = energy.reshape(self.crystal.rkgrid[0],self.crystal.rkgrid[1],self.crystal.rkgrid[2],ns,norb,norb)

            for js in range(ns):
                for iorb in range(norb):
                    ax.plot_surface(kx[:,:,0],ky[:,:,0],energy[:,:,0,js,iorb,iorb])

            ax.view_init(azim=-120,elev=0)
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_zlabel('Energy eV')
            if fn is None:
                plt.show()
            elif fn is not None:
                fig.savefig(fn)
        
        return None

    def Mixing(self, iter : int, mix : float, Fb : np.ndarray, Fm : np.ndarray) -> np.ndarray:

        #norb = Fb.shape[0]
        #ns = Fb.shape[2]
        #nrk = Fb.shape[3]
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)


        Fnew = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        # print(Fnew.shape)
        if iter == 1:
            mix = 1.0
            Fm = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        Fnew[iorb,jorb,js,irk] = mix*Fb[iorb,jorb,js,irk] + (1.0-mix)*Fm[iorb,jorb,js,irk]

        return Fnew
    
    def ChemEmbedding(self,mu : float) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        chem = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')

        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    chem[iorb,iorb,js,irk] = mu

        return chem

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')

        matout = DiagE.dyson.flatstc(mat1,mat2)

        return matout

    def Projection(self, matin : np.ndarray):

        norb = len(self.crystal.fin)
        ns = self.crystal.ns
        norbc = self.crystal.fprojector.shape[1]
        nspace = self.crystal.fprojector.shape[3]

        matout = np.zeros((norbc,norbc,ns,nspace),dtype=np.complex64,order='F')

        for ispace in range(nspace):
            matout[...,ispace] = DiagE.projection.flatstc(matin,self.crystal.fprojector[...,ispace])

        return matout
    
    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[3]

        # if os.path.exists('flatstc'):
        #     pass
        # else:
        #     os.mkdir("flatstc")
        # os.chdir("flatstc")
        with open(fn+'.txt','w') as f:
            f.write("#iorb, jorb, is, ik, Re(F(k)), Im(F(k))\n")
            for irk in range(nrk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            f.write(f"{iorb} {jorb} {js} {irk} {matin[iorb,jorb,js,irk].real} {matin[iorb,jorb,js,irk].imag}\n")
        # os.chdir("..")
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
        matk = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        temp = 0
                        for ir in range(nr):
                            temp += tempmat[iorb,jorb,js,ir]*np.exp(-2.0j*np.pi*(kpoint[ik]@self.crystal.rvec[ir]))
                        [a,m1] = self.crystal.FAtomOrb(iorb)
                        [b,m2] = self.crystal.FAtomOrb(jorb)
                        delta = self.crystal.basisf[a,:]-self.crystal.basisf[b,:]
                        phase = np.exp(-2.0j*np.pi*(kpoint[ik]@delta))
                        matk[iorb,jorb,js,ik] = temp*phase
        
        return matk

    def HermitianCheck(self, matin : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]


        errmessage = 'The matrix is not hermitian. Check the input file again'
        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        err = matin[iorb,jorb,js,ik]-np.conjugate(matin[jorb,iorb,js,ik])
                        if abs(err)>1.0e-6:
                            print(errmessage)
                            sys.exit()
        return None
    
class Hamiltonian(FLatStc):

    def __init__(self, crystal: Crystal, ham : np.ndarray, beta : float = None, sigmah = None, sigmaf = None, sigmac = None):
        super().__init__(crystal)

        self.occ = None
        self.occk = None
        self.occr = None
        self.ham = ham
        self.sigmah = sigmah
        self.sigmaf = sigmaf
        self.sigmac = sigmac
        self.beta = beta
        self.hk = None
        self.hkmu0 = None
        self.mu = 0
        # self.muold = mu
        self.CalMu0()
        self.SearchMu()

    def CalMu0(self) -> np.ndarray:
        
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        tempmat = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        
        tempmat = copy.deepcopy(self.ham)
        

        if (self.sigmah != None):
            tempmat += self.sigmah.hk
        
        if (self.sigmaf != None):
            tempmat += self.sigmaf.fk
        
        if (self.sigmac != None):
            z = self.sigmac.z
            sigma = self.sigmac.stck
            # chem = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order='F')
            tempmat2 = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
            tempmat3 = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
            tempmat4 = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
            tempmat4 = copy.deepcopy(tempmat)
            eigval, eigvec = self.Diagonalize(z,True)
            for ik in range(nrk):
                for js in range(ns):
                    for iorb in range(norb):
                        # chem[iorb,iorb,js,ik] = -self.mu
                        if 0<=(eigval[iorb,iorb,js,ik])<=1:
                            continue
                        else:
                            print("Error : The z-factor was calculated incorrectly. Please rerun the code.")
                            print(eigval[iorb,iorb,js,ik])
                            sys.exit()
                    tempmat2[:,:,js,ik] = np.dot(np.dot(eigvec[:,:,js,ik],np.sqrt(eigval[:,:,js,ik])),np.linalg.inv(eigvec[:,:,js,ik]))
            
            tempmat4 = tempmat4 + sigma

            for ik in range(nrk):
                for js in range(ns):
                    tempmat3[:,:,js,ik] = np.dot(np.dot(tempmat2[:,:,js,ik],tempmat4[:,:,js,ik]),tempmat2[:,:,js,ik])

            tempmat = copy.deepcopy(tempmat3)
            del tempmat2, tempmat3, tempmat4

        self.hkmu0 = copy.deepcopy(tempmat)
        del tempmat
        return None

    def NumOfE(self,  mu : float) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        

        energy = self.Diagonalize(self.hkmu0)

        Ne = 0

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    Ne += 1/(1+np.exp((energy[iorb,iorb,js,ik]-mu)*self.beta))

        Ne /= nk        
        N = self.crystal.nume
        

        return N - Ne
    
    def SearchMu(self):
        
        energy = self.Diagonalize(self.hkmu0)
        norb = energy.shape[0]
        mumin = energy[0,0].min()-1000
        mumax = energy[norb-1,norb-1].max()+1000

        nmin = self.NumOfE(mumin)
        nmax = self.NumOfE(mumax)
        if (nmin < 0) or (nmax>0):
            print("Chemical potential is out of the bisection range")
            sys.exit()
        sol = scipy.optimize.brentq(self.NumOfE,mumin,mumax)
        # try:
        #     sol = scipy.optimize.brentq(self.NumOfE,mumin,mumax)
        # except:
        #     sol = scipy.optimize.newton(self.NumOfE,0,tol=10**(-10))
        self.mu = sol
        
        self.UpdateMu()
        return None


    def Occ(self) -> np.ndarray:
        
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        
        # energy = self.Diagonalize(self.hk)

        occk = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        occ = np.zeros((norb,norb,ns),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb,norb),dtype=float,order='F')

        energy, eigvec = self.Diagonalize(self.hk,True)
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    tempmat[iorb,iorb] = 1/(np.exp(energy[iorb,iorb,js,irk]*self.beta)+1)
                # occk[:,:,js,irk] = np.dot(eigvec[:,:,js,irk],np.dot(tempmat,np.linalg.inv(eigvec[:,:,js,irk])))
                occk[:,:,js,irk] = np.dot(eigvec[:,:,js,irk],np.dot(tempmat,scipy.linalg.inv(eigvec[:,:,js,irk])))
                
            occ += occk[...,irk]
        
        occ /= nrk
        
        self.occ = occ
        self.occk = occk
        self.occr = self.K2R(occk)

        return None
    
    def UpdateMu(self) -> np.ndarray:

        chem = self.ChemEmbedding(self.mu)

        ham = self.hkmu0 - chem

        self.hk = ham

        self.Occ()

        return None
    
    def Save(self, fn: str):
        if os.path.exists('ham'):
            pass
        else:
            os.mkdir('ham')
        os.chdir('ham')
        super().Save(self.hk, fn)
        os.chdir('..')
        return None
        
class NIHamiltonian(FLatStc):

    def __init__(self, crystal: Crystal = None,hoppinglist : list=None, onsitelist : list=None):
        super().__init__(crystal)
        self.hoppinglist = hoppinglist
        self.onsitelist = onsitelist
        print(self.onsitelist)
        self.hamtb = None
        self.hamtbr = None
        # self.Hopping()
        # self.Onsite()

        self.Cal()

    def Cal(self): #GenHam
        
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        kvec = self.crystal.kpoint

        hamtb = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb,norb,ns,self.crystal.rkgrid[0],self.crystal.rkgrid[1],self.crystal.rkgrid[2]),dtype=np.complex64,order='F')
        
        
        for js in range(ns):
            for hopp in self.hoppinglist:
                tij = hopp[0]
                # iorb = hopp[1]
                # jorb = hopp[2]
                (a,m) = hopp[1]
                (b,mp) = hopp[2]
                iorb = self.crystal.FIndex([a,m])
                jorb = self.crystal.FIndex([b,mp])
                R = hopp[3]
                
                # tempmat[iorb,jorb,js,R[0],R[1],R[2]] += -tij
                if (iorb==jorb)and(R==[0,0,0]):
                    print("Wrong value entered, please check the input.in file")
                    sys.exit()
                else:
                    tempmat[iorb,jorb,js,R[0],R[1],R[2]] += -tij
                    tempmat[jorb,iorb,js,-R[0],-R[1],-R[2]] += -tij.conjugate()

                # 0 == -0

        if self.onsitelist != None:
            for js in range(ns):
                for iorb in range(norb):
                    tempmat[iorb,iorb,js,0,0,0] = +self.onsitelist[iorb]
        # Hermitian check
        tempmat = tempmat.reshape((norb,norb,ns,nk),order='F')
        self.hamtbr = tempmat
        hamtb = self.R2K(tempmat)
        self.HermitianCheck(hamtb)

        self.hamtb = hamtb

        return None
    
    def Save(self, fn: str):
        
        if os.path.exists('niham'):
            pass
        else:
            os.mkdir('niham')
        os.chdir('niham')
        super().Save(self.hamtb, fn)
        os.chdir('..')
        return None

    # def Hopping(self):
    #     pass
    
    # def Onsite(self):
    #     pass

class QPHamiltonian(FLatStc):

    def __init__(self, crystal: Crystal, niham : NIHamiltonian, sigma : object, mu : float): # object input
        super().__init__(crystal)
        self.niham = niham
        self.sigma = sigma
        self.mu = mu
        self.hamqp = None

        self.Cal()
    def Cal(self):
        
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        
        hamtb = self.niham.hamtb
        z = self.sigma.zfactor.z
        sigma = self.sigma.sigmastc.sigmastc
        hamqp = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order='F')
        # chem = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order='F')
        tempmat2 = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order='F')

        eigval, eigvec = self.Diagonalize(z,True)
        chem = self.ChemEmbedding(-self.mu)
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    # chem[iorb,iorb,js,ik] = -self.mu
                    if 0<=(eigval[iorb,iorb,js,ik])<=1:
                        continue
                    else:
                        print("Error : The z-factor was calculated incorrectly. Please rerun the code.")
                        print(eigval[iorb,iorb,js,ik])
                        sys.exit()
                tempmat[:,:,js,ik] = np.dot(np.dot(eigvec[:,:,js,ik],np.sqrt(eigval[:,:,js,ik])),np.linalg.inv(eigvec[:,:,js,ik]))
        
        tempmat2 = hamtb + sigma + chem

        for ik in range(nk):
            for js in range(ns):
                hamqp[:,:,js,ik] = np.dot(np.dot(tempmat[:,:,js,ik],tempmat2[:,:,js,ik]),tempmat[:,:,js,ik])
        
        self.hamqp = hamqp

        return None

class SigmaHartree(FLatStc):

    def __init__(self, crystal: Crystal, occ = None , vbare :np.ndarray = None, onsite : np.ndarray = None): # green -> occ
        super().__init__(crystal)
        self.hr = None
        self.hk = None
        self.hdyn = None
        self.vbare = vbare
        self.onsiter = onsite
        self.occ = occ
        
        self.Cal()
        # self.MakeDyn()
    
    def Cal(self):
        # vbare = self.vbare.k
        occ = self.occ
        # vk = self.vbare.Double2Quad(self.vbare.k)
        norbc = len(self.crystal.find) #occk.shape[0]
        ns = self.crystal.ns#occk.shape[2]
        nk = len(self.crystal.kpoint) #occk.shape[3]
        norb = len(self.crystal.bind) #vbare.shape[0]

        # onsite = self.R2K(self.onsiter)
        h = np.zeros((norbc,norbc,ns,nk),dtype=np.complex64,order='F')

        if self.crystal.ns != 1:
        #     for ik in range(nk):
        #         tempmat[...,ik] = self.crystal.OrbSpin2Composite(vbare[...,ik])
            
            # for ik in range(nk):
            #     for ind1 in range(norb*ns):
            #         nn1 = [0]*2
            #         ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
            #         [iorbc1,iorbc2] = self.crystal.b2f[iorb]
                    
            #         for ind2 in range(norb*ns):
            #             nn2 = [0]*2
            #             ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
            #             [iorbc3,iorbc4] = self.crystal.b2f[jorb]
            #             h[iorbc1,iorbc2,js,ik] += tempmat[ind1,ind2,0]*occ[iorbc4,iorbc3,ks]
                        # for jk in range(nk):
                        #     h[iorbc1,iorbc2,js,ik] += tempmat[ind1,ind2,0]*occ[iorbc4,iorbc3,ks,jk]/nk
            for ik in range(nk):
                for ind1 in range(norb*ns):
                    nn1 = [0]*2
                    ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                    [a,[m1,m2]] = self.crystal.BAtomOrb(iorb)
                    iorbc1 = self.crystal.FIndex([a,m1])
                    iorbc2 = self.crystal.FIndex([a,m2])
                    for ind2 in range(norb*ns):
                        nn2 = [0]*2
                        ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
                        [b,[m3,m4]] = self.crystal.BAtomOrb(jorb)
                        iorbc3 = self.crystal.FIndex([b,m3])
                        iorbc4 = self.crystal.FIndex([b,m4])
                        # h[iorbc1,iorbc2,js,ik] += vk[iorbc1,iorbc3,iorbc4,iorbc2,js,ks,0]*occ[iorbc4,iorbc3,ks]
                        h[iorbc1,iorbc2,js,ik] += self.vbare[iorb,jorb,js,ks,0]*occ[iorbc4,iorbc3,ks]
            
        else:
            if(self.crystal.soc == True):
                C = 1
                # for ik in range(nk):
                #     for iorb in range(norb):
                #         iorbc1,iorbc2 = self.crystal.b2f[iorb]
                #         for jorb in range(norb):
                #             iorbc3, iorbc4 = self.crystal.b2f[jorb]
                #             # gtemp = np.zeros((norbc,norbc,1),dtype=np.complex64)
                #             # for jk in range(nk):
                #             #     gtemp[iorbc4,iorbc3,0] += g0kt[iorbc4,iorbc3,0,0,-1]
                #             h[iorbc1,iorbc2,0,ik] += vbare[iorb,jorb,0,0,0]*occ[iorbc4,iorbc3,0]*C #1/nk*gtemp[iorbc4,iorbc3,0]*C
                for ik in range(nk):
                    for ind1 in range(norb*ns):
                        nn1 = [0]*2
                        ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                        [a,[m1,m2]] = self.crystal.BAtomOrb(iorb)
                        iorbc1 = self.crystal.FIndex([a,m1])
                        iorbc2 = self.crystal.FIndex([a,m2])
                        for ind2 in range(norb*ns):
                            nn2 = [0]*2
                            ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
                            [b,[m3,m4]] = self.crystal.BAtomOrb(jorb)
                            iorbc3 = self.crystal.FIndex([b,m3])
                            iorbc4 = self.crystal.FIndex([b,m4])
                            h[iorbc1,iorbc2,js,ik] += self.vbare[iorb,jorb,js,ks,0]*occ[iorbc4,iorbc3,ks]*C
                            
            else:
                C = 2
                # for ik in range(nk):
                #     for iorb in range(norb):
                #         iorbc1,iorbc2 = self.crystal.b2f[iorb]
                #         for jorb in range(norb):
                #             iorbc3, iorbc4 = self.crystal.b2f[jorb]
                #             h[iorbc1,iorbc2,0,ik] += vbare[iorb,jorb,0,0,0]*occ[iorbc4,iorbc3,0]*C 
                #             # for jk in range(nk):
                #             #     h[iorbc1,iorbc2,0,ik] += vbare[iorb,jorb,0,0,0]*occ[iorbc4,iorbc3,0,jk]/nk*C
                for ik in range(nk):
                    for ind1 in range(norb*ns):
                        nn1 = [0]*2
                        ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                        [a,[m1,m2]] = self.crystal.BAtomOrb(iorb)
                        iorbc1 = self.crystal.FIndex([a,m1])
                        iorbc2 = self.crystal.FIndex([a,m2])
                        for ind2 in range(norb*ns):
                            nn2 = [0]*2
                            ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
                            [b,[m3,m4]] = self.crystal.BAtomOrb(jorb)
                            iorbc3 = self.crystal.FIndex([b,m3])
                            iorbc4 = self.crystal.FIndex([b,m4])
                            # h[iorbc1,iorbc2,js,ik] += vk[iorbc1,iorbc3,iorbc4,iorbc2,js,ks,0]*occ[iorbc4,iorbc3,ks]*C
                            h[iorbc1,iorbc2,js,ik] += self.vbare[iorb,jorb,js,ks,0]*occ[iorbc4,iorbc3,ks]*C

        self.hk = h #+onsite
        self.hr = self.K2R(h)

        return None
    
    def Save(self, fn: str):
        
        if os.path.exists('sigmah'):
            pass
        else:
            os.mkdir('sigmah')
        os.chdir('sigmah')
        super().Save(self.hk, fn)
        os.chdir('..')
        return None
    
    # def MakeDyn(self):

    #     norb = self.green.gkf.shape[0]
    #     ns = self.green.gkf.shape[2]
    #     nrk = self.green.gkf.shape[3]
    #     nft = self.green.gkf.shape[4]

    #     tempmat = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')

    #     for ift in range(nft):
    #         tempmat[...,ift] = self.hk
    #     self.hdyn = tempmat

    #     return 


class SigmaFock(FLatStc):

    def __init__(self, crystal: Crystal,occr = None, vbare : np.ndarray = None): # green -> occ
        super().__init__(crystal)
        self.fr = None
        self.fk = None
        self.fdyn = None
        # self.green = green
        self.occr = occr
        self.vbare = vbare
        
        self.Cal()
        # self.MakeDyn()

    def Cal(self):
        
        # g0rt = self.green.glatrt
        occr = self.occr
        # vr = self.vbare.Double2Quad(self.vbare.r)
        
        norbc = len(self.crystal.find)
        ns = occr.shape[2]
        nr = occr.shape[3]
        norb = len(self.crystal.bind)

        fr = np.zeros((norbc,norbc,ns,nr),dtype=np.complex64,order='F')

        # for ir in range(nr):
        #     for js in range(ns):
        #         for iorb in range(norb):
        #             [iorbc1,iorbc4] = self.crystal.b2f[iorb]
        #             for jorb in range(norb):
        #                 [iorbc2,iorbc3] = self.crystal.b2f[jorb]
        #                 fr[iorbc1,iorbc3,js,ir] = -occr[iorbc4,iorbc2,js,ir]*vr[iorb,jorb,js,js,ir]
        for ir in range(nr):
            for ind1 in range(norb*ns):
                nn1 = [0]*2
                ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                iorbc1 = self.crystal.FIndex([a,m1])
                iorbc4 = self.crystal.FIndex([a,m4])
                for ind2 in range(norb*ns):
                    nn2 = [0]*2
                    ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
                    [b,[m3,m2]] = self.crystal.BAtomOrb(jorb)
                    iorbc3 = self.crystal.FIndex([b,m3])
                    iorbc2 = self.crystal.FIndex([b,m2])
                    if js == ks:
                        # fr[iorbc1,iorbc2,js,ir] += -occr[iorbc4,iorbc3,js,ir]*vr[iorbc1,iorbc3,iorbc2,iorbc4,js,ks,ir]
                        fr[iorbc1,iorbc2,js,ir] += -occr[iorbc4,iorbc3,js,ir]*self.vbare[iorb,jorb,js,ks,ir]
                        
                        # fr[iorbc1,iorbc2,js,ir] += -occr[iorbc3,iorbc4,js,ir]*vr[iorbc1,iorbc3,iorbc2,iorbc4,js,ks,ir]

        fk = self.R2K(fr)

        self.fr = fr
        self.fk = fk
        del fr, occr
        return None
    
    def Save(self, fn: str):
        
        if os.path.exists('sigmaf'):
            pass
        else:
            os.mkdir('sigmaf')
        os.chdir('sigmaf')

        super().Save(self.fk, fn)
        os.chdir('..')

        return None


class FLocDyn(object):

    def __init__(self,crystal : Crystal, ft : FT_grid):
        
        self.crystal = crystal
        self.ft = ft

    def Inverse(self, mat : np.ndarray):
        
        norb = mat.shape[0]
        ns = mat.shape[2]
        nft = mat.shape[3]

        matinv = np.zeros((norb,norb,ns,nft),dtype=np.complex64,order='F')

        for ift in range(nft):
            for js in range(ns):
                matinv[:,:,js,ift] = np.linalg.inv(mat[:,:,js,ift])

        return matinv
    
    def Moment(self, ff : np.ndarray, isgreen : int, highzero : int) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]

        moment = np.zeros((norb,norb,ns,3),dtype=np.complex64,order='F')
        high = np.zeros((norb,norb,ns),dtype=np.complex64,order='F')

        moment, high = DiagE.fourier.flocdyn_m(self.ft.omega,ff,isgreen,highzero)

        return moment, high
    
    def F2T(self, ff : np.ndarray, isgreen : int, highzero : int) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        ntau = len(self.ft.tau)

        ftau = np.zeros((norb,norb,ns,ntau),dtype=np.complex64,order='F')

        moment, high = self.Moment(ff,isgreen,highzero)

        ftau = DiagE.fourier.flocdyn_f2t(self.ft.omega,ff,moment,self.ft.tau)

        return ftau
        
    def T2F(self,ftau : np.ndarray) -> np.ndarray:

        norb = ftau.shape[0]
        ns = ftau.shape[2]
        nfreq = len(self.ft.omega)

        ff = np.zeros((norb,norb,ns,nfreq),dtype=np.complex64,order='F')

        ff = DiagE.fourier.flocdyn_t2f(self.ft.tau,ftau,self.ft.omega)

        return ff
    
    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]

        ynew = np.zeros((norb,norb,ns,nft),dtype=np.complex64,order='F')
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

        Fnew = np.zeros((norb,norb,ns,nft),dtype=np.complex64,order='F')

        if iter == 1:
            mix = 1.0
            Fold = np.zeros((norb,norb,ns,nft),dtype=np.complex64,order='F')

        Fnew = mix*Fb+(1.0-mix)*Fold

        return Fnew
    
    def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]
        nft = matimp.shape[3]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,nft,nspace),dtype=np.complex64,order='F')

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

        matimp = np.zeros((norb,norb,ns,nft,nprob),dtype=np.complex64,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            tempmat = np.zeros((norb,norb,ns),dtype=np.complex64)
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

        matout = np.zeros((norb,norb,ns,nfreq),dtype=np.complex64,order='F')
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

        matout = np.zeros((norb,norb,ns,nft),dtype=np.complex64,order='F')

        matout = DiagE.dyson.flocdyn(mat1,mat2)

        return matout

    def Embedding(self, matin : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size
        nspace = self.crystal.fprojector.shape[3]

        matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')

        for ispace in range(nspace):
            matout += DiagE.embedding.flocdyn(nrk,matin[...,ispace],self.crystal.fprojector[...,ispace])

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

    def __init__(self, crystal: Crystal, ft: FT_grid, green : GreenInt):
        
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

        gf = np.zeros((norbc,norbc,ns,nft,nspace),dtype=np.complex64)

        for ispace in range(nspace):
            gf[...,ispace] = DiagE.projection.flatdyn(self.green.gkf,self.crystal.fprojector[...,ispace])

        self.gf = gf
        self.gt = self.F2T(gf,1,1)

        return None

class GreenImp(FLocDyn): # read CTQMC output

    def __init__(self, crystal: Crystal, ft: FT_grid):
        super().__init__(crystal, ft)
        self.Cal()

    def Cal(self):
        super().Dict2Arr()
        pass

class SigmaLoc(FLocDyn):
    
    def __init__(self, crystal: Crystal, ft: FT_grid, sigma : object):
        super().__init__(crystal, ft)
        
        self.sigma = sigma
        self.f = None
        self.Cal()

    def Cal(self): # projection
        
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nft = self.ft.size
        nspace = self.crystal.fprojector.shape[3]

        sigmalocf = np.zeros((norbc,norbc,ns,nft,nspace),dtype=np.complex64,order='F')

        for isapce in range(nspace):
            sigmalocf[...,isapce] = DiagE.projection.flatdyn(self.sigma,self.crystal.fprojector[...,isapce])

        self.f = sigmalocf
        self.t = self.F2T(sigmalocf,0,1)

        return None


class SigmaImp(FLocDyn): # read CTQMC output

    def __init__(self, crystal: Crystal, ft: FT_grid):
        super().__init__(crystal, ft)
        self.Cal()

    def Cal(self):
        super().Dict2Arr()
        pass

class SigmaLGWC(FLocDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid):
        super().__init__(crystal, ft)

        pass
    

class Hybridisation(FLocDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid, implev : object, gimp : GreenImp, sigmaimp : SigmaImp):
        super().__init__(crystal, ft)
        self.Cal()
    
    def Cal(self):
        pass

class FLocStc(object):

    def __init__(self,crystal : Crystal):

        self.crystal = crystal

    def Inverse(self,mat : np.ndarray):

        norb = mat.shape[0]
        ns = mat.shape[2]

        matinv = np.zeros((norb,norb,ns),dtype=np.complex64,order='F')

        for js in range(ns):
            matinv[:,:,js] = np.linalg.inv(mat[:,:,js])

        return matinv
    
    def Mixing(self,iter : int, mix : float, Fb : np.ndarray, Fold : np.ndarray) -> np.ndarray:

        norb = Fb.shape[0]
        ns = Fb.shape[2]

        Fnew = np.zeros((norb,norb,ns),dtype=np.complex64,order='F')

        if iter == 1:
            mix = 1.0
            Fold = np.zeros((norb,norb,ns),dtype=np.complex64,order='F')

        Fnew = mix*Fb + (1.0-mix)*Fold

        return Fnew
    
    def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]


        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,nspace),dtype=np.complex64,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            for ispace in val:
                matloc[...,ispace] = matimp[...,iprob]

        return matloc
        
    def Loc2Imp(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]
    

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,nspace),dtype=np.complex64,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            for ispace in val:
                matloc[...,ispace] = matimp[...,iprob]

        return matloc
    
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
                e /= len(pos)
                matdict[ind+1].append(e)
        
        return matdict
    
    def Dict2Arr(self, equiv : np.ndarray, matdict : dict) -> np.ndarray:

        norb = len(equiv)
        ns = self.crystal.ns
        matout = np.zeros((norb,norb,ns),dtype=np.complex64,order='F')
        nind = np.amax(equiv)

        for js in range(ns):
            for ind in range(nind):
                pos = self.crystal.FindPositions(equiv,ind+1)
                for ii,jj in pos:
                    matout[ii,jj,js] = matdict[str(ind+1)]

        return matout
    
    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

       
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        
        matout = np.zeros((norb,norb,ns),dtype=np.complex64,order='F')

        matout = DiagE.dyson.flocstc(mat1,mat2)

        return matout 

    def Embedding(self, matin : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nspace = self.crystal.fprojector.shape[3]
        
        matout = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        
        for ispace in range(nspace):
            matout += DiagE.embedding.flocstc(nrk,matin[...,ispace],self.crystal.fprojector[...,ispace])

        return matout
    
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
    
    
class ImpurityLevel(FLocStc):

    def __init__(self, crystal: Crystal, niham : NIHamiltonian, mu : float):
        super().__init__(crystal)
        
        self.niham = niham
        self.mu = mu
        self.loc = None
        self.imp = None
        self.Cal()

    def Cal(self):
        
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nspace = self.crystal.fprojector.shape[3]

        ham = self.niham.UpdateMu(self.niham.hamtb,self.mu)

        eimp = np.zeros((norbc,norbc,ns,nspace),dtype=np.complex64,order='F')

        for ispace in range(nspace):
            eimp[...,ispace] = DiagE.projection.flatstc(ham,self.crystal.fprojector[...,ispace])

        self.loc = eimp
        self.imp = self.Loc2Imp(eimp)

        return None
        

class SigmaHLoc(FLocStc):

    def __init__(self, crystal: Crystal, gloc : GreenLoc, vbare : object):
        super().__init__(crystal)
        
        self.gloc = gloc
        self.vbare = vbare
        self.hloc = None
        self.himp = None
        self.hdyn = None
        self.Cal()
        self.MakeDyn()

    def Cal(self):
        
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        norb = self.crystal.bprojector.shape[1]
        nspace = self.crystal.bprojector.shape[3]

        U = np.zeros((norb,norb,ns,ns,nspace),dtype=np.complex64,order='F')
        hloc = np.zeros((norbc,norbc,ns,nspace),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex64,order='F')

        for ispace in range(nspace):
            U[...,ispace] = DiagE.projection.blatstc(self.vbare.k,self.crystal.bprojector[...,ispace])
        
            if ns == 2:
                tempmat = self.crystal.OrbSpin2Composite(U[...,ispace])
                for ind1 in range(norb*ns):
                    nn1 = [0]*2
                    ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                    iorbc1, iorbc2 = self.crystal.b2f[iorb]
                    for ind2 in range(norb*ns):
                        nn2 = [0]*2
                        ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],ind2,nn2)
                        iorbc3,iorbc4 = self.crystal.b2f[jorb]
                        hloc[iorbc1,iorbc2,js,ispace] += -tempmat[ind1,ind2]*self.gloc.gf[iorbc4,iorbc3,ks,-1,ispace]
            else:
                if self.crystal.soc == False:
                    C = 2
                    for iorb in range(norb):
                        iorbc1, iorbc2 = self.crystal.b2f[iorb]
                        for jorb in range(norb):
                            iorbc3,iorbc4 = self.crystal.b2f[jorb]
                            hloc[iorbc1,iorbc2,0,ispace] += -U[iorb,jorb,0,0,ispace]*self.gloc.gf[iorbc4,iorbc3,0,-1,ispace]
                else:
                    C = 1
                    for iorb in range(norb):
                        iorbc1, iorbc2 = self.crystal.b2f[iorb]
                        for jorb in range(norb):
                            iorbc3,iorbc4 = self.crystal.b2f[jorb]
                            hloc[iorbc1,iorbc2,0,ispace] += -U[iorb,jorb,0,0,ispace]*self.gloc.gf[iorbc4,iorbc3,0,-1,ispace]
            
        self.hloc = hloc
        self.himp = self.Loc2Imp(hloc)

        return None
    
    def MakeDyn(self):

        norb = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nft = self.gloc.gf.shape[3]
        nspace = self.crystal.fprojector.shape[3]

        hdyn = np.zeros((norb,norb,ns,nft,nspace),dtype=np.complex64,order='F')
        
        for ift in range(nft):
            hdyn[...,ift,:] = self.hloc
        
        self.hdyn = hdyn

        return None

class SigmaHImp(FLocStc):

    def __init__(self, crystal: Crystal):
        super().__init__(crystal)
        self.Cal()

    def Cal(self):
        pass

class SigmaFLoc(FLocStc):

    def __init__(self, crystal: Crystal, gloc : GreenLoc, vbare : object):
        super().__init__(crystal)

        self.gloc = gloc
        self.vbare = vbare
        self.floc = None
        self.fimp = None
        self.fdyn = None
    
        self.Cal()
        self.MakeDyn()

    def Cal(self):
        
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        norb = self.crystal.bprojector.shape[1]
        nspace = self.crystal.fprojector.shape[3]

        U = np.zeros((norb,norb,ns,ns,nspace),dtype=np.complex64,order='F')
        floc = np.zeros((norbc,norbc,ns,nspace),dtype=np.complex64,order='F')
        

        for ispace in range(nspace):
            U[...,ispace] = DiagE.projection.blatstc(self.vbare.k,self.crystal.bprojector[...,ispace])

            for js in range(ns):
                for iorb in range(norb):
                    iorbc1, iorbc4 = self.crystal.b2f[iorb]
                    for jorb in range(norb):
                        iorbc3, iorbc2 = self.crystal.b2f[jorb]
                        floc[iorbc1,iorbc2,js,ispace] += self.gloc.gf[iorbc4,iorbc3,js,-1,ispace]*U[iorb,jorb,js,js,ispace]

        self.floc = floc
        self.fimp = self.Loc2Imp(floc)
        
        return None



class SigmaFImp(FLocStc):

    def __init__(self, crystal: Crystal):
        super().__init__(crystal)
        self.Cal()

    def Cal(self):
        pass


class BLatDyn(object):

    def __init__(self,crystal : Crystal, ft : FT_grid):

        self.crystal = crystal
        self.ft = ft
        # self.flatdyn = flatdyn

    def Inverse(self,matin : np.ndarray)-> np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]
        nft = matin.shape[5]

        matout = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex64)
        tempmat2 = np.zeros((norb*ns,norb*ns),dtype=np.complex64)

        # Make composite matrix#
        for ift in range(nft):
            for irk in range(nrk):
                tempmat = self.crystal.OrbSpin2Composite(matin[:,:,:,:,irk,ift])
                tempmat2 = np.linalg.inv(tempmat)
                matout[:,:,:,:,irk,ift] = self.crystal.Composite2OrbSpin(tempmat2)
        
        return matout

    def Moment(self,bf : np.ndarray, oddzero : int, highzero : int) -> np.ndarray:

        norb = bf.shape[0]
        ns = bf.shape[2]
        nrk = bf.shape[4]

        moment = np.zeros((norb,norb,ns,ns,nrk,3),dtype=np.complex64,order='F')
        high = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')

        moment, high = DiagE.fourier.blatdyn_m(self.ft.nu,bf,oddzero,highzero)

        return moment, high
    
    def F2T(self,bf : np.ndarray, oddzero : int, highzero : int) -> np.ndarray:

        norb = bf.shape[0]
        ns = bf.shape[2]
        nrk = bf.shape[4]
        ntau = len(self.ft.tau)

        btau = np.zeros((norb,norb,ns,ns,nrk,ntau),dtype=np.complex64,order='F')

        moment, high = self.Moment(bf,oddzero,highzero)

        btau = DiagE.fourier.blatdyn_f2t(self.ft.nu,bf,moment,self.ft.tau)

        return btau
    
    def T2F(self,btau : np.ndarray) -> np.ndarray:

        norb = btau.shape[0]    
        ns = btau.shape[2]
        nrk = btau.shape[4]
        nfreq = len(self.ft.nu)

        bf = np.zeros((norb,norb,ns,ns,nrk,nfreq),dtype=np.complex64,order='F')

        bf = DiagE.fourier.blatdyn_t2f(self.ft.tau,btau,self.ft.nu)

        return bf
    
    def K2R(self, matk : np.ndarray)-> np.ndarray:
        
        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[4]
        nft = matk.shape[5]
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        matr = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

        tempmat = copy.deepcopy(matk)

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                                [b,[m2,m3]] = self.crystal.BAtomOrb(jorb)
                                
                                delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]

                                phase = np.exp(2.0j*np.pi*np.dot(rkvec[irk],delta))
                                tempmat[iorb,jorb,js,ks,irk,ift] *= phase
        
        matr = DiagE.fourier.blatdyn_k2r(rkgrid,tempmat)

        return matr
    
    def R2K(self, matr : np.ndarray)->np.ndarray:

        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[4]
        nft = matr.shape[5]
        rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        # matk = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

        matk = DiagE.fourier.blatdyn_r2k(rkgrid,matr)

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):

                                [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                                [b,[m2,m3]] = self.crystal.BAtomOrb(jorb)

                                delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
                                phase = np.exp(-2.0j*np.pi*np.dot(rkvec[irk],delta))

                                matk[iorb,jorb,js,ks,irk,ift] *= phase

        
        return matk
    
    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):
        
        norb = y.shape[0]
        ns = y.shape[2]
        nrk = y.shape[3]
        nft = y.shape[4]

        ynew = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

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
                            for ks in range(ns):
                                for iorb in range(norb):
                                    for jorb in range(norb):
                                        ynew[iorb,jorb,js,ks,irk,cnt] = sum(dist*y[iorb,jorb,js,ks,irk])/sum(dist)
                    else:
                        ynew[...,irk,cnt] = y[...,irk,cnt]
                cnt += 1

        return ynew

    def Mixing(self,iter : int, mix : float, Bb : np.ndarray, Bold : np.ndarray) -> np.ndarray:

        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nrk = Bb.shape[4]
        nft = Bb.shape[5]

        Bnew = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

        if iter == 1:
            mix = 1.0
            Bold = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')
        
        Bnew = mix*Bb + (1-mix)*Bold

        return Bnew

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        norb = mat1.shape[0]
        ns = mat1.shape[2]
        nrk = mat1.shape[3]
        nft = mat1.shape[4]

        matout = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

        matout = DiagE.dyson.blatdyn(mat1,mat2)

        return matout

    def Projection(self, matin : np.ndarray):

        norbc = self.crystal.bprojector.shape[1]
        ns = self.crystal.ns
        nft = self.ft.size
        nspace = self.crystal.bprojector.shape[3]

        matout = np.zeros((norbc,norbc,ns,ns,nft,nspace),dtype=np.complex64,order='F')

        for ispace in range(nspace):
            matout[...,ispace] = DiagE.projection.blatdyn(matin,self.crystal.bprojector[...,ispace])

        return matout

    def Quad2Double(self,matin : np.ndarray)->np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size
        
        matout = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for ks in range(ns):
                    for js in range(ns):
                        matout[:,:,js,ks,irk,ift] = self.crystal.Quad2Double(matin[:,:,:,:,js,ks,irk,ift])

        return matout
    
    def Double2Quad(self,matin : np.ndarray)->np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros((norb,norb,norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for ks in range(ns):
                    for js in range(ns):
                        matout[:,:,:,:,js,ks,irk,ift] = self.crystal.Double2Quad(matin[:,:,js,ks,irk,ift])

        return matout
    
    def Double2Full(self,matin : np.ndarray)->np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros((norb*norb,norb*norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')
        
        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        matout[:,:,js,ks,irk,ift] = self.crystal.Double2Full(matin[:,:,js,ks,irk,ift])

        return matout
    
    def Full2Double(self, matin : np.ndarray) -> np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        matout[:,:,js,ks,irk,ift] = self.crystal.Full2Double(matin[:,:,js,ks,irk,ift])

        return matout
    
    def Quad2Full(self,matin : np.ndarray) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros((norb*norb,norb*norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        matout[:,:,js,ks,irk,ift] = self.crystal.Quad2Full(matin[:,:,:,:,js,ks,irk,ift])

        return matout
    
    def Full2Quad(self, matin : np.ndarray)->np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size

        matout = np.zeros((norb,norb,norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        matout[:,:,:,:,js,ks,irk,ift] = self.crystal.Full2Quad(matin[:,:,js,ks,irk,ift])

        return matout
    
    def StcEmbedding(self,matin : np.ndarray)->np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]
        nft = self.ft.size

        matout = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

        for ift in range(nft):
            matout[...,ift] += matin
        
        return matout

    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]
        nft = matin.shape[5]

        if os.path.exists('blatdyn'):
            pass
        else:
            os.mkdir("blatdyn")
        os.chdir('blatdyn')

        with open(fn+'.txt','w') as f:
            f.write("iorb, jorb, is, js, irk, ift, Re(B(k,w)), Im(B(k,w))\n")
            for ift in range(nft):
                for irk in range(nrk):
                    for ks in range(ns):
                        for js in range(ns):
                            for jorb in range(norb):
                                for iorb in range(norb):
                                    f.write(f"{iorb} {jorb} {js} {ks} {irk} {ift} {matin[iorb,jorb,js,ks,irk,ift].real} {matin[iorb,jorb,js,ks,irk,ift].imag}\n")
        
        os.chdir('..')

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
        nft = matr.shape[4]

        self.crystal.Rvec()
        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb,norb,ns,ns,nk,nft),dtype=complex,order='F')

        for ift in range(nft):
            for ik in range(nk):
                for ks in range(ns):
                    for js in range(ns):
                        for jorb in range(norb):
                            for iorb in range(norb):
                                temp = 0
                                for ir in range(nr):
                                    temp += tempmat[iorb,jorb,js,ks,ir,ift]*np.exp(-2.0j*np.pi*(kpoint[ik]@self.crystal.rvec[ir]))
                                [a,m1] = self.crystal.FAtomOrb(iorb)
                                [b,m2] = self.crystal.FAtomOrb(jorb)
                                delta = self.crystal.basisf[a,:]-self.crystal.basisf[b,:]
                                phase = np.exp(-2.0j*np.pi*(kpoint[ik]@delta))
                                matk[iorb,jorb,js,ks,ik,ift] = temp*phase
        
        return matk

class PolLat(BLatDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid,green : np.ndarray = None):
        super().__init__(crystal, ft)
        self.polrt = None # rt to kf
        self.polrf = None
        self.polkt = None
        self.polkf = None
        if green is None:
            print("Error, There is no Green's function.")
            sys.exit()
        self.green = green

        self.Cal()
        self.polkt = self.R2K(self.polrt)
        self.polrf = self.T2F(self.polrt)
        self.polkf = self.R2K(self.polrf)

    def Cal(self):
        grt = self.green
        # norbc = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        ntau = len(self.ft.tau)
        norb = len(self.crystal.bind)
        polrt = np.zeros((norb,norb,ns,ns,nrk,ntau),dtype=np.complex64,order='F')

        gmrt = self.crystal.RT2mRmT(grt)
        
        if ns == 2:
            for itau in range(ntau):
                for irk in range(nrk):
                    for js in range(ns):
                        for ks in range(ns):
                            for iorb in range(norb):
                                [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                                iorbc = self.crystal.FIndex([a,m1])
                                lorbc = self.crystal.FIndex([a,m4])
                                for jorb in range(norb):
                                    [b,[m2,m3]] = self.crystal.BAtomOrb(jorb)
                                    jorbc = self.crystal.FIndex([b,m2])
                                    korbc = self.crystal.FIndex([b,m3])
                                    if js == ks:
                                        polrt[iorb,jorb,js,ks,irk,itau] = gmrt[korbc,iorbc,js,irk,itau]*grt[lorbc,jorbc,ks,irk,itau]
                                
        else:
            if self.crystal.soc == True:
                C = 1
                for itau in range(ntau):
                    for irk in range(nrk):
                        for iorb in range(norb):
                            [a,[m1,m3]] = self.crystal.BAtomOrb(iorb)
                            iorbc = self.crystal.FIndex([a,m1])
                            korbc = self.crystal.FIndex([a,m3])
                            for jorb in range(norb):
                                [b,[m4,m2]] = self.crystal.BAtomOrb(jorb)
                                lorbc = self.crystal.FIndex([b,m4])
                                jorbc = self.crystal.FIndex([b,m2])
                                polrt[iorb,jorb,0,0,irk,itau] = gmrt[jorbc,iorbc,0,irk,itau]*grt[korbc,lorbc,0,irk,itau]*C
            else:
                C = 2
                for itau in range(ntau):
                    for irk in range(nrk):
                        for iorb in range(norb):
                            [a,[m1,m3]] = self.crystal.BAtomOrb(iorb)
                            iorbc = self.crystal.FIndex([a,m1])
                            korbc = self.crystal.FIndex([a,m3])
                            for jorb in range(norb):
                                [b,[m4,m2]] = self.crystal.BAtomOrb(jorb)
                                lorbc = self.crystal.FIndex([b,m4])
                                jorbc = self.crystal.FIndex([b,m2])
                                # if (iorb==0)and(jorb==0)and(irk==0):
                                #     print(iorbc,jorbc,korbc,lorbc,irk,itau,gmrt[jorbc,iorbc,0,irk,itau],grt[korbc,lorbc,0,irk,itau])
                                polrt[iorb,jorb,0,0,irk,itau] = gmrt[jorbc,iorbc,0,irk,itau]*grt[korbc,lorbc,0,irk,itau]*C
        

        self.polrt = polrt

        return None

class WLat(BLatDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid,pol : np.ndarray = None, vbare : object = None):
        super().__init__(crystal, ft)
        self.wrt = None #rt to kf
        self.wrf = None
        self.wkt = None
        self.wkf = None
        self.wcrt = None #rt to kf
        self.wcrf = None
        self.wckt = None
        self.wckf = None
        if pol is None:
            print("Error, polarizability doesn't exist")
            sys.exit()
        if vbare is None:
            print("Error, bare coulomb interaction doesn't exist")
            sys.exit()
        self.pol = pol
        self.vbare = vbare

        self.Cal()

        # self.wkt = self.F2T(self.wkf,1,1)
        # self.wrf = self.K2R(self.wkf)
        # self.wrt = self.K2R(self.wkt)

        self.wckt = self.F2T(self.wckf,1,1)
        self.wcrf = self.K2R(self.wckf)
        self.wcrt = self.K2R(self.wckt)

    def Cal(self): # calculate W and Wc
        
        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nfreq = len(self.ft.nu)
        ####### Initialization #######
        tempmat = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nk,nfreq),dtype=np.complex64,order='F')
        wkf = np.zeros((norb,norb,ns,ns,nk,nfreq),dtype=np.complex64,order='F')
        wckf = np.zeros((norb,norb,ns,ns,nk,nfreq),dtype=np.complex64,order='F')
        vdyn = np.zeros((norb,norb,ns,ns,nk,nfreq),dtype=np.complex64,order='F')

        # for ifreq in range(nfreq):
        #     vdyn[...,ifreq] = self.vbare.k
        vdyn = self.StcEmbedding(self.vbare.k)
        polcomp = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nk,nfreq),dtype=np.complex64,order='F')
        vcomp = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nk,nfreq),dtype=np.complex64,order='F')
        ####### Initialization #######
        polcomp = self.Double2Full(self.pol)
        vcomp = self.Double2Full(vdyn)
        
        tempmat = self.Dyson(vcomp,polcomp)
        wkf = self.Full2Double(tempmat)
        
        self.wkf = wkf

        wckf = wkf - vdyn

        self.wckf = wckf

        return None

# class WcLat(BLatDyn):

#     def __init__(self, crystal: Crystal, ft: FT_grid, w ):
#         super().__init__(crystal, ft, flatdyn)

#         pass

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

class VBare(BLatStc):

    def __init__(self, crystal: Crystal,vloc = None, orboption : dict = None, intamp : list = None):
        super().__init__(crystal)
        self.k = None
        self.r = None
        self.intamp = intamp
        self.nonlock = None
        self.nonlocr = None
        self.sigmaonsiter = None
        if vloc == None:
            if orboption != None:
                self.vloc = VLoc(crystal,orboption)
            else:
                print("Error, orboption is not exsist. v local can't generate in here")
        else:
            self.vloc = vloc
        if intamp != None:
            # self.InteractingAmplitue(intamp)
            self.Cal()
        self.LocPlusNonLoc()
        # self.GetOnsiteEnergy()
        

    def Cal(self):

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
                        tempmat[iorb,jorb,js,ks,R[0],R[1],R[2]] += vij
                
                    else:
                        tempmat[iorb,jorb,js,ks,R[0],R[1],R[2]] += vij
                        tempmat[jorb,iorb,js,ks,-R[0],-R[1],-R[2]] += vij
                    
        
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
    
    def Save(self, fn: str):

        if os.path.exists('vbare'):
            pass
        else:
            os.mkdir('vbare')
        
        os.chdir('vbare')
        super().Save(self.k, fn)
        os.chdir('..')

        return None
    
    # def GetOnsiteEnergy(self):

    #     norbc = len(self.crystal.find)
    #     ns = self.crystal.ns
    #     nrk = len(self.crystal.kpoint)
    #     norb = len(self.crystal.bind)

    #     tempmat = np.zeros((norbc,norbc,ns,nrk),dtype=np.complex64,order='F')

    #     for js in range(ns):
    #         for iorb in range(norb):
    #             [a,[m1,m2]] = self.crystal.BAtomOrb(iorb)
    #             iorbc = self.crystal.FIndex([a,m1])
    #             jorbc = self.crystal.FIndex([a,m2])
    #             if iorbc==jorbc:
    #                 tempmat[iorbc,iorbc,js,0] = -self.vloc.vloc[iorb,iorb,js,js]

    #     self.sigmaonsiter = tempmat
        
    #     return None    


class BLocDyn(object):

    def __init__(self, crystal : Crystal, ft : FT_grid):

        self.crystal = crystal
        self.ft = ft

    def Inverse(self, matin : np.ndarray)-> np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]
        nft = self.ft.size

        matout = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex64)
        tempmat2 = np.zeros((norb*ns,norb*ns),dtype=np.complex64)

        for ift in range(nft):
            tempmat = self.crystal.OrbSpin2Composite(matin[...,ift])
            tempmat2 = np.linalg.inv(tempmat)
            matout[...,ift] = self.crystal.Composite2OrbSpin(tempmat2)
        
        return matout

    def Moment(self, bf : np.ndarray, oddzero : int, highzero : int) -> np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns

        moment = np.zeros((norb,norb,ns,ns,3),dtype=np.complex64,order='F')
        high = np.zeros((norb,norb,ns,ns),dtype=np.complex64,order='F')
        moment, high = DiagE.fourier.blocdyn_m(self.ft.nu,bf,oddzero,highzero)

        return moment,high
    
    def F2T(self,bf : np.ndarray, oddzero : int, highzero : int) -> np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nft = self.ft.size

        btau = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex64,order='F')

        moment, high = self.Moment(bf,oddzero,highzero)

        btau = DiagE.fourier.blocdyn_f2t(self.ft.nu,bf,moment,self.ft.tau)

        return btau

    def T2F(self, btau : np.ndarray) -> np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nft = self.ft.size 

        bf = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex64,order='F')

        bf = DiagE.fourier.blocdyn_t2f(self.ft.tau,btau,self.ft.nu)

        return bf

    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]

        ynew = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex64,order='F')
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
                        for ks in range(ns):
                            for iorb in range(norb):
                                for jorb in range(norb):
                                    ynew[iorb,jorb,js,ks,cnt] = sum(dist*y[iorb,jorb,js,ks])/sum(dist)
                else:
                    ynew[...,cnt] = y[...,cnt]
            cnt += 1

        return ynew
    
    def Mixing(self,iter : int, mix : float, Bb : np.ndarray, Bold : np.ndarray):

        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nft = Bb.shape[4]

        Bnew = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex64,order='F')

        if iter == 1:
            mix = 1.0
            Bold = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex64,order='F')

        Bnew = mix*Bb + (1-mix)*Bold

        return Bnew
    
    def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]
        nft = matimp.shape[3]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,ns,nft,nspace),dtype=np.complex64,order='F')

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

        matimp = np.zeros((norb,norb,ns,ns,nft,nprob),dtype=np.complex64,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            tempmat = np.zeros((norb,norb,ns),dtype=np.complex64)
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
                for ks in range(ns):
                    e = 0
                    for ii, jj in pos:
                        e += matin[ii,jj,js,ks]
                    e /=len(pos)
                    matdict[ind+1].append(e.tolist())
        
        return matdict
    
    def Dict2Arr(self,equiv : np.ndarray, matdict : np.ndarray) -> np.ndarray:

        norb = len(equiv)
        ns = self.crystal.ns
        nfreq = len(matdict["1"])                

        matout = np.zeros((norb,norb,ns,ns,nfreq),dtype=np.complex64,order='F')
        nind = np.amax(equiv)

        for js in range(ns):
            for ks in range(ns):
                for ind in range(nind):
                    pos = self.crystal.FindPositions(equiv,ind+1)
                    for ii, jj in pos:
                        matout[ii,jj,js,ks] = matdict[str(ind+1)]

        return matout

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = mat1.shape[0]
        ns = self.crystal.ns
        nft = self.ft.size

        matout = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex64,order='F')

        matout = DiagE.dyson.blocdyn(mat1,mat2)

        return matout

    def Embedding(self, matin : np.ndarray):

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size
        nspace = self.crystal.bprojector.shape[3]

        matout = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex64,order='F')

        for ispace in range(nspace):
            matout += DiagE.embedding.blocdyn(nrk,matin[...,ispace],self.crystal.bprojector[...,ispace])

        return matout
    
    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]
        nft = matin.shape[4]

        if os.path.exists('blocdyn'):
            pass
        else:
            os.mkdir('blocdyn')
        os.chdir('blocdyn')

        with open(fn+'txt','w') as f:
            f.write("iorb, jorb, is, js, ift, Re(B(w)), Im(B(w))\n")
            for ift in range(nft):
                for ks in range(ns):
                    for js in range(ns):
                        for jorb in range(norb):
                            for iorb in range(norb):
                                f.write(f"{iorb} {jorb} {js} {ks} {matin[iorb,jorb,js,ks,ift].real} {matin[iorb,jorb,js,ks,ift].imag}\n")
        
        os.chdir('..')
        return None

class PolLoc(BLocDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid, green, pol : object):
        super().__init__(crystal, ft)
        self.Cal()

    def Cal(self):
        pass

class PolImp(BLocDyn): # read Polarizability from CTQMC

    def __init__(self, crystal: Crystal, ft: FT_grid):
        super().__init__(crystal, ft)

        pass

class WLoc(BLocDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid, flocdyn: FLocDyn):
        super().__init__(crystal, ft, flocdyn)

        pass

class WImp(BLocDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid, flocdyn: FLocDyn):
        super().__init__(crystal, ft, flocdyn)

        pass

class WcLoc(BLocDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid, flocdyn: FLocDyn):
        super().__init__(crystal, ft, flocdyn)

        pass

class WcImp(BLocDyn):

    def __init__(self, crystal: Crystal, ft: FT_grid, flocdyn: FLocDyn):
        super().__init__(crystal, ft, flocdyn)

        pass

class BLocStc(object):

    def __init__(self,crystal : Crystal):

        self.crystal = crystal

    def Inverse(self, matin : np.ndarray)-> np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]

        matout = np.zeros((norb,norb,ns,ns),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex64)
        tempmat2 = np.zeros((norb*ns,norb*ns),dtype=np.complex64)

        
        tempmat = self.crystal.OrbSpin2Composite(matin)
        tempmat2 = np.linalg.inv(tempmat)
        matout = self.crystal.Composite2OrbSpin(tempmat2)
        
        return matout
    
    def Mixing(self, iter : int, mix : float, Bb : np.ndarray, Bold : np.ndarray)-> np.ndarray:

        norb = Bb.shape[0]
        ns = Bb.shape[2]

        Bnew = np.zeros((norb,norb,ns,ns),dtype=np.complex64,order='F')

        if iter == 1:
            mix = 1.0
            Bold = np.zeros((norb,norb,ns,ns),dtype=np.complex64,order='F')
        
        Bnew = mix*Bb + (1-mix)*Bold

        return Bnew

    def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]


        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,ns,nspace),dtype=np.complex64,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            for ispace in val:
                matloc[...,ispace] = matimp[...,iprob]

        return matloc
        
    def Loc2Imp(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]
    

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,ns,nspace),dtype=np.complex64,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            for ispace in val:
                matloc[...,ispace] = matimp[...,iprob]

        return matloc
    
    def Arr2Dict(self, equiv : np.ndarray, matin : np.ndarray) -> dict:

        ns = matin.shape[2]
        nind = np.amax(equiv)
        matdict = {}

        
        for ind in range(nind):
            matdict[ind+1] = []
            pos = self.crystal.FindPositions(equiv,ind+1)
            for js in range(ns):
                for ks in range(ns):
                    e = 0
                    for ii, jj in pos:
                        e += matin[ii,jj,js,ks]
                    e /= len(pos)
                    matdict[ind+1].append(e)
        
        return matdict
    
    def Dict2Arr(self, equiv : np.ndarray, matdict : dict) -> np.ndarray:

        norb = len(equiv)
        ns = self.crystal.ns
        matout = np.zeros((norb,norb,ns),dtype=np.complex64,order='F')
        nind = np.amax(equiv)

        for js in range(ns):
            for ks in range(ns):
                for ind in range(nind):
                    pos = self.crystal.FindPositions(equiv,ind+1)
                    for ii,jj in pos:
                        matout[ii,jj,js,ks] = matdict[str(ind+1)]

        return matout

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = mat1.shape[0]
        ns = mat1.shape[2]

        matout = np.zeros((norb,norb,ns,ns),dtype=np.complex64,order='F')

        matout = DiagE.dyson.blocstc(mat1,mat2)

        return matout

    def Embedding(self, matin : np.ndarray):

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nspace = self.crystal.bprojector.shape[3]

        matout = np.zeros((norb,norb,ns,ns,nrk),dtype=np.complex64,order='F')

        for ispace in range(nspace):
            matout += DiagE.embedding.blocstc(nrk,matin[...,ispace],self.crystal.bprojector.shape[...,ispace])

        return matout
    
    def Double2Quad(self, matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norbc,norbc,norbc,norbc,ns,ns),dtype=np.complex64,order='F')
        
        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.crystal.Double2Quad(matin[...,js,ks])

        return matout
    
    def Quad2Double(self,matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norb,norb,ns,ns),dtype=np.complex64,order='F')

        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.crystal.Quad2Double(matin[...,js,ks])

        return matout
    
    def Double2Full(self,matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norbc**2,norbc**2,ns,ns),dtype=np.complex64)

        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.Double2Full(matin[...,js,ks])
        
        return matin
    
    def Full2Double(self,matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norb,norb,ns,ns),dtype=np.complex64,order='F')

        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.crystal.Full2Double(matin[...,js,ks])

        return matout
    
    def Quad2Full(self,matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norbc*norbc,norbc*norbc,ns,ns),dtype=np.complex64,order='F')

        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.Quad2Full(matin[...,js,ks])
        
        return matout
    
    def Full2Quad(self,matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norbc,norbc,norbc,norbc,ns,ns),dtype=np.complex64,order='F')

        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.crystal.Full2Quad(matout[...,js,ks])

        return matout
    
    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]

        if os.path.exists('blocstc'):
            pass
        else:
            os.mkdir('blocstc')
        os.chdir('blocstc')
        
        with open(fn+'.txt','w') as f:
            f.write("iorb, jorb, is, js, Re(B), Im(B)\n")
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            f.write(f"{iorb} {jorb} {js} {ks} {matin[iorb,jorb,js,ks].real} {matin[iorb,jorb,js,ks].imag}\n")
        
        os.chdir("..")
        return None

class VLoc(BLocStc):

    def __init__(self, crystal: Crystal,voption : dict = None):
        super().__init__(crystal)
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        self.onsitelist = None
        self.vloc = np.zeros((norb,norb,ns,ns),dtype=float,order='F')
        if voption is None:
            print("voption does not exist")
            sys.exit()
        self.SetLocalInteracting(voption)
        # self.GenOnsite()

    def SetLocalInteracting(self,voption : dict):
        
        ns = self.crystal.ns

        if voption["Parameter"] == "Kanamori":
            for key, val in voption["option"].items():
                atom = int(key-1)
                norbc = len(val["orbitals"])
                if norbc > len(self.crystal.find):
                    print("Invalid l value set")
                    sys.exit()
                tempmat = self.KanamoriParameter(norb=norbc,val=val["value"])
                for js in range(ns):
                    for ks in range(ns):
                        for m1,m2,m3,m4 in itertools.product(val["orbitals"],val["orbitals"],val["orbitals"],val["orbitals"]):
                            iorb = self.crystal.BIndex([atom,[m1,m4]])
                            jorb = self.crystal.BIndex([atom,[m2,m3]])
                            if (iorb is not None)and(jorb is not None):
                                self.vloc[iorb,jorb,js,ks] = tempmat[m1,m2,m3,m4,js,ks]
        if voption["Parameter"] == "Slater":
            for key, val in voption["option"].items():
                atom = int(key-1)
                norbc = len(val["orbitals"])
                if norbc > len(self.crystal.find):
                    print("Invalid l value set")
                    sys.exit()
                tempmat = self.SlaterParameter(l=val["l"],norbc=norbc,val=val["value"])
                for js, ks in itertools.product(list(range(ns)),list(range(ns))):
                    for m1,m2,m3,m4 in itertools.product(val["orbitals"],val["orbitals"],val["orbitals"],val["orbitals"]):
                        iorb = self.crystal.BIndex([atom,[m1,m4]])
                        jorb = self.crystal.BIndex([atom,[m2,m3]])
                        if (iorb is not None)and(jorb is not None):
                            self.vloc[iorb,jorb,js,ks] = tempmat[m1,m2,m3,m4,js,ks]
        if voption["Parameter"] == "SlaterKanamori":
            for key, val in voption["option"].items():
                atom = int(key-1)
                norbc = len(val["orbitals"])
                if norbc > len(self.crystal.find):
                    print("Invalid l value set")
                    sys.exit()
                tempmat = self.SlaterKanamori(l=val["l"],norb=norbc,val=val["value"])
                for js, ks in itertools.product(list(range(ns)),list(range(ns))):
                    for m1,m2,m3,m4 in itertools.product(val["orbitals"],val["orbitals"],val["orbitals"],val["orbitals"]):
                        iorb = self.crystal.BIndex([atom,[m1,m4]])
                        jorb = self.crystal.BIndex([atom,[m2,m3]])
                        if (iorb is not None)and(jorb is not None):
                            self.vloc[iorb,jorb,js,ks] = tempmat[m1,m2,m3,m4,js,ks]

            
        # for val in orboption.values():
        #     norbc = len(val["orbitals"])
            
        #     if val["Parameter"] == "Kanamori":
        #         tempmat = self.KanamoriParameter(norbc,val["value"])
        #         for js in range(ns):
        #             for ks in range(ns):
        #                 for iorbc in val["orbitals"]:
        #                     for jorbc in val["orbitals"]:
        #                         for korbc in val["orbitals"]:
        #                             for lorbc in val["orbitals"]:
        #                                 [a,m1] = self.crystal.FAtomOrb(iorbc)
        #                                 [b,m2] = self.crystal.FAtomOrb(jorbc)
        #                                 [bp,m3] = self.crystal.FAtomOrb(korbc)
        #                                 [ap,m4] = self.crystal.FAtomOrb(lorbc)
        #                                 if(a==ap)and(b==bp):
        #                                     iorb = self.crystal.BIndex([a,[m1,m4]])
        #                                     jorb = self.crystal.BIndex([b,[m2,m3]])
        #                                     self.vloc[iorb,jorb,js,ks] = tempmat[m1,m2,m3,m4,js,ks]
        #     elif val["Parameter"] == "Slater":
        #         tempmat = self.SlaterParameter(norbc,val["value"])
        #         for js in range(ns):
        #             for ks in range(ns):
        #                 for iorbc in val["orbitals"]:
        #                     for jorbc in val["orbitals"]:
        #                         for korbc in val["orbitals"]:
        #                             for lorbc in val["orbitals"]:
        #                                 [a,m1] = self.crystal.FAtomOrb(iorbc)
        #                                 [b,m2] = self.crystal.FAtomOrb(jorbc)
        #                                 [bp,m3] = self.crystal.FAtomOrb(korbc)
        #                                 [ap,m4] = self.crystal.FAtomOrb(lorbc)
        #                                 if(a==ap)and(b==bp):
        #                                     iorb = self.crystal.BIndex([a,[m1,m4]])
        #                                     jorb = self.crystal.BIndex([b,[m2,m3]])
        #                                     self.vloc[iorb,jorb,js,ks] = tempmat[m1,m2,m3,m4,js,ks]
        #     elif val["Parameter"] == "SlaterKanamori":
        #         print(norbc)
        #         tempmat = self.SlaterKanamori(norbc,val["value"])
        #         for js in range(ns):
        #             for ks in range(ns):
        #                 for iorbc in val["orbitals"]:
        #                     for jorbc in val["orbitals"]:
        #                         for korbc in val["orbitals"]:
        #                             for lorbc in val["orbitals"]:
        #                                 [a,m1] = self.crystal.FAtomOrb(iorbc)
        #                                 [b,m2] = self.crystal.FAtomOrb(jorbc)
        #                                 [bp,m3] = self.crystal.FAtomOrb(korbc)
        #                                 [ap,m4] = self.crystal.FAtomOrb(lorbc)
        #                                 if(a==ap)and(b==bp):
        #                                     iorb = self.crystal.BIndex([a,[m1,m4]])
        #                                     jorb = self.crystal.BIndex([b,[m2,m3]])
        #                                     self.vloc[iorb,jorb,js,ks] = tempmat[m1,m2,m3,m4,js,ks]
        
        return None
    
    def GenOnsite(self):
        
        norbc = len(self.crystal.find)
        ns = self.crystal.ns
        onsitelist = []
        
        tempmat = np.zeros((norbc,norbc,norbc,norbc,ns,ns),dtype=np.complex64,order='F')

        tempmat = self.Double2Quad(self.vloc)

        for js in range(ns):
            for ks in range(ns):
                for iorbc in range(norbc):
                    if (js==ks):
                        onsitelist.append(-tempmat[iorbc,iorbc,iorbc,iorbc,js,ks])
        
        self.onsitelist = onsitelist

        return None

    
    def KanamoriParameter(self, norb : int, val : list) -> np.ndarray:

        # print("Warning : In kanamori interaction, self interaction term has been added")
        ns = self.crystal.ns
        v = np.zeros((norb,norb,norb,norb,ns,ns),dtype=float,order='F')
        U = val[0]
        Up = val[1]
        J = val[2]

        for js in range(ns):
            for ks in range(ns):
                for m1 in range(norb):
                    for m2 in range(norb):
                        for m3 in range(norb):
                            for m4 in range(norb):
                                if (m1==m2==m3==m4)and(js!=ks):
                                    v[m1,m2,m3,m4,js,ks] = U
                                elif(m1==m4)and(m2==m3)and(m1!=m2)and(js!=ks):
                                    v[m1,m2,m3,m4,js,ks] = Up
                                elif(m1==m4)and(m2==m3)and(m1!=m2)and(js==ks):
                                    v[m1,m2,m3,m4,js,ks] = Up-J
                                elif (m1==m3)and(m2==m4)and(m1!=m2)and(js!=ks):
                                    v[m1,m2,m3,m4,js,ks] = J
                                elif (m1==m2)and(m3==m4)and(m1!=m3)and(js!=ks):
                                    v[m1,m2,m3,m4,js,ks] = J
        v *= 0.5
        return v


    def SlaterParameter(self, l : int = None,norbc : int=None, val : list=None, sc : str = 'c') -> np.ndarray:
        
        # error message
        print("Only calculate the odd number of orbitals")
        ns = self.crystal.ns
        norb = 2*l+1
        vtemp = np.zeros((norb,norb,norb,norb,ns,ns),dtype=float,order='F')
        v = np.zeros((norbc,norbc,norbc,norbc,ns,ns),dtype=float,order='F')

        # l = int((norb-1)/2)
        m = list(range(-l,l+1))

        for n, f in enumerate(val):
            k = 2*n
            
            for js in range(ns):
                for ks in range(ns):
                    for m1 in m:
                        for m2 in m:
                            for m3 in m:
                                for m4 in m:
                                    vtemp[m1+l,m2+l,m3+l,m4+l,js,ks] += f*self.AngularIntegral(l,k,m1,m2,m4,m3)
        if sc == 'c':
            for js in range(ns):
                for ks in range(ns):
                    tempmat = vtemp[:,:,:,:,js,ks]
                    tempmat2 = self.Spherical2Cubic(tempmat,l)
                    vtemp[:,:,:,:,js,ks] = tempmat2
            if (l==2)and(norbc==3):
                for ii, iorbc in enumerate([0,1,3]):
                    for jj, jorbc in enumerate([0,1,3]):
                        for kk, korbc in enumerate([0,1,3]):
                            for ll, lorbc in enumerate([0,1,3]):
                                v[ii,jj,kk,ll] = vtemp[iorbc,jorbc,korbc,lorbc]
            elif (l==2)and(norbc==2):
                for ii, iorbc in enumerate([2,4]):
                    for jj, jorbc in enumerate([2,4]):
                        for kk, korbc in enumerate([2,4]):
                            for ll, lorbc in enumerate([2,4]):
                                v[ii,jj,kk,ll] = vtemp[iorbc,jorbc,korbc,lorbc]
            else:
                v = vtemp
            return v
        else:
            return v
    
    
    def SlaterKanamori(self,l : int,norb : int, val : list) -> np.ndarray :

        U = val[0]
        Up = val[1]
        J = val[2]
        ratio = 0.625
        ns = self.crystal.ns
        print(norb)

        v = np.zeros((norb,norb,norb,norb,ns,ns),dtype=float,order='F')

        if norb == 1:
            F0 = U
            F2 = 0
            F4 = 0
            v = self.SlaterParameter(l,norb,[F0,F2,F4])
            return v
        if norb == 3:
            F2 = 441/(27+20*ratio)*J
            F4 = ratio*F2
            F0 = U-4/49*(F2+F4)
            v = self.SlaterParameter(l,norb,[F0,F2,F4])
                
            return v
        if norb == 5:
            # F2 = 14/(1+ratio)*J
            # F4 = ratio*J
            # F0 = U
            F0 = U-8/5*J
            F2 = 49*(1/4+1/7)*J
            F4 = 63/5*J
            v = self.SlaterParameter(l,norb,[F0,F2,F4])
            return v
        
    def AngularIntegral(self,l,k,m1,m2,m3,m4):

        ang_int = 0
        pi = np.pi

        for q in range(-k,k+1):
            ang_int += gaunt(l,k,l,-m1,q,m3)*np.conjugate(gaunt(l,k,l,m4,-q,-m2))*((-1.0 if(m1+q+m2)%2 == 1 else 1.0))

        ang_int *= 4*pi/(2*k+1)

        return ang_int

    def RotationMatrix(self,l : int):

        mrange = int(2*l+1)
        R = np.zeros((mrange,mrange),dtype=np.complex64)
        
        if l == 0:
            R = np.eye(mrange,mrange,dtype=np.complex64)
        elif l == 1:
            '''/n
            py, pz, px
            '''
            R[0,0] = 1j/np.sqrt(2)
            R[2,0] = 1j/np.sqrt(2)

            R[1,1] = 1

            R[0,2] = 1/np.sqrt(2)
            R[2,2] = -1/np.sqrt(2)

        elif l==2:
            '''/n
            xy, yz, z^2, xz, x^2-y^2
            '''

            R[0,0] = 1j/np.sqrt(2)
            R[4,0] = -1j/np.sqrt(2)

            R[1,1] = 1j/np.sqrt(2)
            R[3,1] = 1j/np.sqrt(2)

            R[2,2] = 1

            R[1,3] = 1/np.sqrt(2)
            R[3,3] = -1/np.sqrt(2)

            R[0,4] = 1/np.sqrt(2)
            R[4,4] = 1/np.sqrt(2)

        elif l==3:
            '''/n
            3x^2-y^2 xyz yz^2 xz^2 z(x^2-y^2) x(x^2-3y^2)
            '''

            R[0,0] = 1j/np.sqrt(2)
            R[6,0] = 1j/np.sqrt(2)

            R[1,1] = 1j/np.sqrt(2)
            R[5,1] = -1j/np.sqrt(2)

            R[2,2] = 1j/np.sqrt(2)
            R[4,2] = 1j/np.sqrt(2)

            R[3,3] = 1

            R[2,4] = 1/np.sqrt(2)
            R[4,4] = -1/np.sqrt(2)

            R[1,5] = 1/np.sqrt(2)
            R[5,5] = 1/np.sqrt(2)

            R[0,6] = 1/np.sqrt(2)
            R[6,6] = -1/np.sqrt(2)

        return R
    
    def Spherical2Cubic(self,v : np.ndarray,l : int):
        
        
        R = self.RotationMatrix(l)
        Rdag = np.conjugate(np.transpose(R))
        
        tempmat = np.einsum("ab,cd,bdeg,ef,gh",Rdag,Rdag,v,R,R)
        tempmat = np.real(tempmat)
        
        V = np.array(tempmat,dtype=float,order='F')
        
        return V


    def GetUijklComCTQMC(self):

        pass

class CorrelationFunction(object):

    def __init__(self,latt,basisposition,ns,soc,rkgrid,orboption,N,impdict = None):
        
        self.green = None
        self.sigmah = None
        self.sigmaf = None
        self.sigmagwc = None
        self.ham = None
        self.hamtb = None
        self.hamhf = None
        self.hamqp = None
        self.occ = None
        self.vbare = None
        cry = Crystal(latt=latt,basisposition=basisposition,ns=ns,soc=soc,rkgrid=rkgrid,orboption=orboption,N=N)
        self.cry = cry

    def TighBinding(self,hoppinglist : list, onsitelist : list):

        niham = NIHamiltonian(self.cry,hoppinglist,onsitelist)
        # niham.Save(niham.hamtb,'hamtb')
        self.hamtb = niham.hamtb
        return niham.hamtb

    def HartreeFockH(self,itermax : int,mix : float, T : float, beta : float,size : int, hoppinglist : list, onsitelist : list=None, option : dict = None, intamp : list = None):

        cry = self.cry
        niham = NIHamiltonian(crystal=cry,hoppinglist=hoppinglist,onsitelist=onsitelist)
        niham.Save('hamtb')
        self.hamtb = niham.hamtb
        ft = FT_grid(T=T,beta=beta,size=size)
        vbare = VBare(crystal=cry,orboption=option,intamp=intamp)
        vbare.Save('vbare')
        self.ft = ft
        self.vbare = vbare
        self.hamtb = niham.hamtb
        self.vbare = vbare

        for iter in range(1,itermax+1):
            if iter == 1:
                hold = Hamiltonian(crystal=cry, ham=niham.hamtb, beta=ft.beta)
                hkold = None
                fkold = None
            print(hold.occ)
            hold.Save(f'hamhf.{iter}')
            sigmah = SigmaHartree(crystal=cry,occ=hold.occ,vbare=vbare.k)
            sigmah.Save(f'sigmah.{iter}')
            sigmaf = SigmaFock(crystal=cry,occr=hold.occr,vbare=vbare.r)
            sigmaf.Save(f'sigmaf.{iter}')
            sigmah.hk = sigmah.Mixing(iter,mix,sigmah.hk,hkold)
            sigmaf.fk = sigmaf.Mixing(iter,mix,sigmaf.fk,fkold)
            # print(sigmah.hk[:,:,0,0])
            # print(sigmaf.fk[:,:,0,0])
            hnew = Hamiltonian(crystal=cry,ham=self.TighBinding(hoppinglist=hoppinglist,onsitelist=onsitelist),beta=ft.beta,sigmah=sigmah,sigmaf=sigmaf)

            

            fcheck = self.FermionSCF(hnew.occk,hold.occk)
            mucheck = abs(hnew.mu-hold.mu)
            print(f" iteration : {iter} \n criteria : {fcheck} \n chemicalpotential : {hnew.mu}")
            if (fcheck <=1.0e-4)and(mucheck<=0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.ham = hnew
                self.sigmah = sigmah
                self.sigmaf = sigmaf
                hnew.Save('hamhf')
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                return hnew.hk, sigmah, sigmaf
            elif (iter==itermax):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.ham = hnew
                self.sigmah = sigmah
                self.sigmaf = sigmaf
                hnew.Save('hamhf')
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                return hnew.hk, sigmah, sigmaf
            else:
                hold = hnew
                hkold = sigmah.hk
                fkold = sigmaf.fk   
                del sigmah, sigmaf, hnew         

    def HartreeFock(self,itermax : int,mix : float, T : float, beta :float, size : int, hoppinglist : list, onsitelist : list, option : dict = None, intamp : list = None):
        
        cry = self.cry
        niham = NIHamiltonian(crystal=cry,hoppinglist=hoppinglist,onsitelist=onsitelist)
        niham.Save('hamtb')
        ft = FT_grid(T=T,beta=beta,size=size)
        gbare = GreenBare(crystal=cry,ft=ft,hamtb=niham.hamtb)
        gbare.Save('g0kf')
        vbare = VBare(crystal=cry,orboption=option,intamp=intamp)
        vbare.Save('vbarek')
        self.hamtb = niham.hamtb
        self.vbare=vbare

        for iter in range(1,itermax+1):
            if iter == 1:
                gold = GreenInt(crystal=cry,ft=ft,greenbare=gbare.g0kf)
                # sigmahold = SigmaHartree(crystal=cry,occ=gold.occ,vbare=vbare)
                # sigmafold = SigmaFock(crystal=cry,occr=gold.occr,vbare=vbare)
                hkold = None
                fkold = None
            print(gold.occ)
            gold.Save(f'gkf.{iter}')
            sigmah = SigmaHartree(crystal=cry,occ=gold.occ,vbare=vbare.k,onsite=vbare.sigmaonsiter)
            sigmah.Save(f'sigmah.{iter}')
            sigmaf = SigmaFock(crystal=cry,occr=gold.occr,vbare=vbare.r)
            sigmaf.Save(f'sigmaf.{iter}')
            hk = sigmah.Mixing(iter,1,sigmah.hk,hkold)
            fk = sigmaf.Mixing(iter,1,sigmaf.fk,fkold)
            print(sigmah.hk[:,:,0,0])
            print(hk[:,:,0,0])
            print(sigmaf.fk[:,:,0,0])
            print(fk[:,:,0,0])
            print(sigmah.hk[:,:,0,0])
            print(sigmaf.fk[:,:,0,0])
            gnew = GreenInt(crystal=cry,ft=ft,greenbare=gbare.g0kf,sigmah=hk,sigmaf=fk)
            

            fcheck = self.FermionSCF(gnew.occk,gold.occk)
            mucheck = abs(gnew.mu-gold.mu)
            print(f" iteration : {iter} \n criteria : {fcheck} \n chemicalpotential : {gnew.mu}")
            if (fcheck <=1.0e-3)and(mucheck<=0.001):
                print(f"Self-consistency is achived with {iter}-th")
                self.green = gnew
                self.sigmah = sigmah
                self.sigmaf = sigmaf
                chem = niham.ChemEmbedding(gnew.mu)
                # ham = Hamiltonian(crystal=cry,ham=self.TighBinding(hoppinglist,onsitelist),beta=ft.beta,sigmah=sigmah,sigmaf=sigmaf)
                # ham.SearchMu()
                flatstc = FLatStc(crystal=self.cry)
                self.hamhf = niham.hamtb+sigmah.hk+sigmaf.fk-chem
                gnew.Save('gkf')
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                flatstc.Save(self.hamhf,'hamhf')
                # self.hamhf = ham.hk
                break
            elif (iter==itermax):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.green = gnew
                self.sigmah = sigmah
                self.sigmaf = sigmaf
                chem = niham.ChemEmbedding(gnew.mu)
                # ham = Hamiltonian(crystal=cry,ham=self.TighBinding(hoppinglist,onsitelist),beta=ft.beta,sigmah=sigmah,sigmaf=sigmaf)
                # ham.SearchMu()
                # self.hamhf = ham.hk
                self.hamhf = niham.hamtb+sigmah.hk+sigmaf.fk-chem
                gnew.Save(gnew.gkf,'gkf')
                sigmah.Save(sigmah.hk,'sigmah')
                sigmaf.Save(sigmaf.fk,'sigmaf')
                sigmah.Save(self.hamhf,'hamhf')
            else:
                gold = gnew
                sigmahold = sigmah
                sigmafold = sigmaf
                hkold = sigmah.hk
                fkold = sigmaf.fk
                del sigmah, sigmaf, gnew
                

    def GWApproximation(self,itermax : int = None,mix : float = None, T : float= None, beta : float = None, size : int= None, hoppinglist : list= None, onsitelist : list= None, option : dict= None, intamp : list= None):

        cry = self.cry
        ft = FT_grid(T=T,beta = beta,size=size)
        vbare = VBare(crystal=cry,orboption=option,intamp=intamp)
        vbare.Save('vbarek')
        # if onsitelist == None:
        #     onsitelist = vbare.vloc.onsitelist
        # else:
        #     for i in range(len(onsitelist)):
        #         onsitelist[i] = onsitelist[i]+vbare.vloc.onsitelist[i]
        niham = NIHamiltonian(cry,hoppinglist,onsitelist)
        niham.Save('hamtb')
        gbare = GreenBare(crystal=cry,ft = ft, hamtb=niham.hamtb)
        gbare.Save('g0kf')
        
        
        for iter in range(1,itermax+1):
            if iter == 1:
                # greenold = Green(greenbare=gbare)
                gold = GreenInt(crystal=cry,ft=ft,greenbare=gbare.g0kf)
                pkfold = None
                gwckfold = None
                wold = 0
                print(gold.occ)
                print(gold.gkf[:,:,0,0,0])
            gold.Save(f'gkf.{iter}')
            print("Hartree calculation start")
            sigmah = SigmaHartree(crystal=cry,occ=gold.occ,vbare=vbare.k,onsite=None)
            sigmah.Save(f'sigmah.{iter}')
            # print(vbare.sigmaonsiter[:,:,0,0])
            print(sigmah.hr[:,:,0,0])
            print("Hartree calculation finish")
            print("Fock calculation start")
            sigmaf = SigmaFock(crystal=cry,occr=gold.occr,vbare=vbare.r)
            sigmaf.Save(f'sigmaf.{iter}')
            print(sigmaf.fr[:,:,0,0])
            print("Fock calculation finish")
            print("Polarizability calculation start")
            pol = PolLat(crystal=cry,ft=ft,green=gold.grt)
            pol.Save(pol.polkf,f'pkf.{iter}')
            pkf = pol.Mixing(iter,mix,pol.polkf,pkfold)
            print(pol.polkf[:,:,0,0,0,0])
            print(pol.polkf[:,:,0,0,0,-1])
            print("Polarizability calculation finish")
            print("Screened coulomb interaction calculation start")
            w = WLat(crystal=cry,ft=ft,pol=pkf,vbare=vbare)
            w.Save(w.wkf,f'wkf.{iter}')
            print(w.wkf[:,:,0,0,0,0])
            print(w.wckf[:,:,0,0,0,-1])
            print("Screened coulomb interaction calculation finish")
            print("GW self-energy calculation start")
            # sigmagwc = SigmaGWC(crystal=cry,ft=ft,green=gold,wlat=w)
            sigmagwc = SigmaGWC(crystal=cry,ft=ft,green=gold.grt,wlat=w.wcrt)
            sigmagwc.Save(f'sigmagwckf.{iter}')
            print(sigmagwc.kf[:,:,0,0,0])
            print("GW self-energy calculation finish")
            gwckf = sigmagwc.Mixing(iter,mix,sigmagwc.kf,gwckfold)
            print("GW green's function calculation start")
            gnew = GreenInt(crystal=cry,ft=ft,greenbare=gbare.g0kf,sigmah=sigmah.hk,sigmaf=sigmaf.fk,sigmagwc=gwckf)
            print("GW green's function calculation start")
            print(gnew.occ)
            print(gnew.gkf[:,:,0,0,0])
            # os.mkdir('gloc')
            # os.chdir('gloc')
            # np.savetxt(f'gloc.{iter}.txt',gnew.grf[0,0,0,0])
            # os.chdir('..')
            # check = self.FermionSCF(green.glatkf,greenold.glatkf)
            check = self.FermionSCF(gnew.gkf,gold.gkf)
            wcheck = self.FermionSCF(w.wkf,wold)

            print(f"iteration : {iter} \nfcriteria : {check} \nbcriteria : {wcheck} \nchemicalpotential : {gnew.mu}")

            if (check <=0.005)and(wcheck<=0.02)and(abs(gnew.mu-gold.mu)<=0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.green = gnew
                self.pol = pol
                self.w = w
                self.sigmac = sigmagwc
                self.fock = sigmaf
                self.hartree = sigmah
                gnew.Save('gkf')
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                sigmagwc.Save('sigmackf')
                pol.Save(pol.polkf,'pkf')
                w.Save(w.wkf,'wkf')
                self.sigmac.SigmaStc()
                self.sigmac.Zfactor()
                ham = Hamiltonian(crystal=cry,ham=self.TighBinding(hoppinglist=hoppinglist,onsitelist=onsitelist),sigmah=sigmah,sigmaf=sigmaf,sigmac=self.sigmac,beta=ft.beta)
                self.ham = ham
                break
            elif (iter == itermax):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.green = gnew
                self.pol = pol
                self.w = w
                self.sigmac = sigmagwc
                self.fock = sigmaf
                self.hartree = sigmah
                gnew.Save('gkf')
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                sigmagwc.Save('sigmackf')
                pol.Save(pol.polkf,'pkf')
                w.Save(w.wkf,'wkf')
                pol.Save(pol.polkf,'pkf')
                w.Save(w.wkf,'wkf')
                self.sigmac.SigmaStc()
                self.sigmac.Zfactor()
                # ham = Hamiltonian(crystal=cry,ham=self.TighBinding(hoppinglist=hoppinglist,onsitelist=onsitelist),sigmah=sigmah,sigmaf=sigmaf,sigmac=self.sigmac)
                # self.ham = ham
            else:
                gold = gnew
                gwckfold = sigmagwc.kf
                pkfold = pkf
                wold = w.wkf
                
                del sigmah,sigmaf,pol,w,sigmagwc, gnew

    def FermionSCF(self, mat1, mat2):

        check = 0
        tempmat = abs(mat1-mat2)

        check = tempmat.max()

        return check
