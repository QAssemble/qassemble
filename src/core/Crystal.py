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
diage_path = os.environ.get('DIAGE','')
path = diage_path+"/modules"
sys.path.append(path)
import DiagE

# Ask to professor for change variables
class Crystal(object): # chemical potential object, num of electron 
    def __init__(self, Rvec : list = None, CorF : str = "F", Basis : list = None, Nspin : int = None, SOC : bool = False, Nelec : float = None, KGrid : list = None) -> object:
        
        '''
        Construct Crystal class for preparing calculation. from the input parameter generate the fermionic orbital index, bosonic orbital index, discrete k-point, a-vector, b-vector

        latt, basisposition, soc, rkgrid, orboption, N -> Crystal()
        '''
        latt = np.array(Rvec,dtype=float)
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
        pos = []
        orboption = {}
        for i, ii in enumerate(Basis):
            pos.append(ii[0])
            orboption[i+1] = ii[1]
        # if basisposition["CorF"] == "C":
        #     pos = np.array(basisposition["pos"])
        #     lat = Lattice.from_parameters(np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(c),alpha,beta,gamma)
        #     structure = Structure(lat,["X"]*len(pos),pos,coords_are_cartesian=True)
        #     structurebasisc = []
        #     structurebasisf = []
        #     for site in structure.sites:
        #         structurebasisc.append(site.coords.tolist())
        #         structurebasisf.append(site.frac_coords.tolist())
        #     print(structure)
        # if basisposition["CorF"] == "F":
        #     pos = np.array(basisposition['pos'])
        #     lat = Lattice.from_parameters(np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(c),alpha,beta,gamma)
        #     structure = Structure(lat,["X"]*len(pos),pos)
        #     structurebasisc = []
        #     structurebasisf = []
        #     for site in structure.sites:
        #         structurebasisc.append(site.coords.tolist())
        #         structurebasisf.append(site.frac_coords.tolist())
        #     print(structure)
        if CorF == "C":
            pos = np.array(pos)
            lat = Lattice.from_parameters(np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(c),alpha,beta,gamma)
            structure = Structure(lat,["X"]*len(pos),pos,coords_are_cartesian=True)
            structurebasisc = []
            structurebasisf = []
            for site in structure.sites:
                structurebasisc.append(site.coords.tolist())
                structurebasisf.append(site.frac_coords.tolist())
            print(structure)
        elif CorF == "F":
            pos = np.array(pos)
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
        self.ns = Nspin
        self.soc = SOC
        self.nume = Nelec*(Nspin/2)
        # self.basisf = tempmat
        # self.basisc = np.dot(self.basisf,self.avec)
        self.bvec = np.zeros((3,3))
        self.vol=np.dot(np.cross(latt[:,0], latt[:,1]), latt[:,2])
        self.bvec[:,0]=2*np.pi*np.cross(latt[:,1], latt[:,2])/self.vol
        self.bvec[:,1]=2*np.pi*np.cross(latt[:,2], latt[:,0])/self.vol
        self.bvec[:,2]=2*np.pi*np.cross(latt[:,0], latt[:,1])/self.vol
        
        self.kpath = None
        self.rkgrid = KGrid
        nk = KGrid[0]*KGrid[1]*KGrid[2]
        kpoint_temp=np.array(list(itertools.product(np.linspace(0,1,num=KGrid[2],endpoint=False),np.linspace(0,1,num=KGrid[1],endpoint=False),np.linspace(0,1,num=KGrid[0],endpoint=False))))
        kpoint=np.fliplr(kpoint_temp)
        self.kpoint = kpoint
        self.nk = nk

        self.rvec = None
        self.kpath = None
        self.kdist = None
        self.knode = None

        self.find = {}
        self.bind = {}
        self.full = {}
        self.pbasis = None
        self.bbasis = None
        # self.b2f = None
        self.c2b = None
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
        self.Boson2Fermion()
        self.SetFullBasis()
        self.Boson2Full()
        # if impdict != None:
        #     self.Projector(impdict)

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
                # self.Boson2Fermion(iorb)
            # self.Composite2Boson()
            # self.Composite2Fermion()
            
    
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
            
    def Boson2Fermion(self):
        '''
        Mapping with boson index to fermion index
        '''
        norbc = len(self.find)
        bbasis = np.zeros((norbc,norbc),dtype=int)
        # [a, [m1,m2]] = self.BAtomOrb(ind)
        # iorbc1 = self.FIndex([a,m1])
        # iorbc2 = self.FIndex([a,m2])
        # self.b2f.append([iorbc1,iorbc2])
        for jorbc in range(norbc):
            for iorbc in range(norbc):
                [a,m] = self.FAtomOrb(iorbc)
                [ap,mp] = self.FAtomOrb(jorbc)
                if (a==ap):
                    iorb = self.BIndex([a,[m,mp]])
                    bbasis[iorbc,jorbc] = iorb

        self.bbasis = bbasis

        return None

    def Boson2Full(self):
        
        norb = len(self.bind)
        c2b = np.zeros((norb),dtype=int)

        for iorb in range(norb):
            [a,[m,mp]] = self.BAtomOrb(iorb)
            iorbc = self.FIndex([a,m])
            jorbc = self.FIndex([a,mp])
            ind = self.pbasis[iorbc,jorbc]
            c2b[iorb] = ind
        
        self.c2b = c2b
    
    def SetFullBasis(self):

        norbc = len(self.find)
        full = {}
        pbasis = np.zeros((norbc,norbc),dtype=int)

        for jorbc in range(norbc):
            for iorbc in range(norbc):
                (a,m1) = self.FAtomOrb(iorbc)
                (b,m2) = self.FAtomOrb(jorbc)
                nn = [iorbc,jorbc]
                ind, nn = self.indexing(norbc*norbc,2,[norbc,norbc],1,0,nn)
                full[ind] = [[a,m1],[b,m2]]
                pbasis[iorbc,jorbc] = ind

        self.pbasis = copy.deepcopy(pbasis)
        self.full = copy.deepcopy(full)
        
        return None
    
    def FullIndex(self,val : list):

        for k, v in self.full.items():
            if v == val:
                return k
        
    def FullAtomOrb(self, ind : int):

        return self.full[ind]

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
        norbc = len(self.find)
        matret = np.zeros((norb,norb),dtype=np.complex64)

        for lorbc in range(norbc):
            for korbc in range(norbc):
                for jorbc in range(norbc):
                    for iorbc in range(norbc):
                        (a,m1) = self.FAtomOrb(iorbc)
                        (ap,m4) = self.FAtomOrb(lorbc)
                        (b,m2) = self.FAtomOrb(jorbc)
                        (bp,m3) = self.FAtomOrb(korbc)
                        if (a==ap)and(b==bp):
                            iorb = self.BIndex([a,[m1,m2]])
                            jorb = self.BIndex([b,[m2,m3]])
                            matret[iorb,jorb] = mat[iorbc,jorbc,korbc,lorbc]

        return matret
    
    def Double2Quad(self, mat : np.ndarray) -> np.ndarray:

        norbc = len(self.find)
        norb = len(self.bind)

        matret = np.zeros((norbc,norbc,norbc,norbc),dtype=np.complex64,order='F')

        for jorb in range(norb):
            for iorb in range(norb):
                [a,[m1,m4]] = self.BAtomOrb(iorb)
                [b,[m2,m3]] = self.BAtomOrb(jorb)
                iorbc = self.FIndex([a,m1])
                lorbc = self.FIndex([a,m4])
                jorbc = self.FIndex([b,m2])
                korbc = self.FIndex([b,m3])
                matret[iorbc,jorbc,korbc,lorbc] = mat[iorb,jorb]

        return matret

    def Full2Quad(self,mat : np.ndarray) -> np.ndarray:

        norbc = len(self.find)

        matret = np.zeros((norbc,norbc,norbc,norbc),dtype=np.complex64,order='F')
        
        for lorbc in range(norbc):
            for korbc in range(norbc):
                for jorbc in range(norbc):
                    for iorbc in range(norbc):
                        iorb = self.pbasis[iorbc,lorbc]
                        jorb = self.pbasis[jorbc,korbc]
                        matret[iorbc,jorbc,korbc,lorbc] = mat[iorb,jorb]

        
        return matret
    
    def Quad2Full(self, mat : np.ndarray) -> np.ndarray:

        norbc = len(self.find)

        matret = np.zeros((norbc**2,norbc**2))

        for lorbc in range(norbc):
            for korbc in range(norbc):
                for jorbc in range(norbc):
                    for iorbc in range(norbc):
                        iorb = self.pbasis[iorbc,lorbc]
                        jorb = self.pbasis[jorbc,korbc]
                        matret[iorb,jorb] = mat[iorbc,jorbc,korbc,lorbc]
        
        return matret
    
    def Full2Double(self, mat : np.ndarray) -> np.ndarray:

        norb = len(self.bind)

        matret = np.zeros((norb,norb),dtype=np.complex64,order='F')

        for jorb in range(norb):
            for iorb in range(norb):
                ind1 = self.c2b[iorb]
                ind2 = self.c2b[jorb]
                matret[iorb,jorb] = mat[ind1,ind2]
        
        return matret
    
    def Double2Full(self, mat : np.ndarray) -> np.ndarray:

        nind = len(self.find)**2
        norb = len(self.bind)
        matret = np.zeros((nind,nind),dtype=np.complex64,order='F')

        for jorb in range(norb):
            for iorb in range(norb):
                ind1 = self.c2b[iorb]
                ind2 = self.c2b[jorb]
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
        # for iz in range(self.rkgrid[2]//2 +1):
        #     for iy in range(self.rkgrid[1]//2 + 1):
        #         for ix in range(self.rkgrid[0]//2+1):
        #             nn1 = [ix,iy,iz]
        #             ind1,nn1 = self.indexing(self.rkgrid[0]*self.rkgrid[1]*self.rkgrid[2],3,self.rkgrid,1,0,nn1)
        #             r[ind1] = nn1
        #             if (nn1==[0,0,0]):
        #                 continue
        #             ii = (self.rkgrid[0]-ix) % self.rkgrid[0]
        #             jj = (self.rkgrid[1]-iy) % self.rkgrid[1]
        #             kk = (self.rkgrid[2]-iz) % self.rkgrid[2]
        #             nn2 = [ii,jj,kk]
        #             ind2,nn2 = self.indexing(self.rkgrid[0]*self.rkgrid[1]*self.rkgrid[2],3,self.rkgrid,1,0,nn2)
        #             r[ind2] = [-ix,-iy,-iz]

        for iz in range(self.rkgrid[2]):
            for iy in range(self.rkgrid[1]):
                for ix in range(self.rkgrid[0]):
                    nn1 = [ix,iy,iz]
                    ind1, nn1 = self.indexing(self.rkgrid[0]*self.rkgrid[1]*self.rkgrid[2],3,self.rkgrid,1,0,nn1)
                    if (ix > self.rkgrid[0]//2):
                        xx = ix-self.rkgrid[0]
                    else:
                        xx = ix
                    if (iy > self.rkgrid[1]//2):
                        yy = iy-self.rkgrid[1]
                    else:
                        yy = iy
                    if (iz > self.rkgrid[2]//2):
                        zz = iz-self.rkgrid[2]
                    else:
                        zz = iz
                    r[ind1] = [xx,yy,zz]

        self.rvec = r

        return None