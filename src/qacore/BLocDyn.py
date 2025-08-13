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
# import Crystal, FTGrid
from .Crystal import Crystal
from .FTGrid import FTGrid
from .FLocDyn import FLocDyn
from .BLocStc import VLoc
from .BLatDyn import WLat
qapath = os.environ.get('QAssemble','')
sys.path.append(qapath+'/src/qacore/modules')
import QAFort

class BLocDyn(object):

    def __init__(self, crystal : Crystal, ft : FTGrid):

        self.crystal = crystal
        self.ft = ft

    def Inverse(self, matin : np.ndarray)-> np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]
        nft = len(self.ft.nu)

        matout = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')
        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex128)
        tempmat2 = np.zeros((norb*ns,norb*ns),dtype=np.complex128)

        for ift in range(nft):
            tempmat = self.crystal.OrbSpin2Composite(matin[...,ift])
            tempmat2 = np.linalg.inv(tempmat)
            matout[...,ift] = self.crystal.Composite2OrbSpin(tempmat2)
        
        return matout

    def Moment(self, bf : np.ndarray, oddzero : int, highzero : int) -> np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns

        moment = np.zeros((norb,norb,ns,ns,3),dtype=np.complex128,order='F')
        high = np.zeros((norb,norb,ns,ns),dtype=np.complex128,order='F')
        moment, high = QAFort.fourier.blocdyn_m(self.ft.nu,bf,oddzero,highzero)

        return moment,high
    
    def F2T(self,bf : np.ndarray, oddzero : int, highzero : int) -> np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nft = len(self.ft.nu)

        btau = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        moment, high = self.Moment(bf,oddzero,highzero)

        btau = QAFort.fourier.blocdyn_f2t(self.ft.nu,bf,moment,self.ft.tau)

        return btau

    def T2F(self, btau : np.ndarray) -> np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nft = len(self.ft.nu)

        bf = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        bf = QAFort.fourier.blocdyn_t2f(self.ft.tau,btau,self.ft.nu)

        return bf

    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]

        ynew = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')
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

        Bnew = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        if iter == 1:
            mix = 1.0
            Bold = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        Bnew = mix*Bb + (1-mix)*Bold

        return Bnew
    
    def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]
        nft = matimp.shape[3]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,ns,nft,nspace),dtype=np.complex128,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            for ispace in val:
                matloc[...,ispace] = matimp[...,iprob]

        return matloc
    
    def Loc2Imp(self,matloc : np.ndarray)->np.ndarray:

        nprob = len(self.crystal.probspace)
        norb = matloc.shape[0]
        ns = matloc.shape[2]
        nft = matloc.shape[4]

        matimp = np.zeros((norb,norb,ns,ns,nft,nprob),dtype=np.complex128,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            tempmat = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128)
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

        matout = np.zeros((norb,norb,ns,ns,nfreq),dtype=np.complex128,order='F')
        nind = np.amax(equiv)

        for js in range(ns):
            for ks in range(ns):
                for ind in range(nind):
                    pos = self.crystal.FindPositions(equiv,ind+1)
                    for ii, jj in pos:
                        matout[ii,jj,js,ks] = matdict[str(ind+1)]

        return matout

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = mat1.shape[1]
        ns = self.crystal.ns
        nft = len(self.ft.nu)

        matout = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        matout = QAFort.dyson.blocdyn(mat1,mat2)

        return matout

    def Embedding(self, matin : np.ndarray):

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.ft.nu)
        nspace = self.crystal.bprojector.shape[3]

        matout = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex128,order='F')

        for ispace in range(nspace):
            matout += QAFort.embedding.blocdyn(nrk,matin[...,ispace],self.crystal.bprojector[...,ispace])

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
    
    def StcEmbedding(self,matin : np.ndarray)->np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]
        # nrk = matin.shape[4]
        nft = len(self.ft.nu)#self.ft.size

        matout = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            matout[...,ift] += matin
        # del matin
        # gc.collect()
        return matout


    def Double2Full(self,matin : np.ndarray)->np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        # nrk = len(self.crystal.kpoint)
        nft = len(self.ft.nu)#self.ft.size

        matout = np.zeros((norb*norb,norb*norb,ns,ns,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            for js in range(ns):
                for ks in range(ns):
                    matout[:,:,js,ks,ift] = self.crystal.Double2Full(matin[:,:,js,ks,ift])
        del matin
        # gc.collect()
        return matout

    def Full2Double(self, matin : np.ndarray) -> np.ndarray:

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        # nrk = len(self.crystal.kpoint)
        nft = len(self.ft.nu)#self.ft.size

        matout = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            for js in range(ns):
                for ks in range(ns):
                    matout[:,:,js,ks,ift] = self.crystal.Full2Double(matin[:,:,js,ks,ift])

        return matout



class PolLoc(BLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, green, hdf5file : str = 'glob.h5', group :str = None):
        super().__init__(crystal, ft)
        
        self.rt = None # rt to kf
        self.rf = None
        # self.kt = None
        # self.kf = None
        nprob = len(self.crystal.probspace)

        ##########################################
        ### not sure what they are used for
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        if green is None:
            print("Error, There is no Green's function.")
            sys.exit()
        ##########################################

        self.green = green

        self.Cal()

        #self.kt = self.R2K(self.rt)
        #self.kf = self.T2F(self.kt)

    def Cal(self):
        #pass
        
        grt = self.green.gt

        ##########################################
        ##### taken from class GreenLoc(FlocDyn)
        norb = self.crystal.bprojector.shape[1]
        ns = self.crystal.ns
        # nft = self.ft.size
        ntau = len(self.ft.tau)
        nft=len(self.ft.nu)
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)
        ##########################################

        # polrt = np.zeros((norbc,norbc,ns,ns,ntau,nspace),dtype=np.complex128,order='F')
        polrt = np.zeros((norb,norb,ns,ns,ntau,nprob),dtype=np.complex128,order='F')
        
        gmrt=np.empty_like(grt)
        for iprob in range(nprob):
            gmrt[...,iprob] = self.crystal.T2mT_loc(grt[...,iprob])

        # print(gmrt.shape)
        # print(grt.shape)

        
        if ns == 2:
            for itau in range(ntau):
                for iprob in range(nprob):
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
                                        polrt[iorb,jorb,js,ks,itau,iprob] = gmrt[korbc,iorbc,js,itau,iprob]*grt[lorbc,jorbc,ks,itau,iprob]
        else:
            if self.crystal.soc == True:
                C = 1
                for itau in range(ntau):
                    for iprob in range(nprob):
                        for iorb in range(norb):
                            [a,[m1,m3]] = self.crystal.BAtomOrb(iorb)
                            iorbc = self.crystal.FIndex([a,m1])
                            korbc = self.crystal.FIndex([a,m3])
                            for jorb in range(norb):
                                [b,[m4,m2]] = self.crystal.BAtomOrb(jorb)
                                lorbc = self.crystal.FIndex([b,m4])
                                jorbc = self.crystal.FIndex([b,m2])
                                polrt[iorb,jorb,0,0,itau,iprob] = gmrt[jorbc,iorbc,0,itau,iprob]*grt[korbc,lorbc,0,itau,iprob]*C
            else:
                C = 2
                for itau in range(ntau):
                    for iprob in range(nprob):
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
                                polrt[iorb,jorb,0,0,itau,iprob] = gmrt[jorbc,iorbc,0,itau,iprob]*grt[korbc,lorbc,0,itau,iprob]*C

        self.rt = polrt


        self.rf=np.zeros((norb,norb,ns,ns,nft,nprob),dtype=np.complex128,order='F')
        for iprob in range(nprob):
            self.rf[...,iprob] = self.T2F(self.rt[...,iprob])
		                    
		                    

class PolImp(BLocDyn): # read Polarizability from CTQMC

    def __init__(self, crystal: Crystal, ft: FTGrid):
        super().__init__(crystal, ft)

        pass

class WLoc(BLocDyn): #### contains WLoc and WcLoc

    def __init__(self, crystal: Crystal, ft: FTGrid
    ,pol : PolLoc = None, vLoc : VLoc = None, vDyn : np.array = None, c : float = 1.0, hdf5file : str = 'glob.h5', group : str = None):
    # def __init__(self, crystal: Crystal, ft: FTGrid ,wlat : WLat = None, hdf5file : str = 'glob.h5', group : str = None):
        super().__init__(crystal, ft)

        # pass
        self.rt = None #rt to kf
        self.rf = None
        # self.kt = None
        # self.kf = None
        self.crt = None #rt to kf
        self.crf = None
        # self.ckt = None
        # self.ckf = None
        # self.c = c #### ??

        # self.wlat = wlat


        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__


        # if pol is None:
        #     print("Error, polarizability doesn't exist")
        #     sys.exit()
        # if vLoc is None:
        #     print("Error, bare coulomb interaction doesn't exist")
        #     sys.exit()
        # self.pol = pol
        # self.vDyn = vDyn
        # self.vLoc = vLoc



        self.Cal()

        # self.wkt = self.F2T(self.wkf,1,1)
        # self.wrf = self.K2R(self.wkf)
        # self.wrt = self.K2R(self.wkt)
        # nprob = len(self.crystal.probspace)
        
        # self.crt = self.F2T(self.crf,1,1)
        # self.crf = self.K2R(self.ckf)
        # self.crt = self.K2R(self.ckt)
    
    def Cal(self): # calculate W and Wc

        norb = self.crystal.bprojector.shape[1]
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        # nft = self.ft.size
        ntau = len(self.ft.tau)
        nfreq = len(self.ft.nu)
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)



        # wrf_nspace  = np.zeros((norb,norb,ns,ns,nfreq,nspace),dtype=np.complex128,order='F')

        # wrf_nspace = self.wlat.Projection(self.wlat.kf)

        # self.rf = self.Loc2Imp(wrf_nspace)

        # rt = np.zeros((norb,norb,ns,ns,ntau,nprob),dtype=np.complex128,order='F')
        # for iprob in range(nprob):
        #     rt[...,iprob] = self.F2T(self.rf[...,iprob], 1, 1)
        # self.rt = np.copy(rt)

        


        # crf = self.wlat.Projection(self.wlat.ckf)
        # self.crf = self.Loc2Imp(crf)

        # crt = np.zeros((norb,norb,ns,ns,ntau,nprob),dtype=np.complex128,order='F')
        # for iprob in range(nprob):
        #     crt[...,iprob] = self.F2T(self.crf[...,iprob], 1, 1)
        # self.crt = np.copy(crt)




        
        ####### Initialization #######
        tempmat = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        wrf  = np.zeros((norb,norb,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        wcrf = np.zeros((norb,norb,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        vdyn = np.zeros((norb,norb,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')

        for iprob in range(nprob):
            vdyn[...,iprob] = self.StcEmbedding(self.vLoc[...,iprob]) ####  define StcEmbedding ### remove this part when it is dynamic
        
        print(self.pol.shape)
        print(vdyn.shape)
        
        polcomp = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        vcomp = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        ###### Initialization #######
        polcomp = self.Loc2Imp(self.pol)*self.c #### ??
        # del self.pol
        vcomp = self.Loc2Imp(vdyn) #### ??

        for iprob in range(nprob):
            wrf[...,iprob] = self.Dyson(self.vDyn[...,iprob],self.pol[...,iprob])
        # wrf = self.Imp2Loc(tempmat)

        ## tempmat -> wrf


        #### W
        self.rf = wrf
        rt = np.zeros((norb,norb,ns,ns,ntau,nprob),dtype=np.complex128,order='F')
        for iprob in range(nprob):
            rt[...,iprob] = self.F2T(wrf[...,iprob], 1, 1)
        self.rt = np.copy(rt)


        #### Wc
        wcrf = wrf - self.vLoc  #### wrf - vloc --> static bare V
        self.crf = wcrf
        crt = np.zeros((norb,norb,ns,ns,ntau,nprob),dtype=np.complex128,order='F')
        for iprob in range(nprob):
            crt[...,iprob] = self.F2T(wcrf[...,iprob], 1, 1)
        self.crt = crt




        # for iprob in range(nprob):
        #     wrf[...,iprob] = self.Dyson(self.vDyn[...,iprob],self.pol[...,iprob])
        # # wrf = self.Imp2Loc(tempmat)

        # ## tempmat -> wrf

        # self.rf = np.copy(wrf)
        


        # for iprob in range(nprob):
        #     for ifreq in range(nfreq):
        #         for iis in range(ns):
        #             for jjs in range(ns):
        #                 wcrf[...,iis,jjs,ifreq,iprob] = wrf[...,iis,jjs,ifreq,iprob] - self.vLoc[...,iis,jjs,iprob]  #### wrf - vloc --> static bare V
        

        # self.crf = np.copy(wcrf)
        # crt = np.zeros((norb,norb,ns,ns,ntau,nprob),dtype=np.complex128,order='F')
        # for iprob in range(nprob):
        #     crt[...,iprob] = self.F2T(wcrf[...,iprob], 1, 1)
        # self.crt = np.copy(crt)
        # del wrf, wcrf, crt
        return None



class WLoc_temp(BLocDyn): #### contains WLoc and WcLoc

    def __init__(self, crystal: Crystal, ft: FTGrid
    ,pol : PolLoc = None, vLoc : VLoc = None, c : float = 1.0, hdf5file : str = 'glob.h5', group : str = None):
        super().__init__(crystal, ft)

        # pass
        self.rt = None #rt to kf
        self.rf = None
        # self.kt = None
        # self.kf = None
        self.crt = None #rt to kf
        self.crf = None
        # self.ckt = None
        # self.ckf = None
        self.c = c #### ??
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        if pol is None:
            print("Error, polarizability doesn't exist")
            sys.exit()
        if vLoc is None:
            print("Error, bare coulomb interaction doesn't exist")
            sys.exit()
        self.pol = pol
        self.vLoc = vLoc

        self.Cal()

        # self.wkt = self.F2T(self.wkf,1,1)
        # self.wrf = self.K2R(self.wkf)
        # self.wrt = self.K2R(self.wkt)
        # nprob = len(self.crystal.probspace)
        
        # self.crt = self.F2T(self.crf,1,1)
        # self.crf = self.K2R(self.ckf)
        # self.crt = self.K2R(self.ckt)
    
    def Cal(self): # calculate W and Wc

        norb = self.crystal.bprojector.shape[1]
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        # nft = self.ft.size
        ntau = len(self.ft.tau)
        nfreq = len(self.ft.nu)
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)

        
        ####### Initialization #######
        tempmat = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        wrf  = np.zeros((norb,norb,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        wcrf = np.zeros((norb,norb,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        vdyn = np.zeros((norb,norb,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        polcomp = np.zeros((norb,norb,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        for iprob in range(nprob):
            vdyn[...,iprob] = self.StcEmbedding(self.vLoc[...,iprob]) ####  define StcEmbedding ### remove this part when it is dynamic
            # polcomp[...,iprob] = self.Double2Full(self.pol[...,iprob])*self.c
        # print(self.pol.shape)
        # print(vdyn.shape)
        
        # polcomp = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        # vcomp = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        ####### Initialization #######
        # polcomp = self.Loc2Imp(self.pol)*self.c #### ??
        # # del self.pol
        # vcomp = self.Loc2Imp(vdyn) #### ??
        
        


        # for iprob in range(nprob):
        #     tempmat[...,iprob] = self.Dyson(vdyn[...,iprob],polcomp[...,iprob])

        # for iprob in range(nprob):
        #     wrf[...,iprob] = self.Full2Double(tempmat[...,iprob])

        for iprob in range(nprob):
            wrf[...,iprob] = self.Dyson(vdyn[...,iprob],self.pol[...,iprob])
        # wrf = self.Imp2Loc(tempmat)

        ## tempmat -> wrf

        self.rf = np.copy(wrf)
        rt = np.zeros((norb,norb,ns,ns,ntau,nprob),dtype=np.complex128,order='F')
        for iprob in range(nprob):
            rt[...,iprob] = self.F2T(wrf[...,iprob], 1, 1)
        self.rt = np.copy(rt)



        wcrf = wrf - vdyn  #### wrf - vloc --> static bare V
        self.crf = np.copy(wcrf)
        crt = np.zeros((norb,norb,ns,ns,ntau,nprob),dtype=np.complex128,order='F')
        for iprob in range(nprob):
            crt[...,iprob] = self.F2T(wcrf[...,iprob], 1, 1)
        self.crt = np.copy(crt)
        # del wrf, wcrf, vdyn, crt
        return None

class WImp(BLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
        super().__init__(crystal, ft, flocdyn)

        pass

class WcLoc(BLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
        super().__init__(crystal, ft, flocdyn)

        pass

class WcImp(BLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
        super().__init__(crystal, ft, flocdyn)

        pass



class UImp(BLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, wloc : WLoc, ploc : PolLoc, vloc : np.array):
        super().__init__(crystal, ft)

        self.utilde_rt = None
        self.utilde_rf = None
        self.ubar_rt = None
        self.ubar_rf = None

        self.wloc = wloc
        self.ploc = ploc
        self.vloc = vloc

        self.Cal()
    

    def Cal(self):
        norb = self.crystal.bprojector.shape[1]
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        ntau = len(self.ft.tau)
        nfreq = len(self.ft.nu)
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)

        IdendityMatrix = np.identity(norb)


        uinv_rf_temp = np.zeros((norb,norb),dtype=np.complex128,order='F')
        self.utilde_rf = np.zeros((norb,norb,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')
        self.ubar_rf = np.zeros((norb,norb,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')

        # temp = np.zeros((norb,norb,ns,ns,nfreq),dtype=np.complex128,order='F')
        # for iprob in range(nprob):
        #     temp = self.Inverse(self.wloc[...,iprob]) + self.ploc[..., iprob]
        #     self.utilde_rf[...,iprob] = self.Inverse(temp)
        # # for iprob in range(nprob):
        # #     for ifreq in range(nfreq):
        # #         for iis in range(ns):
        # #             for jjs in range(ns):
        # #                 uinv_rf_temp[...] = temp[...,iis,jjs,ifreq,iprob] + self.ploc[...,iis,jjs,ifreq,iprob]
        # #                 self.utilde_rf[...,iis,jjs,ifreq,iprob] = np.linalg.inv(uinv_rf_temp[...])

        # self.ubar_rf = self.utilde_rf - self.vloc

        temp0 = np.zeros((norb,norb,ns,ns,nfreq),dtype=np.complex128,order='F')
        temp  = np.zeros((norb,norb,ns,ns,nfreq,nprob),dtype=np.complex128,order='F')

        for iprob in range(nprob):
            for ifreq in range(nfreq):
                for iis in range(ns):
                    for jjs in range(ns):
                        temp0[...,iis,jjs,ifreq] = IdendityMatrix + np.dot(self.wloc[...,iis,jjs,ifreq,iprob],self.ploc[...,iis,jjs,ifreq,iprob])
            temp[...,iprob] = self.Inverse(temp0)

        for iprob in range(nprob):
            for ifreq in range(nfreq):
                for iis in range(ns):
                    for jjs in range(ns):
                        # temp = IdendityMatrix + np.dot(self.wloc[...,iis,jjs,ifreq,iprob],self.ploc[...,iis,jjs,ifreq,iprob])
                        # self.utilde_rf[...,iis,jjs,ifreq,iprob] = np.dot(np.linalg.inv(temp),self.wloc[...,iis,jjs,ifreq,iprob])
                        self.utilde_rf[...,iis,jjs,ifreq,iprob] = np.dot(temp[...,iis,jjs,ifreq,iprob],self.wloc[...,iis,jjs,ifreq,iprob])
                        self.ubar_rf[...,iis,jjs,ifreq,iprob] = self.utilde_rf[...,iis,jjs,ifreq,iprob] - self.vloc[...,iis,jjs,iprob]
        
        # self.ubar_rf = self.utilde_rf - self.vloc

        self.ubar_rt = np.zeros((norb,norb,ns,ns,ntau,nprob),dtype=np.complex128,order='F')
        self.utilde_rt = np.zeros((norb,norb,ns,ns,ntau,nprob),dtype=np.complex128,order='F')
        for iprob in range(nprob):
            self.ubar_rt[...,iprob] = self.F2T(self.ubar_rf[...,iprob], 1, 1)
            self.utilde_rt[...,iprob] = self.F2T(self.utilde_rf[...,iprob], 1, 1)
        
