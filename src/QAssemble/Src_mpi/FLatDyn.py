import string as string
import re as re
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import matplotlib.font_manager as fm
from collections import OrderedDict
import os, sys
import scipy.optimize
import copy
import h5py
from .Crystal import Crystal
from .FTGrid import FTGrid
from .FLatStc import FLatStc
from .MPIManager import MPIManager
from .modules.Dyson import Dyson
from .modules.Fourier import Fourier
from .modules.Bare import Bare
from .modules.Common import Common


class FLatDyn(object):

    def __init__(self, crystal : Crystal, ftgrid : FTGrid, nk : int, nw : int, ntau : int, nprock : int, nprocw : int, mpimanager : MPIManager):

        self.crystal = crystal
        self.ftgrid = ftgrid
        self.nk = nk
        self.nw = nw
        self.nprock = nprock
        self.nprocw = nprocw
        self.mpimanager = mpimanager
        self.nodedict = mpimanager.Quary(nk, nw, ntau, nprock, nprocw, self.crystal.rkgrid)

        self.commk = self.nodedict['commk']
        self.commw = self.nodedict['commf']
        self.submatrixkf = self.nodedict['submatrixkf']
        self.submatrixkb = self.nodedict['submatrixkb']
        self.submatrixw = self.nodedict['submatrixw']

        self.commtau = self.nodedict['commtau']

        self.submatrixtau = self.nodedict['submatrixtau']

    def CheckGroup(self, filepath : str, group : str):

        with h5py.File(filepath, 'r') as file:
            return group in file

    def Save(self, hdf5file : str = None, group : str = None, subgroup : str = None, data : np.ndarray = None, dataname : str = None):

        file = h5py.File(hdf5file, 'a', driver='mpio', comm = self.mpimanager.comm)
        
        if group in file:
            g =  file[group]
        else:
            g = file.create_group(group)
        
        if subgroup in g:
            subg = g[subgroup]
        else:
            subg = g.create_group(subgroup)

        
        print('Saving data...')
        rankf = self.nodedict['commfrank']
        rankk = self.nodedict['commkrank']

        nfreq = len(self.mpimanager.floc[rankf])
        nk = len(self.mpimanager.klocal[rankk])
        n4 = data.shape[3]
        n5 = data.shape[4]
        if (nk != n4) or (nfreq != n5):
            print('Data mismatch')
            sys.exit()
        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(self.crystal.ns):
                    fidx = self.mpimanager.FLocal2Global([rankf, ifreq])
                    kidx = self.mpimanager.KLocal2Global([rankk, ik])
                    name = f"{dataname}_w_{fidx+1}_k_{kidx+1}_s_{js+1}"
                    print(f"Saving {name} data to {hdf5file}")
                    subg.create_dataset(name, data=data[:, :, js, ik, ifreq], dtype=np.complex128)
        
        print('Saving data finish')
    
        # self.mpimanager.comm.Barrier()
        file.close()

        return None

    def Load(self, hdf5file : str = None, group : str = None, subgroup : str = None, data : np.ndarray = None, dataname : str = None):

        file = h5py.File(hdf5file, 'r', driver='mpio', comm=self.mpimanager.comm)
        try:
            if group in file:
                g =  file[group]
                if subgroup in g:
                    subg = g[subgroup]
                    if dataname in subg:
                        data = subg[dataname][:]
                        return data
                    else:
                        raise KeyError(f"{dataname} not found in {subgroup}")
                else:
                    raise KeyError(f"{subgroup} not found in {group}")
            else:
                raise KeyError(f"{group} not found in {hdf5file}")
        finally:
            self.mpimanager.comm.Barrier()
            file.close()

    def Inverse(self, matin : np.ndarray) -> np.ndarray:

        
        norb, _, ns, nk, nft = matin.shape

        matout = np.zeros((norb, norb, ns, nk, nft), dtype=np.complex128, order='F')
        

        for ift in range(nft):
            for ik in range(nk):
                for js in range(ns):
                    matout[:, :, js, ik, ift] = Common.MatInv(matin[:, :, js, ik, ift])

        return matout

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:
    
        matout = Dyson.FLatDyn(mat1, mat2)

        return matout
    
    def ChemEmbedding(self,mu : float) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.ftgrid.omega)#self.ft.size

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
        nft = len(self.ftgrid.omega)#self.ft.size

        matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            matout[...,ift] = matin

        return matout

    def K2R(self, matk : np.ndarray) -> np.ndarray:

        norb, _, ns, nk, nf = matk.shape
        rkvec = self.crystal.kpoint
        rank = self.nodedict['commkrank']
        (nkx, nky, nkz) = self.mpimanager.localshapef[self.nodedict['commkrank']]
        if (nk != nkx * nky * nkz):
            raise ValueError(f"Error: nk ({nk}) does not match local shape ({nkx}, {nky}, {nkz})")        
        nr = len(self.mpimanager.rlocal[rank])
        matr = np.zeros((norb, norb, ns, nr, nf), dtype=np.complex128, order='F')
        tempmat = np.zeros((norb, norb, ns, nk, nf), dtype=np.complex128, order='F')

        for iff in range(nf):            
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        for ik in range(nk):
                            a, _ = self.crystal.FAtomOrb(iorb)
                            b, _ = self.crystal.FAtomOrb(jorb)
                            delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                            kidx = self.mpimanager.KLocal2Global([rank, ik])
                            phase = np.exp(2.0j * np.pi * np.dot(rkvec[kidx], delta))
                            tempmat[iorb, jorb, js, ik, iff] = matk[iorb, jorb, js, ik, iff] * phase

        matr = Fourier.FLatDynK2R(self.commk, tempmat, self.mpimanager)

        return matr
    

    def R2K(self, matr : np.ndarray) -> np.ndarray:

        

        norb, _, ns, nr, nf = matr.shape
        rank = self.nodedict['commkrank']
        rkvec = self.crystal.kpoint
        (nx, ny, nz) = self.mpimanager.localshapeb[rank]
        nk = len(self.mpimanager.klocal[rank])
        if (nr != nx * ny * nz):
            print(f"Error: nk ({nr}) does not match local shape ({nx}, {ny}, {nz})")
            sys.exit()
        
        matk = np.zeros((norb, norb, ns, nk, nf), dtype=np.complex128, order='F')

        tempmat = Fourier.FLatDynR2K(self.commk, matr, self.mpimanager)

        for iff in range(nf):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        for ik in range(nk):
                            a, _ = self.crystal.FAtomOrb(iorb)
                            b, _ = self.crystal.FAtomOrb(jorb)
                            delta = self.crystal.basisf[a, :] - self.crystal.basisf[b, :]
                            kidx = self.mpimanager.KLocal2Global([rank, ik])
                            phase = np.exp(-2.0j * np.pi * np.dot(rkvec[kidx], delta))
                            matk[iorb, jorb, js, ik, iff] = tempmat[iorb, jorb, js, ik, iff] * phase

        return matk
    
    def Moment(self, ff : np.ndarray, isgreen : bool, highzero : bool) -> np.ndarray:



        norb, _, ns, nkloc, _ = ff.shape 
        omega = self.ftgrid.omega*1j
        moment = np.zeros((norb, norb, ns, nkloc, 3), dtype=np.complex128, order='F')
        high = np.zeros((norb, norb, ns, nkloc), dtype=np.complex128, order='F')
        

        fflast = self.mpimanager.FMPIBCast(self.commw, ff, len(omega)-1)
        fflast2 = self.mpimanager.FMPIBCast(self.commw, ff, len(omega)-2)

        moment, high = Fourier.FLatDynM(self.ftgrid.omega, fflast, fflast2, isgreen, highzero)

        return moment, high
    
    def F2T(self, ff : np.ndarray, isgreen : bool, highzero : bool):

        rank = self.nodedict['commtaurank']
        ntauloc = self.submatrixtau[rank][1] - self.submatrixtau[rank][0]
        tau = np.zeros((ntauloc), dtype=np.float64, order='F')
        for itau in range(ntauloc):
            tauidx = self.mpimanager.TLocal2Global([rank, itau])
            tau[itau] = self.ftgrid.tau[tauidx]
        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]
        nomega = len(self.ftgrid.omega)
        ftau = np.zeros((norb, norb, ns, nk, ntauloc), dtype=np.complex128, order='F')
        
        moment, high = self.Moment(ff, isgreen, highzero)

        ffglob = np.zeros((norb, norb, ns, nk, nomega), dtype=np.complex128, order='F')

        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        ffglob[iorb, jorb, js, ik] = self.mpimanager.FMPIAllreduce(self.commw, ff[iorb, jorb, js, ik], nomega)
        
        ftau = Fourier.FLatDynF2T(self.ftgrid.omega, ffglob, moment, tau)

        return ftau
    
    def T2F(self, ftau : np.ndarray) -> np.ndarray:

        rank = self.nodedict['commfrank']
        nfloc = self.submatrixw[rank][1]-self.submatrixw[rank][0]
        norb = ftau.shape[0]
        ns = ftau.shape[2]
        nk = ftau.shape[3]
        # ntauloc = ftau.shape[4]
        # tau = self.ftgrid.tau
        ntau = len(self.ftgrid.tau)
        
        # print(f"commfrank : {rank}, total comm rank : {self.mpimanager.rank}")

        ff = np.zeros((norb, norb, ns, nk, nfloc), dtype=np.complex128, order='F')
        omega = np.zeros((nfloc), dtype=np.float64, order='F')

        ftauglob = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')

        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        ftauglob[iorb, jorb, js, ik] = self.mpimanager.TMPIAllreduce(self.commtau, ftau[iorb, jorb, js, ik], ntau)

        for ifloc in range(nfloc):
            fidx = self.mpimanager.FLocal2Global([rank, ifloc])
            omega[ifloc] = self.ftgrid.omega[fidx]

        # ff = QAFort.fourier.flatdyn_t2f(self.ftgrid.tau, self.ftgrid.beta, ftauglob, omega)
        ffglob = Fourier.FLatDynT2F(self.ftgrid.tau, ftauglob, self.ftgrid.omega)

        for ifreq in range(nfloc):
            fidx = self.mpimanager.FLocal2Global([rank, ifreq])
            ff[...,ifreq] = ffglob[...,fidx]

        return ff
    
    def Kfc2Kff(self, fin : np.ndarray, kvec : np.ndarray) -> np.ndarray:

        finv = self.Inverse(fin)
        rankf = self.nodedict['commfrank']
        rankk = self.nodedict['commkrank']
        tempmat = self.K2R(finv)
        norb, _, ns, nk, nfreq = finv.shape
        tempmat2 = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        omega = self.ftgrid.omega * 1j

        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            fidx = self.mpimanager.FLocal2Global([rankf, ifreq])
                            if (iorb == jorb):
                                tempmat2[iorb, jorb, js, ik, ifreq] = (
                                    omega[fidx] - finv[iorb, jorb, js, ik, ifreq]
                                )
                            else:
                                tempmat2[iorb, jorb, js, ik, ifreq] = (
                                    -finv[iorb, jorb, js, ik, ifreq]
                                )
        
        self.crystal.RVec()
        tempmat3 = Fourier.FPathDynR2K(self.commk, tempmat2, self.mpimanager, kvec)

        return tempmat3






        
        

    
class GreenBare(FLatDyn):

    def __init__(self, crystal: Crystal, ftgrid: FTGrid, hamtb : np.ndarray = None, hdf5file : str = None, group : str = None) -> object:
        
        super().__init__(crystal, ftgrid)
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
        
        
        gnotkf = Bare.FLatFreq(self.hamtb, self.ftgrid.omega)
        gnotrf = self.K2R(gnotkf)#######
        
        self.kf = gnotkf
        self.rf = gnotrf

        gnotkt = Bare.FLatTau(self.hamtb,self.ftgrid.tau)
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

    def __init__(self, crystal: Crystal, ftgrid: FTGrid, greenbare : np.ndarray = None, sigmah : np.ndarray = None, sigmaf : np.ndarray = None, sigmagwc : np.ndarray = None, hdf5file : str = 'glob.h5', group : str = None) -> object:
        
        if greenbare is None:
            print("Bare Green's function doesn't exist")
            sys.exit()
        super().__init__(crystal, ftgrid)
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
        nomega = len(self.ftgrid.omega)
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
        

        self.gktmu0 = self.F2T(self.gkfmu0,True, True)
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
        nft = len(self.ftgrid.omega)

        gkfnew = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')
        chem = self.ChemEmbedding(self.mu)
    
    
        gkfnew = self.Dyson(self.gkfmu0,-chem)
        
        self.kf = gkfnew
        self.kt = self.F2T(gkfnew,True, True)
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
        nft = len(self.ftgrid.omega)#self.ft.size
        tempmat = copy.deepcopy(self.gkfmu0)
        chem = self.ChemEmbedding(mu)
        gcalf = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')
        gcalt = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        
        gcalf = self.Dyson(tempmat,-chem)
        gcalt = self.F2T(gcalf, True, True)
        
        
        
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
        mumin = -self.ftgrid.omega[-1]*0.6
        mumax = self.ftgrid.omega[-1]*0.6
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
        nfreq = len(self.ftgrid.omega)#self.ft.size
        beta = self.ftgrid.beta

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

class GreenAB(FLatDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid) -> object:
        super().__init__(crystal, ft)

        glob = h5py.File('../../glob_dat/global.dat', 'r')
        self.i_kerf = glob['full_space']['gw']['i_kref'][:]
        self.kpt_latt = glob['combasis_fermion']['kpt_latt'][:]
        self.nbndf = glob['full_space']['gw']['nbndf'][:]
        self.n_omega = glob['full_space']['gw']['n_omega'][:]
        self.n3 = glob['full_space']['Gfull_n3'][:]
        glob.close()

        self.crystal.MappingKpoint(self.kpt_latt)

    def KI2KF(self):

        rank = self.nodedict['commkrank']
        (nkx, nky, nkz) = self.mpimanager.localshapef[rank]
        nkloc = nkx * nky * nkz
        tempmat = np.zeros((self.nbndf[0], self.nbndf[0], self.n3[0], nkloc, self.crystal.ns), dtype=np.complex128, order='F')

        glob = h5py.File('../../glob_dat/global.dat', 'r')

        for js in range(self.crystal.ns):
            for iw in range(self.n3[0]):
                for ik in range(nkloc):
                    kglob = self.mpimanager.KLocal2Global([rank, ik])
                    kidx = self.crystal.mappingkp[kglob]
                    kk = self.i_kerf[kidx]
                    name = 'Gfull_w_'+str(iw+1)+'_k_'+str(kk)
                    tempmat[..., iw, ik, js] = glob['full_space'][name][:]
                # for ik in range(len(self.kpt_latt)):
                #     kidx = self.i_kerf[ik]
                #     name = 'Gfull_w_'+str(iw+1)+'_k_'+str(kidx)
                #     kglob = self.crystal.mappingkp[kidx]
                #     [rank_temp, kk] = self.mpimanager.KGlobal2Local(kglob)
                #     if (rank_temp == rank):
                #         tempmat[...,iw,kk, js] = glob['full_space'][name][:]
        glob.close()
        # kpt_latt != kpoints

        self.kf = np.copy(tempmat)

        self.kt = self.F2T(tempmat, True, True)
        self.rf = self.K2R(tempmat)
        self.rt = self.K2R(self.kt)

        return None
    

