import numpy as np
import sys, itertools
import scipy.optimize
import scipy.linalg.lapack
import copy
import h5py
import time, datetime
from .Crystal import Crystal
from .FLatStc import FLatStc
from .utility.DLR import DLR
from .utility.Common import Common
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
# qapath = os.environ.get('QAssemble','')
# sys.path.append(qapath+'/src/QAssemble/modules')
# import QAFort

class FLatDyn(object):
    def __init__(self,crystal : Crystal, dlr : DLR) -> object:
        self.crystal = crystal
        self.dlr = dlr
        self.mappingidx = None
        self._fermion_phase_cache_k2r = self._get_fermion_phaseK2R()
        self._fermion_phase_cache_r2k = self._get_fermion_phaseR2K()

    def _get_fermion_phaseK2R(self) -> np.ndarray:
        
        nrk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]

        basis_orb = self.crystal.basisf[self.crystal.forb2atom]

        kv = self.crystal.kpoint[:nrk] @ basis_orb.T

        kv_delta = kv[:, :, None] - kv[:, None, :]

        phases = np.exp(2.0j * np.pi * kv_delta)

        phases_T = np.transpose(phases, (1, 2, 0))
        return phases_T
    
    def _get_fermion_phaseR2K(self) -> np.ndarray:
        
        nrk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]

        basis_orb = self.crystal.basisf[self.crystal.forb2atom]

        kv = self.crystal.kpoint[:nrk] @ basis_orb.T

        kv_delta = kv[:, :, None] - kv[:, None, :]

        phases = np.exp(-2.0j * np.pi * kv_delta)

        phases_T = np.transpose(phases, (1, 2, 0))
        return phases_T
        
    def Inverse(self, mat : np.ndarray) -> np.ndarray:

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]
        nft = mat.shape[4]

        matinv = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    matinv[:,:,js,irk,ift] = Common.MatInv(mat[:,:,js,irk,ift])
        # for js, irk, ift in itertools.product(list(range(ns)),list(range(nrk),list(range(nft)))):
        #     matinv[:,:,js,irk,ift] = np.linalg.inv(mat[:,:,js,irk,ift])
        
        return matinv

    
    def T2F(self,ftau : np.ndarray) -> np.ndarray:

        norb = ftau.shape[0]
        ns = ftau.shape[2]
        nk = ftau.shape[3]
        ntau = ftau.shape[4]

        # Batch DLR transform: pydlr expects (ntau, n1, n2)
        batch = norb * norb * ns * nk
        ftau_flat = ftau.reshape(batch, ntau).T  # (ntau, batch)
        ftau_3d = ftau_flat[:, :, np.newaxis]  # (ntau, batch, 1)

        fxx = self.dlr.dF.dlr_from_tau(ftau_3d)
        ff_3d = self.dlr.dF.matsubara_from_dlr(fxx, beta=self.dlr.beta, xi=-1)
        # ff_3d shape: (nfreq, batch, 1)
        nfreq = ff_3d.shape[0]
        ff = ff_3d[:, :, 0].T.reshape(norb, norb, ns, nk, nfreq)
        ff = np.asfortranarray(ff)

        return ff
    
    def F2T(self,ff : np.ndarray) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]
        nfreq = ff.shape[4]

        # Batch DLR transform: pydlr expects (nfreq, n1, n2)
        batch = norb * norb * ns * nk
        ff_flat = ff.reshape(batch, nfreq).T  # (nfreq, batch)
        ff_3d = ff_flat[:, :, np.newaxis]  # (nfreq, batch, 1)

        fxx = self.dlr.dF.dlr_from_matsubara(ff_3d, beta=self.dlr.beta, xi=-1)
        ftau_3d = self.dlr.dF.tau_from_dlr(fxx)
        # ftau_3d shape: (ntau, batch, 1)
        ntau = ftau_3d.shape[0]
        ftau = ftau_3d[:, :, 0].T.reshape(norb, norb, ns, nk, ntau)
        ftau = np.asfortranarray(ftau)

        return ftau

    
    def Moment(self,ff : np.ndarray, isgreen : bool, highzero : bool) -> tuple:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]

        moment = np.zeros((norb,norb,ns,nk,3),dtype=np.complex128,order='F')
        high = np.zeros((norb,norb,ns,nk),dtype=np.complex128,order='F')

        if ff.shape[4] < 2:
            raise ValueError("Need at least two frequency points to build high-frequency moments.")

        high_freq_slice = ff[..., -1]
        prev_freq_slice = ff[..., -2]

        # moment, high = QAFort.fourier.flatdyn_m(self.dlr.omega,tempmat,isgreen,highzero)
        moment, high = Fourier.FLatDynM(self.dlr.omega, high_freq_slice, prev_freq_slice, isgreen, highzero)

        return moment, high
    
    
    def K2R(self,matk : np.ndarray, rkgrid : list = None) -> np.ndarray:

        rkvec = self.crystal.kpoint
        if rkgrid == None:
            rkgrid = self.crystal.rkgrid

        
        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[3]
        nft = matk.shape[4]

        # basis_orb = self.crystal.basisf[self.crystal.forb2atom]

        # kv = self.crystal.kpoint[:nrk] @ basis_orb.T

        # kv_delta = kv[:, :, None] - kv[:, None, :]

        # phases = np.exp(2.0j * np.pi * kv_delta)

        # phases_T = np.transpose(phases, (1, 2, 0))

        matr = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
        tempmat = np.empty((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
        # phase_view = phases[:, :, np.newaxis, :]
        tempmat = matk.copy()

        tempmat *= self._fermion_phase_cache_k2r[:, :, None, :, None]

        matr = Fourier.FLatDynK2R(tempmat, rkgrid)

        return matr
    
    def R2K(self, matr : np.ndarray) -> np.ndarray:

        rkgrid = self.crystal.rkgrid

        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[3]
        nft = matr.shape[4]

        
        
        matk = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
        tempmat = np.empty((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')

        tempmat = Fourier.FLatDynR2K(matr, rkgrid)

        matk = tempmat * self._fermion_phase_cache_r2k[:, :, None, :, None]

        return matk
    
    
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

        # matout = QAFort.dyson.flatdyn(mat1,mat2)
        return Dyson.FLatDyn(mat1, mat2)
    
    def ChemEmbedding(self,mu : np.float64) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.dlr.omega)#self.ft.size

        chem = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for iorb in range(norb):
                        chem[iorb, iorb, js, irk, ift] = mu

        return chem
    
    def StcEmbedding(self, matin : np.ndarray) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.dlr.omega)#self.ft.size

        matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            matout[...,ift] = matin

        return matout
    
    
    def CheckGroup(self, filepath :str, group : str):
        
        with h5py.File(filepath,'r') as file:
            return group in file
        
    
    def Spectral(self, green : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        nfreq = len(self.dlr.omega)

        akf = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,oder='F')

        akf = -1/np.pi*green.imag

        return akf
    
    def R2KArb(self,matr : np.ndarray = None,kpoint : np.ndarray = None): # R2KAny

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nr = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        nk = len(kpoint)
        nft = matr.shape[4]

        self.crystal.RVec()
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
        omega = self.dlr.omega

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
    
    def R2mR(self, matin : np.ndarray) -> np.ndarray:

        self.crystal.R2mR()

        matout = np.zeros_like(matin, dtype=np.complex128, order='F')

        for rp in self.crystal.mappingidx:
            matout[..., rp[0],:] = matin[..., rp[1], :]

        return matout
    
    def T2mT(self, ftau : np.ndarray) -> np.ndarray:

        taum = self.dlr.beta - self.dlr.tauF

        norb, _, ns, nrk, ntau = ftau.shape

        fout = np.zeros((norb, norb, ns, nrk, ntau), dtype=np.complex128, order='F')

        for irk in range(nrk):
            for js in range(ns):
                fxx = self.dlr.dF.dlr_from_tau(ftau[:, :, js, irk,:].T)
                fout[:, :, js, irk, :] = (self.dlr.dF.eval_dlr_tau(fxx, taum, self.dlr.beta)).T

        return fout      

    def TauB2TauF(self, ftau : np.ndarray) -> np.ndarray:

        norb, _, ns, ns2, nk, ntauB = ftau.shape
        # Reshape: move tau axis first, flatten all other dims into batch
        # ftau shape: (norb, norb, ns, ns, nk, ntauB)
        # pydlr expects: (ntauB, n1, n2) — batch over n1, n2
        batch = norb * norb * ns * ns2 * nk
        ftau_flat = ftau.reshape(batch, ntauB).T  # (ntauB, batch)
        ftau_3d = ftau_flat[:, :, np.newaxis]  # (ntauB, batch, 1)

        fxx = self.dlr.dB.dlr_from_tau(ftau_3d)
        fout_3d = self.dlr.dB.eval_dlr_tau(fxx, self.dlr.tauF, self.dlr.beta)
        # fout_3d shape: (ntauF, batch, 1)
        ntauF = len(self.dlr.tauF)
        fout = fout_3d[:, :, 0].T.reshape(norb, norb, ns, ns2, nk, ntauF)
        fout = np.asfortranarray(fout)

        return fout
    
    def Diagonalize(self, matk : np.ndarray):

        norb, _, ns, nk, nfreq = matk.shape

        eigval = np.zeros((norb, norb, ns, nk, nfreq), dtype=float)
        eigvec = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128)

        for ifreq in range(nfreq):
                for ik in range(nk):
                    for js in range(ns):
                        e, v, info = scipy.linalg.lapack.zheev(matk[:, :, js, ik, ifreq])
                        eigval[:, :, js, ik, ifreq] = np.diag(e)
                        eigvec[:, :, js, ik, ifreq] = v

        return eigval, eigvec

    
class GreenBare(FLatDyn):

    def __init__(self, crystal: Crystal, dlr : DLR, hamtb : np.ndarray = None, hdf5file : str = None, group : str = None) -> object:
        
        super().__init__(crystal, dlr)
        # print(self.niham.hamtb[...,0,0])
        self.hamtb = hamtb
        self.kt = None
        self.kf = None
        self.rt = None
        self.rf = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        print("Bare Green's function Calculation Start")
        start = time.time()
        self.Cal()
        if hdf5file != None:
            self.Save()
        end = time.time()    
        print("Bare Green's function Calculation Finish")
        print(f"Calculation Time : {str(datetime.timedelta(seconds=end-start))}")
        

    def Cal(self): # freq, tau combine
        
        from .utility.Bare import Bare
        # print(self.hamtb[:,:,0,0])
        # gnotkf = QAFort.bare.flatfreq(self.hamtb,self.dlr.omega)
        gnotkf = Bare.FLatFreq(self.dlr.omega, self.hamtb)
        gnotrf = self.K2R(gnotkf)#######
        
        self.kf = gnotkf
        self.rf = gnotrf

        # gnotkt = QAFort.bare.flattau(self.hamtb,self.dlr.tau)
        gnotkt = Bare.FLatTau(tau=self.dlr.tauF, beta=self.dlr.beta, hlatt=self.hamtb)
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
    
class GreenInt(FLatDyn):

    def __init__(self, crystal: Crystal, dlr : DLR, greenbare : np.ndarray = None, sigmah : np.ndarray = None, sigmaf : np.ndarray = None, sigmagwc : np.ndarray = None, hdf5file : str = 'glob.h5', group : str = None) -> object:
        
        if greenbare is None:
            print("Bare Green's function doesn't exist")
            sys.exit()
        super().__init__(crystal, dlr)
        self.flatstc = FLatStc(crystal=crystal)
        norb, _, ns, nk, nfreq = greenbare.shape
        ntau = len(self.dlr.tauF)
        self.kf = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.kt = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.rf = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.rt = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.gkfmu0 = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.gktmu0 = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.grfmu0 = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.grtmu0 = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.gbare = greenbare
        self.sigmah = sigmah
        self.sigmaf = sigmaf
        self.sigmac = sigmagwc
        self.occ = None
        self.occk = None
        self.occr = None
        self.mu = np.float64(0.0)
        self.c = np.float64(0.0)
        # tau_uniform = self.dlr.TauUniform()
        # self._tau_beta = tau_uniform[-1]
        self._tau_beta = self.dlr.beta
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        
        print("Interacting Green's function Calculation Start")
        start = time.time()
        self.CalMu0()
        
        self.SearchMu()
        end = time.time()
        print("Interacting Green's function Calculation Finish")
        print(f"Calculation Time : {str(datetime.timedelta(seconds=end-start))}")

    def CalMu0(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nomega = len(self.dlr.omega)
        sigma = np.zeros((norb,norb,ns,nrk,nomega),dtype=np.complex128,order='F')
        print("Initialization start")
        if (self.sigmah is None)and(self.sigmaf is None)and(self.sigmac is None):
            self.gkfmu0 = self.gbare
        else:
            if (self.sigmah is not None):
                # print(sigma[:,:,0,0,0])
                diag = np.diagonal(self.sigmah[:,:,0,0])
                const = np.mean(diag)
                self.c = np.real(const)
                # print(const)
                sigma += self.StcEmbedding(self.sigmah)
                sigma += self.ChemEmbedding(-const)
                print('Hartree')
                print(sigma[:,:,0,0,0])
            if (self.sigmaf is not None):
                # print(sigma[:,:,0,0,0])
                sigma += self.StcEmbedding(self.sigmaf)
                print('Fock')
                print(sigma[:,:,0,0,0])
            if (self.sigmac is not None):
                # print(sigma[:,:,0,0,0])
                sigma += self.sigmac
                print('GWC')
                print(sigma[:,:,0,0,0])
            self.gkfmu0 = self.Dyson(self.gbare,sigma) 
        

        self.gktmu0 = self.F2T(self.gkfmu0)
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
        # kt = np.copy(self.kt)
        # ntau = 5000
        tau_beta = np.array([self._tau_beta], dtype=np.float64)

        for irk in range(nrk):
            for js in range(ns):
                
                block = self.kt[:, :, js, irk, :].T

                fxx = self.dlr.dF.dlr_from_tau(block)
                fout = self.dlr.dF.eval_dlr_tau(fxx, tau_beta, beta=self.dlr.beta)

                occk[:, :, js, irk] = -fout[0]


        
            
        occ = occk.sum(axis=3)/nrk
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
        nft = len(self.dlr.omega)

        gkfnew = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')
        chem = self.ChemEmbedding(self.mu)
    
    
        gkfnew = self.Dyson(self.gkfmu0,-chem)
    
        
        self.kf = gkfnew
        self.kt = self.F2T(gkfnew)
        # self.grf = self.K2R(self.Dyson(self.gkfmu0,-chem))
        # self.grt = self.K2R(self.F2T(self.Dyson(self.gkfmu0,-chem),1,1))
        self.rf = self.K2R(self.kf)
        self.rt = self.K2R(self.kt)
        print("Chemical potential shift finish")
        self.Occ()

        return None
    
    def NumOfE(self, mu : np.float64):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nfreq = len(self.dlr.omega)

        # Use cached G0inv: G(mu) = (G0inv + mu*I)^{-1}
        mat = self._g0inv_cache.copy()
        diag = np.arange(norb)
        mat[diag, diag, :, :, :] += mu

        # Batch invert: reshape (norb,norb,ns,nk,nfreq) -> (ns*nk*nfreq, norb,norb)
        mat_batch = np.moveaxis(mat, (0, 1), (-2, -1))  # (ns,nk,nfreq,norb,norb)
        orig_shape = mat_batch.shape[:-2]
        mat_flat = mat_batch.reshape(-1, norb, norb)
        gcalf_flat = np.linalg.inv(mat_flat)
        gcalf_batch = gcalf_flat.reshape(orig_shape + (norb, norb))
        gcalf = np.moveaxis(gcalf_batch, (-2, -1), (0, 1))  # (norb,norb,ns,nk,nfreq)

        # Extract diagonal elements only for DLR: shape (norb, ns, nk, nfreq)
        gdiag = gcalf[diag, diag, :, :, :]  # (norb, ns, nk, nfreq)

        # Batch DLR: Matsubara -> single tau point
        # Reshape to (nfreq, norb*ns*nk, 1) for dlr API
        gdiag_perm = np.ascontiguousarray(np.moveaxis(gdiag, -1, 0))  # (nfreq, norb, ns, nk)
        batch_shape = gdiag_perm.shape[1:]
        gdiag_flat = gdiag_perm.reshape(nfreq, -1, 1)
        fxx = self.dlr.dF.dlr_from_matsubara(gdiag_flat, beta=self.dlr.beta, xi=-1)
        fout = self.dlr.dF.eval_dlr_tau(fxx, self._tau_beta_cache, beta=self.dlr.beta)
        gtau_beta = fout[0, :, 0].reshape(batch_shape)  # (norb, ns, nk)

        Ne = -np.real(gtau_beta.sum()) / nrk

        return (self.crystal.nume - Ne)

    def SearchMu(self):

        print("Finding chemical potential start")
        mumin = self.dlr.omega[0]
        mumax = self.dlr.omega[-1]
        print(f"minimum : {mumin}, maximum : {mumax}")

        # Precompute G0^{-1} for vectorized NumOfE
        norb = len(self.crystal.find)
        g0 = self.gkfmu0  # (norb, norb, ns, nk, nfreq)
        g0_batch = np.moveaxis(g0, (0, 1), (-2, -1))  # (..., norb, norb)
        orig_shape = g0_batch.shape[:-2]
        g0_flat = g0_batch.reshape(-1, norb, norb)
        g0inv_flat = np.linalg.inv(g0_flat)
        g0inv_batch = g0inv_flat.reshape(orig_shape + (norb, norb))
        self._g0inv_cache = np.moveaxis(g0inv_batch, (-2, -1), (0, 1))  # (norb, norb, ns, nk, nfreq)
        self._tau_beta_cache = np.array([self._tau_beta], dtype=np.float64)

        nmin = self.NumOfE(mumin)
        nmax = self.NumOfE(mumax)
        if (nmin < 0) or (nmax>0):
            print("Chemical potential is out of the bisection range")
            print(f"nmin : {nmin}, nmax : {nmax}")
            sys.exit()
        sol = scipy.optimize.brentq(self.NumOfE,mumin,mumax,xtol=1.0e-6)
        self.mu = sol
        print("Finding chemical potential finish")

        # Clean up caches
        del self._g0inv_cache
        del self._tau_beta_cache

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

    def __init__(self, crystal: Crystal, dlr : DLR, green : np.ndarray = None, wlat : np.ndarray = None, hdf5file : str = 'glob.h5',group : str = None) -> object:
        super().__init__(crystal, dlr)
        self.flatstc = FLatStc(crystal=crystal)
        norb, _, ns, nk, nfreq = green.shape
        ntau = len(self.dlr.tauF)
        self.rt = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.rf = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.kt = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.kf = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.stck = np.zeros((norb, norb, ns, nk), dtype=np.complex128, order='F')
        self.z = np.zeros((norb, norb, ns, nk), dtype=np.complex128, order='F')
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

        print("GWC self-energy Calculation Start")
        start = time.time()
        self.Cal()
        end = time.time()
        print("GWC self-energy Calculation Finish")
        print(f"Calculation Time : {str(datetime.timedelta(seconds=end-start))}")

    def Cal(self)->np.ndarray: #SigmaGWC
        '''
        Generate correlated self-energy
        input : Wc(R,t), G(R,t)

        return : crtau, crfreq, cktau, ckfreq
        '''

        norbc = self.green.shape[0]
        ns = self.green.shape[2]
        nr = self.green.shape[3]
        ntau = len(self.dlr.tauF)
        norb = self.wlat.shape[0]

        G = self.green
        Wc = self.TauB2TauF(self.wlat)

        bbasis = self.crystal.bbasis
        s_idx = np.arange(ns)
        Wc_diag = Wc[:, :, s_idx, s_idx, :, :]  # (norb, norb, ns, nr, ntau)

        # Flatten (ns, nr, ntau) into single batch dim S for efficient BLAS dispatch
        S = ns * nr * ntau
        Wc_flat = Wc_diag.reshape(norb, norb, S)
        G_flat = np.ascontiguousarray(G).reshape(norbc, norbc, S)
        out_flat = np.zeros((norbc, norbc, S), dtype=np.complex128)

        # Group fermion orbitals by atom
        atom_groups = {}
        for i in range(norbc):
            a = int(self.crystal.forb2atom[i])
            atom_groups.setdefault(a, []).append(i)

        for orbs_a in atom_groups.values():
            oa = np.array(orbs_a)
            na = len(oa)
            bb_a = bbasis[np.ix_(oa, oa)]  # (na, na)

            for orbs_b in atom_groups.values():
                ob = np.array(orbs_b)
                nb = len(ob)
                bb_b = bbasis[np.ix_(ob, ob)]  # (nb, nb)

                G_block = G_flat[np.ix_(oa, ob)]  # (na, nb, S)

                # out[k,p,S] = sum_{a,b} Wc[a,b,S] * (sum_{i:bb_a[k,i]=a} G[i,:,S])
                #                                     * (sum_{j:bb_b[j,p]=b} delta)
                # Precompute indicator matrices (small, orbital-sized):
                #   Ma[a, k, i] = delta(bb_a[k,i], a)
                #   Mb[b, j, p] = delta(bb_b[j,p], b)
                # Only allocate for boson indices that actually appear.

                unique_a = np.unique(bb_a)
                unique_b = np.unique(bb_b)

                # Ma_dict[a] -> (na, na) indicator: Ma[k,i] = delta(bb_a[k,i], a)
                # Contract: temp_a[a][k, j, S] = sum_i Ma[k,i] * G[i, j, S]
                temp_a = {}
                for a in unique_a:
                    mask = (bb_a == a).astype(np.float64)  # (na, na)
                    # mask[k,i] * G[i,j,S] -> [k,j,S]
                    temp_a[a] = np.einsum('ki,ijS->kjS', mask, G_block)  # (na, nb, S)

                # For each (a,b) pair, accumulate:
                # out[k,p,S] -= Wc[a,b,S] * sum_j temp_a[a][k,j,S] * Mb[j,p]
                result = np.zeros((na, nb, S), dtype=np.complex128)
                for a in unique_a:
                    for b in unique_b:
                        Wc_ab = Wc_flat[a, b]  # (S,)
                        mask_b = (bb_b == b).astype(np.float64)  # (nb, nb) where mask_b[j,p]
                        # sum_j temp_a[a][k,j,S] * mask_b[j,p] -> (na, nb, S)
                        contracted = np.einsum('kjS,jp->kpS', temp_a[a], mask_b)  # (na, nb, S)
                        result += Wc_ab[np.newaxis, np.newaxis, :] * contracted

                out_flat[np.ix_(oa, ob)] -= result

        crtau = np.asfortranarray(out_flat.reshape(norbc, norbc, ns, nr, ntau))
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
        nfreq = len(self.dlr.omega)#self.ft.size

        sigma0 = self.kf[..., 0]
        sigma0_dag = np.transpose(np.conjugate(sigma0), (1, 0, 2, 3))
        sigmastc = 0.5 * (sigma0 + sigma0_dag)

        self.stck = np.asfortranarray(sigmastc, dtype=np.complex128)
        # self.Save('sigmastc',obj=sigmastc)

        return None
    
    def Zfactor(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nfreq = len(self.dlr.omega)#self.ft.size
        beta = self.dlr.beta

        sigma0 = self.kf[..., 0]
        sigma0_dag = np.transpose(np.conjugate(sigma0), (1, 0, 2, 3))
        iw = 1j * beta / (2.0 * np.pi)
        tempmat = np.asfortranarray(iw * (sigma0 - sigma0_dag), dtype=np.complex128)

        diag_idx = np.arange(norb)
        tempmat[diag_idx, diag_idx, :, :] += 1.0

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

    def __init__(self, crystal: Crystal, dlr : DLR) -> object:
        super().__init__(crystal, dlr)

        glob = h5py.File('../../glob_dat/global.dat', 'r')
        self.i_kerf = glob['full_space']['gw']['i_kref'][:]
        self.kpt_latt = glob['combasis_fermion']['kpt_latt'][:]
        self.nbndf = glob['full_space']['gw']['nbndf'][:]
        self.n_omega = glob['full_space']['gw']['n_omega'][:]
        self.n3 = glob['full_space']['Gfull_n3'][:]
        glob.close()

    def KI2KF(self):

        tempmat = np.zeros((self.nbndf[0], self.nbndf[0], self.n3[0], len(self.kpt_latt), self.crystal.ns), dtype=np.complex128, order='F')

        glob = h5py.File('../../glob_dat/global.dat', 'r')

        for js in range(self.crystal.ns):
            for iw in range(self.n3[0]):
                for ik in range(len(self.kpt_latt)):
                    kidx = self.i_kerf[ik]
                    name = 'Gfull_w_'+str(iw+1)+'_k_'+str(kidx)
                    tempmat[...,iw,ik, js] = glob['full_space'][name][:]
        glob.close()
        # kpt_latt != kpoints

        self.kf = np.copy(tempmat)

        self.kt = self.F2T(tempmat, 1, 1)
        self.rf = self.K2R(tempmat)
        self.rt = self.K2R(self.kt)

        return None
    
