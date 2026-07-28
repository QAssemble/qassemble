"""High-level orchestration of tight-binding, Hartree-Fock, and GW calculations."""
import numpy as np
import matplotlib.pyplot as plt
import sys, os, time
import gc
import h5py
from .Crystal import Crystal
# from .FTGrid import FTGrid
from .utility.DLR import DLR
from .FLatDyn import *
from .FLatStc import *
from .FLocDyn import *
from .FLocStc import *
from .BLatDyn import *
from .BLatStc import *
from .BLocDyn import *
from .BLocStc import *

class CorrelationFunction(object):
    """Facade for constructing and running many-body calculation stages."""

    def __init__(self, cry : dict = None, ft : dict = None, c = 1.0):
        """Initialize crystal and DLR state for correlation-function workflows."""

        self.c = c
        self.h0 = None
        self.g = None
        self.g0 = None
        self.sigh = None
        self.sigf = None
        self.siggwc = None
        self.h = None
        self.occ = None
        self.v = None
        self.p = None
        self.w = None

        # cry = Crystal(latt=latt,basisposition=basisposition,ns=ns,soc=soc,rkgrid=rkgrid,orboption=orboption,N=N)
        #cry = Crystal#(Rvec=Rvec,CorF=CorF,Basis=Basis,Nspin=Nspin,SOC=SOC,Nelec=Nelec,#KGrid=KGrid)
        #self.cry = cry
        #ft = FTGrid(T=T,beta=beta,cutoff=cutoff)
        #self.ft = ft
        self.crystal = Crystal(cry=cry)
        # self.ft = FTGrid(ft=ft)
        self.dlr = DLR(ft)

        # if os.path.exists('work'):
        #     pass
        # else:
        #     os.mkdir('work')

    def SCFCheck(self, mat1 : np.ndarray, mat2 : np.ndarray):
        """Return the maximum absolute difference between two SCF matrices."""

        check = 0
        tempmat = abs(mat1-mat2)
        check = tempmat.max()
        return check

    def TightBinding(self, hopping : dict = None, onsite : dict = None, spin : bool = False, site : bool = False, valley : bool = False, hdf5file : str = 'glob.h5'):
        """Build and save the tight-binding Hamiltonian stage."""

        # file = h5py.File(fn+'.h5','w')
        # tb = file.create_group('tb')

        group = 'tb'
        errmessage = "missing input for tight binding calculation"
        if (hopping == None):
            print(errmessage)
            sys.exit()
        h0 = H0(crystal=self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,hdf5file=hdf5file,group=group)
        self.h0 = h0
        # file.close()

        return None

    def HartreeFock(self, itermax : int, mix : float, hopping : dict = None,mode : str = "FromScratch", onsite : dict = None, spin : bool = False, valley : bool = False, avalley : bool = False, site : bool = False, asite : bool = False, aferro : bool = False, loccoulomb : dict = None, nonloccoulomb : list = None, ohno : bool = False, jth : bool = False, ohnoyuka : bool = False, hdf5file : str = 'glob.h5', group : str = 'hf'):
        """Run the self-consistent Hartree-Fock workflow."""

        errmessage = "missing input for HF calculation"
        if (hopping==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()
        
        if (mode == 'FromScratch'):
            
            h0 = H0(self.crystal,hopping=hopping,onsite=onsite,hdf5file=hdf5file,group=group)
            v = V(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=hdf5file,group=group)
            self.v = v

        elif (mode == 'Restart'):
            group = group + '_restart'
            h0 = H0(self.crystal,hopping=hopping,onsite=onsite,hdf5file=hdf5file,group=group)
            v = V(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=hdf5file,group=group)
            self.v = v





        for iter in range(1, itermax+1):
            if iter==1:
                # onsite_temp = None
                # if self.crystal.ns == 2:
                #     onsite_temp = {}
                #     for js in range(self.crystal.ns):
                #         onsite_temp[js] = {}
                #         for iorb in range(len(self.crystal.find)):
                #             ii, m1 = self.crystal.FAtomOrb(iorb)
                #             if js == 0:
                #                 onsite_temp[js][(ii,m1)] = 1.0
                #             else:
                #                 onsite_temp[js][(ii,m1)] = -1.0 
                if mode == "FromScratch":
                    h0_temp = H0(self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,site=site,aferro=aferro, hdf5file=hdf5file,group='test_hf', avalley=avalley, asite=asite)
                    hold = H(crystal=self.crystal,h0=h0_temp.k,beta=self.dlr.beta,hdf5file=hdf5file,group=group)
                elif mode == "Restart":
                    h0_temp = H0(self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,site=site,aferro=aferro, hdf5file=None,group='test_hf', avalley=avalley, asite=asite)
                    glob = h5py.File(hdf5file,'r')
                    hf = glob['hf']
                    hk = hf['H']['hk'][:]
                    glob.close()
                    hold = H(crystal=self.crystal,h0=hk,beta=self.dlr.beta,hdf5file=hdf5file,group=group)
                    
                    

                hartreeold = None
                fockold = None

            print(hold.occ)
            sigh = SigH(crystal=self.crystal,occ=hold.occ,v=v.k,hdf5file=hdf5file,group=group)
            sigh.k = sigh.Mixing(iter=iter,mix=mix,Fb=sigh.k,Fm=hartreeold)
            if (iter % 50 == 0):
                sigh.Save(f'sigh.{iter}')
            sigf = SigF(crystal=self.crystal,occr=hold.occr,v=v.r,hdf5file=hdf5file,group=group)
            sigf.k = sigf.Mixing(iter=iter,mix=mix,Fb=sigf.k,Fm=fockold)
            if (iter % 50 == 0):
                sigf.Save(f'sigf.{iter}')
            hnew = H(crystal=self.crystal,h0=h0.k,beta=self.dlr.beta,sigh=sigh.k,sigf=sigf.k,hdf5file=hdf5file,group=group)
            # hnew = H(crystal=self.crystal,h0=h0.k,beta=self.ft.beta,sigh=None,sigf=sigf,hdf5file=fn,group=group)
            if (iter % 50 == 0):
                hnew.Save(f'hk.{iter}')

            fcheck = self.SCFCheck(hnew.k,hold.k)
            mucheck = abs(hnew.mu-hold.mu)
            print(f"iteration : {iter}\ncriteria : {fcheck}\nchemical potential : {hnew.mu}")
            if (fcheck<=1.0e-7)and(mucheck<=0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.h=hnew
                self.sigf = sigf
                self.sigh = sigh
                hnew.Save('hk',True)
                sigh.Save('sigh')
                sigf.Save('sigf')
                del hnew, sigh, sigf, hold
                # del hnew, sigf, hold
                gc.collect()
                break
            elif(iter==itermax):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.h=hnew
                self.sigf = sigf
                self.sigh = sigh
                hnew.Save('hk',True)
                sigh.Save('sigh')
                sigf.Save('sigf')
                del hnew, sigh, sigf, hold
                # del hnew, sigf, hold
                gc.collect()
            else:
                # hnew.OccMixing(iter=iter, mix=mix, occkb = hnew.occk, occkm=hold.occk)
                hold=hnew
                hartreeold = sigh.k
                fockold = sigf.k
                del sigf,sigh,hnew
                # del sigf,hnew
                gc.collect()


    def GWApproximation(self, itermax : int, mix : float, hoppinglist : list = None, onsitelist : list = None, spin : bool = False, valley : bool = False, site : bool = False, aferro : bool = False, loccoulomb : dict = None, nonloccoulomb : list = None,ohno : bool = False, jth : bool = False, ohnoyuka : bool = False, hdf5file : str = 'glob.h5', group : str = 'gw'):
        """Run the self-consistent GW approximation workflow."""

        errmessage = "missing input for GW calculation"
        if (hoppinglist==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()
        
        h0 = H0(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,hdf5file=hdf5file,group=group)
        g0 = G0(crystal=self.crystal,dlr=self.dlr,h0=h0.k,hdf5file=hdf5file,group=group)
        v = V(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=hdf5file,group=group)

        p_mixer = Mixing()
        siggwc_mixer = Mixing()

        self.gw_object_times = []
        for iter in range(1,itermax+1):
            iter_timing = {"iter": iter}
            if iter == 1:
                # niham_temp = H0(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin, valley=valley, hdf5file=hdf5file,group='test')
                # niham_temp = H0(self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin,aferro=aferro, valley=valley,site=site,hdf5file=hdf5file,group='test_gw')
                t0 = time.perf_counter()
                gold = G(crystal=self.crystal,dlr=self.dlr,g0=g0.kf,hdf5file=hdf5file,group=group)
                print(f"Initial chemical potential : {gold.mu}")
                iter_timing["G_init"] = time.perf_counter() - t0
                gold.Save(f'gkf_ini')
                wold = 0
                # g0.Save('g0')

            print("Density Matrix :")
            print(gold.occ)
            # print("Hartree calculation start")
            sigh = SigH(crystal=self.crystal,occ=gold.occ,v=v.k,hdf5file=hdf5file,group=group)
            # if (iter % 50 == 0)or(iter == 1):
            sigh.Save(f'sigmah.{iter}')
            # print("Hartree calculation finish")
            # print("Fock calculation start")
            sigf = SigF(crystal=self.crystal,occr=gold.occr,v=v.r,hdf5file=hdf5file,group=group)
            # if (iter % 50 == 0)or(iter == 1):
            sigf.Save(f'sigmaf.{iter}')
            # print("Fock calculation finish")
            # print("P calculation start")
            t0 = time.perf_counter()
            p = P(crystal=self.crystal,dlr=self.dlr,g=gold.rt,hdf5file=hdf5file,group=group)
            iter_timing["P"] = time.perf_counter() - t0
            if iter == 1:
                pkfold = np.zeros_like(p.kf)
            p.kf = p_mixer(iter=iter, mix=mix, Fnew=p.kf, Fold=pkfold)
            p.Save(f'pkf.{iter}')
            # print("P calculation finish")
            # print("Screened coulomb interaction calculation start")
            t0 = time.perf_counter()
            w = W(crystal=self.crystal,dlr=self.dlr,p=p.kf,v=v,c=self.c,hdf5file=hdf5file,group=group)
            iter_timing["W"] = time.perf_counter() - t0
            # if (iter % 50 == 0)or(iter == 1):
            w.Save(f'wkf.{iter}')
            # w.Save(w.ckf,f'wckf.{iter}')
            # print("Screened coulomb interaction calculation finish")
            # print("GW self-energy calculation start")
            t0 = time.perf_counter()
            siggwc = SigGWC(crystal=self.crystal,dlr=self.dlr,g=gold.rt,w=w.crt,hdf5file=hdf5file,group=group)
            iter_timing["SigGWC"] = time.perf_counter() - t0
            if iter == 1:
                ckfold = np.zeros_like(siggwc.kf)
            siggwc.kf = siggwc_mixer(iter=iter, mix=mix, Fnew=siggwc.kf, Fold=ckfold)
            siggwc.Save(f'sigmagwckf.{iter}')
            # print("GW self-energy calculation finish")
            # print("GW green's function calculation start")
            t0 = time.perf_counter()

            gnew = G(crystal=self.crystal,dlr=self.dlr,g0=g0.kf,sigh=sigh.k,sigf=sigf.k,siggwc=siggwc.kf,hdf5file=hdf5file,group=group)
            iter_timing["G"] = time.perf_counter() - t0
            # if (iter % 50 == 0)or(iter == 1):
            gnew.Save(f'gkf.{iter}')
            # print("GW green's function calculation start")
            self.gw_object_times.append(iter_timing)
            init_msg = ""
            if "G_init" in iter_timing:
                init_msg = f", G_init: {iter_timing['G_init']:.4f}s"
            print(
                f"[GW timing][iter {iter}] G: {iter_timing['G']:.4f}s, "
                f"P: {iter_timing['P']:.4f}s, "
                f"W: {iter_timing['W']:.4f}s, "
                f"SigGWC: {iter_timing['SigGWC']:.4f}s{init_msg}"
            )
            fcheck = self.SCFCheck(gnew.kf,gold.kf)
            
            bcheck = self.SCFCheck(w.kf,wold)
            mucheck = abs(gnew.mu-gold.mu)

            print(f"iteration : {iter} \nfcriteria : {fcheck} \nbcriteria : {bcheck} \nchemicalpotential : {gnew.mu+gnew.c}")
            # print(f"iteration : {iter} \nfcriteria : {fcheck} \nchemicalpotential : {gnew.mu}")

            if (iter > p_mixer.npulay)and(fcheck <=1.0e-6)and(mucheck<=0.01)and(bcheck<=1.0e-4):
                print(f"Self-consistency is achived with {iter}-th")
                self.g = gnew
                self.p = p
                self.w = w
                self.siggwc = siggwc
                self.sigf = sigf
                self.sigh = sigh
                gnew.Save('gkf',chem=True)
                sigh.Save('sigmah')
                sigf.Save('sigmaf')
                siggwc.Save('sigmagwckf')
                p.Save('pkf')
                w.Save('wkf')
                # self.siggwc.SigmaStc()
                # self.siggwc.Zfactor()
                del h0, v, g0, gnew, gold, sigf, sigh, siggwc, p, w
                gc.collect()
                break
            elif (iter==itermax):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.g = gnew
                self.p = p
                self.w = w
                self.siggwc = siggwc
                self.sigf = sigf
                self.sigh = sigh
                gnew.Save('gkf',chem=True)
                sigh.Save('sigmah')
                sigf.Save('sigmaf')
                siggwc.Save('sigmagwckf')
                p.Save('pkf')
                w.Save('wkf')
                # self.siggwc.SigmaStc()
                # self.siggwc.Zfactor()
                del h0, v, g0, gnew, gold, sigf, sigh, siggwc, p, w
                gc.collect()
            else:
                gold = gnew
                ckfold = siggwc.kf
                pkfold = p.kf
                wold = w.kf

                del gnew, sigh, sigf, siggwc, p, w
                gc.collect()
