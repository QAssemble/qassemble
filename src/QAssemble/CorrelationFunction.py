import copy
import logging
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
from .Projector import Projector
from .CTQMC import CTQMC

logger = logging.getLogger("QAssemble")

class CorrelationFunction(object):

    def __init__(self, control : dict):

        self.control = control
        self.c = control["run"]["cw"]

        self.green = None
        self.niham = None
        self.greenbare = None
        self.sigmah = None
        self.sigmaf = None
        self.sigmagwc = None
        self.ham = None
        self.occ = None
        self.vbare = None
        self.pol = None
        self.w = None
        cry = control['crystal']
        ft = control['ft']
        self.crystal = Crystal(cry=cry)
        self.dlr = DLR(ft)

        
        onebody = control["ham"].get("onebody")

        self.niham = NIHamiltonian(crystal=self.crystal, onebody=onebody, hdf5file=control["run"]["fn"]+'.h5', group='init')

        self.vbare = VBare(crystal=self.crystal, twobody=control['ham'].get('twobody'), hdf5file=control["run"]["fn"]+'.h5', group='init')

        self.greenbare = GreenBare(crystal=self.crystal, dlr=self.dlr, hamtb=self.niham.k, hdf5file=control["run"]["fn"]+'.h5', group='init')
        

    def SCFCheck(self, mat1 : np.ndarray, mat2 : np.ndarray):

        check = 0
        tempmat = abs(mat1-mat2)
        check = tempmat.max()
        return check

    def TightBinding(self):

        # file = h5py.File(fn+'.h5','w')
        # tb = file.create_group('tb')

        group = 'tb'
        errmessage = "missing input for tight binding calculation"
        hdf5file = self.control["run"]["fn"] + '.h5'
        # niham = NIHamiltonian(crystal=self.cry,hoppinglist=hoppinglist,onsitelist=onsitelist,hdf5file=tb)
        niham = self.niham
        # file.close()

        return niham

    def HartreeFock(self):

        errmessage = "missing input for HF calculation"

        itermax = self.control["run"]["nscf"]
        mix = self.control["run"]["mix"]
        hdf5file = self.control["run"]["fn"] + '.h5'
        mode = self.control["run"]["mode"]
        group = 'hf'

        if (mode == 'FromScratch'):
            
            niham = self.niham
            vbare = self.vbare
            

        elif (mode == 'Restart'):
            group = group + '_restart'
            niham = self.niham
            vbare = self.vbare


        onebody = self.control["ham"].get("onebody")
        # twobody = self.control["ham"].get("twobody")


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
                    niham_temp = NIHamiltonian(self.crystal,onebody=onebody, hdf5file=hdf5file,group='test_hf')
                    hold = Hamiltonian(crystal=self.crystal,ham=niham_temp.k,beta=self.dlr.beta,hdf5file=hdf5file,group=group)
                elif mode == "Restart":
                    niham_temp = NIHamiltonian(self.crystal,onebody=onebody, hdf5file=None,group='test_hf')
                    glob = h5py.File(hdf5file,'r')
                    hf = glob['hf']
                    hk = hf['Hamiltonian']['hk'][:]
                    glob.close()
                    hold = Hamiltonian(crystal=self.crystal,ham=hk,beta=self.dlr.beta,hdf5file=hdf5file,group=group)
                    
                    

                hartreeold = None
                fockold = None

            logger.debug(hold.occ)
            sigmah = SigmaHartree(crystal=self.crystal,occ=hold.occ,vbare=vbare.k,hdf5file=hdf5file,group=group)
            sigmah.k = sigmah.Mixing(iter=iter,mix=mix,Fb=sigmah.k,Fm=hartreeold)
            if (iter % 50 == 0):
                sigmah.Save(f'sigh.{iter}')
            sigmaf = SigmaFock(crystal=self.crystal,occr=hold.occr,vbare=vbare.r,hdf5file=hdf5file,group=group)
            sigmaf.k = sigmaf.Mixing(iter=iter,mix=mix,Fb=sigmaf.k,Fm=fockold)
            if (iter % 50 == 0):
                sigmaf.Save(f'sigf.{iter}')
            hnew = Hamiltonian(crystal=self.crystal,ham=niham.k,beta=self.dlr.beta,sigmah=sigmah.k,sigmaf=sigmaf.k,hdf5file=hdf5file,group=group)
            # hnew = Hamiltonian(crystal=self.crystal,ham=niham.k,beta=self.ft.beta,sigmah=None,sigmaf=sigmaf,hdf5file=fn,group=group)
            if (iter % 50 == 0):
                hnew.Save(f'hk.{iter}')

            fcheck = self.SCFCheck(hnew.k,hold.k)
            mucheck = abs(hnew.mu-hold.mu)
            logger.info(f"iteration : {iter}\ncriteria : {fcheck}\nchemical potential : {hnew.mu}")
            if (fcheck<=1.0e-7)and(mucheck<=0.01):
                logger.info(f"Self-consistency is achived with {iter}-th")
                self.ham=hnew
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                hnew.Save('hk',True)
                sigmah.Save('sigh')
                sigmaf.Save('sigf')
                del hnew, sigmah, sigmaf, hold
                # del hnew, sigmaf, hold
                gc.collect()
                break
            elif(iter==itermax):
                logger.warning(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.ham=hnew
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                hnew.Save('hk',True)
                sigmah.Save('sigh')
                sigmaf.Save('sigf')
                del hnew, sigmah, sigmaf, hold
                # del hnew, sigmaf, hold
                gc.collect()
            else:
                # hnew.OccMixing(iter=iter, mix=mix, occkb = hnew.occk, occkm=hold.occk)
                hold=hnew
                hartreeold = sigmah.k
                fockold = sigmaf.k
                del sigmaf,sigmah,hnew
                # del sigmaf,hnew
                gc.collect()


    def GWApproximation(self):

        errmessage = "missing input for GW calculation"
    
        itermax = self.control["run"]["nscf"]
        mix = self.control["run"]["mix"]
        hdf5file = self.control["run"]["fn"] + '.h5'
        # mode = self.control["run"]["mode"]
        group = 'gw'

        niham = self.niham
        gbare = self.greenbare
        vbare = self.vbare

        pol_mixer = Mixing()
        sig_mixer = Mixing()

        self.gw_object_times = []
        for iter in range(1,itermax+1):
            iter_timing = {"iter": iter}
            if iter == 1:
                
                t0 = time.perf_counter()
                gold = GreenInt(crystal=self.crystal,dlr=self.dlr,greenbare=gbare.kf,hdf5file=hdf5file,group=group)
                logger.info(f"Initial chemical potential : {gold.mu}")
                iter_timing["GreenInt_init"] = time.perf_counter() - t0
                gold.Save(f'gkf_ini')
                wold = 0
                # gbare.Save('gbare')

            logger.info("Density Matrix :")
            logger.debug(gold.occ)
            # print("Hartree calculation start")
            sigmah = SigmaHartree(crystal=self.crystal,occ=gold.occ,vbare=vbare.k,hdf5file=hdf5file,group=group)
            # if (iter % 50 == 0)or(iter == 1):
            sigmah.Save(f'sigmah.{iter}')
            # print("Hartree calculation finish")
            # print("Fock calculation start")
            sigmaf = SigmaFock(crystal=self.crystal,occr=gold.occr,vbare=vbare.r,hdf5file=hdf5file,group=group)
            # if (iter % 50 == 0)or(iter == 1):
            sigmaf.Save(f'sigmaf.{iter}')
            # print("Fock calculation finish")
            # print("Polarizability calculation start")
            t0 = time.perf_counter()
            pol = PolLat(crystal=self.crystal,dlr=self.dlr,green=gold.rt,hdf5file=hdf5file,group=group)
            iter_timing["Polarizability"] = time.perf_counter() - t0
            if iter == 1:
                pkfold = np.zeros_like(pol.kf)
            pol.kf = pol_mixer(iter=iter, mix=mix, Fnew=pol.kf, Fold=pkfold)
            pol.Save(f'pkf.{iter}')
            # print("Polarizability calculation finish")
            # print("Screened coulomb interaction calculation start")
            t0 = time.perf_counter()
            w = WLat(crystal=self.crystal,dlr=self.dlr,pol=pol.kf,vbare=vbare,c=self.c,hdf5file=hdf5file,group=group)
            iter_timing["WLat"] = time.perf_counter() - t0
            # if (iter % 50 == 0)or(iter == 1):
            w.Save(f'wkf.{iter}')
            # w.Save(w.ckf,f'wckf.{iter}')
            # print("Screened coulomb interaction calculation finish")
            # print("GW self-energy calculation start")
            t0 = time.perf_counter()
            sigmagwc = SigmaGWC(crystal=self.crystal,dlr=self.dlr,green=gold.rt,wlat=w.crt,hdf5file=hdf5file,group=group)
            iter_timing["SigmaGW"] = time.perf_counter() - t0
            if iter == 1:
                ckfold = np.zeros_like(sigmagwc.kf)
            sigmagwc.kf = sig_mixer(iter=iter, mix=mix, Fnew=sigmagwc.kf, Fold=ckfold)
            sigmagwc.Save(f'sigmagwckf.{iter}')
            # print("GW self-energy calculation finish")
            # print("GW green's function calculation start")
            t0 = time.perf_counter()

            gnew = GreenInt(crystal=self.crystal,dlr=self.dlr,greenbare=gbare.kf,sigmah=sigmah.k,sigmaf=sigmaf.k,sigmagwc=sigmagwc.kf,hdf5file=hdf5file,group=group)
            iter_timing["GreenInt"] = time.perf_counter() - t0
            # if (iter % 50 == 0)or(iter == 1):
            gnew.Save(f'gkf.{iter}')
            # print("GW green's function calculation start")
            self.gw_object_times.append(iter_timing)
            init_msg = ""
            if "GreenInt_init" in iter_timing:
                init_msg = f", GreenInt_init: {iter_timing['GreenInt_init']:.4f}s"
            logger.info(
                f"[GW timing][iter {iter}] GreenInt: {iter_timing['GreenInt']:.4f}s, "
                f"Polarizability: {iter_timing['Polarizability']:.4f}s, "
                f"WLat: {iter_timing['WLat']:.4f}s, "
                f"SigmaGW: {iter_timing['SigmaGW']:.4f}s{init_msg}"
            )
            fcheck = self.SCFCheck(gnew.kf,gold.kf)
            
            bcheck = self.SCFCheck(w.kf,wold)
            mucheck = abs(gnew.mu-gold.mu)

            logger.info(f"iteration : {iter} \nfcriteria : {fcheck} \nbcriteria : {bcheck} \nchemicalpotential : {gnew.mu+gnew.c}")
            # print(f"iteration : {iter} \nfcriteria : {fcheck} \nchemicalpotential : {gnew.mu}")

            if (iter > pol_mixer.npulay)and(fcheck <=1.0e-6)and(mucheck<=0.01)and(bcheck<=1.0e-4):
                logger.info(f"Self-consistency is achived with {iter}-th")
                self.green = gnew
                self.pol = pol
                self.w = w
                self.sigmagwc = sigmagwc
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                gnew.Save('gkf',chem=True)
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                sigmagwc.Save('sigmagwckf')
                pol.Save('pkf')
                w.Save('wkf')
                # self.sigmagwc.SigmaStc()
                # self.sigmagwc.Zfactor()
                del niham, vbare, gbare, gnew, gold, sigmaf, sigmah, sigmagwc, pol, w
                gc.collect()
                break
            elif (iter==itermax):
                logger.info(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.green = gnew
                self.pol = pol
                self.w = w
                self.sigmagwc = sigmagwc
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                gnew.Save('gkf',chem=True)
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                sigmagwc.Save('sigmagwckf')
                pol.Save('pkf')
                w.Save('wkf')
                # self.sigmagwc.SigmaStc()
                # self.sigmagwc.Zfactor()
                del niham, vbare, gbare, gnew, gold, sigmaf, sigmah, sigmagwc, pol, w
                gc.collect()
            else:
                gold = gnew
                ckfold = sigmagwc.kf
                pkfold = pol.kf
                wold = w.kf

                del gnew, sigmah, sigmaf, sigmagwc, pol, w
                gc.collect()

    def _DMFTImpurityConfig(self) -> dict:
        config = self.control.get("impurity", self.control.get("dmft", {}))
        impdict = config.get("impdict", config.get("ImpDict"))
        equiv = config.get("equiv", config.get("Equiv"))

        if impdict is None:
            raise KeyError("DMFT requires control['impurity']['impdict']")
        if equiv is None:
            raise KeyError("DMFT requires control['impurity']['equiv']")

        return {
            "impdict": copy.deepcopy(impdict),
            "equiv": copy.deepcopy(equiv),
        }

    def SCFCheckImpurityGreen(self, gloc : dict, gimp : dict) -> float:
        check = 0.0
        for key in gloc.keys():
            if key not in gimp:
                raise KeyError(f"Missing impurity Green's function for problem key '{key}'")
            check = max(check, self.SCFCheck(gloc[key], gimp[key]))
        return check

    def ImpuritySolver(self):

        errmessage = "missing input for DMFT calculation"

        itermax = self.control["run"]["nscf"]
        dmft_tol = self.control["run"].get("dmft_tol", 1.0e-6)
        hdf5file = self.control["run"]["fn"] + '.h5'
        group = 'dmft'

        niham = self.niham
        gbare = self.greenbare
        impurity = self._DMFTImpurityConfig()
        projector = Projector(
            basisindex=self.crystal._basis_index,
            impdict=impurity["impdict"],
            equiv=impurity["equiv"],
        )

        voption = self.control["ham"]["twobody"].get("Local")
        if voption is None:
            raise KeyError("DMFT requires control['ham']['twobody']['Local']")
        vloc = VLoc(crystal=self.crystal, voption=voption)

        self.dmft_object_times = []
        self.ctqmc = None
        for iter in range(1, itermax+1):
            iter_timing = {"iter": iter}

            t0 = time.perf_counter()
            if iter == 1:
                green = GreenInt(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    greenbare=gbare.kf,
                    hdf5file=hdf5file,
                    group=group,
                )
            else:
                green = GreenInt(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    greenbare=gbare.kf,
                    hdf5file=hdf5file,
                    group=group,
                )
            iter_timing["GreenInt"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            gloc = GLoc(
                crystal=self.crystal,
                dlr=self.dlr,
                projector=projector,
                green=green.kf,
                hdf5file=hdf5file,
                group=group,
            )
            iter_timing["GLoc"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            eimp = EImp(
                crystal=self.crystal,
                projector=projector,
                hamtb=niham.k,
                mu=green.mu,
            )
            hyb = Hyb(
                crystal=self.crystal,
                dlr=self.dlr,
                projector=projector,
                green=gloc.f,
                eimp=eimp.e,
            )
            fweiss = FWeiss(
                crystal=self.crystal,
                dlr=self.dlr,
                projector=projector,
                eimp=eimp,
                hyb=hyb,
            )
            bweiss = BWeiss(
                crystal=self.crystal,
                dlr=self.dlr,
                projector=projector,
                vloc=vloc,
                ploc=None,
                wloc=None,
            )
            iter_timing["Weiss"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            ctqmc = CTQMC(
                dlr=self.dlr,
                fweiss=fweiss,
                bweiss=bweiss,
                control=self.control["run"],
            )
            ctqmc.PreProcessing(iter=iter)
            ctqmc.Run(iter=iter)
            Green_imp, Sigma_hf_imp, Sigma_bare_imp, susceptibility_imp = ctqmc.PostProcessing(iter=iter)
            iter_timing["CTQMC"] = time.perf_counter() - t0

            gcheck = self.SCFCheckImpurityGreen(gloc.f, Green_imp)

            self.dmft_object_times.append(iter_timing)
            logger.info(
                f"[DMFT timing][iter {iter}] GreenInt: {iter_timing['GreenInt']:.4f}s, "
                f"GLoc: {iter_timing['GLoc']:.4f}s, "
                f"Weiss: {iter_timing['Weiss']:.4f}s, "
                f"CTQMC: {iter_timing['CTQMC']:.4f}s"
            )
            logger.info(f"iteration : {iter} \nimpurity Green criteria : {gcheck}")

            self.green = green
            self.gloc = gloc
            self.eimp = eimp
            self.hyb = hyb
            self.fweiss = fweiss
            self.bweiss = bweiss
            self.ctqmc = ctqmc
            self.impurity_green = Green_imp
            self.impurity_sigma_hf = Sigma_hf_imp
            self.impurity_sigma_bare = Sigma_bare_imp
            self.impurity_susceptibility = susceptibility_imp

            if gcheck <= dmft_tol:
                logger.info(f"DMFT self-consistency is achieved with {iter}-th iteration")
                break
            elif iter == itermax:
                logger.info(f"DMFT reaches max iteration {itermax}; impurity Green criteria = {gcheck}")

            del green, gloc, eimp, hyb, fweiss, bweiss
            gc.collect()
