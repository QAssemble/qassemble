import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gc
import h5py
from .Crystal import Crystal
from .FTGrid import FTGrid
from .FLatDyn import *
from .FLatStc import *
from .FLocDyn import *
from .FLocStc import *
from .BLatDyn import *
from .BLatStc import *
from .BLocDyn import *
from .BLocStc import *
from .CTQMC import *

import time, datetime

class CorrelationFunction(object):

    def __init__(self, cry : dict = None, ft : dict = None, c = 1.0):

        self.c = c
        self.niham = None
        self.green = None
        self.greenbare = None
        self.sigmah = None
        self.sigmaf = None
        self.sigmagwc = None
        self.ham = None
        self.occ = None
        self.vbare = None
        self.pol = None
        self.w = None


        ### for GW+EDMFT
        self.full_sigmah_k = None
        self.full_sigmaf_k = None
        self.full_sigmac_kf = None
        self.full_pol_kf = None

        # cry = Crystal(latt=latt,basisposition=basisposition,ns=ns,soc=soc,rkgrid=rkgrid,orboption=orboption,N=N)
        #cry = Crystal#(Rvec=Rvec,CorF=CorF,Basis=Basis,Nspin=Nspin,SOC=SOC,Nelec=Nelec,#KGrid=KGrid)
        #self.cry = cry
        #ft = FTGrid(T=T,beta=beta,cutoff=cutoff)
        #self.ft = ft
        self.crystal = Crystal(cry=cry)
        self.ft = FTGrid(ft=ft)

        # if os.path.exists('work'):
        #     pass
        # else:
        #     os.mkdir('work')








    def SCFCheck(self, mat1 : np.ndarray, mat2 : np.ndarray):

        check = 0
        tempmat = abs(mat1-mat2)
        check = tempmat.max()
        return check

    def TightBinding(self, hopping : dict = None, onsite : dict = None, spin : bool = False, site : bool = False, valley : bool = False, fn : str = 'glob.h5'):

        # file = h5py.File(fn+'.h5','w')
        # tb = file.create_group('tb')

        group = 'tb'
        errmessage = "missing input for tight binding calculation"
        if (hopping == None):
            print(errmessage)
            sys.exit()
        # niham = NIHamiltonian(crystal=self.cry,hoppinglist=hoppinglist,onsitelist=onsitelist,hdf5file=tb)
        niham = NIHamiltonian(crystal=self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,hdf5file=fn,group=group)
        self.niham = niham
        # file.close()

        return None

    def HartreeFock(self, itermax : int, mix : float, hopping : dict = None,mode : str = "FromScratch", onsite : dict = None, spin : bool = False, valley : bool = False, avalley : bool = False, site : bool = False, asite : bool = False, aferro : bool = False, loccoulomb : dict = None, nonloccoulomb : list = None, ohno : bool = False, jth : bool = False, ohnoyuka : bool = False, fn : str = 'glob.h5', group : str = 'hf'):

        errmessage = "missing input for HF calculation"
        if (hopping==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()
        
        if (mode == 'FromScratch'):
            
            niham = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,hdf5file=fn,group=group)
            vbare = VBare(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=fn,group=group)
            self.vbare = vbare
        elif (mode == 'Restart'):
            group = group + '_restart'
            niham = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,hdf5file=fn,group=group)
            vbare = VBare(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=fn,group=group)
            self.vbare = vbare





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
                    niham_temp = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,site=site,aferro=aferro, hdf5file=fn,group='test_hf', avalley=avalley, asite=asite)
                    hold = Hamiltonian(crystal=self.crystal,ham=niham_temp.k,beta=self.ft.beta,hdf5file=fn,group=group)
                    hcheck = hold
                elif mode == "Restart":
                    niham_temp = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,site=site,aferro=aferro, hdf5file=None,group='test_hf', avalley=avalley, asite=asite)
                    glob = h5py.File(fn,'r')
                    hf = glob['hf']
                    hk = hf['Hamiltonian']['hk'][:]
                    glob.close()
                    hold = Hamiltonian(crystal=self.crystal,ham=hk,beta=self.ft.beta,hdf5file=fn,group=group)
                    hcheck = hold
                    
                    

                hartreeold = None
                fockold = None

            print(hold.occ)
            sigmah = SigmaHartree(crystal=self.crystal,occ=hold.occ,vbare=vbare.k,hdf5file=fn,group=group)
            # sigmah.k = sigmah.Mixing(iter=iter,mix=mix,Fb=sigmah.k,Fm=hartreeold)
            if (iter % 50 == 0):
                sigmah.Save(f'sigh.{iter}')
            sigmaf = SigmaFock(crystal=self.crystal,occr=hold.occr,vbare=vbare.r,hdf5file=fn,group=group)
            # sigmaf.k = sigmaf.Mixing(iter=iter,mix=mix,Fb=sigmaf.k,Fm=fockold)
            if (iter % 50 == 0):
                sigmaf.Save(f'sigf.{iter}')
            hnew = Hamiltonian(crystal=self.crystal,ham=niham.k,beta=self.ft.beta,sigmah=sigmah.k,sigmaf=sigmaf.k,hdf5file=fn,group=group)
            # hnew = Hamiltonian(crystal=self.crystal,ham=niham.k,beta=self.ft.beta,sigmah=None,sigmaf=sigmaf,hdf5file=fn,group=group)
            if (iter % 50 == 0):
                hnew.Save(f'hk.{iter}')
            print(f"iteration : {iter}")
            if (iter % 10 == 0):
                fcheck = self.SCFCheck(hnew.kbare,hcheck.kbare)
                mucheck = abs(hnew.mu-hold.mu)
                hcheck = hnew
                print(f"criteria : {fcheck}\nchemical potential : {hnew.mu}")
            else:
                fcheck = 1
                mucheck = 1
            if (fcheck<=1.0e-9)and(mucheck<=0.01):
                print(f"Self-consistency is achived with {iter}-th")
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
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
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
                hnew.HMixing(iter=iter, mix=mix, hm=hold.k)
                hold=hnew
                hartreeold = sigmah.k
                fockold = sigmaf.k
                del sigmaf,sigmah,hnew
                # del sigmaf,hnew
                gc.collect()

        return None


    def GWApproximation(self, itermax : int, mix : float, hoppinglist : list = None, onsitelist : list = None, spin : bool = False, valley : bool = False, site : bool = False, aferro : bool = False, loccoulomb : dict = None, nonloccoulomb : list = None,ohno : bool = False, jth : bool = False, ohnoyuka : bool = False, hdf5file : str = 'glob.h5', group : str = 'gw'):

        errmessage = "missing input for GW calculation"
        if (hoppinglist==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()

        niham = NIHamiltonian(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,hdf5file=hdf5file,group=group)
        gbare = GreenBare(crystal=self.crystal,ft=self.ft,hamtb=niham.k,hdf5file=hdf5file,group=group)
        vbare = VBare(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=hdf5file,group=group)
        self.vbare = vbare

        for iter in range(1,itermax+1):
            if iter == 1:
                # niham_temp = NIHamiltonian(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin, valley=valley, hdf5file=hdf5file,group='test') 
                niham_temp = NIHamiltonian(self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin,aferro=aferro, valley=valley,site=site,hdf5file=hdf5file,group='test_gw')
                gbare_temp = GreenBare(crystal=self.crystal,ft=self.ft,hamtb=niham_temp.k,hdf5file=hdf5file,group='test') 
                gold = GreenInt(crystal=self.crystal,ft=self.ft,greenbare=gbare_temp.kf,hdf5file=hdf5file,group=group)
                pkfold = None
                ckfold = None
                wold = 0
                # gbare.Save('gbare')


            print(gold.occ)
            print("Hartree calculation start")
            sigmah = SigmaHartree(crystal=self.crystal,occ=gold.occ,vbare=vbare.k,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                sigmah.Save(f'sigmah.{iter}')
            print("Hartree calculation finish")
            # exit()
            print("Fock calculation start")
            sigmaf = SigmaFock(crystal=self.crystal,occr=gold.occr,vbare=vbare.r,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                sigmaf.Save(f'sigmaf.{iter}')
            print("Fock calculation finish")
            print("Polarizability calculation start")
            pol = PolLat(crystal=self.crystal,ft=self.ft,green=gold.rt,hdf5file=hdf5file,group=group)
            pol.kf = pol.Mixing(iter=iter,mix=mix,Bb=pol.kf,Bold=pkfold)
            if (iter % 50 == 0):
                pol.Save(f'pkf.{iter}')
            print("Polarizability calculation finish")
            print("Screened coulomb interaction calculation start")
            w = WLat(crystal=self.crystal,ft=self.ft,pol=pol.kf,vbare=vbare,c=self.c,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                w.Save(f'wkf.{iter}')
            # w.Save(w.ckf,f'wckf.{iter}')
            print("Screened coulomb interaction calculation finish")
            print("GW self-energy calculation start")
            sigmagwc = SigmaGWC(crystal=self.crystal,ft=self.ft,green=gold.rt,wlat=w.crt,hdf5file=hdf5file,group=group)
            sigmagwc.kf = sigmagwc.Mixing(iter=iter,mix=mix,Fb=sigmagwc.kf,Fm=ckfold)
            if (iter % 50 == 0):
                sigmagwc.Save(f'sigmagwckf.{iter}')
            print("GW self-energy calculation finish")
            print("GW green's function calculation start")
            gnew = GreenInt(crystal=self.crystal,ft=self.ft,greenbare=gbare.kf,sigmah=sigmah.k,sigmaf=sigmaf.k,sigmagwc=sigmagwc.kf,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                gnew.Save(f'gkf.{iter}')
            print("GW green's function calculation start")

            fcheck = self.SCFCheck(gnew.kf,gold.kf)
            # bcheck = self.SCFCheck(w.kf,wold)
            mucheck = abs(gnew.mu-gold.mu)

            print(f"iteration : {iter} \nfcriteria : {fcheck} \nchemicalpotential : {gnew.mu}")

            if (fcheck <=1.0e-6)and(mucheck<=0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.greenbare = gbare
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
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.greenbare = gbare
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
            
            # self.green = gbare_temp
                
        return None
    


    def GWApproximation_new(self, itermax : int, mix : float, hoppinglist : list = None, onsitelist : list = None, spin : bool = False, valley : bool = False, site : bool = False, aferro : bool = False, loccoulomb : dict = None, nonloccoulomb : list = None,ohno : bool = False, jth : bool = False, ohnoyuka : bool = False, hdf5file : str = 'glob.h5', group : str = 'gw'):

        errmessage = "missing input for GW calculation"
        if (hoppinglist==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()

        start = time.time()
        niham = NIHamiltonian(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,hdf5file=hdf5file,group=group)
        end = time.time()
        tiem_delta = end-start
        print("NIHamiltonian code      - ",round(tiem_delta,5),'seconds')

        start = time.time()
        gbare = GreenBare_new(crystal=self.crystal,ft=self.ft,hamtb=niham.k,hdf5file=hdf5file,group=group)
        end = time.time()
        tiem_delta = end-start
        print("GreenBare code      - ",round(tiem_delta,5),'seconds')

        start = time.time()
        vbare = VBare(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=hdf5file,group=group)
        self.vbare = vbare
        end = time.time()
        tiem_delta = end-start
        print("VBare code      - ",round(tiem_delta,5),'seconds')

        for iter in range(1,itermax+1):
            if iter == 1:
                start = time.time()
                # niham_temp = NIHamiltonian(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin, valley=valley, hdf5file=hdf5file,group='test') 
                niham_temp = NIHamiltonian(self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin,aferro=aferro, valley=valley,site=site,hdf5file=hdf5file,group='test_gw')
                end = time.time()
                tiem_delta = end-start
                print("NIHamiltonian code      - ",round(tiem_delta,5),'seconds')   
                start = time.time()
                gbare_temp = GreenBare_new(crystal=self.crystal,ft=self.ft,hamtb=niham_temp.k,hdf5file=hdf5file,group='test') 
                end = time.time()
                tiem_delta = end-start
                print("GreenBare code      - ",round(tiem_delta,5),'seconds')
                start = time.time()
                gold = GreenInt_new(crystal=self.crystal,ft=self.ft,greenbare=gbare_temp.kf,hdf5file=hdf5file,group=group)
                end = time.time()
                tiem_delta = end-start
                print("GreenInt code      - ",round(tiem_delta,5),'seconds')
                pkfold = None
                ckfold = None
                wold = 0
                # gbare.Save('gbare')
            
            # exit()


            print(gold.occ)
            print("Hartree calculation start")
            sigmah = SigmaHartree(crystal=self.crystal,occ=gold.occ,vbare=vbare.k,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                sigmah.Save(f'sigmah.{iter}')
            print("Hartree calculation finish")
            # exit()
            print("Fock calculation start")
            sigmaf = SigmaFock(crystal=self.crystal,occr=gold.occr,vbare=vbare.r,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                sigmaf.Save(f'sigmaf.{iter}')
            print("Fock calculation finish")
            print("Polarizability calculation start")
            # start = time.time()
            pol = PolLat_new(crystal=self.crystal,ft=self.ft,green=gold.rt,hdf5file=hdf5file,group=group)
            pol.kf = pol.Mixing(iter=iter,mix=mix,Bb=pol.kf,Bold=pkfold)
            if (iter % 50 == 0):
                pol.Save(f'pkf.{iter}')
            # end = time.time()
            # tiem_delta = end-start
            # print("PolLat_new code      - ",round(tiem_delta,5),'seconds')
            print("Polarizability calculation finish")
            print("Screened coulomb interaction calculation start")
            start = time.time()
            w = WLat_new(crystal=self.crystal,ft=self.ft,pol=pol.kf,vbare=vbare,c=self.c,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                w.Save(f'wkf.{iter}')
            # w.Save(w.ckf,f'wckf.{iter}')
            end = time.time()
            tiem_delta = end-start
            print("WLat_new code      - ",round(tiem_delta,5),'seconds')
            print("Screened coulomb interaction calculation finish")
            print("GW self-energy calculation start")
            start = time.time()
            sigmagwc = SigmaGWC_new(crystal=self.crystal,ft=self.ft,green=gold.rt,wlat=w.crt,hdf5file=hdf5file,group=group)
            sigmagwc.kf = sigmagwc.Mixing(iter=iter,mix=mix,Fb=sigmagwc.kf,Fm=ckfold)
            if (iter % 50 == 0):
                sigmagwc.Save(f'sigmagwckf.{iter}')
            end = time.time()
            tiem_delta = end-start
            print("SigmaGWC_new code      - ",round(tiem_delta,5),'seconds')
            print("GW self-energy calculation finish")
            print("GW green's function calculation start")
            gnew = GreenInt_new(crystal=self.crystal,ft=self.ft,greenbare=gbare.kf,sigmah=sigmah.k,sigmaf=sigmaf.k,sigmagwc=sigmagwc.kf,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                gnew.Save(f'gkf.{iter}')
            print("GW green's function calculation start")

            fcheck = self.SCFCheck(gnew.kf,gold.kf)
            # bcheck = self.SCFCheck(w.kf,wold)
            mucheck = abs(gnew.mu-gold.mu)

            print(f"iteration : {iter} \nfcriteria : {fcheck} \nchemicalpotential : {gnew.mu}")

            if (fcheck <=1.0e-6)and(mucheck<=0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.greenbare = gbare
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
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.greenbare = gbare
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
            
            # self.green = gbare_temp
                
        return None
    



    def GW_EDMFT(self, itermax, imp, equiv, 
                 mix : float, hoppinglist : list = None, onsitelist : list = None, spin : bool = False, 
                 valley : bool = False, site : bool = False, aferro : bool = False, 
                 loccoulomb : dict = None, nonloccoulomb : list = None,
                 ohno : bool = False, jth : bool = False, ohnoyuka : bool = False, hdf5file : str = 'glob.h5', group : str = 'gw_edmft'):
    
        """
        Implement GW directly under the GW+EDMFT cycle (copy/paste Seongjun's GWApproximation method)

        1: GW cycle
        2: Double Counting cycle
        3: EDMFT cycle
        4: compute total G and W

        Minor change in GW part (changed the names): 
        sigmah -> sigmah_gw
        sigmaf -> sigmaf_gw
        sigmagwc -> sigmac_gw
        pol -> pol_gw

        use GreenInt/WLat classes to compute the total G and W (Check out GreenInt_EDMFT/W_EDMFT classes)

        Question: 
        Do we need to include the Mixing procedure in the DC and EDMFT parts?
        How should the Mixing procedure of pol_gw and sigmac_gw be computed after completing the full self-consistency cycles?

        """

        errmessage = "missing input for GW_EDMFT calculation"
        if (hoppinglist==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()


        niham = NIHamiltonian(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,hdf5file=hdf5file,group=group)
        gbare = GreenBare(crystal=self.crystal,ft=self.ft,hamtb=niham.k,hdf5file=hdf5file,group=group)
        vbare = VBare(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=hdf5file,group=group)
        self.vbare = vbare
        self.greenbare = gbare



        ########################################################################################
        ### build CTQMC class in advance & get the number of problem space for key iteration ###
        ########################################################################################
            
        ### build a CTQMC class in advance
        ctqmc = CTQMC(crystal=self.crystal,ft=self.ft)
        ### get the number of problem space for key iteration
        nprob = len(self.crystal.probspace)

        ###############################
        ### compute vloc in advance ###
        ###############################
        vloc = self.vbare.vloc                   ### get the vloc from GWApproximation result
        vloc.compute_vloc_projection_on_space()  ### compute projected complex vloc with the nspace dimension
        ### shouldn't it be in nprob, not in nspace?











        iter = 0

        #############################
        
        # niham_temp = NIHamiltonian(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin, valley=valley, hdf5file=hdf5file,group='test') 
        niham_temp = NIHamiltonian(self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin,aferro=aferro, valley=valley,site=site,hdf5file=hdf5file,group='test_gw')
        gbare_temp = GreenBare(crystal=self.crystal,ft=self.ft,hamtb=niham_temp.k,hdf5file=hdf5file,group='test') 
        gold = GreenInt(crystal=self.crystal,ft=self.ft,greenbare=gbare_temp.kf,hdf5file=hdf5file,group=group)
        pkfold = None
        ckfold = None
        # wold = 0
        # gbare.Save('gbare')


        # print(gold.occ)
        print("Hartree calculation start")
        sigmah_gw = SigmaHartree(crystal=self.crystal,occ=gold.occ,vbare=vbare.k,hdf5file=hdf5file,group=group)
        # if (iter % 50 == 0):
        #     sigmah.Save(f'sigmah.{iter}')
        print("Hartree calculation finish")

        print("Fock calculation start")
        sigmaf_gw = SigmaFock(crystal=self.crystal,occr=gold.occr,vbare=vbare.r,hdf5file=hdf5file,group=group)
        # if (iter % 50 == 0):
        #     sigmaf.Save(f'sigmaf.{iter}')
        print("Fock calculation finish")

        print("Polarizability calculation start")
        pol_gw = PolLat(crystal=self.crystal,ft=self.ft,green=gold.rt,hdf5file=hdf5file,group=group)
        # pol_gw.kf = pol_gw.Mixing(iter=iter+1,mix=mix,Bb=pol_gw.kf,Bold=pkfold)
        # if (iter % 50 == 0):
        #     pol.Save(f'pkf.{iter}')
        print("Polarizability calculation finish")

        print("Screened coulomb interaction calculation start")
        wnew = WLat(crystal=self.crystal,ft=self.ft,pol=pol_gw.kf,vbare=vbare,c=self.c,hdf5file=hdf5file,group=group)
        # if (iter % 50 == 0):
        #     w.Save(f'wkf.{iter}')
        # w.Save(w.ckf,f'wckf.{iter}')
        print("Screened coulomb interaction calculation finish")

        print("GW self-energy calculation start")
        sigmac_gw = SigmaGWC(crystal=self.crystal,ft=self.ft,green=gold.rt,wlat=wnew.crt,hdf5file=hdf5file,group=group)
        # sigmac_gw.kf = sigmac_gw.Mixing(iter=iter+1,mix=mix,Fb=sigmac_gw.kf,Fm=ckfold)
        # if (iter % 50 == 0):
        #     sigmagwc.Save(f'sigmagwckf.{iter}')
        print("GW self-energy calculation finish")

        ############################
        ### Green's function - G ###
        ############################
        gnew = GreenInt(crystal=self.crystal,ft=self.ft,greenbare=gbare.kf,sigmah=sigmah_gw.k,sigmaf=sigmaf_gw.k,sigmagwc=sigmac_gw.kf,hdf5file=hdf5file,group=group)
        # if (iter % 50 == 0):
        #     gnew.Save(f'gkf.{iter}')

        fcheck = self.SCFCheck(gnew.kf,gold.kf)
        # bcheck = self.SCFCheck(w.kf,wold)
        mucheck = abs(gnew.mu-gold.mu)
        print(f"iteration : {iter} \nfcriteria : {fcheck} \nchemicalpotential : {gnew.mu}")


        # # gold = gnew
        # # # ckfold = sigmac_gw.kf
        # # # pkfold = pol.kf
        # # # wold = w.kf

        # # ########################################
        # # ### Screen's Coulomb Interaction - W ###
        # # ########################################
        # # # wold = np.copy(wnew)
        # # wold = wnew


        # gloc, wloc = self.Loc_Projection(gnew,wnew)

        # sigmaf_dc,sigmac_dc,pol_dc = self.compute_DC_part(gloc,wloc,vloc) ## passing VLoc class

        # for key in range(1,nprob+1):
        #     # if iter == 1:
        #     ## First iteration uses DC part for IMP calculations
        #     (sigmah_edmft,
        #     sigmaf_edmft,
        #     sigmac_edmft,
        #     pol_edmft,
        #     sigmah_dc) = self.compute_IMP_part(iter,ctqmc,key,imp,equiv,gnew.mu,gloc,wloc,vloc,sigmah_gw,sigmaf_gw,pol_dc,sigmaf_dc,sigmac_dc)


        # # # sigmah_edmft = SigmaHImp(self.crystal)
        # # sigmaf_edmft = SigmaFImp(self.crystal)
        # # sigmac_edmft = SigmaCImp(self.crystal,self.ft)
        # # pol_edmft = PolImp(self.crystal,self.ft)

        # # # sigmah_edmft.k_embedded = sigmah_dc.k_embedded
        # # sigmaf_edmft.k_embedded = sigmaf_dc.k_embedded
        # # sigmac_edmft.kf_embedded = sigmac_dc.kf_embedded

        # # pol_edmft.kf_embedded = pol_dc.kf_embedded

        # # exit()

        # ####################################################################################
        # ##############    Combined GW, DC, and EDMFT to build full G and W    ##############
        # ####################################################################################

        # #### merge GW+EDMFT+DC to compute the total G and W (namely update total G and W)
        # print("\n*** Compute the total Screened Coulomb Interaction -- W_EDMFT")
        # wnew = self.W_EDMFT(self.vbare, pol_gw.kf, pol_edmft.kf_embedded, pol_dc.kf_embedded)

        # print("\n*** Compute the total Interacting Green's function -- GreenInt_EDMFT")
        # gnew = self.GreenInt_EDMFT(self.greenbare,sigmah_gw.k,sigmaf_gw.k,sigmac_gw.kf
        #                         ,sigmaf_edmft.k_embedded,sigmac_edmft.kf_embedded
        #                         ,sigmaf_dc.k_embedded,sigmac_dc.kf_embedded)






        
        
        
        
        
        
        
        """
        input : gold,
                wold,
                vbare, 
                vloc, 
                nprob, 
                imp, 
                CTQMC
        """

        gold = gnew
        wold = wnew

        for iter in range(1,itermax+1):

            #############################################################
            ##############    GW part (Seongjun's code)    ##############
            #############################################################

            sigmah_gw, sigmaf_gw, sigmac_gw, pol_gw = self.GW_update(gold,wold,vbare,mix,hdf5file,group)


            #######################################################
            ##############    Loc projection part    ##############
            #######################################################

            gloc, wloc = self.Loc_Projection(gold,wold)


            ########################################################
            ##############    Double Counting part    ##############
            ########################################################

            sigmaf_dc, sigmac_dc, pol_dc = self.compute_DC_part(gloc,wloc,vloc) ## passing VLoc class


            ###################################################
            ##############    EDMFT(IMP) part    ##############
            ###################################################

            #### EDMFT part ####
            for key in range(1,nprob+1):
                if iter == 1:
                    ## First iteration uses DC part for IMP calculations
                    (sigmah_edmft,
                    sigmaf_edmft,
                    sigmac_edmft,
                    pol_edmft,
                    sigmah_dc) = self.compute_IMP_part(iter,ctqmc,key,imp,equiv,gold.mu,gloc,wloc,vloc,sigmah_gw,sigmaf_gw,pol_dc,sigmaf_dc,sigmac_dc)
                else:
                    (sigmah_edmft,
                    sigmaf_edmft,
                    sigmac_edmft,
                    pol_edmft,
                    sigmah_dc) = self.compute_IMP_part(iter,ctqmc,key,imp,equiv,gold.mu,gloc,wloc,vloc,sigmah_gw,sigmaf_gw,pol_dc,sigmaf_dc,sigmac_dc,sigmah_edmft,sigmaf_edmft,sigmac_edmft)
                
            



            ####################################################################################
            ##############    Combined GW, DC, and EDMFT to build full G and W    ##############
            ####################################################################################

            # #### merge GW+EDMFT+DC to compute the total G and W (namely update total G and W)
            # print("\n*** Compute the total Screened Coulomb Interaction -- W_EDMFT")
            # wnew = self.W_EDMFT(self.vbare, pol_gw.kf, pol_edmft.kf_embedded, pol_dc.kf_embedded)

            # print("\n*** Compute the total Interacting Green's function -- GreenInt_EDMFT")
            # gnew = self.GreenInt_EDMFT(self.greenbare,sigmah_gw.k,sigmaf_gw.k,sigmac_gw.kf
            #                         ,sigmaf_edmft.k_embedded,sigmac_edmft.kf_embedded
            #                         ,sigmaf_dc.k_embedded,sigmac_dc.kf_embedded)
            

            #### merge GW+EDMFT+DC to compute the full G and W (namely update full G and W)
            self.full_sigmah_k = sigmah_gw.k
            self.full_sigmaf_k = sigmaf_gw.k + sigmaf_edmft.k_embedded - sigmaf_dc.k_embedded
            self.full_sigmac_kf = sigmac_gw.kf + sigmac_edmft.kf_embedded - sigmac_dc.kf_embedded
            gnew = GreenInt(crystal=self.crystal,ft=self.ft,greenbare=gbare.kf,sigmah=self.full_sigmah_k,sigmaf=self.full_sigmaf_k,sigmagwc=self.full_sigmac_kf)

            self.full_pol_kf = pol_gw.kf + pol_edmft.kf_embedded - pol_dc.kf_embedded
            wnew = WLat(crystal=self.crystal,ft=self.ft,pol=self.full_pol_kf,vbare=self.vbare,c=self.c)


            ##########################
            ###    SCF Checking    ###
            ##########################
            
            fcheck = self.SCFCheck(gnew.kf,gold.kf)
            # bcheck = self.SCFCheck(w.kf,wold)
            mucheck = abs(gnew.mu-gold.mu)
            print(f"iteration : {iter} \nfcriteria : {fcheck} \nchemicalpotential : {gnew.mu}")

            if (fcheck <=1.0e-6)and(mucheck<=0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.greenbare = gbare
                self.green = gnew
                # self.pol = pol
                self.w = wnew
                self.sigmagwc = sigmac_gw
                self.sigmaf = sigmaf_gw
                self.sigmah = sigmah_gw
                # gnew.Save('gkf',chem=True)
                # sigmah.Save('sigmah')
                # sigmaf.Save('sigmaf')
                # sigmagwc.Save('sigmagwckf')
                # pol.Save('pkf')
                # w.Save('wkf')
                # # self.sigmagwc.SigmaStc()
                # # self.sigmagwc.Zfactor()
                # del niham, vbare, gbare, gnew, gold, sigmaf, sigmah, sigmagwc, pol, w
                # gc.collect()
                break
            elif (iter==itermax):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.greenbare = gbare
                self.green = gnew
                # self.pol = pol
                self.w = wnew
                self.sigmagwc = sigmac_gw
                self.sigmaf = sigmaf_gw
                self.sigmah = sigmah_gw
                # gnew.Save('gkf',chem=True)
                # sigmah.Save('sigmah')
                # sigmaf.Save('sigmaf')
                # sigmagwc.Save('sigmagwckf')
                # pol.Save('pkf')
                # w.Save('wkf')
                # # self.sigmagwc.SigmaStc()
                # # self.sigmagwc.Zfactor()
                # del niham, vbare, gbare, gnew, gold, sigmaf, sigmah, sigmagwc, pol, w
                # gc.collect()
            else:
                gold = gnew
                wold = wnew
                # ckfold = sigmac_gw.kf
                # pkfold = pol_gw.kf
                

                # del gnew, sigmah, sigmaf, sigmagwc, pol, w
                # gc.collect()
            
            
            print(f"\niteration : {iter} is done")
            print('=================================================\n')
            print('\n')



        return None
    




    # def GW_EDMFT_x(self, itermax, imp, equiv, gint : GreenInt=None, vloc : VLoc=None): ### gint should be class GreenInt and vloc should be class VLoc(BLocStc)

    #     #### could put GWApproximation directly here to calculate GW first

    #     # self.GWApproximation(
    #     # itermax=itermax,
    #     # mix=mix,
    #     # hoppinglist=hopping,
    #     # loccoulomb=locoption,
    #     # nonloccoulomb=nonlocoption,
    #     # )


    #     #### if GW is pre-calculated, then get the gint and vloc from GW calculation

    #     if (gint is None) and (vloc is None):
    #         gint = self.green                        ### get the gint from GWApproximation result
    #         vloc = self.vbare.vloc                   ### get the vloc from GWApproximation result
    #         vloc.compute_vloc_projection_on_space()  ### compute projected complex vloc with the nspace dimension
    #         ### shouldn't it be in nprob, not in nspace?


    #     ### build a CTQMC class in advance
    #     ctqmc = CTQMC(crystal=self.crystal,ft=self.ft)
    #     ### get the number of problem space for key iteration
    #     nprob = len(self.crystal.probspace)




    #     ########################
    #     ### iteration starts ###
    #     ########################

    #     ### iteration start from 0 -- 0-based (note that inside the calculation it will be converted to 1-based)
    #     iter = 0

    #     ### Loc (Projection) part first
    #     gloc, wloc = self.Loc_Projection(gint,self.w)

    #     #### DC (Double Counting) part #### input: vloc/gloc/wloc, output: Sigmaf_dc, sigmac_dc, pi_dc
    #     sigmaf_dc,sigmac_dc,pi_dc = self.compute_DC_part(gloc,wloc,vloc) ## passing VLoc class to the argument

    #     #### EDMFT part ####
    #     for key in range(1,nprob+1):
    #         (sigmah_edmft,
    #         sigmaf_edmft,
    #         sigmac_edmft,
    #         pi_edmft,
    #         sigmah_dc) = self.compute_IMP_part(iter,key,imp,equiv,gint.mu,gloc,wloc,vloc,pi_dc,sigmaf_dc,sigmac_dc,ctqmc) ## passing VLoc class to the argument
    #         ### remove gint, instead input gint.mu, niham as input

    #     #### merge GW+EDMFT+DC to compute the total G and W (namely update total G and W)
    #     print("\n*** Compute the total Screened Coulomb Interaction -- W_EDMFT")
    #     w_edmft = self.W_EDMFT(self.vbare, self.pol, pi_edmft.kf_embedded, pi_dc.kf_embedded)

    #     print("\n*** Compute the total Interacting Green's function -- GreenInt_EDMFT")
    #     gint_edmft = self.GreenInt_EDMFT(self.greenbare,self.sigmah,self.sigmaf,self.sigmagwc
    #                             ,sigmah_edmft.k_embedded,sigmaf_edmft.k_embedded,sigmac_edmft.kf_embedded
    #                             ,sigmah_dc.k_embedded,sigmaf_dc.k_embedded,sigmac_dc.kf_embedded)
        
    #     # exit()
        
    #     for iter in range(1,itermax): ### should be starting from 1, since 0th iteration has already been done above

    #         #### GW part update ####
    #         sigmah_gw,sigmaf_gw,sigmac_gw,pol_gw = self.GW_update(gint_edmft,self.vbare)
    #         ### Loc (Projection) part first
    #         gloc, wloc = self.Loc_Projection(gint_edmft,w_edmft)
    #         #### DC (Double Counting) part ####
    #         sigmaf_dc,sigmac_dc,pi_dc = self.compute_DC_part(gloc,wloc,vloc)
    #         #### EDMFT part ####
    #         for key in range(1,nprob+1):

    #             (sigmah_edmft,
    #             sigmaf_edmft,
    #             sigmac_edmft,
    #             pi_edmft,
    #             sigmah_dc) = self.compute_IMP_part(iter,key,imp,equiv,gint_edmft.mu,gloc,wloc,vloc,pi_edmft,sigmaf_edmft,sigmac_edmft,ctqmc)

    #             ## remove gint instead gint.mu

    #         #### merge GW+EDMFT+DC to compute the total G and W (namely update total G and W)
    #         print("\n*** Compute the total Screened Coulomb Interaction -- W_EDMFT")
    #         w_edmft = self.W_EDMFT(self.vbare, pol_gw, pi_edmft.kf_embedded, pi_dc.kf_embedded)

    #         print("\n*** Compute the total Interacting Green's function -- GreenInt_EDMFT")
    #         gint_edmft = self.GreenInt_EDMFT(self.greenbare,sigmah_gw,sigmaf_gw,sigmac_gw
    #                                 ,sigmah_edmft.k_embedded,sigmaf_edmft.k_embedded,sigmac_edmft.kf_embedded
    #                                 ,sigmah_dc.k_embedded,sigmaf_dc.k_embedded,sigmac_dc.kf_embedded)
                

    #     return None
    






    def Loc_Projection(self,gint,w):
        
        print("**GreenLoc start")
        gloc = GreenLoc(crystal=self.crystal, ft=self.ft, green=gint)
        print("**GreenLoc finish\n")

        print("**WLoc start")
        start = time.time()
        # wloc = WLoc_temp(crystal=self.crystal, ft=self.ft, pol=pi_dc.rf, vLoc=vloc.zvloc)
        wloc = WLoc(crystal=self.crystal, ft=self.ft, wlat=w)
        end = time.time()
        tiem_delta = end-start
        print(round(tiem_delta,5))
        print("**WLoc finish\n")

        return gloc, wloc




    ### used for GW_EDMFT
    def compute_DC_part(self,gloc,wloc,vloc):

        print("**PolLoc start")
        pi_dc = PolLGW(crystal=self.crystal, ft=self.ft, green=gloc)
        # pollat = PolLat(crystal=cf.crystal, ft=cf.ft, green=g_average2)
        print("**PolLoc finish\n")

        print("**SigmaFLoc start")
        sigmaf_dc = SigmaFLoc(crystal=self.crystal, ft=self.ft, occr=gloc.occ, vloc=vloc.zvloc)
        print("**SigmaFLoc finish\n")

        print("**SigmaLGWC start")
        sigmac_dc = SigmaLGWC(crystal=self.crystal, ft=self.ft, green=gloc.gt, wloc=wloc.crt)
        print("**SigmaLGWC finish\n")

        sigmaf_dc.embedding()
        sigmac_dc.embedding()
        pi_dc.embedding()

        return sigmaf_dc,sigmac_dc,pi_dc
        

    ### used for GW_EDMFT
    def compute_IMP_part(self,iter,ctqmc,key,imp,equiv,mu,gloc,wloc,vloc,sigmah,sigmaf,pi_dc,sigmaf_dc,sigmac_dc,sigmah_imp=None,sigmaf_imp=None,sigmac_imp=None): ### used for GW_EDMFT

        print("**BWeiss start")
        norb,_,ns,_,nft,nprob=wloc.crf.shape
        wloc_rf_temp = np.zeros((norb,norb,ns,ns,nft,nprob), dtype=np.complex128, order='F')
        for ift in range(nft):
            wloc_rf_temp[:,:,:,:,ift,:] = wloc.crf[:,:,:,:,ift,:] + vloc.zvloc[...]

        uimp = BWeiss(crystal=self.crystal,ft=self.ft,wloc=wloc_rf_temp,ploc=pi_dc.rf,vloc=vloc.zvloc)
        # uimp = BWeiss(crystal=cf.crystal,ft=cf.ft,wloc=wloc.rf,ploc=pi_dc.rf,vloc=vloc)
        print("**BWeiss finish\n")

        print("**SigmaHLoc start")
        sigmah_dc = SigmaHLoc(crystal=self.crystal, occ=gloc.occ, vloc=uimp.utilde_rf)
        # hloc = SigmaHLoc(crystal=cf.crystal, occ=gloc.occ, vloc=vloc)
        print("**SigmaHLoc finish\n")

        print("***Weiss Green's function start")
        if iter == 1 :
            weiss_green = FWeiss(crystal=self.crystal,ft=self.ft,niham=self.niham.k,mu=mu,hamh=sigmah.k,hamf=sigmaf.k,hloc=sigmah_dc.r,floc=sigmaf_dc.r,gloc=gloc.gf,sigmahimp=sigmah_dc.r,sigmafimp=sigmaf_dc.r,sigmacimp=sigmac_dc.rf)
        else:
            weiss_green = FWeiss(crystal=self.crystal,ft=self.ft,niham=self.niham.k,mu=mu,hamh=sigmah.k,hamf=sigmaf.k,hloc=sigmah_dc.r,floc=sigmaf_dc.r,gloc=gloc.gf,sigmahimp=sigmah_imp.r,sigmafimp=sigmaf_imp.r,sigmacimp=sigmac_imp.rf)
        print("***Weiss Green's function finish")



        print('\n*** write_ctqmc_params start ***')
        ctqmc.PreProcessing(iter=iter, key=key, Eimp=weiss_green.Eimp_r, imp=imp, equiv=equiv, vloc=vloc, hyb=weiss_green.delta_rf, bweiss=uimp.utilde_rf)
        print('*** write_ctqmc_params finish ***')


        print('\n*** run and measure CTQMC start ***')
        ctqmc.Run()
        ctqmc.Measure()
        print('*** run and measure CTQMC finish ***')

        

        print('\n*** impurity postprocessing start ***')
        # (sigmah_edmft, 
        # sigmaf_edmft, 
        # sigmac_edmft, 
        # pi_edmft) = ctqmc.PostProcessing(iter=iter,key=key,equiv=equiv,utilde_rf=uimp.utilde_rf)
        (Green,
         Sigma_hf,
         Sigma_bare,
         susceptibility) = ctqmc.PostProcessing(iter=iter,key=key,equiv=equiv,utilde_rf=uimp.utilde_rf)
        (sigmah_edmft, 
        sigmaf_edmft, 
        sigmac_edmft, 
        pi_edmft) = ctqmc.PostProcessing2(iter=iter,key=key,Green=Green,Sigma_hf=Sigma_hf,Sigma_bare=Sigma_bare,susceptibility=susceptibility,utilde_rf=uimp.utilde_rf)
        print('*** impurity postprocessing finish ***\n')



        print('\n*** compute Embedding part')
        print('** double counting part')
        sigmah_dc.embedding()
        # sigmaf_dc.embedding()
        # sigmac_dc.embedding()
        # pi_dc.embedding()

        print('** edmft part')
        sigmah_edmft.embedding()
        sigmaf_edmft.embedding()
        sigmac_edmft.embedding()
        pi_edmft.embedding()

        return sigmah_edmft,sigmaf_edmft,sigmac_edmft,pi_edmft,sigmah_dc
    

    ### used for GW_EDMFT
    # def GreenInt_EDMFT(self, gbare : GreenBare, sigmah_gw : np.ndarray, sigmaf_gw : np.ndarray, sigmac_gw : np.ndarray
    #                            ,sigmah_edmft : np.ndarray, sigmaf_edmft : np.ndarray, sigmac_edmft : np.ndarray
    #                            ,sigmah_dc : np.ndarray, sigmaf_dc : np.ndarray, sigmac_dc : np.ndarray):
    def GreenInt_EDMFT(self, gbare : GreenBare, sigmah_gw : np.ndarray, sigmaf_gw : np.ndarray, sigmac_gw : np.ndarray
                               ,sigmaf_edmft : np.ndarray, sigmac_edmft : np.ndarray
                               ,sigmaf_dc : np.ndarray, sigmac_dc : np.ndarray):

        sigmah_k = sigmah_gw# + sigmah_edmft - sigmah_dc
        sigmaf_k = sigmaf_gw + sigmaf_edmft - sigmaf_dc
        sigmac_kf = sigmac_gw + sigmac_edmft - sigmac_dc  ## should the sigmac_gw.rf needs to be converted from sigmac_gw.kf ?

        # green_edmft = GreenInt_EDMFT(crystal=self.crystal,ft=self.ft,greenbare=gbare.kf,sigmah=sigmah_k,sigmaf=sigmaf_k,sigmagwc=sigmac_kf)
        green_edmft = GreenInt(crystal=self.crystal,ft=self.ft,greenbare=gbare.kf,sigmah=sigmah_k,sigmaf=sigmaf_k,sigmagwc=sigmac_kf)

        # # N = self.green.NumOfE(self.green.mu)
        # # print("NumofE -- GW             :",N)
        # N = green_edmft.NumOfE(green_edmft.mu)
        # print("NumofE -- after embedding:",N)
        # # print("Chemical potential -- GW             :",self.green.mu)
        # print("Chemical potential -- after embedding:",green_edmft.mu)
        # print("====")
        # # print(self.green.kf[:,:,0,0,0])
        # print(green_edmft.kf[:,:,0,0,0])
        # # print(self.green.rf[:,:,0,0,0])
        # print(green_edmft.rf[:,:,0,0,0])
        

        return green_edmft
    
    ### used for GW_EDMFT
    def W_EDMFT(self, vbare : VBare, pol_gw : np.ndarray, pol_edmft : np.ndarray, pol_dc : np.ndarray):

        #### put embedding part to here

        pol_kf = pol_gw + pol_edmft - pol_dc

        # w_edmft = pol_gw.Dyson(mat1, mat2)
        w_edmft = WLat(crystal=self.crystal,ft=self.ft,pol=pol_kf,vbare=vbare,c=self.c)

        return w_edmft
    
    ### used for GW_EDMFT
    def GW_update(self,gold,wold,vbare,mix,hdf5file,group):

        # print(gold.occ)
        print("Hartree calculation start")
        sigmah_gw = SigmaHartree(crystal=self.crystal,occ=gold.occ,vbare=vbare.k,hdf5file=hdf5file,group=group)
        # if (iter % 50 == 0):
        #     sigmah.Save(f'sigmah.{iter}')
        print("Hartree calculation finish")

        print("Fock calculation start")
        sigmaf_gw = SigmaFock(crystal=self.crystal,occr=gold.occr,vbare=vbare.r,hdf5file=hdf5file,group=group)
        # if (iter % 50 == 0):
        #     sigmaf.Save(f'sigmaf.{iter}')
        print("Fock calculation finish")

        print("Polarizability calculation start")
        pol_gw = PolLat(crystal=self.crystal,ft=self.ft,green=gold.rt,hdf5file=hdf5file,group=group)
        # pol_gw.kf = pol_gw.Mixing(iter=iter,mix=mix,Bb=pol_gw.kf,Bold=pkfold)
        # if (iter % 50 == 0):
        #     pol.Save(f'pkf.{iter}')
        print("Polarizability calculation finish")

        # print("Screened coulomb interaction calculation start")
        # w_gw = WLat(crystal=self.crystal,ft=self.ft,pol=pol_gw.kf,vbare=vbare,c=self.c,hdf5file=hdf5file,group=group)
        # # if (iter % 50 == 0):
        # #     w.Save(f'wkf.{iter}')
        # # w.Save(w.ckf,f'wckf.{iter}')
        # print("Screened coulomb interaction calculation finish")

        print("GW self-energy calculation start")
        sigmac_gw = SigmaGWC(crystal=self.crystal,ft=self.ft,green=gold.rt,wlat=wold.crt,hdf5file=hdf5file,group=group)
        # sigmac_gw.kf = sigmac_gw.Mixing(iter=iter,mix=mix,Fb=sigmac_gw.kf,Fm=ckfold)
        # if (iter % 50 == 0):
        #     sigmagwc.Save(f'sigmagwckf.{iter}')
        print("GW self-energy calculation finish")

        return sigmah_gw,sigmaf_gw,sigmac_gw,pol_gw
