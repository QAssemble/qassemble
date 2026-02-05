import numpy as np
import matplotlib.pyplot as plt
import sys, os
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

        check = 0
        tempmat = abs(mat1-mat2)
        check = tempmat.max()
        return check

    def TightBinding(self, hopping : dict = None, onsite : dict = None, spin : bool = False, site : bool = False, valley : bool = False, hdf5file : str = 'glob.h5'):

        # file = h5py.File(fn+'.h5','w')
        # tb = file.create_group('tb')

        group = 'tb'
        errmessage = "missing input for tight binding calculation"
        if (hopping == None):
            print(errmessage)
            sys.exit()
        # niham = NIHamiltonian(crystal=self.cry,hoppinglist=hoppinglist,onsitelist=onsitelist,hdf5file=tb)
        niham = NIHamiltonian(crystal=self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,hdf5file=hdf5file,group=group)
        self.niham = niham
        # file.close()

        return None

    def HartreeFock(self, itermax : int, mix : float, hopping : dict = None,mode : str = "FromScratch", onsite : dict = None, spin : bool = False, valley : bool = False, avalley : bool = False, site : bool = False, asite : bool = False, aferro : bool = False, loccoulomb : dict = None, nonloccoulomb : list = None, ohno : bool = False, jth : bool = False, ohnoyuka : bool = False, hdf5file : str = 'glob.h5', group : str = 'hf'):

        errmessage = "missing input for HF calculation"
        if (hopping==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()
        
        if (mode == 'FromScratch'):
            
            niham = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,hdf5file=hdf5file,group=group)
            vbare = VBare(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=hdf5file,group=group)
            self.vbare = vbare

        elif (mode == 'Restart'):
            group = group + '_restart'
            niham = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,hdf5file=hdf5file,group=group)
            vbare = VBare(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=hdf5file,group=group)
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
                    niham_temp = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,site=site,aferro=aferro, hdf5file=hdf5file,group='test_hf', avalley=avalley, asite=asite)
                    hold = Hamiltonian(crystal=self.crystal,ham=niham_temp.k,beta=self.dlr.beta,hdf5file=hdf5file,group=group)
                elif mode == "Restart":
                    niham_temp = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,site=site,aferro=aferro, hdf5file=None,group='test_hf', avalley=avalley, asite=asite)
                    glob = h5py.File(hdf5file,'r')
                    hf = glob['hf']
                    hk = hf['Hamiltonian']['hk'][:]
                    glob.close()
                    hold = Hamiltonian(crystal=self.crystal,ham=hk,beta=self.dlr.beta,hdf5file=hdf5file,group=group)
                    
                    

                hartreeold = None
                fockold = None

            print(hold.occ)
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
            print(f"iteration : {iter}\ncriteria : {fcheck}\nchemical potential : {hnew.mu}")
            if (fcheck<=1.0e-7)and(mucheck<=0.01):
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
                hold=hnew
                hartreeold = sigmah.k
                fockold = sigmaf.k
                del sigmaf,sigmah,hnew
                # del sigmaf,hnew
                gc.collect()


    def GWApproximation(self, itermax : int, mix : float, hoppinglist : list = None, onsitelist : list = None, spin : bool = False, valley : bool = False, site : bool = False, aferro : bool = False, loccoulomb : dict = None, nonloccoulomb : list = None,ohno : bool = False, jth : bool = False, ohnoyuka : bool = False, hdf5file : str = 'glob.h5', group : str = 'gw'):

        errmessage = "missing input for GW calculation"
        if (hoppinglist==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()
        
        niham = NIHamiltonian(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,hdf5file=hdf5file,group=group)
        gbare = GreenBare(crystal=self.crystal,dlr=self.dlr,hamtb=niham.k,hdf5file=hdf5file,group=group)
        vbare = VBare(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=hdf5file,group=group)


        for iter in range(1,itermax+1):
            if iter == 1:
                # niham_temp = NIHamiltonian(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin, valley=valley, hdf5file=hdf5file,group='test') 
                # niham_temp = NIHamiltonian(self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin,aferro=aferro, valley=valley,site=site,hdf5file=hdf5file,group='test_gw')
                # gbare_temp = GreenBare(crystal=self.crystal,dlr=self.dlr,hamtb=niham_temp.k,hdf5file=hdf5file,group='test') 
                gold = GreenInt(crystal=self.crystal,dlr=self.dlr,greenbare=gbare.kf,hdf5file=hdf5file,group=group)
                print(f"Initial chemical potential : {gold.mu}")
                gold.Save(f'gkf_ini')
                pkfold = None
                ckfold = None
                wold = 0
                # gbare.Save('gbare')

            print("Density Matrix :")
            print(gold.occ)
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
            pol = PolLat(crystal=self.crystal,dlr=self.dlr,green=gold.rt,hdf5file=hdf5file,group=group)
            # pol.kf = pol.Mixing(iter=iter,mix=mix,Bb=pol.kf,Bold=pkfold)
            # if (iter % 50 == 0)or(iter == 1):
            pol.Save(f'pkf.{iter}')
            # print("Polarizability calculation finish")
            # print("Screened coulomb interaction calculation start")
            w = WLat(crystal=self.crystal,dlr=self.dlr,pol=pol.kf,vbare=vbare,c=self.c,hdf5file=hdf5file,group=group)
            # if (iter % 50 == 0)or(iter == 1):
            w.Save(f'wkf.{iter}')
            # w.Save(w.ckf,f'wckf.{iter}')
            # print("Screened coulomb interaction calculation finish")
            # print("GW self-energy calculation start")
            sigmagwc = SigmaGWC(crystal=self.crystal,dlr=self.dlr,green=gold.rt,wlat=w.crt,hdf5file=hdf5file,group=group)
            # sigmagwc.kf = sigmagwc.Mixing(iter=iter,mix=mix,Fb=sigmagwc.kf,Fm=ckfold)
            # if (iter % 50 == 0)or(iter == 1):
            sigmagwc.Save(f'sigmagwckf.{iter}')
            # print("GW self-energy calculation finish")
            # print("GW green's function calculation start")
            gnew = GreenInt(crystal=self.crystal,dlr=self.dlr,greenbare=gbare.kf,sigmah=sigmah.k,sigmaf=sigmaf.k,sigmagwc=sigmagwc.kf,hdf5file=hdf5file,group=group)
            # if (iter % 50 == 0)or(iter == 1):
            gnew.Save(f'gkf.{iter}')
            # print("GW green's function calculation start")

            fcheck = self.SCFCheck(gnew.kf,gold.kf)
            
            bcheck = self.SCFCheck(w.kf,wold)
            mucheck = abs(gnew.mu-gold.mu)

            print(f"iteration : {iter} \nfcriteria : {fcheck} \nbcriteria : {bcheck} \nchemicalpotential : {gnew.mu+gnew.c}")
            # print(f"iteration : {iter} \nfcriteria : {fcheck} \nchemicalpotential : {gnew.mu}")

            if (fcheck <=1.0e-6)and(mucheck<=0.01)and(bcheck<=1.0e-4):
                print(f"Self-consistency is achived with {iter}-th")
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
