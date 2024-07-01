import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gc
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

class CorrelationFunction(object):

    def __init__(self,latt,basisposition,ns,soc,rkgrid,orboption,N,impdict = None, T = None, beta = None, size = None, c = 1.0):
        
        self.c = c
        self.tbham = None
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
        cry = Crystal(latt=latt,basisposition=basisposition,ns=ns,soc=soc,rkgrid=rkgrid,orboption=orboption,N=N)
        self.cry = cry
        ft = FTGrid(T=T,beta=beta,size=size)
        self.ft = ft

        if os.path.exists('work'):
            pass
        else:
            os.mkdir('work')

        
        

    def SCFCheck(self, mat1 : np.ndarray, mat2 : np.ndarray):

        check = 0
        tempmat = abs(mat1-mat2)
        check = tempmat.max()
        return check
    
    def TightBinding(self, hoppinglist : list = None, onsitelist : list = None):
        
        errmessage = "missing input for tight binding calculation"
        if (hoppinglist == None):
            print(errmessage)
            sys.exit()
        niham = NIHamiltonian(crystal=self.cry,hoppinglist=hoppinglist,onsitelist=onsitelist)
        self.tbham = niham.k

        return niham.k
    
    def HartreeFockH(self, itermax : int, mix : float, hoppinglist : list = None, onsitelist : list = None, loccoulomb : dict = None, nonloccoulomb : list = None,ohno : bool = False):

        errmessage = "missing input for HF calculation"
        if (hoppinglist==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()
        niham = NIHamiltonian(self.cry,hoppinglist=hoppinglist,onsitelist=onsitelist)
        vbare = VBare(crystal=self.cry,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno)
        self.vbare = vbare

        for iter in range(1, itermax):
            if iter==1:
                hold = Hamiltonian(crystal=self.cry,ham=niham.k,beta=self.ft.beta)
                hartreeold = None
                fockold = None
            
            print(hold.occ)
            sigmah = SigmaHartree(crystal=self.cry,occ=hold.occ,vbare=vbare.k)
            sigmah.k = sigmah.Mixing(iter=iter,mix=mix,Fb=sigmah.k,Fm=hartreeold)
            sigmah.Save(f'sigmah.{iter}')
            sigmaf = SigmaFock(crystal=self.cry,occr=hold.occr,vbare=vbare.r)
            sigmaf.k = sigmaf.Mixing(iter=iter,mix=mix,Fb=sigmaf.k,Fm=fockold)
            sigmaf.Save(f'sigmah.{iter}')
            hnew = Hamiltonian(crystal=self.cry,ham=self.TightBinding(hoppinglist=hoppinglist,onsitelist=onsitelist),beta=self.ft.beta,sigmah=sigmah,sigmaf=sigmaf)

            fcheck = self.SCFCheck(hnew.occk,hold.occk)
            mucheck = abs(hnew.mu-hold.mu)
            print(f"iteration : {iter}\ncriteria : {fcheck}\nchemical potential : {hnew.mu}")
            if (fcheck<=1.0e-4)and(mucheck<=0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.ham=hnew
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                hnew.Save('hamhf')
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                del hnew, sigmah, sigmaf, hold
                gc.collect()
                break
            elif(iter==itermax):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.ham=hnew
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                hnew.Save('hamhf')
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                del hnew, sigmah, sigmaf, hold
                gc.collect()
            else:
                hold = hnew
                hartreeold = sigmah.k
                fockold = sigmaf.k
                del sigmaf,sigmah,hnew
                gc.collect()

    def HartreeFock(self, itermax : int, mix : float, hoppinglist : list = None, onsitelist : list = None, loccoulomb : dict = None, nonloccoulomb : list = None,ohno : bool = False):

        errmessage = "missing input for HF calculation"
        if (hoppinglist==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()
        niham = NIHamiltonian(self.cry,hoppinglist=hoppinglist,onsitelist=onsitelist)
        vbare = VBare(crystal=self.cry,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno)
        self.vbare = vbare

        gbare = GreenBare(crystal=self.cry,ft=self.ft,hamtb=niham.k)
        self.greenbare = gbare

        for iter in range(1,itermax+1):
            if iter == 1:
                gold = GreenInt(crystal=self.cry,ft=self.ft, greenbare=gbare.kf)
                hartreeold = None
                fockold = None
            print(gold.occ)
            sigmah = SigmaHartree(crystal=self.cry,occ=gold.occ,vbare=vbare.k)
            sigmah.k = sigmah.Mixing(iter=iter,mix=mix,Fb=sigmah.k,Fm=hartreeold)
            sigmah.Save(f'sigmah.{iter}')
            sigmaf = SigmaFock(crystal=self.cry,occr=gold.occr,vbare=vbare.r)
            sigmaf.k = sigmaf.Mixing(iter=iter,mix=min,Fb=sigmaf.k,Fm=fockold)
            sigmaf.Save(f'sigmaf.{iter}')

            gnew = GreenInt(crystal=self.cry,ft=self.ft,greenbare=gbare.kf,sigmah=sigmah.k,sigmaf=sigmaf.k)

            fcheck = self.SCFCheck(gnew.occk,gold.occk)
            mucheck = abs(gnew.mu-gold.mu)
            print(f" iteration : {iter} \n criteria : {fcheck} \n chemicalpotential : {gnew.mu}")
            if (fcheck <=1.0e-3)and(mucheck<=0.001):
                print(f"Self-consistency is achived with {iter}-th")
                self.green = gnew
                self.sigmah = sigmah
                self.sigmaf = sigmaf
                chem = niham.ChemEmbedding(gnew.mu)
                flatstc = FLatStc(crystal=self.cry)
                self.hamhf = niham.k+sigmah.k+sigmaf.k-chem
                gnew.Save('gkf')
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                flatstc.Save(self.hamhf,'hamhf')
                del gnew, sigmaf, sigmah, flatstc, vbare, niham, gold, gbare
                gc.collect()
                break
            elif (iter==itermax):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.green = gnew
                self.sigmah = sigmah
                self.sigmaf = sigmaf
                chem = niham.ChemEmbedding(gnew.mu)
                flatstc = FLatStc(crystal=self.cry)
                self.hamhf = niham.k+sigmah.k+sigmaf.k-chem
                gnew.Save('gkf')
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                flatstc.Save(self.hamhf,'hamhf')
                del gnew, sigmaf, sigmah, flatstc, vbare, niham, gold, gbare
                gc.collect()
            else:
                gold = gnew
                hartreeold = sigmah.k
                fockold = sigmaf.k
                del gnew, sigmaf, sigmah
                gc.collect()
    
    def GWApproximation(self, itermax : int, mix : float, hoppinglist : list = None, onsitelist : list = None, loccoulomb : dict = None, nonloccoulomb : list = None,ohno : bool = False):

        errmessage = "missing input for GW calculation"
        if (hoppinglist==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()
        
        niham = NIHamiltonian(crystal=self.cry,hoppinglist=hoppinglist,onsitelist=onsitelist)
        niham.Save()
        gbare = GreenBare(crystal=self.cry,ft=self.ft,hamtb=niham.k)
        gbare.Save()
        vbare = VBare(crystal=self.cry,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno)
        vbare.Save()

        for iter in range(1,itermax+1):
            if iter == 1:
                gold = GreenInt(crystal=self.cry,ft=self.ft,greenbare=gbare.kf)
                pkfold = None
                ckfold = None
                wold = 0
                # gbare.Save('gbare')
                

            print(gold.occ)
            print("Hartree calculation start")
            sigmah = SigmaHartree(crystal=self.cry,occ=gold.occ,vbare=vbare.k)
            sigmah.Save(f'sigmah.{iter}')
            print("Hartree calculation finish")
            print("Fock calculation start")
            sigmaf = SigmaFock(crystal=self.cry,occr=gold.occr,vbare=vbare.r)
            sigmaf.Save(f'sigmaf.{iter}')
            print("Fock calculation finish")
            print("Polarizability calculation start")
            pol = PolLat(crystal=self.cry,ft=self.ft,green=gold.rt)
            pol.kf = pol.Mixing(iter=iter,mix=mix,Bb=pol.kf,Bold=pkfold)
            pol.Save(f'pkf.{iter}')
            print("Polarizability calculation finish")
            print("Screened coulomb interaction calculation start")
            w = WLat(crystal=self.cry,ft=self.ft,pol=pol.kf,vbare=vbare,c=self.c)
            w.Save(f'wkf.{iter}')
            # w.Save(w.ckf,f'wckf.{iter}')
            print("Screened coulomb interaction calculation finish")
            print("GW self-energy calculation start")
            sigmagwc = SigmaGWC(crystal=self.cry,ft=self.ft,green=gold.rt,wlat=w.crt)
            sigmagwc.kf = sigmagwc.Mixing(iter=iter,mix=mix,Fb=sigmagwc.kf,Fm=ckfold)
            sigmagwc.Save(f'sigmagwckf.{iter}')
            print("GW self-energy calculation finish")
            print("GW green's function calculation start")
            gnew = GreenInt(crystal=self.cry,ft=self.ft,greenbare=gbare.kf,sigmah=sigmah.k,sigmaf=sigmaf.k,sigmagwc=sigmagwc.kf)
            gnew.Save(f'gkf.{iter}')
            print("GW green's function calculation start")

            fcheck = self.SCFCheck(gnew.kf,gold.kf)
            bcheck = self.SCFCheck(w.kf,wold)
            mucheck = abs(gnew.mu-gold.mu)

            print(f"iteration : {iter} \nfcriteria : {fcheck} \nbcriteria : {bcheck} \nchemicalpotential : {gnew.mu+gnew.c}")

            if (fcheck <=0.005)and(bcheck<=1)and(mucheck<=0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.green = gnew
                self.pol = pol
                self.w = w
                self.sigmagwc = sigmagwc
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                gnew.Save('gkf')
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                sigmagwc.Save('sigmagwckf')
                pol.Save('pkf')
                w.Save('wkf')
                self.sigmagwc.SigmaStc()
                self.sigmagwc.Zfactor()
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
                gnew.Save('gkf')
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                sigmagwc.Save('sigmagwckf')
                pol.Save('pkf')
                w.Save('wkf')
                self.sigmagwc.SigmaStc()
                self.sigmagwc.Zfactor()
                del niham, vbare, gbare, gnew, gold, sigmaf, sigmah, sigmagwc, pol, w
                gc.collect()
            else:
                gold = gnew
                ckfold = sigmagwc.kf
                pkfold = pol.kf
                wold = w.kf

                del gnew, sigmah, sigmaf, sigmagwc, pol, w
                gc.collect()
