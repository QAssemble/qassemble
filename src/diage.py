#!/usr/bin/env python3.9
import numpy as np
import string 
import os, sys, gc
import json
import time, datetime

from core.CorrelationFunction import CorrelationFunction

class Run():
    def __init__(self,test = False) -> None:
        
        self.control = None
        self.func = None
        self.ReadInput()
        if test:
            control = self.control
            func = CorrelationFunction(latt=control['crystal']['lattice'], basisposition=control['crystal']['basispos'], ns=control['crystal']['ns'],soc=control['crystal']['soc'],rkgrid=control['crystal']['rkgrid'],orboption=control['crystal']['orbital'],N=control['crystal']['nume'],T=control['ft']['T'],beta=control['ft']['beta'],size=control['ft']['size'],c=control['run']['cw'])
            self.func = func
        else:
            if (self.control['run']['method']==1)or(self.control['run']['method']==2)or(self.control['run']['method']==3)or(self.control['run']['method']==4):
                self.RunDiagE()
        

    def CheckKeyinString(self, key : str, dictionary : dict):

        if (key not in dictionary):
            print("missing \'"+key+'\' in '+dictionary['name'],flush=True)
            sys.exit()
        return None

    def ReadInput(self):
        
        loc = {}
        glob = {}
        exec(open("input.ini").read(),glob,loc)
        
        control = {}
        control['name'] = 'control'
        control["crystal"] = {}
        control["ft"] = {}
        control["ham"] = {} 
        control["run"] = {}
        inicrystal = loc["Crystal"]
        inicrystal['name'] = 'Crystal'
        
        ham = loc["Hamiltonian"]
        ham['name'] = 'Hamiltonian'
        ham['OneBody']['name'] = "OneBody"
        ham["TwoBody"]['name'] = "TwoBody"
        ham["TwoBody"]['Local']['name'] = "Local"
        ini = loc["Control"]
        ini['name'] = 'Control'
        
        ######## Construct Crystal Structure ########
        self.CheckKeyinString('RVec',inicrystal)
        lattice = inicrystal["RVec"]
        self.CheckKeyinString("Basis",inicrystal)
        CorF = inicrystal.get("CorF","F")
        pos = []
        orboption = {}
        for i, ii in enumerate(inicrystal["Basis"]):
            pos.append(ii[0])
            orboption[i+1] = ii[1]

        print(orboption)

        basispos = {"CorF" : CorF,"pos" : pos}    
        ns = inicrystal.get('NSpin',1)
        soc = inicrystal.get("SOC",False)
        self.CheckKeyinString("KGrid",inicrystal)
        rkgrid = inicrystal["KGrid"]
        self.CheckKeyinString("NElec",inicrystal)
        NumE = inicrystal["NElec"]        
        control['crystal']['lattice'] = lattice
        control['crystal']['basispos'] = basispos
        control["crystal"]['ns'] = ns
        control["crystal"]['soc'] = soc
        control["crystal"]['rkgrid'] = rkgrid
        control["crystal"]["nume"] = NumE
        control["crystal"]["orbital"] = orboption
        

        ######## Construct One-Body Hamiltonian ########
        self.CheckKeyinString('OneBody',ham)
        self.CheckKeyinString('Hopping',ham['OneBody'])
        self.CheckKeyinString('Onsite',ham['OneBody'])
        self.CheckKeyinString('TwoBody',ham)
        self.CheckKeyinString('Local',ham['TwoBody'])
        self.CheckKeyinString('Parameter',ham['TwoBody']['Local'])

        hopplist = []
        print(ham["OneBody"]['Hopping'])
        for orb,val in ham["OneBody"]['Hopping'].items():
            # t = val[0]
            # lat = val[1]
            # print(f"orbital : {orb}, hopping : {t}, lattice : {lat}")
            # for r in lat:
            #     hopplist.append([t,orb[0],orb[1],r])
            for ii in range(len(val)):
                t = val[ii][0]
                lat = val[ii][1]
                print(f"orbital : {orb}, hopping : {t}, lattice : {lat}")
                for r in lat:
                    hopplist.append([t,orb[0],orb[1],r])

        # for key,val in ham['OneBody'].items():
        #     for orb, lat in tb['site'][key].items():
        #         for r in range(len(lat)):
        #             hopplist.append([val,orb[0],orb[1],lat[r]])
        onsitelist = []
        for orb, val in ham['OneBody']['Onsite'].items():
            onsitelist.append(val)
        print(onsitelist)
        control['ham']['hoppinglist'] = hopplist
        control['ham']['onsitelist'] = onsitelist

        ######## Construct Two-Body Hamiltonian ########
        vlocparameter = {}
        vlocparameter["option"] = {}
        vlocparameter["Parameter"] = ham['TwoBody']['Local'].get('Parameter',"SlaterKanamori")
        
        if vlocparameter["Parameter"] is "SlaterKanamori":
            for key,val in ham['TwoBody']['Local']['option'].items():
                l = val.get('l',0)
                U = val.get('U',0)
                J = val.get('J',0)
                Up = val.get('Up',U-2*J)
                (atom,orb) = key
                if type(orb)==int:
                    orblist = [orb]
                elif type(orb)==tuple:
                    orblist = list(orb)
                vlocparameter['option'][atom+1] = {}
                vlocparameter['option'][atom+1]['l'] = l
                vlocparameter['option'][atom+1]['value'] = [U,Up,J]
                vlocparameter['option'][atom+1]['orbitals'] =orblist
        if vlocparameter['Parameter'] is 'Slater':
            for key,val in ham['TwoBody']['Local']['option'].items():
                (atom,orb) = key
                l = val.get('l',0)
                value = []
                if l == 0:
                    F0 = val.get('F0',0)
                    value = [F0]
                elif l == 1:
                    F0 = val.get('F0',0)
                    F2 = val.get('F2',0)
                    value = [F0,F2]
                elif l == 2:
                    F0 = val.get('F0',0)
                    F2 = val.get('F2',0)
                    F4 = val.get('F4',0)
                    value = [F0,F2,F4]
                elif l == 3:
                    F0 = val.get('F0',0)
                    F2 = val.get('F2',0)
                    F4 = val.get('F4',0)
                    F6 = val.get('F6',0)
                    value = [F0,F2,F4,F6]
                if type(orb)==int:
                    orblist = [orb]
                elif type(orb)==tuple:
                    orblist = list(orb)
                vlocparameter['option'][atom+1] = {}
                vlocparameter['option'][atom+1]['l'] = l
                vlocparameter['option'][atom+1]['value'] = value
                vlocparameter['option'][atom+1]['orbitals'] = orblist
        if vlocparameter["Parameter"] is "Kanamori":
            for key,val in ham['TwoBody']['Local']['option'].items():
                l = val.get('l',0)
                U = val.get('U',0)
                J = val.get('J',0)
                Up = val.get('Up',U-2*J)
                (atom,orb) = key
                if type(orb)==int:
                    orblist = [orb]
                elif type(orb)==tuple:
                    orblist = list(orb)
                vlocparameter['option'][atom+1] = {}
                vlocparameter['option'][atom+1]['l'] = l
                vlocparameter['option'][atom+1]['value'] = [U,Up,J]
                vlocparameter['option'][atom+1]['orbitals'] = orblist
        print(vlocparameter)
        
        

        vnonlocparameter = None
        # for orb,val in ham['TwoBody']['NonLocal'].items():
        #         v = val[0]
        #         latt = val[1]
        #         for r in latt:
        #             vnonlocparameter.append([v,orb[0],orb[1],r])
        ohno = False
        if ham['TwoBody']['NonLocal'] == "None":
            pass
        elif ham['TwoBody']['NonLocal'] == "Ohno":
            # vnonlocparameter = OhnoParameterization(U, rkgrid, orboption, lattice, inicrystal["pos"])
            ohno = True
        else:
            vnonlocparameter = []
            for orb,val in ham['TwoBody']['NonLocal'].items():
                # v = val[0]
                # latt = val[1]
                # for r in latt:
                #     vnonlocparameter.append([v,orb[0],orb[1],r])
                for ii in range(len(val)):
                    vij = val[ii][0]
                    lat = val[ii][1]
                    for r in lat:
                        vnonlocparameter.append([vij,orb[0],orb[1],r])
        
        control['ham']['coulomb'] = {}
        control['ham']['coulomb']['local'] = vlocparameter
        control['ham']['coulomb']['nonlocal'] = vnonlocparameter
        control['ham']['coulomb']['ohno'] = ohno
        
        ######## Check the method ########
        self.CheckKeyinString('Method',ini)
        control['run']['method'] = ini.get("Method")
        control['run']['mix'] = ini.get("Mix",0.1)
        control['run']['nscf'] = ini.get("NSCF",100)
        control['run']['cw'] = ini.get("ConstantW",1.0)

        # CheckKeyinString("MatsubaraMesh",ini)
        size = ini.get("MatsubaraMesh",1000)
        kb = 8.6173303*10**-5
        if ('T' not in ini)and('beta' not in ini):
            print('missing T and beta in \''+ini['name'])
            sys.exit()
        if ('T' not in ini)and('beta' in ini):
            beta = ini.get('beta',100)
            T = 1/(beta*kb)
        if ('T' in ini)and('beta' not in ini):
            T = ini.get('T',300)
            beta = 1/(T*kb)
        
        control['ft']['T'] = T
        control['ft']['beta'] = beta
        control['ft']['size'] = size

        self.control = control
        return None

    def RunDiagE(self):
        
        control = self.control
        func = CorrelationFunction(latt=control['crystal']['lattice'], basisposition=control['crystal']['basispos'], ns=control['crystal']['ns'],soc=control['crystal']['soc'],rkgrid=control['crystal']['rkgrid'],orboption=control['crystal']['orbital'],N=control['crystal']['nume'],T=control['ft']['T'],beta=control['ft']['beta'],size=control['ft']['size'],c=control['run']['cw'])

        itermax = control['run']['nscf']
        mix = control['run']['mix']
        method = control['run']['method']

        if method == 1:
            print("Tight-Binding calculation start")
            hoppinglist = control['ham']['hoppinglist']
            onsitelist = control['ham']['onsitelist']
            hamtb = func.TightBinding(hoppinglist=hoppinglist,onsitelist=onsitelist)
            print("Tight-Binding calculation finish")
            # flatstc = FLatStc(crystal=func.cry)
            # energy = flatstc.Diagonalize(hamtb)
            # FLatStcSave(hamtb,'hamtb')
            # FLatStcSave(energy,'energy')
        if method == 2:
            print("Hartree-Fock calculation start")
        
            hoppinglist = control['ham']['hoppinglist']
            onsitelist = control['ham']['onsitelist']
            vloc = control['ham']['coulomb']['local']
            vnonloc = control['ham']['coulomb']['nonlocal']
            ohno = control['ham']['coulomb']['ohno']
            start = time.time()
            func.HartreeFockH(itermax=itermax,mix=mix,hoppinglist=hoppinglist,onsitelist=onsitelist,loccoulomb=vloc,nonloccoulomb=vnonloc,ohno=ohno)
            end = time.time()
            print("Hartree-Fock calculation finish")
            delta = datetime.timedelta(seconds=(end-start))
            print(f"Hartree-Fock loop time = {delta}")

            # FLatStcSave(hamhf,'hamhf')
            # FLatStcSave(sigmah.hk,'sigmahk')
            # FLatStcSave(sigmaf.fk,'sigmafk')
            # BLatStcSave(func.vbare.k,'vk')
        if method==3:
            print("Hartree-Fock calculation start")
            
            hoppinglist = control['ham']['hoppinglist']
            onsitelist = control['ham']['onsitelist']
            vloc = control['ham']['coulomb']['local']
            vnonloc = control['ham']['coulomb']['nonlocal']
            ohno = control['ham']['coulomb']['ohno']
            start = time.time()
            func.HartreeFock(itermax=itermax,mix=mix,hoppinglist=hoppinglist,onsitelist=onsitelist,loccoulomb=vloc,nonloccoulomb=vnonloc,ohno=ohno)
            end = time.time()
            print("Hartree-Fock calculation finish")
            delta = datetime.timedelta(seconds=(end-start))
            print(f"Hartree-Fock loop time = {delta}")
            
            # FLatDynSave(func.green.gkf,'gkfhf')
            # FLatStcSave(func.hamhf,'hamhf')
            # FLatStcSave(func.sigmah.hk,'sigmahk')
            # FLatStcSave(func.sigmaf.fk,'sigmafk')
            # BLatStcSave(func.vbare.k,'vk')
        if method==4:
            print("GW calculation start")
            
            hoppinglist = control['ham']['hoppinglist']
            onsitelist = control['ham']['onsitelist']
            vloc = control['ham']['coulomb']['local']
            vnonloc = control['ham']['coulomb']['nonlocal']
            ohno = control['ham']['coulomb']['ohno']
            start = time.time()
            func.GWApproximation(itermax=itermax,mix=mix,hoppinglist=hoppinglist,onsitelist=onsitelist,loccoulomb=vloc,nonloccoulomb=vnonloc,ohno=ohno)
            end = time.time()
            print("GW calculation finish")
            delta = datetime.timedelta(seconds=(end-start))
            print(f"GW loop time = {delta}")

            # FLatDynSave(func.green.gkf,'gkf')
            # FLatStcSave(func.sigmah.hk,'sigmahk')
            # FLatStcSave(func.sigmaf.fk,'sigmafk')
            # FLatDynSave(func.sigmac.kf,'sigmackf')
            # BLatDynSave(func.w.wkf,'wkf')
            # BLatDynSave(func.pol.polkf,'pkf')
            # BLatStcSave(func.vbare.k,'vk')
        return None    
    
    # def OhnoParameter(self):
    #     '''
    #     Set the non-loc bare coulomb interaction by using Ohno parameterization

    #     V = U/{\kappa_ij(1+cR_{ij}^2)}^{1/2}
    #     '''
    #     kappa = 2.0
    #     vlist = []
    #     rkgrid = self.control["crystal"]['rkgrid']
if __name__ == '__main__':
    print("Calculation Start")
    run = Run()
    print("Calculation Finish")