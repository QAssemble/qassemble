import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gc
import h5py
import json
import subprocess
from .utility.DLR import DLR
from .FLocDyn import *
from .BLocDyn import *
from .BLocStc import *
from .utility.Common import Common

class CTQMC(object):

    def __init__(self, dlr : DLR, fweiss : FWeiss, bweiss : BWeiss, control : dict = None):

        self.dlr = dlr
        self.fweiss = fweiss
        self.bweiss = bweiss
        self.control = control if control is not None else {}
        self.crystal = fweiss.crystal
        self.ft = dlr
        self.projector = fweiss.projector
        self.bprojector = bweiss.projector
        if self.projector is None or self.bprojector is None:
            raise ValueError("fweiss and bweiss must provide Projector objects")
        if set(self.projector.fprojector.keys()) != set(self.bprojector.bprojector.keys()):
            raise ValueError("fweiss and bweiss projectors must use the same problem keys")

        cwd = os.getcwd()
        if os.path.basename(cwd) == "ctqmc":
            self.root_dir = os.path.dirname(cwd)
            self.ctqmc_dir = cwd
        else:
            self.root_dir = cwd
            self.ctqmc_dir = os.path.join(self.root_dir, "ctqmc")
        os.makedirs(self.ctqmc_dir, exist_ok=True)
        os.chdir(self.ctqmc_dir)

    def _problem_key(self, key):
        if key not in self.projector.fprojector:
            raise KeyError(f"Unknown impurity problem key '{key}'")
        if key not in self.bprojector.bprojector:
            raise KeyError(f"Unknown bosonic impurity problem key '{key}'")
        if not isinstance(self.projector.equiv, dict) or key not in self.projector.equiv:
            raise KeyError(f"Projector equivalence matrix is missing key '{key}'")
        return key

    def _ctqmc_matrix_labels(self, equiv : np.ndarray) -> list:
        mat = np.kron(np.eye(2, dtype=int), np.asarray(equiv, dtype=int))
        labels = mat.astype(object).tolist()
        for ii in range(len(labels)):
            for jj in range(len(labels[ii])):
                labels[ii][jj] = "" if labels[ii][jj] == 0 else str(int(labels[ii][jj]))
        return labels

    def _use_dyn(self) -> bool:
        state = False
        if self.control.get("method") == "dmft" or self.control.get('edmft') or self.control.get('gw+edmft'):
            state = True

        return state
        

    def PreProcessing(self, iter : int):

        # iter = iter + 1 ### convert index from 0-based to 1-based

        for key in self.projector.fprojector.keys():
            workdir = f"impurity_{iter}_{key}"
            os.makedirs(workdir, exist_ok=True)
            os.chdir(workdir)
            try:
                Eimp = self.fweiss.e[key]
                equiv = np.asarray(self.projector.equiv[key], dtype=int)
                Eimp_final, ctqmc_mu = self.fweiss.eimp.Eimp_final_input(key, Eimp)
                Eimp_final = np.array(np.real(Eimp_final), dtype=float)
                ctqmc_mu = float(np.real(ctqmc_mu))

                ###########################
                print('*** write hyb.json file ***')

                self.fweiss._write_json_pair('hyb', iter, key, self.fweiss._as_hyb_dict(key))

                ###########################
                ### Write dyn.json file ###
                ###########################
                if self._use_dyn():
                    print('*** write dyn.json file ***')
                    self.bweiss._write_json_pair('dyn', iter, key, self.bweiss._as_dyn_dict(key))

                ##############################
                ### Write params.json file ###
                ##############################
                if self.crystal.soc is False:
                    if self.crystal.ns ==1:
                        params = {}
                        params["hloc"] = {}
                        params["hloc"]['one body'] = np.round(Eimp_final).tolist()
                        params["hloc"]["two body"] = self.bweiss.vloc.GetSlaterByProblem(
                            key=key,
                            projector=self.projector,
                            include_defaults=True,
                        )

                        params["partition"]={}

                        params["partition"]["green basis"]= "matsubara"
                        params["partition"]["green bulla"]= True
                        params["partition"]["green matsubara cutoff"] = self.dlr.cutoff # 50
                        params["partition"]["occupation susceptibility bulla"]=True
                        params["partition"]["occupation susceptibility direct"]=False
                        params["partition"]["quantum number susceptibility"] = True
                        params["partition"]["susceptibility cutoff"]=self.dlr.cutoff # 50
                        params["partition"]["susceptibility tail"]=0 #200
                        params["partition"]["quantum numbers"]={}
                        tempmat = np.ones(Eimp_final.shape[0])
                        params["partition"]["quantum numbers"]["N"]=tempmat.tolist()
                        for ii in range(len(tempmat)):
                            if ii < Eimp_final.shape[0]//2:
                                tempmat[ii]*= 0.5
                            elif ii >= Eimp_final.shape[0]//2:
                                tempmat[ii]*=-0.5
                        params["partition"]["quantum numbers"]["Sz"]=tempmat.tolist() # make 
                        params["partition"]["probabilities"]={}
                        params["partition"]["probabilities"]=["N","energy","Sz"]#["N","energy","S2","Sz"]
                        params["partition"]["density matrix precise"] = True
                        params["partition"]["print eigenstates"] = True
                        params["partition"]["print density matrix"]= True

                        params["beta"]=self.dlr.beta
                        params["complex"] = False
                        params["mu"]=ctqmc_mu
                        params["hybridisation"]={}

                        params["hybridisation"]["matrix"]=self._ctqmc_matrix_labels(equiv)
                        params["hybridisation"]["functions"]="hyb.json"
                        params["thermalisation time"]=1 #imp['thermalization_time']
                        params["quantum number susceptibility"]=True
                        params["occupation susceptibility bulla"]=True        
                        params["green bulla"]=True       
                        params["density matrix precise"]=False #True 
                        params["measurement time"]=3 # 10 # 3 #imp['measurement_time']

                        if self._use_dyn():
                            params["dyn"] = {}
                            params["dyn"]["functions"] = "dyn.json"
                            params["dyn"]["matrix"] = [["1"]]
                            params["dyn"]["quantum numbers"] = [[1] * len(Eimp_final[0])]

                        with open(f'params.{iter}.{key}.json','w') as outfile:
                            json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
                        with open('params.json','w') as outfile:
                            json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
                    elif self.crystal.ns == 2:
                        print("Nspin is not 1")
                        sys.exit()
                elif self.crystal.soc is True:
                    print("SOC is not  False, please change SOC")
                    sys.exit()
            finally:
                os.chdir(self.ctqmc_dir)
        

        return None
    
    def Run(self, iter : int):

        for key in self.projector.fprojector.keys():
            workdir = os.path.join(self.ctqmc_dir, f"impurity_{iter}_{key}")
            if not os.path.isdir(workdir):
                raise FileNotFoundError(
                    f"CTQMC working directory does not exist: {workdir}"
                )

            os.chdir(workdir)
            try:
                self.RunCTQMC()
                self.RunMeasure()
            finally:
                os.chdir(self.ctqmc_dir)

        return None
    
    def RunCTQMC(self):

        run_cmd = 'mpirun -np 4 ~/DiagE/ComCTQMC/bin/CTQMC params'
        # run_cmd = 'mpirun -np '+str()+'~/DiagE/ComCTQMC/bin/CTQMC params'

        ## input nb of processors

        with open('./ctqmc.out', 'w') as logfile, open('./ctqmc.err', 'w') as errfile:
            ret = subprocess.call(run_cmd, shell=True,stdout = logfile, stderr = errfile)
            if ret != 0:
                print("Error in CTQMC. Check ctqmc.err for error message.")
                sys.exit()
        
        return None
    
    def RunMeasure(self):
        
        # run_cmd = 'mpirun -np 4 '+diage_path+'/ComCTQMC/bin/EVALSIM params'
        run_cmd = 'mpirun -np 4 ~/DiagE/ComCTQMC/bin/EVALSIM params'

        with open('./evalsim.out', 'w') as logfile, open('./evalsim.err', 'w') as errfile :
            ret = subprocess.call(run_cmd,shell=True, stdout=logfile, stderr=errfile)
            if ret != 0:
                print("Error in EVALSIM. Check evalsim.err for error message.")
                sys.exit()
        # print("measure self-energy done")

        return None
    
    def PostProcessing(self, iter, **kwargs): 

        Green = {}
        Sigma_hf = {}
        Sigma_bare = {}
        susceptibility = {}

        for key in self.projector.fprojector.keys():
            workdir = os.path.join(self.ctqmc_dir, f"impurity_{iter}_{key}")
            if not os.path.isdir(workdir):
                raise FileNotFoundError(
                    f"CTQMC working directory does not exist: {workdir}"
                )

            os.chdir(workdir)
            try:
                equiv = np.asarray(self.projector.equiv[key], dtype=int)
                # utilde_rf = kwargs['utilde_rf']
                
                print("*****************************")
                print("Impurity Postprocessing Strat")
                print("*****************************")
                print(f'key : {key}')
                fileobs='./params.obs.json'
                filemeas='./params.meas.json'
                
                obsjson = json.load(open(fileobs))
                obsjson = obsjson['partition']

                histo_temp=obsjson["expansion histogram"]
            
                histo=np.zeros((np.shape(histo_temp)[0], 2))
                histo[:,0]=np.arange(np.shape(histo_temp)[0])
                histo[:,1]=histo_temp

                nn=obsjson["scalar"]["N"]       
                ctqmc_sign=obsjson["sign"]
            
                # histogram
                firstmoment=sum(histo[:,0]*histo[:,1])/sum(histo[:,1])
                secondmoment=sum((histo[:,0]-firstmoment)**2*histo[:,1])/sum(histo[:,1])

                print('first moment',  firstmoment)
                print('second moment', secondmoment)

                green = {}
                for green_key, val in obsjson["green"].items():
                    templist = []
                    for ii in range(len(val['function']['real'])):
                        templist.append(val['function']['real'][ii]+val['function']['imag'][ii]*1j)
                    green[green_key]=templist
                Green_uniform = self.dlr.MatsubaraAddNegativeFrequency(self.fweiss.ReadDict(equiv,green))
                Green[key] = self.dlr.MatsubaraUniformGrid2DLR(Green_uniform)

                sigma_bare = {}
                sigma_hf = {}
                for sigma_key, val in obsjson["self-energy"].items():
                    sigma_hf[sigma_key] = complex(val['moments'][0])
                    templist = []
                    for ii in range(len(val['function']['real'])):
                        templist.append(val['function']['real'][ii]+val['function']['imag'][ii]*1j)
                    sigma_bare[sigma_key] = templist
                Sigma_hf[key] = self.fweiss.eimp.ReadDict(equiv,sigma_hf)
                Sigma_bare[key] = self.fweiss.ReadDict(equiv,sigma_bare)

                params = json.load(open('./params.json'))
                cutoff = params["partition"]["green matsubara cutoff"]
                
                # susceptibility[key] = self.read_susceptibility_LocDyn(equiv, obsjson, key=key)
            finally:
                os.chdir(self.ctqmc_dir)

        return Green,Sigma_hf,Sigma_bare,susceptibility




    def PostProcessing2(self,iter,key,**kwargs):

        Green = kwargs['Green']
        Sigma_hf = kwargs['Sigma_hf']
        Sigma_bare = kwargs['Sigma_bare']
        susceptibility = kwargs['susceptibility']
        utilde_rf = kwargs['utilde_rf']


        #############################################################
        #### separate Sigma_h and Sigma_f using Green's function ####
        #### & build classes for SigmaHImp, SigmaFImp, SigmaIGWC ####
        #############################################################
        ### build classes for Sigma_h, Sigma_f, Sigma_c and initialize them
        if int(key)-1 == 0:
            Sigma_h = SigmaHImp(self.crystal)
            Sigma_f = SigmaFImp(self.crystal)
            Sigma_c = SigmaCImp(self.crystal,self.ft)

        ### compute rho using Green's function
        flocdyn = FLocDyn(self.crystal,self.ft)
        Green_tau = flocdyn.F2T(Green,1,1)
        occ = (-1) * Green_tau[:, :, :, -1].copy()
        print('density - Occ')
        print(occ)   ### check density

        ### add Simga_h, Sigma_f and Sigma_c to each key (problem space) index
        Sigma_h.add_key(occ,utilde_rf,int(key)-1)         ## int(key)-1 convert 1-based to 0-based indexing
        Sigma_f.add_key(Sigma_hf,Sigma_h.r,int(key)-1)    ## Sigma_f = Sigma_hf - Sigma_h
        Sigma_c.add_key(Sigma_bare,int(key)-1)


        #############################################################
        #######    read susceptibility and compute Pi_emft    #######
        #############################################################
        ### read susceptibility
        # susceptibility = self.read_susceptibility_LocDyn(equiv, obsjson)

        ### compute Pi_edmft using susceptibility and utilde
        if int(key)-1 == 0:
            Pi = PolImp(self.crystal,self.ft)
        Pi.add_key(susceptibility, utilde_rf, int(key)-1)

        print("******************************")
        print("Impurity Postprocessing Finish")
        print("******************************")


        return Sigma_h, Sigma_f, Sigma_c, Pi 
