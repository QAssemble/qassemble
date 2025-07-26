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
from .CorrelationFunction import *


class DMFT(object):
    def __init__(self, cry : dict = None, ft : dict = None):

        self.crystal = Crystal(cry=cry)
        self.ft = FTGrid(ft=ft)
    


    def write_ctqmc_params(self,iter,key,E_imp : np.ndarray,equiv : np.ndarray):
        
        if self.fermion.SOC is False:
            if self.fermion.ns ==1:
                params = {}
                params["hloc"] = {}
                mu_ctqmc=-np.real(E_imp[0,0,0])
                # print(mu_ctqmc,type(mu_ctqmc))
                E_imp = E_imp[:,:,0]+mu_ctqmc*np.eye(E_imp.shape[0],E_imp.shape[0])
                E_imp = np.array(np.real(E_imp),dtype=float)
                tempmat = np.kron(E_imp,np.eye(2,2))
                params["hloc"]['one body'] = tempmat.tolist()
                self.boson.get_Uijkl_comctqmc(key)
                params["hloc"]["two body"]=self.boson.U_ctqmc.tolist()
                # params["hloc"]["two body"] = {}
                # params["hloc"]["two body"]["parametrisataion"] = "slater-condon"
                # params["hloc"]["two body"]["F0"]=5.0
                # params["hloc"]["two body"]["F2"]=0.0
                # params["hloc"]["two body"]["F4"]=0.0
                # params["hloc"]["two body"]["approximation"] = "none"
                
                params["partition"]={}
                
                params["partition"]["green basis"]= "matsubara"
                params["partition"]["green bulla"]= True
                params["partition"]["green matsubara cutoff"] = 50
                params["partition"]["occupation susceptibility bulla"]=True
                params["partition"]["occupation susceptibility direct"]=False
                params["partition"]["quantum number susceptibility"] = True
                params["partition"]["susceptibility cutoff"]=50
                params["partition"]["susceptibility tail"]=200
                params["partition"]["quantum numbers"]={}
                tempmat = np.ones(E_imp.shape[0]*2)
                params["partition"]["quantum numbers"]["N"]=tempmat.tolist()
                # [1,1,1,1,1,1,1,1,1,1]
                for ii in range(len(tempmat)):
                    if ii < E_imp.shape[0]:
                        tempmat[ii]*= 0.5
                    elif ii >= E_imp.shape[0]:
                        tempmat[ii]*=-0.5
                params["partition"]["quantum numbers"]["Sz"]=tempmat.tolist() # make 
                # [0.5,0.5,0.5,0.5,0.5,-0.5,-0.5,-0.5,-0.5,-0.5]
                # params["partition"]["observables"]={}
                # params["partition"]["observables"]["S2"] = {}
                params["partition"]["probabilities"]={}
                params["partition"]["probabilities"]=["N","energy","Sz"]#["N","energy","S2","Sz"]
                params["partition"]["density matrix precise"] = True
                params["partition"]["print eigenstates"] = True
                params["partition"]["print density matrix"]= True
                
                # params["dyn"]={}
                # params["dyn"]["quantum numbers"] = np.ones(E_imp.shape[0]*2).tolist()
                # # [[1,1,1,1,1,1,1,1,1,1]]
                # params["dyn"]["functions"] = "dyn.json"
                # params["dyn"]["matrix"] = [["F0"]]
                params["beta"]=self.ft.beta
                params["complex"] = False
                params["mu"]=mu_ctqmc
                params["hybridisation"]={}
                # tempmat2 = np.kron(equiv,np.ones((2,2)))
                tempmat2 = np.kron(equiv,np.eye(2,2))
                tempmat2 = tempmat2.tolist()
                for ii in range(len(tempmat2)):
                    for jj in range(len(tempmat2)):
                        if tempmat2[ii][jj]==0.0:
                            tempmat2[ii][jj] = ""
                        else:
                            tempmat2[ii][jj] = str(int(tempmat2[ii][jj]))

                params["hybridisation"]["matrix"]=tempmat2
                params["hybridisation"]["functions"]="hyb.json"
                params["thermalisation time"]=1 #imp['thermalization_time']
                params["quantum number susceptibility"]=True
                params["occupation susceptibility bulla"]=True        
                params["green bulla"]=True       
                params["density matrix precise"]=False #True 
                params["measurement time"]=3 #imp['measurement_time']
                
                with open(f'params.{iter}.{key}.json','w') as outfile:
                    json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
                with open('params.json','w') as outfile:
                    json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
                # print("params.json written", file=self.m_ini.control['h_log'])
            elif self.fermion.ns == 2:
                print("Nspin is not 1")
                sys.exit()
        elif self.fermion.SOC is True:
            print("SOC is not  False, please change SOC")
            sys.exit()

        return None
    
    def write_hyb_json(self,iter,key,hyb : dict):

        if self.fermion.SOC is False:
            if self.fermion.ns == 1:
                json_dict = {}
                for key,val in hyb.items():
                    json_dict[key] = {}
                    json_dict[key]['beta'] = self.ft.beta
                    json_dict[key]['real'] = np.real(val[0]).tolist()
                    json_dict[key]['imag'] = np.imag(val[0]).tolist()

                with open(f'hyb.{iter}.{key}.json','w') as outfile:
                    json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
                with open('hyb.json','w') as outfile:
                    json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
            
            elif self.fermion.ns == 2:
                print("Nspin is not 1")
                sys.exit()
        elif self.fermion.SOC is True:
            print("SOC must be False")
            sys.exit()
        return None
        
    
    def run_ctqmc(self):
        
        #run_cmd = 'mpirun -np 1 '+diage_path+'/ComCTQMC/bin/CTQMC params'
        run_cmd = 'mpirun -np 4 '+diage_path+'/ComCTQMC/bin/CTQMC params'
        print(run_cmd)
        
        with open('./ctqmc.out', 'w') as logfile, open('./ctqmc.err', 'w') as errfile:
            ret = subprocess.call(run_cmd, shell=True,stdout = logfile, stderr = errfile)
            if ret != 0:
                print("Error in CTQMC. Check ctqmc.err for error message.")
                sys.exit()
        
        return None