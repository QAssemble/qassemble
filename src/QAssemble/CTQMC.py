import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gc
import h5py
import json
import subprocess
from .utility.DLR import DLR
from .FLocDyn import *
from .FLocStc import SigHImp, SigFImp
from .BLocDyn import *
from .BLocStc import *
from .utility.Common import Common

class CTQMC(object):

    def __init__(self, dlr : DLR, fweiss : FWeiss, bweiss : BWeiss, key, control : dict = None, hdf5file : str = None, group : str = None):

        self.dlr = dlr
        self.fweiss = fweiss
        self.bweiss = bweiss
        self.control = control if control is not None else {}
        self.hdf5file = hdf5file
        self.group = group
        self.crystal = fweiss.crystal
        self.ft = dlr
        self.projector = fweiss.projector
        self.bprojector = bweiss.projector
        if self.projector is None or self.bprojector is None:
            raise ValueError("fweiss and bweiss must provide Projector objects")
        if set(self.projector.fprojector.keys()) != set(self.bprojector.bprojector.keys()):
            raise ValueError("fweiss and bweiss projectors must use the same problem keys")
        self.key = self._problem_key(key)
        if hasattr(self.fweiss, "key") and self.fweiss.key != self.key:
            raise ValueError(
                f"fweiss key '{self.fweiss.key}' does not match CTQMC key '{self.key}'"
            )
        if hasattr(self.bweiss, "key") and self.bweiss.key != self.key:
            raise ValueError(
                f"bweiss key '{self.bweiss.key}' does not match CTQMC key '{self.key}'"
            )
        self.gimp = None
        self.sighimp = None
        self.sigfimp = None
        self.sigimp = None
        self.chi = None
        self.pimp = None

        self.work_dir = os.path.abspath(os.getcwd())
        cwd = self.work_dir
        if os.path.basename(cwd) == "ctqmc":
            self.root_dir = os.path.dirname(cwd)
            self.ctqmc_dir = cwd
        else:
            self.root_dir = cwd
            self.ctqmc_dir = os.path.join(self.root_dir, "ctqmc")
        os.makedirs(self.ctqmc_dir, exist_ok=True)
        os.chdir(self.ctqmc_dir)

    def _problem_key(self, key):
        pkey = key if key in self.projector.fprojector else str(key)
        if pkey not in self.projector.fprojector:
            raise KeyError(f"Unknown impurity problem key '{key}'")
        if pkey not in self.bprojector.bprojector:
            raise KeyError(f"Unknown bosonic impurity problem key '{pkey}'")
        if not isinstance(self.projector.equiv, dict) or pkey not in self.projector.equiv:
            raise KeyError(f"Projector equivalence matrix is missing key '{pkey}'")
        return pkey

    def _ctqmc_matrix_labels(self, equiv : np.ndarray) -> list:
        mat = np.kron(np.eye(2, dtype=int), np.asarray(equiv, dtype=int))
        labels = mat.astype(object).tolist()
        for ii in range(len(labels)):
            for jj in range(len(labels[ii])):
                labels[ii][jj] = "" if labels[ii][jj] == 0 else str(int(labels[ii][jj]))
        return labels

    def _use_dyn(self) -> bool:
        return getattr(self.bweiss, "ubar_rf", None) is not None

    def PreProcessing(self, iter : int):

        # iter = iter + 1 ### convert index from 0-based to 1-based

        key = self.key
        workdir = f"impurity_{iter}_{key}"
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)
        try:
            Eimp = self.fweiss.e
            equiv = np.asarray(self.projector.equiv[key], dtype=int)
            Eimp_final, ctqmc_mu = self.fweiss.eimp.ToCTQMC(key=key, Eimp=Eimp)
            # Eimp_final = self.fweiss.eimp.ToCTQMC(key=key, Eimp=Eimp)
            Eimp_final = np.array(np.real(Eimp_final), dtype=float)
            # ctqmc_mu = float(np.real(self.fweiss.mu))

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
                    Eimp_final[np.abs(Eimp_final) < 1.0e-12] = 0.0
                    params["hloc"]['one body'] = Eimp_final.tolist()
                    params["hloc"]["two body"] = self.bweiss.vloc.GetUijklComCTQMC(key).tolist()

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

        key = self.key
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

        qassemble_path = os.environ.get("QAssemble")
        if qassemble_path is None:
            print("QAssemble environment variable is not set.")
            sys.exit()

        ctqmc_path = os.path.join(os.path.expanduser(qassemble_path), "CTQMC", "bin", "CTQMC")
        # run_cmd = ["mpirun", "-np", "4", ctqmc_path, "params"]
        run_cmd = "mpirun -np 4 " + ctqmc_path + " params"

        with open('./ctqmc.out', 'w') as logfile, open('./ctqmc.err', 'w') as errfile:
            ret = subprocess.call(run_cmd, stdout=logfile, stderr=errfile, shell=True)
            if ret != 0:
                print("Error in CTQMC. Check ctqmc.err for error message.")
                sys.exit()
        
        return None
    
    def RunMeasure(self):
        
        qassemble_path = os.environ.get("QAssemble")
        if qassemble_path is None:
            print("QAssemble environment variable is not set.")
            sys.exit()

        evalsim_path = os.path.join(os.path.expanduser(qassemble_path), "CTQMC", "bin", "EVALSIM")
        # run_cmd = ["mpirun", "-np", "4", evalsim_path, "params"]
        run_cmd = "mpirun -np 4 " + evalsim_path + " params"

        with open('./evalsim.out', 'w') as logfile, open('./evalsim.err', 'w') as errfile :
            ret = subprocess.call(run_cmd, stdout=logfile, stderr=errfile, shell=True)
            if ret != 0:
                print("Error in EVALSIM. Check evalsim.err for error message.")
                sys.exit()
        # print("measure self-energy done")

        return None
    
    def PostProcessing(self, iter): 

        key = self.key
        workdir = os.path.join(self.ctqmc_dir, f"impurity_{iter}_{key}")
        if not os.path.isdir(workdir):
            raise FileNotFoundError(
                f"CTQMC working directory does not exist: {workdir}"
            )

        os.chdir(workdir)
        try:
            equiv = np.asarray(self.projector.equiv[key], dtype=int)
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

            self.gimp = GImp(
                crystal=self.crystal,
                dlr=self.dlr,
                projector=self.projector,
                key=key,
                green=obsjson["green"],
                hdf5file=self.hdf5file,
                group=self.group,
            )
            static_interaction = self.bweiss.v
            dynamic_interaction = self.bweiss.utilde_rf

            self.sighimp = SigHImp(
                crystal=self.crystal,
                projector=self.projector,
                key=key,
                gimp=self.gimp,
                vloc=dynamic_interaction if dynamic_interaction is not None else static_interaction,
                hdf5file=self.hdf5file,
                group=self.group,
            )
            self.sigfimp = SigFImp(
                crystal=self.crystal,
                projector=self.projector,
                key=key,
                sigma=obsjson["self-energy"],
                sigh=self.sighimp,
                hdf5file=self.hdf5file,
                group=self.group,
            )
            self.sigimp = SigCImp(
                crystal=self.crystal,
                dlr=self.dlr,
                projector=self.projector,
                key=key,
                sigma=obsjson["self-energy"],
                hdf5file=self.hdf5file,
                group=self.group,
            )
            self.chi = Chi(
                crystal=self.crystal,
                dlr=self.dlr,
                projector=self.projector,
                key=key,
                partition=obsjson,
                hdf5file=self.hdf5file,
                group=self.group,
            )
            self.pimp = PImp(
                crystal=self.crystal,
                dlr=self.dlr,
                projector=self.projector,
                key=key,
                chi=self.chi,
                utilde=dynamic_interaction,
                hdf5file=self.hdf5file,
                group=self.group,
            )

            params = json.load(open('./params.json'))
            cutoff = params["partition"]["green matsubara cutoff"]
            
            # susceptibility = self.read_susceptibility_LocDyn(equiv, obsjson, key=key)
            print("******************************")
            print("Impurity Postprocessing Finish")
            print("******************************")
        finally:
            os.chdir(self.work_dir)

        return None
