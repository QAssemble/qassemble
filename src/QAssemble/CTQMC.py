import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gc
import h5py
import json
import subprocess
import logging
from .utility.DLR import DLR
from .FLocDyn import *
from .FLocStc import SigHImp, SigFImp
from .BLocDyn import *
from .BLocStc import *
from .utility.Common import Common

logger = logging.getLogger("QAssemble")

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
        self.key = key if key in self.projector.fprojector else str(key)
        self.gimp = None
        self.sighimp = None
        self.sigfimp = None
        self.sigimp = None
        self.chi = None
        self.pimp = None
        self.wimp = None

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

    def _ctqmc_matrix_labels(self, equiv : np.ndarray) -> list:
        mat = np.kron(np.eye(2, dtype=int), np.asarray(equiv, dtype=int))
        labels = mat.astype(object).tolist()
        for ii in range(len(labels)):
            for jj in range(len(labels[ii])):
                labels[ii][jj] = "" if labels[ii][jj] == 0 else str(int(labels[ii][jj]))
        return labels

    def _use_dyn(self) -> bool:
        return getattr(self.bweiss, "cf", None) is not None

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
            logger.info('*** mix hybridization input ***')
            self.fweiss.Mixing(iter=iter, control=self.control)

            logger.info('*** write hyb.json file ***')

            self.fweiss._write_json_pair('hyb', iter, key, self.fweiss._as_hyb_dict(key))

            ###########################
            ### Write dyn.json file ###
            ###########################
            if self._use_dyn():
                logger.info('*** mix dynamic interaction input ***')
                self.bweiss.Mixing(control=self.control)

                logger.info('*** write dyn.json file ***')
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

                    omega_uniform = self.dlr.MatsubaraFermionUniform()
                    nu_uniform = self.dlr.MatsubaraBosonUniform()
                    green_cutoff = float(omega_uniform[-1])
                    susc_cutoff = float(nu_uniform[-1])

                    params["partition"]={}

                    params["partition"]["green basis"]= "matsubara"
                    params["partition"]["green bulla"]= True
                    params["partition"]["green matsubara cutoff"] = green_cutoff / 100
                    params["partition"]["occupation susceptibility bulla"]=True
                    params["partition"]["occupation susceptibility direct"]=False
                    params["partition"]["quantum number susceptibility"] = True
                    params["partition"]["susceptibility cutoff"] = susc_cutoff / 50
                    # measured up to susc_cutoff/50; EVALSIM fills the rest with the
                    # analytic -M/(nu^2+alpha) tail. 2x the grid maximum guarantees the
                    # output is longer than MatsubaraBosonUniform(); Chi truncates on read.
                    params["partition"]["susceptibility tail"] = 2 * susc_cutoff
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
                    params["thermalisation time"]=3 #imp['thermalization_time']
                    params["quantum number susceptibility"]=True
                    params["occupation susceptibility bulla"]=True        
                    params["green bulla"]=True       
                    params["density matrix precise"]=False #True 
                    params["measurement time"]=20 # 10 # 3 #imp['measurement_time']

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
                    logger.error("Nspin is not 1")
                    sys.exit()
            elif self.crystal.soc is True:
                logger.error("SOC is not  False, please change SOC")
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
            logger.error("QAssemble environment variable is not set.")
            sys.exit()

        ctqmc_path = os.path.join(os.path.expanduser(qassemble_path), "CTQMC", "bin", "CTQMC")
        # run_cmd = ["mpirun", "-np", "4", ctqmc_path, "params"]
        run_cmd = "mpirun -np 64 " + ctqmc_path + " params"

        with open('./ctqmc.out', 'w') as logfile, open('./ctqmc.err', 'w') as errfile:
            ret = subprocess.call(run_cmd, stdout=logfile, stderr=errfile, shell=True)
            if ret != 0:
                logger.error("Error in CTQMC. Check ctqmc.err for error message.")
                sys.exit()
        
        return None
    
    def RunMeasure(self):
        
        qassemble_path = os.environ.get("QAssemble")
        if qassemble_path is None:
            logger.error("QAssemble environment variable is not set.")
            sys.exit()

        evalsim_path = os.path.join(os.path.expanduser(qassemble_path), "CTQMC", "bin", "EVALSIM")
        # run_cmd = ["mpirun", "-np", "4", evalsim_path, "params"]
        run_cmd = "mpirun -np 64 " + evalsim_path + " params"

        with open('./evalsim.out', 'w') as logfile, open('./evalsim.err', 'w') as errfile :
            ret = subprocess.call(run_cmd, stdout=logfile, stderr=errfile, shell=True)
            if ret != 0:
                logger.error("Error in EVALSIM. Check evalsim.err for error message.")
                sys.exit()

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
            logger.info("*****************************")
            logger.info("Impurity Postprocessing Strat")
            logger.info("*****************************")
            logger.info(f'key : {key}')
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

            logger.info(f'first moment {firstmoment}')
            logger.info(f'second moment {secondmoment}')

            self.gimp = GImp(
                crystal=self.crystal,
                dlr=self.dlr,
                projector=self.projector,
                key=key,
                green=obsjson["green"],
                hdf5file=self.hdf5file,
                group=self.group,
                iteration=iter,
            )
            self.sighimp = SigHImp(
                crystal=self.crystal,
                projector=self.projector,
                key=key,
                occ=self.gimp.occ,
                vloc=self.bweiss.f[..., 0]
                if self.bweiss.f is not None
                else self.bweiss.vloc.vproj[key],
                control=self.control,
                hdf5file=self.hdf5file,
                group=self.group,
                iteration=iter,
            )
            # SigH is mixed first; SigF is then constructed from the *mixed*
            # SigH so that SigH + SigF == hf holds exactly.  SigFImp.Mixing()
            # is a re-derivation, not an independent mix -- see its docstring.
            self.sighimp.Mixing()
            self.sigfimp = SigFImp(
                crystal=self.crystal,
                projector=self.projector,
                key=key,
                sigma=obsjson["self-energy"],
                sigh=self.sighimp,
                control=self.control,
                hdf5file=self.hdf5file,
                group=self.group,
                iteration=iter,
            )
            self.sigfimp.Mixing()
            self.sigimp = SigCImp(
                crystal=self.crystal,
                dlr=self.dlr,
                projector=self.projector,
                key=key,
                sigma=obsjson["self-energy"],
                control=self.control,
                hdf5file=self.hdf5file,
                group=self.group,
                iteration=iter,
            )
            self.sigimp.Mixing()
            if self._use_dyn():
                self.chi = Chi(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=self.projector,
                    key=key,
                    partition=obsjson,
                    hdf5file=self.hdf5file,
                    group=self.group,
                    iteration=iter,
                )
                self.pimp = PImp(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=self.projector,
                    key=key,
                    chi=self.chi,
                    utilde=self.bweiss.f_uniform,
                    control=self.control,
                    hdf5file=self.hdf5file,
                    group=self.group,
                    iteration=iter,
                )
                self.pimp.Mixing()
                self.wimp = WImp(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=self.projector,
                    key=key,
                    utilde=self.bweiss.f,
                    polarization=self.pimp.f,
                    hdf5file=self.hdf5file,
                    group=self.group,
                    iteration=iter,
                )

            params = json.load(open('./params.json'))
            cutoff = params["partition"]["green matsubara cutoff"]

            # susceptibility = self.read_susceptibility_LocDyn(equiv, obsjson, key=key)
            self.diagnostics = {
                "sign":     float(ctqmc_sign) if np.isscalar(ctqmc_sign) else float(np.mean(ctqmc_sign)),
                "nimp":     float(nn) if np.isscalar(nn) else float(np.sum(nn)),
                "histo":    histo,
                "histo_m1": float(firstmoment),
                "histo_m2": float(secondmoment),
            }
            logger.info("******************************")
            logger.info("Impurity Postprocessing Finish")
            logger.info("******************************")
        finally:
            os.chdir(self.work_dir)

        self._finalize_outputs(iter)

        return None

    def _finalize_outputs(self, iter : int):

        for attr, fn in (
            ('gimp', 'gimp'),
            ('sighimp', 'sighimp'),
            ('sigfimp', 'sigfimp'),
            ('sigimp', 'sigimp'),
            ('chi', 'chi'),
            ('pimp', 'pimp'),
            ('wimp', 'wimp'),
        ):
            obj = getattr(self, attr, None)
            if obj is not None and hasattr(obj, 'Save'):
                obj.Save(fn)

        return None
