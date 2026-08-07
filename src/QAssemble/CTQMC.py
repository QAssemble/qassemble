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
from .utility.HDF5 import IO

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

    def _control_bool(self, *names, default=False):
        for name in names:
            if name in self.control:
                value = self.control[name]
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    value_lower = value.strip().lower()
                    if value_lower in {"true", "t", "1", "yes", "y", "on"}:
                        return True
                    if value_lower in {"false", "f", "0", "no", "n", "off"}:
                        return False
                return bool(value)
        return default

    def _control_float(self, *names, default):
        for name in names:
            if name in self.control:
                return float(self.control[name])
        return float(default)

    def _control_int(self, *names, default):
        for name in names:
            if name in self.control:
                return int(self.control[name])
        return int(default)

    def _control_string(self, *names, default):
        for name in names:
            if name in self.control:
                return str(self.control[name]).lower()
        return str(default).lower()

    def _read_ctqmc_function_dict(self, data : dict) -> dict:
        matdict = {}
        for sigma_key, val in data.items():
            function = val["function"] if isinstance(val, dict) and "function" in val else val
            real = np.asarray(function["real"], dtype=np.float64)
            imag = np.asarray(function.get("imag", np.zeros_like(real)), dtype=np.float64)
            matdict[sigma_key] = real + 1j * imag
        return matdict

    def _dynamic_dict_to_array(self, equiv : np.ndarray, matdict : dict) -> np.ndarray:
        norb = len(equiv)
        ns = self.crystal.ns
        nind = int(np.amax(equiv))
        sample_key = "1" if "1" in matdict else next(iter(matdict))
        sample = np.asarray(matdict[sample_key], dtype=np.complex128)
        nfreq = sample.shape[0] if ns == 1 else sample.shape[1]
        matout = np.zeros((norb, norb, ns, nfreq), dtype=np.complex128, order="F")

        for ind in range(1, nind + 1):
            key = str(ind) if str(ind) in matdict else ind
            if key not in matdict:
                continue
            val = np.asarray(matdict[key], dtype=np.complex128)
            pos = Common.FindPositions(equiv, ind)
            if ns == 1:
                if val.ndim != 1:
                    raise ValueError(
                        f"CTQMC function '{ind}' must be 1D for ns=1"
                    )
                for ii, jj in pos:
                    matout[ii, jj, 0, :] = val
            else:
                if val.ndim != 2 or val.shape[0] != ns:
                    raise ValueError(
                        f"CTQMC function '{ind}' must have shape (ns, nfreq)"
                    )
                for js in range(ns):
                    for ii, jj in pos:
                        matout[ii, jj, js, :] = val[js, :]
        return np.asfortranarray(matout)

    def _sigma_array_to_ctqmc_dict(
        self,
        equiv : np.ndarray,
        template : dict,
        sigma_grid : np.ndarray,
    ) -> dict:
        out = {}
        nind = int(np.amax(equiv))
        for ind in range(1, nind + 1):
            key = str(ind)
            src_key = key if key in template else ind
            if src_key not in template:
                continue
            pos = Common.FindPositions(equiv, ind)
            if self.crystal.ns == 1:
                val = np.zeros(sigma_grid.shape[-1], dtype=np.complex128)
                for ii, jj in pos:
                    val += sigma_grid[ii, jj, 0, :]
                val /= len(pos)
            else:
                val = np.zeros((self.crystal.ns, sigma_grid.shape[-1]), dtype=np.complex128)
                for js in range(self.crystal.ns):
                    for ii, jj in pos:
                        val[js, :] += sigma_grid[ii, jj, js, :]
                    val[js, :] /= len(pos)
            entry = dict(template[src_key])
            entry["function"] = {
                "real": np.real(val).tolist(),
                "imag": np.imag(val).tolist(),
            }
            out[key] = entry
        return out

    def _read_sigma_error_grid(self, equiv : np.ndarray):
        if not os.path.exists("params.err.json"):
            return None
        try:
            with open("params.err.json") as fh:
                errjson = json.load(fh)
            partition = errjson.get("partition", errjson)
            sigma_err = partition.get("self-energy")
            if sigma_err is None:
                return None
            return self._dynamic_dict_to_array(
                equiv, self._read_ctqmc_function_dict(sigma_err)
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning(f"[sigimp-guard] ignoring params.err.json: {exc}")
            return None

    def _sigimp_guard_metrics(self, sigma_grid : np.ndarray, err_grid=None) -> dict:
        arr = np.asarray(sigma_grid, dtype=np.complex128)
        nfreq = arr.shape[-1]
        nwin = max(1, min(self._control_int(
            "SigImpGuardLowFreqPoints",
            "sigimp_guard_low_freq_points",
            default=80,
        ), nfreq))

        rough_re = 0.0
        rough_im = 0.0
        max_pos_imag = -np.inf
        diag_n = min(arr.shape[0], arr.shape[1])
        for iorb in range(diag_n):
            for js in range(arr.shape[2]):
                vals = arr[iorb, iorb, js, :nwin]
                max_pos_imag = max(max_pos_imag, float(np.nanmax(vals.imag)))
                if vals.size >= 3:
                    d2 = np.diff(vals, n=2)
                    d2_re = np.abs(d2.real)
                    d2_im = np.abs(d2.imag)
                    if np.any(np.isfinite(d2_re)):
                        rough_re = max(rough_re, float(np.nanmean(d2_re)))
                    if np.any(np.isfinite(d2_im)):
                        rough_im = max(rough_im, float(np.nanmean(d2_im)))

        err_mean = np.nan
        err_max = np.nan
        if err_grid is not None:
            err_abs = np.abs(np.asarray(err_grid, dtype=np.complex128))
            err_mean = float(np.nanmean(err_abs))
            err_max = float(np.nanmax(err_abs))

        finite = bool(np.all(np.isfinite(arr.real)) and np.all(np.isfinite(arr.imag)))
        return {
            "finite": finite,
            "raw_roughness_re": float(rough_re),
            "raw_roughness_im": float(rough_im),
            "raw_roughness": float(max(rough_re, rough_im)),
            "err_mean": err_mean,
            "err_max": err_max,
            "max_positive_imag": float(max_pos_imag),
        }

    def _previous_guard_metrics(self, iter : int) -> dict:
        if self.hdf5file is None or self.group is None or iter <= 1:
            return {}
        path = f"{self.group}/SigImpGuard/{self.key}/metrics.{iter - 1}"
        try:
            with h5py.File(self.hdf5file, "r") as handle:
                if path not in handle:
                    return {}
                return {name: handle[path].attrs[name] for name in handle[path].attrs}
        except OSError:
            return {}

    def _has_previous_impurity_output(self, iter : int) -> bool:
        if self.hdf5file is None or self.group is None or iter <= 1:
            return False
        paths = (
            f"{self.group}/SigHImp/sighimp.{iter - 1}.{self.key}",
            f"{self.group}/SigFImp/sigfimp.{iter - 1}.{self.key}",
            f"{self.group}/SigCImp/sigimp.{iter - 1}.{self.key}",
        )
        try:
            with h5py.File(self.hdf5file, "r") as handle:
                return all(path in handle for path in paths)
        except OSError:
            return False

    def _evaluate_sigimp_guard(self, iter : int, sigma_grid : np.ndarray, err_grid=None) -> dict:
        enabled = self._control_bool("SigImpGuard", "sigimp_guard", default=True)
        mode = self._control_string(
            "SigImpGuardMode", "sigimp_guard_mode", default="fallback_previous"
        )
        im_tol = self._control_float(
            "SigImpGuardImTol", "sigimp_guard_im_tol", default=1.0e-10
        )
        rough_factor = self._control_float(
            "SigImpGuardRoughnessFactor",
            "sigimp_guard_roughness_factor",
            default=10.0,
        )
        err_factor = self._control_float(
            "SigImpGuardErrorFactor", "sigimp_guard_error_factor", default=10.0
        )

        metrics = self._sigimp_guard_metrics(sigma_grid, err_grid=err_grid)
        diagnostics = dict(metrics)
        diagnostics.update({
            "guard_enabled": bool(enabled),
            "guard_status": "accepted",
            "guard_reason": "ok",
            "used_fallback": False,
            "previous_iteration": int(iter) - 1 if iter > 1 else -1,
            "smoothing_applied": False,
        })
        if not enabled or mode == "off":
            diagnostics["guard_status"] = "disabled"
            diagnostics["guard_reason"] = "disabled"
            return diagnostics

        reasons = []
        if not metrics["finite"]:
            reasons.append("nonfinite")
        if metrics["max_positive_imag"] > im_tol:
            reasons.append("positive_imag")

        prev = self._previous_guard_metrics(iter)
        eps = 1.0e-30
        if prev:
            prev_rough = float(prev.get("raw_roughness", np.nan))
            if np.isfinite(prev_rough) and prev_rough > eps:
                if metrics["raw_roughness"] > rough_factor * prev_rough:
                    reasons.append("roughness_ratio")

            for name in ("err_mean", "err_max"):
                prev_err = float(prev.get(name, np.nan))
                curr_err = float(metrics.get(name, np.nan))
                if np.isfinite(prev_err) and np.isfinite(curr_err) and prev_err > eps:
                    if curr_err > err_factor * prev_err:
                        reasons.append(f"{name}_ratio")

        if reasons:
            diagnostics["guard_status"] = "failed"
            diagnostics["guard_reason"] = ",".join(reasons)
            diagnostics["used_fallback"] = self._has_previous_impurity_output(iter)
        return diagnostics

    def _smooth_sigma_grid(self, sigma_grid : np.ndarray) -> np.ndarray:
        width = self._control_float(
            "SigImpSmoothingWidth", "sigimp_smoothing_width", default=0.05
        )
        try:
            omega = np.asarray(self.dlr.MatsubaraFermionUniform(), dtype=float)
        except AttributeError:
            return sigma_grid
        if len(omega) != sigma_grid.shape[-1]:
            return sigma_grid
        cutoff = float(omega[-1])
        temperature = 1.0 / float(self.dlr.beta)
        broadener = getattr(self.fweiss, "GaussianLinearBroad", None)
        if broadener is None:
            return sigma_grid
        return broadener(omega, sigma_grid, width, temperature, cutoff)

    def _apply_optional_sigimp_smoothing(
        self,
        iter : int,
        equiv : np.ndarray,
        sigma : dict,
        sigma_grid : np.ndarray,
        err_grid,
        diagnostics : dict,
    ):
        mode = self._control_string(
            "SigImpGuardMode", "sigimp_guard_mode", default="fallback_previous"
        )
        smoothing_enabled = self._control_bool(
            "SigImpSmoothing", "sigimp_smoothing", default=False
        )
        if mode != "smooth_then_check" or not smoothing_enabled:
            return sigma, sigma_grid, diagnostics

        smoothed_grid = self._smooth_sigma_grid(sigma_grid)
        smoothed_diag = self._evaluate_sigimp_guard(iter, smoothed_grid, err_grid=err_grid)
        smoothed_diag["smoothing_applied"] = True
        if smoothed_diag["guard_status"] != "failed":
            sigma = self._sigma_array_to_ctqmc_dict(equiv, sigma, smoothed_grid)
            return sigma, smoothed_grid, smoothed_diag
        return sigma, sigma_grid, smoothed_diag

    def _read_previous_impurity_output(self, iter : int):
        if not self._has_previous_impurity_output(iter):
            return None
        with h5py.File(self.hdf5file, "r") as handle:
            prev = int(iter) - 1
            base = str(self.group)
            sigh = np.asarray(
                handle[f"{base}/SigHImp/sighimp.{prev}.{self.key}"][()],
                dtype=np.complex128,
            )
            sigf = np.asarray(
                handle[f"{base}/SigFImp/sigfimp.{prev}.{self.key}"][()],
                dtype=np.complex128,
            )
            sigc = np.asarray(
                handle[f"{base}/SigCImp/sigimp.{prev}.{self.key}"][()],
                dtype=np.complex128,
            )
            uniform_path = f"{base}/SigCImp/sigimp.{prev}.{self.key}_uniform"
            sigc_uniform = None
            if uniform_path in handle:
                sigc_uniform = np.asarray(handle[uniform_path][()], dtype=np.complex128)
        if sigc_uniform is None and hasattr(self.dlr, "MatsubaraDLR2UniformGrid"):
            sigc_uniform = self.dlr.MatsubaraDLR2UniformGrid(sigc, sign=-1)
        return {
            "sigh": np.asfortranarray(sigh),
            "sigf": np.asfortranarray(sigf),
            "sigc": np.asfortranarray(sigc),
            "sigc_uniform": np.asfortranarray(sigc_uniform)
            if sigc_uniform is not None else None,
        }

    def _make_fallback_impurity_objects(self, previous : dict, iter : int) -> None:
        self.sighimp = SigHImp.__new__(SigHImp)
        self.sighimp.crystal = self.crystal
        self.sighimp.projector = self.projector
        self.sighimp.key = self.key
        self.sighimp.h = previous["sigh"]
        self.sighimp.s = previous["sigh"]
        self.sighimp.hdf5file = self.hdf5file
        self.sighimp.group = self.group
        self.sighimp.subgroup = "SigHImp"
        self.sighimp.iteration = iter

        self.sigfimp = SigFImp.__new__(SigFImp)
        self.sigfimp.crystal = self.crystal
        self.sigfimp.projector = self.projector
        self.sigfimp.key = self.key
        self.sigfimp.s = previous["sigf"]
        self.sigfimp.hdf5file = self.hdf5file
        self.sigfimp.group = self.group
        self.sigfimp.subgroup = "SigFImp"
        self.sigfimp.iteration = iter

        self.sigimp = SigCImp.__new__(SigCImp)
        self.sigimp.crystal = self.crystal
        self.sigimp.dlr = self.dlr
        self.sigimp.projector = self.projector
        self.sigimp.key = self.key
        self.sigimp.f = previous["sigc"]
        self.sigimp.f_uniform = previous["sigc_uniform"]
        self.sigimp.hdf5file = self.hdf5file
        self.sigimp.group = self.group
        self.sigimp.subgroup = "SigCImp"
        self.sigimp.iteration = iter
        try:
            self.sigimp.t = self.sigimp.F2T(self.sigimp.f)
        except Exception:
            self.sigimp.t = None

    def _resolve_sighimp_vloc(self, key):
        if self.bweiss.f is None:
            return self.bweiss.vloc.vproj[key], "bare_vloc"
        if getattr(self.bweiss, "f_uniform", None) is not None:
            return self.bweiss.f_uniform[..., 0], "bweiss_uniform_nu0"
        if hasattr(self.dlr, "MatsubaraDLR2UniformGrid"):
            return (
                self.dlr.MatsubaraDLR2UniformGrid(self.bweiss.f, sign=1)[..., 0],
                "bweiss_dlr_to_uniform_nu0",
            )
        return self.bweiss.vloc.vproj[key], "bare_vloc"

    def _write_sigimp_guard_diagnostics(self, iter : int, diagnostics : dict) -> None:
        if self.hdf5file is None or self.group is None:
            return
        with h5py.File(self.hdf5file, "a") as handle:
            group = IO.Group(handle, self.group, "SigImpGuard", self.key, f"metrics.{iter}")
            for name, value in diagnostics.items():
                if isinstance(value, (bool, np.bool_)):
                    group.attrs[name] = int(value)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    group.attrs[name] = value
                elif isinstance(value, str):
                    group.attrs[name] = value

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
                    params["thermalisation time"]=5 #imp['thermalization_time']
                    params["quantum number susceptibility"]=True
                    params["occupation susceptibility bulla"]=True        
                    params["green bulla"]=True       
                    params["density matrix precise"]=False #True 
                    params["measurement time"]=25 # 10 # 3 #imp['measurement_time']

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

            sigma_input = obsjson["self-energy"]
            sigma_grid = self._dynamic_dict_to_array(
                equiv, self._read_ctqmc_function_dict(sigma_input)
            )
            err_grid = self._read_sigma_error_grid(equiv)
            guard_diag = self._evaluate_sigimp_guard(
                iter, sigma_grid, err_grid=err_grid
            )
            sigma_input, sigma_grid, guard_diag = self._apply_optional_sigimp_smoothing(
                iter, equiv, sigma_input, sigma_grid, err_grid, guard_diag
            )
            vloc, sighimp_vloc_source = self._resolve_sighimp_vloc(key)
            guard_diag["sighimp_vloc_source"] = sighimp_vloc_source

            if guard_diag["guard_status"] == "failed":
                previous = self._read_previous_impurity_output(iter)
                if previous is None:
                    self._write_sigimp_guard_diagnostics(iter, guard_diag)
                    raise RuntimeError(
                        f"SigImp guard failed for key={key}, iter={iter} "
                        f"({guard_diag['guard_reason']}) and no previous "
                        "accepted impurity self-energy is available"
                    )
                guard_diag["used_fallback"] = True
                self._make_fallback_impurity_objects(previous, iter)
                logger.warning(
                    f"[sigimp-guard] key={key} iter={iter}: "
                    f"fallback_previous reason={guard_diag['guard_reason']}"
                )
            else:
                self.sighimp = SigHImp(
                    crystal=self.crystal,
                    projector=self.projector,
                    key=key,
                    occ=self.gimp.occ,
                    vloc=vloc,
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
                    sigma=sigma_input,
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
                    sigma=sigma_input,
                    control=self.control,
                    hdf5file=self.hdf5file,
                    group=self.group,
                    iteration=iter,
                )
                self.sigimp.Mixing()
            self._write_sigimp_guard_diagnostics(iter, guard_diag)

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
                "sigimp_guard_used_fallback": float(bool(guard_diag["used_fallback"])),
                "sigimp_guard_failed": float(guard_diag["guard_status"] == "failed"),
                "sigimp_raw_roughness": float(guard_diag["raw_roughness"]),
                "sigimp_err_mean": float(guard_diag["err_mean"]),
                "sigimp_err_max": float(guard_diag["err_max"]),
                "sigimp_max_positive_imag": float(guard_diag["max_positive_imag"]),
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
