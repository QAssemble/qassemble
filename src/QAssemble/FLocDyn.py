import numpy as np
import logging
import sys
import json
import h5py
from .Crystal import Crystal
from .FLocStc import EImp
from .Projector import Projector
from .utility.DLR import DLR
from .utility.Common import Common
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
from .utility.Mixing import Mixing
from .utility.Projection import Projection as PJ

logger = logging.getLogger("QAssemble")

class FLocDyn(object):

    def __init__(self,crystal : Crystal, dlr : DLR, projector : Projector):
        
        self.crystal = crystal
        self.dlr = dlr
        self.projector = projector

    def Inverse(self, mat : np.ndarray):
        
        norb = mat.shape[0]
        ns = mat.shape[2]
        nft = mat.shape[3]

        matinv = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            for js in range(ns):
                matinv[:,:,js,ift] = Common.MatInv(mat[:,:,js,ift])

        return matinv
    
    
    def F2T(self, ff : np.ndarray) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nfreq = ff.shape[3]

        ff_t = np.moveaxis(ff, -1, 0)

        batch = norb * norb * ns
        ff_2d = np.ascontiguousarray(ff_t).reshape(nfreq, batch)

        fxx = self.dlr.dF.dlr_from_matsubara(ff_2d, beta=self.dlr.beta, xi = -1)
        ftau_2d = self.dlr.dF.tau_from_dlr(fxx)
        ntau = ftau_2d.shape[0]
        ftau = ftau_2d.reshape(ntau, norb, norb, ns)
        ftau = np.moveaxis(ftau, 0, -1)

        ftau = np.asfortranarray(ftau)

        return ftau
        
    def T2F(self,ftau : np.ndarray) -> np.ndarray:

        norb = ftau.shape[0]
        ns = ftau.shape[2]
        ntau = ftau.shape[3]

        ftau_t = np.moveaxis(ftau, -1, 0)  # (ntau, norb, norb, ns)
        batch = norb * norb * ns 
        ftau_2d = np.ascontiguousarray(ftau_t).reshape(ntau, batch)

        fxx = self.dlr.dF.dlr_from_tau(ftau_2d)
        ff_2d = self.dlr.dF.matsubara_from_dlr(fxx, beta=self.dlr.beta, xi=-1)
        nfreq = ff_2d.shape[0]

        ff = ff_2d.reshape(nfreq, norb, norb, ns)
        ff = np.moveaxis(ff, 0, -1)  # (norb, norb, ns, nfreq)
        ff = np.asfortranarray(ff)

        return ff

    def CheckGroup(self, filepath :str, group : str):
        
        with h5py.File(filepath,'r') as file:
            return group in file

    def _as_dynamic_spin_matrix(self, mat : np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=np.complex128)
        if mat.ndim == 3:
            mat = mat[:, :, np.newaxis, :]
        if mat.ndim != 4:
            raise ValueError(f"dynamic matrix must be 3D or 4D, got {mat.ndim}D")
        return np.asfortranarray(mat)

    def _as_hyb_dict(self, key, hyb : np.ndarray = None, equiv : np.ndarray = None) -> dict:
        if hyb is None:
            if not hasattr(self, "h") or self.h is None:
                raise ValueError("Hybridisation data is missing")
            hyb = self.h
        if equiv is None:
            pkey = self.ResolveProblemKey(key)
            if self.projector is None or not isinstance(self.projector.equiv, dict) or pkey not in self.projector.equiv:
                raise KeyError(f"Projector equivalence matrix is missing key '{pkey}'")
            equiv = self.projector.equiv[pkey]

        hyb = self._as_dynamic_spin_matrix(hyb)
        equiv = np.asarray(equiv, dtype=int)
        if self.crystal.ns != 1:
            print("Nspin is not 1")
            sys.exit()

        nind = int(np.amax(equiv))
        hyb_dict = {}
        for ind in range(1, nind + 1):
            pos_row, pos_col = np.where(equiv == ind)
            if len(pos_row) == 0:
                continue
            val = np.zeros(hyb.shape[3], dtype=np.complex128)
            for ii, jj in zip(pos_row, pos_col):
                val += hyb[ii, jj, 0, :]
            val /= len(pos_row)
            hyb_dict[str(ind)] = {
                "beta": self.dlr.beta,
                "real": np.real(val).tolist(),
                "imag": np.imag(val).tolist(),
            }
        return hyb_dict

    def _write_json_pair(self, stem : str, iter : int, key, payload : dict) -> None:
        with open(f'{stem}.{iter}.{key}.json', 'w') as outfile:
            json.dump(payload, outfile, sort_keys=True, indent=4, separators=(',', ': '))
        with open(f'{stem}.json', 'w') as outfile:
            json.dump(payload, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    def ResolveProblemKey(self, key):
        if self.projector is None:
            raise ValueError("projector is required to resolve problem key")
        pkey = key if key in self.projector.fprojector else str(key)
        if pkey not in self.projector.fprojector:
            raise KeyError(f"Unknown impurity problem key '{key}'")
        return pkey
    
    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]

        ynew = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')
        w0 = (1.0 - 3.0*w1)*np.pi*temperature
        widtharray = w0+w1*x
        cnt = 0

        for x0 in x:
            if (x0>cutoff+(w0+w1*cutoff)*3.0):
                ynew[...,cnt]=y[...,cnt]
            else:
                if ((x0>3*widtharray[cnt])and((x[-1]-x0)>3*widtharray[cnt])):
                    dist = 1.0/np.sqrt(2*np.pi)/widtharray[cnt]*np.exp(-(x-x0)**2/2.0/widtharray[cnt]**2)
                    for js in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                ynew[iorb,jorb,js,cnt] = sum(dist*y[iorb,jorb,js])/sum(dist)
                else:
                    ynew[...,cnt] = y[...,cnt]
            cnt += 1

        return ynew
    
    def Mixing(self,iter : int, mix : float, Fb : np.ndarray, Fold : np.ndarray):

        norb = Fb.shape[0]
        ns = Fb.shape[2]
        nft = Fb.shape[3]

        Fnew = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        if iter == 1:
            mix = 1.0
            Fold = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        Fnew = mix*Fb+(1.0-mix)*Fold

        return Fnew

    def _resolve_equiv_matrix(self, imp=None, key=None) -> np.ndarray:
        """Resolve an equivalent-orbital matrix from legacy/new impurity inputs.

        Supported inputs:
        - 2D ndarray/list: used directly as equivalence matrix.
        - 1D ndarray/list: interpreted as diagonal class labels and promoted via ``np.diag``.
        - Legacy dict: ``imp[str(key)]['impurity_matrix']``.
        - Direct dict: ``imp[str(key)]`` is the equivalence matrix itself.
        - Fallback: ``self.projector.equiv[str(key)]`` when ``imp`` is None.
        """
        def _resolve_dict_key(dct, key_):
            if key_ is None:
                if len(dct) == 1:
                    return str(next(iter(dct.keys())))
                raise ValueError(
                    "key is required when multiple impurity problems are present"
                )
            k_ = str(key_)
            if k_ not in dct:
                raise KeyError(f"equiv source does not contain key '{k_}'")
            return k_

        if imp is None:
            if self.projector is None or not isinstance(self.projector.equiv, dict):
                raise ValueError(
                    "imp is None and projector.equiv is not available; "
                    "provide imp or set projector.equiv"
                )
            peq = self.projector.equiv
            k = _resolve_dict_key(peq, key)
            equiv = np.asarray(peq[k])

        elif isinstance(imp, np.ndarray):
            equiv = imp
        elif isinstance(imp, (list, tuple)):
            equiv = np.asarray(imp)
        elif isinstance(imp, dict):
            k = _resolve_dict_key(imp, key)
            if isinstance(imp[k], dict):
                if "impurity_matrix" not in imp[k]:
                    raise KeyError(
                        f"imp['{k}'] must contain an 'impurity_matrix' entry"
                    )
                equiv = np.asarray(imp[k]["impurity_matrix"])
            else:
                equiv = np.asarray(imp[k])
        else:
            raise TypeError(
                "imp must be ndarray/list/tuple (equiv matrix), direct equiv dict, or legacy impurity dict"
            )

        if equiv.ndim == 1:
            equiv = np.diag(equiv)

        if equiv.ndim != 2 or equiv.shape[0] != equiv.shape[1]:
            raise ValueError(
                f"equivalence matrix must be square 2D, got shape {equiv.shape}"
            )

        return np.asarray(equiv, dtype=int)

    def Arr2Dict(self, equiv : np.ndarray, matin : np.ndarray) -> dict:
        """Average local dynamic fermionic matrix over equivalent orbital pairs."""
        if matin.ndim == 3:
            if self.crystal.ns != 1:
                raise ValueError(
                    f"3D input is only allowed for ns=1, crystal ns={self.crystal.ns}"
                )
            matin = matin[:, :, np.newaxis, :]
        elif matin.ndim != 4:
            raise ValueError(f"matin must be 3D or 4D, got {matin.ndim}D")

        norb = matin.shape[0]
        if matin.shape[1] != norb:
            raise ValueError("matin first two dimensions must be square")
        if equiv.shape[0] != norb or equiv.shape[1] != norb:
            raise ValueError(
                f"equiv shape {equiv.shape} is incompatible with matin shape {matin.shape}"
            )

        ns = matin.shape[2]
        nfreq = matin.shape[3]
        if ns != self.crystal.ns:
            raise ValueError(
                f"spin dimension mismatch: matin ns={ns}, crystal ns={self.crystal.ns}"
            )

        nind = int(np.amax(equiv))
        if nind <= 0:
            raise ValueError("equiv labels must be positive integers")

        matdict = {}
        for ind in range(1, nind + 1):
            pos = Common.FindPositions(equiv, ind)
            if len(pos) == 0:
                continue

            if ns == 1:
                avg = np.zeros(nfreq, dtype=np.complex128)
                for ii, jj in pos:
                    avg += matin[ii, jj, 0, :]
                avg /= len(pos)
                matdict[str(ind)] = avg.tolist()
            else:
                avg = np.zeros((ns, nfreq), dtype=np.complex128)
                for js in range(ns):
                    for ii, jj in pos:
                        avg[js, :] += matin[ii, jj, js, :]
                avg /= len(pos)
                matdict[str(ind)] = avg.tolist()

        return matdict

    def Dict2Arr(self, equiv : np.ndarray, matdict : dict) -> np.ndarray:
        """Expand equivalent-orbital dict data back to local dynamic fermionic matrix."""
        norb = len(equiv)
        ns = self.crystal.ns
        nind = int(np.amax(equiv))

        sample = None
        if "1" in matdict:
            sample = matdict["1"]
        elif 1 in matdict:
            sample = matdict[1]
        elif len(matdict) > 0:
            sample = next(iter(matdict.values()))
        else:
            raise ValueError("matdict is empty; cannot infer frequency dimension")

        sample_arr = np.asarray(sample, dtype=np.complex128)
        if ns == 1:
            if sample_arr.ndim != 1:
                raise ValueError("for ns=1, each matdict value must be 1D (nfreq)")
            nfreq = sample_arr.shape[0]
        else:
            if sample_arr.ndim != 2 or sample_arr.shape[0] != ns:
                raise ValueError(
                    f"for ns={ns}, each matdict value must be 2D with shape ({ns}, nfreq)"
                )
            nfreq = sample_arr.shape[1]

        matout = np.zeros((norb, norb, ns, nfreq), dtype=np.complex128, order='F')

        for ind in range(1, nind + 1):
            key = str(ind) if str(ind) in matdict else ind
            if key not in matdict:
                continue
            val = np.asarray(matdict[key], dtype=np.complex128)
            pos = Common.FindPositions(equiv, ind)

            if ns == 1:
                if val.ndim != 1 or val.shape[0] != nfreq:
                    raise ValueError(
                        f"matdict['{ind}'] must be 1D with length {nfreq}"
                    )
                for ii, jj in pos:
                    matout[ii, jj, 0, :] = val
            else:
                if val.ndim != 2 or val.shape[0] != ns or val.shape[1] != nfreq:
                    raise ValueError(
                        f"matdict['{ind}'] must have shape ({ns}, {nfreq})"
                    )
                for js in range(ns):
                    for ii, jj in pos:
                        matout[ii, jj, js, :] = val[js, :]

        return matout

    def ReadDict(self, equiv : np.ndarray, mat_dict : dict) -> np.ndarray:
        """Read equivalent-orbital dict data as a local dynamic fermionic matrix."""
        return self.Dict2Arr(equiv=equiv, matdict=mat_dict)

    def AddNegativeFrequency(self, mat : np.ndarray) -> np.ndarray:
        """Build negative-frequency data from non-negative data using G(-iw)=G(iw)^dagger."""
        return self.dlr.MatsubaraAddNegativeFrequency(mat)

    def UniformGridToDLR(self, mat : np.ndarray, omega_uniform : np.ndarray = None) -> np.ndarray:
        """Fit full uniform Matsubara data and evaluate it on the DLR Matsubara grid."""
        return self.dlr.MatsubaraUniformGrid2DLR(mat, omega=omega_uniform, sign=-1)

    def AverageByEquiv(self, equiv : np.ndarray, matin : np.ndarray, squeeze : bool = True) -> np.ndarray:
        """Average equivalent orbital classes and return array in one pass."""
        if matin.ndim == 3:
            if self.crystal.ns != 1:
                raise ValueError(
                    f"3D input is only allowed for ns=1, crystal ns={self.crystal.ns}"
                )
            matin4 = matin[:, :, np.newaxis, :]
        elif matin.ndim == 4:
            matin4 = matin
        else:
            raise ValueError(f"matin must be 3D or 4D, got {matin.ndim}D")

        norb = matin4.shape[0]
        if matin4.shape[1] != norb:
            raise ValueError("matin first two dimensions must be square")
        if equiv.shape[0] != norb or equiv.shape[1] != norb:
            raise ValueError(
                f"equiv shape {equiv.shape} is incompatible with matin shape {matin4.shape}"
            )
        if matin4.shape[2] != self.crystal.ns:
            raise ValueError(
                f"spin dimension mismatch: matin ns={matin4.shape[2]}, crystal ns={self.crystal.ns}"
            )

        matout = np.array(matin4, dtype=np.complex128, copy=True, order='F')
        nind = int(np.amax(equiv))
        if nind <= 0:
            raise ValueError("equiv labels must be positive integers")

        for ind in range(1, nind + 1):
            pos = Common.FindPositions(equiv, ind)
            if len(pos) == 0:
                continue
            for js in range(self.crystal.ns):
                avg = np.zeros(matin4.shape[3], dtype=np.complex128)
                for ii, jj in pos:
                    avg += matin4[ii, jj, js, :]
                avg /= len(pos)
                for ii, jj in pos:
                    matout[ii, jj, js, :] = avg

        if squeeze and self.crystal.ns == 1:
            return matout[:, :, 0, :]
        return matout

    def UniformGrid(self, mat : np.ndarray) -> np.ndarray:
        self.omega_uniform = self.dlr.MatsubaraFermionUniform()
        return self.dlr.MatsubaraDLR2UniformGrid(mat, sign=-1)

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        return Dyson.FLocDyn(mat1, mat2)

    
    def Projection(self, matin : np.ndarray, key):
        if self.projector is None:
            raise ValueError("projector is required for Projection")

        if matin.ndim != 5:
            raise ValueError(f"matin must be 5D, got {matin.ndim}D")

        pkey = self.ResolveProblemKey(key)
        return PJ.FLatDyn(matin, self.projector.fprojector[pkey])
    
class GLoc(FLocDyn):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, green : np.ndarray, hdf5file : str = None, group : str = None):

        super().__init__(crystal, dlr, projector)

        
        self.green = green
        self.f = {}
        self.t = {}
        self.occ = {}
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.Cal()
        
        

    def Cal(self):

        self.f = {}
        self.t = {}
        self.occ = {}

        for key in self.projector.fprojector.keys():
            mat = self.Projection(self.green, key)
            self.f[key] = mat
            tau = self.F2T(mat)
            self.t[key] = tau
            self.occ[key] = self.Occ(tau)

        return None

    def Occ(self, mat : np.ndarray):

        tau_beta = np.array([self.dlr.beta], dtype=np.float64)
        occ = np.zeros_like(mat[...,0], dtype=np.complex128)
        for js in range(mat.shape[2]):
            block = mat[:, :, js, :].T

            ntau_b = block.shape[0]
            block_2d = block.reshape(ntau_b, -1)

            fxx = self.dlr.dF.dlr_from_tau(block_2d)
            fout = self.dlr.dF.eval_dlr_tau(fxx[:, :, None], tau_beta, beta=self.dlr.beta)

            occ[:, :, js] = -fout[0, :, 0].reshape(mat.shape[0], mat.shape[0])

        return occ
    
    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            gloc = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(gloc, fn, obj, dtype=complex)
            else:
                for key, value in self.f.items():
                    Common.HDF5CreateDataset(
                        gloc,
                        f"{fn}.{str(key)}",
                        value,
                        dtype=complex,
                    )

        return None
    
class GImp(FLocDyn):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, key, green, hdf5file : str = None, group : str = None):

        super().__init__(crystal, dlr, projector)

        self.key = self.ResolveProblemKey(key)
        self.green = green
        self.f = None
        self.t = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.Cal()

    def _read_ctqmc_green(self, green : dict) -> dict:

        matdict = {}
        for green_key, val in green.items():
            function = val["function"] if isinstance(val, dict) and "function" in val else val
            real = np.asarray(function["real"], dtype=np.float64)
            imag = np.asarray(function.get("imag", np.zeros_like(real)), dtype=np.float64)
            matdict[green_key] = real + imag * 1j

        return matdict

    def Cal(self):

        if isinstance(self.green, dict):
            equiv = np.asarray(self.projector.equiv[self.key], dtype=int)
            green = self._read_ctqmc_green(self.green)
            self.f_uniform = self.ReadDict(equiv, green)
            green_uniform = self.dlr.MatsubaraAddNegativeFrequency(self.f_uniform)
            self.f = self.dlr.MatsubaraUniformGrid2DLR(green_uniform)
        else:
            self.f = np.asfortranarray(self.green, dtype=np.complex128)

        self.t = self.F2T(self.f)

        return None
    
    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            gloc = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(gloc, fn, obj, dtype=complex)
            else:
                Common.HDF5CreateDataset(gloc, fn, self.f, dtype=complex)
                Common.HDF5CreateDataset(gloc, fn+'_uniform', self.f_uniform, dtype=complex)

        return None


class SigCImp(FLocDyn):

    def __init__(self,crystal : Crystal,dlr : DLR,projector : Projector,key,sigma,sigma_hf : np.ndarray = None,subtract_static : bool = True,hdf5file : str = None,group : str = None,):

        super().__init__(crystal, dlr, projector)

        self.key = self.ResolveProblemKey(key)
        self.sigma_in = sigma
        self.sigma_hf_in = sigma_hf
        self.subtract_static = subtract_static
        self.hf = None
        self.f = None
        self.t = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.Cal()

    def _read_ctqmc_sigma(self, sigma : dict) -> dict:

        matdict = {}
        for sigma_key, val in sigma.items():
            function = val["function"] if isinstance(val, dict) and "function" in val else val
            real = np.asarray(function["real"], dtype=np.float64)
            imag = np.asarray(function.get("imag", np.zeros_like(real)), dtype=np.float64)
            matdict[sigma_key] = real + imag * 1j

        return matdict

    def _read_ctqmc_sigma_hf(self, sigma : dict) -> dict:

        matdict = {}
        for sigma_key, val in sigma.items():
            if not isinstance(val, dict) or "moments" not in val:
                raise ValueError(
                    "SigCImp requires self-energy moments[0] to subtract "
                    "the static/HF part from CTQMC self-energy"
                )
            matdict[sigma_key] = complex(val["moments"][0])

        return matdict

    def _dict_to_static_arr(self, equiv : np.ndarray, matdict : dict) -> np.ndarray:

        norb = len(equiv)
        ns = self.crystal.ns
        nind = int(np.amax(equiv))
        matout = np.zeros((norb, norb, ns), dtype=np.complex128, order='F')

        for ind in range(1, nind + 1):
            key = str(ind) if str(ind) in matdict else ind
            if key not in matdict:
                continue

            val = np.asarray(matdict[key], dtype=np.complex128)
            pos = Common.FindPositions(equiv, ind)

            if ns == 1:
                if val.ndim > 0 and val.size != 1:
                    raise ValueError(
                        f"static matdict['{ind}'] must be scalar for ns=1"
                    )
                for ii, jj in pos:
                    matout[ii, jj, 0] = val.item()
            else:
                if val.ndim != 1 or val.shape[0] != ns:
                    raise ValueError(
                        f"static matdict['{ind}'] must be a 1D spin array of length {ns}"
                    )
                for js in range(ns):
                    for ii, jj in pos:
                        matout[ii, jj, js] = val[js]

        return matout

    def _resolve_sigma_hf(self) -> np.ndarray:

        if self.sigma_hf_in is not None:
            sigma_hf = self.sigma_hf_in
            if hasattr(sigma_hf, "hf"):
                sigma_hf = sigma_hf.hf
            elif hasattr(sigma_hf, "s"):
                sigma_hf = sigma_hf.s
            elif hasattr(sigma_hf, "h"):
                sigma_hf = sigma_hf.h
            sigma_hf = np.asarray(sigma_hf, dtype=np.complex128)
        elif isinstance(self.sigma_in, dict):
            equiv = np.asarray(self.projector.equiv[self.key], dtype=int)
            sigma_hf = self._dict_to_static_arr(
                equiv, self._read_ctqmc_sigma_hf(self.sigma_in)
            )
        else:
            raise ValueError(
                "SigCImp requires sigma_hf when subtract_static=True and sigma "
                "is not a CTQMC self-energy dict with moments"
            )

        if sigma_hf.ndim == 2:
            sigma_hf = sigma_hf[:, :, np.newaxis]
        if sigma_hf.ndim != 3:
            raise ValueError(f"sigma_hf must be 2D or 3D, got {sigma_hf.ndim}D")
        if sigma_hf.shape[2] != self.crystal.ns:
            raise ValueError(
                f"spin dimension mismatch: sigma_hf ns={sigma_hf.shape[2]}, "
                f"crystal ns={self.crystal.ns}"
            )

        return np.asfortranarray(sigma_hf)

    def Cal(self):

        if isinstance(self.sigma_in, dict):
            equiv = np.asarray(self.projector.equiv[self.key], dtype=int)
            sigma = self._read_ctqmc_sigma(self.sigma_in)
            sigma_grid = self.ReadDict(equiv, sigma)
            if self.subtract_static:
                self.hf = self._resolve_sigma_hf()
                if self.hf.shape != sigma_grid.shape[:3]:
                    raise ValueError(
                        f"sigma_hf shape {self.hf.shape} is incompatible with "
                        f"sigma shape {sigma_grid.shape}"
                    )
                sigma_grid = np.asfortranarray(sigma_grid - self.hf[..., np.newaxis])
            self.f_uniform = sigma_grid
            sigma_uniform = self.dlr.MatsubaraAddNegativeFrequency(sigma_grid)
            sigma_total = self.dlr.MatsubaraUniformGrid2DLR(sigma_uniform)
            if sigma_total.ndim != 4:
                raise ValueError(
                    f"sigma must be 4D after DLR conversion, got {sigma_total.ndim}D"
                )
            if sigma_total.shape[2] != self.crystal.ns:
                raise ValueError(
                    f"spin dimension mismatch: sigma ns={sigma_total.shape[2]}, "
                    f"crystal ns={self.crystal.ns}"
                )
            self.f = sigma_total
        else:
            sigma_total = np.asfortranarray(self.sigma_in, dtype=np.complex128)
            if sigma_total.ndim != 4:
                raise ValueError(
                    f"sigma must be 4D after DLR conversion, got {sigma_total.ndim}D"
                )
            if sigma_total.shape[2] != self.crystal.ns:
                raise ValueError(
                    f"spin dimension mismatch: sigma ns={sigma_total.shape[2]}, "
                    f"crystal ns={self.crystal.ns}"
                )

            if self.subtract_static:
                self.hf = self._resolve_sigma_hf()
                if self.hf.shape != sigma_total.shape[:3]:
                    raise ValueError(
                        f"sigma_hf shape {self.hf.shape} is incompatible with "
                        f"sigma shape {sigma_total.shape}"
                    )
                self.f = np.asfortranarray(sigma_total - self.hf[..., np.newaxis])
            else:
                self.f = sigma_total

        if self.f.ndim != 4:
            raise ValueError(f"sigma must be 4D after DLR conversion, got {self.f.ndim}D")
        if self.f.shape[2] != self.crystal.ns:
            raise ValueError(
                f"spin dimension mismatch: sigma ns={self.f.shape[2]}, "
                f"crystal ns={self.crystal.ns}"
            )

        self.t = self.F2T(self.f)

        return None
    
    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            sigimp = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(sigimp, fn, obj, dtype=complex)
            else:
                Common.HDF5CreateDataset(sigimp, fn, self.f, dtype=complex)
                Common.HDF5CreateDataset(sigimp, fn+'_uniform', self.f_uniform, dtype=complex)

        return None


class Hyb(FLocDyn):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, key, green : np.ndarray, eimp : np.ndarray, sigh : np.ndarray = None, sigf : np.ndarray = None, sigc : np.ndarray = None, hdf5file : str = None, group : str = None):
        
        super().__init__(crystal, dlr, projector)

        self.key = self.ResolveProblemKey(key)
        self.green = green
        self.eimp = eimp
        self.sigh = sigh
        self.sigf = sigf
        self.sigc = sigc
        
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        print("Local Green's Function :", self.green[:, :, 0, 0])
        print("Impurity Level :", self.eimp[:, :, 0])
        print("Hartree Self-Energy :", self.sigh[:, :, 0] if self.sigh is not None else None)
        print("Fock Self-Energy :", self.sigf[:, :, 0] if self.sigf is not None else None)
        print("Correlated Self-Energy :", self.sigc[:, :, 0, 0] if self.sigc is not None else None)

        self.f = None
        self.t = None
        self.Cal()

    def Cal(self):

        tempmat = np.zeros_like(self.green, dtype=np.complex128, order='F')
        sig = np.zeros_like(self.green, dtype=np.complex128, order='F')
        if self.sigh is not None:
            sig += self.sigh
        if self.sigf is not None:
            sig += self.sigf
        if self.sigc is not None:
            sig += self.sigc

        g_inv = self.Inverse(self.green)
        
        e = self.eimp
        I = np.eye(g_inv.shape[0], dtype=np.complex128)
        omega = self.dlr.omega * 1j

        for iomega in range(len(omega)):
            for js in range(g_inv.shape[2]):
                tempmat[..., js, iomega] = omega[iomega]*I - e[..., js] - g_inv[..., js, iomega] - sig[..., js, iomega]
        self.f = tempmat
        self.t = self.F2T(tempmat)

        return None
    
    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            hyb = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(hyb, fn, obj, dtype=complex)
            else:
                Common.HDF5CreateDataset(hyb, fn, self.f, dtype=complex)

        return None

class FWeiss(FLocDyn):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, key, eimp : EImp, hyb : Hyb, mu : float = 0.0, hdf5file : str = None, group : str = None):

        super().__init__(crystal, dlr, projector)

        self.key = self.ResolveProblemKey(key)
        self.eimp = eimp
        self.hyb = hyb.f
        self.mu = mu
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        
        self.e = None
        self.h_dlr = None
        self.h = None
        self.omega_uniform = None

        self.Cal()

    def Cal(self):
        equiv = np.array(self.projector.equiv[self.key])
        self.e = self.eimp.AverageByEquiv(equiv, self.eimp.e)
        self.h_dlr = self.AverageByEquiv(equiv, self.hyb)
        self.h = self.UniformGrid(self.h_dlr)
        
        return None