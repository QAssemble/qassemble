import numpy as np
import logging
import sys
import json
import scipy.optimize
import scipy.linalg.lapack
import copy
import h5py
import time, datetime
from .Crystal import Crystal
from .BLocStc import VLoc
from .Projector import Projector
from .utility.DLR import DLR
from .utility.Common import Common
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
from .utility.Mixing import Mixing
from .utility.Projection import Projection as PJ
from .utility.Causal import CausalProjector

class BLocDyn(object):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector):

        self.crystal = crystal
        self.dlr = dlr
        self.projector = projector

    def ResolveProblemKey(self, key):
        if self.projector is None:
            raise ValueError("projector is required to resolve problem key")
        pkey = key
        if hasattr(self.projector, "bprojector") and key not in self.projector.bprojector:
            pkey = str(key)
        if not hasattr(self.projector, "bprojector") or pkey not in self.projector.bprojector:
            raise KeyError(f"Unknown impurity problem key '{key}'")
        if not isinstance(self.projector.equiv, dict) or pkey not in self.projector.equiv:
            raise KeyError(f"Projector equivalence matrix is missing key '{pkey}'")
        return pkey

    def BosonUniform2DLR(self, mat : np.ndarray, omega : np.ndarray = None, hermitian: bool = False) -> np.ndarray:
        mat = np.asarray(mat, dtype=np.complex128)
        nfreq = mat.shape[-1]
        if omega is None:
            omega = 2.0 * np.pi / self.dlr.beta * np.arange(nfreq, dtype=np.float64)
        else:
            omega = np.asarray(omega, dtype=np.float64)
        if omega.ndim != 1:
            raise ValueError(f"omega must be a 1D Matsubara grid, got {omega.ndim}D")
        if omega.size == 0:
            raise ValueError("omega must be a non-empty 1D Matsubara grid")
        if omega.shape[0] != nfreq:
            raise ValueError(
                f"frequency dimension {nfreq} does not match omega length {omega.shape[0]}"
            )

        if omega.ndim == 1 and omega.size > 0 and np.all(omega >= 0.0):
            istart = 1 if np.isclose(omega[0], 0.0) else 0
            if hermitian:
                neg = np.swapaxes(np.conjugate(mat[..., istart:][..., ::-1]), 0, 1)
            else:
                neg = np.conjugate(mat[..., istart:][..., ::-1])
            mat = np.concatenate((neg, mat), axis=-1)
            omega = np.concatenate((-omega[istart:][::-1], omega))
            nfreq = mat.shape[-1]

        mat_t = np.moveaxis(mat, -1, 0)
        mat_2d = np.ascontiguousarray(mat_t).reshape(nfreq, int(np.prod(mat.shape[:-1])))
        mat_dlr = self.dlr.MatsubaraUniform2DLR(mat_2d, omega=omega, sign=1)
        mat_dlr = mat_dlr.reshape(len(self.dlr.nu), *mat.shape[:-1])
        mat_dlr = np.moveaxis(mat_dlr, 0, -1)

        return np.asfortranarray(mat_dlr)

    def _as_dyn_dict(self, key = None, dyn : dict = None) -> dict:
        if dyn is not None:
            return dyn

        if key is None and hasattr(self, "key"):
            key = self.key
        if key is None or not hasattr(self, "ubar_rf") or self.ubar_rf is None:
            return {"1": np.zeros(len(self.dlr.nu), dtype=float).tolist()}

        pkey = self.ResolveProblemKey(key)
        if hasattr(self, "key") and self.key != pkey:
            raise ValueError(f"object key '{self.key}' does not match requested key '{pkey}'")

        dyn = np.asarray(self.ubar_rf, dtype=np.complex128)

        return {"1": self._dynamic_average(pkey, dyn).real.tolist()}

    def _write_json_pair(self, stem : str, iter : int, key, payload : dict) -> None:
        with open(f'{stem}.{iter}.{key}.json', 'w') as outfile:
            json.dump(payload, outfile, sort_keys=True, indent=4, separators=(',', ': '))
        with open(f'{stem}.json', 'w') as outfile:
            json.dump(payload, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    def Projection(self, matin : np.ndarray, key) -> np.ndarray:
        if self.projector is None:
            raise ValueError("projector is required for Projection")

        matin = np.asarray(matin, dtype=np.complex128)
        if matin.ndim != 5:
            raise ValueError(f"matin must be 5D, got {matin.ndim}D")

        pkey = self.ResolveProblemKey(key)
        return PJ.BLocDyn(matin, self.projector.bprojector[pkey])

    def Inverse(self, matin : np.ndarray)-> np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]
        nft = matin.shape[4]

        matout = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')
        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex128)
        tempmat2 = np.zeros((norb*ns,norb*ns),dtype=np.complex128)

        for ift in range(nft):
            tempmat = self.crystal.OrbSpin2Composite(matin[...,ift])
            tempmat2 = Common.MatInv(tempmat)
            matout[...,ift] = self.crystal.Composite2OrbSpin(tempmat2)
        
        return matout

    def F2T(self,bf : np.ndarray) -> np.ndarray:

        bf = np.asarray(bf, dtype=np.complex128)
        nfreq = bf.shape[-1]
        if nfreq != len(self.dlr.nu):
            raise ValueError(
                f"frequency dimension {nfreq} does not match bosonic DLR size {len(self.dlr.nu)}"
            )

        bf_t = np.moveaxis(bf, -1, 0)
        batch = int(np.prod(bf.shape[:-1]))
        bf_2d = np.ascontiguousarray(bf_t).reshape(nfreq, batch)

        btau_2d = self.dlr.BatchBF2T(bf_2d)

        ntau = btau_2d.shape[0]
        btau = btau_2d.reshape(ntau, *bf.shape[:-1])
        btau = np.moveaxis(btau, 0, -1)
        btau = np.asfortranarray(btau)

        return btau

    def T2F(self, btau : np.ndarray) -> np.ndarray:

        norb = btau.shape[0]
        ns = btau.shape[2]
        ntau = btau.shape[4]

        btau_t = np.moveaxis(btau, -1, 0)
        batch = norb * norb * ns * ns
        btau_2d = np.ascontiguousarray(btau_t).reshape(ntau, batch)

        bf_2d = self.dlr.BatchBT2F(btau_2d)

        nfreq = bf_2d.shape[0]
        bf = bf_2d.reshape(nfreq, norb, norb, ns, ns)
        bf = np.moveaxis(bf, 0, -1)  # (norb, norb, ns, ns, nrk, nfreq)
        bf = np.asfortranarray(bf)

        return bf

    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]

        ynew = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')
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
                        for ks in range(ns):
                            for iorb in range(norb):
                                for jorb in range(norb):
                                    ynew[iorb,jorb,js,ks,cnt] = sum(dist*y[iorb,jorb,js,ks])/sum(dist)
                else:
                    ynew[...,cnt] = y[...,cnt]
            cnt += 1

        return ynew
    
    def Mixing(self,iter : int, mix : float, Bb : np.ndarray, Bold : np.ndarray):

        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nft = Bb.shape[4]

        Bnew = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        if iter == 1:
            mix = 1.0
            Bold = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        Bnew = mix*Bb + (1-mix)*Bold

        return Bnew

    def _resolve_equiv_matrix(self, imp=None, key=None) -> np.ndarray:
        """Resolve an equivalent-orbital matrix from legacy/new impurity inputs."""
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

    def AverageByEquiv(self, equiv : np.ndarray, matin : np.ndarray, squeeze : bool = True) -> np.ndarray:
        """Average equivalent orbital classes and return array in one pass."""
        if matin.ndim == 3:
            if self.crystal.ns != 1:
                raise ValueError(
                    f"3D input is only allowed for ns=1, crystal ns={self.crystal.ns}"
                )
            matin5 = matin[:, :, np.newaxis, np.newaxis, :]
        elif matin.ndim == 5:
            matin5 = matin
        else:
            raise ValueError(f"matin must be 3D or 5D, got {matin.ndim}D")

        norb = matin5.shape[0]
        if matin5.shape[1] != norb:
            raise ValueError("matin first two dimensions must be square")
        if equiv.shape[0] != norb or equiv.shape[1] != norb:
            raise ValueError(
                f"equiv shape {equiv.shape} is incompatible with matin shape {matin5.shape}"
            )
        if matin5.shape[2] != self.crystal.ns or matin5.shape[3] != self.crystal.ns:
            raise ValueError(
                "spin dimension mismatch: "
                f"matin spin shape=({matin5.shape[2]}, {matin5.shape[3]}), "
                f"crystal ns={self.crystal.ns}"
            )

        matout = np.array(matin5, dtype=np.complex128, copy=True, order='F')
        nind = int(np.amax(equiv))
        if nind <= 0:
            raise ValueError("equiv labels must be positive integers")

        for ind in range(1, nind + 1):
            pos = Common.FindPositions(equiv, ind)
            if len(pos) == 0:
                continue
            for js in range(self.crystal.ns):
                for ks in range(self.crystal.ns):
                    avg = np.zeros(matin5.shape[4], dtype=np.complex128)
                    for ii, jj in pos:
                        avg += matin5[ii, jj, js, ks, :]
                    avg /= len(pos)
                    for ii, jj in pos:
                        matout[ii, jj, js, ks, :] = avg

        if squeeze and self.crystal.ns == 1:
            return matout[:, :, 0, 0, :]
        return matout

    def AverageImpurityByEquiv(self, imp=None, matimp : dict = None, squeeze : bool = True) -> dict:
        """Average equivalent orbital classes for all impurity problems at once."""
        if not isinstance(matimp, dict):
            raise TypeError("matimp must be dict keyed by impurity problem key")

        matout = {}
        for key, matin in matimp.items():
            equiv = self._resolve_equiv_matrix(imp=imp, key=key)
            matout[str(key)] = self.AverageByEquiv(equiv=equiv, matin=matin, squeeze=squeeze)

        return matout
    
    def Arr2Dict(self, equiv : np.ndarray, matin : np.ndarray) -> dict:
        
        ns = matin.shape[2]
        nind = np.amax(equiv)
        matdict = {}

        for ind in range(nind):
            matdict[ind+1] = []
            pos = self.crystal.FindPositions(equiv,ind+1)
            for js in range(ns):
                for ks in range(ns):
                    e = 0
                    for ii, jj in pos:
                        e += matin[ii,jj,js,ks]
                    e /=len(pos)
                    matdict[ind+1].append(e.tolist())
        
        return matdict
    
    def Dict2Arr(self,equiv : np.ndarray, matdict : np.ndarray) -> np.ndarray:

        norb = len(equiv)
        ns = self.crystal.ns
        nfreq = len(matdict["1"])                

        matout = np.zeros((norb,norb,ns,ns,nfreq),dtype=np.complex128,order='F')
        nind = np.amax(equiv)

        for js in range(ns):
            for ks in range(ns):
                for ind in range(nind):
                    pos = self.crystal.FindPositions(equiv,ind+1)
                    for ii, jj in pos:
                        matout[ii,jj,js,ks] = matdict[str(ind+1)]

        return matout

    def ReadDict(self, equiv : np.ndarray, mat_dict : dict) -> np.ndarray:
        """Read equivalent-orbital dict data as a local dynamic bosonic matrix."""
        norb = len(equiv)
        ns = self.crystal.ns
        sample = mat_dict["1"] if "1" in mat_dict else mat_dict[1]
        nfreq = len(sample)

        mat_out = np.zeros((norb,norb,ns,ns,nfreq), dtype=np.complex128, order='F')
        nind = int(np.amax(equiv))

        for js in range(ns):
            for ks in range(ns):
                for ind in range(1, nind + 1):
                    key = str(ind) if str(ind) in mat_dict else ind
                    if key not in mat_dict:
                        continue
                    val = np.asarray(mat_dict[key], dtype=np.complex128)
                    pos_row, pos_col = np.where(equiv == ind)
                    for ii, jj in zip(pos_row, pos_col):
                        mat_out[ii,jj,js,ks,:] = val

        return mat_out

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        return Dyson.BLocDyn(mat1, mat2)
    
    def _static_to_dynamic(self, mat : np.ndarray) -> np.ndarray:
        return np.repeat(mat[..., np.newaxis], len(self.dlr.nu), axis=4)

    def _ResolveCausalGrid(self, grid : str) -> np.ndarray:
        """Bosonic Matsubara sampling grid for a causal projection.

        ``grid='dlr'``     -> ``self.dlr.nu`` (sparse DLR sampling grid).
        ``grid='uniform'`` -> ``self.dlr.MatsubaraBosonUniform()`` (uniform
        non-negative-frequency grid covering the DLR range).

        The returned array is what the input data's frequency dimension is
        validated against, so the two can never drift apart.
        """
        if grid == "dlr":
            return np.asarray(self.dlr.nu, dtype=np.float64)
        if grid == "uniform":
            return np.asarray(self.dlr.MatsubaraBosonUniform(), dtype=np.float64)
        raise ValueError(f"grid must be 'dlr' or 'uniform', got {grid!r}")

    def CausalProjection(
        self,
        matin : np.ndarray,
        *,
        grid : str = "dlr",
        coefficient_sign : int = -1,
        reflection_symmetry : bool = True,
        solvers = None,
        max_iter : int = 100000,
        constraint_tol : float = 1.0e-8,
        fit_tol : float = 1.0e-6,
        tail_tol : float = 1.0e-1,
    ) -> np.ndarray:
        """Project diagonal local bosonic channels onto real pole-weight
        causal QP via CausalProjector.

        ``grid`` selects the Matsubara sampling grid the input data lives on:
        ``"dlr"`` (sparse DLR grid, default) or ``"uniform"`` (uniform
        non-negative-frequency grid). The DLR pole basis is unchanged either way;
        uniform output is returned on the DLR grid.

        A frequency-independent ``c0`` is estimated from the input tail, removed
        before the decaying pole projection, and re-added to the returned
        channel.  If ``target - c0`` does not decay at the largest ``|nu|``
        nodes, ``RuntimeError`` is raised (``tail_tol`` controls this guard).

        Only the diagonal blocks ``[iorb, iorb, is_, is_, :]`` are projected;
        all other entries are copied unchanged.
        """

        nu = self._ResolveCausalGrid(grid)
        nfreq = len(nu)

        arr = np.asarray(matin, dtype=np.complex128)
        if arr.ndim != 5:
            raise ValueError(f"matin must be 5D (norb,norb,ns,ns,nfreq), got {arr.ndim}D")

        norb = arr.shape[0]
        ns = self.crystal.ns
        if arr.shape[1] != norb:
            raise ValueError("matin first two dimensions must be square")
        if arr.shape[2] != ns or arr.shape[3] != ns:
            raise ValueError(
                f"spin dimension mismatch: matin ns=({arr.shape[2]},{arr.shape[3]}), "
                f"crystal ns={ns}"
            )
        if arr.shape[4] != nfreq:
            raise ValueError(
                f"frequency dimension mismatch: matin nf={arr.shape[4]}, grid nf={nfreq}"
            )
        if not np.all(np.isfinite(np.real(arr))) or not np.all(np.isfinite(np.imag(arr))):
            raise ValueError("matin contains non-finite values")

        # Estimate physical moments on the input grid before any interpolation,
        # then pass [high, moment...] explicitly to the projector.
        moment, high = self.Moment(arr, grid=grid)

        # For uniform input, interpolate the data onto the DLR basis.  The
        # projection grid is always the DLR grid; uniform output is returned on
        # the DLR grid.
        if grid == "uniform":
            arr = self.dlr.MatsubaraUniformGrid2DLR(arr, omega=nu, sign=1)

        proj_nu = np.asarray(self.dlr.nu, dtype=np.float64)
        projector = CausalProjector(
            statistic="B",
            d=self.dlr.dB,
            beta=self.dlr.beta,
            omega=proj_nu,
            coefficient_sign=coefficient_sign,
            reflection_symmetry=reflection_symmetry,
            solvers=solvers,
            max_iter=max_iter,
            constraint_tol=constraint_tol,
            fit_tol=fit_tol,
            tail_tol=tail_tol,
            raise_on_failure=True,
        )
        out = np.array(arr, dtype=np.complex128, copy=True, order='F')
        for is_ in range(ns):
            for iorb in range(norb):
                tail_coeffs = np.empty(4, dtype=float)
                tail_coeffs[0] = float(np.real(high[iorb, iorb, is_, is_]))
                tail_coeffs[1:] = np.real(moment[iorb, iorb, is_, is_, :])
                out[iorb, iorb, is_, is_, :] = projector.project(
                    arr[iorb, iorb, is_, is_, :],
                    tail_coeffs=tail_coeffs,
                )

        return np.asfortranarray(out)

    def Moment(
        self,
        bf: np.ndarray,
        oddzero: bool = False,
        highzero: bool = False,
        tail_points: int = 5,
        grid: str = "dlr",
    ) -> tuple:
        """Physical high-frequency moments of a local bosonic function.

        ``grid`` selects the sampling grid of ``bf``.  Returns
        ``moment[..., :] = [c1, c2, c3]`` and ``high = c0`` in physical sign.
        ``oddzero``/``highzero`` are accepted for compatibility but do not alter
        the robust tail fit.
        """
        arr = np.asarray(bf, dtype=np.complex128)
        if arr.ndim != 5:
            raise ValueError(f"bf must be 5D (norb,norb,ns,ns,nfreq), got {arr.ndim}D")
        nu = self._ResolveCausalGrid(grid)
        if arr.shape[4] != nu.size:
            raise ValueError(
                f"frequency dimension {arr.shape[4]} does not match {grid} nu size {nu.size}"
            )
        if arr.shape[4] < tail_points:
            raise ValueError(
                f"Need at least {tail_points} frequency points to build "
                "high-frequency moments."
            )

        norb = arr.shape[0]
        ns = arr.shape[2]
        moment = np.zeros((norb, norb, ns, ns, 3), dtype=np.complex128, order="F")
        high = np.zeros((norb, norb, ns, ns), dtype=np.complex128, order="F")
        idx = np.argsort(np.abs(nu))[-tail_points:]
        z = 1j * nu[idx]
        design = np.column_stack(
            [np.ones_like(z), 1.0 / z, 1.0 / z**2, 1.0 / z**3]
        )
        for is_ in range(ns):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        if iorb == jorb and is_ == js:
                            tail = Fourier.BosonTailCoefficients(
                                nu, arr[iorb, jorb, is_, js, :], tail_points
                            ).astype(np.complex128)
                        else:
                            tail, *_ = np.linalg.lstsq(
                                design, arr[iorb, jorb, is_, js, idx], rcond=None
                            )
                        high[iorb, jorb, is_, js] = tail[0]
                        moment[iorb, jorb, is_, js, :] = tail[1:]
        high_orig = high.copy()
        high[...] = 0.5 * (
            high_orig + np.swapaxes(np.swapaxes(high_orig, 0, 1), 2, 3).conj()
        )
        for imom in range(3):
            mom_orig = moment[..., imom].copy()
            moment[..., imom] = 0.5 * (
                mom_orig + np.swapaxes(np.swapaxes(mom_orig, 0, 1), 2, 3).conj()
            )

        return moment, high

class Chi(BLocDyn):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, key, partition : dict = None, hdf5file : str = None, group : str = None):

        super().__init__(crystal, dlr, projector)

        self.key = self.ResolveProblemKey(key)
        self.partition = partition
        self.occupation = None
        self.occupation_susceptibility_bulla = None
        self.occupation_susceptibility_xij = None
        self.f = None
        self.t = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.Cal()

    def _json_complex_array(self, val) -> np.ndarray:

        if isinstance(val, dict):
            if "function" in val:
                return self._json_complex_array(val["function"])
            if "real" in val:
                real = np.asarray(val["real"], dtype=float)
                imag = np.asarray(val.get("imag", np.zeros_like(real)), dtype=float)
                return real + 1j * imag
            if "moments" in val:
                return self._json_complex_array(val["moments"])

        arr = np.asarray(val)
        if arr.dtype == object:
            return arr.astype(np.complex128)
        return np.asarray(arr, dtype=np.complex128)

    def _json_scalar(self, val) -> complex:

        arr = self._json_complex_array(val)
        if arr.ndim == 0:
            return complex(arr)
        if arr.size == 0:
            raise ValueError("cannot read scalar from empty JSON array")
        return complex(arr.flat[0])

    def _read_occupation(self, partition : dict):

        if "occupation" not in partition:
            return None

        occupation = {}
        for occ_key, val in partition["occupation"].items():
            occupation[occ_key] = self._json_scalar(val)

        equiv = np.asarray(self.projector.equiv[self.key], dtype=int)
        norb = len(equiv)
        ns = self.crystal.ns
        matout = np.zeros((norb, norb, ns), dtype=np.complex128, order='F')
        nind = int(np.amax(equiv))

        for js in range(ns):
            for ind in range(1, nind + 1):
                dict_key = str(ind) if str(ind) in occupation else ind
                if dict_key not in occupation:
                    continue
                pos = Common.FindPositions(equiv, ind)
                for ii, jj in pos:
                    matout[ii, jj, js] = occupation[dict_key]

        return matout

    def QuadSusceptibility2Boson(self, chi : np.ndarray) -> np.ndarray:
        chi = np.asarray(chi, dtype=np.complex128)
        if chi.ndim == 5:
            return np.asfortranarray(chi)
        if chi.ndim != 7:
            raise ValueError(f"chi must be 5D or 7D, got {chi.ndim}D")

        nspin = chi.shape[4]
        nfreq = chi.shape[6]
        ns = self.crystal.ns
        norbb = self.projector.bprojector[self.key].shape[1]

        chi_temp = np.zeros((norbb, norbb, nspin, nspin, nfreq), dtype=np.complex128, order='F')
        for ib in range(norbb):
            iorb, lorb = self.projector.ProbBorb2FPair(self.key, ib)
            for jb in range(norbb):
                jorb, korb = self.projector.ProbBorb2FPair(self.key, jb)
                chi_temp[ib, jb, :, :, :] = chi[iorb, jorb, korb, lorb, :, :, :]

        chi_boson = np.zeros((norbb, norbb, ns, ns, nfreq), dtype=np.complex128, order='F')
        if ns == 1 and nspin == 2:
            chi_boson[:, :, 0, 0, :] = 0.5 * (
                chi_temp[:, :, 0, 0, :]
                + chi_temp[:, :, 0, 1, :]
                + chi_temp[:, :, 1, 0, :]
                + chi_temp[:, :, 1, 1, :]
            )
        else:
            ncopy = min(ns, nspin)
            for js in range(ncopy):
                for ks in range(ncopy):
                    chi_boson[:, :, js, ks, :] = chi_temp[:, :, js, ks, :]

        return chi_boson

    def _fermion_orbital_count_from_boson(self) -> int:
        pairs = self.projector.blocal2pair[self.key][0].values()
        if len(pairs) == 0:
            raise ValueError(f"bosonic projector for key '{self.key}' has no local pairs")
        return max(max(pair) for pair in pairs) + 1

    def _read_occupation_susceptibility_bulla(self, partition : dict):

        dict_name = "occupation-susceptibility-bulla"
        if dict_name not in partition:
            return None

        xij_json = partition[dict_name]
        pairs = []
        ndim = 0
        nfreq = None
        for xij_key, val in xij_json.items():
            try:
                ii, jj = [int(ind) for ind in str(xij_key).split("_")]
            except ValueError as exc:
                raise ValueError(
                    f"invalid occupation susceptibility key '{xij_key}', expected '<i>_<j>'"
                ) from exc

            arr = self._json_complex_array(val)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            if nfreq is None:
                nfreq = arr.shape[0]
            elif arr.shape[0] != nfreq:
                raise ValueError(
                    "occupation susceptibility frequency length mismatch: "
                    f"key '{xij_key}' has {arr.shape[0]}, expected {nfreq}"
                )

            pairs.append((ii, jj, arr))
            ndim = max(ndim, ii + 1, jj + 1)

        if nfreq is None:
            return np.zeros((0, 0, 0), dtype=np.complex128)

        xij = np.zeros((nfreq, ndim, ndim), dtype=np.complex128)
        for ii, jj, arr in pairs:
            xij[:, ii, jj] = arr

        self.occupation_susceptibility_xij = np.moveaxis(
            self.BosonUniform2DLR(np.moveaxis(xij, 0, -1)),
            -1,
            0,
        )

        norbc = self._fermion_orbital_count_from_boson()
        nspin = ndim // norbc
        if nspin * norbc != ndim:
            raise ValueError(
                f"occupation susceptibility dimension {ndim} is incompatible with bosonic pair map norbc={norbc}"
            )

        susc = np.zeros((norbc, norbc, norbc, norbc, nspin, nspin, nfreq), dtype=np.complex128, order='F')
        for ind1 in range(ndim):
            nn1 = [0] * 2
            _, [iorb, ispin] = Common.Indexing(ndim, 2, [norbc, nspin], 0, ind1, nn1)
            for ind2 in range(ndim):
                nn2 = [0] * 2
                _, [jorb, jspin] = Common.Indexing(ndim, 2, [norbc, nspin], 0, ind2, nn2)
                susc[iorb, jorb, jorb, iorb, ispin, jspin, :] = xij[:, ind1, ind2]

        return susc

    def Cal(self):

        if self.partition is None:
            return None

        self.occupation = self._read_occupation(self.partition)
        susc_uniform = self._read_occupation_susceptibility_bulla(self.partition)
        if susc_uniform is None:
            return None

        susc_uniform = self.QuadSusceptibility2Boson(susc_uniform)
        self.f = self.BosonUniform2DLR(susc_uniform)
        self.occupation_susceptibility_bulla = self.f
        self.t = self.F2T(self.f)

        return None

    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            chi = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(chi, fn, obj, dtype=complex)
            else:
                Common.HDF5CreateDataset(chi, fn, self.f, dtype=complex)

        return None


class PImp(BLocDyn):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, key, chi : Chi = None, utilde = None, hdf5file : str = None, group : str = None):

        super().__init__(crystal, dlr, projector)

        self.key = self.ResolveProblemKey(key)
        self.chi = chi
        self.utilde = utilde
        self.chi_boson = None
        self.f = None
        self.t = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.Cal()

    def _quad_to_local_boson(self, chi : np.ndarray) -> np.ndarray:

        chi = np.asarray(chi, dtype=np.complex128)
        if chi.ndim != 5:
            raise ValueError(f"PImp expects 5D bosonic susceptibility, got {chi.ndim}D")
        return np.asfortranarray(chi)

    def _dynamic_interaction(self, nfreq : int):

        if self.utilde is None:
            return None

        u = np.asarray(self.utilde, dtype=np.complex128)
        bproj = self.projector.bprojector[self.key]
        norbb = bproj.shape[1]

        if u.ndim == 5:
            u = u[..., :nfreq]
        elif u.ndim == 4:
            if u.shape[0] != norbb:
                u = PJ.BLocStc(u, bproj)
            u = np.repeat(u[..., np.newaxis], nfreq, axis=4)
        else:
            raise ValueError(f"utilde must be 4D or 5D, got {u.ndim}D")

        if u.shape[0] != norbb:
            uproj = np.zeros((norbb, norbb, u.shape[2], u.shape[3], u.shape[4]), dtype=np.complex128, order='F')
            for ifreq in range(u.shape[4]):
                uproj[..., ifreq] = PJ.BLocStc(u[..., ifreq], bproj)
            u = uproj

        return np.asfortranarray(u)

    def Cal(self):

        chi = self.chi.f if hasattr(self.chi, "f") else self.chi
        if chi is None:
            return None

        chi = np.asarray(chi, dtype=np.complex128)
        if chi.shape[-1] != len(self.dlr.nu):
            chi = self.BosonUniform2DLR(chi)

        self.chi_boson = self._quad_to_local_boson(chi)
        utilde = self._dynamic_interaction(self.chi_boson.shape[4])
        if utilde is None:
            self.f = self.chi_boson
        else:
            self.f = self.Dyson(self.chi_boson, -utilde)

        self.t = self.F2T(self.f)

        return None

    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            pimp = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(pimp, fn, obj, dtype=complex)
            else:
                Common.HDF5CreateDataset(pimp, fn, self.f, dtype=complex)

        return None

class BWeiss(BLocDyn):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, key, vloc : VLoc, ploc = None, wloc = None, hdf5file : str = None, group : str = None):

        super().__init__(crystal, dlr, projector)

        self.key = self.ResolveProblemKey(key)
        self.vloc = vloc
        if hasattr(self.vloc, "projector") and self.vloc.projector is None:
            self.vloc.projector = projector
        self.ploc = ploc
        self.wloc = wloc
        self.v = None
        self.v_rf = None
        self.utilde_t = None
        self.utilde_rf = None
        self.ubar_t = None
        self.ubar_rf = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        if not hasattr(self.vloc, "vloc") or self.vloc.vloc is None:
            raise TypeError("BWeiss requires vloc to be a VLoc object with vloc data")

        self.v = self.vloc.Projection(self.vloc.vloc, self.key)
        if (self.wloc is None) != (self.ploc is None):
            raise ValueError("BWeiss requires both wloc and ploc, or neither")
        if self.wloc is not None and self.ploc is not None:
            self.Cal()

    def _dynamic_average(self, key, mat : np.ndarray) -> np.ndarray:
        norbb = mat.shape[0]
        ns = mat.shape[2]
        nfreq = mat.shape[4]
        pkey = self.ResolveProblemKey(key)
        pairs = self.projector.blocal2pair[pkey][0].values()
        if len(pairs) == 0:
            raise ValueError(f"bosonic projector for key '{pkey}' has no local pairs")
        norbc = max(max(pair) for pair in pairs) + 1

        out = np.zeros(nfreq, dtype=np.complex128)
        count = 0
        for iorb in range(norbc):
            ib = self.projector.ProbFPair2Borb(pkey, iorb, iorb)
            for jorb in range(norbc):
                jb = self.projector.ProbFPair2Borb(pkey, jorb, jorb)
                for js in range(ns):
                    for ks in range(ns):
                        out += mat[ib, jb, js, ks, :]
                        count += 1

        if count == 0:
            return out
        return out / float(count)

    def Cal(self):
        nfreq = len(self.dlr.nu)

        if self.wloc is None or self.ploc is None:
            raise ValueError("BWeiss.Cal requires both wloc and ploc")

        self.v_rf = self._static_to_dynamic(self.v)
        w = self.Projection(self.wloc.f, self.key)
        p = self.Projection(self.ploc.f, self.key)
        self.utilde_rf = self.Dyson(w, -p)

        self.ubar_rf = self.utilde_rf - self.v_rf

        if self.utilde_rf.shape[4] != nfreq:
            raise ValueError(
                f"BWeiss frequency mismatch for key '{self.key}': "
                f"{self.utilde_rf.shape[4]} != {nfreq}"
            )

        self.utilde_t = self.F2T(self.utilde_rf)
        self.ubar_t = self.F2T(self.ubar_rf)

        return None

    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            bweiss = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(bweiss, fn, obj, dtype=complex)
            else:
                Common.HDF5CreateDataset(bweiss, fn, self.v, dtype=complex)

        return None
# class PolLoc(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, green, pol : object):
#         super().__init__(crystal, ft)
#         self.Cal()

#     def Cal(self):
#         pass

# class PolImp(BLocDyn): # read Polarizability from CTQMC

#     def __init__(self, crystal: Crystal, ft: FTGrid):
#         super().__init__(crystal, ft)

#         pass

# class WLoc(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass

# class WImp(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass

# class WcLoc(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass

# class WcImp(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass
