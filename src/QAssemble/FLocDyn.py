import numpy as np
import logging
import sys
import json
import h5py
import warnings
from .Crystal import Crystal
from .FLocStc import EImp
from .Projector import Projector
from .utility.DLR import DLR
from .utility.Common import Common
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
from .utility.Projection import Projection as PJ
from .utility.Causal import CausalFermionProjector
from .utility.HDF5 import IO
from .utility.Mixing import Mixing as MixingKernel

logger = logging.getLogger("QAssemble")

# Tail-fit point count for the log-spaced uniform-grid mode.  Validated on
# production glob.h5 data: ~24 log-spaced points over the top |omega| decade
# recover [c1, c2, c3] within the noise (well-conditioned design), whereas the
# historical contiguous 5-point block on the signed uniform grid has condition
# number ~5e11 and produces garbage c2/c3 that make the moment cone infeasible.
_LOG_TAIL_POINTS = 24

class FLocDyn(object):
    mixer = MixingKernel()

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

    def TauB2TauF(self, btau : np.ndarray) -> np.ndarray:

        nborb = btau.shape[0]
        ns = btau.shape[2]
        ntau = btau.shape[4]

        btau_t = np.moveaxis(btau, -1, 0)
        batch = nborb * nborb * ns * ns
        btau_2d = np.ascontiguousarray(btau_t).reshape(ntau, batch)

        bxx = self.dlr.dB.dlr_from_tau(btau_2d)
        btauf_2d = self.dlr.dB.eval_dlr_tau(
            bxx[:, :, None], self.dlr.tauF, self.dlr.beta
        )[:, :, 0]

        ntauF = btauf_2d.shape[0]
        btauf = btauf_2d.reshape(ntauF, nborb, nborb, ns, ns)
        btauf = np.moveaxis(btauf, 0, -1)
        btauf = np.asfortranarray(btauf)

        return btauf

    def _ResolveCausalGrid(self, grid : str) -> np.ndarray:
        """Fermionic Matsubara sampling grid for a causal projection.

        ``grid='dlr'``     -> ``self.dlr.omega`` (sparse DLR sampling grid).
        ``grid='uniform'`` -> ``self.dlr.MatsubaraFermionUniformFull()`` (full
        signed uniform grid covering the DLR range).

        The returned array is what the input data's frequency dimension is
        validated against, so the two can never drift apart.
        """
        if grid == "dlr":
            return np.asarray(self.dlr.omega, dtype=np.float64)
        if grid == "uniform":
            return np.asarray(self.dlr.MatsubaraFermionUniformFull(), dtype=np.float64)
        raise ValueError(f"grid must be 'dlr' or 'uniform', got {grid!r}")

    def _ExpandPositiveUniform(self, arr4 : np.ndarray) -> np.ndarray:
        """Expand a positive-only (iw_n >= 0) uniform fermion input to the full
        signed grid, leaving a full signed input untouched.

        ``arr4`` is ``(norb, norb, ns, nfreq)``.  When ``nfreq`` matches the
        positive-only grid ``MatsubaraFermionUniform()``, the negative half is
        reconstructed via the Hermitian relation G(-iw)=G(iw)dagger using the
        transpose-correct :meth:`DLR.MatsubaraAddNegativeFrequency`; the result
        then matches ``MatsubaraFermionUniformFull()`` so the downstream length
        check, tail fit, and DLR conversion proceed unchanged.  When ``nfreq``
        already matches the full signed grid, the input is returned as-is.
        """
        nfreq_in = arr4.shape[3]
        npos = len(self.dlr.MatsubaraFermionUniform())
        if nfreq_in == npos:
            return self.dlr.MatsubaraAddNegativeFrequency(arr4)
        return arr4

    def CausalProjection(
        self,
        matin : np.ndarray,
        *,
        grid : str = "dlr",
        coefficient_sign : int = -1,
        solvers = None,
        max_iter : int = 100000,
        constraint_tol : float | str = 1.0e-8,
        fit_tol : float = 1.0e-6,
    ) -> np.ndarray:
        """Project diagonal local fermionic channels onto real pole-weight
        causal QP via CausalFermionProjector.

        ``grid`` selects the Matsubara sampling grid the input data lives on:
        ``"dlr"`` (sparse DLR grid, default) or ``"uniform"`` (full signed
        uniform grid). The DLR pole basis is unchanged either way.

        Physical tail coefficients are estimated on the input grid before any
        interpolation and passed to the projector; only the projector's QP
        layer converts them to its internal sign convention.  Data not
        representable by the real-coefficient pole basis (node fit residual
        above ``fit_tol``) raise ``RuntimeError``.
        """

        omega = self._ResolveCausalGrid(grid)
        nfreq = len(omega)

        arr = np.asarray(matin, dtype=np.complex128)
        squeeze_spin = False
        if arr.ndim == 3:
            if self.crystal.ns != 1:
                raise ValueError(
                    f"3D input is only allowed for ns=1, crystal ns={self.crystal.ns}"
                )
            arr4 = arr[:, :, np.newaxis, :]
            squeeze_spin = True
        elif arr.ndim == 4:
            arr4 = arr
        else:
            raise ValueError(f"matin must be 3D or 4D, got {arr.ndim}D")

        norb = arr4.shape[0]
        if arr4.shape[1] != norb:
            raise ValueError("matin first two dimensions must be square")
        if arr4.shape[2] != self.crystal.ns:
            raise ValueError(
                f"spin dimension mismatch: matin ns={arr4.shape[2]}, crystal ns={self.crystal.ns}"
            )
        # Accept positive-only (iw_n >= 0) uniform input by expanding it to the
        # full signed grid via Hermitian symmetry before the length check, so a
        # caller can pass either the positive-only or the full signed grid.
        if grid == "uniform":
            arr4 = self._ExpandPositiveUniform(arr4)
        if arr4.shape[3] != nfreq:
            raise ValueError(
                f"frequency dimension mismatch: matin nf={arr4.shape[3]}, grid nf={nfreq}"
            )
        if not np.all(np.isfinite(np.real(arr4))) or not np.all(np.isfinite(np.imag(arr4))):
            raise ValueError("matin contains non-finite values")

        # Estimate physical moments on the input grid before any interpolation,
        # then pass [high, moment...] explicitly to the projector.  The fit
        # sigmas drive the elastic moment penalties (mu = 1/sigma) so noisy
        # moment estimates can never make the QP infeasible.
        moment, high, sigma = self.Moment(arr4, grid=grid, return_sigma=True)

        # For uniform input, interpolate the data onto the DLR basis.  The
        # projection grid is always the DLR grid; uniform output is returned on
        # the DLR grid.
        if grid == "uniform":
            # conversion preserves the (norb, norb, ns, .) layout; squeeze rule
            # is unchanged
            arr4 = self.dlr.MatsubaraUniformGrid2DLR(arr4, sign=-1)

        proj_omega = np.asarray(self.dlr.omega, dtype=np.float64)
        projector = CausalFermionProjector(
            d=self.dlr.dF,
            beta=self.dlr.beta,
            omega=proj_omega,
            coefficient_sign=coefficient_sign,
            solvers=solvers,
            max_iter=max_iter,
            constraint_tol=constraint_tol,
            fit_tol=fit_tol,
            raise_on_failure=True,
        )
        out = np.array(arr4, dtype=np.complex128, copy=True, order='F')
        for js in range(self.crystal.ns):
            for iorb in range(norb):
                tail_coeffs = np.empty(4, dtype=float)
                tail_coeffs[0] = float(np.real(high[iorb, iorb, js]))
                tail_coeffs[1:] = np.real(moment[iorb, iorb, js, :])
                out[iorb, iorb, js, :] = projector.project(
                    arr4[iorb, iorb, js, :],
                    tail_coeffs=tail_coeffs,
                    moment_sigma=sigma[iorb, iorb, js, 1:],
                )

        if squeeze_spin:
            return np.asfortranarray(out[:, :, 0, :])
        return np.asfortranarray(out)

    def CausalityCheck(
        self,
        matin : np.ndarray,
        *,
        grid : str = "dlr",
        coefficient_sign : int = -1,
        solvers = None,
        max_iter : int = 100000,
        constraint_tol : float | str = 1.0e-8,
        fit_tol : float = 1.0e-6,
    ) -> dict:
        """Diagnose causality of diagonal local channels without projecting.

        ``grid`` selects the input sampling grid (``"dlr"`` default or
        ``"uniform"``), as in ``CausalProjection``.

        Returns per-channel arrays shaped ``(norb, ns)`` for 4D input and
        ``(norb,)`` for 3D input (same squeeze rule as ``CausalProjection``):
        ``causal`` boolean, unscaled ``max_inequality_violation`` /
        ``max_equality_residual``, ``violating_count``, and
        ``node_residual``.  A channel whose ``node_residual`` exceeds
        ``fit_tol`` is not representable by the causal pole basis
        (data-quality problem); its sign verdict describes a garbage fit, so
        judge the residual first.  Never raises on bad data — this is the
        diagnostic counterpart of ``CausalProjection``.
        """

        omega = self._ResolveCausalGrid(grid)
        nfreq = len(omega)

        arr = np.asarray(matin, dtype=np.complex128)
        squeeze_spin = False
        if arr.ndim == 3:
            if self.crystal.ns != 1:
                raise ValueError(
                    f"3D input is only allowed for ns=1, crystal ns={self.crystal.ns}"
                )
            arr4 = arr[:, :, np.newaxis, :]
            squeeze_spin = True
        elif arr.ndim == 4:
            arr4 = arr
        else:
            raise ValueError(f"matin must be 3D or 4D, got {arr.ndim}D")

        norb = arr4.shape[0]
        if arr4.shape[1] != norb:
            raise ValueError("matin first two dimensions must be square")
        if arr4.shape[2] != self.crystal.ns:
            raise ValueError(
                f"spin dimension mismatch: matin ns={arr4.shape[2]}, crystal ns={self.crystal.ns}"
            )
        # Accept positive-only (iw_n >= 0) uniform input by expanding it to the
        # full signed grid via Hermitian symmetry before the length check, so a
        # caller can pass either the positive-only or the full signed grid.
        if grid == "uniform":
            arr4 = self._ExpandPositiveUniform(arr4)
        if arr4.shape[3] != nfreq:
            raise ValueError(
                f"frequency dimension mismatch: matin nf={arr4.shape[3]}, grid nf={nfreq}"
            )
        if not np.all(np.isfinite(np.real(arr4))) or not np.all(np.isfinite(np.imag(arr4))):
            raise ValueError("matin contains non-finite values")

        ns = self.crystal.ns
        # mirror CausalProjection: moments on native input grid, then DLR data.
        moment, high, sigma = self.Moment(arr4, grid=grid, return_sigma=True)
        if grid == "uniform":
            arr4 = self.dlr.MatsubaraUniformGrid2DLR(arr4, sign=-1)

        proj_omega = np.asarray(self.dlr.omega, dtype=np.float64)
        projector = CausalFermionProjector(
            d=self.dlr.dF,
            beta=self.dlr.beta,
            omega=proj_omega,
            coefficient_sign=coefficient_sign,
            solvers=solvers,
            max_iter=max_iter,
            constraint_tol=constraint_tol,
            fit_tol=fit_tol,
            raise_on_failure=True,
        )
        causal = np.zeros((norb, ns), dtype=bool)
        max_inequality = np.zeros((norb, ns), dtype=float)
        max_equality = np.zeros((norb, ns), dtype=float)
        violating_count = np.zeros((norb, ns), dtype=int)
        node_residual = np.zeros((norb, ns), dtype=float)
        c0 = np.zeros((norb, ns), dtype=float)
        for js in range(ns):
            for iorb in range(norb):
                tail_coeffs = np.empty(4, dtype=float)
                tail_coeffs[0] = float(np.real(high[iorb, iorb, js]))
                tail_coeffs[1:] = np.real(moment[iorb, iorb, js, :])
                verdict = projector.check(
                    arr4[iorb, iorb, js, :],
                    enforce_gate=False,
                    tail_coeffs=tail_coeffs,
                    moment_sigma=sigma[iorb, iorb, js, 1:],
                )
                causal[iorb, js] = verdict.causal
                max_inequality[iorb, js] = verdict.max_inequality_violation
                max_equality[iorb, js] = verdict.max_equality_residual
                violating_count[iorb, js] = verdict.violating_count
                node_residual[iorb, js] = verdict.node_residual
                c0[iorb, js] = verdict.c0

        report = {
            "causal": causal,
            "max_inequality_violation": max_inequality,
            "max_equality_residual": max_equality,
            "violating_count": violating_count,
            "node_residual": node_residual,
            "c0": c0,
        }
        if squeeze_spin:
            report = {key: value[:, 0] for key, value in report.items()}
        return report

    def Moment(
        self,
        ff: np.ndarray,
        isgreen: bool = False,
        highzero: bool = False,
        tail_points: int = 5,
        grid: str = "dlr",
        return_sigma: bool = False,
    ) -> tuple:
        """Physical high-frequency moments of a local fermionic function.

        ``grid`` selects the sampling grid of ``ff``.  For uniform fermion input,
        positive-only data is expanded before fitting, matching
        ``CausalProjection``.  Returns ``moment[..., :] = [c1, c2, c3]`` and
        ``high = c0`` in physical sign.

        On the uniform grid the fit points are ``_LOG_TAIL_POINTS`` log-spaced
        frequencies over the top decade (``tail_points`` applies to the DLR
        grid only).  With ``return_sigma=True`` a third array of one-sigma fit
        uncertainties, shape ``(norb, norb, ns, 4)`` for ``[c0, c1, c2, c3]``,
        is appended to the return.  Hermitian symmetrization is applied to
        ``moment``/``high`` only — each sigma stays with its own fit.
        """
        arr4 = self._as_dynamic_spin_matrix(ff)
        omega = self._ResolveCausalGrid(grid)
        if grid == "uniform":
            arr4 = self._ExpandPositiveUniform(arr4)
        if arr4.shape[3] != omega.size:
            raise ValueError(
                f"frequency dimension {arr4.shape[3]} does not match {grid} omega size {omega.size}"
            )
        if arr4.shape[3] < tail_points:
            raise ValueError(
                f"Need at least {tail_points} frequency points to build "
                "high-frequency moments."
            )

        norb = arr4.shape[0]
        ns = arr4.shape[2]
        moment = np.zeros((norb, norb, ns, 3), dtype=np.complex128, order="F")
        high = np.zeros((norb, norb, ns), dtype=np.complex128, order="F")
        sigma = np.zeros((norb, norb, ns, 4), dtype=float, order="F")
        log_spaced = grid == "uniform"
        pts = _LOG_TAIL_POINTS if log_spaced else tail_points
        # one shared index selection for the diagonal (FermionTailCoefficients
        # recomputes the identical selection internally) and the off-diagonal
        # complex lstsq below
        idx = Fourier._tail_fit_indices(omega, pts, log_spaced)
        z = 1j * omega[idx]
        design = np.column_stack(
            [np.ones_like(z), 1.0 / z, 1.0 / z**2, 1.0 / z**3]
        )
        for js in range(ns):
            for jorb in range(norb):
                for iorb in range(norb):
                    if iorb == jorb:
                        tail, sig = Fourier.FermionTailCoefficients(
                            omega,
                            arr4[iorb, jorb, js, :],
                            pts,
                            log_spaced=log_spaced,
                            return_sigma=True,
                        )
                        tail = tail.astype(np.complex128)
                    else:
                        b = arr4[iorb, jorb, js, idx]
                        tail, *_ = np.linalg.lstsq(design, b, rcond=None)
                        residual = design @ tail - b
                        dof = max(2 * idx.size - 8, 1)
                        cov = (
                            float(np.sum(np.abs(residual) ** 2)) / dof
                        ) * np.linalg.pinv((design.conj().T @ design).real)
                        sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                    high[iorb, jorb, js] = tail[0]
                    moment[iorb, jorb, js, :] = tail[1:]
                    sigma[iorb, jorb, js, :] = sig
            h = high[:, :, js].copy()
            high[:, :, js] = 0.5 * (h + h.T.conj())
            for imom in range(3):
                m = moment[:, :, js, imom].copy()
                moment[:, :, js, imom] = 0.5 * (m + m.T.conj())

        if return_sigma:
            return moment, high, sigma
        return moment, high

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
    
    def Mixing(
        self,
        iter: int = None,
        mix: float = None,
        component: str = None,
        value: np.ndarray = None,
        method: str = "pulay",
        npulay: int = 5,
        key=None,
    ) -> np.ndarray:
        if iter is None:
            iter = getattr(self, "iteration", None)
        if key is None:
            key = getattr(self, "key", None)
        return IO.MixComponent(
            hdf5file=getattr(self, "hdf5file", None),
            group=getattr(self, "group", None),
            key=key,
            component=component,
            value=value,
            iter=iter,
            mix=mix,
            method=method,
            npulay=npulay,
            mixer=self.mixer,
        )

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

    def __init__(
        self,
        crystal : Crystal,
        dlr : DLR,
        projector : Projector,
        green : np.ndarray,
        key = None,
        hdf5file : str = None,
        group : str = None,
        iteration: int = None,
        scf : bool = True
    ):

        super().__init__(crystal, dlr, projector)

        
        self.key = self._ResolveGLocKey(key)
        self.green = green
        self.f = None
        self.t = None
        self.occ = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.iteration = iteration
        self.scf = scf
        self.Cal()

        self.Save(fn='gloc', scf=self.scf)
        
        
    def _ResolveGLocKey(self, key):
        if key is not None:
            return self.ResolveProblemKey(key)
        if self.projector is None:
            raise ValueError("projector is required to resolve GLoc problem key")
        keys = list(getattr(self.projector, "fprojector", {}).keys())
        if len(keys) == 1:
            return self.ResolveProblemKey(keys[0])
        if len(keys) == 0:
            raise ValueError("GLoc requires at least one impurity problem")
        raise ValueError(
            "GLoc requires key when multiple impurity problems are present"
        )

    def Cal(self):

        self.f = self.Projection(self.green, self.key)
        self.t = self.F2T(self.f)
        self.occ = self.Occ(self.t)

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
    
    def Save(self, fn: str, scf: bool = True):
        
        fn_write = fn
        if self.iteration == 1:
            fn_write = f"{fn_write}_ini"
        if scf:
            if self.iteration is None:
                raise ValueError("GLoc.Save requires iteration when scf=True")
            fn_write = f"{fn}.{self.iteration}"
        if getattr(self, "key", None) is None:
            raise ValueError("GLoc.Save requires key")
        fn_write = f"{fn_write}.{self.key}"

        with h5py.File(self.hdf5file,'a') as file:
            gloc = IO.Group(file, self.group, self.subgroup)
            IO.CreateDataset(gloc, fn_write, self.f, dtype=complex)

        return None
    
class GImp(FLocDyn):

    def __init__(
        self,
        crystal : Crystal,
        dlr : DLR,
        projector : Projector,
        key,
        green,
        hdf5file : str = None,
        group : str = None,
        iteration: int = None,
    ):

        super().__init__(crystal, dlr, projector)

        self.key = self.ResolveProblemKey(key)
        self.green = green
        self.f = None
        self.t = None
        self.occ = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.iteration = iteration
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
        self.occ = self.Occ(self.t)

        return None

    def Occ(self, mat : np.ndarray):

        tau_beta = np.array([self.dlr.beta], dtype=np.float64)
        occ = np.zeros_like(mat[..., 0], dtype=np.complex128)
        for js in range(mat.shape[2]):
            block = mat[:, :, js, :].T

            ntau_b = block.shape[0]
            block_2d = block.reshape(ntau_b, -1)

            fxx = self.dlr.dF.dlr_from_tau(block_2d)
            fout = self.dlr.dF.eval_dlr_tau(fxx[:, :, None], tau_beta, beta=self.dlr.beta)

            occ[:, :, js] = -fout[0, :, 0].reshape(mat.shape[0], mat.shape[0])

        return occ

    def Save(self, fn: str, obj : np.ndarray = None, scf: bool = True):
        if fn is None:
            raise ValueError("GImp.Save requires fn")
        fn_write = fn
        if scf:
            if self.iteration is None:
                raise ValueError("GImp.Save requires iteration when scf=True")
            fn_write = f"{fn}.{self.iteration}.{self.key}"

        with h5py.File(self.hdf5file,'a') as file:
            gloc = IO.Group(file, self.group, self.subgroup)
            if obj is not None:
                IO.CreateDataset(gloc, fn_write, obj, dtype=complex)
            else:
                IO.CreateDataset(gloc, fn_write, self.f, dtype=complex)
                IO.CreateDataset(gloc, fn_write + '_uniform', self.f_uniform, dtype=complex)

        return None


class SigGWCLoc(FLocDyn):

    def __init__(
        self,
        crystal : Crystal,
        dlr : DLR,
        projector : Projector,
        key,
        green : np.ndarray = None,
        wloc : np.ndarray = None,
        hdf5file : str = None,
        group : str = None,
        iteration: int = None,
    ):

        super().__init__(crystal, dlr, projector)

        self.key = self.ResolveProblemKey(key)
        self.green = green
        self.wloc = wloc
        self.f = None
        self.t = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.iteration = iteration
        self.Cal()

    def Cal(self):

        green = np.asarray(self.green, dtype=np.complex128)
        Wc = self.TauB2TauF(np.asarray(self.wloc, dtype=np.complex128))

        norb = green.shape[0]
        ns = green.shape[2]
        ntau = green.shape[3]

        s_idx = np.arange(ns)
        Wc_diag = Wc[:, :, s_idx, s_idx, :]
        pair_map = np.zeros((norb, norb), dtype=int)
        for iorb in range(norb):
            for jorb in range(norb):
                pair_map[iorb, jorb] = self.projector.ProbFPair2Borb(
                    self.key, iorb, jorb
                )

        S = ns * ntau
        Wc_flat = np.ascontiguousarray(Wc_diag).reshape(Wc_diag.shape[0], Wc_diag.shape[1], S)
        green_flat = np.ascontiguousarray(green).reshape(norb, norb, S)
        sigma_flat = np.zeros((norb, norb, S), dtype=np.complex128)

        temp_by_pair = {}
        unique_pairs = np.unique(pair_map)
        for ib in unique_pairs:
            mask = (pair_map == ib).astype(np.float64)
            temp_by_pair[ib] = np.einsum('ki,ijS->kjS', mask, green_flat)

        for ib in unique_pairs:
            for jb in unique_pairs:
                mask_b = (pair_map == jb).astype(np.float64)
                contracted = np.einsum('kjS,jp->kpS', temp_by_pair[ib], mask_b)
                sigma_flat -= Wc_flat[ib, jb][np.newaxis, np.newaxis, :] * contracted

        sigma_tau = sigma_flat.reshape(norb, norb, ns, ntau)

        self.t = np.asfortranarray(sigma_tau)
        self.f = self.T2F(self.t)

        return None

    def Save(self, fn: str, obj : np.ndarray = None, scf: bool = True):
        if fn is None:
            raise ValueError("SigGWCLoc.Save requires fn")
        fn_write = fn
        if scf:
            if self.iteration is None:
                raise ValueError("SigGWCLoc.Save requires iteration when scf=True")
            fn_write = f"{fn}.{self.iteration}.{self.key}"

        with h5py.File(self.hdf5file,'a') as file:
            siggwc = IO.Group(file, self.group, self.subgroup)
            if obj is not None:
                IO.CreateDataset(siggwc, fn_write, obj, dtype=complex)
            else:
                IO.CreateDataset(siggwc, fn_write, self.f, dtype=complex)

        return None


class SigCImp(FLocDyn):
    component = "sigimp"

    def __init__(self,crystal : Crystal,dlr : DLR,projector : Projector,key,sigma,sigma_hf : np.ndarray = None,subtract_static : bool = True,control=None,hdf5file : str = None,group : str = None,iteration: int = None,):

        super().__init__(crystal, dlr, projector)

        self.key = self.ResolveProblemKey(key)
        self.sigma_in = sigma
        self.sigma_hf_in = sigma_hf
        self.subtract_static = subtract_static
        self.control = control if control is not None else {}
        self.hf = None
        self.f = None
        self.t = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.iteration = iteration
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

        # Re-derive f_uniform from the DLR self-energy for single source of truth.
        if hasattr(self, "f_uniform") and self.f_uniform is not None:
            self.f_uniform = self.dlr.MatsubaraDLR2UniformGrid(self.f, sign=-1)

        self.t = self.F2T(self.f)

        return None

    def Mixing(self) -> None:
        self.f = super().Mixing(
            iter=self.iteration,
            mix=float(self.control["mix"]),
            component=self.component,
            value=self.f,
            method=self.control["mixing_method"],
            npulay=int(self.control["npulay"]),
            key=self.key,
        )
        if hasattr(self, "f_uniform") and self.f_uniform is not None:
            if hasattr(self.dlr, "MatsubaraDLR2UniformGrid"):
                self.f_uniform = self.dlr.MatsubaraDLR2UniformGrid(self.f, sign=-1)
        self.t = self.F2T(self.f)

    def Save(self, fn: str, obj : np.ndarray = None, scf: bool = True):
        if fn is None:
            raise ValueError("SigCImp.Save requires fn")
        fn_write = fn
        if scf:
            if self.iteration is None:
                raise ValueError("SigCImp.Save requires iteration when scf=True")
            fn_write = f"{fn}.{self.iteration}.{self.key}"

        with h5py.File(self.hdf5file,'a') as file:
            sigimp = IO.Group(file, self.group, self.subgroup)
            if obj is not None:
                IO.CreateDataset(sigimp, fn_write, obj, dtype=complex)
            else:
                IO.CreateDataset(sigimp, fn_write, self.f, dtype=complex)
                IO.CreateDataset(sigimp, fn_write + '_uniform', self.f_uniform, dtype=complex)

        return None


class Hyb(FLocDyn):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, key, green : np.ndarray, eimp : np.ndarray, sigh : np.ndarray = None, sigf : np.ndarray = None, sigc : np.ndarray = None, hdf5file : str = None, group : str = None, iteration: int = None):

        super().__init__(crystal, dlr, projector)

        print("Enter the Hyb class")
        self.key = self.ResolveProblemKey(key)
        self.green = green
        self.eimp = eimp
        self.sigh = sigh
        self.sigf = sigf
        self.sigc = sigc

        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.iteration = iteration

        print("Local Green's Function :", self.green[:, :, 0, 0])
        print("Impurity Level :", self.eimp[:, :, 0])
        print("Hartree Self-Energy :", self.sigh[:, :, 0] if self.sigh is not None else None)
        print("Fock Self-Energy :", self.sigf[:, :, 0] if self.sigf is not None else None)
        print("Correlated Self-Energy :", self.sigc[:, :, 0, 0] if self.sigc is not None else None)

        self.f = None
        self.t = None
        self.Cal()

        print("Exit the Hyb class")

    def Cal(self):

        tempmat = np.zeros_like(self.green, dtype=np.complex128, order='F')
        sig = np.zeros_like(self.green, dtype=np.complex128, order='F')
        if self.sigh is not None:
            sig += self.sigh[..., np.newaxis]
        if self.sigf is not None:
            sig += self.sigf[..., np.newaxis]
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
        tempmat_uniform = self.UniformGrid(self.f)
        try:
            # "auto" acceptance tolerance: the uniform->DLR interpolation noise
            # floor exceeds the strict 1e-8 default on clean data.
            self.f = self.CausalProjection(
                tempmat_uniform, grid="uniform", constraint_tol="auto"
            )
        except RuntimeError as err:
            # With elastic moments and the clipped last-resort fallback inside
            # the projector this branch should be unreachable; it survives as a
            # final safety net so an unexpected error cannot kill the run.
            warnings.warn(
                f"Hyb causal projection failed for key '{self.key}'; "
                f"using unprojected hybridization: {err}",
                RuntimeWarning,
            )
        self.t = self.F2T(self.f)
        print(f"[Hyb.Cal] key={self.key}, f[0,0,0,0]={self.f[0,0,0,0]}, f[0,0,0,-1]={self.f[0,0,0,-1]}")

        return None

    def Save(self, fn: str, obj : np.ndarray = None, scf: bool = True):
        if fn is None:
            raise ValueError("Hyb.Save requires fn")
        fn_write = fn
        if scf:
            if self.iteration is None:
                raise ValueError("Hyb.Save requires iteration when scf=True")
            fn_write = f"{fn}.{self.iteration}.{self.key}"

        with h5py.File(self.hdf5file,'a') as file:
            hyb = IO.Group(file, self.group, self.subgroup)
            if obj is not None:
                IO.CreateDataset(hyb, fn_write, obj, dtype=complex)
            else:
                IO.CreateDataset(hyb, fn_write, self.f, dtype=complex)

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
