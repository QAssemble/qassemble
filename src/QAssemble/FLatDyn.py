import numpy as np
import logging
import sys
import scipy.optimize
import scipy.linalg.lapack
import copy
import h5py
import time, datetime
from .Crystal import Crystal
from .FLatStc import FLatStc
from .Projector import Projector
from .utility.DLR import DLR
from .utility.Common import Common, timed_init
from .utility.HDF5 import IO
from .utility.Mixing import Mixing as MixingKernel
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
from .utility.Embedding import Embedding as EB
from .utility.Causal import CausalFermionProjector

logger = logging.getLogger("QAssemble")

class FLatDyn(object):
    mixer = MixingKernel()

    def __init__(self,crystal : Crystal, dlr : DLR, mixing_method: str = "pulay", npulay: int = 5) -> object:
        self.crystal = crystal
        self.dlr = dlr
        self.mappingidx = None
        self._fermion_phase_cache_k2r = self._get_fermion_phaseK2R()
        self._fermion_phase_cache_r2k = self._get_fermion_phaseR2K()

    def _get_fermion_phaseK2R(self) -> np.ndarray:
        
        nrk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]

        basis_orb = self.crystal.basisf[self.crystal.forb2atom]

        kv = self.crystal.kpoint[:nrk] @ basis_orb.T

        kv_delta = kv[:, :, None] - kv[:, None, :]

        phases = np.exp(2.0j * np.pi * kv_delta)

        phases_T = np.transpose(phases, (1, 2, 0))
        return phases_T
    
    def _get_fermion_phaseR2K(self) -> np.ndarray:
        
        nrk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]

        basis_orb = self.crystal.basisf[self.crystal.forb2atom]

        kv = self.crystal.kpoint[:nrk] @ basis_orb.T

        kv_delta = kv[:, :, None] - kv[:, None, :]

        phases = np.exp(-2.0j * np.pi * kv_delta)

        phases_T = np.transpose(phases, (1, 2, 0))
        return phases_T
        
    def Inverse(self, mat : np.ndarray) -> np.ndarray:

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]
        nft = mat.shape[4]

        matinv = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    matinv[:,:,js,irk,ift] = Common.MatInv(mat[:,:,js,irk,ift])
        # for js, irk, ift in itertools.product(list(range(ns)),list(range(nrk),list(range(nft)))):
        #     matinv[:,:,js,irk,ift] = np.linalg.inv(mat[:,:,js,irk,ift])
        
        return matinv

    
    def T2F(self,ftau : np.ndarray) -> np.ndarray:

        norb = ftau.shape[0]
        ns = ftau.shape[2]
        nk = ftau.shape[3]
        ntau = ftau.shape[4]

        ftau_t = np.moveaxis(ftau, -1, 0)  # (ntau, norb, norb, ns, nk)
        batch = norb * norb * ns * nk
        ftau_2d = np.ascontiguousarray(ftau_t).reshape(ntau, batch)

        fxx = self.dlr.dF.dlr_from_tau(ftau_2d)
        ff_2d = self.dlr.dF.matsubara_from_dlr(fxx, beta=self.dlr.beta, xi=-1)
        nfreq = ff_2d.shape[0]
        ff = ff_2d.reshape(nfreq, norb, norb, ns, nk)
        ff = np.moveaxis(ff, 0, -1)  # (norb, norb, ns, nk, nfreq)
        ff = np.asfortranarray(ff)

        return ff

    def F2T(self,ff : np.ndarray) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]
        nfreq = ff.shape[4]

        ff_t = np.moveaxis(ff, -1, 0)  # (nfreq, norb, norb, ns, nk)
        batch = norb * norb * ns * nk
        ff_2d = np.ascontiguousarray(ff_t).reshape(nfreq, batch)

        fxx = self.dlr.dF.dlr_from_matsubara(ff_2d, beta=self.dlr.beta, xi=-1)
        ftau_2d = self.dlr.dF.tau_from_dlr(fxx)
        ntau = ftau_2d.shape[0]
        ftau = ftau_2d.reshape(ntau, norb, norb, ns, nk)
        ftau = np.moveaxis(ftau, 0, -1)  # (norb, norb, ns, nk, ntau)
        ftau = np.asfortranarray(ftau)

        return ftau

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

    def _ExpandPositiveUniform(self, arr : np.ndarray) -> np.ndarray:
        """Expand a positive-only (iw_n >= 0) uniform fermion input to the full
        signed grid, leaving a full signed input untouched.

        ``arr`` is the 5D lattice array ``(norb, norb, ns, nk, nfreq)``.  When
        ``nfreq`` matches the positive-only grid ``MatsubaraFermionUniform()``,
        the negative half is reconstructed per-k via the Hermitian relation
        G(-iw)=G(iw)dagger using the transpose-correct
        :meth:`DLR.MatsubaraAddNegativeFrequency` (which takes the 4D
        ``(norb, norb, ns, nfreq)`` layout); the result then matches
        ``MatsubaraFermionUniformFull()`` so the downstream length check, tail
        fit, and DLR conversion proceed unchanged.  When ``nfreq`` already
        matches the full signed grid, the input is returned as-is.
        """
        nfreq_in = arr.shape[4]
        npos = len(self.dlr.MatsubaraFermionUniform())
        if nfreq_in != npos:
            return arr
        nk = arr.shape[3]
        out = None
        for ik in range(nk):
            expanded = self.dlr.MatsubaraAddNegativeFrequency(arr[:, :, :, ik, :])
            if out is None:
                out = np.zeros(
                    (arr.shape[0], arr.shape[1], arr.shape[2], nk, expanded.shape[3]),
                    dtype=np.complex128,
                    order="F",
                )
            out[:, :, :, ik, :] = expanded
        return out

    def CausalProjection(
        self,
        matin : np.ndarray,
        *,
        grid : str = "dlr",
        coefficient_sign : int = -1,
        solvers = None,
        max_iter : int = 100000,
        constraint_tol : float = 1.0e-8,
        fit_tol : float = 1.0e-6,
    ) -> np.ndarray:
        """Project diagonal lattice fermionic channels onto real pole-weight
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
        if arr.ndim != 5:
            raise ValueError(f"matin must be 5D, got {arr.ndim}D")
        norb = arr.shape[0]
        if arr.shape[1] != norb:
            raise ValueError("matin first two dimensions must be square")
        if arr.shape[2] != self.crystal.ns:
            raise ValueError(
                f"spin dimension mismatch: matin ns={arr.shape[2]}, crystal ns={self.crystal.ns}"
            )
        if arr.shape[3] != len(self.crystal.kpoint):
            raise ValueError(
                f"k dimension mismatch: matin nk={arr.shape[3]}, crystal nk={len(self.crystal.kpoint)}"
            )
        # Accept positive-only (iw_n >= 0) uniform input by expanding it to the
        # full signed grid via Hermitian symmetry before the length check, so a
        # caller can pass either the positive-only or the full signed grid.
        if grid == "uniform":
            arr = self._ExpandPositiveUniform(arr)
        if arr.shape[4] != nfreq:
            raise ValueError(
                f"frequency dimension mismatch: matin nf={arr.shape[4]}, grid nf={nfreq}"
            )
        if not np.all(np.isfinite(np.real(arr))) or not np.all(np.isfinite(np.imag(arr))):
            raise ValueError("matin contains non-finite values")

        nk = arr.shape[3]
        ns = self.crystal.ns
        # Estimate physical moments on the input grid before any interpolation,
        # then pass [high, moment...] explicitly to the projector.
        moment, high = self.Moment(arr, isgreen=False, highzero=False, grid=grid)

        # Uniform input: interpolate to the DLR grid (per-k, since
        # MatsubaraUniformGrid2DLR handles the 4D fermion case only).  Output is
        # on the DLR grid.
        if grid == "uniform":
            ndlr = len(self.dlr.omega)
            converted = np.zeros((norb, norb, ns, nk, ndlr), dtype=np.complex128)
            for ik in range(nk):
                converted[:, :, :, ik, :] = self.dlr.MatsubaraUniformGrid2DLR(
                    arr[:, :, :, ik, :], sign=-1
                )
            arr = converted

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
        out = np.array(arr, dtype=np.complex128, copy=True, order='F')
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    tail_coeffs = np.empty(4, dtype=float)
                    tail_coeffs[0] = float(np.real(high[iorb, iorb, js, ik]))
                    tail_coeffs[1:] = np.real(moment[iorb, iorb, js, ik, :])
                    out[iorb, iorb, js, ik, :] = projector.project(
                        arr[iorb, iorb, js, ik, :],
                        tail_coeffs=tail_coeffs,
                    )

        return np.asfortranarray(out)

    def CausalityCheck(
        self,
        matin : np.ndarray,
        *,
        grid : str = "dlr",
        coefficient_sign : int = -1,
        solvers = None,
        max_iter : int = 100000,
        constraint_tol : float = 1.0e-8,
        fit_tol : float = 1.0e-6,
    ) -> dict:
        """Diagnose causality of diagonal lattice channels without projecting.

        ``grid`` selects the input sampling grid (``"dlr"`` default or
        ``"uniform"``), as in ``CausalProjection``.

        Returns per-channel ``(norb, ns, nk)`` arrays: ``causal`` boolean,
        unscaled ``max_inequality_violation`` / ``max_equality_residual``,
        ``violating_count``, and ``node_residual``.  A channel whose
        ``node_residual`` exceeds ``fit_tol`` is not representable by the
        causal pole basis (data-quality problem); its sign verdict describes
        a garbage fit, so judge the residual first.  Never raises on bad
        data — this is the diagnostic counterpart of ``CausalProjection``.
        """

        omega = self._ResolveCausalGrid(grid)
        nfreq = len(omega)

        arr = np.asarray(matin, dtype=np.complex128)
        if arr.ndim != 5:
            raise ValueError(f"matin must be 5D, got {arr.ndim}D")
        norb = arr.shape[0]
        if arr.shape[1] != norb:
            raise ValueError("matin first two dimensions must be square")
        if arr.shape[2] != self.crystal.ns:
            raise ValueError(
                f"spin dimension mismatch: matin ns={arr.shape[2]}, crystal ns={self.crystal.ns}"
            )
        if arr.shape[3] != len(self.crystal.kpoint):
            raise ValueError(
                f"k dimension mismatch: matin nk={arr.shape[3]}, crystal nk={len(self.crystal.kpoint)}"
            )
        # Accept positive-only (iw_n >= 0) uniform input by expanding it to the
        # full signed grid via Hermitian symmetry before the length check, so a
        # caller can pass either the positive-only or the full signed grid.
        if grid == "uniform":
            arr = self._ExpandPositiveUniform(arr)
        if arr.shape[4] != nfreq:
            raise ValueError(
                f"frequency dimension mismatch: matin nf={arr.shape[4]}, grid nf={nfreq}"
            )
        if not np.all(np.isfinite(np.real(arr))) or not np.all(np.isfinite(np.imag(arr))):
            raise ValueError("matin contains non-finite values")

        nk = arr.shape[3]
        ns = self.crystal.ns
        # mirror CausalProjection: moments on native input grid, then DLR data.
        moment, high = self.Moment(arr, isgreen=False, highzero=False, grid=grid)
        if grid == "uniform":
            ndlr = len(self.dlr.omega)
            converted = np.zeros((norb, norb, ns, nk, ndlr), dtype=np.complex128)
            for ik in range(nk):
                converted[:, :, :, ik, :] = self.dlr.MatsubaraUniformGrid2DLR(
                    arr[:, :, :, ik, :], sign=-1
                )
            arr = converted

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
        causal = np.zeros((norb, ns, nk), dtype=bool)
        max_inequality = np.zeros((norb, ns, nk), dtype=float)
        max_equality = np.zeros((norb, ns, nk), dtype=float)
        violating_count = np.zeros((norb, ns, nk), dtype=int)
        node_residual = np.zeros((norb, ns, nk), dtype=float)
        c0 = np.zeros((norb, ns, nk), dtype=float)
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    tail_coeffs = np.empty(4, dtype=float)
                    tail_coeffs[0] = float(np.real(high[iorb, iorb, js, ik]))
                    tail_coeffs[1:] = np.real(moment[iorb, iorb, js, ik, :])
                    verdict = projector.check(
                        arr[iorb, iorb, js, ik, :],
                        enforce_gate=False,
                        tail_coeffs=tail_coeffs,
                    )
                    causal[iorb, js, ik] = verdict.causal
                    max_inequality[iorb, js, ik] = verdict.max_inequality_violation
                    max_equality[iorb, js, ik] = verdict.max_equality_residual
                    violating_count[iorb, js, ik] = verdict.violating_count
                    node_residual[iorb, js, ik] = verdict.node_residual
                    c0[iorb, js, ik] = verdict.c0

        return {
            "causal": causal,
            "max_inequality_violation": max_inequality,
            "max_equality_residual": max_equality,
            "violating_count": violating_count,
            "node_residual": node_residual,
            "c0": c0,
        }

    
    def Moment(
        self,
        ff : np.ndarray,
        isgreen : bool = False,
        highzero : bool = False,
        tail_points : int = 5,
        grid: str = "dlr",
    ) -> tuple:
        """High-frequency tail coefficients of a lattice fermionic function.

        For each orbital matrix element ``(iorb, jorb, js, ik)`` fit
        ``G(iw) ~ c0 + c1/(iw) + c2/(iw)^2 + c3/(iw)^3`` by a robust
        least squares over the ``tail_points`` largest ``|omega|`` points
        (``Fourier.FermionTailCoefficients``).  Returns
        ``moment[..., 0:3] = [c1, c2, c3]`` and ``high = c0`` in physical sign.
        ``grid`` selects the sampling grid of ``ff``; for uniform fermion input,
        positive-only data is expanded before fitting, matching
        ``CausalProjection``.

        All orbital matrix elements are fitted, then each moment matrix is
        Hermitian-symmetrized as in the original ``FLocDynM`` path.
        ``isgreen``/``highzero`` are accepted for backward compatibility but no
        longer alter the fit (the robust fit determines ``c0`` and all moments
        directly from the data).
        """
        arr = np.asarray(ff, dtype=np.complex128)
        if arr.ndim != 5:
            raise ValueError(f"ff must be 5D (norb,norb,ns,nk,nfreq), got {arr.ndim}D")
        if grid == "uniform":
            arr = self._ExpandPositiveUniform(arr)

        norb = arr.shape[0]
        ns = arr.shape[2]
        nk = arr.shape[3]

        moment = np.zeros((norb, norb, ns, nk, 3), dtype=np.complex128, order='F')
        high = np.zeros((norb, norb, ns, nk), dtype=np.complex128, order='F')

        omega = self._ResolveCausalGrid(grid)
        if arr.shape[4] != omega.size:
            raise ValueError(
                f"frequency dimension {arr.shape[4]} does not match {grid} omega size {omega.size}"
            )
        if arr.shape[4] < tail_points:
            raise ValueError(
                f"Need at least {tail_points} frequency points to build "
                "high-frequency moments."
            )
        idx = np.argsort(np.abs(omega))[-tail_points:]
        z = 1j * omega[idx]
        design = np.column_stack(
            [np.ones_like(z), 1.0 / z, 1.0 / z**2, 1.0 / z**3]
        )
        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        if iorb == jorb:
                            tail = Fourier.FermionTailCoefficients(
                                omega, arr[iorb, jorb, js, ik, :], tail_points
                            ).astype(np.complex128)
                        else:
                            tail, *_ = np.linalg.lstsq(
                                design, arr[iorb, jorb, js, ik, idx], rcond=None
                            )
                        high[iorb, jorb, js, ik] = tail[0]
                        moment[iorb, jorb, js, ik, :] = tail[1:]
                h = high[:, :, js, ik].copy()
                high[:, :, js, ik] = 0.5 * (h + h.T.conj())
                for imom in range(3):
                    m = moment[:, :, js, ik, imom].copy()
                    moment[:, :, js, ik, imom] = 0.5 * (m + m.T.conj())

        return moment, high
    
    
    def K2R(self,matk : np.ndarray, rkgrid : list = None) -> np.ndarray:

        rkvec = self.crystal.kpoint
        if rkgrid == None:
            rkgrid = self.crystal.rkgrid

        
        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[3]
        nft = matk.shape[4]

        # basis_orb = self.crystal.basisf[self.crystal.forb2atom]

        # kv = self.crystal.kpoint[:nrk] @ basis_orb.T

        # kv_delta = kv[:, :, None] - kv[:, None, :]

        # phases = np.exp(2.0j * np.pi * kv_delta)

        # phases_T = np.transpose(phases, (1, 2, 0))

        matr = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
        tempmat = np.empty((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
        # phase_view = phases[:, :, np.newaxis, :]
        tempmat = matk.copy()

        tempmat *= self._fermion_phase_cache_k2r[:, :, None, :, None]

        matr = Fourier.FLatDynK2R(tempmat, rkgrid)

        return matr
    
    def R2K(self, matr : np.ndarray) -> np.ndarray:

        rkgrid = self.crystal.rkgrid

        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[3]
        nft = matr.shape[4]

        
        
        matk = np.zeros((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')
        tempmat = np.empty((norb, norb, ns, nrk, nft), dtype=np.complex128, order='F')

        tempmat = Fourier.FLatDynR2K(matr, rkgrid)

        matk = tempmat * self._fermion_phase_cache_r2k[:, :, None, :, None]

        return matk
    
    
    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nrk = y.shape[3]
        nft = y.shape[4]

        ynew = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        w0 = (1.0 - 3.0*w1)*np.pi*temperature
        widtharray = w0+w1*x
        cnt = 0
        for irk in range(nrk):
            for x0 in x:
                if (x0>cutoff+(w0+w1*cutoff)*3.0):
                    ynew[...,irk,cnt] = y[...,irk,cnt]
                else : 
                    if ((x0>3*widtharray[cnt])and((x[-1]-x0)>3*widtharray[cnt])):
                        dist = 1.0/np.sqrt(2*np.pi)/widtharray[cnt]*np.exp(-(x-x0)**2/2.0/widtharray[cnt]**2)
                        for js in range(ns):
                            for iorb in range(norb):
                                for jorb in range(norb):
                                    ynew[iorb,jorb,js,irk,cnt] = sum(dist*y[iorb,jorb,js,irk])/sum(dist)
                    else:
                        ynew[...,irk,cnt] = y[...,irk,cnt]
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
            key = "global"
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
    
    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        # matout = QAFort.dyson.flatdyn(mat1,mat2)
        return Dyson.FLatDyn(mat1, mat2)
    
    def ChemEmbedding(self,mu : np.float64) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.dlr.omega)#self.ft.size

        chem = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for iorb in range(norb):
                        chem[iorb, iorb, js, irk, ift] = mu

        return chem
    
    def StcEmbedding(self, matin : np.ndarray) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.dlr.omega)#self.ft.size

        matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ift in range(nft):
            matout[...,ift] = matin

        return matout
    
    
    def CheckGroup(self, filepath :str, group : str):
        
        with h5py.File(filepath,'r') as file:
            return group in file
        
    
    def AnalyticContinuation(self, mat_freq : np.ndarray, wreal : np.ndarray, eta : float = 0.05) -> tuple:
        """Analytically continue a Matsubara/DLR quantity to the real axis via DLR.

        Uses the DLR spectral representation
            rho_DLR(w) = -sum_k g_k delta(w - w_k)
        so that the retarded function is the pole sum
            G^R(w) = sum_k g_k / (w + i*eta - w_k).
        This is exact (no ill-posed fit) for *noiseless* DLR-represented
        quantities such as a Dyson-built G or a DLR self-energy. It is NOT a
        substitute for MaxEnt/Nevanlinna on noisy (QMC) data: the continuation
        amplifies error for poles poorly captured by the sparse DLR Matsubara
        nodes and is unstable under statistical noise.

        Convention (verified against the single-pole G(iv)=1/(iv-e0)): pydlr's
        eval_dlr_freq builds the kernel 1/(z + w_x/beta) and conjugates the
        result, so the physically correct retarded continuation is obtained
        with z = -w + i*eta, giving A(w) = -Im G^R(w)/pi >= 0 peaked at the
        right energy.

        Parameters
        ----------
        mat_freq : (norb, norb, ns, nk, nfreq_dlr) complex
            Quantity on the DLR Matsubara grid (self.dlr.omega).
        wreal : (nw,) float
            Real-frequency grid.
        eta : float
            Lorentzian broadening (imaginary shift). Default 0.05.

        Returns
        -------
        gret : (norb, norb, ns, nk, nw) complex
            Retarded function G^R(w).
        akf : (norb, norb, ns, nk, nw) float
            Spectral function -Im G^R(w)/pi.
        """
        mat_freq = np.asarray(mat_freq, dtype=np.complex128)
        if mat_freq.ndim != 5:
            raise ValueError(f"AnalyticContinuation expects 5D input, got {mat_freq.ndim}D")
        nfreq_dlr = mat_freq.shape[4]
        if nfreq_dlr != len(self.dlr.omega):
            raise ValueError(
                f"frequency dimension mismatch: mat_freq nfreq={nfreq_dlr}, "
                f"dlr nfreq={len(self.dlr.omega)}"
            )

        norb, _, ns, nk, _ = mat_freq.shape
        wreal = np.asarray(wreal, dtype=np.float64)
        nw = wreal.shape[0]

        dF = self.dlr.dF
        beta = self.dlr.beta

        # Move DLR axis to front and flatten orbital/spin/k into a single batch,
        # matching the reshape idiom used by T2F/F2T.
        mat_t = np.moveaxis(mat_freq, -1, 0)  # (nfreq_dlr, norb, norb, ns, nk)
        batch = norb * norb * ns * nk
        mat_2d = np.ascontiguousarray(mat_t).reshape(nfreq_dlr, batch)

        coeffs = dF.dlr_from_matsubara(mat_2d, beta, xi=-1)  # (nfreq_dlr, batch)

        # Retarded continuation: z = -w + i*eta (see docstring for the sign).
        # z = -wreal + 1j * eta
        z= wreal + 1j * eta
        # eval_dlr_freq expects rank-3 coeffs (n, m, m); treat the batch as the
        # second "orbital" axis so a single call covers every column.
        gret_3d = dF.eval_dlr_freq(coeffs[:, None, :], z, beta, xi=-1)  # (nw, batch, 1)
        gret_2d = gret_3d.reshape(nw, batch)

        gret = gret_2d.reshape(nw, norb, norb, ns, nk)
        gret = np.moveaxis(gret, 0, -1)  # (norb, norb, ns, nk, nw)
        gret = np.asfortranarray(gret)

        akf = np.asfortranarray(-gret.imag / np.pi)

        return gret, akf

    def Spectral(self, green : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        nfreq = len(self.dlr.omega)

        akf = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,order='F')

        akf = -1/np.pi*green.imag

        return akf
    
    def R2KArb(self,matr : np.ndarray = None,kpoint : np.ndarray = None): # R2KAny

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nr = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        nk = len(kpoint)
        nft = matr.shape[4]

        self.crystal.RVec()
        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb,norb,ns,nk,nft),dtype=complex,order='F')

        for ift in range(nft):
            for ik in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            temp = 0
                            for ir in range(nr):
                                temp += tempmat[iorb,jorb,js,ir,ift]*np.exp(-2.0j*np.pi*(kpoint[ik]@self.crystal.rvec[ir]))
                            [a,m1] = self.crystal.FAtomOrb(iorb)
                            [b,m2] = self.crystal.FAtomOrb(jorb)
                            delta = self.crystal.basisf[a,:]-self.crystal.basisf[b,:]
                            phase = np.exp(-2.0j*np.pi*(kpoint[ik]@delta))
                            matk[iorb,jorb,js,ik,ift] = temp*phase
        
        return matk

    def KArb(self, matr : np.ndarray = None, kpoint : np.ndarray = None):

        norb = matr.shape[0]
        ns = matr.shape[2]
        nr = matr.shape[3]
        nfreq = matr.shape[4]
        nk = len(kpoint)

        tempmat = np.zeros((norb,norb,ns,nr,nfreq),dtype=complex,order='F')
        matkinv = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,order='F')

        matrinv = self.Inverse(matr)
        omega = self.dlr.omega

        for ifreq in range(nfreq):
            for ir in range(nr):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            if iorb==jorb:
                                tempmat[iorb,jorb,js,ir,ifreq] = 1j*omega[ifreq]-matrinv[iorb,jorb,js,ir,ifreq]
                            else:
                                tempmat[iorb,jorb,js,ir,ifreq] = -matrinv[iorb,jorb,js,ir,ifreq]

        tempmat2 = self.R2KArb(tempmat,kpoint)

        for ifreq in range(nfreq):
            for ik in range(nk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            if iorb==jorb:
                                matkinv[iorb,jorb,js,ik,ifreq] = 1j*omega[ifreq]-tempmat2[iorb,jorb,js,ik,ifreq]
                            else:
                                matkinv[iorb,jorb,js,ik,ifreq] = -tempmat2[iorb,jorb,js,ik,ifreq]
        
        matk = self.Inverse(matkinv)

        return matk
    
    def R2mR(self, matin : np.ndarray) -> np.ndarray:

        mappingidx = Common.R2mRMapping(self.crystal.kpoint)

        matout = np.zeros_like(matin, dtype=np.complex128, order='F')

        for rp in mappingidx:
            matout[..., rp[0],:] = matin[..., rp[1], :]

        return matout
    
    def T2mT(self, ftau : np.ndarray) -> np.ndarray:

        taum = self.dlr.beta - self.dlr.tauF

        norb, _, ns, nrk, ntau = ftau.shape

        fout = np.zeros((norb, norb, ns, nrk, ntau), dtype=np.complex128, order='F')

        for irk in range(nrk):
            for js in range(ns):
                block = ftau[:, :, js, irk, :].T  # (ntau, norb, norb)
                ntau_b = block.shape[0]
                block_2d = block.reshape(ntau_b, -1)

                fxx = self.dlr.dF.dlr_from_tau(block_2d)
                fout_3d = self.dlr.dF.eval_dlr_tau(fxx[:, :, None], taum, self.dlr.beta)
                fout[:, :, js, irk, :] = fout_3d[:, :, 0].reshape(-1, norb, norb).T

        return fout

    def TauB2TauF(self, ftau : np.ndarray) -> np.ndarray:

        norb, _, ns, ns2, nk, ntauB = ftau.shape
        ftau_t = np.moveaxis(ftau, -1, 0)  # (ntauB, norb, norb, ns, ns2, nk)
        batch = norb * norb * ns * ns2 * nk
        ftau_2d = np.ascontiguousarray(ftau_t).reshape(ntauB, batch)

        fxx = self.dlr.dB.dlr_from_tau(ftau_2d)
        fout_3d = self.dlr.dB.eval_dlr_tau(fxx[:, :, None], self.dlr.tauF, self.dlr.beta)
        ntauF = len(self.dlr.tauF)
        fout = fout_3d[:, :, 0].reshape(ntauF, norb, norb, ns, ns2, nk)
        fout = np.moveaxis(fout, 0, -1)  # (norb, norb, ns, ns2, nk, ntauF)
        fout = np.asfortranarray(fout)

        return fout
    
    def Diagonalize(self, matk : np.ndarray):

        norb, _, ns, nk, nfreq = matk.shape

        eigval = np.zeros((norb, norb, ns, nk, nfreq), dtype=float)
        eigvec = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128)

        for ifreq in range(nfreq):
                for ik in range(nk):
                    for js in range(ns):
                        e, v, info = scipy.linalg.lapack.zheev(matk[:, :, js, ik, ifreq])
                        eigval[:, :, js, ik, ifreq] = np.diag(e)
                        eigvec[:, :, js, ik, ifreq] = v

        return eigval, eigvec
    
    # def Embedding(self, matin : np.ndarray):

    #     norb = len(self.crystal.find)
    #     ns = self.crystal.ns
    #     nrk = len(self.crystal.kpoint)
    #     nft = self.ft.size
    #     nspace = self.crystal.fprojector.shape[3]

    #     matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

    #     for ispace in range(nspace):
    #         matout += QAFort.embedding.flocdyn(nrk,matin[...,ispace],self.crystal.fprojector[...,ispace])

    #     return matout

    def Embedding(self, matin: np.ndarray, projector: Projector, key) -> np.ndarray:
        if projector is None:
            raise ValueError("projector is required for Embedding")

        nrk = len(self.crystal.kpoint)
        pkey = key if key in projector.fprojector else str(key)
        if pkey not in projector.fprojector:
            raise KeyError(f"Unknown impurity problem key '{key}'")

        matin = np.asarray(matin, dtype=np.complex128)
        if matin.ndim != 4:
            raise ValueError(f"matin must be 4D, got {matin.ndim}D")
        if matin.shape[2] != self.crystal.ns:
            raise ValueError(
                f"spin dimension mismatch: matin ns={matin.shape[2]}, crystal ns={self.crystal.ns}"
            )
        if matin.shape[3] != len(self.dlr.omega):
            raise ValueError(
                f"frequency dimension mismatch: matin nf={matin.shape[3]}, dlr nf={len(self.dlr.omega)}"
            )

        proj = projector.fprojector[pkey]
        rep_emb = EB.FLatDyn(matin, proj, nrk)
        expanded = np.zeros_like(rep_emb, dtype=np.complex128, order="F")
        rep_orbs = projector.fimpdict[pkey][0]

        for tgt_orbs in projector.fimpdict[pkey]:
            if len(tgt_orbs) != len(rep_orbs):
                raise ValueError(
                    f"Equivalent spaces in key '{pkey}' have different orbital counts"
                )

            expanded[np.ix_(tgt_orbs, tgt_orbs)] = rep_emb[np.ix_(rep_orbs, rep_orbs)]

        return expanded


@timed_init
class G0(FLatDyn):

    def __init__(self, crystal: Crystal, dlr : DLR, hamtb : np.ndarray = None, hdf5file : str = None, group : str = None) -> object:
        
        super().__init__(crystal, dlr)
        # print(self.niham.hamtb[...,0,0])
        self.hamtb = hamtb
        self.kt = None
        self.kf = None
        self.rt = None
        self.rf = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        logger.info("Bare Green's function Calculation Start")
        start = time.time()
        self.Cal()
        if hdf5file != None:
            self.Save()
        end = time.time()
        logger.info("Bare Green's function Calculation Finish")
        logger.info(f"Calculation Time : {str(datetime.timedelta(seconds=end-start))}")
        

    def Cal(self): # freq, tau combine
        
        from .utility.Bare import Bare
        # print(self.hamtb[:,:,0,0])
        # gnotkf = QAFort.bare.flatfreq(self.hamtb,self.dlr.omega)
        gnotkf = Bare.FLatFreq(self.dlr.omega, self.hamtb)
        gnotrf = self.K2R(gnotkf)#######
        
        self.kf = gnotkf
        self.rf = gnotrf

        # gnotkt = QAFort.bare.flattau(self.hamtb,self.dlr.tau)
        gnotkt = Bare.FLatTau(tau=self.dlr.tauF, beta=self.dlr.beta, hlatt=self.hamtb)
        gnotrt = self.K2R(gnotkt)

        self.kt = gnotkt
        self.rt = gnotrt

        return None
    
    def Save(self):

        # if os.path.exists('gbare'):
        #     pass
        # else:
        #     os.mkdir('gbare')

        with h5py.File(self.hdf5file,'a') as file:
            if self.CheckGroup(self.hdf5file,self.group):
                group = file[self.group]
                if self.subgroup in group:
                    gbare = group[self.subgroup]
                else:
                    gbare = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                gbare = group.create_group(self.subgroup)
            IO.CreateDataset(gbare, 'g0kf', self.kf, dtype=complex)

        return None
    
@timed_init
class G(FLatDyn):

    def __init__(
        self,
        crystal: Crystal,
        dlr : DLR,
        greenbare : np.ndarray = None,
        sigmah : np.ndarray = None,
        sigmaf : np.ndarray = None,
        sigmagwc : np.ndarray = None,
        hdf5file : str = 'glob.h5',
        group : str = None,
        iteration: int = None,
        mu_reference: float = None,
        mu_search_mode: str = "reference_nearest",
        mu_search_ecut: float = 10.0,
        mu_search_scan_points: int = 41,
        mu_search_max_iter: int = 1000,
        mu_search_density_tol: float = 1.0e-7,
    ) -> object:
        
        if greenbare is None:
            logger.error("Bare Green's function doesn't exist")
            sys.exit()
        super().__init__(crystal, dlr)
        self.flatstc = FLatStc(crystal=crystal)
        norb, _, ns, nk, nfreq = greenbare.shape
        ntau = len(self.dlr.tauF)
        self.kf = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.kt = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.rf = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.rt = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.gkfmu0 = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.gktmu0 = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.grfmu0 = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.grtmu0 = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.gbare = greenbare
        self.sigmah = sigmah
        self.sigmaf = sigmaf
        self.sigmac = sigmagwc
        self.occ = None
        self.occk = None
        self.occr = None
        self.mu = np.float64(0.0)
        self.c = np.float64(0.0)
        # tau_uniform = self.dlr.TauUniform()
        # self._tau_beta = tau_uniform[-1]
        self._tau_beta = self.dlr.tauF[-1]
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.iteration = iteration
        self.mu_reference = self._coerce_optional_float(mu_reference)
        self.mu_search_mode = str(mu_search_mode).lower()
        self.mu_search_ecut = float(mu_search_ecut)
        self.mu_search_scan_points = int(mu_search_scan_points)
        self.mu_search_max_iter = int(mu_search_max_iter)
        self.mu_search_density_tol = float(mu_search_density_tol)
        self.mu_search_diagnostics = {}
        self._validate_mu_search_options()
        
        logger.info("Interacting Green's function Calculation Start")
        start = time.time()
        self.CalMu0()

        self.SearchMu()
        end = time.time()
        logger.info(f"chemical potential : {self.mu}")
        logger.debug(self.occ)
        logger.info("Interacting Green's function Calculation Finish")
        logger.info(f"Calculation Time : {str(datetime.timedelta(seconds=end-start))}")

    def CalMu0(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nomega = len(self.dlr.omega)
        sigma = np.zeros((norb,norb,ns,nrk,nomega),dtype=np.complex128,order='F')
        logger.info("Initialization start")
        if (self.sigmah is None)and(self.sigmaf is None)and(self.sigmac is None):
            self.gkfmu0 = self.gbare
        else:
            if (self.sigmah is not None):
                sigma += self.StcEmbedding(self.sigmah)
                logger.info('Hartree')
                logger.debug(sigma[:,:,0,0,0])
            if (self.sigmaf is not None):
                # print(sigma[:,:,0,0,0])
                sigma += self.StcEmbedding(self.sigmaf)
                logger.info('Fock')
                logger.debug(sigma[:,:,0,0,0])
            if (self.sigmac is not None):
                # print(sigma[:,:,0,0,0])
                sigma += self.sigmac
                logger.info('GWC')
                logger.debug(sigma[:,:,0,0,0])
            self.gkfmu0 = self.Dyson(self.gbare,sigma)


        self.gktmu0 = self.F2T(self.gkfmu0)
        self.grfmu0 = self.K2R(self.gkfmu0)
        self.grtmu0 = self.K2R(self.gktmu0)
        logger.debug(f"[G.CalMu0] c={self.c}, gkfmu0[0,0,0,0,-1]={self.gkfmu0[0,0,0,0,-1]}")
        logger.info("Initialization finish")
        return None
    
    def Occ(self):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        
        
        occk = np.zeros((norb,norb,ns,nrk),dtype=np.complex128,order='F')
        occ = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')
        
        logger.info("Density matrixy calculation start")
        # kt = np.copy(self.kt)
        # ntau = 5000
        tau_beta = np.array([self._tau_beta], dtype=np.float64)

        for irk in range(nrk):
            for js in range(ns):

                block = self.kt[:, :, js, irk, :].T  # (ntau, norb, norb)
                ntau_b = block.shape[0]
                block_2d = block.reshape(ntau_b, -1)

                fxx = self.dlr.dF.dlr_from_tau(block_2d)
                fout = self.dlr.dF.eval_dlr_tau(fxx[:, :, None], tau_beta, beta=self.dlr.beta)

                occk[:, :, js, irk] = -fout[0, :, 0].reshape(norb, norb)


        
            
        occ = occk.sum(axis=3)/nrk
        self.occ = occ
        self.occk = occk
        
        self.occr = self.flatstc.K2R(occk)
        logger.info("Density matrixy calculation finish")
        return None
    
    def UpdateMu(self) -> np.ndarray:

        logger.info("Chemical potential shift start")
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = len(self.dlr.omega)

        gkfnew = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')
        chem = self.ChemEmbedding(self.mu)
    
    
        gkfnew = self.Dyson(self.gkfmu0,-chem)
    
        
        self.kf = gkfnew
        self.kt = self.F2T(gkfnew)
        # self.grf = self.K2R(self.Dyson(self.gkfmu0,-chem))
        # self.grt = self.K2R(self.F2T(self.Dyson(self.gkfmu0,-chem),1,1))
        self.rf = self.K2R(self.kf)
        self.rt = self.K2R(self.kt)
        logger.info("Chemical potential shift finish")
        self.Occ()

        return None
    
    def NumOfE(self, mu : np.float64):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nfreq = len(self.dlr.omega)

        # Use cached G0inv: G(mu) = (G0inv + mu*I)^{-1}
        mat = self._g0inv_cache.copy()
        diag = np.arange(norb)
        mat[diag, diag, :, :, :] += mu

        # Batch invert: reshape (norb,norb,ns,nk,nfreq) -> (ns*nk*nfreq, norb,norb)
        mat_batch = np.moveaxis(mat, (0, 1), (-2, -1))  # (ns,nk,nfreq,norb,norb)
        orig_shape = mat_batch.shape[:-2]
        mat_flat = mat_batch.reshape(-1, norb, norb)
        gcalf_flat = np.linalg.inv(mat_flat)
        gcalf_batch = gcalf_flat.reshape(orig_shape + (norb, norb))
        gcalf = np.moveaxis(gcalf_batch, (-2, -1), (0, 1))  # (norb,norb,ns,nk,nfreq)

        # Extract diagonal elements only for DLR: shape (norb, ns, nk, nfreq)
        gdiag = gcalf[diag, diag, :, :, :]  # (norb, ns, nk, nfreq)

        # Batch DLR: Matsubara -> single tau point
        gdiag_perm = np.ascontiguousarray(np.moveaxis(gdiag, -1, 0))  # (nfreq, norb, ns, nk)
        batch_shape = gdiag_perm.shape[1:]
        gdiag_flat = gdiag_perm.reshape(nfreq, -1)
        fxx = self.dlr.dF.dlr_from_matsubara(gdiag_flat, beta=self.dlr.beta, xi=-1)
        fout = self.dlr.dF.eval_dlr_tau(fxx[:, :, None], self._tau_beta_cache, beta=self.dlr.beta)
        gtau_beta = fout[0, :, 0].reshape(batch_shape)  # (norb, ns, nk)

        Ne = -np.real(gtau_beta.sum()) / nrk

        return (self.crystal.nume - Ne)

    @staticmethod
    def _coerce_optional_float(value):
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value

    def _validate_mu_search_options(self):
        allowed = {"global", "reference_bisect", "reference_nearest"}
        if self.mu_search_mode not in allowed:
            raise ValueError(
                f"Unknown MuSearchMode={self.mu_search_mode!r}; "
                f"expected one of {sorted(allowed)}"
            )
        if not np.isfinite(self.mu_search_ecut) or self.mu_search_ecut <= 0.0:
            raise ValueError("MuSearchEcut/Ecut must be positive and finite")
        if self.mu_search_scan_points < 3:
            raise ValueError("MuSearchScanPoints must be at least 3")
        if self.mu_search_max_iter < 1:
            raise ValueError("MuSearchMaxIter must be positive")
        if not np.isfinite(self.mu_search_density_tol) or self.mu_search_density_tol <= 0.0:
            raise ValueError("MuSearchDensityTol must be positive and finite")

    def _load_mu_reference_from_hdf5(self):
        if self.hdf5file is None or self.group is None or self.iteration is None:
            return None
        if int(self.iteration) <= 1:
            return None
        try:
            with h5py.File(self.hdf5file, "r") as file:
                path = f"{self.group}/{self.subgroup}/mu.{int(self.iteration) - 1}"
                if path in file:
                    return float(file[path][()])
        except OSError:
            return None
        return None

    def _prepare_mu_search_cache(self):
        norb = len(self.crystal.find)
        g0 = self.gkfmu0  # (norb, norb, ns, nk, nfreq)
        g0_batch = np.moveaxis(g0, (0, 1), (-2, -1))  # (..., norb, norb)
        orig_shape = g0_batch.shape[:-2]
        g0_flat = g0_batch.reshape(-1, norb, norb)
        g0inv_flat = np.linalg.inv(g0_flat)
        g0inv_batch = g0inv_flat.reshape(orig_shape + (norb, norb))
        self._g0inv_cache = np.moveaxis(g0inv_batch, (-2, -1), (0, 1))  # (norb, norb, ns, nk, nfreq)
        self._tau_beta_cache = np.array([self._tau_beta], dtype=np.float64)

    def _global_mu_bracket(self):
        # Keep the legacy global bracket as the reproducibility fallback.
        shift_est = 0.0
        shift_spread = 0.0
        if self.sigmah is not None:
            diag = np.real(np.diagonal(self.sigmah[:, :, 0, 0]))
            shift_est = float(np.mean(diag))
            shift_spread = float(np.max(np.abs(diag - shift_est)))
        safety = 1.5
        mumin = self.dlr.omega[0] + shift_est - safety * shift_spread
        mumax = self.dlr.omega[-1] + shift_est + safety * shift_spread
        return float(mumin), float(mumax), shift_est, shift_spread

    def _solve_global_mu(self):
        mumin, mumax, shift_est, shift_spread = self._global_mu_bracket()
        logger.info(
            f"global mu search range : {mumin}, {mumax} "
            f"(shift_est={shift_est}, spread={shift_spread})"
        )

        nmin = self.NumOfE(mumin)
        nmax = self.NumOfE(mumax)
        expand_tries = 0
        while ((nmin < 0) or (nmax > 0)) and expand_tries < 3:
            width = mumax - mumin
            mumin -= 0.5 * width
            mumax += 0.5 * width
            nmin = self.NumOfE(mumin)
            nmax = self.NumOfE(mumax)
            expand_tries += 1
            logger.info(
                f"expand {expand_tries}: [{mumin}, {mumax}], nmin={nmin}, nmax={nmax}"
            )
        if (nmin < 0) or (nmax > 0):
            logger.error("Chemical potential is out of the bisection range after expansion")
            logger.error(f"nmin : {nmin}, nmax : {nmax}")
            sys.exit()

        sol = scipy.optimize.brentq(self.NumOfE, mumin, mumax, xtol=1.0e-6)
        return float(sol), {
            "search_mode": "global",
            "bracket_min": float(mumin),
            "bracket_max": float(mumax),
            "global_expand_tries": int(expand_tries),
            "used_global_fallback": False,
        }

    @staticmethod
    def _solve_reference_bisect(
        func,
        mu_reference,
        ecut,
        density_tol=1.0e-7,
        max_iter=1000,
    ):
        lower_delta = -float(ecut)
        upper_delta = float(ecut)
        delta_mu = 0.0
        residual = float(func(float(mu_reference) + delta_mu))
        iterations = 0

        for iterations in range(1, int(max_iter) + 1):
            residual = float(func(float(mu_reference) + delta_mu))
            if abs(residual) < float(density_tol):
                break
            if residual <= 0.0:
                upper_delta = delta_mu
            else:
                lower_delta = delta_mu
            delta_mu = 0.5 * (lower_delta + upper_delta)

        sol = float(mu_reference) + float(delta_mu)
        residual = float(func(sol))
        converged = abs(residual) < float(density_tol)
        return sol, {
            "search_mode": "reference_bisect",
            "mu_reference": float(mu_reference),
            "Ecut": float(ecut),
            "delta_mu": float(delta_mu),
            "bracket_min": float(mu_reference) - float(ecut),
            "bracket_max": float(mu_reference) + float(ecut),
            "reference_iterations": int(iterations),
            "reference_residual": float(residual),
            "used_global_fallback": False,
            "local_root_found": bool(converged),
            "local_root_count": int(converged),
        }

    @staticmethod
    def _solve_reference_nearest(
        func,
        mu_reference,
        ecut,
        xtol=1.0e-6,
        density_tol=1.0e-7,
        scan_points=41,
    ):
        mu_reference = float(mu_reference)
        ecut = float(ecut)
        grid = np.linspace(mu_reference - ecut, mu_reference + ecut, int(scan_points))
        values = np.asarray([float(func(mu)) for mu in grid], dtype=float)
        candidates = []

        for mu, value in zip(grid, values):
            if np.isfinite(value) and abs(value) < float(density_tol):
                candidates.append(float(mu))

        for idx in range(len(grid) - 1):
            f_left = values[idx]
            f_right = values[idx + 1]
            if not (np.isfinite(f_left) and np.isfinite(f_right)):
                continue
            if f_left == 0.0 or f_right == 0.0:
                continue
            if np.sign(f_left) == np.sign(f_right):
                continue
            root = scipy.optimize.brentq(
                func,
                float(grid[idx]),
                float(grid[idx + 1]),
                xtol=float(xtol),
            )
            candidates.append(float(root))

        unique = []
        for root in candidates:
            if not any(abs(root - old) <= float(xtol) for old in unique):
                unique.append(root)

        if not unique:
            return None, {
                "search_mode": "reference_nearest",
                "mu_reference": mu_reference,
                "Ecut": ecut,
                "bracket_min": mu_reference - ecut,
                "bracket_max": mu_reference + ecut,
                "used_global_fallback": True,
                "local_root_found": False,
                "local_root_count": 0,
            }

        def sort_key(root):
            return (abs(root - mu_reference), abs(float(func(root))))

        sol = float(min(unique, key=sort_key))
        return sol, {
            "search_mode": "reference_nearest",
            "mu_reference": mu_reference,
            "Ecut": ecut,
            "delta_mu": sol - mu_reference,
            "bracket_min": mu_reference - ecut,
            "bracket_max": mu_reference + ecut,
            "used_global_fallback": False,
            "local_root_found": True,
            "local_root_count": int(len(unique)),
            "scan_points": int(scan_points),
        }

    def SearchMu(self):

        logger.info("Finding chemical potential start")

        self._prepare_mu_search_cache()
        try:
            reference = self.mu_reference
            if reference is None:
                reference = self._load_mu_reference_from_hdf5()
                self.mu_reference = reference

            diagnostics = {}
            if self.mu_search_mode == "global" or reference is None:
                if self.mu_search_mode != "global":
                    logger.info("No mu reference available; falling back to global mu search")
                sol, diagnostics = self._solve_global_mu()
                diagnostics["used_global_fallback"] = self.mu_search_mode != "global"
                diagnostics.setdefault("mu_reference", np.nan)
                diagnostics.setdefault("Ecut", float(self.mu_search_ecut))
                diagnostics.setdefault("delta_mu", np.nan)
                diagnostics.setdefault("local_root_found", False)
                diagnostics.setdefault("local_root_count", 0)
            elif self.mu_search_mode == "reference_bisect":
                sol, diagnostics = self._solve_reference_bisect(
                    self.NumOfE,
                    reference,
                    self.mu_search_ecut,
                    density_tol=self.mu_search_density_tol,
                    max_iter=self.mu_search_max_iter,
                )
            else:
                sol, diagnostics = self._solve_reference_nearest(
                    self.NumOfE,
                    reference,
                    self.mu_search_ecut,
                    density_tol=self.mu_search_density_tol,
                    scan_points=self.mu_search_scan_points,
                )
                if sol is None:
                    logger.warning(
                        "No local mu root found in "
                        f"[{reference - self.mu_search_ecut}, {reference + self.mu_search_ecut}]; "
                        "falling back to global mu search"
                    )
                    sol, global_diag = self._solve_global_mu()
                    diagnostics.update(global_diag)
                    diagnostics["search_mode"] = "reference_nearest"
                    diagnostics["used_global_fallback"] = True

            final_residual = float(self.NumOfE(sol))
            diagnostics["mu_final"] = float(sol)
            diagnostics["delta_mu"] = float(sol - reference) if reference is not None else np.nan
            diagnostics["occupation_at_mu"] = float(self.crystal.nume - final_residual)
            diagnostics["residual_at_mu"] = final_residual
            self.mu_search_diagnostics = diagnostics
            self.mu = float(sol)  # physical chemical potential
        finally:
            if hasattr(self, "_g0inv_cache"):
                del self._g0inv_cache
            if hasattr(self, "_tau_beta_cache"):
                del self._tau_beta_cache

        logger.info(
            "Finding chemical potential finish: "
            f"mode={self.mu_search_diagnostics.get('search_mode')}, "
            f"mu_ref={self.mu_search_diagnostics.get('mu_reference')}, "
            f"Ecut={self.mu_search_diagnostics.get('Ecut')}, "
            f"delta_mu={self.mu_search_diagnostics.get('delta_mu')}, "
            f"mu={self.mu}, "
            f"occupation={self.mu_search_diagnostics.get('occupation_at_mu')}, "
            f"fallback={self.mu_search_diagnostics.get('used_global_fallback')}"
        )

        self.UpdateMu()
        return None
    
    def Save(self, fn: str, scf: bool = True):
        if fn is None:
            raise ValueError("G.Save requires fn")
        fn_write = fn
        mu_write = "mu"
        if scf:
            if self.iteration is None:
                raise ValueError("G.Save requires iteration when scf=True")
            fn_write = f"{fn}.{self.iteration}"
            mu_write = f"mu.{self.iteration}"

        
        with h5py.File(self.hdf5file,'a') as file:
            if self.CheckGroup(self.hdf5file,self.group):
                group = file[self.group]
                if self.subgroup in group:
                    green = group[self.subgroup]
                else:
                    green = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                green = group.create_group(self.subgroup)
            IO.CreateDataset(green, fn_write, self.kf, dtype=complex)
            
            
            # mureal = np.real(self.mu)
            IO.CreateDataset(green, mu_write, self.mu, dtype=float)
            diag = getattr(self, "mu_search_diagnostics", {})
            suffix = f".{self.iteration}" if scf else ""
            scalar_diagnostics = {
                "mu_reference": diag.get("mu_reference", np.nan),
                "mu_ecut": diag.get("Ecut", np.nan),
                "delta_mu": diag.get("delta_mu", np.nan),
                "occupation_at_mu": diag.get("occupation_at_mu", np.nan),
                "residual_at_mu": diag.get("residual_at_mu", np.nan),
                "used_global_fallback": int(bool(diag.get("used_global_fallback", False))),
                "local_root_found": int(bool(diag.get("local_root_found", False))),
                "local_root_count": int(diag.get("local_root_count", 0)),
            }
            for key, value in scalar_diagnostics.items():
                dtype = int if key in {"used_global_fallback", "local_root_found", "local_root_count"} else float
                IO.CreateDataset(green, f"{key}{suffix}", value, dtype=dtype)
            IO.CreateDataset(
                green,
                f"mu_search_mode{suffix}",
                str(diag.get("search_mode", getattr(self, "mu_search_mode", "global"))),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

        return None


class SigC(FLatDyn):

    def __init__(
        self,
        crystal: Crystal,
        dlr: DLR,
        sigh: np.ndarray = None,
        sigf: np.ndarray = None,
        siggwc: np.ndarray = None,
        mixing_method: str = "pulay",
        npulay: int = 5,
    ) -> object:
        super().__init__(crystal, dlr, mixing_method=mixing_method, npulay=npulay)
        self.flatstc = FLatStc(crystal=crystal)

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        nfreq = len(self.dlr.omega)
        self._static_shape = (norb, norb, ns, nk)
        self._dynamic_shape = (norb, norb, ns, nk, nfreq)

        self.sigh = self._validated_array("sigh", sigh, self._static_shape)
        self.sigf = self._validated_array("sigf", sigf, self._static_shape)
        self.siggwc = self._validated_array("siggwc", siggwc, self._dynamic_shape)
        self.sigimp = None
        self.kf = np.zeros(self._dynamic_shape, dtype=np.complex128, order="F")

    def _validated_array(
        self,
        name: str,
        value: np.ndarray,
        expected_shape: tuple,
    ) -> np.ndarray:
        if value is None:
            return None

        arr = np.asarray(value, dtype=np.complex128)
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, got {arr.shape}"
            )
        return np.array(arr, dtype=np.complex128, order="F", copy=True)

    def _accumulate_static(self, name: str, value: np.ndarray) -> None:
        arr = self._validated_array(name, value, self._static_shape)
        current = getattr(self, name)
        if current is None:
            setattr(self, name, arr)
        else:
            current += arr

        return None

    def _accumulate_dynamic(self, name: str, value: np.ndarray) -> None:
        arr = self._validated_array(name, value, self._dynamic_shape)
        current = getattr(self, name)
        if current is None:
            setattr(self, name, arr)
        else:
            current += arr

        return None

    def ImpEmbedding(
        self,
        sigimp: np.ndarray = None,
        sighimp: np.ndarray = None,
        sigfimp: np.ndarray = None,
        projector: Projector = None,
        key = None,
    ) -> None:
        if projector is None:
            raise ValueError("projector is required for ImpEmbedding")
        if key is None:
            raise ValueError("key is required for ImpEmbedding")
        if sigimp is None and sighimp is None and sigfimp is None:
            raise ValueError("at least one impurity self-energy is required")

        if sigimp is not None:
            self._accumulate_dynamic(
                "sigimp",
                self.Embedding(sigimp, projector=projector, key=key),
            )
        if sighimp is not None:
            self._accumulate_static(
                "sigh",
                self.flatstc.Embedding(sighimp, projector=projector, key=key),
            )
        if sigfimp is not None:
            self._accumulate_static(
                "sigf",
                self.flatstc.Embedding(sigfimp, projector=projector, key=key),
            )

        return None

    def __call__(self) -> None:
        self.kf.fill(0.0)

        if self.siggwc is not None:
            self.kf += self.siggwc
        if self.sigimp is not None:
            self.kf += self.sigimp
        if self.sigh is not None:
            self.kf += self.sigh[..., np.newaxis]
        if self.sigf is not None:
            self.kf += self.sigf[..., np.newaxis]

        return None

    
@timed_init
class SigGWC(FLatDyn):
    component = "siggwc"

    def __init__(
        self,
        crystal: Crystal,
        dlr : DLR,
        green : np.ndarray = None,
        wlat : np.ndarray = None,
        hdf5file : str = 'glob.h5',
        group : str = None,
        iteration: int = None,
    ) -> object:
        super().__init__(crystal, dlr)
        self.flatstc = FLatStc(crystal=crystal)
        norb, _, ns, nk, nfreq = green.shape
        ntau = len(self.dlr.tauF)
        self.rt = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.rf = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.kt = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')
        self.kf = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')
        self.stck = np.zeros((norb, norb, ns, nk), dtype=np.complex128, order='F')
        self.z = np.zeros((norb, norb, ns, nk), dtype=np.complex128, order='F')
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.iteration = iteration

        if green is None:
            logger.error("Error, green doesn't exist")
            sys.exit()

        if wlat is None:
            logger.error("Error, wlat doesn't exist")
            sys.exit()
        self.green = green
        self.wlat = wlat

        logger.info("GWC self-energy Calculation Start")
        start = time.time()
        self.Cal()
        end = time.time()
        logger.info("GWC self-energy Calculation Finish")
        logger.info(f"Calculation Time : {str(datetime.timedelta(seconds=end-start))}")

    def Cal(self)->np.ndarray: #SigmaGWC
        '''
        Generate correlated self-energy
        input : Wc(R,t), G(R,t)

        return : crtau, crfreq, cktau, ckfreq
        '''

        norbc = self.green.shape[0]
        ns = self.green.shape[2]
        nr = self.green.shape[3]
        ntau = len(self.dlr.tauF)
        norb = self.wlat.shape[0]

        G = self.green
        Wc = self.TauB2TauF(self.wlat)

        bbasis = self.crystal.bbasis
        s_idx = np.arange(ns)
        Wc_diag = Wc[:, :, s_idx, s_idx, :, :]  # (norb, norb, ns, nr, ntau)

        # Flatten (ns, nr, ntau) into single batch dim S for efficient BLAS dispatch
        S = ns * nr * ntau
        Wc_flat = Wc_diag.reshape(norb, norb, S)
        G_flat = np.ascontiguousarray(G).reshape(norbc, norbc, S)
        out_flat = np.zeros((norbc, norbc, S), dtype=np.complex128)

        # Group fermion orbitals by atom
        atom_groups = {}
        for i in range(norbc):
            a = int(self.crystal.forb2atom[i])
            atom_groups.setdefault(a, []).append(i)

        for orbs_a in atom_groups.values():
            oa = np.array(orbs_a)
            na = len(oa)
            bb_a = bbasis[np.ix_(oa, oa)]  # (na, na), 1-based

            for orbs_b in atom_groups.values():
                ob = np.array(orbs_b)
                nb = len(ob)
                bb_b = bbasis[np.ix_(ob, ob)]  # (nb, nb), 1-based

                G_block = G_flat[np.ix_(oa, ob)]  # (na, nb, S)

                
                # Only allocate for boson indices that actually appear.

                unique_a = np.unique(bb_a)
                unique_b = np.unique(bb_b)

                # Ma_dict[a] -> (na, na) indicator: Ma[k,i] = delta(bb_a[k,i], a)
                # Contract: temp_a[a][k, j, S] = sum_i Ma[k,i] * G[i, j, S]
                temp_a = {}
                for a in unique_a:
                    mask = (bb_a == a).astype(np.float64)  # (na, na)
                    # mask[k,i] * G[i,j,S] -> [k,j,S]
                    temp_a[a] = np.einsum('ki,ijS->kjS', mask, G_block)  # (na, nb, S)

                # For each (a,b) pair, accumulate:
                # out[k,p,S] -= Wc[a,b,S] * sum_j temp_a[a][k,j,S] * Mb[j,p]
                # bb_a/bb_b are 1-based boson indices from bbasis, so subtract
                # 1 when indexing into Wc_flat.
                result = np.zeros((na, nb, S), dtype=np.complex128)
                for a in unique_a:
                    for b in unique_b:
                        Wc_ab = Wc_flat[a - 1, b - 1]  # (S,)
                        mask_b = (bb_b == b).astype(np.float64)  # (nb, nb) where mask_b[j,p]
                        # sum_j temp_a[a][k,j,S] * mask_b[j,p] -> (na, nb, S)
                        contracted = np.einsum('kjS,jp->kpS', temp_a[a], mask_b)  # (na, nb, S)
                        result += Wc_ab[np.newaxis, np.newaxis, :] * contracted

                out_flat[np.ix_(oa, ob)] -= result

        crtau = np.asfortranarray(out_flat.reshape(norbc, norbc, ns, nr, ntau))
        cktau = self.R2K(crtau)
        ckfreq = self.T2F(cktau)
        crfreq = self.T2F(crtau)

        self.rt = crtau
        self.kt = cktau
        self.rf = crfreq
        self.kf = ckfreq

        return None

    def Mixing(
        self,
        iter: int = None,
        mix: float = None,
        method: str = "pulay",
        npulay: int = 5,
        key=None,
    ) -> None:
        self.kf = super().Mixing(
            iter=iter,
            mix=mix,
            component=self.component,
            value=self.kf,
            method=method,
            npulay=npulay,
            key=key,
        )
        self.kt = self.F2T(self.kf)
        self.rf = self.K2R(self.kf)
        self.rt = self.K2R(self.kt)
    
    
    def Save(self, fn: str, obj : np.ndarray = None, scf: bool = True):
        if fn is None:
            raise ValueError("SigGWC.Save requires fn")
        fn_write = fn
        if scf:
            if self.iteration is None:
                raise ValueError("SigGWC.Save requires iteration when scf=True")
            fn_write = f"{fn}.{self.iteration}"

        with h5py.File(self.hdf5file,'a') as file:
            if self.CheckGroup(self.hdf5file,self.group):
                group = file[self.group]
                if self.subgroup in group:
                    sigmac = group[self.subgroup]
                else:
                    sigmac = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                sigmac = group.create_group(self.subgroup)
            

            if obj is not None:
                IO.CreateDataset(sigmac, fn_write, obj, dtype=complex)
            else:
                IO.CreateDataset(sigmac, fn_write, self.kf, dtype=complex)

        return None

class GreenAB(FLatDyn):

    def __init__(self, crystal: Crystal, dlr : DLR) -> object:
        super().__init__(crystal, dlr)

        glob = h5py.File('../../glob_dat/global.dat', 'r')
        self.i_kerf = glob['full_space']['gw']['i_kref'][:]
        self.kpt_latt = glob['combasis_fermion']['kpt_latt'][:]
        self.nbndf = glob['full_space']['gw']['nbndf'][:]
        self.n_omega = glob['full_space']['gw']['n_omega'][:]
        self.n3 = glob['full_space']['Gfull_n3'][:]
        glob.close()

    def KI2KF(self):

        tempmat = np.zeros((self.nbndf[0], self.nbndf[0], self.n3[0], len(self.kpt_latt), self.crystal.ns), dtype=np.complex128, order='F')

        glob = h5py.File('../../glob_dat/global.dat', 'r')

        for js in range(self.crystal.ns):
            for iw in range(self.n3[0]):
                for ik in range(len(self.kpt_latt)):
                    kidx = self.i_kerf[ik]
                    name = 'Gfull_w_'+str(iw+1)+'_k_'+str(kidx)
                    tempmat[...,iw,ik, js] = glob['full_space'][name][:]
        glob.close()
        # kpt_latt != kpoints

        self.kf = np.copy(tempmat)

        self.kt = self.F2T(tempmat, 1, 1)
        self.rf = self.K2R(tempmat)
        self.rt = self.K2R(self.kt)

        return None
    
