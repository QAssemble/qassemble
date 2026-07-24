"""
Fourier transform utilities for Green's functions and self-energies.
Serial implementation without MPI dependencies.
"""
import numpy as np
from scipy.linalg import lstsq
from typing import Tuple, Optional

ArrayLike = np.ndarray

class Fourier:
    """
    Static methods for Fourier transforms of Green's functions.
    
    Handles transformations between:
    - Imaginary-time (τ) and Matsubara frequency (iω_n or iν_n) representations
    - k-space and real-space representations
    
    All methods now run serially without MPI.
    """
    
    @staticmethod
    def _tail_fit_indices(
        freq: np.ndarray,
        tail_points: int,
        log_spaced: bool = False,
    ) -> np.ndarray:
        """Select grid indices for a high-frequency tail fit.

        ``log_spaced=False`` reproduces the historical selection: the
        ``tail_points`` largest-``|freq|`` entries.  On a signed uniform grid
        those are contiguous points (with +/- duplicates of the same
        ``|freq|``), which makes the 4-column tail design matrix severely
        ill-conditioned and the fitted ``c2``/``c3`` moments garbage.

        ``log_spaced=True`` instead snaps ~``tail_points`` log-spaced targets
        in ``[|freq|_max/10, |freq|_max]`` onto actual positive-branch grid
        points (``np.unique`` collapses duplicate snaps).  Spreading the fit
        over a decade keeps the design well conditioned; only real grid
        frequencies are ever used.  Falls back to the largest-``|freq|``
        selection when fewer than 4 distinct points survive.
        """
        freq = np.asarray(freq, dtype=np.float64)
        if log_spaced:
            pos = np.flatnonzero(freq > 0.0)
            if pos.size >= 4:
                fmax = float(np.max(freq[pos]))
                targets = np.geomspace(fmax / 10.0, fmax, int(tail_points))
                nearest = pos[
                    np.argmin(np.abs(freq[pos][None, :] - targets[:, None]), axis=1)
                ]
                idx = np.unique(nearest)
                if idx.size >= 4:
                    return idx
            tail_points = min(int(tail_points), freq.size)
        return np.argsort(np.abs(freq))[-int(tail_points):]

    @staticmethod
    def _tail_fit_sigma(
        design_ri: np.ndarray,
        b_ri: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """One-sigma lstsq uncertainties for the fitted tail coefficients."""
        residual = b_ri - design_ri @ c
        dof = max(b_ri.size - design_ri.shape[1], 1)
        cov = (residual @ residual / dof) * np.linalg.pinv(design_ri.T @ design_ri)
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))

    @staticmethod
    def FermionTailCoefficients(
        omega: np.ndarray,
        target: np.ndarray,
        tail_points: int = 5,
        *,
        log_spaced: bool = False,
        return_sigma: bool = False,
    ):
        """Robust fermionic high-frequency tail ``[c0, c1, c2, c3]``.

        Fits ``G(iw) ~ c0 + c1/(iw) + c2/(iw)^2 + c3/(iw)^3`` for a single
        scalar channel by a real (real/imag-stacked) least squares over the
        ``tail_points`` largest ``|omega|`` points.  This is the robust,
        over-determined replacement for the former two-point ``FLocDynM``
        formula and is stable against high-frequency noise.

        The returned coefficients use the physical high-frequency expansion
        sign directly.  The causal projector owns any internal sign conversion
        needed to match its pole-moment convention.

        Parameters
        ----------
        omega : ndarray[nfreq], dtype=float
            Fermionic Matsubara frequencies (the sampling grid).
        target : ndarray[nfreq], dtype=complex
            Scalar channel values G(i*omega) on that grid.
        tail_points : int
            Number of largest-|omega| points used in the fit (>= 4), or the
            log-spaced target count when ``log_spaced=True``.
        log_spaced : bool, keyword-only
            Select fit points via :meth:`_tail_fit_indices` with log spacing
            over the top decade instead of the largest-|omega| block.
        return_sigma : bool, keyword-only
            Also return the one-sigma lstsq uncertainties of the coefficients.

        Returns
        -------
        ndarray[4], dtype=float
            ``[c0, c1, c2, c3]`` in physical high-frequency expansion sign.
            When ``return_sigma=True`` a ``(c, sigma)`` tuple is returned
            instead, ``sigma`` being ndarray[4] of one-sigma uncertainties.
        """
        omega = np.asarray(omega, dtype=np.float64)
        target = np.asarray(target, dtype=np.complex128)
        if omega.ndim != 1 or target.ndim != 1 or omega.shape != target.shape:
            raise ValueError("omega and target must be 1D arrays of equal length")
        if tail_points < 4 or (not log_spaced and tail_points > omega.size):
            raise ValueError(
                f"tail_points must be in [4, {omega.size}], got {tail_points}"
            )

        idx = Fourier._tail_fit_indices(omega, tail_points, log_spaced)
        z = 1j * omega[idx]
        design = np.column_stack([np.ones_like(z), 1.0 / z, 1.0 / z**2, 1.0 / z**3])
        b = target[idx]
        design_ri = np.vstack([design.real, design.imag])
        b_ri = np.concatenate([b.real, b.imag])
        c, *_ = lstsq(design_ri, b_ri)
        c = np.asarray(c, dtype=float)
        if not np.all(np.isfinite(c)):
            raise ValueError("tail coefficient fit produced non-finite values")
        if return_sigma:
            return c, Fourier._tail_fit_sigma(design_ri, b_ri, c)
        return c

    @staticmethod
    def BosonTailCoefficients(
        nu: np.ndarray,
        target: np.ndarray,
        tail_points: int = 5,
        *,
        log_spaced: bool = False,
        return_sigma: bool = False,
    ):
        """Robust bosonic high-frequency tail ``[c0, c1, c2, c3]``.

        Fits ``chi(inu) ~ c0 + c1/(inu) + c2/(inu)^2 + c3/(inu)^3`` for a single
        scalar bosonic channel by a real (real/imag-stacked) least squares over
        the ``tail_points`` largest ``|nu|`` points.  This is the bosonic
        counterpart of :meth:`FermionTailCoefficients`: same robust,
        over-determined fit, applied on the bosonic Matsubara grid.

        The returned coefficients use the physical high-frequency expansion
        sign directly.  The causal projector owns any internal sign conversion
        needed to match its pole-moment convention.  ``c0`` is a genuine
        ``(inu)^0`` constant and keeps its physical sign.

        Parameters
        ----------
        nu : ndarray[nfreq], dtype=float
            Bosonic Matsubara frequencies (the sampling grid; positive-only on
            the native uniform grid, signed-symmetric on the DLR grid).
        target : ndarray[nfreq], dtype=complex
            Scalar channel values chi(i*nu) on that grid.
        tail_points : int
            Number of largest-|nu| points used in the fit (>= 4), or the
            log-spaced target count when ``log_spaced=True``.
        log_spaced : bool, keyword-only
            Select fit points via :meth:`_tail_fit_indices` with log spacing
            over the top decade instead of the largest-|nu| block.
        return_sigma : bool, keyword-only
            Also return the one-sigma lstsq uncertainties of the coefficients.

        Returns
        -------
        ndarray[4], dtype=float
            ``[c0, c1, c2, c3]`` in physical high-frequency expansion sign.
            When ``return_sigma=True`` a ``(c, sigma)`` tuple is returned
            instead, ``sigma`` being ndarray[4] of one-sigma uncertainties.
        """
        nu = np.asarray(nu, dtype=np.float64)
        target = np.asarray(target, dtype=np.complex128)
        if nu.ndim != 1 or target.ndim != 1 or nu.shape != target.shape:
            raise ValueError("nu and target must be 1D arrays of equal length")
        if tail_points < 4 or (not log_spaced and tail_points > nu.size):
            raise ValueError(
                f"tail_points must be in [4, {nu.size}], got {tail_points}"
            )

        idx = Fourier._tail_fit_indices(nu, tail_points, log_spaced)
        z = 1j * nu[idx]
        design = np.column_stack([np.ones_like(z), 1.0 / z, 1.0 / z**2, 1.0 / z**3])
        b = target[idx]
        design_ri = np.vstack([design.real, design.imag])
        b_ri = np.concatenate([b.real, b.imag])
        c, *_ = lstsq(design_ri, b_ri)
        c = np.asarray(c, dtype=float)
        if not np.all(np.isfinite(c)):
            raise ValueError("tail coefficient fit produced non-finite values")
        if return_sigma:
            return c, Fourier._tail_fit_sigma(design_ri, b_ri, c)
        return c

    @staticmethod
    def BLocDynM(freq: np.ndarray, ff: np.ndarray, oddzero: bool,
                 highzero: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate high-frequency moments for local bosonic function.
        
        Bosonic functions have expansion χ(iν_n) ~ c₀ + c₁/(iν_n) + c₂/(iν_n)² + ...
        
        Parameters
        ----------
        freq : ndarray[nfreq]
            Bosonic Matsubara frequencies ν_n = 2πn/β
        ff : ndarray[norb, norb, ns, ns, nfreq], dtype=complex
            Local bosonic function
        oddzero : bool
            True if odd moments vanish (c₁ = c₃ = 0)
        highzero : bool
            True to enforce c₀ = 0
            
        Returns
        -------
        moment : ndarray[norb, norb, ns, ns, 3]
            Tail coefficients
        high : ndarray[norb, norb, ns, ns]
            Constant term
        """
        ai = 1j
        norb, _, ns, _, _ = ff.shape
        moment = np.zeros((norb, norb, ns, ns, 3), dtype=np.complex128, order='F')
        high = np.zeros((norb, norb, ns, ns), dtype=np.complex128, order='F')
        
        if oddzero:
            # Only even moments: simplified fitting
            if highzero:
                moment[..., 1] = ff[..., -1] * (freq[-1] * ai)**2
            else:
                # Two-point finite difference for c₂
                moment[..., 1] = (
                    (ff[..., -1] - ff[..., -2]) * -1.0 
                    * (freq[-1] * ai * freq[-2] * ai)**2 
                    / ((freq[-1] * ai + freq[-2] * ai) * (freq[-1] * ai - freq[-2] * ai))
                )
                high = ff[..., -1] - moment[..., 1] / (freq[-1] * ai)**2
        else:
            # General case with all moments
            if highzero:
                for is_ in range(ns):
                    for js in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                # c₁ from anti-hermitian part
                                moment[iorb, jorb, is_, js, 0] += (
                                    (ff[iorb, jorb, is_, js, -1] 
                                     - np.conj(ff[jorb, iorb, js, is_, -1])) 
                                    / (2.0 * (freq[-1] * ai))
                                )
                                # c₂ from hermitian part
                                moment[iorb, jorb, is_, js, 1] += (
                                    (ff[iorb, jorb, is_, js, -1] 
                                     + np.conj(ff[jorb, iorb, js, is_, -1])) 
                                    / (2.0 * (freq[-1] * ai)**2)
                                )
            else:
                # Full 4-parameter fit
                amat = np.zeros((4, 4), dtype=np.complex128)
                bmat = np.zeros((4, 1), dtype=np.complex128)
                
                for is_ in range(ns):
                    for js in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                # Build constraint matrix
                                amat[0, :] = [1.0, 1.0/(freq[-1]*ai), 
                                            1.0/(freq[-1]*ai)**2, 1.0/(freq[-1]*ai)**3]
                                amat[1, :] = [1.0, -1.0/(freq[-1]*ai), 
                                            1.0/(freq[-1]*ai)**2, -1.0/(freq[-1]*ai)**3]
                                amat[2, :] = [1.0, 1.0/(freq[-2]*ai), 
                                            1.0/(freq[-2]*ai)**2, 1.0/(freq[-2]*ai)**3]
                                amat[3, :] = [1.0, -1.0/(freq[-2]*ai), 
                                            1.0/(freq[-2]*ai)**2, -1.0/(freq[-2]*ai)**3]
                                
                                bmat[0, 0] = ff[iorb, jorb, is_, js, -1]
                                bmat[1, 0] = np.conj(ff[jorb, iorb, js, is_, -1])
                                bmat[2, 0] = ff[iorb, jorb, is_, js, -2]
                                bmat[3, 0] = np.conj(ff[jorb, iorb, js, is_, -2])
                                
                                x = np.linalg.solve(amat, bmat)
                                
                                high[iorb, jorb, is_, js] = x[0, 0]
                                moment[iorb, jorb, is_, js, 0] = x[1, 0]
                                moment[iorb, jorb, is_, js, 1] = x[2, 0]
                                moment[iorb, jorb, is_, js, 2] = x[3, 0]
        
        # Symmetrize
        for iorb in range(norb):
            for jorb in range(norb):
                for is_ in range(ns):
                    for js in range(ns):
                        high[iorb, jorb, is_, js] = (
                            np.conj(high[jorb, iorb, js, is_]) + high[iorb, jorb, is_, js]
                        ) / 2.0
                        for ii in range(3):
                            moment[iorb, jorb, is_, js, ii] = (
                                np.conj(moment[jorb, iorb, js, is_, ii]) 
                                + moment[iorb, jorb, is_, js, ii]
                            ) / 2.0
        
        return moment, high
    
    @staticmethod
    def BLatDynM(freq: np.ndarray, ff: np.ndarray, oddzero: bool, 
                 highzero: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate high-frequency moments for lattice bosonic function.
        
        Parameters
        ----------
        freq : ndarray[nfreq]
            Bosonic Matsubara frequencies
        ff : ndarray[norb, norb, ns, ns, nk, nfreq], dtype=complex
            Lattice bosonic function
        oddzero : bool
            True if odd moments vanish
        highzero : bool
            True to enforce c₀ = 0
            
        Returns
        -------
        moment : ndarray[norb, norb, ns, ns, nk, 3]
            k-dependent tail coefficients
        high : ndarray[norb, norb, ns, ns, nk]
            k-dependent constant term
        """
        norb, _, ns, _, nk, _ = ff.shape
        moment = np.zeros((norb, norb, ns, ns, nk, 3), dtype=np.complex128, order='F')
        high = np.zeros((norb, norb, ns, ns, nk), dtype=np.complex128, order='F')
        
        for ik in range(nk):
            moment[..., ik, :], high[..., ik] = Fourier.BLocDynM(
                freq, ff[..., ik, :], oddzero, highzero
            )
        
        return moment, high
    
    @staticmethod
    def _validate_kgrid(kgrid: Tuple[int, int, int], size: int) -> Tuple[int, int, int]:
        if len(kgrid) != 3:
            raise ValueError(f"kgrid must have three entries (nx, ny, nz), got {kgrid}.")
        nx, ny, nz = (int(val) for val in kgrid)
        if nx * ny * nz != size:
            raise ValueError(
                f"Incompatible kgrid {kgrid}: product {nx * ny * nz} "
                f"does not match data size {size}."
            )
        return nx, ny, nz
    
    @staticmethod
    def FLatStcK2R(fin: np.ndarray, kgrid: Tuple[int, int, int]) -> np.ndarray:
        """
        Transform fermionic static quantity from k-space to real-space.
        
        Uses inverse FFT: f(R) = (1/N_k) Σ_k f(k) e^{ik·R}
        
        Parameters
        ----------
        fin : ndarray[norb, norb, ns, nk], dtype=complex
            k-space fermionic function
        kgrid : tuple(nx, ny, nz)
            k-point mesh dimensions
            
        Returns
        -------
        fout : ndarray[norb, norb, ns, nr], dtype=complex
            Real-space fermionic function
        """
        norb, _, ns, nk = fin.shape
        nx, ny, nz = Fourier._validate_kgrid(kgrid, nk)
        grid = np.asfortranarray(fin, dtype=np.complex128).reshape(
            (norb, norb, ns, nx, ny, nz), order="F"
        )
        real_grid = np.fft.ifftn(grid, axes=(-3, -2, -1))
        return real_grid.reshape((norb, norb, ns, nk), order="F")
    
    @staticmethod
    def FLatDynK2R(fin: np.ndarray, kgrid: Tuple[int, int, int]) -> np.ndarray:
        """
        Transform fermionic dynamic quantity from k-space to real-space.
        
        Parameters
        ----------
        fin : ndarray[norb, norb, ns, nk, nfreq], dtype=complex
            k-space fermionic function at all frequencies
        kgrid : tuple(nx, ny, nz)
            k-point mesh dimensions
            
        Returns
        -------
        fout : ndarray[norb, norb, ns, nr, nfreq], dtype=complex
            Real-space fermionic function at all frequencies
        """
        norb, _, ns, nk, nfreq = fin.shape
        nx, ny, nz = Fourier._validate_kgrid(kgrid, nk)
        grid = np.asfortranarray(fin, dtype=np.complex128).reshape(
            (norb, norb, ns, nx, ny, nz, nfreq), order="F"
        )
        real_grid = np.fft.ifftn(grid, axes=(-4, -3, -2))
        return real_grid.reshape((norb, norb, ns, nk, nfreq), order="F")
    
    @staticmethod
    def FLatStcR2K(fin: np.ndarray, kgrid: Tuple[int, int, int]) -> np.ndarray:
        """
        Transform fermionic static quantity from real-space to k-space.
        
        Uses FFT: f(k) = Σ_R f(R) e^{-ik·R}
        
        Parameters
        ----------
        fin : ndarray[norb, norb, ns, nr], dtype=complex
            Real-space fermionic function
        kgrid : tuple(nx, ny, nz)
            k-point mesh dimensions
            
        Returns
        -------
        fout : ndarray[norb, norb, ns, nk], dtype=complex
            k-space fermionic function
        """
        norb, _, ns, nr = fin.shape
        nx, ny, nz = Fourier._validate_kgrid(kgrid, nr)
        grid = np.asfortranarray(fin, dtype=np.complex128).reshape(
            (norb, norb, ns, nx, ny, nz), order="F"
        )
        k_grid = np.fft.fftn(grid, axes=(-3, -2, -1))
        return k_grid.reshape((norb, norb, ns, nr), order="F")
    
    @staticmethod
    def FLatDynR2K(fin: np.ndarray, kgrid: Tuple[int, int, int]) -> np.ndarray:
        """
        Transform fermionic dynamic quantity from real-space to k-space.
        
        Parameters
        ----------
        fin : ndarray[norb, norb, ns, nr, nfreq], dtype=complex
            Real-space fermionic function at all frequencies
        kgrid : tuple(nx, ny, nz)
            k-point mesh dimensions
            
        Returns
        -------
        fout : ndarray[norb, norb, ns, nk, nfreq], dtype=complex
            k-space fermionic function at all frequencies
        """
        norb, _, ns, nr, nfreq = fin.shape
        nx, ny, nz = Fourier._validate_kgrid(kgrid, nr)
        grid = np.asfortranarray(fin, dtype=np.complex128).reshape(
            (norb, norb, ns, nx, ny, nz, nfreq), order="F"
        )
        k_grid = np.fft.fftn(grid, axes=(-4, -3, -2))
        return k_grid.reshape((norb, norb, ns, nr, nfreq), order="F")
    
    @staticmethod
    def BLatStcK2R(fin: np.ndarray, kgrid: Tuple[int, int, int]) -> np.ndarray:
        """
        Transform bosonic static quantity from k-space to real-space.
        
        Parameters
        ----------
        fin : ndarray[norb, norb, ns, ns, nk], dtype=complex
            k-space bosonic function
        kgrid : tuple(nx, ny, nz)
            k-point mesh dimensions
            
        Returns
        -------
        fout : ndarray[norb, norb, ns, ns, nr], dtype=complex
            Real-space bosonic function
        """
        norb, _, ns, _, nk = fin.shape
        nx, ny, nz = Fourier._validate_kgrid(kgrid, nk)
        grid = np.asfortranarray(fin, dtype=np.complex128).reshape(
            (norb, norb, ns, ns, nx, ny, nz), order="F"
        )
        real_grid = np.fft.ifftn(grid, axes=(-3, -2, -1))
        return real_grid.reshape((norb, norb, ns, ns, nk), order="F")
    
    @staticmethod
    def BLatDynK2R(fin: np.ndarray, kgrid: Tuple[int, int, int]) -> np.ndarray:
        """
        Transform bosonic dynamic quantity from k-space to real-space.
        
        Parameters
        ----------
        fin : ndarray[norb, norb, ns, ns, nk, nfreq], dtype=complex
            k-space bosonic function at all frequencies
        kgrid : tuple(nx, ny, nz)
            k-point mesh dimensions
            
        Returns
        -------
        fout : ndarray[norb, norb, ns, ns, nr, nfreq], dtype=complex
            Real-space bosonic function at all frequencies
        """
        norb, _, ns, _, nk, nfreq = fin.shape
        nx, ny, nz = Fourier._validate_kgrid(kgrid, nk)
        grid = np.asfortranarray(fin, dtype=np.complex128).reshape(
            (norb, norb, ns, ns, nx, ny, nz, nfreq), order="F"
        )
        real_grid = np.fft.ifftn(grid, axes=(-4, -3, -2))
        return real_grid.reshape((norb, norb, ns, ns, nk, nfreq), order="F")
    
    @staticmethod
    def BLatStcR2K(fin: np.ndarray, kgrid: Tuple[int, int, int]) -> np.ndarray:
        """
        Transform bosonic static quantity from real-space to k-space.
        
        Parameters
        ----------
        fin : ndarray[norb, norb, ns, ns, nr], dtype=complex
            Real-space bosonic function
        kgrid : tuple(nx, ny, nz)
            k-point mesh dimensions
            
        Returns
        -------
        fout : ndarray[norb, norb, ns, ns, nk], dtype=complex
            k-space bosonic function
        """
        norb, _, ns, _, nr = fin.shape
        nx, ny, nz = Fourier._validate_kgrid(kgrid, nr)
        grid = np.asfortranarray(fin, dtype=np.complex128).reshape(
            (norb, norb, ns, ns, nx, ny, nz), order="F"
        )
        k_grid = np.fft.fftn(grid, axes=(-3, -2, -1))
        return k_grid.reshape((norb, norb, ns, ns, nr), order="F")
    
    @staticmethod
    def BLatDynR2K(fin: np.ndarray, kgrid: Tuple[int, int, int]) -> np.ndarray:
        """
        Transform bosonic dynamic quantity from real-space to k-space.
        
        Parameters
        ----------
        fin : ndarray[norb, norb, ns, ns, nr, nfreq], dtype=complex
            Real-space bosonic function at all frequencies
        kgrid : tuple(nx, ny, nz)
            k-point mesh dimensions
            
        Returns
        -------
        fout : ndarray[norb, norb, ns, ns, nk, nfreq], dtype=complex
            k-space bosonic function at all frequencies
        """
        norb, _, ns, _, nr, nfreq = fin.shape
        nx, ny, nz = Fourier._validate_kgrid(kgrid, nr)
        grid = np.asfortranarray(fin, dtype=np.complex128).reshape(
            (norb, norb, ns, ns, nx, ny, nz, nfreq), order="F"
        )
        k_grid = np.fft.fftn(grid, axes=(-4, -3, -2))
        return k_grid.reshape((norb, norb, ns, ns, nr, nfreq), order="F")
    
    @staticmethod
    def FPathStcR2K(fin: np.ndarray, kvec: np.ndarray, rvec: np.ndarray) -> np.ndarray:
        """
        Transform fermionic static quantity from real-space to arbitrary k-path.
        
        Direct evaluation: f(k) = Σ_R f(R) e^{-2πik·R}
        
        Parameters
        ----------
        fin : ndarray[norb, norb, ns, nr], dtype=complex
            Real-space fermionic function
        kvec : ndarray[nk_path, 3], dtype=float
            k-points along path (fractional coordinates)
        rvec : ndarray[nr, 3], dtype=float
            Real-space lattice vectors (fractional coordinates)
            
        Returns
        -------
        fout : ndarray[norb, norb, ns, nk_path], dtype=complex
            k-space fermionic function along path
        """
        norb, _, ns, nr = fin.shape
        nk_path = len(kvec)
        
        fout = np.zeros((norb, norb, ns, nk_path), dtype=np.complex128, order='F')
        
        # Precompute phase factors
        phase = np.exp(-2.0j * np.pi * kvec @ rvec.T)  # [nk_path, nr]
        
        for js in range(ns):
            for jorb in range(norb):
                for iorb in range(norb):
                    # Direct summation over R
                    fout[iorb, jorb, js, :] = phase @ fin[iorb, jorb, js, :]
        
        return fout
    
    @staticmethod
    def FPathDynR2K(fin: np.ndarray, kvec: np.ndarray, rvec: np.ndarray) -> np.ndarray:
        """
        Transform fermionic static quantity from real-space to arbitrary k-path.
        
        Direct evaluation: f(k) = Σ_R f(R) e^{-2πik·R}
        
        Parameters
        ----------
        fin : ndarray[norb, norb, ns, nr], dtype=complex
            Real-space fermionic function
        kvec : ndarray[nk_path, 3], dtype=float
            k-points along path (fractional coordinates)
        rvec : ndarray[nr, 3], dtype=float
            Real-space lattice vectors (fractional coordinates)
            
        Returns
        -------
        fout : ndarray[norb, norb, ns, nk_path], dtype=complex
            k-space fermionic function along path
        """
        norb, _, ns, nr, nft = fin.shape
        nk_path = len(kvec)
        
        fout = np.zeros((norb, norb, ns, nk_path, nft), dtype=np.complex128, order='F')
        
        # Precompute phase factors
        # phase = np.exp(-2.0j * np.pi * kvec @ rvec.T)  # [nk_path, nr]
        
        # for js in range(ns):
        #     for jorb in range(norb):
        #         for iorb in range(norb):
        #             # Direct summation over R
        #             fout[iorb, jorb, js, :] = phase @ fin[iorb, jorb, js, :]
        for ift in range(nft):
            fout[..., ift] = Fourier.FPathStcR2K(fin[..., ift], kvec, rvec)
        
        return fout

class FourierMPI:

    pass
