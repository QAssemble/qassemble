"""
Fourier transform utilities for Green's functions and self-energies.
Serial implementation without MPI dependencies.
"""
import numpy as np
from scipy.linalg import solve
# from numba import jit
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
    def FLocDynM(freq: np.ndarray, ff1: np.ndarray, ff2: np.ndarray, 
                 isgreen: bool, highzero: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate high-frequency moments of local fermionic Green's function.
        
        Extracts tail coefficients G(iω_n) ~ c₀ + c₁/(iω_n) + c₂/(iω_n)² + c₃/(iω_n)³
        using values at highest frequencies.
        
        Parameters
        ----------
        freq : ndarray[nfreq], dtype=float
            Fermionic Matsubara frequencies ω_n = (2n+1)π/β
        ff1 : ndarray[norb, norb, ns], dtype=complex
            G or Σ at highest frequency point
        ff2 : ndarray[norb, norb, ns], dtype=complex  
            G or Σ at second-highest frequency point
        isgreen : bool
            True for Green's function (c₁ = 1), False for self-energy
        highzero : bool
            True to enforce c₀ = 0 (constant term)
            
        Returns
        -------
        moment : ndarray[norb, norb, ns, 3], dtype=complex
            Tail coefficients [c₁, c₂, c₃]
        high : ndarray[norb, norb, ns], dtype=complex
            Constant term c₀
        """
        norb, _, ns = ff1.shape
        nfreq = len(freq)
        
        moment = np.zeros((norb, norb, ns, 3), dtype=np.complex128, order='F')
        high = np.zeros((norb, norb, ns), dtype=np.complex128, order='F')
        
        ai = 1j
        
        if isgreen:
            # Green's function: enforce c₁ = δᵢⱼ
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        moment[iorb, jorb, js, 0] = 1.0 if iorb == jorb else 0.0
                        
                        # c₂ from hermiticity constraint
                        moment[iorb, jorb, js, 1] = (
                            (ff1[iorb, jorb, js] + np.conj(ff1[jorb, iorb, js])) / 2.0 
                            * (freq[nfreq-1] * ai)**2
                        )
                        
                        # c₃ from anti-hermiticity constraint  
                        moment[iorb, jorb, js, 2] = (
                            (ff1[iorb, jorb, js] - np.conj(ff1[jorb, iorb, js]) 
                             - moment[iorb, jorb, js, 0] * 2.0 / (freq[nfreq-1] * ai)) / 2.0 
                            * (freq[nfreq-1] * ai)**3
                        )
        else:
            # Self-energy fitting
            if highzero:
                # Simplified case: c₀ = 0
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            moment[iorb, jorb, js, 0] = (
                                (ff1[iorb, jorb, js] - np.conj(ff1[jorb, iorb, js])) / 2.0 
                                * (freq[nfreq-1] * ai)
                            )
                            moment[iorb, jorb, js, 1] = (
                                (ff1[iorb, jorb, js] + np.conj(ff1[jorb, iorb, js])) / 2.0 
                                * (freq[nfreq-1] * ai)**2
                            )
            else:
                # Full 4-parameter fit using two frequency points
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            # Build linear system for [c₀, c₁, c₂, c₃]ᵀ
                            amat = np.zeros((4, 4), dtype=np.complex128, order='F')
                            bmat = np.zeros((4, 1), dtype=np.complex128, order='F')
                            
                            # Constraints from ff1 and its hermitian conjugate
                            amat[0, :] = [1.0, 1.0/(freq[nfreq-1]*ai), 
                                         1.0/(freq[nfreq-1]*ai)**2, 1.0/(freq[nfreq-1]*ai)**3]
                            amat[1, :] = [1.0, -1.0/(freq[nfreq-1]*ai), 
                                         1.0/(freq[nfreq-1]*ai)**2, -1.0/(freq[nfreq-1]*ai)**3]
                            # Constraints from ff2 and its hermitian conjugate
                            amat[2, :] = [1.0, 1.0/(freq[nfreq-2]*ai), 
                                         1.0/(freq[nfreq-2]*ai)**2, 1.0/(freq[nfreq-2]*ai)**3]
                            amat[3, :] = [1.0, -1.0/(freq[nfreq-2]*ai), 
                                         1.0/(freq[nfreq-2]*ai)**2, -1.0/(freq[nfreq-2]*ai)**3]
                            
                            bmat[0, 0] = ff1[iorb, jorb, js]
                            bmat[1, 0] = np.conj(ff1[jorb, iorb, js])
                            bmat[2, 0] = ff2[iorb, jorb, js]
                            bmat[3, 0] = np.conj(ff2[jorb, iorb, js])
                            
                            sol = solve(amat, bmat)
                            
                            high[iorb, jorb, js] = sol[0, 0]
                            moment[iorb, jorb, js, 0] = sol[1, 0]
                            moment[iorb, jorb, js, 1] = sol[2, 0]
                            moment[iorb, jorb, js, 2] = sol[3, 0]
        
        # Enforce hermiticity of tail coefficients
        for js in range(ns):
            high[:, :, js] = (high[:, :, js].T.conj() + high[:, :, js]) / 2.0
            for ii in range(3):
                moment[:, :, js, ii] = (moment[:, :, js, ii].T.conj() + moment[:, :, js, ii]) / 2.0
        
        return moment, high
    
    @staticmethod
    def FLatDynM(freq: np.ndarray, ff1: np.ndarray, ff2: np.ndarray, 
                 isgreen: bool, highzero: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate high-frequency moments for lattice fermionic function.
        
        Parameters
        ----------
        freq : ndarray[nfreq]
            Fermionic Matsubara frequencies
        ff1 : ndarray[norb, norb, ns, nk], dtype=complex
            Lattice function at highest frequency
        ff2 : ndarray[norb, norb, ns, nk], dtype=complex
            Lattice function at second-highest frequency
        isgreen : bool
            True for Green's function, False for self-energy
        highzero : bool
            True to enforce c₀ = 0
            
        Returns
        -------
        moment : ndarray[norb, norb, ns, nk, 3]
            k-dependent tail coefficients
        high : ndarray[norb, norb, ns, nk]
            k-dependent constant term
        """
        norb, _, ns, nk = ff1.shape
        
        moment = np.zeros((norb, norb, ns, nk, 3), dtype=np.complex128, order='F')
        high = np.zeros((norb, norb, ns, nk), dtype=np.complex128, order='F')
        
        # Process each k-point independently
        for ik in range(nk):
            moment[..., ik, :], high[..., ik] = Fourier.FLocDynM(
                freq, ff1[..., ik], ff2[..., ik], isgreen, highzero
            )
        
        return moment, high
    
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
