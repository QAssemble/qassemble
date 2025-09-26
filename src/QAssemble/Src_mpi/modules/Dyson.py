"""
This module provides a set of static methods for solving the Dyson equation to compute
renormalized Green's functions from bare Green's functions and self-energies.
"""
import numpy as np
# import scipy
from .Common import Common

class Dyson:
    """
    A collection of static methods to solve the Dyson equation for various systems.

    The Dyson equation relates the bare Green's function (G₀) and the full (or dressed)
    Green's function (G) via the self-energy (Σ):
    G = G₀ + G₀ Σ G
    which can be rewritten as:
    G = (G₀⁻¹ - Σ)⁻¹

    This class provides methods to perform this calculation for fermionic and bosonic systems,
    in both static (frequency-independent self-energy) and dynamic (frequency-dependent
    self-energy) cases, and for both local and lattice Hamiltonians.
    """

    @staticmethod
    def FLocStc(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:
        """
        Solves the static Dyson equation for a local fermionic system.

        Args:
            ffin (np.ndarray): The bare local fermionic Green's function (norb, norb, ns).
            sig (np.ndarray): The static self-energy (norb, norb, ns).

        Returns:
            np.ndarray: The full local fermionic Green's function (norb, norb, ns).
        """

        norb, _, ns = ffin.shape
        ffout = np.zeros((norb, norb, ns), dtype=np.complex128, order='F')

        for js in range(ns):
            tempmat = np.zeros((norb, norb), dtype=np.complex128, order='F')
            tempmat2 = np.zeros((norb, norb), dtype=np.complex128, order='F')
            tempmat = -np.dot(sig[..., js], ffin[..., js])
            
            for iorb in range(norb):
                tempmat[iorb, iorb] += 1.0
            
            tempmat2 = Common.MatInv(tempmat)

            ffout[..., js] = np.dot(ffin[...,js], tempmat2)

        return ffout
    
    @staticmethod
    def FLatStc(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:
        """
        Solves the static Dyson equation for a lattice fermionic system.

        Args:
            ffin (np.ndarray): The bare lattice fermionic Green's function (norb, norb, ns, nk).
            sig (np.ndarray): The static self-energy (norb, norb, ns, nk).

        Returns:
            np.ndarray: The full lattice fermionic Green's function (norb, norb, ns, nk).
        """

        norb, _, ns, nk = ffin.shape
        ffout = np.zeros((norb, norb, ns, nk),dtype=np.complex128, order='F')

        for ik in range(nk):
            ffout[..., ik] = Dyson.FLocStc(ffin[..., ik], sig[...,ik])
        
        return ffout
    
    @staticmethod
    def FLocDyn(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:
        """
        Solves the dynamic Dyson equation for a local fermionic system.

        Args:
            ffin (np.ndarray): The bare local fermionic Green's function (norb, norb, ns, nfreq).
            sig (np.ndarray): The dynamic self-energy (norb, norb, ns, nfreq).

        Returns:
            np.ndarray: The full local fermionic Green's function (norb, norb, ns, nfreq).
        """

        nfreq = ffin.shape[3]
        ffout = np.zeros_like(ffin, dtype=np.complex128, order='F')

        for ifreq in range(nfreq):
            ffout[...,ifreq] = Dyson.FLocStc(ffin[..., ifreq], sig[..., ifreq])
        
        return ffout
    
    @staticmethod
    def FLatDyn(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:
        """
        Solves the dynamic Dyson equation for a lattice fermionic system.

        Args:
            ffin (np.ndarray): The bare lattice fermionic Green's function (norb, norb, ns, nk, nfreq).
            sig (np.ndarray): The dynamic self-energy (norb, norb, ns, nk, nfreq).

        Returns:
            np.ndarray: The full lattice fermionic Green's function (norb, norb, ns, nk, nfreq).
        """

        nfreq = ffin.shape[4]
        ffout = np.zeros_like(ffin, dtype=np.complex128, order='F')

        for ifreq in range(nfreq):
            ffout[..., ifreq] = Dyson.FLatStc(ffin[..., ifreq], sig[..., ifreq])

        return ffout
    
    @staticmethod
    def BLocStc(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:
        """
        Solves the static Dyson equation for a local bosonic system.

        Args:
            ffin (np.ndarray): The bare local bosonic Green's function (norb, norb, ns, ns).
            sig (np.ndarray): The static self-energy (norb, norb, ns, ns).

        Returns:
            np.ndarray: The full local bosonic Green's function (norb, norb, ns, ns).
        """

        norb, _, ns, _ = ffin.shape

        ffout = np.zeros((norb, norb, ns, ns), dtype=np.complex128, order='F')
        tempmat = np.zeros((norb*ns, norb*ns), dtype=np.complex128, order='F')
        tempmat2 = np.zeros((norb*ns, norb*ns), dtype=np.complex128, order='F')
        ffintemp = np.zeros((norb*ns, norb*ns), dtype=np.complex128, order='F')
        sigtemp = np.zeros((norb*ns, norb*ns), dtype=np.complex128, order='F')

        ndim = norb*ns

        for ks in range(ns):
            for jorb in range(norb):
                nn2 = [jorb, ks]
                ind2, nn2 = Common.Indexing(ndim, 2, [norb, ns], 1, 0, nn2)
                for js in range(ns):
                    for iorb in range(norb):
                        nn1 = [iorb ,js]
                        ind1, nn1 = Common.Indexing(ndim, 2, [norb, ns], 1, 0, nn1)
                        sigtemp[ind1, ind2] = sig[iorb, jorb, js, ks]
                        ffintemp[ind1, ind2] = ffin[iorb, jorb, js, ks]

        tempmat = -np.dot(sigtemp, ffintemp)
        for ind in range(ndim):
            tempmat[ind, ind] += 1.0
        
        tempmat2 = Common.MatInv(tempmat)
        tempmat3 = np.dot(ffintemp, tempmat2)

        for ks in range(ns):
            for jorb in range(norb):
                nn2 = [jorb, ks]
                ind2, nn2 = Common.Indexing(ndim, 2, [norb, ns], 1, 0, nn2)
                for js in range(ns):
                    for iorb in range(norb):
                        nn1 = [iorb ,js]
                        ind1, nn1 = Common.Indexing(ndim, 2, [norb, ns], 1, 0, nn1)
                        ffout[iorb, jorb, js, ks] = tempmat3[ind1, ind2]

        return ffout
    
    @staticmethod
    def BLocDyn(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:
        """
        Solves the dynamic Dyson equation for a local bosonic system.

        Args:
            ffin (np.ndarray): The bare local bosonic Green's function (norb, norb, ns, ns, nfreq).
            sig (np.ndarray): The dynamic self-energy (norb, norb, ns, ns, nfreq).

        Returns:
            np.ndarray: The full local bosonic Green's function (norb, norb, ns, ns, nfreq).
        """

        nfreq = ffin.shape[4]
        ffout = np.zeros_like(ffin, dtype=np.complex128, order='F')

        for ifreq in range(nfreq):
            ffout[...,ifreq] = Dyson.BLocStc(ffin[...,ifreq], sig[...,ifreq])

        return ffout
    
    @staticmethod
    def BLatStc(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:
        """
        Solves the static Dyson equation for a lattice bosonic system.

        Args:
            ffin (np.ndarray): The bare lattice bosonic Green's function (norb, norb, ns, ns, nk).
            sig (np.ndarray): The static self-energy (norb, norb, ns, ns, nk).

        Returns:
            np.ndarray: The full lattice bosonic Green's function (norb, norb, ns, ns, nk).
        """

        nk = ffin.shape[4]
        ffout = np.zeros_like(ffin, dtype=np.complex128, order='F')

        for ik in range(nk):
            ffout[...,ik] = Dyson.BLocStc(ffin[..., ik], sig[..., ik])

        return ffout
    
    @staticmethod
    def BLatDyn(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:
        """
        Solves the dynamic Dyson equation for a lattice bosonic system.

        Args:
            ffin (np.ndarray): The bare lattice bosonic Green's function (norb, norb, ns, ns, nk, nfreq).
            sig (np.ndarray): The dynamic self-energy (norb, norb, ns, ns, nk, nfreq).

        Returns:
            np.ndarray: The full lattice bosonic Green's function (norb, norb, ns, ns, nk, nfreq).
        """

        nfreq = ffin.shape[5]
        ffout = np.zeros_like(ffin, dtype=np.complex128, order='F')

        for ifreq in range(nfreq):
            ffout[..., ifreq] = Dyson.BLatStc(ffin[..., ifreq], sig[..., ifreq])

        return ffout