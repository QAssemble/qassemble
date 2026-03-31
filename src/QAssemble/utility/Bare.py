"""
This module provides a collection of static methods to calculate bare Green's functions
for fermionic and bosonic systems in various representations (frequency/imaginary time,
local/lattice).
"""
from .Common import Common
from sys import exit
import numpy as np

class Bare:
    """
    A collection of static methods to compute bare Green's functions for non-interacting systems.

    The methods handle fermionic and bosonic statistics, and can compute Green's functions
    in both frequency and imaginary-time (Matsubara) domains. It also provides functions
    to compute these for both single-site (local) and multi-site (lattice) Hamiltonians.
    """

    @staticmethod
    def FFreq(freq : np.ndarray, energy : np.ndarray) -> np.ndarray:
        """
        Calculates the bare fermionic Green's function in frequency domain.

        Args:
            freq (np.ndarray): An array of fermionic Matsubara frequencies (iω_n).
            energy (np.ndarray): An array of single-particle energy eigenvalues.

        Returns:
            np.ndarray: The Green's function G(iω_n) = 1 / (iω_n - E).
        """

        nfreq = len(freq)
        gfreq = np.ndarray((nfreq), dtype=np.ndarray, order='F')

        for ifreq in range(nfreq):
            gfreq[ifreq] = 1.0/(1j*freq[ifreq] - energy)

        return gfreq
    
    @staticmethod
    def FTau(tau : np.ndarray, beta : np.float64, energy : np.ndarray) -> np.ndarray:
        """
        Calculates the bare fermionic Green's function in imaginary time domain.

        Args:
            tau (np.ndarray): An array of imaginary time points (τ).
            energy (np.ndarray): An array of single-particle energy eigenvalues.

        Returns:
            np.ndarray: The Green's function G(τ).
        """
        
        tau = np.asarray(tau, dtype=np.float64, order='F')
        ntau = len(tau)
        gtau = np.ndarray((ntau), dtype=np.complex128, order='F')

        pi = np.pi
        # beta = tau[0]/(np.cos(pi*(ntau-0.5)/ntau) + 1.0)*2.0
        machep = np.finfo(np.float64).eps
        
        for itau in range(ntau):
            taumod = (tau[itau] % beta)
            unitnum = int(tau[itau] - taumod)/beta

            if (taumod < machep):
                unitnum = unitnum-1

            # Ensure unitnum is a clean integer to avoid power operation warnings
            # unitnum = int(np.round(unitnum))
            taunew = tau[itau] - beta * unitnum

            if (energy > 0):
                gtau[itau] = complex(-1)**(unitnum+1)*np.exp(np.complex128(-energy*taunew)) \
                    * (1 - 1/(np.exp(np.complex128(energy*beta)) + 1))
            else:
                gtau[itau] = complex(-1)**(unitnum+1)* np.exp(np.complex128(-energy*(taunew-beta))) \
                    * (1.0/(np.exp(np.complex128(energy*beta)) + 1))

        return gtau

    @staticmethod
    def BFreq(freq : np.ndarray, energy : np.ndarray) -> np.ndarray:
        """
        Calculates the bare bosonic Green's function in frequency domain.

        Args:
            freq (np.ndarray): An array of bosonic Matsubara frequencies (iΩ_n).
            energy (np.ndarray): An array of single-particle energy eigenvalues.

        Returns:
            np.ndarray: The Green's function W(iΩ_n) = 1 / (iΩ_n - E).
        """

        nfreq = len(freq)
        wfreq = np.zeros((nfreq), dtype=np.complex128, order='F')

        for ifreq in range(nfreq):
            wfreq[ifreq] = 1.0 / (1j * freq[ifreq] - energy)

        return wfreq
    
    @staticmethod
    def BTau(tau : np.ndarray, energy : np.ndarray) -> np.ndarray:
        """
        Calculates the bare bosonic Green's function in imaginary time domain.

        Args:
            tau (np.ndarray): An array of imaginary time points (τ).
            energy (np.ndarray): An array of single-particle energy eigenvalues.

        Returns:
            np.ndarray: The Green's function W(τ).
        """

        ntau = len(tau)
        wtau = np.zeros((ntau), dtype=np.complex128, order='F')

        pi = np.pi
        beta = tau[0]/(np.cos(pi*(ntau-0.5)/ntau) + 1.0)*2.0
        machep = np.finfo(np.float64).eps

        if (abs(energy) < 1.0e-12):
            print("Zero energy in Bare.BTau. impossible")
            exit()

        for itau in range(ntau):
            taumod = (tau[itau] % beta)
            unitnum = int(tau[itau] - taumod)/beta
            if (taumod < machep):
                unitnum = unitnum - 1
            taunew = tau[itau] - beta*unitnum

            if (energy > 0):
                wtau[itau] = -np.exp(np.complex128(-energy*beta))\
                    * (1.0 - 1.0/(np.exp(np.complex128(energy*beta)) - 1))
            else:
                wtau[itau] = -np.exp(np.complex128(-energy*(taunew-beta)))\
                    * (1.0/np.exp(np.complex128(energy*beta)) - 1)
        return wtau


    @staticmethod
    def FLocFreq(freq : np.ndarray, hloc : np.ndarray)->np.ndarray:
        """
        Calculates the local fermionic Green's function in frequency domain from a local Hamiltonian.

        Args:
            freq (np.ndarray): Array of fermionic Matsubara frequencies.
            hloc (np.ndarray): The local Hamiltonian matrix (norb, norb, ns).

        Returns:
            np.ndarray: The local Green's function G_loc(iω_n) (norb, norb, ns, nfreq).
        """

        norb, _, ns = hloc.shape
        nfreq = len(freq)
        tempmat = np.zeros((norb, norb), dtype=np.complex128, order='F')
        tempmat2 = np.zeros((norb, norb), dtype=np.complex128, order='F')
        gloc = np.zeros((norb, norb, ns, nfreq), dtype=np.complex128, order='F')

        for js in range(ns):
            tempmat = hloc[:, :, js]
            w, v = Common.HermitianEigenCmplx(tempmat)
            gfreq = np.zeros((nfreq, norb), dtype=np.complex128, order='F')

            for iorb in range(norb):
                gfreq[:, iorb] = Bare.FFreq(freq, w[iorb])

            for ifreq in range(nfreq):
                for jorb in range(norb):
                    for iorb in range(norb):
                        tempmat2[iorb, jorb] = v[iorb, jorb]*gfreq[ifreq, jorb]
            
                gloc[:, :, js, ifreq] = tempmat2@(np.conjugate(v.T))

        return gloc
    
    @staticmethod
    def FLatFreq(freq : np.ndarray, hlatt : np.ndarray) -> np.ndarray:
        """
        Calculates the lattice fermionic Green's function in frequency domain from a lattice Hamiltonian.

        Args:
            freq (np.ndarray): Array of fermionic Matsubara frequencies.
            hlatt (np.ndarray): The lattice Hamiltonian (norb, norb, ns, nk).

        Returns:
            np.ndarray: The lattice Green's function G_latt(k, iω_n) (norb, norb, ns, nk, nfreq).
        """

        norb, _, ns, nk = hlatt.shape
        nfreq = len(freq)
        glatt = np.zeros((norb, norb, ns, nk, nfreq),dtype=np.complex128, order='F')

        for ik in range(nk):
            glatt[..., ik, :] = Bare.FLocFreq(freq, hlatt[..., ik])
        
        return glatt
    
    @staticmethod
    def FLocTau(tau : np.ndarray, beta : np.float64,  hloc : np.ndarray) -> np.ndarray:
        """
        Calculates the local fermionic Green's function in imaginary time from a local Hamiltonian.

        Args:
            tau (np.ndarray): Array of imaginary time points.
            hloc (np.ndarray): The local Hamiltonian matrix (norb, norb, ns).

        Returns:
            np.ndarray: The local Green's function G_loc(τ) (norb, norb, ns, ntau).
        """
        
        norb, _, ns = hloc.shape
        ntau = len(tau)
        tempmat = np.zeros((norb, norb), dtype=np.complex128, order='F')
        tempmat2 = np.zeros((norb, norb), dtype=np.complex128, order='F')
        gloc = np.zeros((norb, norb, ns, ntau), dtype=np.complex128, order='F')

        for js in range(ns):
            tempmat = hloc[:, :, js]
            w, v = Common.HermitianEigenCmplx(tempmat)
            gtau = np.zeros((ntau, norb), dtype=np.complex128, order='F')

            for iorb in range(norb):
                gtau[:, iorb] = Bare.FTau(tau, beta, w[iorb])
                
            for itau in range(ntau):
                for jorb in range(norb):
                    for iorb in range(norb):
                        tempmat2[iorb, jorb] = v[iorb, jorb]*gtau[itau, jorb]
                
                gloc[:, :, js, itau] = tempmat2@(np.conjugate(v.T))
        
        return gloc
    
    @staticmethod
    def FLatTau(tau : np.ndarray, beta : np.float64, hlatt : np.ndarray) -> np.ndarray:
        """
        Calculates the lattice fermionic Green's function in imaginary time from a lattice Hamiltonian.

        Args:
            tau (np.ndarray): Array of imaginary time points.
            hlatt (np.ndarray): The lattice Hamiltonian (norb, norb, ns, nk).

        Returns:
            np.ndarray: The lattice Green's function G_latt(k, τ) (norb, norb, ns, nk, ntau).
        """

        norb, _, ns, nk = hlatt.shape
        ntau = len(tau)
        glatt = np.zeros((norb, norb, ns, nk, ntau),dtype=np.complex128, order='F')

        for ik in range(nk):
            glatt[..., ik, :] = Bare.FLocTau(tau, beta, hlatt[..., ik])

        return glatt

    @staticmethod
    def BLocFreq(freq : np.ndarray, hloc : np.ndarray) -> np.ndarray:
        """
        Calculates the local bosonic Green's function in frequency domain from a local Hamiltonian.

        Args:
            freq (np.ndarray): Array of bosonic Matsubara frequencies.
            hloc (np.ndarray): The local Hamiltonian matrix (norb, norb, ns, ns).

        Returns:
            np.ndarray: The local Green's function W_loc(iΩ_n) (norb, norb, ns, ns, nfreq).
        """

        norb, _, ns, _ = hloc.shape
        nfreq = len(freq)
        tempmat = np.zeros((norb, norb), dtype=np.complex128, order='F')
        tempmat2 = np.zeros((norb, norb), dtype=np.complex128, order='F')
        wloc = np.zeros((norb, norb, ns, ns, nfreq), dtype=np.complex128, order='F')

        for ks in range(ns):
            for js in range(ns):
                tempmat[:, :, js, ks] = hloc[:, :, js, ks]
                w, v = Common.HermitianEigenCmplx(tempmat)
                wfreq = np.zeros((nfreq, norb), dtype=np.complex128, order='F')

                for iorb in range(norb):
                    wfreq[:, iorb] = Bare.BFreq(freq, w[iorb])
                
                for ifreq in range(nfreq):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            tempmat2[iorb, jorb] = v[iorb, jorb]*wfreq[ifreq, jorb]
                    
                    wloc[:, :, js, ks, ifreq] = tempmat2@(np.conjugate(v.T))
        
        return wloc
    
    @staticmethod
    def BLatFreq(freq : np.ndarray, hlatt : np.ndarray) -> np.ndarray:
        """
        Calculates the lattice bosonic Green's function in frequency domain from a lattice Hamiltonian.

        Args:
            freq (np.ndarray): Array of bosonic Matsubara frequencies.
            hlatt (np.ndarray): The lattice Hamiltonian (norb, norb, ns, ns, nk).

        Returns:
            np.ndarray: The lattice Green's function W_latt(k, iΩ_n) (norb, norb, ns, ns, nk, nfreq).
        """

        norb, _, ns, _, nk = hlatt.shape
        nfreq = len(freq)
        wlatt = np.zeros((norb, norb, ns, ns, nk, nfreq), dtype=np.complex128, order='F')

        for ik in range(nk):
            wlatt[..., ik, :] = Bare.BLocFreq(freq, hlatt[..., ik])

        return wlatt
    
    @staticmethod
    def BLocTau(tau : np.ndarray, hloc : np.ndarray) -> np.ndarray:
        """
        Calculates the local bosonic Green's function in imaginary time from a local Hamiltonian.

        Args:
            tau (np.ndarray): Array of imaginary time points.
            hloc (np.ndarray): The local Hamiltonian matrix (norb, norb, ns, ns).

        Returns:
            np.ndarray: The local Green's function W_loc(τ) (norb, norb, ns, ns, ntau).
        """

        norb, _, ns, _ = hloc.shape
        ntau = len(tau)
        tempmat = np.zeros((norb, norb), dtype=np.complex128, order='F')
        tempmat2 = np.zeros((norb, norb), dtype=np.complex128, order='F')
        wloc = np.zeros((norb, norb, ns, ns, ntau), dtype=np.complex128, order='F')

        for ks in range(ns):
            for js in range(ns):
                tempmat = hloc[:, :, js, ks]
                w, v = Common.HermitianEigenCmplx(tempmat)
                wtau = np.zeros((ntau, norb), dtype=np.complex128, order='F')

                for iorb in range(norb):
                    wtau[:, iorb] = Bare.BTau(tau, w[iorb])
                
                for itau in range(ntau):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            tempmat2[iorb, jorb] = v[iorb, jorb]*wtau[itau, jorb]
                    
                    wloc[:, :, js, ks, itau] = tempmat2@(np.conjugate(v.T))
        
        return wloc
    
    @staticmethod
    def BLatTau(tau : np.ndarray, hlatt : np.ndarray) -> np.ndarray:
        """
        Calculates the lattice bosonic Green's function in imaginary time from a lattice Hamiltonian.

        Args:
            tau (np.ndarray): Array of imaginary time points.
            hlatt (np.ndarray): The lattice Hamiltonian (norb, norb, ns, ns, nk).

        Returns:
            np.ndarray: The lattice Green's function W_latt(k, τ) (norb, norb, ns, ns, nk, ntau).
        """

        norb, _, ns, _, nk = hlatt.shape
        ntau = len(tau)
        wlatt = np.zeros((norb, norb, ns, ns, nk, ntau), dtype=np.complex128, order='F')

        for ik in range(nk):
            wlatt[..., ik, :] = Bare.BLocTau(tau, hlatt[..., ik])

        return wlatt