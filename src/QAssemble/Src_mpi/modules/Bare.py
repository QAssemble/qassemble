from Common import Common
from sys import exit
import numpy as np

class Bare:

    @staticmethod
    def FFreq(freq : np.ndarray, energy : np.ndarray) -> np.ndarray:

        nfreq = len(freq)
        gfreq = np.ndarray((nfreq), dtype=np.ndarray, order='F')

        for ifreq in range(nfreq):
            gfreq[ifreq] = 1.0/(1j*freq[ifreq] - energy)

        return gfreq
    
    @staticmethod
    def FTau(tau : np.ndarray, energy : np.ndarray) -> np.ndarray:

        ntau = len(tau)
        gtau = np.ndarray((ntau), dtype=np.complex128, order='F')

        pi = np.pi
        beta = tau[0]/(np.cos(pi*(ntau-0.5)/ntau) + 1.0)*2.0
        machep = np.finfo(np.float64).eps
        
        for itau in range(ntau):
            taumod = (tau[itau] % beta)
            unitnum = int(tau[itau] - taumod)/beta
            
            if (taumod < machep):
                unitnum = unitnum-1

            taunew = tau[itau] - beta * unitnum
            
            if (energy > 0):
                gtau[itau] = (-1)**(unitnum+1)*np.exp(-energy*taunew, dtype=np.complex128) \
                    * (1 - 1/(np.exp(energy*beta, dtype=np.complex128) + 1))
            else:
                gtau[itau] = (-1)**(unitnum+1)* np.exp(-energy*(taunew-beta), dtype=np.complex128)\
                    * (1.0/(np.exp(energy*beta, dtype=np.complex128) + 1))

        return gtau

    @staticmethod
    def BFreq(freq : np.ndarray, energy : np.ndarray) -> np.ndarray:

        nfreq = len(freq)
        wfreq = np.zeros((nfreq), dtype=np.complex128, order='F')

        for ifreq in range(nfreq):
            wfreq[ifreq] = 1.0 / (1j * freq[ifreq] - energy)

        return wfreq
    
    @staticmethod
    def BTau(tau : np.ndarray, energy : np.ndarray) -> np.ndarray:

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
                wtau[itau] = -np.exp(-energy*beta, dtype=np.complex128) \
                    * (1.0 - 1.0/(np.exp(energy*beta, dtype=np.complex128) - 1))
            else:
                wtau[itau] = -np.exp(-energy*(taunew-beta), dtype=np.complex128) \
                    * (1.0/np.exp(energy*beta, dtype=np.complex128) - 1)
        return wtau


    @staticmethod
    def FLocFreq(freq : np.ndarray, hloc : np.ndarray)->np.ndarray:

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

        norb, _, ns, nk = hlatt.shape
        nfreq = len(freq)
        glatt = np.zeros((norb, norb, ns, nk, nfreq),dtype=np.complex128, order='F')

        for ik in range(nk):
            glatt[..., ik, :] = Bare.FLocFreq(freq, hlatt[..., ik])
        
        return glatt
    
    @staticmethod
    def FLocTau(tau : np.ndarray, hloc : np.ndarray) -> np.ndarray:
        
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
                gtau[:, iorb] = Bare.FTau(tau, w[iorb])
                
            for itau in range(ntau):
                for jorb in range(norb):
                    for iorb in range(norb):
                        tempmat2[iorb, jorb] = v[iorb, jorb]*gtau[itau, jorb]
                
                gloc[:, :, js, itau] = tempmat2@(np.conjugate(v.T))
        
        return gloc
    
    @staticmethod
    def FLatTau(tau : np.ndarray, hlatt : np.ndarray) -> np.ndarray:

        norb, _, ns, nk = hlatt.shape
        ntau = len(tau)
        glatt = np.zeros((norb, norb, ns, nk, ntau),dtype=np.complex128, order='F')

        for ik in range(nk):
            glatt[..., ik, :] = Bare.FLocTau(tau, hlatt[..., ik])

        return glatt

    @staticmethod
    def BLocFreq(freq : np.ndarray, hloc : np.ndarray) -> np.ndarray:

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

        norb, _, ns, _, nk = hlatt.shape
        nfreq = len(freq)
        wlatt = np.zeros((norb, norb, ns, ns, nk, nfreq), dtype=np.complex128, order='F')

        for ik in range(nk):
            wlatt[..., ik, :] = Bare.BLocFreq(freq, hlatt[..., ik])

        return wlatt
    
    @staticmethod
    def BLocTau(tau : np.ndarray, hloc : np.ndarray) -> np.ndarray:

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

        norb, _, ns, _, nk = hlatt.shape
        ntau = len(tau)
        wlatt = np.zeros((norb, norb, ns, ns, nk, ntau), dtype=np.complex128, order='F')

        for ik in range(nk):
            wlatt[..., ik, :] = Bare.BLocTau(tau, hlatt[..., ik])

        return wlatt
    
