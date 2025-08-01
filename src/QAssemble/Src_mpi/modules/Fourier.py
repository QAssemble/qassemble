import numpy as np
from Common import Common
import finufft
from scipy.linalg import solve

class Fourier:

    @staticmethod
    def FLocDynT2F(tau : np.ndarray, ftau : np.ndarray, freq : np.ndarray) -> np.ndarray:

        norb, _, ns, ntau = ftau.shape
        nfreq = len(freq)
        pi = np.pi
        beta = pi/freq[0]
        
        ntau_finu = 2*ntau
        nfreq_finu = 4*nfreq-1
        
        ff = np.zeros((norb, norb, ns, nfreq), dtype=np.complex128, order='F')

        taurad_finu = np.zeros((ntau_finu), dtype=np.float64, order='F')
        for itau in range(ntau):
            # itheta = Common.Ttind(itau, ntau)
            taurad_finu[itau + ntau] = tau[itau] / beta * np.pi
            taurad_finu[itau] = -taurad_finu[itau]
            # taurad_finu[itheta] = -taurad_finu[itau]

        for iorb in range(norb):
            for jorb in range(norb):
                for js in range(ns):
                    ftau_finu = np.zeros((ntau_finu), dtype=np.complex128, order='F')
                    ff_finu = np.zeros((nfreq_finu), dtype=np.complex128, order='F')
                    for itau in range(ntau):
                        ftau_finu[itau + ntau] = ftau[iorb, jorb, js, itau] * \
                                    np.sqrt(tau[itau] * (beta - tau[itau])) * pi / ntau
                        ftau_finu[itau] = -ftau_finu[itau]

                    ff_finu = finufft.nufft1d1(taurad_finu, ftau_finu, nfreq_finu, isign=1, eps=1e-12, nthreads=1)

                    for ifreq in range(nfreq*2):
                        if (ifreq %2 == 1):
                            ff[iorb, jorb, js, (ifreq-1)//2] = ff_finu[ifreq] / 2.0

        return ff
    
    @staticmethod
    def FLatDynT2F(tau : np.ndarray, ftau : np.ndarray, freq : np.ndarray) -> np.ndarray:

        norb, _, ns, nk, _ = ftau.shape
        nfreq = len(freq)
        ff = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')

        for ik in range(nk):
            ff[..., ik, :] = Fourier.FLocDynT2F(tau, ftau[..., ik, :], freq)

        return ff
    
    @staticmethod
    def FLocDynF2T(freq : np.ndarray, ff : np.ndarray, moment : np.ndarray, tau : np.ndarray) -> np.ndarray:

        pi = np.pi
        beta = pi / freq[0]
        
        norb, _, ns, nfreq = ff.shape
        ntau = len(tau)
        ftau = np.zeros((norb, norb, ns, ntau), dtype=np.complex128, order='F')

        ntau_finu = ntau
        nfreq_finu = nfreq*4 - 1

        momega_finu = np.zeros((nfreq_finu, 3), dtype=np.complex128, order='F')
        
        for ifreq in range(-2*nfreq+1, 2*nfreq):
            if (ifreq % 2 ==1):
                momega_finu[ifreq + 2*nfreq -1 , 0] = 1.0/(pi/beta * ifreq * 1j)
                momega_finu[ifreq + 2*nfreq -1 , 1] = 1.0/(pi/beta * ifreq * 1j)**2
                momega_finu[ifreq + 2*nfreq -1 , 2] = 1.0/(pi/beta * ifreq * 1j)**2

        taurad_finu = tau / beta * pi

        mtau_finu = np.zeros((ntau_finu), dtype=np.complex128, order='F')
        for ii in range(3):
            mtau_finu[:, ii] = finufft.nufft1d2(taurad_finu, momega_finu[:, ii], isign=-1, eps=1e-12, nthreads=1)

        for js in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    ff_finu = np.zeros((nfreq_finu), dtype=np.complex128, order='F')

                    for ifreq in range(-2*nfreq+1, 2*nfreq):
                        if (ifreq % 2 == 1):
                            if (ifreq > 0):
                                ff_finu[ifreq + 2*nfreq -1] = ff[iorb, jorb, js, (ifreq-1)//2]
                            else:
                                ff_finu[ifreq + 2*nfreq -1] = np.conjugate(ff[jorb, iorb, js, (-ifreq-1)//2])
                    
                    ftau_finu = finufft.nufft1d2(taurad_finu, ff_finu, isign=-1, eps=1e-12, nthreads=1)

                    for itau in range(ntau):
                        xx = tau[itau] / beta
                        ftau[iorb, jorb, js, itau] = ftau_finu[itau] / beta

                        for ii in range(3):
                            ftau[iorb, jorb, js, itau] -= moment[iorb, jorb, js, ii] * \
                            mtau_finu[itau, ii] / beta
                            ftau[iorb, jorb, js, itau] += 0.5*beta**(ii) / Common.FactorialInt(ii) * (-1) ** (ii + 1) * \
                            Common.EulerPolynomial(xx, ii) * moment[iorb, jorb, js, ii]

        return ftau
    
    @staticmethod
    def FLatDynF2T(freq : np.ndarray, ff : np.ndarray, moment : np.ndarray, tau : np.ndarray) -> np.ndarray:

        norb, _, ns, nk, _ = ff.shape
        ntau = len(tau)
        ftau = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')

        for ik in range(nk):
            ftau[..., ik, :] = Fourier.FLocDynF2T(freq, ff[..., ik, :], moment[..., ik], tau)

        return ftau
    
    @staticmethod
    def FLocDynM(freq : np.ndarray, ff : np.ndarray, isgreen : bool, highzero : bool) -> tuple:

        norb, _, ns, _ = ff.shape
        nfreq = len(freq)

        moment = np.zeros((norb, norb, ns, 3), dtype=np.complex128, order='F')
        high = np.zeros((norb, norb, ns), dtype=np.complex128, order='F')

        ai = 1j

        if (isgreen):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        if (iorb == jorb):
                            moment[iorb, jorb, js, 0] = 1.0
                        else:
                            moment[iorb, jorb, js, 0] = 0.0
                        
                        moment[iorb, jorb, js, 1] = (ff[iorb, jorb, js, nfreq-1] + 
                                                      np.conj(ff[jorb, iorb, js, nfreq-1])) / 2.0 * (freq[nfreq-1] * ai)**2
                        
                        moment[iorb, jorb, js, 2] = (ff[iorb, jorb, js, nfreq-1] - 
                                                      np.conj(ff[jorb, iorb, js, nfreq-1]) - 
                                                      moment[iorb, jorb, js, 0] * 2.0 / (freq[nfreq-1] * ai)) / 2.0 * (freq[nfreq-1] * ai)**3
        else:
            if (highzero):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            moment[iorb, jorb, js , 0] = (ff[iorb, jorb, js, nfreq-1] - 
                                                          np.conjugate(ff[jorb, iorb, js, nfreq-1])) / 2.0 * (freq[nfreq-1] * ai)
                            
                            moment[iorb, jorb, js, 1] = (ff[iorb, jorb, js, nfreq-1] + 
                                                         np.conjugate(ff[jorb, iorb, js, nfreq-1])) / 2.0 * (freq[nfreq-1] * ai)**2
                            
            else:

                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            amat = np.zeros((4, 4), dtype=np.complex128, order='F')
                            bmat = np.zeros((4, 1), dtype=np.complex128, order='F')

                            amat[0, :] = [1.0, 1.0/(freq[nfreq-1]*ai), 1.0/(freq[nfreq-1]*ai)**2, 1.0/(freq[nfreq-1]*ai)**3]
                            amat[1, :] = [1.0, -1.0/(freq[nfreq-1]*ai), 1.0/(freq[nfreq-1]*ai)**2, -1.0/(freq[nfreq-1]*ai)**3]
                            amat[2, :] = [1.0, 1.0/(freq[nfreq-2]*ai), 1.0/(freq[nfreq-2]*ai)**2, 1.0/(freq[nfreq-2]*ai)**3]
                            amat[3, :] = [1.0, -1.0/(freq[nfreq-2]*ai), 1.0/(freq[nfreq-2]*ai)**2, -1.0/(freq[nfreq-2]*ai)**3]

                            bmat[0, 0] = ff[iorb, jorb, js, nfreq-1]
                            bmat[1, 0] = np.conjugate(ff[jorb, iorb, js, nfreq-1])
                            bmat[2, 0] = ff[iorb, jorb, js, nfreq-2]
                            bmat[3, 0] = np.conjugate(ff[jorb, iorb, js, nfreq-2])

                            sol = solve(amat, bmat)

                            high[iorb, jorb, js] = sol[0, 0]
                            moment[iorb, jorb, js, 0] = sol[1, 0]
                            moment[iorb, jorb, js, 1] = sol[2, 0]
                            moment[iorb, jorb, js, 2] = sol[3, 0]

        for js in range(ns):
            high[:, :, js] = (high[:, :, js].T.conj() + high[:, :, js]) / 2.0
            for ii in range(3):
                moment[:, :, js, ii] = (moment[:, :, js, ii].T.conj() + moment[:, :, js, ii]) / 2.0

        return moment, high