import numpy as np
# import scipy
from Common import Common

class Dyson:

    @staticmethod
    def FLocStc(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:

        norb, _, ns = ffin.shape
        ffout = np.zeros((norb, norb, ns), dtype=np.complex128, order='F')

        for js in range(ns):
            tempmat = np.zeros((norb, norb), dtype=np.complex128, order='F')
            tempmat2 = np.zeros((norb, norb), dtype=np.complex128, order='F')
            tempmat = -np.dot(sig[..., js], ffin[..., js])
            
            for iorb in range(norb):
                tempmat[iorb, iorb] += 1.0
            
            tempmat2 = Common.CmplxMatInv(tempmat)

            ffout[..., js] = np.dot(ffin[...,js], tempmat2)

        return ffout
    
    @staticmethod
    def FLatStc(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:

        norb, _, ns, nk = ffin.shape
        ffout = np.zeros((norb, norb, ns, nk),dtype=np.complex128, order='F')

        for ik in range(nk):
            ffout[..., ik] = Dyson.FLocStc(ffin[..., ik], sig[...,ik])
        
        return ffout
    
    @staticmethod
    def FLocDyn(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:

        nfreq = ffin.shape[3]
        ffout = np.zeros_like(ffin, dtype=np.complex128, order='F')

        for ifreq in range(nfreq):
            ffout[...,ifreq] = Dyson.FLocStc(ffin[..., ifreq], sig[..., ifreq])
        
        return ffout
    
    @staticmethod
    def FLatDyn(ffin : np.ndarray, sig : np.ndarray) -> np.ndarray:

        nfreq = ffin.shape[4]
        ffout = np.zeros_like(ffin, dtype=np.complex128, order='F')

        for ifreq in range(nfreq):
            ffout[..., ifreq] = Dyson.FLatStc(ffin[..., ifreq], sig[..., ifreq])

        return ffout