import numpy as np
from pydlr import dlr

class DLR(object):

    def __init__(self, ft : dict = None) -> object:

        #self.T = ft.get('T',300) #ft['T']
        #self.beta = ft['beta']
        if ('T' not in ft):
            self.beta = ft['beta']
            self.T = 1/(self.beta*8.6173303*10**-5)

        elif ('beta' not in ft):
            self.T = ft['T']
            self.beta = 1/(self.T*8.6173303*10**-5)
        else:
            self.T = ft['T']
            self.beta = ft['beta']
        self.cutoff = ft.get('cutoff',10.0)
        self.eps = ft.get('eps',1e-8)
        
        dF = dlr(lamb = self.beta*self.cutoff, eps=self.eps, nmax=self.cutoff*self.beta)
        dB = dlr(lamb= self.beta*self.cutoff, eps=self.eps, nmax=self.cutoff*self.beta, xi=1)

        self.dF = dF
        self.dB = dB
        self.tau = dF.get_tau(self.beta)
        self.omega = dF.get_matsubara_frequencies(self.beta).imag
        self.nu = dB.get_matsubara_frequencies(self.beta).imag

    def TauUniform(self, ntau : int = 100) -> np.ndarray:
        return np.linspace(0, self.beta, ntau, dtype=np.float64)
    
    def MatsubaraFermionUniform(self) -> np.ndarray:
        
        cutoff = self.omega[-1].imag
        nomega = int((self.beta/np.pi*cutoff - 1)/2)
        omega = np.zeros((nomega), dtype=np.complex128, order='F')
        for iomega in range(nomega):
            omega[iomega] = 1j*np.pi/self.beta*(2*iomega+1)

        return omega
    
    def MatsubaraBosonUniform(self) -> np.ndarray:
        
        cutoff = self.nu[-1].imag
        nnu = int((self.beta/np.pi*cutoff)/2)
        nu = np.zeros((nnu), dtype=np.complex128, order='F')
        for inu in range(nnu):
            nu[inu] = 1j*2*np.pi/self.beta*inu

        return nu
