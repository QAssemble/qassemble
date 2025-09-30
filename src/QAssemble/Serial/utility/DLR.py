import numpy as np
from pydlr import dlr
from .Common import Common

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
        nmax = int((self.beta/np.pi * self.cutoff -1)/2)
        dF = dlr(lamb = self.beta*self.cutoff, eps=self.eps, nmax=nmax, dense_imfreq=True)
        dB = dlr(lamb= self.beta*self.cutoff, eps=self.eps, nmax=nmax, xi=1, dense_imfreq=True)

        self.dF = dF
        self.dB = dB
        self.tau = dF.get_tau(self.beta)
        self.omega = dF.get_matsubara_frequencies(self.beta).imag
        self.nu = dB.get_matsubara_frequencies(self.beta).imag

    def TauUniform(self, ntau : int = 100) -> np.ndarray:
        # return np.linspace(0, self.beta, ntau, dtype=np.float64)
        tau = np.zeros((ntau),dtype=float,order='F')
        for itau in range(ntau):
            itheta = Common.Ttind(itau,ntau)
            tau[itau] = self.beta/2.0*(np.cos(np.pi*(itheta+0.5)/ntau)+1.0)

        # self.tau = tau

        return tau
    
    def MatsubaraFermionUniform(self) -> np.ndarray:
        
        Emax = self.omega[-1]
        Emin = self.omega[0]
        # nomega = int((self.beta/np.pi*Emax - 1)/2)
        nstart = int((self.beta/np.pi * Emin -1)/2)
        nend = int((self.beta/np.pi * Emax -1)/2)
        # omega = np.zeros((2*nomega+1), dtype=np.float64, order='F')
        number = np.arange(nstart, nend)
        omega = []
        for iomega in number:
            omega.append(np.float64(np.pi/self.beta*(2*iomega + 1)))
        omega = np.array(omega, dtype=np.float64, order='F')

        return omega
    
    def MatsubaraBosonUniform(self) -> np.ndarray:
        
        Emax = self.nu[-1]
        Emin = self.nu[0]
        
        nstart = int((self.beta/np.pi * Emin)/2)
        nend = int((self.beta/np.pi * Emax)/2)
        
        number = np.arange(nstart, nend)
        nu = []
        
        for inu in number:
            nu.append(np.float64(np.pi/self.beta * (2 * inu)))
        
        nu = np.array(nu, dtype=np.float64, order='F')

        return nu

    def FT2F(self, ftau : np.ndarray):
        '''
        Input :
        ftau : (norb, norb, ntau) array like

        Output:
        ff : (norb, norb, nfreq) array like
        '''
        
        fxx = self.dF.dlr_from_tau(ftau.T)
        ff = (self.dF.matsubara_from_dlr(fxx, beta=self.beta, xi=-1)).T

        return ff
    
    def FF2T(self, ff : np.ndarray):
        '''
        Input :
        ff : (norb, norb, nfreq) array like

        Output:
        ftau : (norb, norb, ntau) array like
        '''

        fxx = self.dF.dlr_from_matsubara(ff.T, beta=self.beta, xi=-1)
        ftau = (self.dF.tau_from_dlr(fxx)).T

        return ftau
    
    def BT2F(self, btau : np.ndarray):

        bxx = self.dB.dlr_from_tau(btau.T)
        bf = (self.dB.matsubara_from_dlr(bxx, beta=self.beta, xi=+1)).T

        return bf
    
    def BF2T(self, bf : np.ndarray):

        bxx = self.dB.dlr_from_matsubara(bf.T, beta=self.beta, xi=+1)

        btau = (self.dB.tau_from_dlr(bxx)).T

        return btau
    
    def TauDLR2Uniform(self, ftau : np.ndarray, ntau : int = 100):

        fxx = self.dF.dlr_from_tau(ftau.T)

        fout = (self.dF.eval_dlr_tau(fxx, self.TauUniform(ntau), beta=self.beta)).T

        return fout
    
    def MatsubaraDLR2Uniform(self, ff : np.ndarray, sign : int = -1):

        if (sign == -1):
            fxx = self.dF.dlr_from_matsubara(ff.T, beta=self.beta, xi=sign)
            z = self.MatsubaraFermionUniform()*1j
            fout = (self.dF.eval_dlr_freq(fxx, z, beta=self.beta, xi=sign)).T
        else:
            fxx = self.dB.dlr_from_matsubara(ff.T, beta=self.beta, xi=sign)
            z = self.MatsubaraBosonUniform()*1j
            fout = (self.dB.eval_dlr_freq(fxx, z, beta=self.beta, xi=sign)).T

        return fout