from tempfile import tempdir
import numpy as np
from pydlr import dlr
from .Common import Common


class DLR(object):
    def __init__(self, ft: dict = None) -> object:
        # self.T = ft.get('T',300) #ft['T']
        # self.beta = ft['beta']
        if "T" not in ft:
            self.beta = ft["beta"]
            self.T = 1 / (self.beta * 8.6173303 * 10**-5)

        elif "beta" not in ft:
            self.T = ft["T"]
            self.beta = 1 / (self.T * 8.6173303 * 10**-5)
        else:
            self.T = ft["T"]
            self.beta = ft["beta"]
        self.cutoff = ft.get("cutoff", 10.0)
        self.eps = ft.get("eps", 1e-15)
        self.lambF = (self.beta / np.pi * self.cutoff - 1) / 2
        self.lambB = self.beta * self.cutoff / (2 * np.pi)  
        
        dF = dlr(lamb=self.lambF, eps=self.eps, dense_imfreq=False)
        dB = dlr(lamb=self.lambB, eps=self.eps, xi=1, dense_imfreq=False)
        

        self.dF = dF
        self.dB = dB
        self.tauF = dF.get_tau(self.beta)
        self.tauB = dB.get_tau(self.beta)
        self.omega = dF.get_matsubara_frequencies(self.beta).imag
        self.nu = dB.get_matsubara_frequencies(self.beta).imag

    def TauUniform(self) -> np.ndarray:
        ntau = int((self.beta / np.pi * self.cutoff - 1) / 2) * 2
        tau = np.zeros((ntau), dtype=float, order="F")
        for itau in range(ntau):
            itheta = Common.Ttind(itau, ntau)
            tau[itau] = self.beta / 2.0 * (np.cos(np.pi * (itheta + 0.5) / ntau) + 1.0)

        # tau = np.linspace(0, self.beta, num=ntau)
        # self.tau = tau

        return tau

    def MatsubaraFermionUniform(self, Emax : np.float64 = None, beta : np.float64 = None) -> np.ndarray:
        Emax = self.omega[-1]
        Emin = self.omega[0]
        # nomega = int((self.beta/np.pi*Emax - 1)/2)
        if Emax is None:
            Emax = self.cutoff
        if beta is None:
            beta = self.beta
        nstart = int((beta / np.pi * Emin - 1) / 2) * 0
        nend = int((beta / np.pi * Emax - 1) / 2)
        # omega = np.zeros((2*nomega+1), dtype=np.float64, order='F')
        number = np.arange(nstart, nend)
        omega = []
        for iomega in number:
            omega.append(np.float64(np.pi / beta * (2 * iomega + 1)))
        omega = np.array(omega, dtype=np.float64, order="F")

        return omega

    def MatsubaraBosonUniform(self) -> np.ndarray:
        Emax = self.nu[-1]
        Emin = self.nu[0]

        nstart = int((self.beta / np.pi * Emin) / 2)
        nend = int((self.beta / np.pi * Emax) / 2)

        number = np.arange(nstart, nend)
        nu = []

        for inu in number:
            nu.append(np.float64(np.pi / self.beta * (2 * inu)))

        nu = np.array(nu, dtype=np.float64, order="F")

        return nu

    def FT2F(self, ftau: np.ndarray):
        """
        Input :
        ftau : (norb, norb, ntau) array like

        Output:
        ff : (norb, norb, nfreq) array like
        """
        ntau = len(ftau)
        ftau = ftau.reshape(ntau, 1, 1)
        fxx = self.dF.dlr_from_tau(ftau)
        tempmat = self.dF.matsubara_from_dlr(fxx, beta=self.beta, xi=-1)
        ff = tempmat[:, 0, 0]

        return ff

    def FF2T(self, ff: np.ndarray):
        """
        Input :
        ff : (norb, norb, nfreq) array like

        Output:
        ftau : (norb, norb, ntau) array like
        """
        nfreq = len(ff)
        ff = ff.reshape(nfreq, 1, 1)
        fxx = self.dF.dlr_from_matsubara(ff, beta=self.beta, xi=-1)
        tempmat = self.dF.tau_from_dlr(fxx)
        ftau = tempmat[:, 0, 0]

        return ftau

    def BT2F(self, btau: np.ndarray):
        # print(btau.shape)
        ntau = len(btau)
        btau = btau.reshape(ntau, 1, 1)
        bxx = self.dB.dlr_from_tau(btau)
        tempmat = self.dB.matsubara_from_dlr(bxx, beta=self.beta, xi=+1)
        bf = tempmat[:, 0, 0]

        return bf

    def BF2T(self, bf: np.ndarray):
        # if ((bf.ndim) != 3):
        #     nf = bf.shape[0]
        #     tempmat = np.zeros((nf, 1, 1), dtype=np.complex128)
        #     tempmat[:,0,0] = bf
        #     bf = tempmat

        nfreq = len(bf)
        bf = bf.reshape(nfreq, 1, 1)
        bxx = self.dB.dlr_from_matsubara(bf, beta=self.beta, xi=+1)

        tempmat = self.dB.tau_from_dlr(bxx)
        btau = tempmat[:, 0, 0]

        return btau

    def TauDLR2Uniform(self, ftau: np.ndarray):
        ntau = len(ftau)
        ftau = ftau.reshape(ntau, 1, 1)

        fxx = self.dF.dlr_from_tau(ftau)

        fout = self.dF.eval_dlr_tau(fxx, self.TauUniform(), beta=self.beta)
        # print(fout.shape)

        return fout

    def TauDLR2Points(self, ftau: np.ndarray, tau) -> np.ndarray:
        """
        Evaluate a DLR-sampled imaginary-time function at specific tau values.

        Args:
            ftau: Array sampled on the DLR tau grid.
            tau: Target tau value(s) as a float or array-like.

        Returns:
            np.ndarray: Function values at the requested tau points.
        """
        tau = np.atleast_1d(tau)
        ntau = len(ftau)
        ftau = ftau.reshape(ntau, 1, 1)
        fxx = self.dF.dlr_from_tau(ftau)
        fout = self.dF.eval_dlr_tau(fxx, tau, beta=self.beta)

        return fout[:, 0, 0]

    def TauDLR2Uniform_v2(self, ftau: np.ndarray):
        fxx = self.dF.dlr_from_tau(ftau.T)

        tau = np.linspace(self.beta - 1, self.beta, num=1000)

        fout = (self.dF.eval_dlr_tau(fxx, tau, beta=self.beta)).T

        return fout

    def TauUniform2DLR(self, ftau: np.ndarray):
        # shape = ftau.shape
        tau = self.TauUniform()
        fxx = self.dF.lstsq_dlr_from_tau(tau_i=tau, G_iaa=ftau.T, beta=self.beta)

        fout = (self.dF.tau_from_dlr(G_xaa=fxx)).T

        return fout

    def MatsubaraDLR2Uniform(self, ff: np.ndarray, sign: int = -1):
        nfreq = len(ff)
        ff = ff.reshape(nfreq, 1, 1)
        if sign == -1:
            fxx = self.dF.dlr_from_matsubara(ff, beta=self.beta, xi=sign)
            z = self.MatsubaraFermionUniform() * 1j
            fout = self.dF.eval_dlr_freq(fxx, z, beta=self.beta, xi=sign)
        else:
            fxx = self.dB.dlr_from_matsubara(ff, beta=self.beta, xi=sign)
            z = self.MatsubaraBosonUniform() * 1j
            fout = self.dB.eval_dlr_freq(fxx, z, beta=self.beta, xi=sign)

        return fout

    def T2mT(self, ftau: np.ndarray, tau: np.ndarray = None) -> np.ndarray:
        if tau is None:
            tau = self.tauB
        taum = self.beta - tau

        ntau = len(ftau)
        ftau = ftau.reshape((ntau, 1, 1))
        fout = np.zeros((ntau), dtype=np.complex128, order="F")

        fxx = self.dB.dlr_from_tau(ftau)
        tempmat = self.dB.eval_dlr_tau(fxx, taum, beta=self.beta)
        fout = -tempmat[:, 0, 0]
        # fxx = self.dF.dlr_from_tau(ftau[:, :, js, irk, :].T)
        # fout[:, :, js, irk, :] = -(self.dF.eval_dlr_tau(fxx, taum, self.beta)).T

        return fout
    
    def TauF2TauB(self, ftau : np.ndarray) -> np.ndarray:

        ntau = len(ftau)
        tempmat = np.zeros((ntau, 1, 1), dtype=np.complex128, order='F')
        tempmat[:, 0, 0] = ftau
        fxx = self.dF.dlr_from_tau(tempmat)
        tempmat2 = self.dF.eval_dlr_tau(fxx, self.tauB, self.beta)
        fout = tempmat2[:, 0, 0]

        return fout
    
    def TauB2TauF(self, ftau : np.ndarray) -> np.ndarray:

        ntau = len(ftau)
        tempmat = np.zeros((ntau, 1, 1), dtype=np.complex128, order='F')
        tempmat[:, 0, 0] = ftau
        fxx = self.dB.dlr_from_tau(tempmat)
        tempmat2 = self.dB.eval_dlr_tau(fxx, self.tauF, self.beta)
        fout = tempmat2[:, 0, 0]

        return fout

    # def FDLR2Tau(self, fdlr: np.ndarray) -> np.ndarray:
    #     ftau = self.dF.tau_from_dlr(fdlr)

    #     return ftau

    # def FDLR2Matsubara(self, fdlr: np.ndarray) -> np.ndarray:
    #     ff = self.dF.matsubara_from_dlr(fdlr, beta=self.beta, xi=-1)

    #     return ff

    # def FTau2DLR(self, ftau: np.ndarray) -> np.ndarray:
    #     fdlr = self.dF.dlr_from_tau(ftau)

    #     return fdlr

    # def FMatsubara2DLR(self, ff: np.ndarray) -> np.ndarray:
    #     nfreq = len(ff)
    #     ff = ff.reshape(nfreq, 1, 1)

    #     fdlr = self.dF.dlr_from_matsubara(ff, self.beta, xi=-1)

    #     return fdlr

    # def BDLR2Tau(self, fdlr: np.ndarray) -> np.ndarray:
    #     ftau = self.dB.tau_from_dlr(fdlr)

    #     return ftau

    # def BDLR2Matsubara(self, fdlr: np.ndarray) -> np.ndarray:
    #     ff = self.dB.matsubara_from_dlr(fdlr, beta=self.beta, xi=-1)

    #     return ff

    # def BTau2DLR(self, ftau: np.ndarray) -> np.ndarray:
    #     fdlr = self.dB.dlr_from_tau(ftau)

    #     return fdlr

    # def BMatsubara2DLR(self, ff: np.ndarray) -> np.ndarray:
    #     fdlr = self.dB.dlr_from_matsubara(ff, self.beta, xi=-1)

    #     return fdlr
