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
        self.lambB = self.beta * self.cutoff / (2 * np.pi) * 2
        
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
        if Emax is None:
            Emax = max(abs(float(self.omega[0])), abs(float(self.omega[-1])))
        if beta is None:
            beta = self.beta
        nend = int(np.floor((beta / np.pi * Emax - 1.0) / 2.0))
        if nend < 0:
            raise ValueError("Cannot build non-negative fermion Matsubara grid")
        number = np.arange(nend + 1)
        omega = np.array(np.pi / beta * (2*number + 1), dtype=np.float64, order="F")

        return omega

    def MatsubaraFermionUniformFull(self, Emax : np.float64 = None, beta : np.float64 = None) -> np.ndarray:
        omega = self.MatsubaraFermionUniform(Emax=Emax, beta=beta)
        return np.concatenate((-omega[::-1], omega))

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

    def _as_dynamic_spin_matrix(self, mat : np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=np.complex128)
        if mat.ndim == 3:
            mat = mat[:, :, np.newaxis, :]
        if mat.ndim != 4:
            raise ValueError(f"dynamic matrix must be 3D or 4D, got {mat.ndim}D")
        return np.asfortranarray(mat)

    def MatsubaraAddNegativeFrequency(self, mat : np.ndarray) -> np.ndarray:
        """Build full fermionic Matsubara data from non-negative frequencies."""
        mat = self._as_dynamic_spin_matrix(mat)
        nfreq = mat.shape[3]

        matout = np.zeros(
            (mat.shape[0], mat.shape[1], mat.shape[2], 2*nfreq),
            dtype=np.complex128,
            order="F",
        )
        matout[..., nfreq:] = mat
        matout[..., :nfreq] = np.swapaxes(np.conjugate(mat[..., ::-1]), 0, 1)

        return matout

    def MatsubaraDLR2UniformGrid(self, ff : np.ndarray, sign : int = -1) -> np.ndarray:
        """Evaluate DLR Matsubara data on the corresponding uniform grid."""
        ff = self._as_dynamic_spin_matrix(ff)
        omega_dlr = self.omega if sign == -1 else self.nu
        nfreq = ff.shape[3]
        if nfreq != len(omega_dlr):
            raise ValueError(
                f"frequency dimension {nfreq} does not match DLR omega size {len(omega_dlr)}"
            )

        ff_t = np.moveaxis(ff, -1, 0)
        batch = ff.shape[0] * ff.shape[1] * ff.shape[2]
        ff_2d = np.ascontiguousarray(ff_t).reshape(nfreq, batch)

        out_2d = self.MatsubaraDLR2Uniform(ff_2d, sign=sign)
        if out_2d.ndim == 3:
            out_2d = out_2d[:, :, 0]

        nfreq_uniform = out_2d.shape[0]
        out = out_2d.reshape(nfreq_uniform, ff.shape[0], ff.shape[1], ff.shape[2])
        out = np.moveaxis(out, 0, -1)

        return np.asfortranarray(out)

    def MatsubaraUniformGrid2DLR(
        self,
        ff : np.ndarray,
        omega : np.ndarray = None,
        sign : int = -1,
    ) -> np.ndarray:
        """Fit full uniform Matsubara data and evaluate it on the DLR grid."""
        if sign != -1:
            raise NotImplementedError("MatsubaraUniformGrid2DLR is currently implemented for fermions only")

        ff = self._as_dynamic_spin_matrix(ff)
        omega = self.MatsubaraFermionUniformFull() if omega is None else np.asarray(omega, dtype=np.float64)

        out = np.zeros(
            (ff.shape[0], ff.shape[1], ff.shape[2], len(self.omega)),
            dtype=np.complex128,
            order="F",
        )
        for js in range(ff.shape[2]):
            block = np.moveaxis(ff[:, :, js, :], -1, 0)
            block_dlr = self.MatsubaraUniform2DLR(block, omega=omega, sign=sign)
            out[:, :, js, :] = np.moveaxis(block_dlr, 0, -1)

        return out

    def FT2F(self, ftau: np.ndarray):
        """1D ftau -> 1D ff (Fermion, tau -> Matsubara)"""
        fxx = self.dF.dlr_from_tau(ftau)
        ff = self.dF.matsubara_from_dlr(fxx, beta=self.beta, xi=-1)
        return ff

    def FF2T(self, ff: np.ndarray):
        """1D ff -> 1D ftau (Fermion, Matsubara -> tau)"""
        fxx = self.dF.dlr_from_matsubara(ff, beta=self.beta, xi=-1)
        ftau = self.dF.tau_from_dlr(fxx)
        return ftau

    def BT2F(self, btau: np.ndarray):
        """1D btau -> 1D bf (Boson, tau -> Matsubara)"""
        from scipy.linalg import lu_solve
        bxx = lu_solve((self.dB.dlrit2cf, self.dB.it2cfpiv), btau)
        bf = self.beta * np.dot(self.dB.T_qx * self.dB.bosonic_corr_x[None, :], bxx)
        return bf

    def BF2T(self, bf: np.ndarray):
        """1D bf -> 1D btau (Boson, Matsubara -> tau)"""
        from scipy.linalg import lu_solve
        bxx = lu_solve((self.dB.dlrmf2cf, self.dB.mf2cfpiv), bf / self.beta)
        bxx /= self.dB.bosonic_corr_x
        btau = np.dot(self.dB.T_lx, bxx)
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
        fxx = self.dF.dlr_from_tau(ftau)
        fout = self.dF.eval_dlr_tau(fxx[:, None, None], tau, beta=self.beta)

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
        from scipy.linalg import lu_solve
        if sign == -1:
            fxx = self.dF.dlr_from_matsubara(ff, beta=self.beta, xi=sign)
            z = self.MatsubaraFermionUniform() * 1j
            fout = self.dF.eval_dlr_freq(fxx[:, None, None], z, beta=self.beta, xi=sign)
        else:
            fxx = lu_solve((self.dB.dlrmf2cf, self.dB.mf2cfpiv), ff / self.beta)
            fxx /= self.dB.bosonic_corr_x
            z = self.MatsubaraBosonUniform() * 1j
            fout = self.dB.eval_dlr_freq(fxx[:, None, None], z, beta=self.beta, xi=sign)

        return fout

    def MatsubaraUniform2DLR(self, ff: np.ndarray, omega: np.ndarray = None, sign: int = -1):
        if sign != -1:
            raise NotImplementedError("MatsubaraUniform2DLR is currently implemented for fermions only")

        ff = np.asarray(ff, dtype=np.complex128)
        omega = self.MatsubaraFermionUniformFull() if omega is None else np.asarray(omega, dtype=np.float64)
        if ff.shape[0] != len(omega):
            raise ValueError(
                f"frequency dimension {ff.shape[0]} does not match omega length {len(omega)}"
            )
        if omega[0] > self.omega[0] or omega[-1] < self.omega[-1]:
            raise ValueError("uniform Matsubara grid does not cover the DLR frequency range")

        fxx = self.dF.lstsq_dlr_from_matsubara(
            w_q=omega * 1j,
            G_qaa=ff,
            beta=self.beta,
        )
        return self.dF.matsubara_from_dlr(G_xaa=fxx, beta=self.beta, xi=-1)

    def T2mT(self, ftau: np.ndarray, tau: np.ndarray = None) -> np.ndarray:
        if tau is None:
            tau = self.tauB
        taum = self.beta - tau

        fxx = self.dB.dlr_from_tau(ftau)
        tempmat = self.dB.eval_dlr_tau(fxx[:, None, None], taum, beta=self.beta)
        fout = -tempmat[:, 0, 0]

        return fout
    
    def TauF2TauB(self, ftau : np.ndarray) -> np.ndarray:
        fxx = self.dF.dlr_from_tau(ftau)
        tempmat = self.dF.eval_dlr_tau(fxx[:, None, None], self.tauB, self.beta)
        return tempmat[:, 0, 0]

    def TauB2TauF(self, ftau : np.ndarray) -> np.ndarray:
        fxx = self.dB.dlr_from_tau(ftau)
        tempmat = self.dB.eval_dlr_tau(fxx[:, None, None], self.tauF, self.beta)
        return tempmat[:, 0, 0]

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
