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

        nstart = int(np.floor((self.beta / np.pi * Emin) / 2))
        nend = int(np.ceil((self.beta / np.pi * Emax) / 2))

        number = np.arange(nstart, nend + 1)
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

    def _as_bosonic_dynamic_matrix(self, mat : np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=np.complex128)
        if mat.ndim != 5:
            raise ValueError(f"bosonic dynamic matrix must be 5D, got {mat.ndim}D")
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
        ff = self._as_dynamic_spin_matrix(ff) if sign == -1 else self._as_bosonic_dynamic_matrix(ff)
        omega_dlr = self.omega if sign == -1 else self.nu
        nfreq = ff.shape[-1]
        if nfreq != len(omega_dlr):
            raise ValueError(
                f"frequency dimension {nfreq} does not match DLR omega size {len(omega_dlr)}"
            )

        ff_t = np.moveaxis(ff, -1, 0)
        batch = int(np.prod(ff.shape[:-1]))
        ff_2d = np.ascontiguousarray(ff_t).reshape(nfreq, batch)

        out_2d = self.MatsubaraDLR2Uniform(ff_2d, sign=sign)
        if out_2d.ndim == 3:
            out_2d = out_2d[:, :, 0]

        nfreq_uniform = out_2d.shape[0]
        out = out_2d.reshape(nfreq_uniform, *ff.shape[:-1])
        out = np.moveaxis(out, 0, -1)

        return np.asfortranarray(out)

    def MatsubaraUniformGrid2DLR(
        self,
        ff : np.ndarray,
        omega : np.ndarray = None,
        sign : int = -1,
    ) -> np.ndarray:
        """Fit full uniform Matsubara data and evaluate it on the DLR grid."""
        if sign == 1:
            ff = self._as_bosonic_dynamic_matrix(ff)
            omega = self.MatsubaraBosonUniform() if omega is None else np.asarray(omega, dtype=np.float64)
            nfreq = ff.shape[-1]

            block = np.moveaxis(ff, -1, 0)
            block = np.ascontiguousarray(block).reshape(nfreq, int(np.prod(ff.shape[:-1])))
            block_dlr = self.MatsubaraUniform2DLR(block, omega=omega, sign=sign)
            out = block_dlr.reshape(len(self.nu), *ff.shape[:-1])
            out = np.moveaxis(out, 0, -1)

            return np.asfortranarray(out)

        if sign != -1:
            raise ValueError("sign must be -1 for fermions or 1 for bosons")

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

    def BatchBF2T(self, bf_2d: np.ndarray) -> np.ndarray:
        """(nfreq_dlr, batch) boson Matsubara -> (ntau_dlr, batch) tau.

        Batched form of :meth:`BF2T`: solve DLR coefficients on the DLR
        Matsubara grid, apply the bosonic correction (pydlr's
        ``dlr_from_matsubara`` divide for xi=1, here broadcast over the batch
        axis), then map to the DLR tau grid via ``T_lx``.  Shared core for the
        ``F2T`` methods on ``BLocDyn`` / ``BLatDyn`` so the bosonic correction
        can not diverge between them.
        """
        from scipy.linalg import lu_solve
        bf_2d = np.asarray(bf_2d, dtype=np.complex128)
        G_xaa = lu_solve((self.dB.dlrmf2cf, self.dB.mf2cfpiv), bf_2d / self.beta)
        G_xaa /= self.dB.bosonic_corr_x[:, None]
        return np.tensordot(self.dB.T_lx, G_xaa, axes=(1, 0))

    def BatchBT2F(self, btau_2d: np.ndarray) -> np.ndarray:
        """(ntau_dlr, batch) boson tau -> (nfreq_dlr, batch) Matsubara.

        Batched inverse of :meth:`BatchBF2T` / batched form of :meth:`BT2F`:
        solve DLR coefficients on the DLR tau grid, then map to the DLR
        Matsubara grid via ``T_qx`` with the bosonic correction folded in (the
        multiply that inverts ``BatchBF2T``'s divide).  Shared core for the
        ``T2F`` methods on ``BLocDyn`` / ``BLatDyn``.
        """
        from scipy.linalg import lu_solve
        btau_2d = np.asarray(btau_2d, dtype=np.complex128)
        fxx = lu_solve((self.dB.dlrit2cf, self.dB.it2cfpiv), btau_2d)
        return self.beta * np.tensordot(
            self.dB.T_qx * self.dB.bosonic_corr_x[None, :], fxx, axes=(1, 0)
        )

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
        ff = np.asarray(ff, dtype=np.complex128)
        ff_ndim = ff.ndim
        if ff_ndim == 1:
            ff = ff[:, np.newaxis, np.newaxis]
        elif ff_ndim == 2:
            ff = ff[:, :, np.newaxis]
        elif ff_ndim != 3:
            raise ValueError(f"Matsubara data must be 1D, 2D, or 3D, got {ff_ndim}D")

        if sign not in (-1, 1):
            raise ValueError("sign must be -1 for fermions or 1 for bosons")

        omega_dlr = self.omega if sign == -1 else self.nu
        omega_default = self.MatsubaraFermionUniformFull if sign == -1 else self.MatsubaraBosonUniform
        omega = omega_default() if omega is None else np.asarray(omega, dtype=np.float64)

        if ff.shape[0] != len(omega):
            raise ValueError(
                f"frequency dimension {ff.shape[0]} does not match omega length {len(omega)}"
            )

        # np.interp requires an ascending source grid; sort defensively and
        # permute the data identically (the default grids are already ascending,
        # but a caller-supplied omega may not be).
        order = np.argsort(omega)
        x_src = np.asarray(omega, dtype=np.float64)[order]
        ff = ff[order]

        # Coverage check AFTER the sort (so an unsorted caller-supplied omega can
        # not bypass it), for both statistics.  ``np.interp`` would otherwise
        # silently clamp out-of-range DLR nodes to the endpoint value.  The
        # tolerance absorbs the bosonic outermost-|nu| node, which floor/ceil
        # rounding in MatsubaraBosonUniform can leave marginally outside the
        # source range; that single node is then clamped (decaying boundary).
        x_target = np.asarray(omega_dlr, dtype=np.float64)
        scale = max(abs(float(x_target[0])), abs(float(x_target[-1])), 1.0)
        tol = 1.0e-9 * scale
        if x_src[0] > x_target.min() + tol or x_src[-1] < x_target.max() - tol:
            raise ValueError("uniform Matsubara grid does not cover the DLR frequency range")

        # Step 1: linearly interpolate the uniform data onto the DLR Matsubara
        # sampling grid (element-wise, no transpose — the pydlr transforms below
        # own the single (a, b) transpose).
        interp = self._interp_to_grid(ff, x_src, x_target)
        # Step 2: exact DLR coefficients from the interpolated data (square LU
        # solve; row count == DLR rank is guaranteed by interpolating onto the
        # DLR grid).  Inserting the interpolation first stabilises this versus
        # fitting DLR coefficients directly from the raw uniform grid.  pydlr's
        # dlr_from_matsubara wraps scipy lu_solve, which rejects a 3D RHS (and
        # its bosonic_corr divide assumes a 3D shape), so flatten the (a, b)
        # matrix axes to a 2D batch, solve directly, and apply the bosonic
        # correction by hand (mirrors BF2T).
        from scipy.linalg import lu_solve

        d = self.dF if sign == -1 else self.dB
        rank, bi, bj = interp.shape
        fxx2 = lu_solve(
            (d.dlrmf2cf, d.mf2cfpiv),
            interp.reshape(rank, bi * bj) / self.beta,
        )
        if sign == 1:
            fxx2 = fxx2 / d.bosonic_corr_x[:, None]
        fxx = np.asarray(fxx2).reshape(rank, bi, bj)
        # Step 3: re-evaluate the coefficients on the DLR grid (applies the
        # single trailing-axis transpose the ndim-restore rule is written for).
        out = d.matsubara_from_dlr(G_xaa=fxx, beta=self.beta, xi=sign)
        if ff_ndim == 1:
            return out[:, 0, 0]
        if ff_ndim == 2:
            return out[:, 0, :]
        return out

    def _interp_to_grid(
        self,
        ff: np.ndarray,
        x_src: np.ndarray,
        x_target: np.ndarray,
    ) -> np.ndarray:
        """Linearly interpolate Matsubara data onto a target frequency grid.

        Step 1 of the uniform->DLR pipeline.  ``ff`` has shape
        ``(nfreq_src, bi, bj)`` on the ascending source grid ``x_src``; real and
        imaginary parts are interpolated independently onto ``x_target`` and the
        result is returned **element-wise with no matrix-axis transpose** — the
        downstream pydlr transforms (``dlr_from_matsubara`` / ``matsubara_from_dlr``)
        apply the single ``(a, b)`` transpose.

        The caller guarantees coverage; ``np.interp`` still clamps to the
        endpoint value for any node marginally outside ``x_src`` (e.g. the
        bosonic outermost-|nu| node from floor/ceil grid rounding), which for a
        decaying boundary value is acceptable.
        """
        x_target = np.asarray(x_target, dtype=np.float64)
        _, bi, bj = ff.shape

        out = np.empty((x_target.size, bi, bj), dtype=np.complex128)
        for ai in range(bi):
            for aj in range(bj):
                col = ff[:, ai, aj]
                out[:, ai, aj] = np.interp(x_target, x_src, col.real) + 1j * np.interp(
                    x_target, x_src, col.imag
                )
        return out

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
