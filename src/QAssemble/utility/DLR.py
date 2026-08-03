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

    def MatsubaraBosonUniform(self, Emax : np.float64 = None, beta : np.float64 = None) -> np.ndarray:
        if Emax is None:
            Emax = max(abs(float(self.nu[0])), abs(float(self.nu[-1])))
        if beta is None:
            beta = self.beta
        nend = int(np.ceil(beta * Emax / (2.0 * np.pi)))
        if nend < 0:
            raise ValueError("Cannot build non-negative boson Matsubara grid")
        number = np.arange(nend + 1)
        nu = np.array(2.0 * np.pi / beta * number, dtype=np.float64, order="F")

        return nu

    def MatsubaraBosonUniformFull(self, Emax : np.float64 = None, beta : np.float64 = None) -> np.ndarray:
        nu = self.MatsubaraBosonUniform(Emax=Emax, beta=beta)
        return np.concatenate((-nu[:0:-1], nu))

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

    def MatsubaraDLR2UniformGrid(
        self,
        ff : np.ndarray,
        sign : int = -1,
        *,
        method : str = "interp",
    ) -> np.ndarray:
        """Evaluate DLR Matsubara data on the corresponding uniform grid.

        ``method="interp"`` (default) interpolates the DLR node values directly
        onto the uniform grid, never forming DLR coefficients.  ``method="dlr"``
        keeps the legacy coefficient round trip; see ``MatsubaraDLR2Uniform``
        for why it is no longer the default.

        The Hermitian fold happens *here* rather than in the flattened core,
        because the matrix-valued relation carries a transpose and only this
        level still has the orbital axes: ``G(-iw) = G(iw)^dagger`` for fermions
        (mirroring ``MatsubaraAddNegativeFrequency``) and
        ``F_ij(-nu) = conj(F_ji(+nu))`` over orbital *and* spin axes for bosons
        (mirroring the reverse direction in ``MatsubaraUniformGrid2DLR``).
        """
        ff = self._as_dynamic_spin_matrix(ff) if sign == -1 else self._as_bosonic_dynamic_matrix(ff)
        omega_dlr = self.omega if sign == -1 else self.nu
        nfreq = ff.shape[-1]
        if nfreq != len(omega_dlr):
            raise ValueError(
                f"frequency dimension {nfreq} does not match DLR omega size {len(omega_dlr)}"
            )

        if method == "interp":
            if sign == -1:
                return self._MatsubaraDLR2UniformInterpGrid(ff, omega_dlr)
            return self._MatsubaraDLR2UniformInterpGridBoson(ff, omega_dlr)

        ff_t = np.moveaxis(ff, -1, 0)
        batch = int(np.prod(ff.shape[:-1]))
        ff_2d = np.ascontiguousarray(ff_t).reshape(nfreq, batch)

        out_2d = self.MatsubaraDLR2Uniform(ff_2d, sign=sign, method=method)
        out_2d = np.asarray(out_2d).reshape(out_2d.shape[0], batch)

        nfreq_uniform = out_2d.shape[0]
        out = out_2d.reshape(nfreq_uniform, *ff.shape[:-1])
        out = np.moveaxis(out, 0, -1)

        return np.asfortranarray(out)

    def _MatsubaraDLR2UniformInterpGrid(
        self,
        ff : np.ndarray,
        omega_dlr : np.ndarray,
    ) -> np.ndarray:
        """Fermionic DLR->uniform interpolation with a Hermitian fold.

        ``ff`` is the 4D ``(norb, norb, ns, nfreq)` fermionic array.  The
        negative DLR nodes are mirrored with ``G(-iw) = G(iw)^dagger`` (the
        orbital transpose is why this cannot live in the flattened core), and
        the folded data is then interpolated onto the uniform grid with real and
        imaginary parts treated independently.
        """
        omega_dlr = np.asarray(omega_dlr, dtype=np.float64)

        # Hermitian mirror: conjugate-transpose on the orbital axes.
        mirrored = np.swapaxes(np.conjugate(ff), 0, 1)

        x_all = np.concatenate([omega_dlr, -omega_dlr])
        f_all = np.concatenate([ff, mirrored], axis=-1)

        keep = x_all >= 0.0
        x_all, f_all = x_all[keep], f_all[..., keep]

        order = np.argsort(x_all)
        x_all, f_all = x_all[order], f_all[..., order]

        _, unique_idx = np.unique(np.round(x_all, 12), return_index=True)
        unique_idx = np.sort(unique_idx)
        x_all, f_all = x_all[unique_idx], f_all[..., unique_idx]

        self.omega_uniform = self.MatsubaraFermionUniform()

        batch = int(np.prod(f_all.shape[:-1]))
        f_2d = np.ascontiguousarray(np.moveaxis(f_all, -1, 0)).reshape(
            x_all.size, batch, 1
        )

        out = self._interp_to_grid(
            f_2d, x_all, self.omega_uniform, variable="inverse", kind="cubic"
        )

        out = out.reshape(self.omega_uniform.size, *f_all.shape[:-1])
        return np.asfortranarray(np.moveaxis(out, 0, -1))

    def _MatsubaraDLR2UniformInterpGridBoson(
        self,
        ff : np.ndarray,
        nu_dlr : np.ndarray,
    ) -> np.ndarray:
        """Bosonic DLR->uniform interpolation with a Hermitian fold.

        The bosonic analogue of ``_MatsubaraDLR2UniformInterpGrid``.  Two things
        differ from the fermionic case:

        * the mirror is ``F_ij(-nu) = conj(F_ji(+nu))`` on the 5D
          ``(norb, norb, ns, ns, nfreq)`` array — the transpose covers the spin
          axes as well as the orbital ones, matching the reverse direction in
          ``MatsubaraUniformGrid2DLR``;
        * ``nu=0`` is a real node of both the DLR and the uniform grid, so the
          ``1/nu`` interpolation variable is shifted by ``2*pi/beta`` (one
          bosonic Matsubara spacing).  The shift only has to clear the
          singularity: the result is flat to within a factor 1.5 over a 1000x
          range of offsets, so this is not a tuning knob.
        """
        nu_dlr = np.asarray(nu_dlr, dtype=np.float64)

        # Hermitian mirror: conjugate-transpose on the orbital AND spin axes.
        mirrored = np.conjugate(np.swapaxes(np.swapaxes(ff, 0, 1), 2, 3))

        x_all = np.concatenate([nu_dlr, -nu_dlr])
        f_all = np.concatenate([ff, mirrored], axis=-1)

        keep = x_all >= 0.0
        # np.abs normalises the -0.0 that mirroring the zero node produces.
        x_all, f_all = np.abs(x_all[keep]), f_all[..., keep]

        order = np.argsort(x_all)
        x_all, f_all = x_all[order], f_all[..., order]

        _, unique_idx = np.unique(np.round(x_all, 12), return_index=True)
        unique_idx = np.sort(unique_idx)
        x_all, f_all = x_all[unique_idx], f_all[..., unique_idx]

        nu_uniform = self.MatsubaraBosonUniform()

        batch = int(np.prod(f_all.shape[:-1]))
        f_2d = np.ascontiguousarray(np.moveaxis(f_all, -1, 0)).reshape(
            x_all.size, batch, 1
        )

        out = self._interp_to_grid(
            f_2d,
            x_all,
            nu_uniform,
            variable="inverse",
            kind="cubic",
            offset=2.0 * np.pi / self.beta,
        )

        out = out.reshape(nu_uniform.size, *f_all.shape[:-1])
        return np.asfortranarray(np.moveaxis(out, 0, -1))

    def MatsubaraUniformGrid2DLR(
        self,
        ff : np.ndarray,
        omega : np.ndarray = None,
        sign : int = -1,
    ) -> np.ndarray:
        """Fit uniform Matsubara data and evaluate it on the DLR grid.

        ``omega=None`` means the full signed default grid.  Positive-only input is
        expanded only when the caller passes an explicit non-negative 1D
        ``omega``; no data-length inference is performed.
        """
        if sign == 1:
            ff = self._as_bosonic_dynamic_matrix(ff)
            omega_was_none = omega is None
            omega = self.MatsubaraBosonUniformFull() if omega_was_none else np.asarray(omega, dtype=np.float64)
            if omega.ndim != 1:
                raise ValueError(f"omega must be a 1D Matsubara grid, got {omega.ndim}D")
            if omega.size == 0:
                raise ValueError("omega must be a non-empty 1D Matsubara grid")
            if omega_was_none and ff.shape[-1] != omega.size:
                raise ValueError(
                    f"frequency dimension {ff.shape[-1]} does not match full signed omega length {omega.size}; "
                    "pass an explicit positive omega to enable positive-only expansion"
                )
            if omega.ndim == 1 and omega.size > 0 and np.all(omega >= 0.0):
                istart = 1 if np.isclose(omega[0], 0.0) else 0
                # F_ij(-nu) = conj(F_ji(+nu)): transpose orbital and spin axes,
                # mirroring the fermionic MatsubaraAddNegativeFrequency.
                negative = np.conjugate(
                    np.swapaxes(np.swapaxes(ff[..., istart:][..., ::-1], 0, 1), 2, 3)
                )
                ff = np.concatenate((negative, ff), axis=-1)
                omega = np.concatenate((-omega[istart:][::-1], omega))
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
        omega_was_none = omega is None
        omega = self.MatsubaraFermionUniformFull() if omega_was_none else np.asarray(omega, dtype=np.float64)
        if omega.ndim != 1:
            raise ValueError(f"omega must be a 1D Matsubara grid, got {omega.ndim}D")
        if omega.size == 0:
            raise ValueError("omega must be a non-empty 1D Matsubara grid")
        if omega_was_none and ff.shape[-1] != omega.size:
            raise ValueError(
                f"frequency dimension {ff.shape[-1]} does not match full signed omega length {omega.size}; "
                "pass an explicit positive omega to enable positive-only expansion"
            )

        if omega.ndim == 1 and omega.size > 0 and np.all(omega >= 0.0):
            ff = self.MatsubaraAddNegativeFrequency(ff)
            omega = np.concatenate((-omega[::-1], omega))

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

    def MatsubaraDLR2Uniform(self, ff: np.ndarray, sign: int = -1, *, method: str = "interp"):
        """Evaluate flattened DLR Matsubara data on the uniform grid.

        ``method="dlr"`` solves for DLR coefficients and evaluates the
        expansion.  That path is exact in exact arithmetic but loses the signal
        entirely at production parameters: at ``beta=100``, ``cutoff=300``,
        ``eps=1e-15`` the coefficient recovery has relative error ~1 (recovered
        ``sum|c|`` 19725 against a true 75), and a clean hybridization whose
        ``Re(Delta)*w**2`` is a constant -7.64 comes back swinging between +344
        and +10475.

        ``method="interp"`` (default) interpolates the node values directly and
        never forms coefficients.  Prefer ``MatsubaraDLR2UniformGrid``, which
        applies the Hermitian fold with the correct orbital transpose; this
        flattened core can only mirror element-wise, so it is exact for diagonal
        channels only.

        Bosons interpolate too; ``nu=0`` sits on both grids, so the ``1/nu``
        variable is shifted by one Matsubara spacing (see
        ``_MatsubaraDLR2UniformInterpGridBoson``).
        """
        from scipy.linalg import lu_solve

        if method not in ("interp", "dlr"):
            raise ValueError("method must be 'interp' or 'dlr'")

        if method == "interp":
            ff2 = np.asarray(ff, dtype=np.complex128)
            squeeze = ff2.ndim == 1
            if squeeze:
                ff2 = ff2[:, None]
            if sign == -1:
                nodes, target, offset = self.omega, self.MatsubaraFermionUniform(), 0.0
            else:
                nodes = self.nu
                target = self.MatsubaraBosonUniform()
                offset = 2.0 * np.pi / self.beta
            x_all, f_all = self._fold_to_nonnegative(nodes, ff2)
            out = self._interp_to_grid(
                f_all[:, :, None],
                np.abs(x_all),
                target,
                variable="inverse",
                kind="cubic",
                offset=offset,
            )[:, :, 0]
            return out[:, 0] if squeeze else out

        if sign == -1:
            fxx = self.dF.dlr_from_matsubara(ff, beta=self.beta, xi=sign)
            z = self.MatsubaraFermionUniform() * 1j
            fout = self.dF.eval_dlr_freq(fxx[:, None, None], z, beta=self.beta, xi=sign)
        else:
            fxx = lu_solve((self.dB.dlrmf2cf, self.dB.mf2cfpiv), ff / self.beta)
            corr_shape = (self.dB.bosonic_corr_x.size,) + (1,) * (np.ndim(fxx) - 1)
            fxx = fxx / self.dB.bosonic_corr_x.reshape(corr_shape)
            z = self.MatsubaraBosonUniform() * 1j
            fout = self.dB.eval_dlr_freq(fxx[:, None, None], z, beta=self.beta, xi=sign)

        return fout

    def MatsubaraUniform2DLR(self, ff: np.ndarray, omega: np.ndarray = None, sign: int = -1):
        """Convert full signed uniform Matsubara data to the DLR grid.

        Positive-only Hermitian expansion is a wrapper-level operation, where the
        orbital axes are still available.  This flattened core expects ``omega`` to
        describe every input row directly.
        """
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
        omega_default = self.MatsubaraFermionUniformFull if sign == -1 else self.MatsubaraBosonUniformFull
        omega_was_none = omega is None
        omega = omega_default() if omega_was_none else np.asarray(omega, dtype=np.float64)

        if omega.ndim != 1:
            raise ValueError(f"omega must be a 1D Matsubara grid, got {omega.ndim}D")
        if omega.size == 0:
            raise ValueError("omega must be a non-empty 1D Matsubara grid")

        if ff.shape[0] != omega.size:
            message = f"frequency dimension {ff.shape[0]} does not match omega length {omega.size}"
            if omega_was_none:
                message += "; omega=None uses the default full signed grid"
            raise ValueError(
                message
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
        # rounding in MatsubaraBosonUniformFull can leave marginally outside the
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
        *,
        variable: str = "linear",
        kind: str = "linear",
        offset: float = 0.0,
    ) -> np.ndarray:
        """Interpolate Matsubara data onto a target frequency grid.

        Step 1 of the uniform->DLR pipeline, and also the core of the
        DLR->uniform direction.  ``ff`` has shape ``(nfreq_src, bi, bj)`` on the
        ascending source grid ``x_src``; real and imaginary parts are
        interpolated independently onto ``x_target`` and the result is returned
        **element-wise with no matrix-axis transpose** — the downstream pydlr
        transforms (``dlr_from_matsubara`` / ``matsubara_from_dlr``) apply the
        single ``(a, b)`` transpose.

        ``variable="inverse"`` interpolates in ``u = 1/(w + offset)`` instead of
        ``w``.  A Matsubara tail ``c1/(iw) + c2/(iw)**2`` is a low-order
        polynomial in ``1/w``, so this removes the systematic bias linear
        interpolation would otherwise accumulate across the sparse log-spaced
        DLR nodes.  It requires ``w + offset`` to be strictly positive.
        ``offset=0`` (the default) is right for fermions, whose smallest
        Matsubara frequency is ``pi/beta``; bosons pass ``offset=2*pi/beta`` to
        shift the ``nu=0`` node off the singularity.

        ``kind="cubic"`` upgrades to a cubic spline, which in the ``1/w``
        variable is near-exact for a decaying tail.  It needs at least four
        source points and falls back to linear below that.

        The default arguments reproduce the original plain-linear-in-``w``
        behaviour exactly, which ``MatsubaraUniform2DLR`` depends on.

        The caller guarantees coverage; interpolation still clamps to the
        endpoint value for any node marginally outside ``x_src`` (e.g. the
        bosonic outermost-|nu| node from floor/ceil grid rounding), which for a
        decaying boundary value is acceptable.
        """
        x_src = np.asarray(x_src, dtype=np.float64)
        x_target = np.asarray(x_target, dtype=np.float64)
        _, bi, bj = ff.shape

        if variable not in ("linear", "inverse"):
            raise ValueError("variable must be 'linear' or 'inverse'")
        if kind not in ("linear", "cubic"):
            raise ValueError("kind must be 'linear' or 'cubic'")

        # Clamp the target into the source span before any change of variable:
        # np.interp clamps on its own, but scipy's interp1d raises instead, and
        # the folded DLR grid can start marginally above the first uniform node.
        x_eval = np.clip(x_target, x_src.min(), x_src.max())

        if variable == "inverse":
            # A folded bosonic grid carries a -0.0 at the zero node; adding 0.0
            # normalises the sign so the shifted value is strictly positive.
            s_src, s_eval = x_src + offset + 0.0, x_eval + offset + 0.0
            if s_src.min() <= 0.0 or s_eval.min() <= 0.0:
                raise ValueError(
                    "variable='inverse' requires positive frequencies after the offset"
                )
            u_src, u_eval = 1.0 / s_src, 1.0 / s_eval
        else:
            u_src, u_eval = x_src, x_eval

        # The change of variable can reverse the ordering (1/w descends).
        order = np.argsort(u_src)
        u_src = u_src[order]
        ff = ff[order]

        use_cubic = kind == "cubic" and u_src.size >= 4

        out = np.empty((x_target.size, bi, bj), dtype=np.complex128)
        for ai in range(bi):
            for aj in range(bj):
                col = ff[:, ai, aj]
                if use_cubic:
                    from scipy.interpolate import interp1d

                    re = interp1d(u_src, col.real, kind="cubic", assume_sorted=True)(u_eval)
                    im = interp1d(u_src, col.imag, kind="cubic", assume_sorted=True)(u_eval)
                else:
                    re = np.interp(u_eval, u_src, col.real)
                    im = np.interp(u_eval, u_src, col.imag)
                out[:, ai, aj] = re + 1j * im
        return out

    @staticmethod
    def _fold_to_nonnegative(
        x_nodes: np.ndarray,
        ff: np.ndarray,
    ) -> tuple:
        """Mirror negative Matsubara nodes onto the non-negative axis.

        The DLR Matsubara grid straddles zero while the uniform target grids
        (``MatsubaraFermionUniform`` / ``MatsubaraBosonUniform``) are
        non-negative.  Keeping only the positive branch is not enough: the
        uniform grid is sized from the *signed* extreme (see
        ``MatsubaraFermionUniform``), which is frequently the negative node, so
        it can extend past the largest positive node and the interpolation would
        silently clamp there.

        Folding uses ``f(-w) = conj(f(w))``, which holds exactly for a
        Matsubara function of real data.  Callers that own the orbital axes are
        responsible for the matrix-valued form ``G(-iw) = G(iw)^dagger`` (the
        extra transpose); this helper only applies the conjugate, so it is
        correct as-is for data whose matrix axes have already been handled.

        Returns the ascending folded grid and the matching data.
        """
        x_nodes = np.asarray(x_nodes, dtype=np.float64)
        ff = np.asarray(ff, dtype=np.complex128)

        x_all = np.concatenate([x_nodes, -x_nodes])
        f_all = np.concatenate([ff, np.conjugate(ff)], axis=0)

        keep = x_all >= 0.0
        x_all, f_all = x_all[keep], f_all[keep]

        order = np.argsort(x_all)
        x_all, f_all = x_all[order], f_all[order]

        # A node at exactly w=0 (bosonic) appears twice after folding; so can a
        # symmetric pair.  Keep the first occurrence of each distinct frequency.
        _, unique_idx = np.unique(np.round(x_all, 12), return_index=True)
        unique_idx = np.sort(unique_idx)

        return x_all[unique_idx], f_all[unique_idx]

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
