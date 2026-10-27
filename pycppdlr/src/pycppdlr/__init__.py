"""cppdlr's native kernel convention; this is not a pydlr compatibility layer.

With u=tau/beta and w=beta*E, A_tau=-exp(-u*w)/(1+exp(-w)).
A_F(n,w)=1/(i*(2*n+1)*pi-w); A_B(n,w)=tanh(w/2)/(i*2*n*pi-w).
At n=w=0 the boson kernel is -1/2. Frequency matrices are dimensionless:
physical values contain one factor of beta, applied inside cppdlr. Bosonic tanh
is already included. Basis delegates transforms and kernel evaluation to cppdlr;
its Python wrapper validates arrays and preserves trailing axes, without
transpose or conjugation. Time evaluation takes native relative coordinates.
Symmetrization selects paired nodes/poles; it does not enforce real/even data.
PairedBoson supplies a reflection-even kernel basis with native pivoted node
selection and real LAPACK LU solves; NumPy contracts its native kernel matrices.
Only checkout-based builds with the sibling cppdlr submodule are supported.
"""

import numpy as np

from . import _core
from ._core import ImFreq, ImTime, build_dlr_rf

__all__ = ["Basis", "PairedBoson", "build_dlr_rf", "ImTime", "ImFreq"]


def _2d(values, rows):
    values = np.asarray(values)
    if values.ndim == 0 or values.shape[0] != rows:
        raise ValueError(f"first axis must have {rows} rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    return np.ascontiguousarray(values.reshape(rows, -1), dtype=np.complex128), values


def _call(method, values, rows):
    data, original = _2d(values, rows)
    out = method(data)
    out = out.reshape((out.shape[0], *original.shape[1:]))
    return out.real if not np.iscomplexobj(original) and not np.any(out.imag) else out


class Basis:
    """Native cppdlr basis; statistic is 'fermion' or 'boson'.

    it and ifreq retain cppdlr operators and their prepared LU factors.
    rf and native-order t_rel are dimensionless; n contains integer indices.
    u is an absolute-time view FOR DISPLAY ONLY: converting t_rel to u loses
    endpoint precision. Do not use u for sampling or evaluation.
    eval_tau(coefficients, t_rel) takes relative times in [-0.5, 0.5], or +1
    for the right endpoint; data order always follows the native nodes.
    eval_freq(coefficients, n, beta=1) takes integer Matsubara indices.
    All numerical work, including the symmetric-boson least-squares solve,
    is delegated to cppdlr with its own native solver settings.
    """

    def __init__(self, lamb, eps=1e-14, statistic="fermion", symmetrize=False):
        if statistic not in ("fermion", "boson"):
            raise ValueError("statistic must be 'fermion' or 'boson'")
        self.lamb, self.eps = float(lamb), float(eps)
        self.statistic, self.symmetrize = statistic, bool(symmetrize)
        self.rf = build_dlr_rf(self.lamb, self.eps, self.symmetrize)
        self.it = _core.ImTime(self.lamb, self.rf, self.symmetrize)
        self.ifreq = _core.ImFreq(self.lamb, self.rf, statistic == "boson", self.symmetrize)
        self.rank = self.it.rank
        self.t_rel, self.n = self.it.nodes, self.ifreq.nodes
        self.u = np.where(self.t_rel < 0, 1 + self.t_rel, self.t_rel)

    def tau_to_coef(self, values):
        return _call(self.it.vals2coefs, values, self.rank)

    def coef_to_tau(self, coefficients):
        return _call(self.it.coefs2vals, coefficients, self.rank)

    def freq_to_coef(self, values, beta=1.0):
        return _call(lambda data: self.ifreq.vals2coefs(beta, data), values, len(self.n))

    def coef_to_freq(self, coefficients, beta=1.0):
        return _call(lambda data: self.ifreq.coefs2vals(beta, data), coefficients, self.rank)

    def eval_tau(self, coefficients, t_rel):
        t_rel = np.atleast_1d(np.asarray(t_rel))
        if (t_rel.ndim != 1 or np.iscomplexobj(t_rel) or not np.all(np.isfinite(t_rel))
                or np.any(((t_rel < -0.5) | (t_rel > 0.5)) & (t_rel != 1))):
            raise ValueError("t_rel must be finite 1D relative times in [-0.5, 0.5], or +1")
        t_rel = np.ascontiguousarray(t_rel, dtype=np.float64)
        return _call(lambda data: self.it.coefs2eval(data, t_rel), coefficients, self.rank)

    def eval_freq(self, coefficients, n, beta=1.0):
        n = np.atleast_1d(np.asarray(n))
        if (n.ndim != 1 or np.iscomplexobj(n) or not np.all(np.isfinite(n))
                or np.any(n != np.floor(n)) or np.any((n < -1073741824) | (n > 1073741823))):
            raise ValueError("n must be 1D integer indices in [-1073741824, 1073741823]")
        n = np.ascontiguousarray(n, dtype=np.int64)
        return _call(lambda data: self.ifreq.coefs2eval(beta, data, n), coefficients, self.rank)


class PairedBoson:
    """Even boson kernels on cppdlr fine grids, selected by native pivrgs.

    Positive poles use the sum of the +/-pole kernels; zero uses one native
    kernel (-1/2 in time, -beta/2 at n=0). Samples expose both mirror partners,
    but only the independent half is used by square native RealLU solves.
    Complex data is solved componentwise without real/even sample projection.
    """

    statistic = "boson"
    symmetrize = True

    def __init__(self, lamb, eps=1e-14):
        self.lamb, self.eps = float(lamb), float(eps)
        self.rf, t_half, n_half = _core.paired_boson_nodes(self.lamb, self.eps)
        self.rank = self.rf.size
        self.t_rel = np.r_[t_half, -t_half[::-1][t_half[::-1] != .5]]
        self.u = np.where(self.t_rel < 0, 1 + self.t_rel, self.t_rel)
        self.n = np.unique(np.r_[-n_half, n_half])
        self._tau_rows = np.arange(t_half.size)
        self._freq_rows = np.searchsorted(self.n, n_half)
        self._kt = _core.paired_boson_k_tau(self.t_rel, self.rf)
        self._kf = _core.paired_boson_k_freq(self.n, self.rf)
        self._tau_lu = _core.RealLU(np.ascontiguousarray(self._kt[self._tau_rows]))
        self._freq_lu = _core.RealLU(np.ascontiguousarray(self._kf[self._freq_rows]))

    def _solve(self, lu, values, rows, independent):
        _, original = _2d(values, rows)
        rhs = original[independent].reshape(self.rank, -1)
        out = lu.solve(np.ascontiguousarray(rhs.real, dtype=float))
        if np.iscomplexobj(original):
            out = out + 1j * lu.solve(np.ascontiguousarray(rhs.imag, dtype=float))
        return out.reshape(self.rank, *original.shape[1:])

    def _apply(self, kernel, coefficients):
        _, original = _2d(coefficients, self.rank)
        return np.einsum("nx,x...->n...", kernel, original)

    @staticmethod
    def _check_beta(beta):
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")

    def tau_to_coef(self, values):
        return self._solve(self._tau_lu, values, self.t_rel.size, self._tau_rows)

    def coef_to_tau(self, coefficients):
        return self._apply(self._kt, coefficients)

    def freq_to_coef(self, values, beta=1.0):
        self._check_beta(beta)
        return self._solve(self._freq_lu, np.asarray(values) / beta, self.n.size, self._freq_rows)

    def coef_to_freq(self, coefficients, beta=1.0):
        self._check_beta(beta)
        return beta * self._apply(self._kf, coefficients)

    def eval_tau(self, coefficients, t_rel):
        t_rel = np.atleast_1d(np.asarray(t_rel))
        if t_rel.ndim != 1 or np.iscomplexobj(t_rel):
            raise ValueError("t_rel must be 1D real relative times")
        kernel = _core.paired_boson_k_tau(np.ascontiguousarray(t_rel, dtype=float), self.rf)
        return self._apply(kernel, coefficients)

    def eval_freq(self, coefficients, n, beta=1.0):
        self._check_beta(beta)
        n = np.atleast_1d(np.asarray(n))
        if (n.ndim != 1 or np.iscomplexobj(n) or not np.all(np.isfinite(n))
                or np.any(n != np.floor(n)) or np.any((n < -1073741823) | (n > 1073741823))):
            raise ValueError("n must be 1D integer indices in [-1073741823, 1073741823]")
        kernel = _core.paired_boson_k_freq(np.ascontiguousarray(n, dtype=np.int64), self.rf)
        return beta * self._apply(kernel, coefficients)
