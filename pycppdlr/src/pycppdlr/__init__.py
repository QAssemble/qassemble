"""cppdlr's native kernel convention; this is not a pydlr compatibility layer.

With u=tau/beta and w=beta*E, A_tau=-exp(-u*w)/(1+exp(-w)).
A_F(n,w)=1/(i*(2*n+1)*pi-w); A_B(n,w)=tanh(w/2)/(i*2*n*pi-w).
At n=w=0 the boson kernel is -1/2. Frequency matrices are dimensionless:
physical values contain one factor of beta, applied inside cppdlr. Bosonic tanh
is already included. cppdlr performs all transforms and kernel evaluations;
Python only validates arrays and preserves their trailing axes, without
transpose or conjugation. Time evaluation takes native relative coordinates.
Symmetrization selects paired nodes/poles; it does not enforce real/even data.
Only checkout-based builds with the sibling cppdlr submodule are supported.
"""

import numpy as np

from . import _core
from ._core import ImFreq, ImTime, build_dlr_rf

__all__ = ["Basis", "build_dlr_rf", "ImTime", "ImFreq"]


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
