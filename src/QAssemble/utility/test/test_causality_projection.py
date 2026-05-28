"""Unit tests for matrix-valued CausalityProjection (Han & Choi PRB 104, 115112).

Run with: pytest src/QAssemble/utility/test/test_causality_projection.py -v
Or:       python src/QAssemble/utility/test/test_causality_projection.py
"""
import numpy as np
import pytest

from QAssemble.utility.DLR import DLR
from QAssemble.utility.Common import Common


def _build_dlr(beta: float = 20.0, cutoff: float = 4.0):
    return DLR({"beta": beta, "cutoff": cutoff, "eps": 1e-12})


def _causal_lorentzian_tau(dlr: DLR, eps_pole: float = 0.5, V2: float = 1.0):
    """G(tau) for a single causal pole: G(iw) = V^2 / (iw - eps_pole)."""
    tau = dlr.tauF
    beta = dlr.beta
    if eps_pole >= 0:
        gtau = -V2 * np.exp(-eps_pole * tau) / (1.0 + np.exp(-eps_pole * beta))
    else:
        gtau = -V2 * np.exp(-eps_pole * (tau - beta)) / (1.0 + np.exp(eps_pole * beta))
    return gtau.astype(np.complex128)


def _causal_lorentzian_freq(dlr: DLR, eps_pole: float = 0.5, V2: float = 1.0):
    omega = dlr.omega
    iw = 1j * omega
    return V2 / (iw - eps_pole)


class _DummyFLocDyn:
    """Minimal stub exposing only the bits CausalityProjection needs."""

    def __init__(self, dlr: DLR, ns: int = 1):
        self.dlr = dlr
        self._causality_ops = None

        class _Crystal:
            pass
        c = _Crystal()
        c.ns = ns
        self.crystal = c

    def F2T(self, ff):
        # ff shape: (norb, norb, ns, nfreq)
        norb = ff.shape[0]
        ns = ff.shape[2]
        nfreq = ff.shape[3]
        out = np.zeros(
            (norb, norb, ns, len(self.dlr.tauF)), dtype=np.complex128, order="F"
        )
        for i in range(norb):
            for j in range(norb):
                for js in range(ns):
                    out[i, j, js, :] = self.dlr.FF2T(ff[i, j, js, :])
        return out

    def T2F(self, ftau):
        norb = ftau.shape[0]
        ns = ftau.shape[2]
        ntau = ftau.shape[3]
        out = np.zeros(
            (norb, norb, ns, len(self.dlr.omega)), dtype=np.complex128, order="F"
        )
        for i in range(norb):
            for j in range(norb):
                for js in range(ns):
                    out[i, j, js, :] = self.dlr.FT2F(ftau[i, j, js, :])
        return out


# Attach CausalityProjection methods to the dummy via the real class methods.
from QAssemble.FLocDyn import FLocDyn as _RealFLocDyn

_DummyFLocDyn._causality_get_ops = _RealFLocDyn._causality_get_ops
_DummyFLocDyn.CausalityProjection = _RealFLocDyn.CausalityProjection
_DummyFLocDyn._causality_project_block = _RealFLocDyn._causality_project_block


@pytest.fixture(scope="module")
def dlr():
    return _build_dlr()


@pytest.fixture(scope="module")
def projector(dlr):
    return _DummyFLocDyn(dlr, ns=1)


def test_operators_shape(dlr):
    tau = dlr.tauF
    D2, L4 = Common.CausalityOperators(tau)
    N = len(tau)
    assert D2.shape == (N, N)
    assert L4.shape == (N - 1, N)
    # D2 acting on a linear function should give ~0.
    lin = tau.copy().astype(np.complex128)
    d2_lin = D2 @ lin.real
    assert np.max(np.abs(d2_lin)) < 1e-6


def test_l4_sign_scale_on_dlr_grid(dlr):
    """Lock in D2/L4 signs on the actual non-uniform DLR tau grid."""
    g_tau = _causal_lorentzian_tau(dlr, eps_pole=0.5, V2=1.0).real
    D2, L4 = Common.CausalityOperators(dlr.tauF)
    assert np.max(D2 @ g_tau) <= 1e-8
    assert np.max(L4 @ g_tau) <= 1e-8


def test_idempotency_diagonal(projector, dlr):
    """Already-causal G should be returned essentially unchanged."""
    g_freq_1d = _causal_lorentzian_freq(dlr, eps_pole=0.7, V2=1.0)
    mat = g_freq_1d.reshape(1, 1, 1, -1).astype(np.complex128)
    out = projector.CausalityProjection(
        mat, sign=+1, eps=1e-10, eta=1e-10, fallback="warn"
    )
    # On positive Matsubara, ImG <= 0 for a causal G.
    pos = dlr.omega > 0
    assert np.all(out[0, 0, 0, pos].imag <= 1e-7)
    rel = np.linalg.norm(out - mat) / np.linalg.norm(mat)
    assert rel < 0.5, f"Idempotency relative error {rel}"


def test_idempotency_sigma_sign_negative(projector, dlr):
    """Already-causal dynamic Sigma should be stable with the sign=-1 cone."""
    epsilon = 0.7
    omega = dlr.omega
    sigma_freq = -1.0 / (1j * omega - epsilon)
    mat = sigma_freq.reshape(1, 1, 1, -1).astype(np.complex128)
    out = projector.CausalityProjection(
        mat, sign=-1, eps=1e-10, eta=1e-10, fallback="warn"
    )
    rel = np.linalg.norm(out - mat) / np.linalg.norm(mat)
    assert rel < 5e-3, f"Sigma idempotency relative error {rel}"


def test_acausal_flip_diagonal(projector, dlr):
    """Flipping a few Im G points should be cleaned up."""
    g_freq = _causal_lorentzian_freq(dlr, eps_pole=0.7, V2=1.0)
    corrupted = g_freq.copy()
    for idx in (-1, -3, -5):
        corrupted[idx] = corrupted[idx].real - 1j * corrupted[idx].imag

    mat = corrupted.reshape(1, 1, 1, -1).astype(np.complex128)
    out = projector.CausalityProjection(
        mat, sign=+1, eps=1e-10, eta=1e-10, fallback="warn"
    )
    pos = dlr.omega > 0
    assert np.all(out[0, 0, 0, pos].imag <= 1e-7)


def test_hermiticity_n2_tau(projector, dlr):
    """A 2x2 block must come out tau-space Hermitian: G_{ji}(τ) = conj(G_{ij}(τ)).

    Matsubara-space Hermiticity is G_{ji}(iω) = conj(G_{ij}(-iω)), which the
    asymmetric DLR Matsubara grid cannot express by a simple index swap, so
    we verify the tau-space invariant the algorithm actually enforces.
    """
    N = len(dlr.omega)
    g1 = _causal_lorentzian_freq(dlr, eps_pole=0.5)
    g2 = _causal_lorentzian_freq(dlr, eps_pole=-0.3)
    mat = np.zeros((2, 2, 1, N), dtype=np.complex128)
    mat[0, 0, 0, :] = g1
    mat[1, 1, 0, :] = g2
    mat[0, 1, 0, :] = 0.05 * (g1 + g2)
    mat[1, 0, 0, :] = np.conjugate(mat[0, 1, 0, :])

    out = projector.CausalityProjection(
        mat, sign=+1, eps=1e-10, eta=1e-10, fallback="regularize"
    )
    # Convert back to tau and check Hermitian symmetry on tau grid.
    out_tau = projector.F2T(out)
    diff = np.max(np.abs(out_tau[0, 1, 0, :] - np.conjugate(out_tau[1, 0, 0, :])))
    assert diff < 1e-6, f"tau-Hermiticity violated by {diff}"


def test_round_trip_stability(projector, dlr):
    """project(project(x)) ~ project(x)."""
    g_freq = _causal_lorentzian_freq(dlr, eps_pole=0.5)
    # Mild corruption.
    corrupted = g_freq.copy()
    corrupted[-2] = corrupted[-2].real - 1j * corrupted[-2].imag

    mat = corrupted.reshape(1, 1, 1, -1).astype(np.complex128)
    out1 = projector.CausalityProjection(
        mat, sign=+1, eps=1e-10, eta=1e-10, fallback="warn"
    )
    out2 = projector.CausalityProjection(
        out1, sign=+1, eps=1e-10, eta=1e-10, fallback="warn"
    )
    rel = np.linalg.norm(out2 - out1) / max(np.linalg.norm(out1), 1e-12)
    assert rel < 0.05, f"Round-trip relative drift {rel}"


def test_psd_invariant_n3(projector, dlr):
    """Projected 3x3 blocks should satisfy -sign * L_alpha(Gc) PSD."""
    N = len(dlr.omega)
    mat = np.zeros((3, 3, 1, N), dtype=np.complex128)
    diag = [
        _causal_lorentzian_freq(dlr, eps_pole=0.4, V2=1.0),
        _causal_lorentzian_freq(dlr, eps_pole=0.8, V2=0.8),
        _causal_lorentzian_freq(dlr, eps_pole=1.2, V2=0.6),
    ]
    for i, g in enumerate(diag):
        mat[i, i, 0, :] = g
    mat[1, 1, 0, -2] = mat[1, 1, 0, -2].real - 1j * mat[1, 1, 0, -2].imag
    mat[0, 1, 0, :] = 0.002 * (diag[0] + diag[1])
    mat[1, 0, 0, :] = np.conjugate(mat[0, 1, 0, :])
    mat[1, 2, 0, :] = 0.002j * (diag[1] + diag[2])
    mat[2, 1, 0, :] = np.conjugate(mat[1, 2, 0, :])

    # Use the library default fallback explicitly: this invariant test checks
    # the final projected cone while stricter production call sites raise.
    out = projector.CausalityProjection(
        mat, sign=+1, eps=1e-10, eta=1e-10, fallback="regularize"
    )
    tau = projector.F2T(out)[:, :, 0, :]
    D2, L4 = Common.CausalityOperators(dlr.tauF)
    for rows in (np.eye(len(dlr.tauF)), D2, L4):
        for c_alpha in rows:
            block = -np.einsum("l,ijl->ij", c_alpha, tau)
            block = 0.5 * (block + np.conjugate(block.T))
            assert np.linalg.eigvalsh(block).min() >= -1e-5


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
