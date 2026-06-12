"""Tests for FLatDyn.AnalyticContinuation (DLR -> real-axis continuation).

Run directly with:  python test_analytic_continuation.py
"""
import numpy as np
import pytest

from QAssemble.utility.DLR import DLR
from QAssemble.FLatDyn import FLatDyn as _RealFLatDyn


def _build_dlr(beta: float = 20.0, cutoff: float = 4.0):
    return DLR({"beta": beta, "cutoff": cutoff, "eps": 1e-12})


class _DummyFLatDyn:
    """Minimal stand-in: AnalyticContinuation only touches self.dlr."""

    def __init__(self, dlr):
        self.dlr = dlr

    # Borrow the real method under test.
    AnalyticContinuation = _RealFLatDyn.AnalyticContinuation


@pytest.fixture(scope="module")
def dlr():
    return _build_dlr()


def _single_pole_matsubara(dlr, e0):
    """G(iv) = 1/(iv - e0) on the DLR Matsubara grid, shaped (1,1,1,1,nfreq)."""
    wn = dlr.dF.get_matsubara_frequencies(dlr.beta)  # complex i*nu
    giw = 1.0 / (wn - e0)
    return giw.reshape(1, 1, 1, 1, -1)


def test_single_pole_recovery(dlr):
    """Well-resolved pole at e0=0.5: peak at e0, A>=0, integral ~ 1."""
    fl = _DummyFLatDyn(dlr)
    e0 = 0.5
    mat = _single_pole_matsubara(dlr, e0)

    wreal = np.linspace(-3.0, 3.0, 601)
    gret, akf = fl.AnalyticContinuation(mat, wreal, eta=0.05)

    a = akf[0, 0, 0, 0, :]

    # Peak lands at the pole energy (within grid spacing).
    dw = wreal[1] - wreal[0]
    peak = a.max()
    assert abs(wreal[np.argmax(a)] - e0) <= 2 * dw

    # Spectral function is positive-dominated. Bare DLR continuation rings
    # slightly negative in the tails (a few % of the peak); we only require
    # no large negative excursion, which would signal a sign error.
    assert a.min() >= -0.05 * peak

    # Spectral weight integrates to ~1 (single unit-residue pole).
    trapz = getattr(np, "trapezoid", np.trapz)
    assert trapz(a, wreal) == pytest.approx(1.0, abs=2e-2)


def test_return_shapes_and_dtypes(dlr):
    """Output contract: (norb,norb,ns,nk,nw), gret complex, akf real."""
    fl = _DummyFLatDyn(dlr)
    norb, ns, nk = 2, 1, 3
    nfreq = len(dlr.omega)
    mat = (np.random.randn(norb, norb, ns, nk, nfreq)
           + 1j * np.random.randn(norb, norb, ns, nk, nfreq))

    wreal = np.linspace(-2.0, 2.0, 51)
    gret, akf = fl.AnalyticContinuation(mat, wreal, eta=0.1)

    assert gret.shape == (norb, norb, ns, nk, len(wreal))
    assert akf.shape == (norb, norb, ns, nk, len(wreal))
    assert np.iscomplexobj(gret)
    assert not np.iscomplexobj(akf)
    # akf is exactly -Im gret / pi.
    np.testing.assert_allclose(akf, -gret.imag / np.pi)


def test_input_validation(dlr):
    fl = _DummyFLatDyn(dlr)
    wreal = np.linspace(-1.0, 1.0, 5)

    # Wrong ndim.
    with pytest.raises(ValueError):
        fl.AnalyticContinuation(np.zeros((1, 1, 1, 1), dtype=complex), wreal)

    # Wrong DLR frequency count.
    bad = np.zeros((1, 1, 1, 1, len(dlr.omega) + 1), dtype=complex)
    with pytest.raises(ValueError):
        fl.AnalyticContinuation(bad, wreal)


def test_matsubara_roundtrip(dlr):
    """Guard against a pydlr API/shape change: evaluating the fitted DLR
    coefficients back on the Matsubara nodes must reproduce the input."""
    e0 = 0.5
    mat = _single_pole_matsubara(dlr, e0)[0, 0, 0, 0, :]  # (nfreq,)

    coeffs = dlr.dF.dlr_from_matsubara(mat[:, None], dlr.beta, xi=-1)
    wn = dlr.dF.get_matsubara_frequencies(dlr.beta)
    recon = dlr.dF.eval_dlr_freq(coeffs[:, :, None], wn, dlr.beta, xi=-1).reshape(-1)

    assert np.abs(recon - mat).max() < 1e-10


@pytest.mark.xfail(
    reason="Known limitation: poles poorly captured by the sparse DLR Matsubara "
           "nodes (here e0=-1.0) are not faithfully continued; noisy/edge poles "
           "need MaxEnt/Nevanlinna. Recorded so the limitation is not silently "
           "broken.",
    strict=False,
)
def test_poorly_resolved_pole_known_limitation(dlr):
    fl = _DummyFLatDyn(dlr)
    e0 = -1.0
    mat = _single_pole_matsubara(dlr, e0)

    wreal = np.linspace(-3.0, 3.0, 601)
    _, akf = fl.AnalyticContinuation(mat, wreal, eta=0.05)
    a = akf[0, 0, 0, 0, :]

    dw = wreal[1] - wreal[0]
    assert abs(wreal[np.argmax(a)] - e0) <= 2 * dw
    assert (a >= -1e-6).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
