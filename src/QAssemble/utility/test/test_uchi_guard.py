"""Tests for the Uchi stability guard on the bosonic Dyson solves.

``BLocDyn._UChiGuard`` flags frequencies where ``max|eig(sigma g0)| >= 1``
(the EDMFT charge-instability boundary, where ``1 - sigma g0`` is singular
and the Dyson solve amplifies CTQMC noise without bound).  ``PImp.Cal``,
``BWeiss.Cal`` and ``WImp.Cal`` substitute the previous iteration's value at
the flagged frequencies; on the first iteration (no cache) they warn and
keep the computed value.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from QAssemble.BLocDyn import BLocDyn, PImp, BWeiss, WImp


class _FakeDLR:
    """Minimal DLR stub with same-size uniform and DLR grids, so the
    DLR<->uniform conversions in ``Cal`` are identity maps (same idiom as
    test_pimp_sign_convention.py)."""

    def __init__(self, nfreq):
        self.beta = 1.0
        self.nu = np.arange(nfreq, dtype=float)
        self.tauB = np.arange(nfreq, dtype=float)

    def MatsubaraBosonUniform(self):
        return self.nu

    def MatsubaraDLR2UniformGrid(self, value, sign=1):
        return np.asarray(value, dtype=np.complex128)

    def BatchBF2T(self, bf_2d):
        return np.asarray(bf_2d, dtype=np.complex128)


class _FakeProjector:
    def __init__(self, norbb):
        self.bprojector = {"1": np.ones((1, norbb, norbb), dtype=float)}
        self.equiv = {"1": np.eye(norbb, dtype=int)}
        self.blocal2pair = {"1": [{i: (i, i) for i in range(norbb)}]}

    def ProbBorb2FPair(self, key, iorbc, ispace=0):
        return (int(iorbc), int(iorbc))


def _scalar_series(values):
    """(1, 1, 1, 1, nfreq) array from a list of per-frequency scalars."""
    arr = np.asarray(values, dtype=np.complex128)
    return np.asfortranarray(arr.reshape(1, 1, 1, 1, -1))


def _guard_host():
    obj = BLocDyn.__new__(BLocDyn)
    obj.key = "1"
    obj.iteration = 3
    return obj


def test_uchi_guard_flags_boundary_crossing():
    obj = _guard_host()
    g0 = _scalar_series([1.0, 1.0, 1.0])
    sigma = _scalar_series([1.2, 0.5, 0.5])

    with pytest.warns(RuntimeWarning, match=r"\[uchi-guard\]"):
        bad = obj._UChiGuard(g0, sigma, "pimp")

    assert bad.tolist() == [0]


def test_uchi_guard_quiet_when_stable(recwarn, caplog):
    obj = _guard_host()
    g0 = _scalar_series([1.0, 1.0, 1.0])
    sigma = _scalar_series([0.5, 0.5, 0.5])

    with caplog.at_level(logging.WARNING, logger="QAssemble"):
        bad = obj._UChiGuard(g0, sigma, "pimp")

    assert bad.size == 0
    assert len(recwarn) == 0
    assert not any("[uchi-guard]" in rec.message for rec in caplog.records)


def test_uchi_guard_logs_large_condition_number(recwarn, caplog):
    # max|lambda| = 0.999 < 1: no trip, but cond(1 - m) = 0.9/0.001 = 900
    # must be surfaced as a diagnostic log line.  A 1x1 block always has
    # cond == 1, so this needs a genuine 2x2 spread of eigenvalues.
    obj = _guard_host()
    norb, nfreq = 2, 2
    g0 = np.zeros((norb, norb, 1, 1, nfreq), dtype=np.complex128)
    sigma = np.zeros_like(g0)
    for i in range(norb):
        g0[i, i, 0, 0, :] = 1.0
    sigma[0, 0, 0, 0, :] = [0.999, 0.1]
    sigma[1, 1, 0, 0, :] = [0.1, 0.1]

    with caplog.at_level(logging.WARNING, logger="QAssemble"):
        bad = obj._UChiGuard(g0, sigma, "bweiss")

    assert bad.size == 0
    assert len(recwarn) == 0
    assert any("[uchi-guard]" in rec.message for rec in caplog.records)


def test_uchi_guard_multiorbital_layout():
    # Non-commuting 2x2 blocks: the guard's batched reshape must reproduce a
    # per-frequency loop over sigma @ g0 in the (norb*ns) composite basis.
    rng = np.random.default_rng(7)
    norb, ns, nfreq = 2, 1, 4
    g0 = rng.normal(size=(norb, norb, ns, ns, nfreq)) + 0.1j * rng.normal(
        size=(norb, norb, ns, ns, nfreq)
    )
    sigma = rng.normal(size=(norb, norb, ns, ns, nfreq)) + 0.1j * rng.normal(
        size=(norb, norb, ns, ns, nfreq)
    )

    dim = norb * ns
    expected = []
    for ifreq in range(nfreq):
        g0_mat = g0[..., ifreq].transpose(0, 2, 1, 3).reshape(dim, dim)
        sigma_mat = sigma[..., ifreq].transpose(0, 2, 1, 3).reshape(dim, dim)
        lam = np.abs(np.linalg.eigvals(sigma_mat @ g0_mat)).max()
        expected.append(lam >= 1.0)
    expected_bad = np.flatnonzero(expected)
    assert expected_bad.size > 0, "fixture must trip at least one frequency"
    assert expected_bad.size < nfreq, "fixture must leave some frequencies stable"

    obj = _guard_host()
    with pytest.warns(RuntimeWarning, match=r"\[uchi-guard\]"):
        bad = obj._UChiGuard(g0, sigma, "wimp")

    np.testing.assert_array_equal(bad, expected_bad)


# --------------------------------------------------------------------------
# PImp.Cal wiring


def _build_pimp_guard(chi, utilde, prev):
    """PImp with identity causal projection and a controllable brd cache;
    ``Cal`` is NOT called so tests can wrap it in pytest.warns."""
    norbb = chi.shape[0]
    nfreq = chi.shape[-1]
    obj = PImp.__new__(PImp)
    obj.crystal = SimpleNamespace(ns=1)
    obj.dlr = _FakeDLR(nfreq)
    obj.projector = _FakeProjector(norbb)
    obj.key = "1"
    obj.chi = SimpleNamespace(f_uniform=chi)
    obj.utilde = utilde
    obj.control = {}
    obj.hdf5file = None
    obj.group = None
    obj.subgroup = "PImp"
    obj.iteration = 1
    obj.chi_boson = None
    obj.chi_boson_uniform = None
    obj.utilde_uniform = None
    obj.f = None
    obj.f_uniform = None
    obj.t = None
    obj.CausalProjection = lambda value, **kwargs: np.asfortranarray(value)
    obj.ReadBrdPrev = lambda stem, shape: prev
    return obj


def test_pimp_cal_splices_previous_on_guard_trip():
    # chi*U = [2.0, 0.5, 0.4]: index 0 crosses the instability boundary and
    # the raw solve there gives P = -2/(1-2) = +2, an unphysical positive
    # polarization -- exactly what the guard must replace with the cache.
    chi = _scalar_series([2.0, 0.5, 0.4])
    utilde = _scalar_series([1.0, 1.0, 1.0])
    prev = _scalar_series([-7.0, -8.0, -9.0])
    obj = _build_pimp_guard(chi, utilde, prev)

    with pytest.warns(RuntimeWarning, match=r"\[uchi-guard\]"):
        obj.Cal()

    expected = -chi / (1.0 - chi * utilde)
    np.testing.assert_allclose(obj.f_uniform[..., 0], prev[..., 0])
    np.testing.assert_allclose(
        obj.f_uniform[..., 1:], expected[..., 1:], rtol=1e-12
    )


def test_pimp_cal_warn_only_without_history():
    chi = _scalar_series([2.0, 0.5, 0.4])
    utilde = _scalar_series([1.0, 1.0, 1.0])
    obj = _build_pimp_guard(chi, utilde, prev=None)

    with pytest.warns(RuntimeWarning, match=r"\[uchi-guard\]"):
        obj.Cal()

    # No cache: the computed value survives everywhere, including the
    # flagged frequency.
    expected = -chi / (1.0 - chi * utilde)
    np.testing.assert_allclose(obj.f_uniform, expected, rtol=1e-12)


# --------------------------------------------------------------------------
# BWeiss.Cal wiring


def test_bweiss_cal_splices_previous_on_guard_trip():
    # p*w = [-1.5, ...]: |lambda| trips on the negative side too.  The
    # "bweiss" cache stores the projected correlated bath cf = f - v, so the
    # spliced f must come back as prev_cf + v.
    nfreq = 3
    w = _scalar_series([1.0, 1.0, 1.0])
    p = _scalar_series([1.5, -0.2, -0.1])
    v4 = np.full((1, 1, 1, 1), 0.3, dtype=np.complex128)
    prev_cf = _scalar_series([-0.05, -0.04, -0.03])

    obj = BWeiss.__new__(BWeiss)
    obj.crystal = SimpleNamespace(ns=1)
    obj.dlr = _FakeDLR(nfreq)
    obj.projector = _FakeProjector(1)
    obj.key = "1"
    obj.vloc = SimpleNamespace(vproj={"1": v4})
    obj.w = w
    obj.p = p
    obj.hdf5file = None
    obj.group = None
    obj.subgroup = "BWeiss"
    obj.iteration = 1
    obj.f = None
    obj.cf = None
    obj.f_uniform = None
    obj.cf_uniform = None
    obj.t = None
    obj.ct = None
    obj.CausalProjection = lambda value, **kwargs: np.asfortranarray(value)
    obj.ReadBrdPrev = lambda stem, shape: prev_cf
    obj.WriteBrdPrev = lambda stem, value: None

    with pytest.warns(RuntimeWarning, match=r"\[uchi-guard\]"):
        obj.Cal()

    # Flagged frequency: previous correlated bath plus the static part.
    np.testing.assert_allclose(obj.f[..., 0], prev_cf[..., 0] + v4)
    # Unflagged frequencies: the raw Dyson solve w / (1 + p w).
    expected = w / (1.0 + p * w)
    np.testing.assert_allclose(obj.f[..., 1:], expected[..., 1:], rtol=1e-12)


# --------------------------------------------------------------------------
# WImp.Cal wiring


def test_wimp_cal_splices_previous_and_seeds_cache():
    # WImp had no brd cache before the guard; Cal must now both splice from
    # and seed the "wimp" cache, and the seeded value must be the final
    # (spliced) result so the next iteration falls back to good data.
    nfreq = 3
    utilde = _scalar_series([1.0, 1.0, 1.0])
    polarization = _scalar_series([1.2, -0.3, -0.2])  # |lambda| trips index 0
    prev = _scalar_series([0.7, 0.8, 0.9])
    writes = []

    obj = WImp.__new__(WImp)
    obj.crystal = SimpleNamespace(ns=1)
    obj.dlr = _FakeDLR(nfreq)
    obj.projector = _FakeProjector(1)
    obj.key = "1"
    obj.utilde = utilde
    obj.polarization = polarization
    obj.hdf5file = None
    obj.group = None
    obj.subgroup = "WImp"
    obj.iteration = 1
    obj.f = None
    obj.f_uniform = None
    obj.t = None
    obj.ReadBrdPrev = lambda stem, shape: prev
    obj.WriteBrdPrev = lambda stem, value: writes.append(
        (stem, np.array(value, copy=True))
    )

    with pytest.warns(RuntimeWarning, match=r"\[uchi-guard\]"):
        obj.Cal()

    np.testing.assert_allclose(obj.f[..., 0], prev[..., 0])
    expected = utilde / (1.0 - polarization * utilde)
    np.testing.assert_allclose(obj.f[..., 1:], expected[..., 1:], rtol=1e-12)

    assert len(writes) == 1
    stem, written = writes[0]
    assert stem == "wimp"
    np.testing.assert_allclose(written, obj.f)
