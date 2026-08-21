"""Denominator diagnostics for the bosonic impurity Weiss field."""

import h5py
import numpy as np
import pytest

from QAssemble.BLocDyn import BWeiss


def _boson_blocks(norb, nfreq=1):
    shape = (norb, norb, 1, 1, nfreq)
    return np.zeros(shape, dtype=np.complex128, order="F")


def test_bweiss_denominator_metrics_well_conditioned():
    w = _boson_blocks(2)
    p = _boson_blocks(2)
    w[:, :, 0, 0, 0] = [[2.0, 0.5], [0.0, 1.0]]
    p[:, :, 0, 0, 0] = [[0.1, 0.0], [0.2, -0.1]]

    smin, smax, cond, frequency, bad = BWeiss._denominator_metrics(w, p)
    expected = np.eye(2) + w[:, :, 0, 0, 0] @ p[:, :, 0, 0, 0]
    singular_values = np.linalg.svd(expected, compute_uv=False)

    np.testing.assert_allclose(smin, singular_values[-1])
    np.testing.assert_allclose(smax, singular_values[0])
    np.testing.assert_allclose(cond, singular_values[0] / singular_values[-1])
    assert frequency == 0
    assert bad is False


def test_bweiss_denominator_metrics_detects_smin_floor():
    w = _boson_blocks(1, nfreq=2)
    p = _boson_blocks(1, nfreq=2)
    w[0, 0, 0, 0, :] = 1.0
    p[0, 0, 0, 0, :] = [0.0, -0.995]

    smin, _, _, frequency, bad = BWeiss._denominator_metrics(w, p)

    np.testing.assert_allclose(smin, 5.0e-3)
    assert frequency == 1
    assert bad is True


def test_bweiss_denominator_metrics_detects_condition_ceiling():
    w = _boson_blocks(2)
    p = _boson_blocks(2)
    w[:, :, 0, 0, 0] = np.eye(2)
    p[:, :, 0, 0, 0] = np.diag([0.0, 100.0])

    smin, smax, cond, frequency, bad = BWeiss._denominator_metrics(w, p)

    np.testing.assert_allclose([smin, smax, cond], [1.0, 101.0, 101.0])
    assert frequency == 0
    assert bad is True


class _IdentityDLR:
    nu = np.arange(2, dtype=float)

    @staticmethod
    def MatsubaraDLR2UniformGrid(value, sign=1):
        return np.asfortranarray(value)


def _cal_object(iteration, previous_cf=None):
    obj = object.__new__(BWeiss)
    obj.key = "1"
    obj.iteration = iteration
    obj.dlr = _IdentityDLR()
    v = np.zeros((1, 1, 1, 1), dtype=np.complex128, order="F")
    obj.vloc = type("VLoc", (), {"vproj": {"1": v}})()
    obj.w = type("W", (), {"f": _boson_blocks(1, 2)})()
    obj.p = type("P", (), {"f": _boson_blocks(1, 2)})()
    obj.w.f[0, 0, 0, 0, :] = 1.0
    obj.p.f[0, 0, 0, 0, :] = [0.0, -0.995]
    obj.Dyson = lambda w, sigma: np.asfortranarray(w * 2.0)
    obj.CausalProjection = lambda value, **kwargs: np.asfortranarray(value)
    obj.ReadBrdPrev = lambda stem, shape: previous_cf
    obj.WriteBrdPrev = lambda stem, value: None
    obj.F2T = lambda value: np.asfortranarray(value)
    obj._build_solver_consistent = lambda: None
    return obj


def test_bweiss_first_iteration_warns_and_continues():
    obj = _cal_object(iteration=1)

    with pytest.warns(RuntimeWarning, match="continuing without fallback"):
        obj.Cal()

    np.testing.assert_allclose(obj.f, obj.w.f * 2.0)
    assert obj.denominator_bad is True
    assert obj.denominator_fallback is False
    assert np.isfinite(obj.projection_delta_rel)


def test_bweiss_later_iteration_reuses_previous_bath(tmp_path):
    previous_cf = np.full_like(_boson_blocks(1, 2), 7.0)
    obj = _cal_object(iteration=2, previous_cf=previous_cf)
    obj.Dyson = lambda w, sigma: pytest.fail("Dyson must be skipped")
    obj.CausalProjection = lambda value, **kwargs: pytest.fail(
        "fresh projection must be skipped"
    )

    with pytest.warns(RuntimeWarning, match="reusing previous bath"):
        obj.Cal()

    np.testing.assert_allclose(obj.cf, previous_cf)
    np.testing.assert_allclose(obj.f, previous_cf)
    assert obj.denominator_fallback is True
    assert np.isnan(obj.projection_delta_abs)
    assert np.isnan(obj.projection_delta_rel)

    obj.hdf5file = str(tmp_path / "fallback.h5")
    obj.group = "calc"
    obj.subgroup = "BWeiss"
    obj.is_bare = False
    obj.f_to_solver = None
    obj.Save("bweiss")
    base = "calc/BWeiss/bweiss.2.1_"
    with h5py.File(obj.hdf5file, "r") as handle:
        assert handle[base + "denominator_fallback"][()] == 1
        assert np.isnan(handle[base + "projection_delta_abs"][()])
        assert np.isnan(handle[base + "projection_delta_rel"][()])


def test_bweiss_later_iteration_without_previous_bath_fails():
    obj = _cal_object(iteration=2)

    with pytest.raises(RuntimeError, match="no previous bath is available"):
        obj.Cal()


def test_bweiss_save_records_denominator_and_projection_diagnostics(tmp_path):
    obj = _cal_object(iteration=1)
    obj.hdf5file = str(tmp_path / "glob.h5")
    obj.group = "calc"
    obj.subgroup = "BWeiss"
    obj.is_bare = False
    obj.f_to_solver = None

    with pytest.warns(RuntimeWarning, match="continuing without fallback"):
        obj.Cal()
    obj.Save("bweiss")

    base = "calc/BWeiss/bweiss.1.1_"
    with h5py.File(obj.hdf5file, "r") as handle:
        np.testing.assert_allclose(handle[base + "denominator_smin"][()], 5.0e-3)
        np.testing.assert_allclose(handle[base + "denominator_smax"][()], 1.0)
        np.testing.assert_allclose(handle[base + "denominator_cond"][()], 1.0)
        assert handle[base + "denominator_bad_frequency"][()] == 1
        assert handle[base + "denominator_fallback"][()] == 0
        np.testing.assert_allclose(handle[base + "projection_delta_abs"][()], 0.0)
        np.testing.assert_allclose(handle[base + "projection_delta_rel"][()], 0.0)
