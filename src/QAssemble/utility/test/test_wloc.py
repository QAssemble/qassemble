from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from QAssemble.BLocDyn import WLoc
from QAssemble.utility.Projection import Projection as PJ


class _FakeDLR:
    def __init__(self, nfreq=3):
        self.nu = np.arange(nfreq, dtype=float)
        self.tauB = np.arange(nfreq, dtype=float)

    def BatchBF2T(self, bf_2d):
        return 2.0 * np.asarray(bf_2d, dtype=np.complex128)


class _FakeProjector:
    def __init__(self):
        self.bprojector = {"1": np.ones((1, 1, 1), dtype=float)}
        self.equiv = {"1": np.eye(1, dtype=int)}


@pytest.fixture(autouse=True)
def _forbid_projection_and_cache(monkeypatch):
    def _fail(*args, **kwargs):
        pytest.fail("WLoc must not call causal projection or fallback cache methods")

    for method in ("CausalProjection", "ReadBrdPrev", "WriteBrdPrev"):
        monkeypatch.setattr(WLoc, method, _fail)


def test_wloc_projects_lattice_screened_interaction_and_correlation_part():
    dlr = _FakeDLR(nfreq=3)
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()

    vloc = np.zeros((1, 1, 1, 1), dtype=np.complex128, order="F")
    vloc[0, 0, 0, 0] = 2.0
    wlat = np.zeros((1, 1, 1, 1, 2, 3), dtype=np.complex128, order="F")
    wlat[0, 0, 0, 0, 0, :] = np.array([3.0, 4.0, 5.0])
    wlat[0, 0, 0, 0, 1, :] = np.array([5.0, 6.0, 7.0])

    wloc = WLoc(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        wlat=wlat,
        vloc=vloc,
    )

    expected = np.mean(wlat, axis=4)
    np.testing.assert_allclose(wloc.f, expected)
    np.testing.assert_allclose(wloc.cf, expected - vloc[..., np.newaxis])

    np.testing.assert_array_equal(wloc.f, PJ.BLatDyn(wlat, projector.bprojector["1"]))
    assert wloc.cf.flags.f_contiguous


def test_wloc_builds_tau_quantities_through_f2t():
    dlr = _FakeDLR(nfreq=3)
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()

    vloc = np.ones((1, 1, 1, 1), dtype=np.complex128, order="F")
    wlat = np.ones((1, 1, 1, 1, 2, 3), dtype=np.complex128, order="F")
    wlat[..., 1, :] *= 3.0

    wloc = WLoc(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        wlat=wlat,
        vloc=vloc,
    )

    np.testing.assert_allclose(wloc.t, 2.0 * wloc.f)
    np.testing.assert_allclose(wloc.ct, 2.0 * wloc.cf)


def test_wloc_without_vloc_preserves_full_f():
    dlr = _FakeDLR(nfreq=3)
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()

    wlat = np.ones((1, 1, 1, 1, 2, 3), dtype=np.complex128, order="F") * 4.0

    wloc = WLoc(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        wlat=wlat,
        vloc=None,
    )

    assert wloc.cf is None
    assert wloc.ct is None
    np.testing.assert_allclose(wloc.f, np.mean(wlat, axis=4))
    np.testing.assert_array_equal(wloc.t, 2.0 * wloc.f)


@pytest.mark.parametrize("with_vloc", [False, True])
def test_wloc_leaves_brd_prev_cache_untouched(tmp_path, with_vloc):
    dlr = _FakeDLR(nfreq=3)
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()
    path = str(tmp_path / "glob.h5")

    vloc = np.ones((1, 1, 1, 1), dtype=np.complex128) if with_vloc else None
    wlat = np.ones((1, 1, 1, 1, 2, 3), dtype=np.complex128, order="F") * 4.0

    def _build():
        return WLoc(
            crystal=crystal,
            dlr=dlr,
            projector=projector,
            key="1",
            wlat=wlat,
            vloc=vloc,
            hdf5file=path,
            group="calc",
        )

    with h5py.File(path, "w"):
        pass
    first = _build()
    with h5py.File(path, "r") as file:
        assert list(file) == []
    cache_path = "calc/WLoc/wloc_brd_prev.1"
    cache = np.full(first.f.shape, 17.0 + 3.0j)
    with h5py.File(path, "a") as file:
        file[cache_path] = cache
    second = _build()
    np.testing.assert_array_equal(second.f, first.f)
    if with_vloc:
        np.testing.assert_array_equal(second.cf, first.cf)
    with h5py.File(path, "r") as file:
        np.testing.assert_array_equal(file[cache_path][()], cache)


@pytest.mark.parametrize("offdiag", [0.0, 1e-15 + 2e-15j])
def test_wloc_preserves_multiorbital_spatial_projection(offdiag):
    # A non-unit projector makes direct projection differ from a k-average.
    projector = SimpleNamespace(
        bprojector={"1": np.diag([0.5, 2.0])[..., None]},
        equiv={"1": np.eye(2, dtype=int)},
    )
    wlat = np.zeros((2, 2, 1, 1, 2, 3), dtype=np.complex128)
    wlat[0, 0, 0, 0] = [[3, 4, 5], [5, 6, 7]]
    wlat[1, 1, 0, 0] = [[7, 8, 9], [9, 10, 11]]
    # Reconstructing f as (f - v) + v would lose this small diagonal value.
    wlat[0, 0, 0, 0, :, 2] = 1e-18 + 1e-19j
    wlat[0, 1] = offdiag
    wlat[1, 0] = np.conjugate(offdiag)
    vloc = np.diag([0.5, 1.5])[:, :, None, None]
    wloc = WLoc(SimpleNamespace(ns=1), _FakeDLR(), projector, "1", wlat, vloc)
    expected = PJ.BLatDyn(wlat, projector.bprojector["1"])
    np.testing.assert_array_equal(wloc.f, expected)
    np.testing.assert_array_equal(wloc.cf, expected - vloc[..., None])
    np.testing.assert_array_equal(wloc.t, 2.0 * expected)
    np.testing.assert_array_equal(wloc.ct, 2.0 * wloc.cf)
    mask = ~np.eye(2, dtype=bool)
    if offdiag == 0:
        assert np.count_nonzero(wloc.cf[mask]) == 0
    else:
        assert np.all(wloc.cf[mask] != 0)
