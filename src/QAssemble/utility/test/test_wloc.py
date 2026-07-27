from types import SimpleNamespace

import numpy as np

from QAssemble.BLocDyn import WLoc


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


def _identity_projection(monkeypatch, calls):
    def _fake(self, matin, **kwargs):
        calls.append((np.array(matin, copy=True), kwargs))
        return np.asfortranarray(np.asarray(matin, dtype=np.complex128))

    monkeypatch.setattr(WLoc, "CausalProjection", _fake)


def test_wloc_projects_lattice_screened_interaction_and_correlation_part(monkeypatch):
    dlr = _FakeDLR(nfreq=3)
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()
    calls = []
    _identity_projection(monkeypatch, calls)

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

    # The exact static v is split off and the decaying dynamic part is
    # projected on the DLR grid with the wlocbrd sign convention.
    assert len(calls) == 1
    projected_input, kwargs = calls[0]
    np.testing.assert_allclose(projected_input, expected - vloc[..., np.newaxis])
    assert kwargs["grid"] == "dlr"
    assert kwargs["coefficient_sign"] == -1
    assert kwargs["oddzero"] is True
    assert kwargs["highzero"] is True
    assert kwargs["fallback_matrix"] is None


def test_wloc_builds_tau_quantities_through_f2t(monkeypatch):
    dlr = _FakeDLR(nfreq=3)
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()
    _identity_projection(monkeypatch, [])

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


def test_wloc_without_vloc_projects_full_f(monkeypatch):
    dlr = _FakeDLR(nfreq=3)
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()
    calls = []
    _identity_projection(monkeypatch, calls)

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
    # No static to split: the full f is projected with the default c0
    # tail-fit split (no oddzero/highzero overrides).
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["grid"] == "dlr"
    assert kwargs["coefficient_sign"] == -1
    assert "oddzero" not in kwargs
    assert "highzero" not in kwargs


def test_wloc_seeds_and_reuses_brd_prev_cache(monkeypatch, tmp_path):
    dlr = _FakeDLR(nfreq=3)
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()
    path = str(tmp_path / "glob.h5")
    calls = []
    _identity_projection(monkeypatch, calls)

    vloc = np.zeros((1, 1, 1, 1), dtype=np.complex128, order="F")
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

    first = _build()
    assert calls[0][1]["fallback_matrix"] is None

    _build()
    # The second iteration receives the first one's projected cf as fallback.
    np.testing.assert_allclose(calls[1][1]["fallback_matrix"], first.cf)
