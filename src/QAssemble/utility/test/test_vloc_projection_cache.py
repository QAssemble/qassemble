from types import SimpleNamespace

import numpy as np

from QAssemble.BLocStc import VLoc
from QAssemble.BLocDyn import BWeiss


class _FakeProjector:
    def __init__(self):
        self.bprojector = {
            "1": np.array([[[1.0]], [[0.0]]]),
            "2": np.array([[[0.0]], [[1.0]]]),
        }
        self.equiv = {"1": np.eye(1, dtype=int), "2": np.eye(1, dtype=int)}


class _FakeDLR:
    def __init__(self, nfreq=2):
        self.nu = np.arange(nfreq, dtype=float)

    def BatchBF2T(self, bf_2d):
        return np.asarray(bf_2d, dtype=np.complex128)


def _vloc_with_data(projector=None):
    obj = VLoc.__new__(VLoc)
    obj.crystal = SimpleNamespace(ns=1)
    obj.projector = projector
    obj.vloc = np.zeros((2, 2, 1, 1), dtype=np.complex128, order="F")
    obj.vloc[:, :, 0, 0] = np.array([[2.0, 0.5], [0.25, 3.0]])
    obj.vproj = {}
    return obj


def test_vloc_build_projection_populates_keyed_cache():
    projector = _FakeProjector()
    vloc = _vloc_with_data(projector=projector)

    out = vloc.BuildProjection()

    assert out is vloc.vproj
    assert set(vloc.vproj.keys()) == {"1", "2"}
    np.testing.assert_allclose(vloc.vproj["1"], vloc.Projection(vloc.vloc, "1"))
    np.testing.assert_allclose(vloc.vproj["2"], vloc.Projection(vloc.vloc, "2"))
    np.testing.assert_allclose(vloc.vproj["1"][0, 0, 0, 0], 2.0)
    np.testing.assert_allclose(vloc.vproj["2"][0, 0, 0, 0], 3.0)


def test_vloc_build_projection_accepts_late_projector():
    projector = _FakeProjector()
    vloc = _vloc_with_data(projector=None)

    out = vloc.BuildProjection(projector)

    assert vloc.projector is projector
    assert out is vloc.vproj
    np.testing.assert_allclose(vloc.vproj["1"][0, 0, 0, 0], 2.0)
    np.testing.assert_allclose(vloc.vproj["2"][0, 0, 0, 0], 3.0)


def test_bweiss_uses_cached_projected_vloc_and_projected_dynamic_inputs():
    projector = _FakeProjector()
    crystal = SimpleNamespace(ns=1)
    dlr = _FakeDLR(nfreq=2)

    vproj = np.zeros((1, 1, 1, 1), dtype=np.complex128, order="F")
    vproj[0, 0, 0, 0] = 2.0
    vloc = SimpleNamespace(vloc=np.zeros((2, 2, 1, 1)), vproj={"1": vproj})

    w = np.zeros((1, 1, 1, 1, 2), dtype=np.complex128, order="F")
    p = np.zeros_like(w)
    w[0, 0, 0, 0, :] = np.array([3.0, 4.0])
    p[0, 0, 0, 0, :] = np.array([0.1, -0.2])

    bweiss = BWeiss(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        vloc=vloc,
        ploc=SimpleNamespace(f=p),
        wloc=SimpleNamespace(f=w),
    )

    expected_utilde = w / (1.0 + p * w)
    expected_ubar = expected_utilde - vproj[..., np.newaxis]
    np.testing.assert_allclose(bweiss.f, expected_utilde)
    np.testing.assert_allclose(bweiss.cf, expected_ubar)
    np.testing.assert_allclose(bweiss.t, expected_utilde)
    np.testing.assert_allclose(bweiss.ct, expected_ubar)

    removed_attrs = (
        "v",
        "v" + "_rf",
        "utilde" + "_rf",
        "utilde" + "_t",
        "ubar" + "_rf",
        "ubar" + "_t",
    )
    for attr in removed_attrs:
        assert not hasattr(bweiss, attr)
