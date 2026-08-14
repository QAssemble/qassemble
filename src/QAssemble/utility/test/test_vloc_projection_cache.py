from types import SimpleNamespace

import numpy as np

from QAssemble.BLocStc import VLoc
from QAssemble.BLocDyn import BLocDyn, BWeiss


class _FakeProjector:
    def __init__(self):
        self.bprojector = {
            "1": np.array([[[1.0]], [[0.0]]]),
            "2": np.array([[[0.0]], [[1.0]]]),
        }
        self.equiv = {"1": np.eye(1, dtype=int), "2": np.eye(1, dtype=int)}
        self.blocal2pair = {"1": [{0: (0, 0)}], "2": [{0: (0, 0)}]}

    def ProbFPair2Borb(self, key, iorb, jorb, ispace=0):
        return 0


class _FakeDLR:
    def __init__(self, nfreq=2):
        self.nu = np.arange(nfreq, dtype=float)

    def BatchBF2T(self, bf_2d):
        return np.asarray(bf_2d, dtype=np.complex128)

    def MatsubaraDLR2UniformGrid(self, value, sign=1):
        assert sign == 1
        return np.asfortranarray(np.asarray(value, dtype=np.complex128) + 10.0)


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


def test_blocdyn_projects_lattice_polarization_by_problem_key():
    projector = _FakeProjector()
    bloc = BLocDyn(
        crystal=SimpleNamespace(ns=1),
        dlr=_FakeDLR(nfreq=2),
        projector=projector,
    )
    lattice = np.zeros((2, 2, 1, 1, 2, 2), dtype=np.complex128)
    lattice[0, 0, 0, 0, :, :] = 3.0
    lattice[1, 1, 0, 0, :, :] = 5.0

    np.testing.assert_allclose(bloc.Projection(lattice, "1"), 3.0)
    np.testing.assert_allclose(bloc.Projection(lattice, "2"), 5.0)


def test_bweiss_uses_cached_projected_vloc_and_projected_dynamic_inputs(monkeypatch):
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

    # With p set, Cal causally projects cf before deriving the uniform/tau
    # views.  Stub the projection to undo the fake uniform-grid offset so the
    # wiring assertions below stay exact, and record the call.
    projection_calls = []

    def _fake_projection(self, matin, **kwargs):
        projection_calls.append(kwargs)
        return np.asfortranarray(np.asarray(matin, dtype=np.complex128) - 10.0)

    monkeypatch.setattr(BWeiss, "CausalProjection", _fake_projection)

    bweiss = BWeiss(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        vloc=vloc,
        w=SimpleNamespace(f=w),
        p=SimpleNamespace(f=p),
    )

    expected_utilde = w / (1.0 + p * w)
    expected_ubar = expected_utilde - vproj[..., np.newaxis]
    assert len(projection_calls) == 1
    assert projection_calls[0]["grid"] == "uniform"
    assert projection_calls[0]["coefficient_sign"] == -1
    assert projection_calls[0]["oddzero"] is True
    assert projection_calls[0]["highzero"] is True
    np.testing.assert_allclose(bweiss.f, expected_utilde)
    np.testing.assert_allclose(bweiss.cf, expected_ubar)
    np.testing.assert_allclose(bweiss.f_uniform, expected_utilde + 10.0)
    np.testing.assert_allclose(bweiss.cf_uniform, expected_ubar + 10.0)
    np.testing.assert_allclose(bweiss.t, expected_utilde)
    np.testing.assert_allclose(bweiss.ct, expected_ubar)
    # Single density pair: the solver-consistent view degenerates to f/cf.
    np.testing.assert_allclose(bweiss.f_to_solver, expected_utilde)
    np.testing.assert_allclose(bweiss.cf_to_solver, expected_ubar[0, 0, 0, 0, :])

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


def test_bweiss_without_p_copies_w_and_projects_bare_bath(monkeypatch):
    # The bare path (p is None, first iteration) used to skip the causal
    # projection entirely, feeding an unprojected W_loc - v into dyn.json.
    # It now projects cf like the correlated path does.
    projector = _FakeProjector()
    crystal = SimpleNamespace(ns=1)
    dlr = _FakeDLR(nfreq=2)

    vproj = np.zeros((1, 1, 1, 1), dtype=np.complex128, order="F")
    vloc = SimpleNamespace(vloc=np.zeros((2, 2, 1, 1)), vproj={"1": vproj})
    w = np.ones((1, 1, 1, 1, 2), dtype=np.complex128, order="F") * 3.0

    projection_calls = []

    def _fake_projection(self, matin, **kwargs):
        projection_calls.append(kwargs)
        return np.asfortranarray(np.asarray(matin, dtype=np.complex128) - 10.0)

    monkeypatch.setattr(BWeiss, "CausalProjection", _fake_projection)

    bweiss = BWeiss(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        vloc=vloc,
        w=SimpleNamespace(f=w),
        p=None,
    )

    assert len(projection_calls) == 1
    assert projection_calls[0]["grid"] == "uniform"
    assert projection_calls[0]["coefficient_sign"] == -1
    assert projection_calls[0]["oddzero"] is True
    assert projection_calls[0]["highzero"] is True
    np.testing.assert_allclose(bweiss.f, w)
    assert bweiss.f is not w
    assert bweiss.is_bare


def test_bweiss_seeds_and_reuses_brd_prev_cache(monkeypatch, tmp_path):
    projector = _FakeProjector()
    crystal = SimpleNamespace(ns=1)
    path = str(tmp_path / "glob.h5")

    vproj = np.zeros((1, 1, 1, 1), dtype=np.complex128, order="F")
    vproj[0, 0, 0, 0] = 2.0
    vloc = SimpleNamespace(vloc=np.zeros((2, 2, 1, 1)), vproj={"1": vproj})
    w = np.zeros((1, 1, 1, 1, 2), dtype=np.complex128, order="F")
    p = np.zeros_like(w)
    w[0, 0, 0, 0, :] = np.array([3.0, 4.0])
    p[0, 0, 0, 0, :] = np.array([0.1, -0.2])

    projection_calls = []

    def _fake_projection(self, matin, **kwargs):
        projection_calls.append(kwargs)
        return np.asfortranarray(np.asarray(matin, dtype=np.complex128) - 10.0)

    monkeypatch.setattr(BWeiss, "CausalProjection", _fake_projection)

    def _build():
        return BWeiss(
            crystal=crystal,
            dlr=_FakeDLR(nfreq=2),
            projector=projector,
            key="1",
            vloc=vloc,
            w=SimpleNamespace(f=w),
            p=SimpleNamespace(f=p),
            hdf5file=path,
            group="calc",
        )

    first = _build()
    assert projection_calls[0]["fallback_matrix"] is None

    _build()
    # The second iteration receives the first one's projected cf as fallback.
    np.testing.assert_allclose(projection_calls[1]["fallback_matrix"], first.cf)
