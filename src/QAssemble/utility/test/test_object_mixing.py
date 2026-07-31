from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from QAssemble.BLatDyn import BLatDyn, P
from QAssemble.BLatStc import BLatStc
from QAssemble.BLocDyn import BWeiss, PImp
from QAssemble.FLatDyn import FLatDyn, G, SigGWC
from QAssemble.FLatStc import FLatStc, H, SigF, SigH
from QAssemble.FLocDyn import FWeiss, GLoc, SigCImp
from QAssemble.FLocStc import SigFImp, SigHImp


class _FakeDLR:
    def MatsubaraDLR2UniformGrid(self, value, sign=-1):
        return np.asfortranarray(np.asarray(value, dtype=np.complex128) + 10.0)


def _patch_gloc_calculation(monkeypatch):
    def fake_projection(self, green, key):
        self.projected_key = key
        return np.asfortranarray(np.asarray(green, dtype=np.complex128) + 1.0)

    def fake_f2t(self, mat):
        return np.asfortranarray(np.asarray(mat, dtype=np.complex128) + 2.0)

    def fake_occ(self, mat):
        return np.asfortranarray(np.asarray(mat, dtype=np.complex128)[..., 0])

    monkeypatch.setattr(GLoc, "Projection", fake_projection)
    monkeypatch.setattr(GLoc, "F2T", fake_f2t)
    monkeypatch.setattr(GLoc, "Occ", fake_occ)


def _seed_common(obj, path):
    obj.hdf5file = str(path)
    obj.group = "calc"
    obj.key = "1"
    obj.control = {"mix": 0.5, "mixing_method": "linear", "npulay": 2}
    obj.iteration = 1


def _seed_save(obj, path, subgroup, iteration=None, key=None):
    obj.hdf5file = str(path)
    obj.group = "calc"
    obj.subgroup = subgroup
    obj.iteration = iteration
    if key is not None:
        obj.key = key


def test_global_save_appends_iteration_and_scf_false_preserves_name(tmp_path):
    path = tmp_path / "save.h5"
    obj = object.__new__(SigH)
    _seed_save(obj, path, "SigH", iteration=3)
    obj.k = np.asarray([1.0], dtype=np.complex128)

    obj.Save("sigh")
    obj.Save("sigh_final", scf=False)

    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/SigH/sigh.3"][()], [1.0])
        np.testing.assert_allclose(handle["calc/SigH/sigh_final"][()], [1.0])


def test_local_save_appends_iteration_and_key(tmp_path):
    path = tmp_path / "save.h5"
    obj = object.__new__(SigCImp)
    _seed_save(obj, path, "SigCImp", iteration=4, key="1")
    obj.f = np.asarray([2.0], dtype=np.complex128)
    obj.f_uniform = np.asarray([3.0], dtype=np.complex128)

    obj.Save("sigimp")
    obj.Save("sigimp_final", scf=False)

    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/SigCImp/sigimp.4.1"][()], [2.0])
        np.testing.assert_allclose(handle["calc/SigCImp/sigimp.4.1_uniform"][()], [3.0])
        np.testing.assert_allclose(handle["calc/SigCImp/sigimp_final"][()], [2.0])
        np.testing.assert_allclose(handle["calc/SigCImp/sigimp_final_uniform"][()], [3.0])


def test_gloc_save_appends_iteration_and_key(tmp_path):
    path = tmp_path / "save.h5"
    obj = object.__new__(GLoc)
    _seed_save(obj, path, "GLoc", iteration=0, key="1")
    obj.f = np.asarray([4.0], dtype=np.complex128)

    obj.Save("gloc")
    obj.iteration = 1
    obj.Save("gloc")

    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/GLoc/gloc.0.1"][()], [4.0])
        np.testing.assert_allclose(handle["calc/GLoc/gloc.1.1"][()], [4.0])


def test_gloc_auto_selects_single_problem_key(monkeypatch, tmp_path):
    _patch_gloc_calculation(monkeypatch)
    projector = SimpleNamespace(fprojector={"1": None})
    green = np.ones((1, 1, 1, 1), dtype=np.complex128)

    obj = GLoc(
        crystal=SimpleNamespace(ns=1),
        dlr=SimpleNamespace(),
        projector=projector,
        green=green,
        hdf5file=str(tmp_path / "gloc.h5"),
        group="calc",
        iteration=0,
    )

    assert obj.key == "1"
    assert obj.projected_key == "1"
    np.testing.assert_allclose(obj.f, green + 1.0)


def test_gloc_requires_key_for_multiple_problems():
    projector = SimpleNamespace(fprojector={"1": None, "2": None})

    with pytest.raises(ValueError, match="requires key"):
        GLoc(
            crystal=SimpleNamespace(ns=1),
            dlr=SimpleNamespace(),
            projector=projector,
            green=np.ones((1, 1, 1, 1), dtype=np.complex128),
        )


def test_gloc_uses_explicit_problem_key(monkeypatch, tmp_path):
    _patch_gloc_calculation(monkeypatch)
    projector = SimpleNamespace(fprojector={"1": None, "2": None})
    green = np.ones((1, 1, 1, 1), dtype=np.complex128)

    obj = GLoc(
        crystal=SimpleNamespace(ns=1),
        dlr=SimpleNamespace(),
        projector=projector,
        green=green,
        key="2",
        hdf5file=str(tmp_path / "gloc.h5"),
        group="calc",
        iteration=0,
    )

    assert obj.key == "2"
    assert obj.projected_key == "2"
    np.testing.assert_allclose(obj.f, green + 1.0)


def test_g_and_h_save_mu_with_matching_scf_suffix(tmp_path):
    path = tmp_path / "save.h5"

    g = object.__new__(G)
    _seed_save(g, path, "G", iteration=6)
    g.kf = np.asarray([5.0], dtype=np.complex128)
    g.mu = 0.25
    g.Save("gkf")
    g.Save("gkf_final", scf=False)

    h = object.__new__(H)
    _seed_save(h, path, "H", iteration=7)
    h.k = np.asarray([6.0], dtype=np.complex128)
    h.mu = 0.5
    h.Save("hk")
    h.Save("hk_final", scf=False)

    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/G/gkf.6"][()], [5.0])
        np.testing.assert_allclose(handle["calc/G/mu.6"][()], 0.25)
        np.testing.assert_allclose(handle["calc/G/gkf_final"][()], [5.0])
        np.testing.assert_allclose(handle["calc/G/mu"][()], 0.25)
        np.testing.assert_allclose(handle["calc/H/hk.7"][()], [6.0])
        np.testing.assert_allclose(handle["calc/H/mu.7"][()], 0.5)
        np.testing.assert_allclose(handle["calc/H/hk_final"][()], [6.0])
        np.testing.assert_allclose(handle["calc/H/mu"][()], 0.5)


def test_save_requires_iteration_when_scf_true(tmp_path):
    obj = object.__new__(SigH)
    _seed_save(obj, tmp_path / "save.h5", "SigH", iteration=None)
    obj.k = np.asarray([1.0], dtype=np.complex128)

    with pytest.raises(ValueError, match="iteration"):
        obj.Save("sigh")


def test_sighimp_mixing_assigns_h_and_s(tmp_path):
    path = tmp_path / "mix.h5"
    obj = object.__new__(SigHImp)
    _seed_common(obj, path)
    obj.h = np.asarray([0.0], dtype=np.complex128)
    obj.s = None

    assert obj.Mixing() is None
    obj.h = np.asarray([2.0], dtype=np.complex128)
    obj.iteration = 2
    assert obj.Mixing() is None

    np.testing.assert_allclose(obj.h, [1.0])
    assert obj.s is obj.h
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/sighimp/last"][()], [1.0])


def test_sigfimp_mixing_rederives_instead_of_mixing(tmp_path):
    """SigF carries no mixing history of its own.

    ``SigFImp.Mixing`` re-runs ``Cal`` so that ``s = hf - sigh`` is recomputed
    from the already-mixed SigH.  Mixing it independently would break
    ``SigH + SigF == hf`` by one iteration of lag; see the method docstring and
    test_sighimp_sigfimp_hf_identity.py.
    """
    path = tmp_path / "mix.h5"
    obj = object.__new__(SigFImp)
    _seed_common(obj, path)
    obj.projector = SimpleNamespace(equiv={"1": np.asarray([[1]], dtype=int)})
    obj.crystal = SimpleNamespace(ns=1)
    obj.sigma_in = np.asarray([4.0], dtype=np.complex128)
    obj.sigh = np.asarray([1.0], dtype=np.complex128)
    obj.hf = None
    # Seeded so an independent mix would have a finite value to blend and would
    # therefore succeed; the assertions below are what must reject it.
    obj.s = np.asarray([0.0], dtype=np.complex128)

    assert obj.Mixing() is None
    # s is the algebraic complement of sigh, not a blend with any history.
    np.testing.assert_allclose(obj.s, [3.0])

    # At iter > 1 an independent mix would blend against stored history.  The
    # control dict and hdf5file are fully populated, so that path would succeed
    # if it were still taken -- these assertions are what reject it.
    obj.iteration = 2
    assert obj.Mixing() is None
    np.testing.assert_allclose(obj.s, [3.0])

    # Changing the mixed sigh propagates in full: under the 0.5 linear mix
    # seeded above only half of it would land.
    obj.sigh = np.asarray([2.5], dtype=np.complex128)
    assert obj.Mixing() is None
    np.testing.assert_allclose(obj.s, [1.5])

    # No sigfimp mixing history was ever created -- the re-derivation never
    # touches HDF5, so the file itself is never even opened for writing.
    assert not path.exists()


def test_sigcimp_mixing_assigns_f_and_uniform_grid(tmp_path):
    path = tmp_path / "mix.h5"
    obj = object.__new__(SigCImp)
    _seed_common(obj, path)
    obj.dlr = _FakeDLR()
    obj.f = np.asarray([0.0], dtype=np.complex128)
    obj.f_uniform = np.asarray([-1.0], dtype=np.complex128)
    obj.F2T = lambda value: np.asfortranarray(
        np.asarray(value, dtype=np.complex128) + 20.0
    )

    assert obj.Mixing() is None
    obj.f = np.asarray([6.0], dtype=np.complex128)
    obj.iteration = 2
    assert obj.Mixing() is None

    np.testing.assert_allclose(obj.f, [3.0])
    np.testing.assert_allclose(obj.f_uniform, [13.0])
    np.testing.assert_allclose(obj.t, [23.0])
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/sigimp/last"][()], [3.0])


def test_pimp_mixing_assigns_f_and_tau(monkeypatch, tmp_path):
    path = tmp_path / "mix.h5"
    obj = object.__new__(PImp)
    _seed_common(obj, path)
    obj.dlr = _FakeDLR()
    obj.f = np.asarray([0.0], dtype=np.complex128)
    obj.f_uniform = np.asarray([-1.0], dtype=np.complex128)
    obj.t = None
    obj.F2T = lambda value: np.asfortranarray(
        np.asarray(value, dtype=np.complex128) + 20.0
    )

    projection_calls = []

    def fake_causal_projection(self, value, *, grid="dlr", **kwargs):
        projection_calls.append((grid, kwargs))
        return np.asfortranarray(np.asarray(value, dtype=np.complex128))

    monkeypatch.setattr(PImp, "CausalProjection", fake_causal_projection)

    assert obj.Mixing() is None
    obj.f = np.asarray([8.0], dtype=np.complex128)
    obj.iteration = 2
    assert obj.Mixing() is None

    np.testing.assert_allclose(obj.f, [4.0])
    np.testing.assert_allclose(obj.t, [24.0])
    # f_uniform is re-derived from the mixed+projected f (fake DLR adds 10).
    np.testing.assert_allclose(obj.f_uniform, [14.0])
    # The mixed value is re-projected on the DLR grid with the zero-static
    # pimpbrd policy.
    assert len(projection_calls) == 2
    grid, kwargs = projection_calls[-1]
    assert grid == "dlr"
    assert kwargs["coefficient_sign"] == -1
    assert kwargs["oddzero"] is True
    assert kwargs["highzero"] is True
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/pimp/last"][()], [4.0])
        # The projected result seeds the previous-iteration fallback cache.
        np.testing.assert_allclose(handle["calc/PImp/pimp_brd_prev.1"][()], [4.0])


def test_pimp_mixing_overwrites_last_with_projected_value(monkeypatch, tmp_path):
    # Pins the FullGWEDMFT semantics: the stored mixing "last" holds the
    # *projected* value the run consumes, so the next iteration's fold (and
    # its residuals) are based on the causal result -- while the appended
    # history entries still reflect the pre-overwrite mixing pipeline.
    path = tmp_path / "mix.h5"
    obj = object.__new__(PImp)
    _seed_common(obj, path)
    obj.dlr = _FakeDLR()
    obj.f = np.asarray([0.0], dtype=np.complex128)
    obj.f_uniform = None
    obj.t = None
    obj.F2T = lambda value: np.asfortranarray(
        np.asarray(value, dtype=np.complex128) + 20.0
    )

    fallbacks = []

    def fake_causal_projection(self, value, *, grid="dlr", **kwargs):
        fallbacks.append(kwargs.get("fallback_matrix"))
        return np.asfortranarray(np.asarray(value, dtype=np.complex128) + 1.0)

    monkeypatch.setattr(PImp, "CausalProjection", fake_causal_projection)

    assert obj.Mixing() is None
    # iter 1: passthrough mixing [0.0] -> projected [1.0] overwrites last.
    np.testing.assert_allclose(obj.f, [1.0])
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/pimp/last"][()], [1.0])

    obj.f = np.asarray([8.0], dtype=np.complex128)
    obj.iteration = 2
    assert obj.Mixing() is None

    # iter 2 folds against the *projected* last: 0.5*8 + 0.5*1 = 4.5, then
    # the projection (+1) gives 5.5, which again overwrites last and the cache.
    np.testing.assert_allclose(obj.f, [5.5])
    # The previous iteration's projected value arrived as the QP fallback.
    np.testing.assert_allclose(fallbacks[-1], [1.0])
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/pimp/last"][()], [5.5])
        np.testing.assert_allclose(handle["calc/PImp/pimp_brd_prev.1"][()], [5.5])


def test_fweiss_mixing_mixes_hyb_reprojects_and_recomputes_h(monkeypatch, tmp_path):
    path = tmp_path / "mix.h5"
    obj = object.__new__(FWeiss)
    _seed_common(obj, path)
    obj.hyb = np.asarray([0.0], dtype=np.complex128)
    obj.h = None

    projection_calls = []

    def fake_causal_projection(self, value, *, grid="dlr", **kwargs):
        projection_calls.append((grid, kwargs))
        return np.asfortranarray(np.asarray(value, dtype=np.complex128) + 1.0)

    def fake_cal(self):
        self.h = np.asarray(self.hyb, dtype=np.complex128) + 100.0

    monkeypatch.setattr(FWeiss, "CausalProjection", fake_causal_projection)
    monkeypatch.setattr(FWeiss, "Cal", fake_cal)

    assert obj.Mixing(iter=1, control=obj.control) is None
    # iter 1 passthrough [0.0] -> projected [1.0]; Cal rebuilds the averaged
    # uniform hybridization hyb.json is written from (fake Cal adds 100).
    np.testing.assert_allclose(obj.hyb, [1.0])
    np.testing.assert_allclose(obj.h, [101.0])

    obj.hyb = np.asarray([8.0], dtype=np.complex128)
    assert obj.Mixing(iter=2, control=obj.control) is None

    # iter 2 folds against the *projected* last: 0.5*8 + 0.5*1 = 4.5, then
    # the projection (+1) gives 5.5, which again overwrites last.
    np.testing.assert_allclose(obj.hyb, [5.5])
    np.testing.assert_allclose(obj.h, [105.5])
    assert projection_calls[-1][0] == "dlr"
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/hyb/last"][()], [5.5])


def test_fweiss_mixing_survives_projection_failure(monkeypatch, tmp_path):
    path = tmp_path / "mix.h5"
    obj = object.__new__(FWeiss)
    _seed_common(obj, path)
    obj.hyb = np.asarray([3.0], dtype=np.complex128)

    def broken_projection(self, value, **kwargs):
        raise RuntimeError("infeasible")

    monkeypatch.setattr(FWeiss, "CausalProjection", broken_projection)
    monkeypatch.setattr(FWeiss, "Cal", lambda self: None)

    with pytest.warns(RuntimeWarning, match="re-projection failed"):
        assert obj.Mixing(iter=1, control=obj.control) is None
    # The unprojected mixed value is kept and still becomes the stored last.
    np.testing.assert_allclose(obj.hyb, [3.0])
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/hyb/last"][()], [3.0])


def test_bweiss_mixing_mixes_reprojects_and_rebuilds_derived(monkeypatch, tmp_path):
    path = tmp_path / "mix.h5"
    obj = object.__new__(BWeiss)
    _seed_common(obj, path)
    obj.dlr = _FakeDLR()
    obj.subgroup = "BWeiss"
    obj.vloc = SimpleNamespace(vproj={"1": np.asarray(2.0)})
    obj.cf = np.asarray([0.0], dtype=np.complex128)
    obj.f = None
    obj.f_uniform = None
    obj.cf_uniform = None
    obj.t = None
    obj.ct = None
    obj.F2T = lambda value: np.asfortranarray(
        np.asarray(value, dtype=np.complex128) + 20.0
    )

    projection_calls = []

    def fake_causal_projection(self, value, *, grid="dlr", **kwargs):
        projection_calls.append((grid, kwargs))
        return np.asfortranarray(np.asarray(value, dtype=np.complex128) + 1.0)

    monkeypatch.setattr(BWeiss, "CausalProjection", fake_causal_projection)

    assert obj.Mixing(obj.control) is None
    # iter 1 passthrough [0.0] -> projected [1.0]; every derived quantity is
    # rebuilt from the mixed bath (fake DLR->uniform adds 10, fake F2T adds 20).
    np.testing.assert_allclose(obj.cf, [1.0])
    np.testing.assert_allclose(obj.f, [3.0])
    np.testing.assert_allclose(obj.f_uniform, [13.0])
    np.testing.assert_allclose(obj.cf_uniform, [11.0])
    np.testing.assert_allclose(obj.t, [23.0])
    np.testing.assert_allclose(obj.ct, [21.0])

    obj.cf = np.asarray([8.0], dtype=np.complex128)
    obj.iteration = 2
    assert obj.Mixing(obj.control) is None

    # iter 2 folds against the *projected* last: 0.5*8 + 0.5*1 = 4.5, then the
    # projection (+1) gives 5.5, which again overwrites last and the cache.
    np.testing.assert_allclose(obj.cf, [5.5])
    np.testing.assert_allclose(obj.f, [7.5])
    grid, kwargs = projection_calls[-1]
    assert grid == "dlr"
    assert kwargs["coefficient_sign"] == -1
    assert kwargs["oddzero"] is True
    assert kwargs["highzero"] is True
    # The previous iteration's projected bath arrived as the QP fallback.
    np.testing.assert_allclose(kwargs["fallback_matrix"], [1.0])
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/bweiss/last"][()], [5.5])
        np.testing.assert_allclose(handle["calc/BWeiss/bweiss_brd_prev.1"][()], [5.5])


def test_bweiss_mixing_skips_static_bath(tmp_path):
    path = tmp_path / "mix.h5"
    obj = object.__new__(BWeiss)
    _seed_common(obj, path)
    obj.cf = None

    assert obj.Mixing(obj.control) is None
    assert not path.exists()


def test_lattice_parent_mixing_uses_global_key(tmp_path):
    path = tmp_path / "mix.h5"
    for cls in (FLatStc, FLatDyn, BLatStc, BLatDyn):
        obj = object.__new__(cls)
        obj.hdf5file = str(path)
        obj.group = cls.__name__

        obj.Mixing(
            iter=1,
            mix=0.5,
            component="quantity",
            value=np.asarray([0.0], dtype=np.complex128),
            method="linear",
            npulay=2,
        )
        mixed = obj.Mixing(
            iter=2,
            mix=0.5,
            component="quantity",
            value=np.asarray([2.0], dtype=np.complex128),
            method="linear",
            npulay=2,
        )

        np.testing.assert_allclose(mixed, [1.0])
        with h5py.File(path, "r") as handle:
            np.testing.assert_allclose(
                handle[f"{cls.__name__}/Mixing/global/quantity/last"][()],
                [1.0],
            )


def test_lattice_concrete_mixing_updates_owned_state(tmp_path):
    path = tmp_path / "mix.h5"

    sig_h = object.__new__(SigH)
    sig_h.hdf5file = str(path)
    sig_h.group = "calc"
    sig_h.k = np.asarray([0.0], dtype=np.complex128)
    sig_h.K2R = lambda value: np.asfortranarray(
        np.asarray(value, dtype=np.complex128) + 10.0
    )
    assert sig_h.Mixing(iter=1, mix=0.5, method="linear", npulay=2) is None
    sig_h.k = np.asarray([2.0], dtype=np.complex128)
    assert sig_h.Mixing(iter=2, mix=0.5, method="linear", npulay=2) is None
    np.testing.assert_allclose(sig_h.k, [1.0])
    np.testing.assert_allclose(sig_h.r, [11.0])

    sig_f = object.__new__(SigF)
    sig_f.hdf5file = str(path)
    sig_f.group = "calc"
    sig_f.k = np.asarray([0.0], dtype=np.complex128)
    sig_f.K2R = lambda value: np.asfortranarray(
        np.asarray(value, dtype=np.complex128) + 20.0
    )
    assert sig_f.Mixing(iter=1, mix=0.5, method="linear", npulay=2) is None
    sig_f.k = np.asarray([4.0], dtype=np.complex128)
    assert sig_f.Mixing(iter=2, mix=0.25, method="linear", npulay=2) is None
    np.testing.assert_allclose(sig_f.k, [1.0])
    np.testing.assert_allclose(sig_f.r, [21.0])

    siggwc = object.__new__(SigGWC)
    siggwc.hdf5file = str(path)
    siggwc.group = "calc"
    siggwc.kf = np.asarray([0.0], dtype=np.complex128)
    siggwc.F2T = lambda value: np.asfortranarray(
        np.asarray(value, dtype=np.complex128) + 30.0
    )
    siggwc.K2R = lambda value: np.asfortranarray(
        np.asarray(value, dtype=np.complex128) + 40.0
    )
    assert siggwc.Mixing(iter=1, mix=0.5, method="linear", npulay=2) is None
    siggwc.kf = np.asarray([6.0], dtype=np.complex128)
    assert siggwc.Mixing(iter=2, mix=0.5, method="linear", npulay=2) is None
    np.testing.assert_allclose(siggwc.kf, [3.0])
    np.testing.assert_allclose(siggwc.kt, [33.0])
    np.testing.assert_allclose(siggwc.rf, [43.0])
    np.testing.assert_allclose(siggwc.rt, [73.0])

    pol = object.__new__(P)
    pol.hdf5file = str(path)
    pol.group = "calc"
    pol.kf = np.asarray([0.0], dtype=np.complex128)
    pol.F2T = lambda value: np.asfortranarray(
        np.asarray(value, dtype=np.complex128) + 50.0
    )
    pol.K2R = lambda value: np.asfortranarray(
        np.asarray(value, dtype=np.complex128) + 60.0
    )
    assert pol.Mixing(iter=1, mix=0.5, method="linear", npulay=2) is None
    pol.kf = np.asarray([8.0], dtype=np.complex128)
    assert pol.Mixing(iter=2, mix=0.5, method="linear", npulay=2) is None
    np.testing.assert_allclose(pol.kf, [4.0])
    np.testing.assert_allclose(pol.kt, [54.0])
    np.testing.assert_allclose(pol.rf, [64.0])
    np.testing.assert_allclose(pol.rt, [114.0])

    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/global/sigh/last"][()], [1.0])
        np.testing.assert_allclose(handle["calc/Mixing/global/sigf/last"][()], [1.0])
        np.testing.assert_allclose(handle["calc/Mixing/global/siggwc/last"][()], [3.0])
        np.testing.assert_allclose(handle["calc/Mixing/global/pol/last"][()], [4.0])
