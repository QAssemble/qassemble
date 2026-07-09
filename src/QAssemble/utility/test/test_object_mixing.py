from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from QAssemble.BLatDyn import BLatDyn, P
from QAssemble.BLatStc import BLatStc
from QAssemble.BLocDyn import PImp
from QAssemble.FLatDyn import FLatDyn, G, SigGWC
from QAssemble.FLatStc import FLatStc, H, SigF, SigH
from QAssemble.FLocDyn import GLoc, SigCImp
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

    assert obj.Mixing(iter=1, mix=0.5, method="linear", npulay=2) is None
    obj.h = np.asarray([2.0], dtype=np.complex128)
    assert obj.Mixing(iter=2, mix=0.5, method="linear", npulay=2) is None

    np.testing.assert_allclose(obj.h, [1.0])
    assert obj.s is obj.h
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/sighimp/last"][()], [1.0])


def test_sigfimp_mixing_assigns_s(tmp_path):
    path = tmp_path / "mix.h5"
    obj = object.__new__(SigFImp)
    _seed_common(obj, path)
    obj.s = np.asarray([0.0], dtype=np.complex128)

    assert obj.Mixing(iter=1, mix=0.5, method="linear", npulay=2) is None
    obj.s = np.asarray([4.0], dtype=np.complex128)
    assert obj.Mixing(iter=2, mix=0.25, method="linear", npulay=2) is None

    np.testing.assert_allclose(obj.s, [1.0])
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/sigfimp/last"][()], [1.0])


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

    assert obj.Mixing(iter=1, mix=0.5, method="linear", npulay=2) is None
    obj.f = np.asarray([6.0], dtype=np.complex128)
    assert obj.Mixing(iter=2, mix=0.5, method="linear", npulay=2) is None

    np.testing.assert_allclose(obj.f, [3.0])
    np.testing.assert_allclose(obj.f_uniform, [13.0])
    np.testing.assert_allclose(obj.t, [23.0])
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/sigimp/last"][()], [3.0])


def test_pimp_mixing_assigns_f_and_tau(tmp_path):
    path = tmp_path / "mix.h5"
    obj = object.__new__(PImp)
    _seed_common(obj, path)
    obj.f = np.asarray([0.0], dtype=np.complex128)
    obj.t = None
    obj.F2T = lambda value: np.asfortranarray(
        np.asarray(value, dtype=np.complex128) + 20.0
    )

    assert obj.Mixing(iter=1, mix=0.5, method="linear", npulay=2) is None
    obj.f = np.asarray([8.0], dtype=np.complex128)
    assert obj.Mixing(iter=2, mix=0.5, method="linear", npulay=2) is None

    np.testing.assert_allclose(obj.f, [4.0])
    np.testing.assert_allclose(obj.t, [24.0])
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/Mixing/1/pimp/last"][()], [4.0])


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
