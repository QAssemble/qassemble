from types import SimpleNamespace

import h5py
import numpy as np

from QAssemble.CTQMC import CTQMC


def _fake_ctqmc(tmp_path, *, control=None):
    projector = SimpleNamespace(
        fprojector={"1": np.ones((1, 1, 1), dtype=float)},
        bprojector={"1": np.ones((1, 1, 1, 1), dtype=float)},
        equiv={"1": np.eye(1, dtype=int)},
    )
    obj = CTQMC.__new__(CTQMC)
    obj.crystal = SimpleNamespace(ns=1, soc=False)
    obj.projector = projector
    obj.key = "1"
    obj.control = control or {}
    obj.hdf5file = str(tmp_path / "glob.h5")
    obj.group = "calc"
    obj.dlr = SimpleNamespace(
        beta=10.0,
        MatsubaraFermionUniform=lambda: np.asarray([1.0, 3.0, 5.0]),
        MatsubaraDLR2UniformGrid=lambda arr, sign=-1: np.asarray(arr),
    )
    obj.fweiss = SimpleNamespace()
    obj.bweiss = SimpleNamespace(
        f=None,
        f_uniform=None,
        vloc=SimpleNamespace(vproj={"1": np.asarray([[[[2.0]]]])}),
    )
    return obj


def _write_previous_state(ctqmc, iter_no=2):
    with h5py.File(ctqmc.hdf5file, "a") as handle:
        guard = handle.require_group(f"{ctqmc.group}/SigImpGuard/1/metrics.{iter_no - 1}")
        guard.attrs["raw_roughness"] = 1.0e-3
        guard.attrs["err_mean"] = 1.0e-3
        guard.attrs["err_max"] = 2.0e-3
        sigh = handle.require_group(f"{ctqmc.group}/SigHImp")
        sigf = handle.require_group(f"{ctqmc.group}/SigFImp")
        sigc = handle.require_group(f"{ctqmc.group}/SigCImp")
        sigh.create_dataset(
            f"sighimp.{iter_no - 1}.1",
            data=np.asarray([[[1.0 + 0.0j]]]),
        )
        sigf.create_dataset(
            f"sigfimp.{iter_no - 1}.1",
            data=np.asarray([[[2.0 + 0.0j]]]),
        )
        sigc.create_dataset(
            f"sigimp.{iter_no - 1}.1",
            data=np.asarray([[[[3.0 + 0.0j]]]]),
        )
        sigc.create_dataset(
            f"sigimp.{iter_no - 1}.1_uniform",
            data=np.asarray([[[[4.0 + 0.0j, 5.0 + 0.0j]]]]),
        )


def test_sigimp_guard_rejects_nonfinite_values(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)
    sigma = np.asarray([[[[0.0 + 0.0j, np.nan + 0.0j, 0.0 + 0.0j]]]])

    diag = ctqmc._evaluate_sigimp_guard(1, sigma)

    assert diag["guard_status"] == "failed"
    assert diag["guard_reason"] == "nonfinite"
    assert diag["used_fallback"] is False


def test_sigimp_guard_rejects_positive_diagonal_imaginary_part(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)
    sigma = np.asarray([[[[0.0 - 0.1j, 0.0 + 1.0e-4j, 0.0 - 0.2j]]]])

    diag = ctqmc._evaluate_sigimp_guard(1, sigma)

    assert diag["guard_status"] == "failed"
    assert "positive_imag" in diag["guard_reason"]


def test_sigimp_guard_first_iteration_skips_relative_roughness(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)
    sigma = np.asarray([[[[0.0 + 0.0j, 100.0 + 0.0j, -100.0 + 0.0j]]]])

    diag = ctqmc._evaluate_sigimp_guard(1, sigma)

    assert diag["guard_status"] == "accepted"
    assert diag["used_fallback"] is False


def test_sigimp_guard_roughness_uses_previous_fallback(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)
    _write_previous_state(ctqmc, iter_no=2)
    sigma = np.asarray([[[[0.0 + 0.0j, 100.0 + 0.0j, -100.0 + 0.0j]]]])

    diag = ctqmc._evaluate_sigimp_guard(2, sigma)

    assert diag["guard_status"] == "failed"
    assert "roughness_ratio" in diag["guard_reason"]
    assert diag["used_fallback"] is True


def test_sigimp_guard_builds_fallback_objects_and_saves_current_iter(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)
    _write_previous_state(ctqmc, iter_no=2)

    previous = ctqmc._read_previous_impurity_output(2)
    ctqmc._make_fallback_impurity_objects(previous, 2)
    ctqmc.sighimp.Save("sighimp")
    ctqmc.sigfimp.Save("sigfimp")
    ctqmc.sigimp.Save("sigimp")

    with h5py.File(ctqmc.hdf5file, "r") as handle:
        np.testing.assert_allclose(
            handle["calc/SigHImp/sighimp.2.1"][()], previous["sigh"]
        )
        np.testing.assert_allclose(
            handle["calc/SigFImp/sigfimp.2.1"][()], previous["sigf"]
        )
        np.testing.assert_allclose(
            handle["calc/SigCImp/sigimp.2.1"][()], previous["sigc"]
        )
        np.testing.assert_allclose(
            handle["calc/SigCImp/sigimp.2.1_uniform"][()],
            previous["sigc_uniform"],
        )


def test_sighimp_uses_solver_consistent_uniform_zero_frequency(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)
    ctqmc.bweiss.f = np.asarray([[[[[999.0]]]]])
    ctqmc.bweiss.f_uniform = np.asarray([[[[[123.0, 124.0]]]]])
    ctqmc.bweiss.f_to_solver_uniform = np.asarray([[[[[321.0, 322.0]]]]])

    vloc, source = ctqmc._resolve_sighimp_vloc("1")

    assert source == "bweiss_solver_uniform_nu0"
    np.testing.assert_allclose(vloc, np.asarray([[[[321.0]]]]))


def test_sighimp_uses_solver_consistent_dlr_fallback(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)
    ctqmc.bweiss.f = np.asarray([[[[[999.0]]]]])
    ctqmc.bweiss.f_uniform = np.asarray([[[[[123.0, 124.0]]]]])
    ctqmc.bweiss.f_to_solver = np.asarray([[[[[456.0]]]]])
    ctqmc.dlr = SimpleNamespace(
        MatsubaraDLR2UniformGrid=lambda arr, sign=1: np.concatenate(
            (arr, arr + 1.0), axis=-1
        )
    )

    vloc, source = ctqmc._resolve_sighimp_vloc("1")

    assert source == "bweiss_solver_dlr_to_uniform_nu0"
    np.testing.assert_allclose(vloc, np.asarray([[[[456.0]]]]))


def test_sighimp_legacy_uniform_fallback_warns(tmp_path, caplog):
    ctqmc = _fake_ctqmc(tmp_path)
    ctqmc.bweiss.f = np.asarray([[[[[999.0]]]]])
    ctqmc.bweiss.f_uniform = np.asarray([[[[[123.0, 124.0]]]]])

    vloc, source = ctqmc._resolve_sighimp_vloc("1")

    assert source == "bweiss_uniform_nu0"
    assert "legacy matrix-valued interaction" in caplog.text
    np.testing.assert_allclose(vloc, np.asarray([[[[123.0]]]]))


def test_sighimp_legacy_dlr_fallback_warns(tmp_path, caplog):
    ctqmc = _fake_ctqmc(tmp_path)
    ctqmc.bweiss.f = np.asarray([[[[[999.0]]]]])
    ctqmc.bweiss.f_uniform = None
    ctqmc.dlr = SimpleNamespace(
        MatsubaraDLR2UniformGrid=lambda arr, sign=1: np.asarray([[[[[456.0, 457.0]]]]])
    )

    vloc, source = ctqmc._resolve_sighimp_vloc("1")

    assert source == "bweiss_dlr_to_uniform_nu0"
    assert "legacy matrix-valued interaction" in caplog.text
    np.testing.assert_allclose(vloc, np.asarray([[[[456.0]]]]))


def test_sighimp_static_bweiss_uses_bare_interaction(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)

    vloc, source = ctqmc._resolve_sighimp_vloc("1")

    assert source == "bare_vloc"
    np.testing.assert_allclose(vloc, np.asarray([[[[2.0]]]]))
