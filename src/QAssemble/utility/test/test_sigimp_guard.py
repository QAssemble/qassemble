from types import SimpleNamespace
import importlib
import json

import h5py
import numpy as np
import pytest

from QAssemble.CTQMC import CTQMC
from QAssemble.FLocDyn import FWeiss
from QAssemble.utility.DLR import DLR


def _fake_ctqmc(tmp_path, *, control=None, equiv=None, ns=1):
    equiv = np.eye(1, dtype=int) if equiv is None else np.asarray(equiv, dtype=int)
    norb = len(equiv)
    projector = SimpleNamespace(
        fprojector={"1": np.eye(norb)[:, :, None].repeat(ns, axis=2)},
        bprojector={"1": np.eye(norb ** 2)[:, :, None, None]},
        equiv={"1": equiv},
    )
    obj = CTQMC.__new__(CTQMC)
    obj.crystal = SimpleNamespace(ns=ns, soc=False)
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


def test_sigimp_smoothing_preserves_first_two_points_across_beta(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)
    ctqmc.fweiss = FWeiss.__new__(FWeiss)
    sigma = np.full((1, 1, 1, 12), -1j)
    sigma[..., 1:3] = [-0.2j, -0.8j]
    for beta in range(10, 401):
        omega = (2 * np.arange(12) + 1) * np.pi / beta
        ctqmc.dlr = SimpleNamespace(
            beta=beta, cutoff=300.0, MatsubaraFermionUniform=lambda: omega
        )
        smoothed = ctqmc._smooth_sigma_grid(sigma)
        np.testing.assert_array_equal(smoothed[..., :2], sigma[..., :2], err_msg=f"beta={beta}")
        assert smoothed[0, 0, 0, 2] != sigma[0, 0, 0, 2], f"beta={beta}"


def _write_previous_sigma(ctqmc, sigma, name="sigimp_input.1.1"):
    with h5py.File(ctqmc.hdf5file, "a") as handle:
        handle.require_group(f"{ctqmc.group}/SigCImp").create_dataset(name, data=sigma)


def _two_class_sigma(ctqmc):
    sigma = np.zeros((2, 2, ctqmc.crystal.ns, 3), dtype=complex)
    sigma[0, 0] = 2.0 - 0.5j
    sigma[1, 1] = 7.0 + 0.1j
    template = {"1": {"moments": [2.0]}, "2": {"moments": [7.0]}}
    sigma_input = ctqmc._sigma_array_to_ctqmc_dict(
        ctqmc.projector.equiv["1"], template, sigma
    )
    return sigma_input, sigma


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_sigimp_guard_rejects_nonfinite_values(tmp_path, value):
    ctqmc = _fake_ctqmc(tmp_path)
    sigma = np.asarray([[[[0.0 + 0.0j, value + 0.0j, 0.0 + 0.0j]]]])

    diag = ctqmc._evaluate_sigimp_guard(1, sigma)

    assert diag["guard_status"] == "failed"
    assert diag["guard_reason"] == "nonfinite"
    assert diag["broken"] == [(1, 0)]
    assert diag["used_fallback"] is False


def test_sigimp_guard_rejects_positive_diagonal_imaginary_part(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)
    sigma = np.asarray([[[[0.0 - 0.1j, 0.0 + 1.0e-4j, 0.0 - 0.2j]]]])

    diag = ctqmc._evaluate_sigimp_guard(1, sigma)

    assert diag["guard_status"] == "failed"
    assert "positive_imag" in diag["guard_reason"]
    assert diag["broken"] == [(1, 0)]


def test_sigimp_guard_roughness_and_error_are_diagnostic_only(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)
    with h5py.File(ctqmc.hdf5file, "a") as handle:
        guard = handle.require_group("calc/SigImpGuard/1/metrics.1")
        guard.attrs["raw_roughness"] = 1.0e-3
        guard.attrs["err_mean"] = 1.0e-3
        guard.attrs["err_max"] = 1.0e-3
    sigma = np.asarray([[[[0.0 + 0.0j, 100.0 + 0.0j, -100.0 + 0.0j]]]])

    diag = ctqmc._evaluate_sigimp_guard(2, sigma, err_grid=np.ones_like(sigma))

    assert diag["guard_status"] == "accepted"
    assert diag["broken"] == []
    assert diag["used_fallback"] is False
    assert diag["raw_roughness"] == 300.0
    assert diag["err_mean"] == diag["err_max"] == 1.0


def test_sigimp_guard_checks_all_frequencies(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path)
    sigma = np.full((1, 1, 1, 100), -0.1j)
    sigma[..., 90] = 0.01j

    diag = ctqmc._evaluate_sigimp_guard(1, sigma)

    assert diag["guard_status"] == "failed"
    assert diag["broken"] == [(1, 0)]


def test_sigimp_guard_checks_unique_diagonal_classes_per_spin(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path, equiv=[[1, 3, 0], [3, 2, 0], [0, 0, 2]], ns=2)
    sigma = np.zeros((3, 3, 2, 3), dtype=complex)
    sigma[0, 1] = np.nan
    sigma[1, 0] = 1j
    sigma[1, 1, 1] = sigma[2, 2, 1] = 0.1j

    diag = ctqmc._evaluate_sigimp_guard(1, sigma)

    assert diag["broken"] == [(2, 1)]
    assert diag["guard_reason"] == "positive_imag"
    unchanged = sigma.copy()
    ctqmc._replace_broken_sigma(1, {"2": {"moments": [7.0]}}, sigma, diag)
    np.testing.assert_array_equal(sigma[[1, 2], [1, 2], 1], 7.0)
    assert sigma[:, :, 0].tobytes() == unchanged[:, :, 0].tobytes()
    assert sigma[0, 1].tobytes() == unchanged[0, 1].tobytes()
    assert sigma[1, 0].tobytes() == unchanged[1, 0].tobytes()


@pytest.mark.parametrize("control", [{"SigImpGuard": False}, {"SigImpGuardMode": "off"}])
def test_sigimp_guard_can_be_disabled(tmp_path, control):
    ctqmc = _fake_ctqmc(tmp_path, control=control)

    diag = ctqmc._evaluate_sigimp_guard(1, np.full((1, 1, 1, 3), 1j))

    assert diag["guard_status"] == "disabled"
    assert diag["broken"] == []


@pytest.mark.parametrize("iteration", [1, 2])
def test_sigimp_guard_zeros_only_broken_class_without_previous_input(tmp_path, iteration, caplog):
    ctqmc = _fake_ctqmc(tmp_path, equiv=np.diag([1, 2]), ns=2)
    sigma_input, sigma = _two_class_sigma(ctqmc)
    sigma[1, 1, 0] = 7.0 - 0.5j
    unchanged = sigma.copy()
    diag = ctqmc._evaluate_sigimp_guard(iteration, sigma)

    ctqmc._replace_broken_sigma(iteration, sigma_input, sigma, diag)

    assert diag["broken"] == [(2, 1)]
    assert diag["fallback_source"] == "zero"
    assert diag["used_fallback"] is False
    assert sigma[0, 0].tobytes() == unchanged[0, 0].tobytes()
    assert sigma[1, 1, 0].tobytes() == unchanged[1, 1, 0].tobytes()
    np.testing.assert_array_equal(sigma[1, 1, 1], 7.0)
    assert sigma_input["2"]["moments"] == [7.0]
    assert "class=2 spin=1" in caplog.text


def test_sigimp_guard_uses_previous_raw_input_before_mixed_output(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path, equiv=np.diag([1, 2]))
    sigma_input, sigma = _two_class_sigma(ctqmc)
    unchanged = sigma.copy()
    previous = np.full_like(sigma, -0.25j)
    previous[1, 1] = [0.5 - 0.25j, 1.0 - 0.5j, 2.0 - 1j]
    _write_previous_sigma(ctqmc, previous)
    _write_previous_sigma(ctqmc, 2 * previous, "sigimp.1.1_uniform")
    diag = ctqmc._evaluate_sigimp_guard(2, sigma)

    ctqmc._replace_broken_sigma(2, sigma_input, sigma, diag)

    assert diag["used_fallback"] is True
    assert diag["fallback_source"] == "sigimp_input"
    np.testing.assert_array_equal(sigma[1, 1], 7.0 + previous[1, 1])
    assert sigma[0, 0].tobytes() == unchanged[0, 0].tobytes()


def test_sigimp_guard_legacy_mixed_output_fallback_warns(tmp_path, caplog):
    ctqmc = _fake_ctqmc(tmp_path, equiv=np.diag([1, 2]))
    sigma_input, sigma = _two_class_sigma(ctqmc)
    previous = np.full_like(sigma, -0.25j)
    _write_previous_sigma(ctqmc, previous, "sigimp.1.1_uniform")
    diag = ctqmc._evaluate_sigimp_guard(2, sigma)

    ctqmc._replace_broken_sigma(2, sigma_input, sigma, diag)

    assert diag["used_fallback"] is True
    assert diag["fallback_source"] == "sigimp_mixed"
    assert "legacy restart" in caplog.text
    np.testing.assert_array_equal(sigma[1, 1], 7.0 + previous[1, 1])


@pytest.mark.parametrize("name", ["sigimp_input.1.1", "sigimp.1.1_uniform"])
def test_sigimp_guard_rejects_previous_grid_shape_mismatch(tmp_path, name):
    ctqmc = _fake_ctqmc(tmp_path, equiv=np.diag([1, 2]))
    sigma_input, sigma = _two_class_sigma(ctqmc)
    _write_previous_sigma(ctqmc, np.zeros_like(sigma[..., :2]), name)
    diag = ctqmc._evaluate_sigimp_guard(2, sigma)

    with pytest.raises(ValueError, match="shape mismatch.*expected.*got"):
        ctqmc._replace_broken_sigma(2, sigma_input, sigma, diag)


def test_sigimp_guard_saves_broken_classes_as_string_attribute(tmp_path):
    ctqmc = _fake_ctqmc(tmp_path, equiv=np.diag([1, 2]))
    sigma_input, sigma = _two_class_sigma(ctqmc)
    sigma[0, 0] = 0.1j
    diag = ctqmc._evaluate_sigimp_guard(1, sigma)
    ctqmc._replace_broken_sigma(1, sigma_input, sigma, diag)

    ctqmc._write_sigimp_guard_diagnostics(1, diag)

    with h5py.File(ctqmc.hdf5file, "r") as handle:
        attrs = handle["calc/SigImpGuard/1/metrics.1"].attrs
        assert attrs["broken"] == "1/0,2/0"
        assert attrs["fallback_source"] == "zero"
        assert attrs["used_fallback"] == 0


@pytest.mark.parametrize("method", ["edmft", "gw+edmft"])
@pytest.mark.parametrize("bad_value", [0.1j, np.nan + 0.0j])
def test_postprocessing_replaces_before_dlr_and_keeps_raw_input_before_mixing(
    tmp_path, monkeypatch, method, bad_value
):
    ctqmc = _fake_ctqmc(
        tmp_path, equiv=np.diag([1, 2]),
        control={"method": method, "mix": 0.25, "mixing_method": "linear", "npulay": 3},
    )
    ctqmc.projector.ProbBorb2FPair = lambda key, ind: divmod(ind, 2)
    ctqmc.bweiss.vloc.vproj["1"] = 2.0 * np.eye(4)[:, :, None, None]
    ctqmc.dlr = DLR({"beta": 20.0, "cutoff": 8.0, "eps": 1.0e-12})
    omega = ctqmc.dlr.MatsubaraFermionUniform()
    ctqmc.work_dir = str(tmp_path)
    ctqmc.ctqmc_dir = str(tmp_path / "ctqmc")
    module = importlib.import_module("QAssemble.CTQMC")
    monkeypatch.setattr(
        module, "GImp", lambda **kwargs: SimpleNamespace(occ=rho, Save=lambda *args: None)
    )
    monkeypatch.chdir(tmp_path)
    previous_h = previous_fock = previous_f = None
    for iteration in (1, 2, 3):
        rho = np.diag([0.1 * iteration, 0.2 * iteration])[:, :, None]
        moments = np.diag([2.0 + iteration, 7.0 + iteration])[:, :, None]
        sigma = np.zeros((2, 2, 1, omega.size), dtype=complex)
        sigma[0, 0, 0] = moments[0, 0, 0] + iteration / (1.0 + 1j * omega)
        sigma[1, 1, 0] = moments[1, 1, 0] + (
            -0.5j / (1.0 + omega) if iteration == 2 else bad_value
        )
        template = {
            str(ind): {"moments": [moments[ind - 1, ind - 1, 0]]} for ind in (1, 2)
        }
        partition = {
            "expansion histogram": [1, 2, 1], "scalar": {"N": 0.6 * iteration},
            "sign": 1.0, "green": {},
            "self-energy": ctqmc._sigma_array_to_ctqmc_dict(
                ctqmc.projector.equiv["1"], template, sigma
            ),
        }
        work = tmp_path / "ctqmc" / f"impurity_{iteration}_1"
        work.mkdir(parents=True)
        (work / "params.obs.json").write_text(json.dumps({"partition": partition}))
        (work / "params.json").write_text(json.dumps({"partition": {"green matsubara cutoff": 3}}))

        ctqmc.PostProcessing(iteration)

        np.testing.assert_array_equal(ctqmc.sigimp.hf, moments)
        np.testing.assert_array_equal(ctqmc.sigfimp.hf, moments)
        expected_h = 4.0 * rho
        if previous_h is not None:
            expected_h = 0.25 * expected_h + 0.75 * previous_h
            assert not np.array_equal(ctqmc.sigfimp.s, previous_fock)
        np.testing.assert_allclose(ctqmc.sighimp.h, expected_h)
        assert np.all(np.isfinite(ctqmc.sigimp.f))
        assert ctqmc.diagnostics["sigimp_guard_failed"] == float(iteration != 2)
        assert ctqmc.diagnostics["sigimp_guard_used_fallback"] == float(iteration == 3)
        with h5py.File(ctqmc.hdf5file, "r") as handle:
            raw = handle[f"calc/SigCImp/sigimp_input.{iteration}.1"][()]
            assert raw.shape == sigma.shape
            assert raw[0, 0].tobytes() == (sigma - moments[..., None])[0, 0].tobytes()
            if iteration == 1:
                np.testing.assert_array_equal(raw[1, 1], 0.0)
                np.testing.assert_array_equal(ctqmc.sigimp.f[1, 1], 0.0)
            elif iteration == 2:
                np.testing.assert_array_equal(raw[1, 1], (sigma - moments[..., None])[1, 1])
                assert not np.allclose(raw[1, 1], ctqmc.sigimp.f_uniform[1, 1])
            else:
                np.testing.assert_array_equal(raw[1, 1], handle["calc/SigCImp/sigimp_input.2.1"][()][1, 1])
                np.testing.assert_allclose(ctqmc.sigimp.f[1, 1], 1.75 * previous_f[1, 1])
                attrs = handle["calc/SigImpGuard/1/metrics.3"].attrs
                assert attrs["broken"] == "2/0"
                assert attrs["fallback_source"] == "sigimp_input"
        previous_h = ctqmc.sighimp.h.copy()
        previous_fock = ctqmc.sigfimp.s.copy()
        previous_f = ctqmc.sigimp.f.copy()


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
