from types import SimpleNamespace

import os
import json
import h5py
import numpy as np
import pytest

from QAssemble.CTQMC import CTQMC
from QAssemble.FLocStc import EImp
from QAssemble.utility.DLR import DLR


class _Output:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def Mixing(self, iter=None, mix=None, method="pulay", npulay=5):
        self.events.append(
            (self.name, "mix", iter, mix, method, npulay)
        )

    def Save(self, name):
        self.events.append((self.name, "save", name))


def _fake_projector():
    return SimpleNamespace(
        fprojector={"1": np.zeros((1, 1, 1), dtype=np.complex128)},
        bprojector={"1": np.zeros((1, 1, 1, 1), dtype=np.complex128)},
        equiv={"1": np.eye(1, dtype=int)},
    )


def test_ctqmc_keeps_run_control_for_impurity_objects(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    projector = _fake_projector()
    fweiss = SimpleNamespace(crystal=SimpleNamespace(), projector=projector, key="1")
    bweiss = SimpleNamespace(projector=projector, key="1")

    ctqmc = CTQMC(
        dlr="dlr",
        fweiss=fweiss,
        bweiss=bweiss,
        key=1,
        control={
            "mix": 0.25,
            "mixing_method": "linear",
            "npulay": 3,
        },
    )

    assert ctqmc.control == {
        "mix": 0.25,
        "mixing_method": "linear",
        "npulay": 3,
    }
    assert not hasattr(ctqmc, "mix_sig")
    assert not hasattr(ctqmc, "mix_p")
    assert os.getcwd() == str(tmp_path / "ctqmc")


@pytest.mark.parametrize("static_fit", [False, True])
@pytest.mark.parametrize("node_max", [4.0, 22.0])
@pytest.mark.parametrize("cutoffs", [
    {},
    {"GreenCutoff": 2.0, "SusceptibilityCutoff": 3.0},
    {"green_cutoff": 2.0, "susceptibility_cutoff": 3.0},
])
def test_preprocessing_mixes_inputs_before_writing_json(
    monkeypatch, tmp_path, static_fit, node_max, cutoffs
):
    monkeypatch.chdir(tmp_path)
    events = []
    control = {"mix": 0.25, "mixing_method": "linear", "npulay": 3}
    control.update(cutoffs)

    ctqmc = object.__new__(CTQMC)
    ctqmc.key = "1"
    ctqmc.control = control
    ctqmc.crystal = SimpleNamespace(soc=False, ns=1)
    ctqmc.projector = _fake_projector()
    ctqmc.dlr = SimpleNamespace(
        beta=10.0, cutoff=8.0,
        nu=np.asarray([-node_max, 0., node_max]), omega=np.asarray([-node_max, node_max]),
        MatsubaraFermionUniform=lambda: np.asarray([1.0, 3.0, 5.0, 7.0]),
        MatsubaraBosonUniform=lambda: np.asarray([0.0, 2.0]),
    )
    ctqmc.fweiss = SimpleNamespace(
        e=np.zeros((2, 2, 1)),
        h=np.zeros((1, 1, 1, 4), dtype=np.complex128),
        eimp=SimpleNamespace(ToCTQMC=lambda key, Eimp: (np.zeros((2, 2)), 0.5)),
        Mixing=lambda iter=None, control=None: events.append(
            ("fweiss_mix", iter, control)
        ),
        _write_json_pair=lambda stem, iter, key, payload: events.append(
            ("write", stem, iter, key)
        ),
        _as_hyb_dict=lambda key: {},
    )
    ctqmc.bweiss = SimpleNamespace(
        cf=np.asarray([1.0]),
        static_fit=static_fit,
        Save=lambda name: events.append(("bweiss_save", name)),
        vloc=SimpleNamespace(GetUijklComCTQMC=lambda key: np.zeros((1, 1, 1, 1))),
        Mixing=lambda control=None: events.append(("bweiss_mix", control)),
        _write_json_pair=lambda stem, iter, key, payload: events.append(
            ("write", stem, iter, key)
        ),
        _as_dyn_dict=lambda key: {},
    )
    ctqmc.work_dir = str(tmp_path)
    ctqmc.root_dir = str(tmp_path)
    ctqmc.ctqmc_dir = str(tmp_path)

    ctqmc.PreProcessing(iter=2)
    import json
    with open(tmp_path / "impurity_2_1" / "params.json") as stream:
        partition = json.load(stream)["partition"]
    if cutoffs:
        assert partition["green matsubara cutoff"] == 2.0
        assert partition["susceptibility cutoff"] == 3.0
    else:
        assert partition["green matsubara cutoff"] == 0.08
        assert partition["susceptibility cutoff"] == 0.16
    spacing = 2*np.pi/ctqmc.dlr.beta
    n_tail = int(partition["susceptibility tail"]/spacing)
    assert (n_tail-1)*spacing >= np.max(np.abs(ctqmc.dlr.nu))
    assert partition["susceptibility tail"] >= 2*ctqmc.dlr.cutoff
    if node_max == 4.0:
        assert partition["susceptibility tail"] == 2*ctqmc.dlr.cutoff

    # Both inputs are mixed strictly before their json files are written, and
    # each Mixing receives the run control (hyb additionally the iteration).
    assert events == [
        ("fweiss_mix", 2, control),
        ("write", "hyb", 2, "1"),
        ("bweiss_mix", control),
        ("bweiss_save", "bweiss"),
        ("write", "dyn", 2, "1"),
    ]


@pytest.mark.parametrize("control,expected", [
    ({}, 0.08), ({"GreenCutoff": 2.0}, 2.0), ({"green_cutoff": 3.0}, 3.0),
])
def test_sigma_smoothing_uses_green_cutoff(control, expected):
    ctqmc = object.__new__(CTQMC)
    ctqmc.control = control
    omega = np.asarray([1.0, 3.0, 5.0])
    sigma = np.full((1, 1, 1, 3), -0.1j)
    ctqmc.dlr = SimpleNamespace(
        beta=10.0, cutoff=8.0, MatsubaraFermionUniform=lambda: omega
    )
    calls = []

    def broadener(grid, data, width, temperature, cutoff):
        calls.append((grid, data, width, temperature, cutoff))
        return data

    ctqmc.fweiss = SimpleNamespace(GaussianLinearBroad=broadener)
    assert ctqmc._smooth_sigma_grid(sigma) is sigma
    assert len(calls) == 1
    grid, data, width, temperature, cutoff = calls[0]
    np.testing.assert_array_equal(grid, omega)
    assert data is sigma
    assert (width, temperature, cutoff) == (0.05, 0.1, expected)


def test_preprocessing_static_bath_skips_bweiss_mixing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    events = []
    control = {"mix": 0.25, "mixing_method": "linear", "npulay": 3}

    ctqmc = object.__new__(CTQMC)
    ctqmc.key = "1"
    ctqmc.control = control
    ctqmc.crystal = SimpleNamespace(soc=False, ns=1)
    ctqmc.projector = _fake_projector()
    ctqmc.dlr = SimpleNamespace(
        beta=10.0, cutoff=8.0,
        nu=np.asarray([-22., 0., 22.]), omega=np.asarray([-15., 15.]),
        MatsubaraFermionUniform=lambda: np.asarray([1.0, 3.0, 5.0, 7.0]),
        MatsubaraBosonUniform=lambda: np.asarray([0.0, 2.0]),
    )
    ctqmc.fweiss = SimpleNamespace(
        e=np.zeros((2, 2, 1)),
        h=np.zeros((1, 1, 1, 4), dtype=np.complex128),
        eimp=SimpleNamespace(ToCTQMC=lambda key, Eimp: (np.zeros((2, 2)), 0.5)),
        Mixing=lambda iter=None, control=None: events.append(
            ("fweiss_mix", iter, control)
        ),
        _write_json_pair=lambda stem, iter, key, payload: events.append(
            ("write", stem, iter, key)
        ),
        _as_hyb_dict=lambda key: {},
    )
    ctqmc.bweiss = SimpleNamespace(
        cf=None,
        vloc=SimpleNamespace(GetUijklComCTQMC=lambda key: np.zeros((1, 1, 1, 1))),
        Mixing=lambda control=None: events.append(("bweiss_mix", control)),
        _write_json_pair=lambda stem, iter, key, payload: events.append(
            ("write", stem, iter, key)
        ),
        _as_dyn_dict=lambda key: {},
    )
    ctqmc.work_dir = str(tmp_path)
    ctqmc.root_dir = str(tmp_path)
    ctqmc.ctqmc_dir = str(tmp_path)

    ctqmc.PreProcessing(iter=1)

    assert events == [
        ("fweiss_mix", 1, control),
        ("write", "hyb", 1, "1"),
    ]


@pytest.mark.parametrize("method,iteration,use_dyn,has_previous", [
    ("gw+edmft", 2, True, True),
    ("gw+edmft", 1, True, True),
    ("edmft", 2, True, True),
    ("dmft", 2, True, True),
    ("gw+edmft", 2, False, True),
    ("gw+edmft", 2, True, False),
])
def test_preprocessing_current_bath_hartree(
    monkeypatch, tmp_path, caplog, method, iteration, use_dyn, has_previous
):
    monkeypatch.chdir(tmp_path)
    ctqmc = object.__new__(CTQMC)
    ctqmc.key = "1"
    ctqmc.control = {"method": method, "mix": 0.1, "mixing_method": "linear", "npulay": 3}
    ctqmc.crystal = SimpleNamespace(soc=False, ns=1)
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ctqmc.projector = SimpleNamespace(
        fprojector={"1": np.eye(2)[:, :, None]},
        bprojector={"1": np.eye(4)[:, :, None]},
        equiv={"1": np.array([[1, 2], [2, 1]])},
        ProbBorb2FPair=lambda key, index: pairs[index],
    )
    ctqmc.dlr = DLR(dict(beta=10.0, cutoff=8.0))
    dlr = ctqmc.dlr
    rotation = np.array([[0.8, -0.6], [0.6, 0.8]])
    energies = np.array([-0.12, 0.24])
    green = np.einsum(
        "ik,kf,jk->ijf", rotation,
        1 / (1j * dlr.omega[None, :] - energies[:, None]), rotation,
    )[:, :, None, :]
    # Analytic single-pole densities at the same last tau node used by GImp.Occ.
    rho = rotation @ np.diag(
        np.exp(-dlr.tauF[-1] * energies) / (1 + np.exp(-dlr.beta * energies))
    ) @ rotation.T
    ctqmc.hdf5file = str(tmp_path / "impurity.h5")
    ctqmc.group = "GWEDMFT"
    with h5py.File(ctqmc.hdf5file, "w") as handle:
        group = handle.create_group(f"{ctqmc.group}/GImp")
        if has_previous:
            group.create_dataset("gimp.1.1", data=green)
        group.create_dataset("gimp.1.1_uniform", data=np.zeros_like(green))

    hloc = np.array([[0.4, 0.05], [0.05, 0.6]])[:, :, None]
    eimp = EImp(
        ctqmc.crystal, ctqmc.projector, "1",
        hamtb=np.array([[1.1, 0.2], [0.2, 1.7]])[:, :, None, None],
        mu=0.3, hloc=hloc.copy(), floc=np.zeros_like(hloc),
    )
    old_one_body, old_mu = eimp.ToCTQMC()
    omega = dlr.MatsubaraFermionUniform()
    hyb = {"1": {"real": np.zeros_like(omega).tolist(), "imag": (-0.2 / omega).tolist()}}
    written_hyb = []

    def write_json(stem, iteration, key, payload):
        with open(f"{stem}.json", "w") as stream:
            json.dump(payload, stream, sort_keys=True)
        if stem == "hyb":
            written_hyb.append((tmp_path / f"impurity_{iteration}_{key}" / "hyb.json").read_bytes())

    ctqmc.fweiss = SimpleNamespace(
        e=eimp.e.copy(), eimp=eimp,
        h=np.broadcast_to((0.2 / (1j * omega)), (2, 2, 1, omega.size)).copy(),
        Mixing=lambda **kwargs: None,
        _write_json_pair=write_json, _as_hyb_dict=lambda key: hyb,
    )
    m = np.array([[1.2, 0.3], [0.3, 0.8]])
    bare_v = 2 * np.outer(m.ravel(), m.ravel())[:, :, None, None]
    solver_u = 0.75 * bare_v

    def mix_bweiss(**kwargs):
        ctqmc.bweiss.f_to_solver_uniform = np.stack([solver_u, bare_v], axis=-1)

    ctqmc.bweiss = SimpleNamespace(
        cf=np.ones(1) if use_dyn else None,
        f=0.5 * bare_v[..., None], f_uniform=0.25 * bare_v[..., None],
        f_to_solver_uniform=None,
        vloc=SimpleNamespace(vproj={"1": bare_v}, GetUijklComCTQMC=lambda key: np.zeros((2, 2, 2, 2))),
        Mixing=mix_bweiss, Save=lambda name: None,
        _write_json_pair=write_json,
        _as_dyn_dict=lambda key: {"D0": (solver_u - bare_v).ravel().tolist()},
    )
    ctqmc.ctqmc_dir = str(tmp_path)

    def no_hartree_mixing(self):
        pytest.fail("The solver-input Hartree must not be mixed")

    monkeypatch.setattr("QAssemble.CTQMC.SigHImp.Mixing", no_hartree_mixing)
    with caplog.at_level("INFO", logger="QAssemble"):
        ctqmc.PreProcessing(iter=iteration)

    workdir = tmp_path / f"impurity_{iteration}_1"
    params = json.loads((workdir / "params.json").read_text())
    applied = method == "gw+edmft" and iteration > 1 and use_dyn and has_previous
    if applied:
        # H_ab = 2 sum_cd U_(ab),(cd) rho_dc; average equivalent shifts.
        h_new = (3 * m * np.sum(m * rho.T))[:, :, None]
        shift = h_new - hloc
        shift[0, 0, 0] = shift[1, 1, 0] = (shift[0, 0, 0] + shift[1, 1, 0]) / 2
        expected = old_one_body.real - np.kron(np.eye(2), shift[..., 0])
        np.testing.assert_allclose(params["hloc"]["one body"], expected, rtol=0, atol=1e-11)
        # Spin-up orbitals precede spin-down orbitals; mu is not re-centered.
        np.testing.assert_allclose(np.asarray(params["hloc"]["one body"])[:2, 2:], 0, atol=0)
        with h5py.File(ctqmc.hdf5file, "r") as handle:
            diagnostics = handle[f"{ctqmc.group}/SigHImp"]
            assert set(diagnostics) == {"sighimp_input.2.1", "sighimp_shift.2.1"}
            np.testing.assert_allclose(diagnostics["sighimp_input.2.1"][()], h_new, rtol=0, atol=1e-11)
            np.testing.assert_allclose(diagnostics["sighimp_shift.2.1"][()], shift, rtol=0, atol=1e-11)
            assert f"{ctqmc.group}/Mixing" not in handle
        assert "key=1 iter=2" in caplog.text
        assert "max|dH|=" in caplog.text
    else:
        np.testing.assert_array_equal(params["hloc"]["one body"], old_one_body.real)
        with h5py.File(ctqmc.hdf5file, "r") as handle:
            assert f"{ctqmc.group}/SigHImp" not in handle
        if not has_previous:
            assert "GImp/gimp.1.1" in caplog.text
            assert "WARNING" in caplog.text
    assert params["mu"] == old_mu
    np.testing.assert_array_equal(eimp.hloc, hloc)
    assert (workdir / "hyb.json").read_bytes() == written_hyb[0]
    assert json.loads((workdir / "hyb.json").read_text()) == hyb


@pytest.mark.parametrize("missing", ["control", "hloc", "hdf5file", "group"])
def test_current_bath_hartree_skips_missing_optional_context(tmp_path, missing):
    ctqmc = object.__new__(CTQMC)
    ctqmc.control = {"method": "gw+edmft"}
    ctqmc.bweiss = SimpleNamespace(cf=np.ones(1))
    ctqmc.fweiss = SimpleNamespace(eimp=SimpleNamespace(hloc=np.ones((1, 1, 1))))
    ctqmc.hdf5file = str(tmp_path / "unused.h5")
    ctqmc.group = "GWEDMFT"
    delattr(ctqmc.fweiss.eimp if missing == "hloc" else ctqmc, missing)

    assert ctqmc._current_bath_hartree_shift(iter=2, key="1") is None
    assert not (tmp_path / "unused.h5").exists()


def test_hybridization_tail_guard_rejects_nondecaying_constant():
    omega = np.arange(1.0, 33.0, 2.0)
    iw = 1j * omega
    ctqmc = object.__new__(CTQMC)
    ctqmc.key = "1"
    ctqmc.dlr = SimpleNamespace(MatsubaraFermionUniform=lambda: omega)
    ctqmc.fweiss = SimpleNamespace(
        h=(0.7 / iw + 0.2 / iw**2).reshape(1, 1, 1, -1)
    )

    ctqmc._validate_hybridization_tail(iteration=1)

    ctqmc.fweiss.h = ctqmc.fweiss.h - 3.7
    with pytest.raises(RuntimeError, match="non-decaying high-frequency constant"):
        ctqmc._validate_hybridization_tail(iteration=1)


def test_run_ctqmc_propagates_solver_exit_code(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QAssemble", str(tmp_path))
    monkeypatch.setattr(
        "QAssemble.CTQMC.subprocess.call", lambda *args, **kwargs: 7
    )
    ctqmc = object.__new__(CTQMC)
    ctqmc.control = {}

    with pytest.raises(SystemExit) as exc_info:
        ctqmc.RunCTQMC()

    assert exc_info.value.code == 7


@pytest.mark.parametrize(
    ("control", "expected"),
    [({}, (64, 64)), ({"CTQMCMPIRanks": 32, "MeasureMPIRanks": 16}, (32, 16))],
)
def test_solver_commands_use_independent_mpi_ranks(
    monkeypatch, tmp_path, caplog, control, expected
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QAssemble", str(tmp_path))
    commands = []

    def fake_call(command, **kwargs):
        commands.append(command)
        return 0

    monkeypatch.setattr("QAssemble.CTQMC.subprocess.call", fake_call)
    ctqmc = object.__new__(CTQMC)
    ctqmc.control = control
    with caplog.at_level("INFO", logger="QAssemble"):
        ctqmc.RunCTQMC()
        ctqmc.RunMeasure()

    assert commands == [
        f"mpirun -np {expected[0]} {tmp_path}/CTQMC/bin/CTQMC params",
        f"mpirun -np {expected[1]} {tmp_path}/CTQMC/bin/EVALSIM params",
    ]
    assert f"Running CTQMC with {expected[0]} MPI ranks" in caplog.text
    assert f"Running EVALSIM with {expected[1]} MPI ranks" in caplog.text


def test_ctqmc_finalize_only_saves_outputs():
    events = []
    ctqmc = object.__new__(CTQMC)
    ctqmc.bweiss = SimpleNamespace(cf=np.asarray([1.0]))
    ctqmc.gimp = _Output("gimp", events)
    ctqmc.sighimp = _Output("sighimp", events)
    ctqmc.sigfimp = _Output("sigfimp", events)
    ctqmc.sigimp = _Output("sigimp", events)
    ctqmc.chi = _Output("chi", events)
    ctqmc.pimp = _Output("pimp", events)
    ctqmc.wimp = _Output("wimp", events)

    ctqmc._finalize_outputs(iter=4)

    assert events == [
        ("gimp", "save", "gimp"),
        ("sighimp", "save", "sighimp"),
        ("sigfimp", "save", "sigfimp"),
        ("sigimp", "save", "sigimp"),
        ("chi", "save", "chi"),
        ("pimp", "save", "pimp"),
        ("wimp", "save", "wimp"),
    ]


def test_ctqmc_finalize_saves_only_present_static_outputs():
    events = []
    ctqmc = object.__new__(CTQMC)
    ctqmc.bweiss = SimpleNamespace(cf=None)
    ctqmc.gimp = _Output("gimp", events)
    ctqmc.sighimp = _Output("sighimp", events)
    ctqmc.sigfimp = _Output("sigfimp", events)
    ctqmc.sigimp = _Output("sigimp", events)
    ctqmc.chi = None
    ctqmc.pimp = None
    ctqmc.wimp = None

    ctqmc._finalize_outputs(iter=4)

    assert events == [
        ("gimp", "save", "gimp"),
        ("sighimp", "save", "sighimp"),
        ("sigfimp", "save", "sigfimp"),
        ("sigimp", "save", "sigimp"),
    ]


def test_ctqmc_finalize_skips_missing_optional_bosonic_outputs():
    events = []
    ctqmc = object.__new__(CTQMC)
    ctqmc.bweiss = SimpleNamespace(cf=None)
    ctqmc.gimp = _Output("gimp", events)
    ctqmc.sighimp = _Output("sighimp", events)
    ctqmc.sigfimp = _Output("sigfimp", events)
    ctqmc.sigimp = _Output("sigimp", events)
    ctqmc.chi = None
    ctqmc.pimp = None
    ctqmc.wimp = None

    ctqmc._finalize_outputs(iter=2)

    assert events == [
        ("gimp", "save", "gimp"),
        ("sighimp", "save", "sighimp"),
        ("sigfimp", "save", "sigfimp"),
        ("sigimp", "save", "sigimp"),
    ]
