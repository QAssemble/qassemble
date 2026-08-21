from types import SimpleNamespace

import os
import numpy as np
import pytest

from QAssemble.CTQMC import CTQMC


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


def test_preprocessing_mixes_inputs_before_writing_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    events = []
    control = {"mix": 0.25, "mixing_method": "linear", "npulay": 3}

    ctqmc = object.__new__(CTQMC)
    ctqmc.key = "1"
    ctqmc.control = control
    ctqmc.crystal = SimpleNamespace(soc=False, ns=1)
    ctqmc.projector = _fake_projector()
    ctqmc.dlr = SimpleNamespace(
        beta=10.0,
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

    # Both inputs are mixed strictly before their json files are written, and
    # each Mixing receives the run control (hyb additionally the iteration).
    assert events == [
        ("fweiss_mix", 2, control),
        ("write", "hyb", 2, "1"),
        ("bweiss_mix", control),
        ("write", "dyn", 2, "1"),
    ]


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
        beta=10.0,
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

    with pytest.raises(SystemExit) as exc_info:
        ctqmc.RunCTQMC()

    assert exc_info.value.code == 7


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
