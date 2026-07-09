from types import SimpleNamespace

import os
import numpy as np

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


def test_ctqmc_init_reads_mixing_parameters_from_run_control(monkeypatch, tmp_path):
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
            "mix_sig": 0.35,
            "mix_p": 0.45,
            "mixing_method": "linear",
            "npulay": 3,
        },
    )

    assert ctqmc.mix == 0.25
    assert ctqmc.mix_sig == 0.35
    assert ctqmc.mix_p == 0.45
    assert ctqmc.mixing_method == "linear"
    assert ctqmc.npulay == 3
    assert os.getcwd() == str(tmp_path / "ctqmc")


def test_ctqmc_finalize_mixes_selected_outputs_before_saving_all():
    events = []
    ctqmc = object.__new__(CTQMC)
    ctqmc.mix = 0.25
    ctqmc.mix_sig = 0.35
    ctqmc.mix_p = 0.45
    ctqmc.mixing_method = "linear"
    ctqmc.npulay = 3
    ctqmc.bweiss = SimpleNamespace(cf=np.asarray([1.0]))
    ctqmc.gimp = _Output("gimp", events)
    ctqmc.sighimp = _Output("sighimp", events)
    ctqmc.sigfimp = _Output("sigfimp", events)
    ctqmc.sigimp = _Output("sigimp", events)
    ctqmc.chi = _Output("chi", events)
    ctqmc.pimp = _Output("pimp", events)

    ctqmc._finalize_outputs(iter=4)

    assert events == [
        ("sighimp", "mix", 4, 0.35, "linear", 3),
        ("sigfimp", "mix", 4, 0.35, "linear", 3),
        ("sigimp", "mix", 4, 0.35, "linear", 3),
        ("pimp", "mix", 4, 0.45, "linear", 3),
        ("gimp", "save", "gimp"),
        ("sighimp", "save", "sighimp"),
        ("sigfimp", "save", "sigfimp"),
        ("sigimp", "save", "sigimp"),
        ("chi", "save", "chi"),
        ("pimp", "save", "pimp"),
    ]


def test_ctqmc_does_not_mix_p_for_static_bweiss():
    events = []
    ctqmc = object.__new__(CTQMC)
    ctqmc.mix = 0.25
    ctqmc.mix_sig = 0.35
    ctqmc.mix_p = 0.45
    ctqmc.mixing_method = "linear"
    ctqmc.npulay = 3
    ctqmc.bweiss = SimpleNamespace(cf=None)
    ctqmc.gimp = _Output("gimp", events)
    ctqmc.sighimp = _Output("sighimp", events)
    ctqmc.sigfimp = _Output("sigfimp", events)
    ctqmc.sigimp = _Output("sigimp", events)
    ctqmc.chi = None
    ctqmc.pimp = _Output("pimp", events)

    ctqmc._finalize_outputs(iter=4)

    assert events == [
        ("sighimp", "mix", 4, 0.35, "linear", 3),
        ("sigfimp", "mix", 4, 0.35, "linear", 3),
        ("sigimp", "mix", 4, 0.35, "linear", 3),
        ("gimp", "save", "gimp"),
        ("sighimp", "save", "sighimp"),
        ("sigfimp", "save", "sigfimp"),
        ("sigimp", "save", "sigimp"),
        ("pimp", "save", "pimp"),
    ]


def test_ctqmc_finalize_skips_missing_optional_bosonic_outputs():
    events = []
    ctqmc = object.__new__(CTQMC)
    ctqmc.mix = None
    ctqmc.mix_sig = None
    ctqmc.mix_p = None
    ctqmc.mixing_method = "pulay"
    ctqmc.npulay = 5
    ctqmc.bweiss = SimpleNamespace(cf=None)
    ctqmc.gimp = _Output("gimp", events)
    ctqmc.sighimp = _Output("sighimp", events)
    ctqmc.sigfimp = _Output("sigfimp", events)
    ctqmc.sigimp = _Output("sigimp", events)
    ctqmc.chi = None
    ctqmc.pimp = None

    ctqmc._finalize_outputs(iter=2)

    assert events == [
        ("gimp", "save", "gimp"),
        ("sighimp", "save", "sighimp"),
        ("sigfimp", "save", "sigfimp"),
        ("sigimp", "save", "sigimp"),
    ]
