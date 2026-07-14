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
