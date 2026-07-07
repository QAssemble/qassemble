import importlib
from types import SimpleNamespace

import numpy as np

from QAssemble import Method as method_mod


class _SavedObject:
    def __init__(self, **attrs):
        self.saved = []
        for key, value in attrs.items():
            setattr(self, key, value)

    def Save(self, name, *args, **kwargs):
        self.saved.append((name, args, kwargs))


class _FakeConvergence:
    def __init__(self):
        self.seeded = []
        self.checked = []
        self._conv_hdf5_group = None

    def Start(self):
        pass

    def seed_prev(self, name, value, kind):
        self.seeded.append((name, value, kind))

    def StartIter(self, iter, ready_after=None):
        self.iter = iter
        self.ready_after = ready_after

    def CheckSelf(self, name, value, kind):
        self.checked.append((name, value, kind))

    def RecordDiagnostics(self, diagnostics):
        self.diagnostics = diagnostics

    def Commit(self, iter, will_continue):
        return True, {
            "self": {
                "F": {"abs": 0.0},
                "B": {"abs": 0.0},
            }
        }


def test_gw_approximation_uses_current_hf_and_gw_results(monkeypatch):
    cf_mod = importlib.import_module("QAssemble.CorrelationFunction")
    calls = {
        "G": [],
        "W": [],
        "HF": [],
        "GW": [],
    }

    occ = np.asarray([1.0])
    occr = np.asarray([2.0])
    g_initial_kf = np.asarray([3.0])
    g_final_kf = np.asarray([4.0])
    w_initial_kf = np.asarray([5.0])
    w_final_kf = np.asarray([6.0])
    sigh_k = np.asarray([7.0])
    sigf_k = np.asarray([8.0])
    siggwc_kf = np.asarray([9.0])
    pol_kf = np.asarray([10.0])

    class FakeG:
        instances = []

        def __init__(self, *args, **kwargs):
            calls["G"].append({"args": args, "kwargs": kwargs})
            FakeG.instances.append(self)
            self.saved = []
            self.subgroup = "G"
            self.kf = g_initial_kf if len(calls["G"]) == 1 else g_final_kf
            self.rt = np.asarray([11.0])
            self.occ = occ
            self.occr = occr
            self.mu = 0.25

        def Save(self, name, *args, **kwargs):
            self.saved.append((name, args, kwargs))

    class FakeW:
        instances = []

        def __init__(self, **kwargs):
            calls["W"].append(kwargs)
            FakeW.instances.append(self)
            self.saved = []
            self.kf = w_initial_kf if len(calls["W"]) == 1 else w_final_kf

        def Save(self, name, *args, **kwargs):
            self.saved.append((name, args, kwargs))

    def fake_hf(**kwargs):
        calls["HF"].append(kwargs)

        class FakeHF:
            def __call__(self):
                return method_mod.HFResult(
                    sigh=_SavedObject(k=sigh_k),
                    sigf=_SavedObject(k=sigf_k),
                )

        return FakeHF()

    def fake_gw(**kwargs):
        calls["GW"].append(kwargs)

        class FakeGW:
            def __call__(self):
                return method_mod.GWResult(
                    siggwc=_SavedObject(kf=siggwc_kf),
                    pol=_SavedObject(kf=pol_kf),
                )

        return FakeGW()

    monkeypatch.setattr(cf_mod, "G", FakeG)
    monkeypatch.setattr(cf_mod, "W", FakeW)
    monkeypatch.setattr(cf_mod, "HF", fake_hf)
    monkeypatch.setattr(cf_mod, "GW", fake_gw)

    corr = object.__new__(cf_mod.CorrelationFunction)
    corr.control = {
        "run": {
            "nscf": 1,
            "mix": 0.5,
            "fn": "calc",
            "mixing_method": "linear",
            "npulay": 2,
        }
    }
    corr.crystal = "crystal"
    corr.dlr = SimpleNamespace(nu=[0, 1])
    corr.c = 1.0
    corr.niham = SimpleNamespace()
    corr.greenbare = SimpleNamespace(kf=np.asarray([12.0]))
    corr.vbare = SimpleNamespace(k=np.zeros((1, 1, 1, 1)))
    corr.conv = _FakeConvergence()

    corr.GWApproximation()

    assert len(calls["HF"]) == 1
    assert calls["HF"][0]["occ"] is occ
    assert calls["HF"][0]["occr"] is occr
    assert calls["HF"][0]["v"] is corr.vbare
    assert calls["HF"][0]["hdf5file"] == "calc.h5"
    assert calls["HF"][0]["group"] == "gw"
    assert calls["HF"][0]["iteration"] == 1
    assert calls["HF"][0]["mix"] == 0.5
    assert calls["HF"][0]["mixing_method"] == "linear"
    assert calls["HF"][0]["npulay"] == 2
    assert len(calls["GW"]) == 1
    assert calls["GW"][0]["g"] is FakeG.instances[0]
    assert calls["GW"][0]["w"] is FakeW.instances[0]
    assert calls["GW"][0]["hdf5file"] == "calc.h5"
    assert calls["GW"][0]["group"] == "gw"
    assert calls["GW"][0]["iteration"] == 1
    assert calls["GW"][0]["mix"] == 0.5
    assert calls["GW"][0]["mixing_method"] == "linear"
    assert calls["GW"][0]["npulay"] == 2
    assert calls["G"][1]["args"][3] is sigh_k
    assert calls["G"][1]["args"][4] is sigf_k
    assert calls["G"][1]["args"][5] is siggwc_kf
    assert calls["W"][1]["pol"] is pol_kf
    assert corr.green.kf is g_final_kf
    assert corr.w.kf is w_final_kf
    assert corr.pol.kf is pol_kf
    assert corr.siggwc.kf is siggwc_kf
    assert corr.sigh.k is sigh_k
    assert corr.sigf.k is sigf_k
    assert [item[0] for item in corr.sigh.saved] == ["sigh"]
    assert [item[0] for item in corr.sigf.saved] == ["sigf"]
    assert [item[0] for item in corr.siggwc.saved] == ["siggwckf", "siggwckf"]
    assert FakeG.instances[0].subgroup == "G"
    assert FakeG.instances[1].subgroup == "G"


def test_gw_timing_constructor_targets_are_decorated():
    from QAssemble.BLatDyn import P, W
    from QAssemble.FLatDyn import G, G0, SigGWC
    from QAssemble.FLatStc import SigF, SigH

    for cls in (G0, G, SigGWC, SigH, SigF, P, W):
        assert hasattr(cls.__init__, "__wrapped__")
