from types import SimpleNamespace

import numpy as np

from QAssemble import Method as method_mod


class _SavedObject:
    def __init__(self, **attrs):
        self.saved = []
        self.mixed = []
        for key, value in attrs.items():
            setattr(self, key, value)

    def Save(self, name):
        self.saved.append(name)

    def Mixing(self, iter=None, mix=None, method="pulay", npulay=5):
        self.mixed.append(
            {
                "iter": iter,
                "mix": mix,
                "method": method,
                "npulay": npulay,
            }
        )


def _install_fake_ctqmc(monkeypatch, *, missing=(), include_bosons=False):
    class FakeCTQMC:
        instances = []

        def __init__(self, dlr, fweiss, bweiss, key, control=None, hdf5file=None, group=None):
            self.dlr = dlr
            self.fweiss = fweiss
            self.bweiss = bweiss
            self.key = key
            self.control = control
            self.hdf5file = hdf5file
            self.group = group
            self.calls = []
            self.gimp = None
            self.sighimp = None
            self.sigfimp = None
            self.sigimp = None
            self.chi = None
            self.pimp = None
            self.wimp = None
            FakeCTQMC.instances.append(self)

        def PreProcessing(self, iter):
            self.calls.append(("PreProcessing", iter))

        def Run(self, iter):
            self.calls.append(("Run", iter))

        def PostProcessing(self, iter):
            self.calls.append(("PostProcessing", iter))
            self.gimp = _SavedObject(f=np.asarray([10.0]))
            self.sighimp = _SavedObject(h=np.asarray([1.0]))
            self.sigfimp = _SavedObject(s=np.asarray([2.0]))
            self.sigimp = _SavedObject(f=np.asarray([3.0]))
            self.diagnostics = {"sign": 1.0}
            if include_bosons:
                self.chi = _SavedObject(f=np.asarray([4.0]))
                self.pimp = _SavedObject(f=np.asarray([5.0]))
                self.wimp = _SavedObject(f=np.asarray([6.0]))
            for attr in missing:
                setattr(self, attr, None)

    monkeypatch.setattr(method_mod, "CTQMC", FakeCTQMC)
    return FakeCTQMC


def _weiss_pair(tmp_path):
    fweiss = SimpleNamespace(
        dlr="dlr",
        key="imp0",
        hdf5file=str(tmp_path / "calc.h5"),
        group="impurity_solver",
    )
    bweiss = SimpleNamespace(key="imp0")
    return fweiss, bweiss


def test_impurity_action_runs_ctqmc_and_returns_outputs(monkeypatch, tmp_path):
    fake_ctqmc = _install_fake_ctqmc(monkeypatch)
    fweiss, bweiss = _weiss_pair(tmp_path)

    result = method_mod.ImpurityAction(
        fweiss,
        bweiss,
        control={"method": "dmft"},
        iteration=7,
    )()

    ctqmc = fake_ctqmc.instances[0]
    assert ctqmc.calls == [
        ("PreProcessing", 7),
        ("Run", 7),
        ("PostProcessing", 7),
    ]
    assert ctqmc.dlr == "dlr"
    assert ctqmc.key == "imp0"
    assert ctqmc.control == {"method": "dmft"}
    assert ctqmc.hdf5file == str(tmp_path / "calc.h5")
    assert ctqmc.group == "impurity_solver"
    assert isinstance(result, method_mod.ImpurityActionResult)
    assert result.ctqmc is ctqmc
    assert result.key == "imp0"
    np.testing.assert_allclose(result.gimp.f, [10.0])
    np.testing.assert_allclose(result.sighimp.h, [1.0])
    np.testing.assert_allclose(result.sigfimp.s, [2.0])
    np.testing.assert_allclose(result.sigimp.f, [3.0])
    assert result.chi is None
    assert result.pimp is None
    assert result.wimp is None
    assert result.diagnostics == {"sign": 1.0}


def test_impurity_action_leaves_output_finalization_to_ctqmc(monkeypatch, tmp_path):
    fake_ctqmc = _install_fake_ctqmc(monkeypatch, include_bosons=True)
    fweiss, bweiss = _weiss_pair(tmp_path)

    result = method_mod.ImpurityAction(fweiss, bweiss, iteration=2)()

    assert result.gimp.saved == []
    assert result.sighimp.saved == []
    assert result.sigfimp.saved == []
    assert result.sigimp.saved == []
    assert result.chi.saved == []
    assert result.pimp.saved == []
    assert result.wimp.saved == []
    assert fake_ctqmc.instances[0].calls[-1] == ("PostProcessing", 2)


def test_impurity_action_passes_control_to_ctqmc(monkeypatch, tmp_path):
    _install_fake_ctqmc(monkeypatch, include_bosons=True)
    fweiss, bweiss = _weiss_pair(tmp_path)
    control = {
        "mix": 0.25,
        "mixing_method": "linear",
        "npulay": 3,
    }

    result = method_mod.ImpurityAction(
        fweiss,
        bweiss,
        control=control,
        iteration=4,
    )()

    ctqmc = method_mod.CTQMC.instances[0]
    assert ctqmc.control is control
    assert result.gimp.mixed == []
    assert result.sighimp.mixed == []
    assert result.sigfimp.mixed == []
    assert result.sigimp.mixed == []
    assert result.chi.mixed == []
    assert result.pimp.mixed == []
    assert result.wimp.mixed == []


def test_impurity_action_can_return_missing_optional_outputs_without_validation(monkeypatch, tmp_path):
    _install_fake_ctqmc(monkeypatch, missing=("sigimp",))
    fweiss, bweiss = _weiss_pair(tmp_path)

    result = method_mod.ImpurityAction(fweiss, bweiss, iteration=3)()

    assert result.sigimp is None


def test_hf_result_supports_fields_and_tuple_unpacking():
    result = method_mod.HFResult(sigh="sigh", sigf="sigf")

    assert result.sigh == "sigh"
    assert result.sigf == "sigf"
    sigh, sigf = result
    assert sigh == "sigh"
    assert sigf == "sigf"


def test_hfloc_builds_local_hf_objects_and_saves_on_first_iteration(monkeypatch):
    def fake_ctor(label):
        def _ctor(**kwargs):
            return _SavedObject(label=label, kwargs=kwargs)

        return _ctor

    class FakeVLoc:
        def __init__(self):
            self.vproj = {}
            self.projector = None

        def BuildProjection(self, projector):
            self.projector = projector
            self.vproj["1"] = "vproj"

    monkeypatch.setattr(method_mod, "SigHLoc", fake_ctor("sighloc"))
    monkeypatch.setattr(method_mod, "SigFLoc", fake_ctor("sigfloc"))

    gloc = SimpleNamespace(
        crystal="crystal",
        projector="projector",
        key="1",
        occ="occ",
    )
    vloc = FakeVLoc()

    result = method_mod.HFLoc(
        gloc=gloc,
        vloc=vloc,
        key="1",
        hdf5file="calc.h5",
        group="hfloc",
        iteration=1,
    )()

    assert isinstance(result, method_mod.HFResult)
    assert vloc.projector == "projector"
    assert result.sigh.label == "sighloc"
    assert result.sigh.kwargs == {
        "crystal": "crystal",
        "projector": "projector",
        "key": "1",
        "occ": "occ",
        "vloc": "vproj",
        "hdf5file": "calc.h5",
        "group": "hfloc",
        "iteration": 1,
    }
    assert result.sigf.label == "sigfloc"
    assert result.sigf.kwargs == {
        "crystal": "crystal",
        "projector": "projector",
        "key": "1",
        "occ": "occ",
        "vloc": "vproj",
        "hdf5file": "calc.h5",
        "group": "hfloc",
        "iteration": 1,
    }
    assert result.sigh.saved == ["sighloc"]
    assert result.sigf.saved == ["sigfloc"]
    sigh, sigf = result
    assert (sigh, sigf) == (result.sigh, result.sigf)


def test_hfloc_skips_save_on_non_save_iteration(monkeypatch):
    def fake_ctor(label):
        def _ctor(**kwargs):
            return _SavedObject(label=label, kwargs=kwargs)

        return _ctor

    monkeypatch.setattr(method_mod, "SigHLoc", fake_ctor("sighloc"))
    monkeypatch.setattr(method_mod, "SigFLoc", fake_ctor("sigfloc"))

    gloc = SimpleNamespace(
        crystal="crystal",
        projector="projector",
        key="1",
        occ="occ",
    )
    vloc = SimpleNamespace(vproj={"1": "vproj"})

    result = method_mod.HFLoc(gloc=gloc, vloc=vloc, key="1", iteration=2)()

    assert result.sigh.saved == []
    assert result.sigf.saved == []


def test_gw_accepts_legacy_keywords_and_returns_unpackable_result(monkeypatch):
    def fake_ctor(label):
        def _ctor(**kwargs):
            return _SavedObject(label=label, kwargs=kwargs)

        return _ctor

    monkeypatch.setattr(method_mod, "SigGWC", fake_ctor("siggwc"))
    monkeypatch.setattr(method_mod, "P", fake_ctor("pol"))

    g = SimpleNamespace(
        crystal="crystal",
        dlr="dlr",
        rt="rt",
    )
    w = SimpleNamespace(crt="wcrt")

    result = method_mod.GW(Ginit=g, W=w, hdf5file="calc.h5", group="gw", iteration=1)()

    assert isinstance(result, method_mod.GWResult)
    assert result.siggwc.label == "siggwc"
    assert result.pol.label == "pol"
    siggwc, pol = result
    assert (siggwc, pol) == (
        result.siggwc,
        result.pol,
    )


def test_gwloc_builds_local_gw_objects_and_saves_on_first_iteration(monkeypatch):
    def fake_ctor(label):
        def _ctor(**kwargs):
            return _SavedObject(label=label, kwargs=kwargs)

        return _ctor

    monkeypatch.setattr(method_mod, "SigGWCLoc", fake_ctor("siggwcloc"))
    monkeypatch.setattr(method_mod, "PLoc", fake_ctor("ploc"))

    gloc = SimpleNamespace(
        crystal="crystal",
        dlr="dlr",
        projector="projector",
        key="1",
        t="gtau",
    )
    wloc = SimpleNamespace(key="1", ct="wct")

    result = method_mod.GWLoc(
        gloc=gloc,
        wloc=wloc,
        hdf5file="calc.h5",
        group="gwloc",
        iteration=1,
    )()

    assert isinstance(result, method_mod.GWResult)
    assert result.siggwc.label == "siggwcloc"
    assert result.siggwc.kwargs == {
        "crystal": "crystal",
        "dlr": "dlr",
        "projector": "projector",
        "key": "1",
        "green": "gtau",
        "wloc": "wct",
        "hdf5file": "calc.h5",
        "group": "gwloc",
        "iteration": 1,
    }
    assert result.pol.label == "ploc"
    assert result.pol.kwargs == {
        "crystal": "crystal",
        "dlr": "dlr",
        "projector": "projector",
        "key": "1",
        "gloc": "gtau",
        "hdf5file": "calc.h5",
        "group": "gwloc",
        "iteration": 1,
    }
    assert result.siggwc.saved == ["siggwcloc.f"]
    assert result.pol.saved == ["ploc.f"]
    siggwc, pol = result
    assert (siggwc, pol) == (result.siggwc, result.pol)


def test_gwloc_skips_save_on_non_save_iteration(monkeypatch):
    def fake_ctor(label):
        def _ctor(**kwargs):
            return _SavedObject(label=label, kwargs=kwargs)

        return _ctor

    monkeypatch.setattr(method_mod, "SigGWCLoc", fake_ctor("siggwcloc"))
    monkeypatch.setattr(method_mod, "PLoc", fake_ctor("ploc"))

    gloc = SimpleNamespace(
        crystal="crystal",
        dlr="dlr",
        projector="projector",
        key="1",
        t="gtau",
    )
    wloc = SimpleNamespace(key="1", ct="wct")

    result = method_mod.GWLoc(gloc=gloc, wloc=wloc, iteration=2)()

    assert result.siggwc.saved == []
    assert result.pol.saved == []
