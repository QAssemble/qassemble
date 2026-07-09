import importlib
from types import SimpleNamespace

import numpy as np
import pytest


class _SavedObject:
    def __init__(self, **attrs):
        self.saved = []
        self.mixing_calls = []
        for key, value in attrs.items():
            setattr(self, key, value)

    def Save(self, name, *args, **kwargs):
        self.saved.append((name, args, kwargs))

    def Mixing(self, **kwargs):
        self.mixing_calls.append(kwargs)


class _FakeConvergence:
    def __init__(self, converge_after=1):
        self.converge_after = converge_after
        self.min_iter = 0
        self.start_iters = []
        self.self_checks = []
        self.cross_checks = []
        self.self_hdf5_checks = []
        self.cross_hdf5_checks = []
        self.diagnostics = None

    def Start(self):
        pass

    def StartIter(self, iter, ready_after=None):
        self.iter = iter
        self.start_iters.append(iter)

    def CheckSelf(self, name, value, kind):
        self.self_checks.append((name, value, kind))

    def CheckCross(self, name_a, a, name_b, b, kind):
        self.cross_checks.append((name_a, name_b, a, b, kind))

    def CheckSelfHDF5(self, name, **kwargs):
        self.self_hdf5_checks.append((name, kwargs))

    def CheckCrossHDF5(self, name_a, **kwargs):
        self.cross_hdf5_checks.append((name_a, kwargs))

    def RecordDiagnostics(self, diagnostics):
        self.diagnostics = diagnostics

    def Commit(self, iter, will_continue):
        return iter >= self.converge_after, {
            "self": {
                "GLoc": {"abs": 0.0},
                "mu": {"abs": 0.0},
                "WLoc": {"abs": 0.0},
            },
            "cross": {
                "GLoc-GImp": {"abs": 0.0},
            },
        }


def _install_fake_edmft_stack(monkeypatch, *, missing=()):
    cf_mod = importlib.import_module("QAssemble.CorrelationFunction")
    missing = set(missing)

    class FakeProjector:
        def __init__(self, basisindex=None, impdict=None, equiv=None):
            self.basisindex = basisindex
            self.impdict = impdict
            self.equiv = {"1": np.eye(1, dtype=int)}
            self.fprojector = {"1": np.ones((1, 1, 1), dtype=float)}
            self.bprojector = {"1": np.ones((1, 1, 1), dtype=float)}

    class FakeG:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeG.instances.append(self)
            self.args = args
            self.kwargs = kwargs
            self.kf = np.asarray([len(FakeG.instances)], dtype=np.complex128)
            self.mu = 0.25
            self.saved = []

        def Save(self, name, *args, **kwargs):
            self.saved.append((name, args, kwargs))

    class FakeGLoc:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeGLoc.instances.append(self)
            self.args = args
            self.kwargs = kwargs
            self.key = kwargs["key"]
            idx = len(FakeGLoc.instances)
            self.f = np.asarray([10.0 + idx], dtype=np.complex128)
            self.t = np.asarray([20.0 + idx], dtype=np.complex128)
            self.saved = []

        def Projection(self, matin, key):
            return np.ones((1, 1, 1), dtype=np.complex128) * np.asarray(matin).flat[0]

        def Save(self, name, *args, **kwargs):
            self.saved.append((name, args, kwargs))

    class FakePLoc:
        instances = []

        def __init__(self, *args, **kwargs):
            FakePLoc.instances.append(self)
            self.args = args
            self.kwargs = kwargs
            self.f = np.ones((1, 1, 1, 1, 2), dtype=np.complex128) * 0.2
            self.saved = []

        def Save(self, name, *args, **kwargs):
            self.saved.append((name, args, kwargs))

    class FakeWLoc:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeWLoc.instances.append(self)
            self.args = args
            self.kwargs = kwargs
            self.f = np.asarray(kwargs["pol"], dtype=np.complex128) + 1.0
            self.saved = []

        def Save(self, name, *args, **kwargs):
            self.saved.append((name, args, kwargs))

    class FakeEImp:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeEImp.instances.append(self)
            self.args = args
            self.kwargs = kwargs
            self.e = np.asarray([0.0])
            self.saved = []

        def Projection(self, matin, key):
            return np.ones((1, 1, 1), dtype=np.complex128) * np.asarray(matin).flat[0]

        def Save(self, name, *args, **kwargs):
            self.saved.append((name, args, kwargs))

    class FakeHyb:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeHyb.instances.append(self)
            self.args = args
            self.kwargs = kwargs
            self.f = np.asarray([0.1])
            self.saved = []

        def Save(self, name, *args, **kwargs):
            self.saved.append((name, args, kwargs))

    class FakeFWeiss:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeFWeiss.instances.append(self)
            self.args = args
            self.kwargs = kwargs

    class FakeBWeiss:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeBWeiss.instances.append(self)
            self.args = args
            self.kwargs = kwargs
            self.ploc = kwargs["ploc"]
            self.wloc = kwargs["wloc"]
            self.cf = np.asarray([1.0])
            self.saved = []

        def Save(self, name, *args, **kwargs):
            self.saved.append((name, args, kwargs))

    class FakeSigC:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeSigC.instances.append(self)
            self.args = args
            self.kwargs = kwargs
            self.embedded = []
            self.kf = np.asarray([0.0])
            self.sigh = np.asarray([1.0])
            self.sigf = np.asarray([2.0])
            self.sigimp = np.asarray([3.0])

        def ImpEmbedding(self, **kwargs):
            self.embedded.append(kwargs)

        def __call__(self):
            self.kf = np.asarray([4.0])

    class FakeImpurityAction:
        instances = []
        results = []

        def __init__(self, *args, **kwargs):
            FakeImpurityAction.instances.append(self)
            self.args = args
            self.kwargs = kwargs

        def __call__(self):
            iter_no = float(self.kwargs["iteration"])
            result = SimpleNamespace(
                ctqmc=SimpleNamespace(),
                diagnostics={"sign": iter_no, "nimp": iter_no},
                gimp=_SavedObject(f=np.asarray([1.0])),
                sighimp=_SavedObject(h=np.asarray([2.0])),
                sigfimp=_SavedObject(s=np.asarray([3.0])),
                sigimp=_SavedObject(f=np.asarray([4.0])),
                chi=_SavedObject(f=np.asarray([5.0])),
                pimp=_SavedObject(
                    f=np.ones((1, 1, 1, 1, 2), dtype=np.complex128) * 0.4
                ),
            )
            for attr in missing:
                setattr(result, attr, None)
            FakeImpurityAction.results.append(result)
            return result

    monkeypatch.setattr(cf_mod, "Projector", FakeProjector)
    monkeypatch.setattr(cf_mod, "G", FakeG)
    monkeypatch.setattr(cf_mod, "GLoc", FakeGLoc)
    monkeypatch.setattr(cf_mod, "PLoc", FakePLoc)
    monkeypatch.setattr(cf_mod, "WLoc", FakeWLoc)
    monkeypatch.setattr(cf_mod, "EImp", FakeEImp)
    monkeypatch.setattr(cf_mod, "Hyb", FakeHyb)
    monkeypatch.setattr(cf_mod, "FWeiss", FakeFWeiss)
    monkeypatch.setattr(cf_mod, "BWeiss", FakeBWeiss)
    monkeypatch.setattr(cf_mod, "SigC", FakeSigC)
    monkeypatch.setattr(cf_mod, "ImpurityAction", FakeImpurityAction)

    return SimpleNamespace(
        cf_mod=cf_mod,
        G=FakeG,
        GLoc=FakeGLoc,
        PLoc=FakePLoc,
        WLoc=FakeWLoc,
        Hyb=FakeHyb,
        BWeiss=FakeBWeiss,
        SigC=FakeSigC,
        ImpurityAction=FakeImpurityAction,
    )


class _FakeVLoc:
    def __init__(self):
        self.projector = None
        self.vproj = {}

    def BuildProjection(self, projector):
        self.projector = projector
        self.vproj = {
            "1": np.ones((1, 1, 1, 1), dtype=np.complex128),
        }
        return self.vproj


def _edmft_correlation_object(cf_mod, tmp_path, *, nscf=1, conv=None):
    corr = object.__new__(cf_mod.CorrelationFunction)
    corr.control = {
        "run": {
            "nscf": nscf,
            "fn": str(tmp_path / "calc"),
            "mix": 0.5,
            "mixing_method": "pulay",
            "npulay": 5,
        },
        "impurity": {
            "impdict": {"1": [[(0, 0)]]},
            "equiv": {"1": np.eye(1, dtype=int)},
        },
    }
    corr.crystal = SimpleNamespace(_basis_index="basis")
    corr.dlr = SimpleNamespace(nu=np.asarray([0.0, 1.0]))
    corr.c = 1.0
    corr.niham = SimpleNamespace(k=np.asarray([0.0]))
    corr.greenbare = SimpleNamespace(kf=np.asarray([0.0]))
    corr.vbare = SimpleNamespace(vloc=_FakeVLoc())
    corr.conv = conv if conv is not None else _FakeConvergence()
    return corr


def _dmft_correlation_object(cf_mod, tmp_path, *, nscf=1, conv=None):
    corr = _edmft_correlation_object(
        cf_mod,
        tmp_path,
        nscf=nscf,
        conv=conv,
    )
    return corr


def test_dmft_projects_gloc_in_problem_loop_and_uses_hdf5_convergence(
    monkeypatch,
    tmp_path,
):
    stack = _install_fake_edmft_stack(monkeypatch)
    conv = _FakeConvergence(converge_after=99)
    corr = _dmft_correlation_object(
        stack.cf_mod,
        tmp_path,
        nscf=2,
        conv=conv,
    )

    corr.DMFT()

    assert [item.kwargs["key"] for item in stack.GLoc.instances] == ["1", "1"]
    assert [item.kwargs["iteration"] for item in stack.GLoc.instances] == [0, 1]
    assert stack.Hyb.instances[0].kwargs["green"] is stack.GLoc.instances[0].f
    assert stack.Hyb.instances[1].kwargs["green"] is stack.GLoc.instances[1].f

    assert conv.start_iters == [1]
    assert conv.self_hdf5_checks == [
        (
            "GLoc",
            {
                "group": "dmft",
                "subgroup": "GLoc",
                "current": "gloc.1",
                "previous": "gloc.0",
                "keys": ["1"],
            },
        )
    ]
    assert conv.cross_hdf5_checks == [
        (
            "GLoc",
            {
                "name_b": "GImp",
                "group": "dmft",
                "subgroup_a": "GLoc",
                "subgroup_b": "GImp",
                "stem_a": "gloc.1",
                "stem_b": "gimp.1",
                "keys": ["1"],
            },
        )
    ]
    assert [name for name, _, _ in conv.self_checks] == ["mu"]
    assert conv.cross_checks == []
    assert conv.diagnostics == {"1": {"sign": 1.0, "nimp": 1.0}}


def test_dmft_skips_convergence_on_first_loop(monkeypatch, tmp_path):
    stack = _install_fake_edmft_stack(monkeypatch)
    conv = _FakeConvergence(converge_after=99)
    corr = _dmft_correlation_object(
        stack.cf_mod,
        tmp_path,
        nscf=1,
        conv=conv,
    )

    corr.DMFT()

    assert [item.kwargs["iteration"] for item in stack.GLoc.instances] == [0]
    assert conv.start_iters == []
    assert conv.self_hdf5_checks == []
    assert conv.cross_hdf5_checks == []


def test_edmft_uses_initial_dynamic_bath_and_wloc_self_check(monkeypatch, tmp_path):
    stack = _install_fake_edmft_stack(monkeypatch)
    corr = _edmft_correlation_object(stack.cf_mod, tmp_path)

    corr.EDMFT()

    assert len(stack.PLoc.instances) == 1
    assert [item.kwargs["key"] for item in stack.GLoc.instances] == ["1", "1"]
    assert len(stack.WLoc.instances) == 2
    assert len(stack.BWeiss.instances) == 1
    assert stack.BWeiss.instances[0].ploc is stack.PLoc.instances[0]
    assert stack.BWeiss.instances[0].wloc is stack.WLoc.instances[0]
    assert stack.Hyb.instances[0].kwargs["green"] is stack.GLoc.instances[0].f
    assert stack.PLoc.instances[0].kwargs["gloc"] is stack.GLoc.instances[0].t
    assert stack.WLoc.instances[0].kwargs["pol"] is stack.PLoc.instances[0].f

    result = stack.ImpurityAction.results[0]
    assert result.sighimp.mixing_calls == []
    assert result.sigfimp.mixing_calls == []
    assert result.sigimp.mixing_calls == []
    assert result.pimp.mixing_calls == []

    self_check_names = [name for name, _, _ in corr.conv.self_checks]
    assert "GLoc" in self_check_names
    assert "mu" in self_check_names
    assert "WLoc" in self_check_names
    gloc_self = next(value for name, value, _ in corr.conv.self_checks if name == "GLoc")
    assert gloc_self["1"] is stack.GLoc.instances[-1].f
    assert len(corr.conv.cross_checks) == 1
    name_a, name_b, cross_a, cross_b, kind = corr.conv.cross_checks[0]
    assert (name_a, name_b, kind) == ("GLoc", "GImp", "dict")
    assert cross_a["1"] is stack.GLoc.instances[-1].f
    np.testing.assert_allclose(cross_b["1"], [1.0])
    assert all("WLoc" not in item[:2] for item in corr.conv.cross_checks)


def test_edmft_reuses_previous_mixed_pimp_as_next_boson_input(monkeypatch, tmp_path):
    stack = _install_fake_edmft_stack(monkeypatch)
    corr = _edmft_correlation_object(
        stack.cf_mod,
        tmp_path,
        nscf=2,
        conv=_FakeConvergence(converge_after=2),
    )

    corr.EDMFT()

    assert len(stack.PLoc.instances) == 1
    assert len(stack.BWeiss.instances) == 2
    assert stack.BWeiss.instances[0].ploc is stack.PLoc.instances[0]
    assert stack.BWeiss.instances[1].ploc is stack.ImpurityAction.results[0].pimp


@pytest.mark.parametrize("missing_attr", ["chi", "pimp"])
def test_edmft_requires_bosonic_impurity_outputs(monkeypatch, tmp_path, missing_attr):
    stack = _install_fake_edmft_stack(monkeypatch, missing=(missing_attr,))
    corr = _edmft_correlation_object(stack.cf_mod, tmp_path)

    with pytest.raises(RuntimeError, match=f"no {missing_attr}"):
        corr.EDMFT()


def test_run_dispatch_calls_edmft(monkeypatch):
    run_mod = importlib.import_module("QAssemble.Run")
    called = []

    class FakeCorrelationFunction:
        def __init__(self, control):
            self.control = control

        def EDMFT(self):
            called.append(self.control)

    monkeypatch.setattr(run_mod, "CorrelationFunction", FakeCorrelationFunction)

    runner = object.__new__(run_mod.Run)
    runner.control = {"run": {"method": "edmft"}}

    runner.RunDiagE()

    assert called == [runner.control]
