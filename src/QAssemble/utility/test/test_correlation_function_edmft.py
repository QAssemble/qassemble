import importlib
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from QAssemble.FLocStc import EImp
from QAssemble.utility.Projection import Projection as PJ


class _SavedObject:
    def __init__(self, **attrs):
        self.saved = []
        self.mixing_calls = []
        self.projection_calls = []
        for key, value in attrs.items():
            setattr(self, key, value)

    def Save(self, name, *args, **kwargs):
        self.saved.append((name, args, kwargs))

    def Mixing(self, **kwargs):
        self.mixing_calls.append(kwargs)
        return getattr(self, "mixing_result", kwargs["value"])

    def Projection(self, matin, key):
        self.projection_calls.append((matin, key))
        if hasattr(self, "projector"):
            return PJ.BLatDyn(matin, self.projector.bprojector[str(key)])
        return np.asarray(matin)


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


def _install_fake_edmft_stack(monkeypatch, *, missing=(), hdf5_offset=0.0):
    cf_mod = importlib.import_module("QAssemble.CorrelationFunction")
    missing = set(missing)

    class FakeProjector:
        def __init__(self, basisindex=None, impdict=None, equiv=None):
            self.basisindex = basisindex
            self.impdict = impdict
            keys = [str(key) for key in impdict]
            self.equiv = {key: np.eye(1, dtype=int) for key in keys}
            self.fprojector = {
                key: np.ones((1, 1, 1), dtype=float) for key in keys
            }
            self.bprojector = {}
            for index, key in enumerate(keys):
                projection = np.zeros((len(keys), 1, 1), dtype=float)
                projection[index, 0, 0] = 1.0
                self.bprojector[key] = projection

    class FakeG:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeG.instances.append(self)
            self.args = args
            self.kwargs = kwargs
            self.kf = np.asarray([len(FakeG.instances)], dtype=np.complex128)
            self.mu = 0.25
            self.occ = np.asarray([1.0])
            self.occr = np.asarray([2.0])
            self.rt = np.asarray([3.0])
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
            self.crystal = kwargs["crystal"]
            self.dlr = kwargs["dlr"]
            self.projector = kwargs["projector"]
            idx = len(FakeGLoc.instances)
            self.f = np.asarray([10.0 + idx], dtype=np.complex128)
            self.t = np.asarray([20.0 + idx], dtype=np.complex128)
            self.occ = np.asarray([1.0])
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
            self.f = np.mean(np.asarray(kwargs["wlat"], dtype=np.complex128), axis=4)
            self.ct = np.asarray([7.0])
            self.key = kwargs["key"]
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
            self.p = kwargs.get("p")
            self.w = kwargs["w"]
            self.f = None if self.w is None else np.asarray(self.w.f)
            self.cf = None if self.w is None else np.asarray([1.0])
            self.saved = []

        def Save(self, name, *args, **kwargs):
            self.saved.append((name, args, kwargs))

        def Dyson(self, value, polarization):
            return np.asarray(value) + np.asarray(polarization)

    class FakeW:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeW.instances.append(self)
            self.kwargs = kwargs
            self.is_bare = kwargs.get("pol") is None
            value = 1.0 if self.is_bare else 2.0
            self.kf = np.ones((1, 1, 1, 1, 1, 2), dtype=np.complex128) * value
            self.crt = np.asarray([4.0])
            self.saved = []

        def Save(self, name, *args, **kwargs):
            self.saved.append((name, args, kwargs))

    class FakePolC:
        instances = []

        def __init__(self, *args, **kwargs):
            FakePolC.instances.append(self)
            self.embedded = []
            self.kf = np.ones((1, 1, 1, 1, 1, 2), dtype=np.complex128) * 0.4
            self.pimp = None

        def ImpEmbedding(self, value, projector, key):
            self.embedded.append((value, projector, key))
            local = np.asarray(
                value.f if hasattr(value, "f") else value,
                dtype=np.complex128,
            )
            projection = projector.bprojector[str(key)][:, 0, 0]
            if self.pimp is None:
                norb = projection.shape[0]
                self.pimp = np.zeros(
                    (norb, norb, 1, 1, 1, local.shape[-1]),
                    dtype=np.complex128,
                )
            orbital_block = projection[:, None] * projection.conj()[None, :]
            self.pimp[:, :, 0, 0, 0, :] += (
                orbital_block[..., None] * local[0, 0, 0, 0, :]
            )

        def GWContribution(self, value):
            self.gw = value

        def GWDoubleCounting(self, value, projector, key):
            self.dc = (value, projector, key)

        def __call__(self):
            return self.kf

    class FakeHF:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeHF.instances.append(self)
            self.kwargs = kwargs

        def __call__(self):
            return SimpleNamespace(
                sigh=_SavedObject(k=np.asarray([10.0])),
                sigf=_SavedObject(k=np.asarray([20.0])),
            )

    class FakeGW:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeGW.instances.append(self)
            self.kwargs = kwargs

        def __call__(self):
            return SimpleNamespace(
                siggwc=_SavedObject(kf=np.asarray([30.0])),
                pol=_SavedObject(kf=np.asarray([40.0])),
            )

    class FakeHFLoc:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeHFLoc.instances.append(self)
            self.kwargs = kwargs

        def __call__(self):
            self.result = SimpleNamespace(
                sigh=_SavedObject(hloc=np.asarray([50.0])),
                sigf=_SavedObject(floc=np.asarray([5.0])),
            )
            return self.result

    class FakeGWLoc:
        instances = []

        def __init__(self, *args, **kwargs):
            FakeGWLoc.instances.append(self)
            self.kwargs = kwargs

        def __call__(self):
            self.result = SimpleNamespace(
                siggwc=_SavedObject(f=np.asarray([6.0])),
                pol=_SavedObject(
                    f=np.ones((1, 1, 1, 1, 2), dtype=np.complex128) * 0.2,
                    projector=self.kwargs["gloc"].projector,
                ),
            )
            return self.result

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
            key_no = float(self.kwargs["key"])
            key_offset = 10.0 * (key_no - 1.0)
            result = SimpleNamespace(
                ctqmc=SimpleNamespace(),
                diagnostics={"sign": iter_no, "nimp": iter_no},
                gimp=_SavedObject(f=np.asarray([1.0])),
                sighimp=_SavedObject(h=np.asarray([2.0 + key_offset])),
                sigfimp=_SavedObject(s=np.asarray([3.0 + key_offset])),
                sigimp=_SavedObject(f=np.asarray([4.0 + key_offset])),
                chi=_SavedObject(f=np.asarray([5.0])),
                pimp=_SavedObject(
                    f=np.ones((1, 1, 1, 1, 2), dtype=np.complex128)
                    * (0.3 + 0.1 * key_no)
                ),
                wimp=_SavedObject(
                    f=np.ones((1, 1, 1, 1, 2), dtype=np.complex128) * 0.6
                ),
            )
            for attr in missing:
                setattr(result, attr, None)
            with h5py.File(self.kwargs["hdf5file"], "a") as handle:
                for subgroup, name, attr, field in (
                    ("SigHImp", "sighimp", "sighimp", "h"),
                    ("SigFImp", "sigfimp", "sigfimp", "s"),
                    ("SigCImp", "sigimp", "sigimp", "f"),
                    ("PImp", "pimp", "pimp", "f"),
                ):
                    obj = getattr(result, attr)
                    if obj is not None:
                        handle.require_group(
                            f"{self.kwargs['group']}/{subgroup}"
                        ).create_dataset(
                            f"{name}.{int(iter_no)}.{self.kwargs['key']}",
                            data=getattr(obj, field) + hdf5_offset,
                        )
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
    monkeypatch.setattr(cf_mod, "W", FakeW)
    monkeypatch.setattr(cf_mod, "PolC", FakePolC)
    monkeypatch.setattr(cf_mod, "SigC", FakeSigC)
    monkeypatch.setattr(cf_mod, "HF", FakeHF)
    monkeypatch.setattr(cf_mod, "GW", FakeGW)
    monkeypatch.setattr(cf_mod, "HFLoc", FakeHFLoc)
    monkeypatch.setattr(cf_mod, "GWLoc", FakeGWLoc)
    monkeypatch.setattr(cf_mod, "ImpurityAction", FakeImpurityAction)

    return SimpleNamespace(
        cf_mod=cf_mod,
        G=FakeG,
        GLoc=FakeGLoc,
        PLoc=FakePLoc,
        WLoc=FakeWLoc,
        Hyb=FakeHyb,
        BWeiss=FakeBWeiss,
        W=FakeW,
        PolC=FakePolC,
        SigC=FakeSigC,
        HF=FakeHF,
        GW=FakeGW,
        HFLoc=FakeHFLoc,
        GWLoc=FakeGWLoc,
        ImpurityAction=FakeImpurityAction,
    )


class _FakeVLoc:
    def __init__(self):
        self.projector = None
        self.vproj = {}

    def BuildProjection(self, projector):
        self.projector = projector
        self.vproj = {
            key: np.ones((1, 1, 1, 1), dtype=np.complex128)
            for key in projector.bprojector
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


def test_eimp_static_moment_subtracts_impurity_hartree_and_fock_dc():
    projector = SimpleNamespace(
        fprojector={"1": np.ones((1, 1, 1), dtype=np.complex128)}
    )
    hamtb = np.asarray([4.0, 8.0], dtype=np.complex128).reshape(1, 1, 1, 2)
    sigh = np.asarray([2.0, 4.0], dtype=np.complex128).reshape(1, 1, 1, 2)
    sigf = np.asarray([-1.0, 3.0], dtype=np.complex128).reshape(1, 1, 1, 2)
    sigma_imp_h = np.asarray([11.0], dtype=np.complex128).reshape(1, 1, 1)
    sigma_dc_f = np.asarray([-3.0], dtype=np.complex128).reshape(1, 1, 1)

    eimp = EImp(
        crystal=SimpleNamespace(),
        projector=projector,
        key="1",
        hamtb=hamtb,
        mu=1.0,
        sigh=sigh,
        sigf=sigf,
        hloc=sigma_imp_h,
        floc=sigma_dc_f,
    )

    lattice_hf_moment = np.mean(hamtb - 1.0 + sigh + sigf, axis=3)
    np.testing.assert_allclose(
        eimp.e,
        lattice_hf_moment - sigma_imp_h - sigma_dc_f,
    )


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
        ),
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
        ),
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


def test_edmft_uses_initial_dynamic_bath_and_skips_first_convergence(
    monkeypatch,
    tmp_path,
):
    stack = _install_fake_edmft_stack(monkeypatch)
    corr = _edmft_correlation_object(stack.cf_mod, tmp_path)

    corr.EDMFT()

    assert len(stack.PLoc.instances) == 0
    assert [item.kwargs["key"] for item in stack.GLoc.instances] == ["1"]
    assert [item.kwargs["iteration"] for item in stack.GLoc.instances] == [0]
    assert len(stack.WLoc.instances) == 1
    assert [item.kwargs["iteration"] for item in stack.WLoc.instances] == [0]
    assert stack.WLoc.instances[0].saved == [("wloc", (), {})]
    assert len(stack.BWeiss.instances) == 1
    assert stack.BWeiss.instances[0].p is None
    assert stack.BWeiss.instances[0].w is stack.WLoc.instances[0]
    assert stack.Hyb.instances[0].kwargs["green"] is stack.GLoc.instances[0].f
    assert stack.W.instances[0].is_bare
    assert stack.WLoc.instances[0].kwargs["wlat"] is stack.W.instances[0].kf

    result = stack.ImpurityAction.results[0]
    assert result.sighimp.mixing_calls == []
    assert result.sigfimp.mixing_calls == []
    assert result.sigimp.mixing_calls == []
    assert result.pimp.mixing_calls == []

    assert corr.conv.start_iters == []
    assert corr.conv.self_checks == []
    assert corr.conv.self_hdf5_checks == []
    assert corr.conv.cross_checks == []
    assert corr.conv.cross_hdf5_checks == []
    assert not hasattr(corr, "gloc")
    assert not hasattr(corr, "wloc")
    assert corr.w is stack.W.instances[1]


def test_edmft_projects_gloc_in_problem_loop_and_uses_hdf5_convergence(
    monkeypatch,
    tmp_path,
):
    stack = _install_fake_edmft_stack(monkeypatch)
    conv = _FakeConvergence(converge_after=99)
    corr = _edmft_correlation_object(
        stack.cf_mod,
        tmp_path,
        nscf=2,
        conv=conv,
    )

    corr.EDMFT()

    assert [item.kwargs["key"] for item in stack.GLoc.instances] == ["1", "1"]
    assert [item.kwargs["iteration"] for item in stack.GLoc.instances] == [0, 1]
    assert stack.Hyb.instances[0].kwargs["green"] is stack.GLoc.instances[0].f
    assert stack.Hyb.instances[1].kwargs["green"] is stack.GLoc.instances[1].f

    assert [item.kwargs["iteration"] for item in stack.WLoc.instances] == [0, 1]
    assert [item.saved for item in stack.WLoc.instances] == [
        [("wloc", (), {})],
        [("wloc", (), {})],
    ]

    assert conv.start_iters == [1]
    assert conv.self_hdf5_checks == [
        (
            "GLoc",
            {
                "group": "edmft",
                "subgroup": "GLoc",
                "current": "gloc.1",
                "previous": "gloc.0",
                "keys": ["1"],
            },
        ),
        (
            "WLoc",
            {
                "group": "edmft",
                "subgroup": "WLoc",
                "current": "wloc.1",
                "previous": "wloc.0",
                "keys": ["1"],
            },
        ),
    ]
    assert conv.cross_hdf5_checks == [
        (
            "GLoc",
            {
                "name_b": "GImp",
                "group": "edmft",
                "subgroup_a": "GLoc",
                "subgroup_b": "GImp",
                "stem_a": "gloc.1",
                "stem_b": "gimp.1",
                "keys": ["1"],
            },
        ),
        (
            "WLoc",
            {
                "name_b": "WImp",
                "group": "edmft",
                "subgroup_a": "WLoc",
                "subgroup_b": "WImp",
                "stem_a": "wloc.1",
                "stem_b": "wimp.1",
                "keys": ["1"],
            },
        ),
    ]
    assert [name for name, _, _ in conv.self_checks] == ["mu"]
    assert conv.cross_checks == []
    assert conv.diagnostics == {"1": {"sign": 1.0, "nimp": 1.0}}
    assert not hasattr(corr, "gloc")
    assert not hasattr(corr, "wloc")


def test_edmft_feeds_mixed_pimp_into_next_lattice_iteration(monkeypatch, tmp_path):
    stack = _install_fake_edmft_stack(monkeypatch)
    corr = _edmft_correlation_object(
        stack.cf_mod,
        tmp_path,
        nscf=2,
        conv=_FakeConvergence(converge_after=2),
    )

    corr.EDMFT()

    assert len(stack.PLoc.instances) == 0
    assert len(stack.BWeiss.instances) == 2
    assert stack.BWeiss.instances[0].p is None
    assert stack.BWeiss.instances[1].p is stack.ImpurityAction.results[0].pimp
    assert stack.W.instances[0].is_bare
    assert not stack.W.instances[1].is_bare
    assert not stack.W.instances[2].is_bare
    assert stack.PolC.instances[0].embedded[0][0] is stack.ImpurityAction.results[0].pimp


@pytest.mark.parametrize("missing_attr", ["chi", "pimp", "wimp"])
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


def test_gwedmft_composes_gw_dc_and_impurity_without_quantity_dicts(
    monkeypatch,
    tmp_path,
):
    stack = _install_fake_edmft_stack(monkeypatch)
    corr = _edmft_correlation_object(stack.cf_mod, tmp_path, nscf=1)

    corr.GWEDMFT()

    hf_dc = stack.HFLoc.instances[0].result
    gw_dc = stack.GWLoc.instances[0].result
    assert hf_dc.sigh.mixing_calls == []
    for obj, component, value in (
        (hf_dc.sigf, "sigfdc", hf_dc.sigf.floc),
        (gw_dc.siggwc, "siggwcdc", gw_dc.siggwc.f),
        (gw_dc.pol, "pdc", gw_dc.pol.f),
    ):
        assert len(obj.mixing_calls) == 1
        call = obj.mixing_calls[0]
        assert call["iter"] == 1
        assert call["mix"] == 0.5
        assert call["component"] == component
        assert call["method"] == "pulay"
        assert call["npulay"] == 5
        np.testing.assert_allclose(call["value"], value)

    sigc = stack.SigC.instances[0]
    assert sigc.kwargs["sigh"] == pytest.approx(np.asarray([10.0]))
    assert sigc.kwargs["sigf"] == pytest.approx(np.asarray([20.0]))
    assert sigc.kwargs["siggwc"] == pytest.approx(np.asarray([30.0]))
    assert len(sigc.embedded) == 2
    assert "sighimp" not in sigc.embedded[0]
    np.testing.assert_allclose(sigc.embedded[0]["sigfimp"], -5.0)
    np.testing.assert_allclose(sigc.embedded[0]["sigimp"], -6.0)
    assert "sighimp" not in sigc.embedded[1]
    np.testing.assert_allclose(sigc.embedded[1]["sigfimp"], 5.0)
    np.testing.assert_allclose(sigc.embedded[1]["sigimp"], 6.0)

    eimp = stack.cf_mod.EImp.instances[0]
    np.testing.assert_allclose(eimp.kwargs["sigh"], 10.0)
    np.testing.assert_allclose(eimp.kwargs["sigf"], 20.0)
    np.testing.assert_allclose(eimp.kwargs["hloc"], 50.0)
    np.testing.assert_allclose(eimp.kwargs["floc"], 5.0)

    hyb = stack.Hyb.instances[0]
    np.testing.assert_allclose(hyb.kwargs["sigh"], 50.0)
    np.testing.assert_allclose(hyb.kwargs["sigf"], 5.0)
    np.testing.assert_allclose(hyb.kwargs["sigc"], 6.0)

    polc = stack.PolC.instances[0]
    assert hasattr(polc.gw, "kf")
    assert polc.dc[0] is not None
    assert polc.embedded[0][0] is polc.dc[0]
    np.testing.assert_allclose(polc.embedded[0][0].f, 0.2)

    np.testing.assert_allclose(
        stack.GLoc.instances[1].kwargs["green"], stack.G.instances[1].kf
    )
    np.testing.assert_allclose(
        stack.WLoc.instances[1].kwargs["wlat"], stack.W.instances[1].kf
    )

    assert not any(name.endswith("_by_key") for name in vars(corr))
    assert not hasattr(corr, "impurity_state")
    assert corr.green is stack.G.instances[-1]
    assert corr.w is stack.W.instances[-1]
    assert corr.pol is polc
    assert corr.sigc is sigc


def test_gwedmft_uses_mixed_non_hartree_dc_in_lattice_and_bath(
    monkeypatch, tmp_path
):
    stack = _install_fake_edmft_stack(monkeypatch)
    original_hf_call = stack.HFLoc.__call__
    original_gw_call = stack.GWLoc.__call__

    def mixed_hf_call(self):
        result = original_hf_call(self)
        result.sigh.mixing_result = np.asarray([51.0])
        result.sigf.mixing_result = np.asarray([7.0])
        return result

    def mixed_gw_call(self):
        result = original_gw_call(self)
        result.siggwc.mixing_result = np.asarray([8.0])
        result.pol.mixing_result = np.full_like(result.pol.f, 0.25)
        return result

    monkeypatch.setattr(stack.HFLoc, "__call__", mixed_hf_call)
    monkeypatch.setattr(stack.GWLoc, "__call__", mixed_gw_call)

    corr = _edmft_correlation_object(stack.cf_mod, tmp_path, nscf=1)
    corr.GWEDMFT()

    sigc = stack.SigC.instances[0]
    assert stack.HFLoc.instances[0].result.sigh.mixing_calls == []
    assert "sighimp" not in sigc.embedded[0]
    np.testing.assert_allclose(sigc.embedded[0]["sigfimp"], -7.0)
    np.testing.assert_allclose(sigc.embedded[0]["sigimp"], -8.0)
    assert "sighimp" not in sigc.embedded[1]
    np.testing.assert_allclose(sigc.embedded[1]["sigfimp"], 7.0)
    np.testing.assert_allclose(sigc.embedded[1]["sigimp"], 8.0)

    eimp = stack.cf_mod.EImp.instances[0]
    np.testing.assert_allclose(eimp.kwargs["hloc"], 50.0)
    np.testing.assert_allclose(eimp.kwargs["floc"], 7.0)
    hyb = stack.Hyb.instances[0]
    np.testing.assert_allclose(hyb.kwargs["sigh"], 50.0)
    np.testing.assert_allclose(hyb.kwargs["sigf"], 7.0)
    np.testing.assert_allclose(hyb.kwargs["sigc"], 8.0)
    np.testing.assert_allclose(stack.PolC.instances[0].dc[0].f, 0.25)


def test_gwedmft_builds_bosonic_weiss_from_previous_impurity_polarization(
    monkeypatch,
    tmp_path,
):
    stack = _install_fake_edmft_stack(monkeypatch, hdf5_offset=100.0)
    conv = _FakeConvergence(converge_after=2)
    corr = _edmft_correlation_object(
        stack.cf_mod,
        tmp_path,
        nscf=2,
        conv=conv,
    )

    corr.GWEDMFT()

    np.testing.assert_allclose(stack.BWeiss.instances[0].p.f, 0.2)
    np.testing.assert_allclose(stack.BWeiss.instances[1].p, 100.4)
    assert stack.GWLoc.instances[0].result.pol.projection_calls == []
    assert stack.GWLoc.instances[1].result.pol.projection_calls == []
    np.testing.assert_allclose(stack.PolC.instances[1].dc[0].f, 0.2)
    first_hyb, second_hyb = stack.Hyb.instances
    np.testing.assert_allclose(first_hyb.kwargs["sigh"], 50.0)
    np.testing.assert_allclose(first_hyb.kwargs["sigf"], 5.0)
    np.testing.assert_allclose(first_hyb.kwargs["sigc"], 6.0)
    np.testing.assert_allclose(second_hyb.kwargs["sigh"], 102.0)
    np.testing.assert_allclose(second_hyb.kwargs["sigf"], 103.0)
    np.testing.assert_allclose(second_hyb.kwargs["sigc"], 104.0)
    assert stack.PolC.instances[0].embedded[0][0] is stack.PolC.instances[0].dc[0]
    np.testing.assert_allclose(stack.PolC.instances[1].embedded[0][0], 100.4)
    assert conv.start_iters == [1, 2]
    assert conv.self_hdf5_checks[0][1]["current"] == "gloc.1"
    assert conv.self_hdf5_checks[0][1]["previous"] == "gloc.0"
    assert conv.self_hdf5_checks[2][1]["current"] == "gloc.2"
    assert conv.self_hdf5_checks[2][1]["previous"] == "gloc.1"
    assert conv.cross_hdf5_checks[0][1]["stem_a"] == "gloc.1"
    assert conv.cross_hdf5_checks[0][1]["stem_b"] == "gimp.1"


def test_gwedmft_runs_each_problem_without_impurity_quantity_dicts(
    monkeypatch,
    tmp_path,
):
    stack = _install_fake_edmft_stack(monkeypatch)
    corr = _edmft_correlation_object(
        stack.cf_mod,
        tmp_path,
        nscf=2,
        conv=_FakeConvergence(converge_after=2),
    )
    corr.control["impurity"] = {
        "impdict": {"1": [[(0, 0)]], "2": [[(0, 0)]]},
        "equiv": {
            "1": np.eye(1, dtype=int),
            "2": np.eye(1, dtype=int),
        },
    }

    corr.GWEDMFT()

    assert [item.kwargs["key"] for item in stack.GLoc.instances] == [
        "1", "2", "1", "2", "1", "2"
    ]
    assert [item.kwargs["key"] for item in stack.HFLoc.instances] == [
        "1", "2", "1", "2"
    ]
    assert [item.kwargs["key"] for item in stack.GWLoc.instances] == [
        "1", "2", "1", "2"
    ]
    assert [item.kwargs["key"] for item in stack.ImpurityAction.instances] == [
        "1", "2", "1", "2"
    ]
    np.testing.assert_allclose(stack.BWeiss.instances[0].p.f, 0.2)
    np.testing.assert_allclose(stack.BWeiss.instances[1].p.f, 0.2)
    np.testing.assert_allclose(stack.BWeiss.instances[2].p, 0.4)
    np.testing.assert_allclose(stack.BWeiss.instances[3].p, 0.5)
    assert stack.GWLoc.instances[2].result.pol.projection_calls == []
    assert stack.GWLoc.instances[3].result.pol.projection_calls == []
    for hyb in stack.Hyb.instances[:2]:
        np.testing.assert_allclose(hyb.kwargs["sigh"], 50.0)
        np.testing.assert_allclose(hyb.kwargs["sigf"], 5.0)
        np.testing.assert_allclose(hyb.kwargs["sigc"], 6.0)
    np.testing.assert_allclose(stack.Hyb.instances[2].kwargs["sigh"], 2.0)
    np.testing.assert_allclose(stack.Hyb.instances[2].kwargs["sigf"], 3.0)
    np.testing.assert_allclose(stack.Hyb.instances[2].kwargs["sigc"], 4.0)
    np.testing.assert_allclose(stack.Hyb.instances[3].kwargs["sigh"], 12.0)
    np.testing.assert_allclose(stack.Hyb.instances[3].kwargs["sigf"], 13.0)
    np.testing.assert_allclose(stack.Hyb.instances[3].kwargs["sigc"], 14.0)
    np.testing.assert_allclose(
        np.asarray(
            [item.kwargs["hloc"] for item in stack.cf_mod.EImp.instances]
        ).reshape(-1),
        [50.0, 50.0, 2.0, 12.0],
    )
    assert len(stack.SigC.instances[0].embedded) == 4
    assert all(
        "sighimp" not in entry for entry in stack.SigC.instances[0].embedded
    )
    assert len(stack.PolC.instances[0].embedded) == 2
    assert not any(name.endswith("_by_key") for name in vars(corr))


@pytest.mark.parametrize("missing_attr", ["pimp", "wimp", "sigimp"])
def test_gwedmft_requires_complete_impurity_pipeline(
    monkeypatch,
    tmp_path,
    missing_attr,
):
    stack = _install_fake_edmft_stack(monkeypatch, missing=(missing_attr,))
    corr = _edmft_correlation_object(stack.cf_mod, tmp_path)

    with pytest.raises(RuntimeError, match=f"no {missing_attr}"):
        corr.GWEDMFT()


def test_run_dispatch_calls_gwedmft(monkeypatch):
    run_mod = importlib.import_module("QAssemble.Run")
    called = []

    class FakeCorrelationFunction:
        def __init__(self, control):
            self.control = control

        def GWEDMFT(self):
            called.append(self.control)

    monkeypatch.setattr(run_mod, "CorrelationFunction", FakeCorrelationFunction)
    runner = object.__new__(run_mod.Run)
    runner.control = {"run": {"method": "gw+edmft"}}

    runner.RunDiagE()

    assert called == [runner.control]
