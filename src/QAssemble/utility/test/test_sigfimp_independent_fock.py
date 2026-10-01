"""Bare-V impurity Fock, distinct from the solver's measured HF moment."""
from types import SimpleNamespace
import importlib
import json

import h5py
import numpy as np
import pytest

from QAssemble.FLocStc import SigFImp, SigFLoc
from QAssemble.utility.HDF5 import IO
from QAssemble.BLocStc import VLoc


class _Projector:
    def __init__(self, norb=1, ns=1):
        self.fprojector = {"1": np.eye(norb)[:, :, None].repeat(ns, axis=2)}
        self.equiv = {"1": np.eye(norb, dtype=int)}
        self.pairs = [(i, j) for i in range(norb) for j in range(norb)]
        self.bprojector = {"1": np.eye(norb ** 2)[:, :, None, None]}

    def ProbBorb2FPair(self, key, iorb):
        return self.pairs[iorb]


def _imp(occ, v, **kwargs):
    return SigFImp(
        SimpleNamespace(), _Projector(occ.shape[0], occ.shape[2]), "1",
        np.full(occ.shape, 99.0), sigh=np.full(occ.shape, 7.0),
        occ=occ, vloc=v, **kwargs,
    )


@pytest.mark.parametrize("ns", [1, 2])
def test_single_orbital_same_spin_fock_has_no_hartree_multiplicity(ns):
    occ = np.linspace(0.2, 0.6, ns).reshape(1, 1, ns)
    v = np.full((1, 1, ns, ns), 100.0)
    for s in range(ns):
        v[0, 0, s, s] = 3.0
    imp = _imp(occ, v)
    np.testing.assert_allclose(imp.s, -3.0 * occ)
    np.testing.assert_allclose(imp.hf, 99.0)
    assert not np.allclose(imp.s, imp.hf - imp.sigh)


def test_offdiagonal_density_matches_matrix_exchange_and_local_fock():
    m = np.array([[2.0, 0.4], [0.4, 1.5]])
    rho = np.array([[0.7, 0.2], [0.2, 0.3]])
    occ = rho[:, :, None]
    v = np.einsum("ab,cd->abcd", m, m).reshape(4, 4, 1, 1)
    imp = _imp(occ, v)
    loc = SigFLoc(SimpleNamespace(), imp.projector, "1", occ=occ, vloc=v)
    np.testing.assert_allclose(imp.s[:, :, 0], -m @ rho @ m)
    np.testing.assert_allclose(imp.s, loc.floc)
    np.testing.assert_allclose(imp.s, imp.s.conj().transpose(1, 0, 2))
    assert imp.s.flags.f_contiguous


def test_diagonal_density_matches_fullgwedmft_four_index_contraction():
    rng = np.random.default_rng(17)
    tensor = rng.normal(size=(2, 2, 2, 2))
    rho = np.diag([0.25, 0.6])
    # QA's pair-flat V[(a,b),(c,d)] maps to UT[i,j,k,l]=V[(i,l),(j,k)].
    full_tensor = tensor.transpose(0, 2, 3, 1)
    expected = np.zeros((2, 2))
    # FullGWEDMFT dc_f0.F: V(i,l,j,k) * G(beta)(k,l), G(beta)=-rho.
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    expected[i, j] -= full_tensor[i, l, j, k] * rho[k, l]
    imp = _imp(rho[:, :, None], tensor.reshape(4, 4, 1, 1))
    np.testing.assert_allclose(imp.s[:, :, 0], expected, atol=1e-10)


def test_independent_fock_mixing_rewinds_and_resumes_its_own_history(tmp_path):
    path = str(tmp_path / "mix.h5")
    with h5py.File(path, "w"):
        pass
    control = {"mix": 0.25, "mixing_method": "linear", "npulay": 3}
    v = np.full((1, 1, 1, 1), 2.0)

    def run(iteration, density):
        obj = _imp(np.full((1, 1, 1), density), v, control=control,
                   hdf5file=path, group="gwedmft", iteration=iteration)
        obj.Mixing()
        return obj.s.copy()

    np.testing.assert_allclose(run(1, 0.2), -0.4)
    second = run(2, 0.6)
    np.testing.assert_allclose(second, -0.6)
    uninterrupted = run(3, 0.8)
    actions = IO.AlignMixingHistory(path, "gwedmft", ["1"], 2)
    assert actions["1/sigfimp"] == "rewound"
    np.testing.assert_allclose(run(3, 0.8), uninterrupted)
    with h5py.File(path) as handle:
        np.testing.assert_allclose(handle["gwedmft/Mixing/1/sigfimp/last"], uninterrupted)


def test_independent_fock_requires_bare_interaction():
    with pytest.raises(ValueError, match="bare vloc"):
        SigFImp(SimpleNamespace(), _Projector(), "1", np.zeros((1, 1, 1)),
                occ=np.zeros((1, 1, 1)))


def test_solver_two_body_keeps_real_bare_tensor_and_existing_cutoff():
    projector = _Projector()
    projector.fimpdict = {"1": [[0]]}
    vloc = VLoc.__new__(VLoc)
    vloc.projector = projector
    vloc.crystal = SimpleNamespace(
        find=[0], ns=1, soc=False,
        Double2Quad=lambda v: v.reshape(1, 1, 1, 1),
    )
    vloc.vloc = np.full((1, 1, 1, 1), 2.0 + 0.3j)
    solver = vloc.GetUijklComCTQMC("1").reshape(2, 2, 2, 2)
    for si in range(2):
        for sj in range(2):
            assert solver[si, sj, sj, si] == 2.0
    vloc.vloc[:] = 0.0005
    assert not np.any(vloc.GetUijklComCTQMC("1"))


@pytest.mark.parametrize("method", ["gw+edmft", "edmft"])
def test_postprocessing_routes_independent_fock_only_for_gwedmft(monkeypatch, tmp_path, method):
    mod = importlib.import_module("QAssemble.CTQMC")
    obj = mod.CTQMC.__new__(mod.CTQMC)
    obj.key = "1"
    obj.projector = _Projector()
    obj.crystal = SimpleNamespace(ns=1, soc=False)
    obj.dlr = SimpleNamespace(MatsubaraFermionUniform=lambda: np.array([1., 3., 5.]))
    obj.control = {"method": method, "sigimp_guard": False}
    obj.hdf5file = None
    obj.group = "gwedmft"
    obj.ctqmc_dir = obj.work_dir = str(tmp_path)
    work = tmp_path / "impurity_1_1"
    work.mkdir()
    partition = {
        "expansion histogram": [1, 2, 1], "scalar": {"N": 0.3}, "sign": 1.,
        "green": {}, "self-energy": {"1": {"function": {
            "real": [1., 1., 1.], "imag": [-0.1, -0.1, -0.1]}, "moments": [1.]}},
    }
    (work / "params.obs.json").write_text(json.dumps({"partition": partition}))
    (work / "params.json").write_text(json.dumps({"partition": {"green matsubara cutoff": 3}}))
    rho = np.full((1, 1, 1), 0.3)
    bare = np.full((1, 1, 1, 1), 2.0)
    obj.bweiss = SimpleNamespace(cf=None, f=None, vloc=SimpleNamespace(vproj={"1": bare}))
    calls = []

    def output(**kwargs):
        return SimpleNamespace(occ=rho, Mixing=lambda: None, Save=lambda *args: None)

    def fock(**kwargs):
        calls.append(kwargs)
        return output(**kwargs)

    for name in ("GImp", "SigHImp", "SigCImp"):
        monkeypatch.setattr(mod, name, output)
    monkeypatch.setattr(mod, "SigFImp", fock)
    monkeypatch.setattr(obj, "_read_sigma_error_grid", lambda equiv: None)
    monkeypatch.chdir(tmp_path)
    obj.PostProcessing(1)
    assert len(calls) == 1
    if method == "gw+edmft":
        assert calls[0]["occ"] is rho
        assert calls[0]["vloc"] is bare
    else:
        assert "occ" not in calls[0] and "vloc" not in calls[0]
