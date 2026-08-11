from types import SimpleNamespace

import numpy as np
import pytest

from QAssemble.BLocDyn import Chi, PImp


class _FakeDLR:
    def __init__(self, nfreq=2, nuniform=3):
        self.beta = 1.0
        self.nu = np.arange(nfreq, dtype=float)
        self.tauB = np.arange(nfreq, dtype=float)
        self.nu_uniform = np.arange(nuniform, dtype=float)
        self.uniform_to_dlr_calls = 0
        self.dlr_to_uniform_calls = []

    def BatchBF2T(self, bf_2d):
        return np.asarray(bf_2d, dtype=np.complex128)

    def MatsubaraBosonUniform(self):
        return self.nu_uniform

    def MatsubaraUniform2DLR(self, value, omega=None, sign=1):
        self.uniform_to_dlr_calls += 1
        raise AssertionError("unexpected direct uniform->DLR conversion in test")

    def MatsubaraDLR2UniformGrid(self, value, sign=1):
        self.dlr_to_uniform_calls.append(sign)
        return np.asarray(value, dtype=np.complex128) + 1000.0


class _FakeProjector:
    def __init__(self):
        self.bprojector = {"1": np.ones((1, 1, 1), dtype=float)}
        self.equiv = {"1": np.eye(1, dtype=int)}
        self.blocal2pair = {"1": [{0: (0, 0)}]}

    def ProbBorb2FPair(self, key, iorbc, ispace=0):
        assert key == "1"
        assert int(iorbc) == 0
        assert ispace == 0
        return (0, 0)


def _chi_reader():
    obj = Chi.__new__(Chi)
    obj.crystal = SimpleNamespace(ns=1)
    obj.dlr = _FakeDLR(nfreq=2)
    obj.projector = _FakeProjector()
    obj.key = "1"
    return obj


def test_chi_quad_susceptibility_converts_ctqmc_7d_to_bosonic_5d():
    reader = _chi_reader()
    raw = np.zeros((1, 1, 1, 1, 2, 2, 2), dtype=np.complex128, order="F")
    raw[0, 0, 0, 0, 0, 0, :] = [1.0, 2.0]
    raw[0, 0, 0, 0, 0, 1, :] = [3.0, 4.0]
    raw[0, 0, 0, 0, 1, 0, :] = [5.0, 6.0]
    raw[0, 0, 0, 0, 1, 1, :] = [7.0, 8.0]

    out = reader.QuadSusceptibility2Boson(raw)

    expected = 0.5 * np.array([16.0, 20.0], dtype=np.complex128)
    assert out.shape == (1, 1, 1, 1, 2)
    np.testing.assert_allclose(out[0, 0, 0, 0, :], expected)


def test_chi_quad_susceptibility_rejects_non_ctqmc_shape():
    reader = _chi_reader()
    chi = np.zeros((1, 1, 1, 1, 2), dtype=np.complex128)

    with pytest.raises(ValueError, match="chi must be 7D"):
        reader.QuadSusceptibility2Boson(chi)


def test_chi_cal_projects_ctqmc_susceptibility_causally(monkeypatch):
    crystal = SimpleNamespace(ns=1)
    dlr = _FakeDLR(nfreq=2, nuniform=3)
    projector = _FakeProjector()
    partition = {
        "occupation-susceptibility-bulla": {
            "0_0": [1.0, 2.0, 3.0],
            "0_1": [10.0, 20.0, 30.0],
            "1_0": [100.0, 200.0, 300.0],
            "1_1": [1000.0, 2000.0, 3000.0],
        },
    }

    projection_calls = []

    def fake_causal_projection(self, value, *, grid="dlr", **kwargs):
        projection_calls.append((np.array(value, copy=True), grid, kwargs))
        return np.asfortranarray(value[..., : len(self.dlr.nu)] + 100.0)

    monkeypatch.setattr(Chi, "CausalProjection", fake_causal_projection)

    chi = Chi(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        partition=partition,
    )

    raw = 0.5 * np.array([1111.0, 2222.0, 3333.0], dtype=np.complex128)
    value, grid, kwargs = projection_calls[0]
    assert grid == "uniform"
    assert kwargs["coefficient_sign"] == 1
    assert kwargs["oddzero"] is True
    assert kwargs["highzero"] is True
    assert kwargs["constraint_tol"] == "auto"
    # No hdf5 cache target: the previous-iteration fallback is absent.
    assert kwargs["fallback_matrix"] is None
    np.testing.assert_allclose(value[0, 0, 0, 0, :], raw)
    # Raw solver data stays available for diagnostics.
    np.testing.assert_allclose(
        chi.occupation_susceptibility_bulla[0, 0, 0, 0, :], raw
    )
    # f holds the projected data on the DLR grid.
    projected = raw[:2] + 100.0
    assert chi.f.shape == (1, 1, 1, 1, 2)
    np.testing.assert_allclose(chi.f[0, 0, 0, 0, :], projected)
    # f_uniform is re-derived from the projected f (fake adds 1000).
    assert dlr.dlr_to_uniform_calls == [1]
    np.testing.assert_allclose(chi.f_uniform[0, 0, 0, 0, :], projected + 1000.0)
    # t follows the projected f (fake BF2T is the identity).
    np.testing.assert_allclose(chi.t, chi.f)
    np.testing.assert_allclose(
        chi.occupation_susceptibility_xij_uniform[:, 0, 1],
        [10.0, 20.0, 30.0],
    )


def test_pimp_uses_uniform_chi_and_utilde_before_causal_projection(monkeypatch):
    crystal = SimpleNamespace(ns=1)
    dlr = _FakeDLR(nfreq=2, nuniform=3)
    projector = _FakeProjector()
    chi_uniform = np.zeros((1, 1, 1, 1, 3), dtype=np.complex128, order="F")
    utilde_uniform = np.zeros_like(chi_uniform)
    chi_uniform[0, 0, 0, 0, :] = [0.1, 0.2, 0.3]
    utilde_uniform[0, 0, 0, 0, :] = [1.0, 2.0, 3.0]
    projected_inputs = []

    def fake_causal_projection(self, value, *, grid="dlr", **kwargs):
        assert grid == "uniform"
        # Cal uses the zero-static pimpbrd policy (Pimp decays to zero).
        assert kwargs["coefficient_sign"] == -1
        assert kwargs["oddzero"] is True
        assert kwargs["highzero"] is True
        assert kwargs["fallback_matrix"] is None
        projected_inputs.append(np.array(value, copy=True))
        return np.asfortranarray(value[..., : len(self.dlr.nu)] + 100.0)

    monkeypatch.setattr(PImp, "CausalProjection", fake_causal_projection)

    pimp = PImp(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        chi=SimpleNamespace(f_uniform=chi_uniform),
        utilde=utilde_uniform,
    )

    # P = -chi (1 - U chi)^-1 (negative for a positive susceptibility).  The
    # last frequency point is deliberately near-resonant (U*chi = 0.9), so the
    # expected value there is amplified to -3.0; that is the formula, not a bug.
    expected_uniform = -chi_uniform / (1.0 - utilde_uniform * chi_uniform)
    np.testing.assert_allclose(pimp.chi_boson_uniform, chi_uniform)
    np.testing.assert_allclose(pimp.chi_boson, chi_uniform)
    np.testing.assert_allclose(pimp.utilde_uniform, utilde_uniform)
    np.testing.assert_allclose(pimp.f_uniform, expected_uniform)
    np.testing.assert_allclose(projected_inputs[0], expected_uniform)
    np.testing.assert_allclose(pimp.f, expected_uniform[..., :2] + 100.0)
    np.testing.assert_allclose(pimp.t, pimp.f)


def test_chi_cal_seeds_and_reuses_brd_prev_cache(monkeypatch, tmp_path):
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()
    partition = {
        "occupation-susceptibility-bulla": {
            "0_0": [1.0, 2.0, 3.0],
            "0_1": [1.0, 2.0, 3.0],
            "1_0": [1.0, 2.0, 3.0],
            "1_1": [1.0, 2.0, 3.0],
        },
    }
    path = str(tmp_path / "glob.h5")

    projection_calls = []

    def fake_causal_projection(self, value, *, grid="dlr", **kwargs):
        projection_calls.append(kwargs)
        return np.asfortranarray(value[..., : len(self.dlr.nu)] + 100.0)

    monkeypatch.setattr(Chi, "CausalProjection", fake_causal_projection)

    first = Chi(
        crystal=crystal, dlr=_FakeDLR(nfreq=2, nuniform=3),
        projector=projector, key="1", partition=partition,
        hdf5file=path, group="calc",
    )
    assert projection_calls[0]["fallback_matrix"] is None

    second = Chi(
        crystal=crystal, dlr=_FakeDLR(nfreq=2, nuniform=3),
        projector=projector, key="1", partition=partition,
        hdf5file=path, group="calc",
    )
    # The second iteration receives the first one's projected f as fallback.
    np.testing.assert_allclose(projection_calls[1]["fallback_matrix"], first.f)
    assert second.f.shape == first.f.shape


class _FakeProjector2Orb:
    """Two real fermion orbitals (nspin inferred as 1 from the pair map)."""

    def __init__(self):
        self.bprojector = {"1": np.ones((1, 2, 2), dtype=float)}
        self.equiv = {"1": np.eye(2, dtype=int)}
        self.blocal2pair = {"1": [{0: (0, 0), 1: (1, 1)}]}

    def ProbBorb2FPair(self, key, iorbc, ispace=0):
        return (int(iorbc), int(iorbc))


def test_chi_cal_symmetrizes_boson_susceptibility(monkeypatch):
    # "0_1" and "1_0" are independently measured by EVALSIM and differ by
    # CTQMC noise; Cal must hand the causal projection the exact-symmetry
    # average chi[ib, jb, s1, s2] = chi[jb, ib, s2, s1] while keeping the raw
    # solver data available for diagnostics.
    crystal = SimpleNamespace(ns=1)
    dlr = _FakeDLR(nfreq=2, nuniform=3)
    projector = _FakeProjector2Orb()
    x01 = [10.0, 20.0, 30.0]
    x10 = [40.0, 60.0, 80.0]
    partition = {
        "occupation-susceptibility-bulla": {
            "0_0": [1.0, 2.0, 3.0],
            "0_1": x01,
            "1_0": x10,
            "1_1": [4.0, 5.0, 6.0],
        },
    }

    projection_calls = []

    def fake_causal_projection(self, value, *, grid="dlr", **kwargs):
        projection_calls.append(np.array(value, copy=True))
        return np.asfortranarray(value[..., : len(self.dlr.nu)])

    monkeypatch.setattr(Chi, "CausalProjection", fake_causal_projection)

    chi = Chi(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        partition=partition,
    )

    value = projection_calls[0]
    mean = 0.5 * (np.array(x01) + np.array(x10))
    np.testing.assert_allclose(value[0, 1, 0, 0, :], mean)
    np.testing.assert_allclose(value[1, 0, 0, 0, :], mean)
    # Diagonal channels are untouched by the symmetrization.
    np.testing.assert_allclose(value[0, 0, 0, 0, :], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(value[1, 1, 0, 0, :], [4.0, 5.0, 6.0])
    # The enforced symmetry transposes orbital and spin axes together.
    np.testing.assert_allclose(value, value.transpose(1, 0, 3, 2, 4))
    # Raw (asymmetric) solver data stays available for diagnostics.
    np.testing.assert_allclose(
        chi.occupation_susceptibility_xij_uniform[:, 0, 1], x01
    )
    np.testing.assert_allclose(
        chi.occupation_susceptibility_xij_uniform[:, 1, 0], x10
    )
