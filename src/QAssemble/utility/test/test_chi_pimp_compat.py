from types import SimpleNamespace

import h5py
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
    def __init__(self, classes=(1,), pairs=None):
        if pairs is None:
            pairs = [(i, i) for i in range(len(classes))]
        self.bprojector = {"1": np.ones((1, len(pairs), 1), dtype=float)}
        self.equiv = {"1": np.diag(classes)}
        self.blocal2pair = {"1": [dict(enumerate(pairs))]}

    def ProbBorb2FPair(self, key, iorbc, ispace=0):
        assert key == "1"
        assert ispace == 0
        return self.blocal2pair[key][ispace][int(iorbc)]


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

    expected = np.array([16.0, 20.0], dtype=np.complex128)
    assert out.shape == (1, 1, 1, 1, 2)
    np.testing.assert_allclose(out[0, 0, 0, 0, :], expected)


def test_chi_quad_susceptibility_rejects_non_ctqmc_shape():
    reader = _chi_reader()
    chi = np.zeros((1, 1, 1, 1, 2), dtype=np.complex128)

    with pytest.raises(ValueError, match="chi must be 7D"):
        reader.QuadSusceptibility2Boson(chi)


@pytest.mark.parametrize("classes,groups", [
    ((1, 1, 1), [
        [(1, 1), (2, 2), (4, 4)],
        [(1, 2), (1, 4), (2, 1), (2, 4), (4, 1), (4, 2)],
    ]),
    ((1, 2, 3), []),
    ((1, 1, 2), [
        [(2, 2), (4, 4)],
        [(2, 4), (4, 2)],
        [(1, 2), (1, 4), (2, 1), (4, 1)],
    ]),
], ids=["equivalent", "distinct", "mixed"])
def test_chi_density_equiv_average_preserves_charge_and_other_pairs(classes, groups):
    reader = _chi_reader()
    # Deliberately shuffled boson indices, including non-density pairs.
    reader.projector = _FakeProjector(
        classes, pairs=[(0, 1), (2, 2), (0, 0), (1, 0), (1, 1)]
    )
    values = np.arange(5 * 5 * 2 * 2 * 3).reshape(5, 5, 2, 2, 3) ** 2
    raw = np.asfortranarray(6.0 * values + 3.0j * values[::-1])
    original = raw.copy()
    expected = raw.copy()
    for group in groups:
        mean = sum(raw[bi, bj] for bi, bj in group) / len(group)
        for bi, bj in group:
            expected[bi, bj] = mean

    out = reader._average_density_by_equiv(raw)

    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(raw, original)
    density = np.ix_([1, 2, 4], [1, 2, 4])
    np.testing.assert_array_equal(out[density].sum(axis=(0, 1)),
                                  raw[density].sum(axis=(0, 1)))
    np.testing.assert_array_equal(out[[0, 3]], raw[[0, 3]])
    np.testing.assert_array_equal(out[:, [0, 3]], raw[:, [0, 3]])
    if not groups:
        assert out.tobytes() == raw.tobytes()


@pytest.mark.parametrize("enabled", [None, False, True], ids=["default", "off", "on"])
def test_chi_cal_equiv_average_and_raw_save(monkeypatch, tmp_path, enabled):
    crystal = SimpleNamespace(ns=2)
    projector = _FakeProjector(classes=(1, 1, 1))
    values = np.arange(6 * 6 * 3).reshape(6, 6, 3) ** 2
    solver = 6.0 * values + 3.0j * values[::-1]
    partition = {"occupation-susceptibility-bulla": {
        f"{i}_{j}": {"real": solver[i, j].real.tolist(),
                     "imag": solver[i, j].imag.tolist()}
        for i in range(6) for j in range(6)
    }}
    raw = np.empty((3, 3, 2, 2, 3), dtype=complex, order="F")
    for js in range(2):
        for ks in range(2):
            raw[:, :, js, ks] = solver[3 * js:3 * js + 3, 3 * ks:3 * ks + 3]
    expected = raw.copy()
    if enabled:
        diagonal = np.eye(3, dtype=bool)
        expected[diagonal] = raw[diagonal].mean(axis=0)
        expected[~diagonal] = raw[~diagonal].mean(axis=0)
    projection_calls = []

    def fake_causal_projection(self, value, **kwargs):
        projection_calls.append(value.copy())
        return np.asfortranarray(value + 100.0)

    monkeypatch.setattr(Chi, "CausalProjection", fake_causal_projection)
    monkeypatch.setattr(PImp, "CausalProjection", fake_causal_projection)
    options = {} if enabled is None else {"equiv_average": enabled}
    path = str(tmp_path / "chi.h5")
    chi = Chi(crystal, _FakeDLR(nfreq=3), projector, "1", partition=partition,
              hdf5file=path, group="calc", iteration=2, **options)
    assert chi.equiv_average is bool(enabled)
    np.testing.assert_array_equal(projection_calls[0], expected)
    np.testing.assert_array_equal(chi.occupation_susceptibility_bulla_raw, raw)
    np.testing.assert_array_equal(chi.occupation_susceptibility_bulla, expected)
    np.testing.assert_array_equal(chi.f, expected + 100.0)
    np.testing.assert_array_equal(chi.f_uniform, expected + 1100.0)
    np.testing.assert_array_equal(chi.t, chi.f)
    if not enabled:
        assert chi.f.tobytes() == (raw + 100.0).tobytes()
        assert chi.f_uniform.tobytes() == (raw + 1100.0).tobytes()
    chi.Save("chi")
    with h5py.File(path, "r") as handle:
        saved = handle["calc/Chi"]
        names = {"chi.2.1", "chi.2.1_uniform", "chi_brd_prev.1"}
        if enabled:
            names.add("chi.2.1_raw_uniform")
            np.testing.assert_array_equal(saved["chi.2.1_raw_uniform"][()], raw)
        assert set(saved) == names
        np.testing.assert_array_equal(saved["chi.2.1_uniform"][()], chi.f_uniform)

    pimp = PImp(crystal, chi.dlr, projector, "1", chi=chi,
                utilde=np.zeros_like(chi.f_uniform))
    np.testing.assert_array_equal(pimp.chi_boson_uniform, chi.f_uniform)
    np.testing.assert_array_equal(projection_calls[1], -chi.f_uniform)


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

    raw = np.array([1111.0, 2222.0, 3333.0], dtype=np.complex128)
    value, grid, kwargs = projection_calls[0]
    assert grid == "uniform"
    assert kwargs["coefficient_sign"] == 1
    assert kwargs["oddzero"] is True
    assert kwargs["highzero"] is True
    assert kwargs["tail_points"] == 5
    assert kwargs["tail_log_spaced"] is False
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
        assert kwargs["tail_points"] == 5
        assert kwargs["tail_log_spaced"] is False
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


def test_pimp_cal_seeds_and_reuses_brd_prev_cache(monkeypatch, tmp_path):
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()
    chi_uniform = np.full((1, 1, 1, 1, 3), 0.1, dtype=np.complex128)
    utilde_uniform = np.ones_like(chi_uniform)
    path = str(tmp_path / "glob.h5")

    projection_calls = []

    def fake_causal_projection(self, value, *, grid="dlr", **kwargs):
        projection_calls.append(kwargs)
        return np.asfortranarray(value[..., : len(self.dlr.nu)] + 100.0)

    monkeypatch.setattr(PImp, "CausalProjection", fake_causal_projection)

    first = PImp(
        crystal=crystal, dlr=_FakeDLR(nfreq=2, nuniform=3),
        projector=projector, key="1",
        chi=SimpleNamespace(f_uniform=chi_uniform), utilde=utilde_uniform,
        hdf5file=path, group="calc",
    )
    assert projection_calls[0]["fallback_matrix"] is None
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/PImp/pimp_brd_prev.1"][()], first.f)

    second = PImp(
        crystal=crystal, dlr=_FakeDLR(nfreq=2, nuniform=3),
        projector=projector, key="1",
        chi=SimpleNamespace(f_uniform=chi_uniform), utilde=utilde_uniform,
        hdf5file=path, group="calc",
    )
    assert len(projection_calls) == 2
    np.testing.assert_allclose(projection_calls[1]["fallback_matrix"], first.f)
    assert second.f.shape == first.f.shape
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["calc/PImp/pimp_brd_prev.1"][()], second.f)
