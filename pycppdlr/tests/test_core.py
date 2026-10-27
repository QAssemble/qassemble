import gc
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from pycppdlr import Basis, ImFreq, ImTime, PairedBoson, build_dlr_rf
from pycppdlr._core import RealLU, paired_boson_k_freq, paired_boson_k_tau


def exact_tau(u, poles, weights):
    # Evaluate each pole independently, with stable expressions on both sides.
    result = np.zeros(len(u), dtype=np.result_type(np.asarray(weights).dtype, float))
    for w, weight in zip(poles, weights):
        if w >= 0:
            result -= weight * np.exp(-u * w) / (1 + np.exp(-w))
        else:
            result -= weight * np.exp((1 - u) * w) / (1 + np.exp(w))
    return result


def exact_tau_rel(t, poles, weights):
    t = np.asarray(t)
    result = np.empty(len(t), dtype=np.result_type(np.asarray(weights).dtype, float))
    positive = t >= 0
    result[positive] = exact_tau(t[positive], poles, weights)
    # Reflect both variables instead of forming 1+t near the right endpoint.
    result[~positive] = exact_tau(-t[~positive], -np.asarray(poles), weights)
    return result


def exact_freq(n, poles, weights, statistic, beta):
    n = np.asarray(n)
    z = 1j * np.pi * (2 * n.astype(float) + (statistic == "fermion"))
    result = np.zeros(len(n), dtype=complex)
    for w, weight in zip(poles, weights):
        if statistic == "boson" and w == 0:
            result[n == 0] -= beta * weight / 2
        else:
            correction = np.tanh(w / 2) if statistic == "boson" else 1
            result += beta * weight * correction / (z - w)
    return result


def expand_paired(poles, weights):
    poles, weights = np.asarray(poles), np.asarray(weights)
    positive = poles > 0
    return np.r_[poles, -poles[positive]], np.r_[weights, weights[positive]]


@pytest.fixture(scope="module")
def paired():
    return PairedBoson(100, 1e-14)


def test_paired_nodes_are_mirror_complete_and_deterministic(paired):
    other = PairedBoson(100, 1e-14)
    for name in ("rf", "t_rel", "n"):
        np.testing.assert_array_equal(getattr(paired, name), getattr(other, name))
    assert np.all(paired.rf >= 0)
    assert np.all(np.diff(paired.rf) > 0)
    assert paired.t_rel.size in (2*paired.rank-1, 2*paired.rank)
    assert paired.n.size == 2*paired.rank-1
    np.testing.assert_allclose(paired.u, 1-paired.u[::-1], atol=1e-16, rtol=0)
    np.testing.assert_array_equal(paired.n, -paired.n[::-1])
    assert np.count_nonzero(paired.n == 0) == 1


def test_paired_analytic_even_function(paired):
    beta = 7.3
    poles, weights = expand_paired([.913, 37.1], [.3, .7])
    t = evaluation_times()
    n = np.arange(-157, 158)
    samples = (paired.tau_to_coef(exact_tau_rel(paired.t_rel, poles, weights)),
               paired.freq_to_coef(exact_freq(paired.n, poles, weights, "boson", beta), beta))
    for c in samples:
        np.testing.assert_allclose(paired.eval_tau(c, t), exact_tau_rel(t, poles, weights), atol=1e-12, rtol=0)
        np.testing.assert_allclose(paired.eval_freq(c, n, beta), exact_freq(n, poles, weights, "boson", beta),
                                   atol=1e-12, rtol=0)


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_paired_complex_roundtrip_and_axes(paired, shape):
    count = int(np.prod(shape)) if shape else 1
    c = np.zeros((paired.rank, *shape), dtype=complex)
    c[3] = (1 + np.arange(count).reshape(shape)) / count * (.7 + .2j)
    tau, freq = paired.coef_to_tau(c), paired.coef_to_freq(c, 2.7)
    for recovered in (paired.tau_to_coef(tau), paired.freq_to_coef(freq, 2.7)):
        assert recovered.shape == c.shape
        np.testing.assert_allclose(paired.coef_to_tau(recovered), tau, atol=1e-12, rtol=0)
        np.testing.assert_allclose(paired.coef_to_freq(recovered, 2.7), freq, atol=1e-12, rtol=0)


def test_paired_does_not_project_complex_or_mirror_samples(paired):
    c = np.zeros(paired.rank, dtype=complex)
    c[3] = .7 + .2j
    for values, rows, solve in (
        (paired.coef_to_tau(c), paired._tau_rows, paired.tau_to_coef),
        (paired.coef_to_freq(c), paired._freq_rows, paired.freq_to_coef),
    ):
        changed = values.copy()
        omitted = np.setdiff1d(np.arange(values.size), rows)
        changed[omitted] += 30 + 20j
        np.testing.assert_array_equal(solve(changed), solve(values))
        recovered = solve(changed)
        assert np.max(np.abs(recovered.imag)) > .1


def test_paired_native_kernels_and_single_zero_normalization():
    poles = np.array([0., .913, 37.1])
    times = np.ascontiguousarray(evaluation_times())
    n = np.arange(-10, 11, dtype=np.int64)
    tau = paired_boson_k_tau(times, poles)
    freq = paired_boson_k_freq(n, poles)
    for j, w in enumerate(poles):
        expanded, weights = expand_paired([w], [1.])
        np.testing.assert_allclose(tau[:, j], exact_tau_rel(times, expanded, weights), atol=1e-14, rtol=0)
        np.testing.assert_allclose(freq[:, j], exact_freq(n, expanded, weights, "boson", 1), atol=1e-14, rtol=0)
    np.testing.assert_array_equal(tau[:, 0], -.5)
    np.testing.assert_array_equal(freq[:, 0], np.where(n == 0, -.5, 0))


def test_real_lu_matches_numpy_and_owns_its_factorization():
    a = np.array([[3., 1., .2], [-.4, 2., 1.], [.7, .3, 4.]])
    b = np.arange(12, dtype=float).reshape(3, 4)
    expected = np.linalg.solve(a, b)
    lu = RealLU(a)
    a[:] = np.nan
    np.testing.assert_allclose(lu.solve(b), expected, atol=1e-14, rtol=1e-14)


def test_real_lu_invalid_inputs():
    for a in (np.empty((0, 0)), np.zeros((2, 3)), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError):
            RealLU(a)
    with pytest.raises(RuntimeError, match="getrf"):
        RealLU(np.zeros((2, 2)))
    lu = RealLU(np.eye(2))
    for b in (np.ones((3, 1)), np.full((2, 1), np.inf)):
        with pytest.raises(ValueError):
            lu.solve(b)


def test_paired_transforms_do_not_call_python_solvers(paired, monkeypatch):
    import scipy.linalg

    def forbidden(*args, **kwargs):
        raise AssertionError("paired node transforms must use native RealLU")

    c = np.zeros(paired.rank)
    c[3] = 1
    tau, freq = paired.coef_to_tau(c), paired.coef_to_freq(c, 2.7)
    for module, name in ((np.linalg, "solve"), (np.linalg, "lstsq"),
                         (scipy.linalg, "lu_factor"), (scipy.linalg, "lu_solve")):
        monkeypatch.setattr(module, name, forbidden)
    np.testing.assert_allclose(paired.coef_to_tau(paired.tau_to_coef(tau)), tau, atol=1e-12, rtol=0)
    np.testing.assert_allclose(paired.coef_to_freq(paired.freq_to_coef(freq, 2.7), 2.7), freq, atol=1e-12, rtol=0)


@pytest.mark.parametrize("lamb,eps", [(0, 1e-14), (np.nan, 1e-14), (np.inf, 1e-14),
                                       (20, 0), (20, np.nan), (20, 1)])
def test_invalid_paired_basis(lamb, eps):
    with pytest.raises(ValueError):
        PairedBoson(lamb, eps)


def test_paired_invalid_transform_inputs(paired):
    with pytest.raises(ValueError):
        paired.tau_to_coef(np.zeros(paired.t_rel.size + 1))
    with pytest.raises(ValueError):
        paired.freq_to_coef(np.full(paired.n.size, np.nan))
    for beta in (0, -1, np.nan):
        with pytest.raises(ValueError):
            paired.coef_to_freq(np.zeros(paired.rank), beta)
    with pytest.raises(ValueError):
        paired.eval_tau(np.zeros(paired.rank), [-.6])
    with pytest.raises(ValueError):
        paired.eval_freq(np.zeros(paired.rank), [.5])


def evaluation_times():
    # 200 off-grid physical times, encoded directly in relative coordinates.
    return np.r_[np.linspace(0, 0.5, 100, endpoint=False),
                 -np.linspace(0.5, 0, 99, endpoint=False), 1.]


def accuracy(basis):
    beta = 7.3
    poles = np.array([-0.371 * basis.lamb, 0.113 * basis.lamb, 0.913])
    weights = np.array([0.7 + 0.2j, -0.3 + 0.4j, 0.6 - 0.1j])
    t = evaluation_times()
    n = np.arange(-157, 158)
    errors = {}
    for label, selected in (("single", slice(0, 1)), ("multi", slice(None))):
        p, a = poles[selected], weights[selected]
        tau = exact_tau_rel(basis.t_rel, p, a)
        freq = exact_freq(basis.n, p, a, basis.statistic, beta)
        for source, c in (("tau", basis.tau_to_coef(tau)), ("freq", basis.freq_to_coef(freq, beta))):
            for target, actual, expected in (
                ("tau", basis.eval_tau(c, t), exact_tau_rel(t, p, a)),
                ("freq", basis.eval_freq(c, n, beta), exact_freq(n, p, a, basis.statistic, beta)),
            ):
                errors[f"{label}_{source}_{target}"] = float(
                    np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))
    return errors


@pytest.fixture(scope="module", params=list(itertools.product(("fermion", "boson"), (False, True))))
def basis(request):
    statistic, symmetric = request.param
    return Basis(100, 1e-14, statistic, symmetric)


def test_native_unit_coefficient_transforms_and_nodes(basis):
    eye = np.eye(basis.rank)
    expected_t = np.column_stack([exact_tau_rel(basis.t_rel, [w], [1.]) for w in basis.rf])
    expected_f = np.column_stack([
        exact_freq(basis.n, [w], [1.], basis.statistic, 1) for w in basis.rf])
    np.testing.assert_allclose(basis.coef_to_tau(eye), expected_t, rtol=0, atol=1e-14)
    np.testing.assert_allclose(basis.coef_to_freq(eye), expected_f, rtol=0, atol=1e-14)
    np.testing.assert_allclose(basis.eval_tau(eye, basis.t_rel), expected_t, rtol=0, atol=1e-14)
    np.testing.assert_allclose(basis.eval_freq(eye, basis.n), expected_f, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(basis.t_rel, basis.it.nodes)
    np.testing.assert_array_equal(basis.n, basis.ifreq.nodes)
    assert basis.it.rank == basis.ifreq.rank == basis.rank
    np.testing.assert_array_equal(basis.u, np.where(basis.t_rel < 0, 1 + basis.t_rel, basis.t_rel))
    assert np.all((basis.u >= 0) & (basis.u <= 1))


def test_off_grid_analytic_accuracy(basis):
    errors = accuracy(basis)
    assert max(errors.values()) <= 1e-12, errors


def test_selected_node_roundtrip(basis):
    c = np.zeros(basis.rank)
    c[basis.rank // 3] = 1
    tau = basis.coef_to_tau(c)
    freq = basis.coef_to_freq(c, 2.7)
    np.testing.assert_allclose(basis.coef_to_tau(basis.freq_to_coef(freq, 2.7)), tau, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(basis.coef_to_freq(basis.tau_to_coef(tau), 2.7), freq, atol=1e-12, rtol=1e-12)


def test_transforms_do_not_call_numpy_solvers(basis, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("production transforms must call cppdlr")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    c = np.zeros(basis.rank)
    c[basis.rank // 3] = 1
    tau = basis.coef_to_tau(c)
    freq = basis.coef_to_freq(c, 2.7)
    np.testing.assert_allclose(basis.coef_to_tau(basis.tau_to_coef(tau)), tau, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(basis.coef_to_freq(basis.freq_to_coef(freq, 2.7), 2.7), freq, atol=1e-12, rtol=1e-12)
    assert basis.eval_tau(c, [0, -0.1, 1]).shape == (3,)
    assert basis.eval_freq(c, [-1, 0, 1], 2.7).shape == (3,)


def test_relative_time_endpoints_and_display_only_u(basis):
    c = np.zeros(basis.rank)
    c[basis.rank // 3] = 1
    points = np.array([0., -1e-16, 1.])
    expected = exact_tau_rel(points, [basis.rf[basis.rank // 3]], [1.])
    np.testing.assert_allclose(basis.eval_tau(c, points), expected, atol=1e-14, rtol=0)
    before = basis.coef_to_tau(c)
    displayed_u = basis.u.copy()
    try:
        basis.u[:] = np.nan
        np.testing.assert_array_equal(basis.coef_to_tau(c), before)
        np.testing.assert_allclose(basis.eval_tau(c, points), expected, atol=1e-14, rtol=0)
    finally:
        basis.u[:] = displayed_u


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (2, 3, 2)])
@pytest.mark.parametrize("layout", ["slice", "fortran"])
def test_complex_axes_and_noncontiguous_data(basis, shape, layout):
    # Different complex entries on each axis expose conjugation and transposition.
    size = int(np.prod(shape)) if shape else 1
    amplitude = (np.arange(size).reshape(shape) + 1) * (0.7 + 0.3j)
    c = np.zeros((basis.rank, *shape), dtype=complex)
    c[basis.rank // 3] = amplitude
    for source, solve in ((basis.coef_to_tau(c), basis.tau_to_coef),
                          (basis.coef_to_freq(c, 3.2), lambda x: basis.freq_to_coef(x, 3.2))):
        if layout == "slice":
            padded = np.zeros((2 * len(source), *shape), dtype=complex)
            padded[::2] = source
            values = padded[::2]
        else:
            values = np.asfortranarray(source)
        actual = solve(values)
        expected = np.stack([solve(values.reshape(len(values), -1)[:, j]) for j in range(size)], axis=1)
        # Compare reconstructed values; coefficients can be ill-conditioned.
        for reconstruct in (basis.coef_to_tau, basis.coef_to_freq):
            np.testing.assert_allclose(
                reconstruct(actual).reshape(-1, size), reconstruct(expected).reshape(-1, size),
                atol=1e-12, rtol=1e-12)
        assert actual.shape == c.shape
    for evaluate, points in ((basis.eval_tau, [0.1, -0.3]), (basis.eval_freq, [-3, 0, 4])):
        actual = evaluate(c, points)
        expected = np.stack([evaluate(c.reshape(basis.rank, -1)[:, j], points) for j in range(size)], axis=1)
        np.testing.assert_allclose(actual.reshape(len(points), -1), expected, atol=1e-12, rtol=1e-12)
        assert actual.shape == (len(points), *shape)


def test_real_output_when_native_imaginary_part_is_exactly_zero(basis):
    c = np.zeros(basis.rank)
    c[3] = 1
    assert not np.iscomplexobj(basis.coef_to_tau(c))
    assert not np.iscomplexobj(basis.eval_tau(c, [0.1, -0.2]))
    assert np.iscomplexobj(basis.coef_to_tau(c.astype(complex)))


def test_symmetry(basis):
    if not basis.symmetrize:
        return
    np.testing.assert_array_equal(basis.rf, -basis.rf[::-1])
    np.testing.assert_allclose(basis.t_rel + basis.t_rel[::-1], 0, rtol=0, atol=2e-16)
    if basis.statistic == "boson":
        np.testing.assert_array_equal(basis.n, -basis.n[::-1])
        assert np.count_nonzero(basis.n == 0) == 1
        assert len(basis.n) == basis.rank + 1
    else:
        np.testing.assert_array_equal(2 * basis.n + 1, -(2 * basis.n[::-1] + 1))


def test_native_frequency_integer_limits(basis):
    c = np.zeros(basis.rank)
    c[3] = 1
    n = np.array([-1073741824, 1073741823], dtype=np.int64)
    expected = exact_freq(n, [basis.rf[3]], [1.], basis.statistic, 1)
    np.testing.assert_allclose(basis.eval_freq(c, n), expected, atol=0, rtol=1e-14)
    for invalid in ([-1073741825], [1073741824], [2**62]):
        with pytest.raises(ValueError):
            basis.eval_freq(c, invalid)


def test_copied_arrays_survive_operator_destruction():
    rf = build_dlr_rf(20, 1e-14)
    it = ImTime(20, np.ascontiguousarray(rf))
    iw = ImFreq(20, np.asfortranarray(rf), False)
    c = np.zeros((len(rf), 1), dtype=complex)
    c[3, 0] = 1
    arrays = [rf, it.nodes, iw.nodes, it.coefs2vals(c), iw.coefs2vals(2.7, c)]
    expected = [x.copy() for x in arrays]
    del it, iw
    gc.collect()
    for _ in range(4):
        ImTime(20, rf)
        ImFreq(20, rf, True)
    for actual, reference in zip(arrays, expected):
        np.testing.assert_array_equal(actual, reference)


def test_boson_zero_pole_limit():
    rf = np.array([-1., 0., 1.])
    iw = ImFreq(10, rf, True)
    c = np.zeros((3, 1), dtype=complex)
    c[1, 0] = 1
    np.testing.assert_array_equal(iw.coefs2vals(1., c)[:, 0], np.where(iw.nodes == 0, -0.5, 0))
    nodes = np.ascontiguousarray(iw.nodes, dtype=np.int64)
    np.testing.assert_array_equal(iw.coefs2eval(1., c, nodes)[:, 0], np.where(nodes == 0, -0.5, 0))


def test_raw_frequency_indices_require_int64():
    rf = build_dlr_rf(10, 1e-14)
    iw = ImFreq(10, rf, False)
    c = np.zeros((len(rf), 1), dtype=np.complex128)
    c[3, 0] = 1
    assert iw.coefs2eval(1., c, np.array([0], dtype=np.int64)).shape == (1, 1)
    with pytest.raises(TypeError):
        iw.coefs2eval(1., c, np.array([0.5]))


@pytest.mark.parametrize("lamb,eps", [(0, 1e-14), (np.nan, 1e-14), (np.inf, 1e-14),
                                       (20, 0), (20, np.nan), (20, 1)])
def test_invalid_basis(lamb, eps):
    with pytest.raises(ValueError):
        Basis(lamb, eps)


@pytest.mark.parametrize("rf", [[], [1, -1], [1, 1], [np.nan], [21]])
def test_invalid_poles(rf):
    with pytest.raises(ValueError):
        ImTime(20, np.asarray(rf, dtype=float))


@pytest.mark.parametrize("rf", [[-1., 0., 1.], [-2., 1.]])
def test_invalid_symmetric_poles(rf):
    with pytest.raises(ValueError):
        ImTime(20, np.asarray(rf), True)


def test_invalid_statistic():
    with pytest.raises(ValueError):
        Basis(20, statistic="unknown")


def test_rank_degenerate_raw_poles_raise_without_crashing():
    code = """
import numpy as np
from pycppdlr import ImTime, ImFreq
cases = [lambda: ImTime(1, np.linspace(-1, 1, 50)),
         lambda: ImFreq(1, np.linspace(-1, 1, 50), False),
         lambda: ImTime(1, np.array([-3e-200, -1e-200, 1e-200, 3e-200]))]
for call in cases:
    try:
        call()
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError('invalid poles were accepted')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_invalid_transform_inputs(basis):
    with pytest.raises(ValueError):
        basis.tau_to_coef(np.zeros(basis.rank + 1))
    with pytest.raises(ValueError):
        basis.coef_to_freq(np.zeros(basis.rank), beta=0)
    for t in ([-0.6], [0.6], [np.nan]):
        with pytest.raises(ValueError):
            basis.eval_tau(np.zeros(basis.rank), t)
    with pytest.raises(ValueError):
        basis.eval_freq(np.zeros(basis.rank), [0.5])
    with pytest.raises(ValueError):
        basis.tau_to_coef(np.full(basis.rank, np.nan))


def test_thread_determinism(tmp_path):
    code = """
import numpy as np, sys
from pycppdlr import Basis, PairedBoson
arrays = {}
for statistic in ('fermion', 'boson'):
    for sym in (False, True):
        b = Basis(1000, 1e-14, statistic, sym)
        for name in ('rf', 't_rel', 'n'):
            arrays[f'{statistic}_{sym}_{name}'] = getattr(b, name)
        eye = np.eye(b.rank)
        arrays[f'{statistic}_{sym}_tau'] = b.coef_to_tau(eye)
        arrays[f'{statistic}_{sym}_freq'] = b.coef_to_freq(eye)
b = PairedBoson(1000, 1e-14)
for name in ('rf', 't_rel', 'n'):
    arrays[f'paired_{name}'] = getattr(b, name)
arrays['paired_tau'] = b.coef_to_tau(np.eye(b.rank))
arrays['paired_freq'] = b.coef_to_freq(np.eye(b.rank))
np.savez(sys.argv[1], **arrays)
"""
    files = []
    for omp, blas in itertools.product((1, 8), repeat=2):
        path = tmp_path / f"{omp}_{blas}.npz"
        env = dict(os.environ, OMP_NUM_THREADS=str(omp), OPENBLAS_NUM_THREADS=str(blas))
        subprocess.run([sys.executable, "-c", code, str(path)], env=env, check=True, capture_output=True, text=True)
        files.append(path)
    with np.load(files[0]) as reference:
        for path in files[1:]:
            with np.load(path) as actual:
                for key in reference.files:
                    assert actual[key].dtype == reference[key].dtype
                    assert actual[key].shape == reference[key].shape
                    assert actual[key].tobytes() == reference[key].tobytes(), (path.name, key)


@pytest.mark.slow
@pytest.mark.parametrize("lamb,statistic,sym,eps", list(itertools.product(
    (1e3, 1e4, 5e4), ("fermion", "boson"), (False, True), (1e-14, 1e-15))))
def test_construction_benchmark_and_large_cutoff(lamb, statistic, sym, eps):
    code = """
import importlib.util, json, sys, time
from pycppdlr import Basis
spec = importlib.util.spec_from_file_location('checks', sys.argv[1])
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
start = time.perf_counter()
b = Basis(float(sys.argv[2]), float(sys.argv[5]), sys.argv[3], sys.argv[4] == 'True')
seconds = time.perf_counter() - start
errors = checks.accuracy(b)
print(json.dumps(dict(lamb=b.lamb, statistic=b.statistic, symmetrize=b.symmetrize,
    eps=b.eps, rank=b.rank, seconds=seconds, errors=errors)))
"""
    result = subprocess.run(
        ["/usr/bin/time", "-l", sys.executable, "-c", code, str(Path(__file__).resolve()),
         str(lamb), statistic, str(sym), str(eps)],
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        capture_output=True, text=True, check=True)
    metrics = json.loads(result.stdout)
    metrics["peak_rss_bytes"] = int(re.search(r"(\d+)\s+maximum resident set size", result.stderr)[1])
    print("BENCHMARK " + json.dumps(metrics, sort_keys=True), flush=True)
    if eps == 1e-14:
        # Keep the requested tau-origin accuracy target unchanged.
        errors = {k: v for k, v in metrics["errors"].items() if "_tau_" in k}
        assert max(errors.values()) <= 1e-12, metrics


@pytest.mark.slow
def test_native_and_numpy_lu_diagnostic():
    b = Basis(5e4, 1e-15, "boson", True)
    poles = np.array([-0.371 * b.lamb, 0.113 * b.lamb, 0.913])
    weights = np.array([0.7 + 0.2j, -0.3 + 0.4j, 0.6 - 0.1j])
    t = evaluation_times()
    expected_t = exact_tau_rel(t, poles, weights)
    expected_f = exact_freq(np.arange(-157, 158), poles, weights, b.statistic, 7.3)
    matrix = b.coef_to_tau(np.eye(b.rank))
    stable_values = exact_tau_rel(b.t_rel, poles, weights)
    absolute_values = exact_tau(b.u, poles, weights)
    coefficients = {
        "native_relative_input": b.tau_to_coef(stable_values),
        "numpy_relative_input": np.linalg.solve(matrix, stable_values),
        "numpy_legacy_absolute_input": np.linalg.solve(matrix, absolute_values),
    }
    diagnostics = {
        "lamb": b.lamb, "eps": b.eps, "rank": b.rank,
        "relative_absolute_input_difference": float(np.max(np.abs(stable_values - absolute_values))),
        "native_numpy_coefficient_difference": float(np.max(np.abs(
            coefficients["native_relative_input"] - coefficients["numpy_relative_input"]))),
        "errors": {},
    }
    for label, c in coefficients.items():
        diagnostics["errors"][label] = {
            "tau": float(np.max(np.abs(b.eval_tau(c, t) - expected_t)) / np.max(np.abs(expected_t))),
            "freq": float(np.max(np.abs(b.eval_freq(c, np.arange(-157, 158), 7.3) - expected_f))
                          / np.max(np.abs(expected_f))),
        }
    # This diagnoses the old solver/input paths; it does not alter acceptance limits.
    print("SOLVER_DIAGNOSTIC " + json.dumps(diagnostics, sort_keys=True), flush=True)
