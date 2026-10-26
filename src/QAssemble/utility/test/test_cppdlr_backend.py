"""Analytic functions independent of either backend's selected nodes."""

import numpy as np
import pytest

from QAssemble.utility.DLR import CppdlrDLR


@pytest.fixture(scope="module", params=[(-1, False), (1, False), (1, True)])
def basis(request):
    xi, symmetry = request.param
    return CppdlrDLR(100, 1e-14, xi, symmetry)


def time_values(u, poles, weights):
    u = np.asarray(u)
    kernel = np.column_stack([
        np.exp(-u * w) / (1 + np.exp(-w)) if w >= 0 else
        np.exp((1 - u) * w) / (1 + np.exp(w)) for w in poles
    ])
    return np.einsum("nx,x...->n...", kernel, weights)


def frequency_values(z, poles, weights, beta, xi, *, evaluation=False):
    kernel = 1 / (z[:, None] + poles / beta) if evaluation else -1 / (z[:, None] - poles / beta)
    if xi == 1:
        kernel *= np.tanh(poles / 2)
    out = np.einsum("nx,x...->n...", kernel, weights)
    if out.ndim == 3:
        out = out.swapaxes(1, 2)
    return out.conj() if evaluation else out


def analytic_data(basis):
    residue = np.array([[1., .2 + .17j], [.2 - .17j, .7]])
    if basis.symmetrize:
        poles = np.array([-37.1, -.913, .913, 37.1])
        weights = -np.array([.7, .3, .3, .7])[:, None, None] * residue.real
    else:
        poles = np.array([-37.1, 11.3, .913])
        weights = -np.array([.7, .3, .6])[:, None, None] * residue
    return poles, weights


def test_analytic_multipole_matrix_transforms_and_evaluation(basis):
    beta = 7.3
    poles, weights = analytic_data(basis)
    tau = time_values(basis.dlrit, poles, weights)
    nodes = basis.get_matsubara_frequencies(beta)
    frequency = frequency_values(nodes, poles, weights, beta, basis.xi)
    # The inverse consumes unswapped matrix axes; only forward/eval transpose.
    coefficients = [basis.dlr_from_tau(tau),
                    basis.dlr_from_matsubara(frequency.swapaxes(1, 2), beta)]
    uniform_tau = np.linspace(0, beta, 97)
    uniform_z = 1j * np.pi / beta * (2 * np.arange(-40, 41) + (basis.xi == -1))
    for c in coefficients:
        np.testing.assert_allclose(basis.tau_from_dlr(c), tau, atol=1e-12, rtol=0)
        np.testing.assert_allclose(basis.matsubara_from_dlr(c, beta), frequency, atol=1e-12, rtol=0)
        np.testing.assert_allclose(basis.eval_dlr_tau(c, uniform_tau, beta),
                                   time_values(uniform_tau / beta, poles, weights), atol=1e-12, rtol=0)
        expected = frequency_values(uniform_z, poles, weights, beta, basis.xi, evaluation=True)
        np.testing.assert_allclose(basis.eval_dlr_freq(c, uniform_z, beta), expected, atol=1e-12, rtol=0)
        if basis.symmetrize:
            np.testing.assert_array_equal(c, c[::-1])
            np.testing.assert_array_equal(c.imag, 0)
            np.testing.assert_array_equal(basis.matsubara_from_dlr(c, beta).imag, 0)
            values = basis.eval_dlr_tau(c, uniform_tau, beta)
            np.testing.assert_allclose(values, values[::-1], atol=1e-14, rtol=0)


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (2, 3, 2)])
def test_unit_coefficients_preserve_axes_sign_and_legacy_frequency_conventions(basis, shape):
    beta = 3.7
    count = int(np.prod(shape)) if shape else 1
    amplitude = (1 + np.arange(count).reshape(shape)) / count * (.7 + .2j)
    c = np.zeros((basis.rank, *shape), dtype=complex)
    c[3] = -amplitude
    if basis.symmetrize:
        c = .5 * (c.real + c[::-1].real)
    expected_tau = time_values(basis.dlrit, basis.dlrrf, c)
    expected_freq = frequency_values(basis.get_matsubara_frequencies(beta), basis.dlrrf, c, beta, basis.xi)
    np.testing.assert_allclose(basis.tau_from_dlr(c), expected_tau, atol=1e-14, rtol=0)
    np.testing.assert_allclose(basis.matsubara_from_dlr(c, beta), expected_freq, atol=1e-14, rtol=0)
    for z in (basis.get_matsubara_frequencies(beta), np.array([-.3 + .13j, .7 + .2j])):
        np.testing.assert_allclose(basis.eval_dlr_freq(c, z, beta),
            frequency_values(z, basis.dlrrf, c, beta, basis.xi, evaluation=True), atol=1e-14, rtol=0)
    assert np.sum(c).real < 0  # Causal residues retain pydlr's negative coefficient sign.


def test_native_node_order_and_beta_scaling(basis):
    assert np.all(np.diff(basis.dlrit) > 0)
    assert np.all(np.diff(basis.dlrmf) > 0)
    c = -np.eye(basis.rank)[:, :2]
    if basis.symmetrize:
        c = .5 * (c + c[::-1])
    np.testing.assert_allclose(basis.tau_from_dlr(c),
        basis.basis.coef_to_tau(-c)[basis._tau_order], atol=0, rtol=0)
    np.testing.assert_allclose(basis.matsubara_from_dlr(c, 2),
        2 * basis.matsubara_from_dlr(c, 1), atol=0, rtol=0)
    if basis.symmetrize:
        assert len(basis.dlrit) == basis.rank
        assert len(basis.dlrmf) == basis.rank + 1
        assert np.count_nonzero(basis.dlrmf == 0) == 1
        assert not np.any(basis.dlrrf == 0)


def test_eps_clamp_and_invalid_inputs(basis):
    assert CppdlrDLR(10, 1e-15, basis.xi, basis.symmetrize).eps == 1e-14
    for invalid in (np.zeros(basis.dlrit.size + 1), np.array(0)):
        with pytest.raises(ValueError, match="rows"):
            basis.dlr_from_tau(invalid)
    with pytest.raises(ValueError, match="rows"):
        basis.dlr_from_matsubara(np.zeros(basis.dlrmf.size + 1), 1)
    with pytest.raises(ValueError):
        basis.dlr_from_matsubara(np.zeros(len(basis.dlrmf)), 1, xi=-basis.xi)
    with pytest.raises(ValueError):
        basis.eval_dlr_tau(np.zeros(basis.rank), [-.1], 1)
    with pytest.raises(ValueError):
        basis.eval_dlr_freq(np.zeros(basis.rank), [np.nan], 1)


def test_absolute_tau_right_endpoint_is_preserved(basis):
    c = -np.ones(basis.rank)
    np.testing.assert_allclose(basis.eval_dlr_tau(c, [0, 2], 2),
        time_values(np.array([0, 1]), basis.dlrrf, c), atol=1e-13, rtol=0)


def test_frequency_outside_native_index_range_uses_complex_kernel(basis):
    c = -np.ones(basis.rank)
    z = 1j * np.pi * (2 * np.array([-2**31, 2**31], dtype=float) + (basis.xi == -1))
    np.testing.assert_allclose(basis.eval_dlr_freq(c, z, 1),
        frequency_values(z, basis.dlrrf, c, 1, basis.xi, evaluation=True), atol=1e-22, rtol=1e-14)


def test_node_transforms_delegate_to_native_solvers(basis, monkeypatch):
    import scipy.linalg

    def forbidden(*args, **kwargs):
        raise AssertionError("node transforms must use cppdlr")

    c = -np.eye(basis.rank)[:, 3]
    if basis.symmetrize:
        c = .5 * (c + c[::-1])
    tau = basis.tau_from_dlr(c)
    freq = basis.matsubara_from_dlr(c, 3.7)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(scipy.linalg, "lu_solve", forbidden)
    np.testing.assert_allclose(basis.tau_from_dlr(basis.dlr_from_tau(tau)), tau, atol=1e-12, rtol=0)
    np.testing.assert_allclose(basis.matsubara_from_dlr(basis.dlr_from_matsubara(freq, 3.7), 3.7),
                               freq, atol=1e-12, rtol=0)


def test_complex_frequency_evaluation_retains_legacy_antilinearity(basis):
    c = -np.ones((basis.rank, 2, 2))
    z = np.array([-.3 + .2j, .7 + .1j])
    if basis.symmetrize:
        np.testing.assert_array_equal(basis.eval_dlr_freq(1j*c, z, 2), 0)
        np.testing.assert_array_equal(basis.matsubara_from_dlr(1j*c, 2), 0)
    else:
        np.testing.assert_allclose(basis.eval_dlr_freq(1j*c, z, 2),
                                   -1j*basis.eval_dlr_freq(c, z, 2), atol=1e-14, rtol=0)
        np.testing.assert_allclose(basis.matsubara_from_dlr(1j*c, 2),
                                   1j*basis.matsubara_from_dlr(c, 2), atol=1e-14, rtol=0)
