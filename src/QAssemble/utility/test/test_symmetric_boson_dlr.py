"""K+ kernel normalization, redundant mirror nodes, and real/even projection."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from pydlr import dlr as NativeDLR

from QAssemble.BLatDyn import BLatDyn
from QAssemble.BLocDyn import BLocDyn
from QAssemble.FLatDyn import FLatDyn
from QAssemble.FLocDyn import FLocDyn
from QAssemble.utility.Causal import CausalBosonProjector
from QAssemble.utility.DLR import DLR, SymmetricBosonDLR


@pytest.fixture
def basis():
    return SymmetricBosonDLR(40.0, eps=1e-14)


def test_nodes_are_mirror_complete_and_deterministic(basis):
    other = SymmetricBosonDLR(40.0, eps=1e-14)
    np.testing.assert_array_equal(basis.dlrrf, other.dlrrf)
    np.testing.assert_array_equal(basis.dlrit, other.dlrit)
    np.testing.assert_array_equal(basis.dlrmf, other.dlrmf)
    assert basis.dlrrf[0] == 0
    assert basis.rank < basis.dlrit.size
    assert basis.rank < basis.dlrmf.size
    assert np.all(np.diff(basis.dlrit) > 0)
    assert np.all(np.diff(basis.dlrmf) > 0)
    np.testing.assert_allclose(basis.dlrit, 1 - basis.dlrit[::-1], atol=1e-16)
    np.testing.assert_array_equal(basis.dlrmf, -basis.dlrmf[::-1])
    assert np.count_nonzero(basis.dlrmf == 0) == 1


@pytest.mark.parametrize("pole", [0, 3, -1])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_single_pole_roundtrip_preserves_axes_and_is_real(basis, pole, batch_shape):
    beta = 20.0
    coeff = np.zeros((basis.rank, *batch_shape))
    coeff[pole] = np.arange(np.prod(batch_shape) or 1).reshape(batch_shape) + 1
    tau = basis.tau_from_dlr(coeff)
    freq = basis.matsubara_from_dlr(coeff, beta)
    tau_back = basis.tau_from_dlr(basis.dlr_from_matsubara(freq, beta))
    freq_back = basis.matsubara_from_dlr(basis.dlr_from_tau(tau), beta)
    assert tau.shape == tau_back.shape == (basis.dlrit.size, *batch_shape)
    assert freq.shape == freq_back.shape == (basis.dlrmf.size, *batch_shape)
    assert np.max(abs(tau_back - tau)) < 1e-12
    assert np.max(abs(freq_back - freq)) < 1e-12
    assert np.count_nonzero(tau_back.imag) == np.count_nonzero(freq_back.imag) == 0


def test_zero_mode_is_constant_in_time_and_delta_in_frequency(basis):
    coeff = np.zeros(basis.rank)
    coeff[0] = 1
    np.testing.assert_array_equal(basis.tau_from_dlr(coeff), np.ones(basis.dlrit.size))
    z = 1j * np.array([-0.3, 0, 0.3])
    np.testing.assert_array_equal(basis.eval_dlr_freq(coeff, z, 20), [0, 20, 0])


def test_even_single_pole_matches_native_pydlr(basis):
    beta, x = 20.0, 1.7
    native = NativeDLR(lamb=40.0, eps=1e-14, xi=1, dense_imfreq=False)

    def exact_tau(t):
        t = t / beta
        return (np.exp(-t * x) + np.exp(-(1 - t) * x)) / (1 + np.exp(-x))

    coeff = basis.dlr_from_tau(exact_tau(basis.get_tau(beta)))
    native_coeff = native.dlr_from_tau(exact_tau(native.get_tau(beta)))
    tau = np.linspace(0, beta, 97)
    z = 2j * np.pi / beta * np.arange(-20, 21)
    np.testing.assert_allclose(basis.eval_dlr_tau(coeff, tau, beta),
                               native.eval_dlr_tau(native_coeff, tau, beta), atol=1e-12, rtol=0)
    np.testing.assert_allclose(basis.eval_dlr_freq(coeff, z, beta),
                               native.eval_dlr_freq(native_coeff, z, beta, xi=1), atol=1e-12, rtol=0)


@pytest.mark.parametrize("domain", ["tau", "Matsubara"])
def test_odd_input_warns_before_projection(basis, domain, caplog):
    values = np.ones(basis.dlrit.size if domain == "tau" else basis.dlrmf.size)
    values[0] += 0.1
    with caplog.at_level(logging.WARNING, logger="QAssemble"):
        coeff = (basis.dlr_from_tau(values) if domain == "tau"
                 else basis.dlr_from_matsubara(values, 20))
    assert f"K+ {domain} projection" in caplog.text
    assert "odd residual 9.090909e-02" in caplog.text
    assert np.count_nonzero(coeff.imag) == 0


def test_zero_input_does_not_warn(basis, caplog):
    np.testing.assert_array_equal(basis.dlr_from_tau(np.zeros(basis.dlrit.size)), 0)
    assert not caplog.records


def test_invalid_sample_shape_and_nonfinite_values_are_rejected(basis):
    with pytest.raises(ValueError, match="rows"):
        basis.dlr_from_tau(np.ones(basis.rank))
    with pytest.raises(ValueError, match="finite"):
        basis.dlr_from_matsubara(np.full(basis.dlrmf.size, np.nan), 20)


@pytest.fixture
def symmetric_dlr():
    return DLR({"beta": 20.0, "cutoff": 8.0, "eps": 1e-12})


def test_symmetry_default_and_legacy_opt_out(symmetric_dlr):
    assert isinstance(symmetric_dlr.dB, SymmetricBosonDLR)
    legacy = DLR({"beta": 20.0, "cutoff": 8.0, "eps": 1e-12, "symmetry": False})
    assert isinstance(legacy.dB, NativeDLR)
    assert legacy.lambB == legacy.lambF == symmetric_dlr.lambB


def test_unsupported_causal_qp_fails_explicitly(symmetric_dlr):
    with pytest.raises(NotImplementedError, match="BosonSymmetry=False"):
        CausalBosonProjector(d=symmetric_dlr.dB, beta=symmetric_dlr.beta,
                             fit_omega=symmetric_dlr.nu)


def test_t2mt_is_exact_permutation_for_non_even_fermions(symmetric_dlr):
    d = symmetric_dlr
    values = -np.exp(-0.3 * d.tauB) / (1 + np.exp(-0.3 * d.beta))
    assert np.max(abs(values - values[::-1])) > 0.1
    np.testing.assert_array_equal(d.T2mT(values), -values[::-1])
    np.testing.assert_array_equal(d.T2mT(values, d.tauB), -values[::-1])
    np.testing.assert_allclose(d.T2mT(values),
                               np.exp(-0.3 * (d.beta - d.tauB)) / (1 + np.exp(-0.3 * d.beta)),
                               atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match="mirror partners"):
        d.T2mT(values, np.array([0.12345]))


@pytest.mark.parametrize("lattice", [False, True])
def test_local_and_lattice_batch_roundtrip_and_w_on_fermion_tau(symmetric_dlr, lattice):
    d = symmetric_dlr
    shape = (2, 2, 2, 2, 3) if lattice else (2, 2, 2, 2)
    rng = np.random.default_rng(90)
    coefficients = rng.normal(size=(d.dB.rank, *shape))
    coefficients *= np.exp(-np.arange(d.dB.rank)).reshape((-1,) + (1,) * len(shape))
    freq = np.moveaxis(d.dB.matsubara_from_dlr(coefficients, d.beta), 0, -1)
    owner = SimpleNamespace(dlr=d)
    boson = BLatDyn if lattice else BLocDyn
    fermion = FLatDyn if lattice else FLocDyn
    tau = boson.F2T(owner, freq)
    assert tau.shape == (*shape, len(d.tauB))
    np.testing.assert_allclose(boson.T2F(owner, tau), freq, atol=1e-12, rtol=0)
    on_fermion_tau = fermion.TauB2TauF(owner, tau)
    expected = np.moveaxis(d.dB.eval_dlr_tau(coefficients, d.tauF, d.beta), 0, -1)
    np.testing.assert_allclose(on_fermion_tau, expected, atol=1e-12, rtol=0)
    assert np.isfortran(tau) and np.isfortran(on_fermion_tau)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_uniform_fit_uses_coefficient_rank_and_preserves_axes(symmetric_dlr, ndim):
    d = symmetric_dlr
    shape = [(), (3,), (2, 3)][ndim - 1]
    coefficients = np.zeros((d.dB.rank, *shape))
    coefficients[4] = np.arange(np.prod(shape) or 1).reshape(shape) + 1
    grid = d.MatsubaraBosonUniformFull(Emax=max(abs(d.nu)))
    uniform = d.dB.eval_dlr_freq(coefficients, 1j * grid, d.beta)
    actual = d.MatsubaraUniform2DLR(uniform, grid, sign=1)
    expected = d.dB.matsubara_from_dlr(coefficients, d.beta)
    assert actual.shape == (len(d.nu), *shape)
    assert np.count_nonzero(actual.imag) == 0
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_uniform_fit_warns_about_odd_input(symmetric_dlr, caplog):
    d = symmetric_dlr
    grid = d.MatsubaraBosonUniformFull(Emax=max(abs(d.nu)))
    values = np.ones(grid.size) + 1j * grid
    with caplog.at_level(logging.WARNING, logger="QAssemble"):
        out = d.MatsubaraUniform2DLR(values, grid, sign=1)
    assert "uniform Matsubara projection: odd residual" in caplog.text
    assert np.count_nonzero(out.imag) == 0


def test_batched_and_scalar_dlr_to_uniform_evaluation(symmetric_dlr):
    d = symmetric_dlr
    coefficients = np.zeros((d.dB.rank, 3))
    coefficients[3] = [1, 2, 3]
    freq = d.dB.matsubara_from_dlr(coefficients, d.beta)
    expected = d.dB.eval_dlr_freq(coefficients, 1j * d.MatsubaraBosonUniform(), d.beta)
    actual = d.MatsubaraDLR2Uniform(freq, sign=1, method="dlr").reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    scalar = d.MatsubaraDLR2Uniform(freq[:, 0], sign=1, method="dlr")
    np.testing.assert_allclose(scalar[:, 0, 0], expected[:, 0], atol=1e-12, rtol=0)


@pytest.mark.parametrize("domain", ["tau", "Matsubara"])
def test_even_imaginary_input_warns_before_real_sector_projection(basis, domain, caplog):
    values = np.full(basis.dlrit.size if domain == "tau" else basis.dlrmf.size, 1 + 0.1j)
    with caplog.at_level(logging.WARNING, logger="QAssemble"):
        coeff = (basis.dlr_from_tau(values) if domain == "tau"
                 else basis.dlr_from_matsubara(values, 20))
    assert "imaginary residual" in caplog.text
    assert np.count_nonzero(coeff.imag) == 0

