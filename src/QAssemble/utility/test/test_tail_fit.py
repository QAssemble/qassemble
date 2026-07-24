"""Tests for the log-spaced tail fit and sigma estimates in Fourier."""
import numpy as np
import pytest

from QAssemble.utility.Fourier import Fourier


def _signed_uniform_fermion_grid(beta: float, nmax: int) -> np.ndarray:
    n = np.arange(-nmax, nmax)
    return (2.0 * n + 1.0) * np.pi / beta


def _tail_model(omega: np.ndarray, c: np.ndarray) -> np.ndarray:
    z = 1j * omega
    return c[0] + c[1] / z + c[2] / z**2 + c[3] / z**3


TRUE_C = np.array([0.0, 1.0, -0.35, 1.8])


def test_tail_fit_indices_log_spaced_properties():
    omega = _signed_uniform_fermion_grid(beta=10.0, nmax=256)
    idx = Fourier._tail_fit_indices(omega, 24, log_spaced=True)
    assert idx.size >= 4
    assert idx.size == np.unique(idx).size
    assert np.all(omega[idx] > 0.0)
    fmax = omega.max()
    assert np.all(omega[idx] >= fmax / 10.0 - np.diff(omega).max())
    assert omega[idx].max() == pytest.approx(fmax)


def test_tail_fit_indices_default_matches_argsort():
    omega = _signed_uniform_fermion_grid(beta=10.0, nmax=64)
    idx = Fourier._tail_fit_indices(omega, 5, log_spaced=False)
    np.testing.assert_array_equal(idx, np.argsort(np.abs(omega))[-5:])


def test_tail_fit_indices_small_grid_falls_back():
    omega = np.array([-3.0, -1.0, 1.0, 3.0])
    idx = Fourier._tail_fit_indices(omega, 24, log_spaced=True)
    assert idx.size == 4


def test_log_spaced_false_is_bit_identical_to_legacy():
    omega = _signed_uniform_fermion_grid(beta=10.0, nmax=128)
    rng = np.random.default_rng(3)
    target = _tail_model(omega, TRUE_C) + 1.0e-8 * (
        rng.standard_normal(omega.size) + 1j * rng.standard_normal(omega.size)
    )
    c_new = Fourier.FermionTailCoefficients(omega, target, 5)
    # Legacy reference computed inline with the historical selection.
    idx = np.argsort(np.abs(omega))[-5:]
    z = 1j * omega[idx]
    design = np.column_stack([np.ones_like(z), 1.0 / z, 1.0 / z**2, 1.0 / z**3])
    design_ri = np.vstack([design.real, design.imag])
    b_ri = np.concatenate([target[idx].real, target[idx].imag])
    c_legacy, *_ = np.linalg.lstsq(design_ri, b_ri, rcond=None)
    np.testing.assert_allclose(c_new, c_legacy, rtol=0.0, atol=1.0e-12)


def test_log_spaced_fit_recovers_moments_where_legacy_fails():
    omega = _signed_uniform_fermion_grid(beta=10.0, nmax=512)
    rng = np.random.default_rng(7)
    noise = 1.0e-7 * (
        rng.standard_normal(omega.size) + 1j * rng.standard_normal(omega.size)
    )
    target = _tail_model(omega, TRUE_C) + noise

    c_log, sigma = Fourier.FermionTailCoefficients(
        omega, target, 24, log_spaced=True, return_sigma=True
    )
    c_old = Fourier.FermionTailCoefficients(omega, target, 5)

    # Log-spaced fit recovers every moment within 3 sigma.
    err = np.abs(c_log - TRUE_C)
    assert np.all(err <= 3.0 * sigma + 1.0e-10)
    assert np.all(sigma >= 0.0)
    # Regression witness: the contiguous 5-point fit on the signed uniform
    # grid is catastrophically ill-conditioned for c2/c3.
    assert abs(c_old[2] - TRUE_C[2]) > 10.0 * abs(c_log[2] - TRUE_C[2])
    assert abs(c_old[3] - TRUE_C[3]) > 10.0 * abs(c_log[3] - TRUE_C[3])


def test_sigma_is_statistically_calibrated():
    omega = _signed_uniform_fermion_grid(beta=10.0, nmax=512)
    rng = np.random.default_rng(11)
    within = 0
    trials = 40
    for _ in range(trials):
        noise = 1.0e-6 * (
            rng.standard_normal(omega.size) + 1j * rng.standard_normal(omega.size)
        )
        target = _tail_model(omega, TRUE_C) + noise
        c, sigma = Fourier.FermionTailCoefficients(
            omega, target, 24, log_spaced=True, return_sigma=True
        )
        if np.all(np.abs(c - TRUE_C) <= 3.0 * sigma + 1.0e-12):
            within += 1
    assert within >= int(0.9 * trials)


def test_boson_log_spaced_fit_and_sigma():
    beta = 8.0
    nu = 2.0 * np.arange(1, 513) * np.pi / beta  # positive-only bosonic grid
    true_c = np.array([0.0, 0.0, -0.9, 0.0])
    rng = np.random.default_rng(5)
    target = _tail_model(nu, true_c) + 1.0e-8 * (
        rng.standard_normal(nu.size) + 1j * rng.standard_normal(nu.size)
    )
    c, sigma = Fourier.BosonTailCoefficients(
        nu, target, 24, log_spaced=True, return_sigma=True
    )
    assert np.all(np.abs(c - true_c) <= 3.0 * sigma + 1.0e-9)
