import matplotlib.pyplot as plt
import numpy as np
import pytest

from QAssemble.Crystal import Crystal
from QAssemble.FLatDyn import FLatDyn
from QAssemble.FLocDyn import FLocDyn
from QAssemble.utility.Causal import CausalFermion, CausalProjection
from QAssemble.utility.DLR import DLR


def _single_band_hubbard():
    crystal = Crystal(
        {
            "RVec": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "Basis": [[[0, 0, 0], 1]],
            "NSpin": 1,
            "NElec": 1.0,
            "KGrid": [2, 2, 2],
        }
    )
    kx, ky, kz = crystal.kpoint[:, 0], crystal.kpoint[:, 1], crystal.kpoint[:, 2]
    eps = -2.0 * (
        np.cos(2.0 * np.pi * kx)
        + np.cos(2.0 * np.pi * ky)
        + np.cos(2.0 * np.pi * kz)
    )
    hamtb = np.zeros((1, 1, 1, crystal.nk), dtype=np.complex128, order="F")
    hamtb[0, 0, 0, :] = eps
    dlr = DLR({"beta": 20.0, "cutoff": 8.0, "eps": 1.0e-12})
    return crystal, dlr, hamtb


class _Verifier:
    """Implementation-independent kernel/moment construction.

    Built directly from ``eval_dlr_freq`` unit vectors and
    ``get_dlr_frequencies()/beta`` so it does not share code with the
    projector under test.
    """

    def __init__(self, dlr):
        self.beta = float(dlr.beta)
        x = np.asarray(dlr.dF.get_dlr_frequencies(), dtype=float)
        self.nodes = x / self.beta
        self.omega = np.asarray(dlr.omega, dtype=float)
        z = 1j * self.omega
        kernel = np.empty((self.omega.size, x.size), dtype=np.complex128)
        for j in range(x.size):
            unit = np.zeros((x.size, 1, 1), dtype=np.complex128)
            unit[j, 0, 0] = 1.0
            kernel[:, j] = dlr.dF.eval_dlr_freq(unit, z, self.beta, xi=-1)[:, 0, 0]
        self.kernel = kernel
        self.moment_rows = np.vstack([self.nodes**power for power in range(3)])

    def fit(self, values):
        lhs = np.vstack((self.kernel.real, self.kernel.imag))
        rhs = np.concatenate((np.real(values), np.imag(values)))
        coeff, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        return np.asarray(coeff, dtype=float)

    def reconstruct(self, coefficients):
        return self.kernel @ np.asarray(coefficients, dtype=float)


def _fermion(dlr, *, constraint_tol=1.0e-7, **kwargs):
    return CausalFermion(
        d=dlr.dF,
        beta=dlr.beta,
        omega=dlr.omega,
        constraint_tol=constraint_tol,
        **kwargs,
    )


def _causal_coefficients(verifier):
    return -0.2 * np.exp(-0.1 * verifier.nodes**2) - 0.05


def _bad_coefficients(verifier, column=0, alpha_extra=0.2):
    moment_rows = verifier.moment_rows
    _, _, vh = np.linalg.svd(moment_rows)
    null_space = vh[3:].T
    direction = null_space[:, column % null_space.shape[1]]
    if direction[np.argmax(np.abs(direction))] < 0.0:
        direction = -direction
    base = _causal_coefficients(verifier)
    idx = int(np.argmax(direction))
    alpha = abs(base[idx]) / direction[idx] + alpha_extra
    bad = base + alpha * direction
    assert np.max(bad) > 0.0
    np.testing.assert_allclose(moment_rows @ bad, moment_rows @ base, atol=1.0e-10)
    return bad


def _hubbard_lattice_target(crystal, dlr):
    verifier = _Verifier(dlr)
    nfreq = len(dlr.omega)
    mat = np.zeros((1, 1, 1, crystal.nk, nfreq), dtype=np.complex128, order="F")
    coeffs = []
    for ik in range(crystal.nk):
        coeff = _bad_coefficients(verifier, column=ik, alpha_extra=0.2 + 0.01 * ik)
        coeffs.append(coeff)
        mat[0, 0, 0, ik, :] = verifier.reconstruct(coeff)
    return mat, np.asarray(coeffs)


def _assert_projected_channel(verifier, values, moment_target):
    coeff = verifier.fit(values)
    assert np.max(coeff) <= 1.0e-5
    np.testing.assert_allclose(
        verifier.moment_rows @ coeff,
        moment_target,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def _show_projection_comparison(
    dlr,
    verifier,
    before_values,
    after_values,
    before_coefficients,
    after_coefficients,
    title,
):
    order = np.argsort(verifier.nodes)
    omega = np.asarray(dlr.omega, dtype=float)
    before_values = np.asarray(before_values, dtype=np.complex128)
    after_values = np.asarray(after_values, dtype=np.complex128)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), constrained_layout=True)
    axes[0].plot(
        verifier.nodes[order],
        np.asarray(before_coefficients)[order],
        "o-",
        ms=3,
        lw=1.0,
        label="before",
    )
    axes[0].plot(
        verifier.nodes[order],
        np.asarray(after_coefficients)[order],
        "o-",
        ms=3,
        lw=1.0,
        label="projected",
    )
    axes[0].axhline(0.0, color="black", lw=0.8, ls=":")
    axes[0].set_xlabel(r"$\epsilon_l$")
    axes[0].set_ylabel(r"$A_l$")
    axes[0].set_title("pole weights")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(omega, before_values.real, lw=1.0, label="before")
    axes[1].plot(omega, after_values.real, lw=1.0, label="projected")
    axes[1].set_xlabel(r"$\omega_n$")
    axes[1].set_ylabel(r"$\mathrm{Re}\,G(i\omega_n)$")
    axes[1].set_title("real part")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(omega, before_values.imag, lw=1.0, label="before")
    axes[2].plot(omega, after_values.imag, lw=1.0, label="projected")
    axes[2].set_xlabel(r"$\omega_n$")
    axes[2].set_ylabel(r"$\mathrm{Im}\,G(i\omega_n)$")
    axes[2].set_title("imaginary part")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle(title)
    plt.show()
    plt.close(fig)


def test_causal_fermion_enforces_sign_and_preserves_fitted_moments():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    bad = _bad_coefficients(verifier)
    target = verifier.reconstruct(bad)

    fermion = _fermion(dlr)
    projected = fermion.project(target)

    assert projected.shape == target.shape
    assert not fermion.last_validation["skipped"]

    coeff = fermion.last_coefficients
    assert np.max(coeff) <= 1.0e-6
    # moments are anchored internally to the unconstrained fit of the input
    np.testing.assert_allclose(
        fermion.moment_rows @ coeff,
        verifier.moment_rows @ bad,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    _assert_projected_channel(verifier, projected, verifier.moment_rows @ bad)

    relative = np.linalg.norm(projected - target) / np.linalg.norm(target)
    assert np.isfinite(relative)
    assert relative < 0.5


def test_causal_fermion_check_reports_unscaled_violations():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    bad = _bad_coefficients(verifier)
    amplitude = 7.0  # push scale > 1 so unscaled reporting is exercised
    target = amplitude * verifier.reconstruct(bad)

    fermion = _fermion(dlr)
    verdict = fermion.check(target)

    assert not verdict.causal
    assert verdict.violating_count >= 1
    # the max violation must match the positive fitted weights in data units
    expected = amplitude * float(np.max(bad))
    assert verdict.max_inequality_violation == pytest.approx(expected, rel=1.0e-2)


def test_causal_fermion_skips_qp_for_causal_input():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    causal_coeff = _causal_coefficients(verifier)
    target = verifier.reconstruct(causal_coeff)

    fermion = _fermion(dlr)
    assert fermion.check(target).causal

    projected = fermion.project(target)
    assert fermion.last_validation["skipped"] is True
    assert fermion.last_solver is None
    assert fermion.last_status == "skipped"
    # the skip path still returns the refit kernel @ reference
    np.testing.assert_allclose(
        projected,
        fermion.kernel @ fermion.last_coefficients,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(projected, target, atol=1.0e-8)


def test_causal_fermion_gate_rejects_nonhermitian_noise():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    target = verifier.reconstruct(_causal_coefficients(verifier))

    rng = np.random.default_rng(20260612)
    noise = 1.0e-3 * (
        rng.normal(size=target.size) + 1j * rng.normal(size=target.size)
    )
    noisy = target + noise

    fermion = _fermion(dlr)
    with pytest.raises(RuntimeError, match="fit_tol"):
        fermion.project(noisy)
    with pytest.raises(RuntimeError, match="fit_tol"):
        fermion.check(noisy)

    # a loosened gate lets the same data through
    loose = _fermion(dlr, fit_tol=1.0)
    loose_projected = loose.project(noisy)
    assert np.all(np.isfinite(loose_projected))


def test_causal_fermion_silent_failure_returns_refit_reference():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    bad = _bad_coefficients(verifier)
    target = verifier.reconstruct(bad)

    fermion = _fermion(
        dlr,
        raise_on_failure=False,
        solvers=("no_such_solver",),
    )
    projected = fermion.project(target)

    # every solver failed: the unprojected (still non-causal) reference is
    # refit through the kernel and the validation flags the violation
    assert fermion.last_validation["valid"] is False
    assert fermion.last_validation["skipped"] is False
    assert np.max(fermion.last_coefficients) > 1.0e-3
    np.testing.assert_allclose(
        projected,
        fermion.kernel @ fermion.last_coefficients,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(projected, target, atol=1.0e-6)


def test_causal_fermion_diagnostics_reset_on_gate_failure():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)

    fermion = _fermion(dlr)
    fermion.project(verifier.reconstruct(_bad_coefficients(verifier)))
    assert fermion.last_validation["valid"] is True
    assert fermion.last_coefficients is not None

    rng = np.random.default_rng(11)
    noisy = verifier.reconstruct(_causal_coefficients(verifier))
    noisy = noisy + 1.0e-3 * (
        rng.normal(size=noisy.size) + 1j * rng.normal(size=noisy.size)
    )
    with pytest.raises(RuntimeError, match="fit_tol"):
        fermion.project(noisy)

    # diagnostics describe the failed call, not the earlier successful one
    assert fermion.last_coefficients is None
    assert fermion.last_solver is None
    assert fermion.last_status == "gate_failed"
    assert fermion.last_validation["gate_failed"] is True
    assert fermion.last_validation["valid"] is False
    assert fermion.last_validation["node_residual"] > 1.0e-6


def test_causal_fermion_normal_equation_diagnostic_is_small():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    target = verifier.reconstruct(_bad_coefficients(verifier))

    fermion = _fermion(dlr)
    fermion.project(target)

    assert fermion.last_validation["node_residual"] <= 1.0e-6
    assert fermion.last_validation["normal_eq_residual"] <= 1.0e-6


def test_weighted_metric_matches_or_beats_identity_metric():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    bad = _bad_coefficients(verifier)
    target = verifier.reconstruct(bad)

    fermion = _fermion(dlr)
    weighted_error = np.linalg.norm(fermion.project(target) - target)

    reference = verifier.fit(target)
    identity = CausalProjection(
        coefficient_sign=-1,
        constraint_tol=1.0e-7,
        raise_on_failure=True,
    )
    identity_result = identity.project(
        reference,
        equality_matrix=verifier.moment_rows,
        equality_target=verifier.moment_rows @ reference,
    )
    identity_error = np.linalg.norm(
        verifier.reconstruct(identity_result.coefficients) - target
    )

    # W = Re(K^H K) minimizes the frequency-space error; never assert strict
    assert weighted_error <= identity_error + 1.0e-8


def test_flatdyn_causal_projection_single_band_hubbard_grid():
    crystal, dlr, hamtb = _single_band_hubbard()
    assert hamtb.shape == (1, 1, 1, crystal.nk)
    flat = FLatDyn(crystal, dlr)
    mat, coeffs = _hubbard_lattice_target(crystal, dlr)
    original = mat.copy()

    projected = flat.CausalProjection(mat, constraint_tol=1.0e-7)

    assert projected.shape == mat.shape
    assert projected.dtype == np.complex128
    assert projected.flags.f_contiguous
    np.testing.assert_allclose(mat, original)

    verifier = _Verifier(dlr)
    for ik in range(crystal.nk):
        _assert_projected_channel(
            verifier,
            projected[0, 0, 0, ik, :],
            verifier.moment_rows @ coeffs[ik],
        )

    ik_plot = 0
    _show_projection_comparison(
        dlr,
        verifier,
        original[0, 0, 0, ik_plot, :],
        projected[0, 0, 0, ik_plot, :],
        coeffs[ik_plot],
        verifier.fit(projected[0, 0, 0, ik_plot, :]),
        "FLatDyn single-band Hubbard k=0",
    )


def test_flocdyn_causal_projection_preserves_fitted_moments():
    crystal, dlr, _ = _single_band_hubbard()
    local = FLocDyn(crystal, dlr, projector=None)
    verifier = _Verifier(dlr)
    bad = _bad_coefficients(verifier)
    nfreq = len(dlr.omega)

    local4 = np.zeros((1, 1, 1, nfreq), dtype=np.complex128, order="F")
    local4[0, 0, 0, :] = verifier.reconstruct(bad)
    local3 = np.asfortranarray(local4[:, :, 0, :])

    projected4 = local.CausalProjection(local4, constraint_tol=1.0e-7)
    projected3 = local.CausalProjection(local3, constraint_tol=1.0e-7)

    assert projected4.shape == local4.shape
    assert projected3.shape == local3.shape
    assert projected4.flags.f_contiguous
    assert projected3.flags.f_contiguous
    np.testing.assert_allclose(projected4[:, :, 0, :], projected3)

    # internal fitted-moment preservation replaces the removed explicit
    # `moments` workflow: the anchor is the unconstrained fit of the input
    _assert_projected_channel(verifier, projected4[0, 0, 0, :], verifier.moment_rows @ bad)
    _show_projection_comparison(
        dlr,
        verifier,
        local4[0, 0, 0, :],
        projected4[0, 0, 0, :],
        bad,
        verifier.fit(projected4[0, 0, 0, :]),
        "FLocDyn single-band Hubbard local channel",
    )


def test_flatdyn_causality_check_reports_channels():
    crystal, dlr, _ = _single_band_hubbard()
    flat = FLatDyn(crystal, dlr)
    mat, _ = _hubbard_lattice_target(crystal, dlr)

    report = flat.CausalityCheck(mat, constraint_tol=1.0e-7)

    assert report["causal"].shape == (1, 1, crystal.nk)
    assert not report["causal"].any()
    assert np.all(report["max_inequality_violation"][0, 0, :] > 0.0)
    assert np.all(report["violating_count"][0, 0, :] >= 1)
    # representable targets: tiny node residuals (data quality is fine)
    assert np.all(report["node_residual"] <= 1.0e-6)

    projected = flat.CausalProjection(mat, constraint_tol=1.0e-7)
    clean = flat.CausalityCheck(projected, constraint_tol=1.0e-6)
    assert clean["causal"].all()


def test_flocdyn_causality_check_squeeze_and_gate_diagnostics():
    crystal, dlr, _ = _single_band_hubbard()
    local = FLocDyn(crystal, dlr, projector=None)
    verifier = _Verifier(dlr)
    nfreq = len(dlr.omega)

    causal_channel = verifier.reconstruct(_causal_coefficients(verifier))
    rng = np.random.default_rng(3)
    broken_channel = causal_channel + 1.0e-3 * (
        rng.normal(size=nfreq) + 1j * rng.normal(size=nfreq)
    )

    local4 = np.zeros((2, 2, 1, nfreq), dtype=np.complex128, order="F")
    local4[0, 0, 0, :] = causal_channel
    local4[1, 1, 0, :] = broken_channel

    report = local.CausalityCheck(local4)

    assert report["causal"].shape == (2, 1)
    assert report["causal"][0, 0]
    assert report["node_residual"][0, 0] <= 1.0e-6
    # the non-representable channel is reported, never raised
    assert report["node_residual"][1, 0] > 1.0e-6

    local3 = np.asfortranarray(local4[:, :, 0, :])
    report3 = local.CausalityCheck(local3)
    assert report3["causal"].shape == (2,)
    np.testing.assert_allclose(report3["node_residual"], report["node_residual"][:, 0])


def test_causal_projection_input_validation():
    crystal, dlr, _ = _single_band_hubbard()
    flat = FLatDyn(crystal, dlr)
    local = FLocDyn(crystal, dlr, projector=None)
    nfreq = len(dlr.omega)

    with pytest.raises(ValueError, match="frequency dimension"):
        flat.CausalProjection(np.zeros((1, 1, 1, crystal.nk, nfreq + 1), dtype=complex))
    with pytest.raises(ValueError, match="first two dimensions"):
        flat.CausalProjection(np.zeros((1, 2, 1, crystal.nk, nfreq), dtype=complex))
    with pytest.raises(ValueError, match="matin must be 5D"):
        flat.CausalProjection(np.zeros((1, 1, 1, nfreq), dtype=complex))
    with pytest.raises(ValueError, match="matin must be 3D or 4D"):
        local.CausalProjection(np.zeros((1, 1, 1, 1, nfreq), dtype=complex))
