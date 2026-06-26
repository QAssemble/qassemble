import matplotlib

matplotlib.use("Agg")  # keep plt.show() non-interactive under pytest

import matplotlib.pyplot as plt
import numpy as np
import pytest

from QAssemble.Crystal import Crystal
from QAssemble.FLatDyn import FLatDyn
from QAssemble.FLocDyn import FLocDyn
from QAssemble.utility.Causal import (
    CausalFermionProjector,
    CausalProjector,
    CausalProjection,
)
from QAssemble.utility.DLR import DLR
from QAssemble.utility.Fourier import Fourier

# the robust tail estimator now lives on Fourier; keep the old test-local name
fermion_tail_coefficients = Fourier.FermionTailCoefficients


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


def _two_orbital_crystal():
    return Crystal(
        {
            "RVec": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "Basis": [[[0, 0, 0], 2]],
            "NSpin": 1,
            "NElec": 2.0,
            "KGrid": [2, 1, 1],
        }
    )


class _Verifier:
    """Implementation-independent kernel/moment construction.

    Built directly from ``eval_dlr_freq`` unit vectors and
    ``get_dlr_frequencies()/beta`` so it does not share code with the
    projector under test.
    """

    def __init__(self, dlr, omega=None):
        self.beta = float(dlr.beta)
        x = np.asarray(dlr.dF.get_dlr_frequencies(), dtype=float)
        self.nodes = x / self.beta
        self.omega = np.asarray(dlr.omega if omega is None else omega, dtype=float)
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


def _fermion(dlr, *, constraint_tol=1.0e-7, omega=None, **kwargs):
    return CausalFermionProjector(
        d=dlr.dF,
        beta=dlr.beta,
        omega=dlr.omega if omega is None else omega,
        constraint_tol=constraint_tol,
        **kwargs,
    )


def _causal_coefficients(verifier):
    return -0.2 * np.exp(-0.1 * verifier.nodes**2) - 0.05


def _tail_model(freq, coeff):
    freq = np.asarray(freq, dtype=float)
    z = 1j * freq
    return (
        coeff[0]
        + coeff[1] / z
        + coeff[2] / z**2
        + coeff[3] / z**3
    )


def _matrix_tail_model(freq, coeff):
    freq = np.asarray(freq, dtype=float)
    z = 1j * freq
    return (
        coeff[0][..., np.newaxis]
        + coeff[1][..., np.newaxis] / z
        + coeff[2][..., np.newaxis] / z**2
        + coeff[3][..., np.newaxis] / z**3
    )


def _hermitian_tail_coefficients(norb):
    coeff = np.zeros((4, norb, norb), dtype=np.complex128)
    for imom in range(4):
        diag = np.linspace(0.2 + imom, 0.6 + imom, norb)
        coeff[imom] += np.diag(diag)
    coeff[:, 0, 1] = np.array([0.3 + 0.2j, -0.7 + 0.1j, 0.5 - 0.4j, 1.2 + 0.3j])
    coeff[:, 1, 0] = np.conj(coeff[:, 0, 1])
    return coeff


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


def _causal_tail_coeffs(verifier, coeff):
    """Analytic physical tail ``[c0, c1, c2, c3]`` of a pure
    decaying channel ``kernel @ coeff``.  Such channels have ``c0 = 0`` and
    the projector's internal ``moment_rows @ coeff`` is
    ``-[c1, c2, c3]``.  Injecting these via
    ``tail_coeffs`` bypasses the noisy 5-point tail fit, which on the coarse
    22-point DLR grid produces a spurious ``c0 ~ 1e-4``; that spurious
    constant is not pole-representable and, amplified by the ill-conditioned
    kernel, destabilizes the *unconstrained* reference fit used by ``check``.
    The QP-based ``project`` regularizes and does not need the injection."""
    return np.concatenate(([0.0], -(verifier.moment_rows @ np.asarray(coeff, float))))


def _assert_projected_channel(verifier, values, moment_target, c0=0.0):
    """Assert ``values`` is a causal, moment-preserving channel.

    ``values = kernel @ coeff + c0`` (the projector re-adds the constant
    ``c0`` on return).  The constant is not pole-representable, so it must be
    subtracted before the (ill-conditioned) refit; otherwise the lstsq blows
    up trying to express it.  After subtraction the recovered pole weights are
    causal and their internal moments equal the negative of the preserved
    physical data-tail moments."""
    coeff = verifier.fit(np.asarray(values, dtype=np.complex128) - c0)
    assert np.max(coeff) <= 1.0e-5
    np.testing.assert_allclose(
        verifier.moment_rows @ coeff,
        -np.asarray(moment_target, dtype=float),
        atol=1.0e-7,
        rtol=1.0e-6,
    )


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


def test_causal_fermion_enforces_sign_and_preserves_data_tail_moments():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    bad = _bad_coefficients(verifier)
    target = verifier.reconstruct(bad)

    fermion = _fermion(dlr)
    projected = fermion.project(target)

    assert projected.shape == target.shape
    assert not fermion.last_validation["skipped"]

    coeff = fermion.last_coefficients
    c0 = fermion.last_validation["c0"]
    assert np.max(coeff) <= 1.0e-6
    # the QP equality target is the internally signed physical DATA tail.
    data_tail = fermion_tail_coefficients(verifier.omega, target)
    np.testing.assert_allclose(
        fermion.moment_rows @ coeff,
        -data_tail[1:],
        atol=1.0e-9,
        rtol=1.0e-9,
    )
    # the returned channel re-adds the unscaled constant tail c0
    np.testing.assert_allclose(
        projected, fermion.kernel @ coeff + c0, atol=1.0e-12
    )
    _assert_projected_channel(verifier, projected, data_tail[1:], c0=c0)

    relative = np.linalg.norm(projected - target) / np.linalg.norm(target)
    assert np.isfinite(relative)
    assert relative < 0.5


def test_fermion_tail_coefficients_are_physical_sign():
    omega = np.array([-9.0, -5.0, -3.0, 3.0, 5.0, 9.0])
    coeff = np.array([0.7, -1.2, 0.4, 2.1])
    target = _tail_model(omega, coeff)

    tail = Fourier.FermionTailCoefficients(omega, target, tail_points=6)

    np.testing.assert_allclose(tail, coeff, atol=1.0e-12)


def test_causal_fermion_check_reports_unscaled_violations():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    bad = _bad_coefficients(verifier)
    amplitude = 7.0  # push scale > 1 so unscaled reporting is exercised
    target = amplitude * verifier.reconstruct(bad)
    # inject the analytic tail (c0 = 0) so the check reflects the genuine
    # sign violation, not the spurious-c0 blowup of the reference fit
    tail = amplitude * _causal_tail_coeffs(verifier, bad)

    fermion = _fermion(dlr)
    verdict = fermion.check(target, tail_coeffs=tail)

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
    tail = _causal_tail_coeffs(verifier, causal_coeff)  # analytic c0 = 0

    fermion = _fermion(dlr)
    assert fermion.check(target, tail_coeffs=tail).causal

    projected = fermion.project(target, tail_coeffs=tail)
    assert fermion.last_validation["skipped"] is True
    assert fermion.last_solver is None
    assert fermion.last_status == "skipped"
    assert fermion.last_validation["c0"] == pytest.approx(0.0, abs=1.0e-12)
    # the skip path still returns the refit kernel @ reference + c0
    np.testing.assert_allclose(
        projected,
        fermion.kernel @ fermion.last_coefficients + fermion.last_validation["c0"],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(projected, target, atol=1.0e-8)


def test_causal_fermion_check_gate_rejects_nonhermitian_noise():
    # project no longer enforces the node-fit gate; only check (enforce_gate
    # default True) raises on data the real-coefficient pole basis cannot
    # represent.
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
        fermion.check(noisy)

    # project tolerates the noise (no gate) and returns a finite causal channel
    projected = fermion.project(noisy)
    assert np.all(np.isfinite(projected))
    assert np.max(fermion.last_coefficients) <= 1.0e-6

    # a loosened gate lets the same data through check, too
    loose = _fermion(dlr, fit_tol=1.0)
    assert np.isfinite(loose.check(noisy).node_residual)


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
    # refit through the kernel, c0 is re-added, and validation flags the
    # violation
    assert fermion.last_validation["valid"] is False
    assert fermion.last_validation["skipped"] is False
    assert np.max(fermion.last_coefficients) > 1.0e-3
    np.testing.assert_allclose(
        projected,
        fermion.kernel @ fermion.last_coefficients + fermion.last_validation["c0"],
        atol=1.0e-12,
    )
    # the refit reference reproduces the input (loosely; c0 + ill-conditioning)
    np.testing.assert_allclose(projected, target, atol=1.0e-4)


def test_causal_fermion_diagnostics_reset_between_calls():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)

    fermion = _fermion(dlr)
    fermion.project(verifier.reconstruct(_bad_coefficients(verifier)))
    assert fermion.last_validation["valid"] is True
    assert fermion.last_coefficients is not None
    first_coeff = fermion.last_coefficients.copy()

    # a second project on a causal channel resets and overwrites the
    # diagnostics; they never describe the earlier call
    causal_coeff = _causal_coefficients(verifier)
    tail = _causal_tail_coeffs(verifier, causal_coeff)
    fermion.project(verifier.reconstruct(causal_coeff), tail_coeffs=tail)
    assert fermion.last_validation["skipped"] is True
    assert fermion.last_status == "skipped"
    assert not np.array_equal(fermion.last_coefficients, first_coeff)


def test_causal_fermion_normal_equation_diagnostic_is_small():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    target = verifier.reconstruct(_bad_coefficients(verifier))

    fermion = _fermion(dlr)
    fermion.project(target)

    # subtracting the spurious tail c0 lifts the node residual off machine
    # zero (measured ~8e-6 on the coarse DLR grid); the normal-equation
    # diagnostic of the decaying part stays tight
    assert fermion.last_validation["node_residual"] <= 1.0e-4
    assert fermion.last_validation["normal_eq_residual"] <= 1.0e-6


def test_causal_fermion_self_energy_preserves_sigma_inf():
    # a self-energy Sigma(iw) = decaying + Sigma_inf: the constant c0 must be
    # preserved (re-added) by the projection, not discarded
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    causal_coeff = _causal_coefficients(verifier)
    sigma_inf = 1.7
    target = verifier.reconstruct(causal_coeff) + sigma_inf

    # fit_tol loosened: subtracting c0 lifts the decaying-part residual above
    # the default 1e-6 on this coarse grid
    fermion = _fermion(dlr, fit_tol=1.0e-3)
    projected = fermion.project(target)

    c0 = fermion.last_validation["c0"]
    assert c0 == pytest.approx(sigma_inf, abs=1.0e-3)
    # the high-frequency limit of the output tends to Sigma_inf
    far = int(np.argmax(np.abs(verifier.omega)))
    assert projected[far].real == pytest.approx(sigma_inf, abs=5.0e-3)
    # causality is enforced on the decaying part
    assert np.max(fermion.last_coefficients) <= 1.0e-5
    np.testing.assert_allclose(
        projected, fermion.kernel @ fermion.last_coefficients + c0, atol=1.0e-12
    )


def _noisy_uniform_g0(crystal, dlr, *, seed=7, amp=1.0e-4):
    """A causal G0 sampled on the uniform grid with low/mid-frequency noise.

    Mirrors the TestCausalFermion notebook: build G0, evaluate on the full
    uniform Matsubara grid, then add noise only where ``|nu| < cutoff*2/3`` so
    the high-frequency tail (used for moment estimation) stays clean.  This is
    the realistic scenario where a fixed 1e-8 tolerance is unattainable.
    """
    from QAssemble.FLatDyn import G0

    hamtb = np.zeros((1, 1, 1, crystal.nk), dtype=np.complex128, order="F")
    kx, ky, kz = crystal.kpoint[:, 0], crystal.kpoint[:, 1], crystal.kpoint[:, 2]
    hamtb[0, 0, 0, :] = -2.0 * (
        np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky) + np.cos(2 * np.pi * kz)
    )
    g0 = G0(crystal=crystal, dlr=dlr, hamtb=hamtb, hdf5file=None)

    z = dlr.MatsubaraFermionUniformFull() * 1j
    nf = len(z)
    g0_kf = np.zeros((1, 1, 1, crystal.nk, nf), dtype=np.complex128, order="F")
    for i in range(crystal.nk):
        fxx = dlr.dF.dlr_from_matsubara(g0.kf[0, 0, 0, i, :], beta=dlr.beta)
        g0_kf[0, 0, 0, i, :] = dlr.dF.eval_dlr_freq(
            fxx[:, None, None], z, beta=dlr.beta, xi=-1
        )[:, 0, 0]

    rng = np.random.default_rng(seed)
    noisy = g0_kf.copy(order="F")
    low = [i for i in range(nf) if abs(z[i].imag) < dlr.cutoff * 2 / 3]
    for i in low:
        noisy[..., i] += rng.uniform(-amp, amp) + 1j * rng.uniform(-amp, amp)
    return g0, noisy


def test_causal_fermion_auto_tol_accepts_noisy_uniform_data():
    # noisy uniform G0 cannot satisfy a fixed 1e-8 tolerance through the
    # uniform->DLR pipeline; constraint_tol="auto" sets the acceptance
    # tolerance to the per-channel noise floor so the projection succeeds.
    # cutoff=50 gives a well-resolved grid (the notebook regime); the coarse
    # default-fixture grid is too sparse for the noisy conversion to be feasible.
    crystal, _, _ = _single_band_hubbard()
    dlr = DLR({"beta": 20.0, "cutoff": 50.0, "eps": 1.0e-12})
    g0, noisy = _noisy_uniform_g0(crystal, dlr)

    # a fixed strict tolerance rejects at least one noisy channel
    with pytest.raises(RuntimeError):
        g0.CausalProjection(noisy, grid="uniform", constraint_tol=1.0e-8)

    # auto accepts the whole grid and returns a finite result on the DLR grid
    out = g0.CausalProjection(noisy, grid="uniform", constraint_tol="auto")
    assert out.shape == (1, 1, 1, crystal.nk, len(dlr.omega))
    assert np.all(np.isfinite(out))


def test_causal_fermion_auto_tol_safety_and_floor():
    # the auto path is exercised at the per-projector level: a noisy channel is
    # accepted with safety=10 and rejected with a tiny safety, and clean data
    # stays at the floor.  Also confirms auto never masks a strong violation.
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)

    # clean causal data: effective tolerance stays at the auto_floor, projection
    # is valid (no spurious loosening from genuine signal)
    causal = verifier.reconstruct(_causal_coefficients(verifier))
    auto_clean = _fermion(dlr, constraint_tol="auto", auto_floor=1.0e-8)
    auto_clean.project(causal)
    assert auto_clean.last_validation["auto_tol"] is True
    assert auto_clean.last_validation["valid"] is True

    # auto still flags a genuinely strong non-causality (not masked by the
    # widened tolerance): a large positive pole weight is far above the floor
    bad_strong = _causal_coefficients(verifier).copy()
    bad_strong[5] = 5.0
    strong = verifier.reconstruct(bad_strong)
    auto = _fermion(dlr, constraint_tol="auto")
    verdict = auto.check(strong, enforce_gate=False)
    assert not verdict.causal


def test_fermion_tail_coefficients_matches_internal_and_injection():
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    target = verifier.reconstruct(_bad_coefficients(verifier)) + 0.9

    fermion = _fermion(dlr, fit_tol=1.0e-3)
    scale = float(max(np.max(np.abs(target)), 1.0))

    # the projector's internal _tail_coefficients agrees with the module-level
    # fermion_tail_coefficients on the same grid (both in scaled units)
    internal = fermion._tail_coefficients(
        target / scale, tail_coeffs=None, scale=scale
    )
    external = fermion_tail_coefficients(verifier.omega, target / scale)
    np.testing.assert_allclose(internal, external, atol=1.0e-12)

    # injecting the unscaled tail reproduces the internal-estimate path exactly
    tail_unscaled = fermion_tail_coefficients(verifier.omega, target)
    out_internal = fermion.project(target)
    out_injected = _fermion(dlr, fit_tol=1.0e-3).project(
        target, tail_coeffs=tail_unscaled
    )
    np.testing.assert_allclose(out_internal, out_injected, atol=1.0e-12)


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
        tail = fermion_tail_coefficients(verifier.omega, original[0, 0, 0, ik, :])
        _assert_projected_channel(
            verifier,
            projected[0, 0, 0, ik, :],
            tail[1:],
            c0=tail[0],
        )

    ik_plot = 0
    c0_plot = fermion_tail_coefficients(
        verifier.omega, original[0, 0, 0, ik_plot, :]
    )[0]
    _show_projection_comparison(
        dlr,
        verifier,
        original[0, 0, 0, ik_plot, :],
        projected[0, 0, 0, ik_plot, :],
        coeffs[ik_plot],
        verifier.fit(projected[0, 0, 0, ik_plot, :] - c0_plot),
        "FLatDyn single-band Hubbard k=0",
    )


def test_flocdyn_causal_projection_preserves_data_tail_moments():
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

    # the preserved moments are the DATA tail [c1, c2, c3] of the input
    tail = fermion_tail_coefficients(verifier.omega, local4[0, 0, 0, :])
    _assert_projected_channel(
        verifier, projected4[0, 0, 0, :], tail[1:], c0=tail[0]
    )
    _show_projection_comparison(
        dlr,
        verifier,
        local4[0, 0, 0, :],
        projected4[0, 0, 0, :],
        bad,
        verifier.fit(projected4[0, 0, 0, :] - tail[0]),
        "FLocDyn single-band Hubbard local channel",
    )


def test_fermion_loc_lat_moment_matches_physical_tail():
    crystal, dlr, _ = _single_band_hubbard()
    local = FLocDyn(crystal, dlr, projector=None)
    flat = FLatDyn(crystal, dlr)
    coeff = np.array([1.3, -0.8, 0.25, 1.1])
    values = _tail_model(dlr.omega, coeff)

    local_values = np.zeros((1, 1, 1, len(dlr.omega)), dtype=np.complex128, order="F")
    local_values[0, 0, 0, :] = values
    moment, high = local.Moment(local_values)
    np.testing.assert_allclose(high[0, 0, 0], coeff[0], atol=1.0e-12)
    np.testing.assert_allclose(moment[0, 0, 0, :], coeff[1:], atol=1.0e-12)
    assert moment.shape == (1, 1, 1, 3)
    assert high.shape == (1, 1, 1)

    lat_values = np.zeros(
        (1, 1, 1, crystal.nk, len(dlr.omega)),
        dtype=np.complex128,
        order="F",
    )
    lat_values[0, 0, 0, :, :] = values
    moment_lat, high_lat = flat.Moment(lat_values, isgreen=False, highzero=False)
    np.testing.assert_allclose(high_lat[0, 0, 0, :], coeff[0], atol=1.0e-12)
    np.testing.assert_allclose(
        moment_lat[0, 0, 0, :, :],
        np.tile(coeff[1:], (crystal.nk, 1)),
        atol=1.0e-12,
    )
    assert moment_lat.shape == (1, 1, 1, crystal.nk, 3)
    assert high_lat.shape == (1, 1, 1, crystal.nk)


def test_fermion_moment_fits_offdiagonal_matrix_tail():
    crystal = _two_orbital_crystal()
    dlr = DLR({"beta": 20.0, "cutoff": 8.0, "eps": 1.0e-12})
    norb = len(crystal.find)
    coeff = _hermitian_tail_coefficients(norb)
    values = _matrix_tail_model(dlr.omega, coeff)

    local = FLocDyn(crystal, dlr, projector=None)
    local_values = np.zeros((norb, norb, 1, len(dlr.omega)), dtype=np.complex128, order="F")
    local_values[:, :, 0, :] = values
    moment, high = local.Moment(local_values)
    expected_moment = np.moveaxis(coeff[1:], 0, -1)[:, :, np.newaxis, :]
    expected_high = coeff[0][:, :, np.newaxis]
    np.testing.assert_allclose(high, expected_high, atol=1.0e-11)
    np.testing.assert_allclose(moment, expected_moment, atol=1.0e-11)
    assert abs(moment[0, 1, 0, 0]) > 0.0
    np.testing.assert_allclose(moment[:, :, 0, 0], moment[:, :, 0, 0].T.conj())

    flat = FLatDyn(crystal, dlr)
    lat_values = np.zeros(
        (norb, norb, 1, crystal.nk, len(dlr.omega)),
        dtype=np.complex128,
        order="F",
    )
    lat_values[:, :, 0, :, :] = values[:, :, np.newaxis, :]
    moment_lat, high_lat = flat.Moment(lat_values)
    np.testing.assert_allclose(
        high_lat,
        np.repeat(expected_high[:, :, :, np.newaxis], crystal.nk, axis=3),
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        moment_lat,
        np.repeat(expected_moment[:, :, :, np.newaxis, :], crystal.nk, axis=3),
        atol=1.0e-11,
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
    # representable targets, but subtracting the spurious tail c0 lifts the
    # node residual off machine zero (measured ~1e-5 on the coarse DLR grid)
    assert np.all(report["node_residual"] <= 1.0e-4)

    # the projection is causal — verified directly (CausalityCheck cannot
    # re-verify projector output on the coarse DLR grid: recomputing c0 from
    # the output's tail plus the ill-conditioned kernel blows up the reference
    # fit, see _assert_projected_channel for the stable check)
    projected = flat.CausalProjection(mat, constraint_tol=1.0e-7)
    verifier = _Verifier(dlr)
    for ik in range(crystal.nk):
        tail = fermion_tail_coefficients(verifier.omega, mat[0, 0, 0, ik, :])
        _assert_projected_channel(
            verifier, projected[0, 0, 0, ik, :], tail[1:], c0=tail[0]
        )


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
    # node_residual is the reliable data-quality diagnostic on the coarse DLR
    # grid: the representable channel sits near machine zero (lifted only by
    # the spurious tail-c0 subtraction), the noisy channel does not.  The
    # `causal` verdict itself is NOT reliable here — the spurious c0 plus the
    # ill-conditioned kernel blow up the unconstrained reference fit even for a
    # genuinely causal channel — so judge the residual first.
    assert report["node_residual"][0, 0] <= 1.0e-4
    # the non-representable (noisy) channel is reported, never raised
    assert report["node_residual"][1, 0] > 1.0e-4

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


def test_resolve_causal_grid_selects_dlr_or_uniform():
    # the grid selection now lives on FLatDyn/FLocDyn as _ResolveCausalGrid
    crystal, dlr, _ = _single_band_hubbard()
    flat = FLatDyn(crystal, dlr)
    dlr_grid = flat._ResolveCausalGrid("dlr")
    uniform_grid = flat._ResolveCausalGrid("uniform")

    np.testing.assert_allclose(dlr_grid, dlr.omega)
    np.testing.assert_allclose(uniform_grid, dlr.MatsubaraFermionUniformFull())
    # the uniform grid is denser than the sparse DLR sampling grid
    assert uniform_grid.size > dlr_grid.size
    with pytest.raises(ValueError, match="grid must be"):
        flat._ResolveCausalGrid("nonsense")

    # FLocDyn exposes the same resolver
    local = FLocDyn(crystal, dlr, projector=None)
    np.testing.assert_allclose(local._ResolveCausalGrid("dlr"), dlr.omega)


def test_causal_projector_fermion_ignores_bosonic_kwargs():
    # statistic='F' silently ignores reflection_symmetry / tail_tol; even a
    # non-default value must not raise and must produce a working projector.
    _, dlr, _ = _single_band_hubbard()
    verifier = _Verifier(dlr)
    target = verifier.reconstruct(_bad_coefficients(verifier))

    fermion = CausalProjector(
        statistic="F",
        d=dlr.dF,
        beta=dlr.beta,
        omega=dlr.omega,
        constraint_tol=1.0e-7,
        reflection_symmetry=False,   # bosonic-only, must be ignored
        tail_tol=-5.0,               # invalid for boson, but ignored for F
    )
    assert not hasattr(fermion, "reflection_symmetry")
    assert not hasattr(fermion, "tail_tol")
    projected = fermion.project(target)
    assert np.max(fermion.last_coefficients) <= 1.0e-6
    assert projected.shape == target.shape


def test_causal_projection_uniform_grid_matches_dlr_grid():
    """Project the SAME underlying function sampled on the DLR grid and on
    the uniform grid; both must be causal and, since uniform output is now
    returned on the DLR grid, they must agree on that common grid."""
    crystal, dlr, _ = _single_band_hubbard()
    flat = FLatDyn(crystal, dlr)
    local = FLocDyn(crystal, dlr, projector=None)

    # one non-causal channel on the DLR grid
    verifier_dlr = _Verifier(dlr)
    bad = _bad_coefficients(verifier_dlr)
    g_dlr = verifier_dlr.reconstruct(bad)  # (len(dlr.omega),)

    # the SAME function expressed on the full uniform grid
    uniform_omega = dlr.MatsubaraFermionUniformFull()
    verifier_uniform = _Verifier(dlr, omega=uniform_omega)
    g_uniform = verifier_uniform.reconstruct(bad)  # (len(uniform_omega),)

    n_dlr = len(dlr.omega)
    n_uniform = len(uniform_omega)

    # --- FLocDyn: 4D local channel on each grid ---
    loc_dlr = np.zeros((1, 1, 1, n_dlr), dtype=np.complex128, order="F")
    loc_dlr[0, 0, 0, :] = g_dlr
    loc_uniform = np.zeros((1, 1, 1, n_uniform), dtype=np.complex128, order="F")
    loc_uniform[0, 0, 0, :] = g_uniform

    out_dlr = local.CausalProjection(loc_dlr, grid="dlr", constraint_tol=1.0e-7)
    out_uniform = local.CausalProjection(loc_uniform, grid="uniform", constraint_tol=1.0e-7)

    # uniform-grid output is returned ON THE DLR GRID, not the uniform grid
    assert out_dlr.shape == loc_dlr.shape
    assert out_uniform.shape == (1, 1, 1, n_dlr)

    # the DLR-grid output is causal: tail / c0 estimated on the DLR grid
    tail_dlr = fermion_tail_coefficients(verifier_dlr.omega, g_dlr)
    _assert_projected_channel(
        verifier_dlr, out_dlr[0, 0, 0, :], tail_dlr[1:], c0=tail_dlr[0]
    )
    # the uniform-grid output is now on the DLR grid, so verify it with the
    # DLR-grid verifier; its tail / c0 were estimated on the uniform grid
    tail_uniform = fermion_tail_coefficients(verifier_uniform.omega, g_uniform)
    _assert_projected_channel(
        verifier_dlr, out_uniform[0, 0, 0, :], tail_uniform[1:], c0=tail_uniform[0]
    )

    # both outputs live on the DLR grid now; the two projections minimize a
    # grid-dependent weighted norm of a non-unique causal target, so the pole
    # weights differ, but the reconstructed functions must track closely.
    rel = (
        np.linalg.norm(out_uniform[0, 0, 0, :] - out_dlr[0, 0, 0, :])
        / np.linalg.norm(out_dlr[0, 0, 0, :])
    )
    assert rel < 5.0e-2, f"cross-grid reconstruction mismatch: rel={rel:.3e}"

    # both outputs are causal — already asserted via _assert_projected_channel
    # above.  CausalityCheck cannot re-verify projector output on the coarse
    # DLR grid (recomputing c0 from the output's own tail blows up the
    # reference fit), so node_residual is the only reliable diagnostic there.
    assert np.all(local.CausalityCheck(out_dlr, grid="dlr")["node_residual"] <= 1.0e-4)
    assert np.all(
        local.CausalityCheck(out_uniform, grid="dlr")["node_residual"] <= 1.0e-4
    )

    # --- FLatDyn: same channel replicated across k ---
    lat_uniform = np.zeros((1, 1, 1, crystal.nk, n_uniform), dtype=np.complex128, order="F")
    for ik in range(crystal.nk):
        lat_uniform[0, 0, 0, ik, :] = g_uniform
    lat_out = flat.CausalProjection(lat_uniform, grid="uniform", constraint_tol=1.0e-7)
    # uniform input -> DLR-grid output
    assert lat_out.shape == (1, 1, 1, crystal.nk, n_dlr)
    for ik in range(crystal.nk):
        _assert_projected_channel(
            verifier_dlr, lat_out[0, 0, 0, ik, :], tail_uniform[1:], c0=tail_uniform[0]
        )
