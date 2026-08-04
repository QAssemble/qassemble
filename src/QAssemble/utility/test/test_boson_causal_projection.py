import warnings

import numpy as np
import pytest

from QAssemble.BLatDyn import BLatDyn
from QAssemble.BLocDyn import BLocDyn
from QAssemble.Crystal import Crystal
from QAssemble.utility.Causal import (
    BosonPoleQPProjector,
    CausalBosonProjector,
    CausalProjector,
    ProjectBosonComponentWithFallback,
)
from QAssemble.utility.DLR import DLR
from QAssemble.utility.Fourier import Fourier


def _boson_dlr():
    return DLR({"beta": 20.0, "cutoff": 8.0, "eps": 1.0e-12})


def _single_band_crystal():
    return Crystal(
        {
            "RVec": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "Basis": [[[0, 0, 0], 1]],
            "NSpin": 1,
            "NElec": 1.0,
            "KGrid": [2, 2, 2],
        }
    )


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


def _boson_tail_model(freq, coeff):
    freq = np.asarray(freq, dtype=float)
    z = 1j * freq
    out = np.full(freq.shape, coeff[0], dtype=np.complex128)
    mask = ~np.isclose(freq, 0.0)
    out[mask] = (
        coeff[0]
        + coeff[1] / z[mask]
        + coeff[2] / z[mask] ** 2
        + coeff[3] / z[mask] ** 3
    )
    return out


def _boson_matrix_tail_model(freq, coeff):
    freq = np.asarray(freq, dtype=float)
    z = 1j * freq
    out = np.repeat(coeff[0][..., np.newaxis], freq.size, axis=-1)
    mask = ~np.isclose(freq, 0.0)
    out[..., mask] = (
        coeff[0][..., np.newaxis]
        + coeff[1][..., np.newaxis] / z[mask]
        + coeff[2][..., np.newaxis] / z[mask] ** 2
        + coeff[3][..., np.newaxis] / z[mask] ** 3
    )
    return out


def _hermitian_tail_coefficients(norb):
    coeff = np.zeros((4, norb, norb), dtype=np.complex128)
    for imom in range(4):
        diag = np.linspace(0.1 + imom, 0.9 + imom, norb)
        coeff[imom] += np.diag(diag)
    coeff[:, 0, 1] = np.array([0.25 - 0.15j, -0.3 + 0.2j, 0.7 + 0.1j, -0.5j])
    coeff[:, 1, 0] = np.conj(coeff[:, 0, 1])
    return coeff


class _BosonVerifier:
    """Independent kernel/moment construction mirroring causal_boson.

    Built directly from ``eval_dlr_freq`` unit vectors (xi=+1) and
    ``get_dlr_frequencies()`` so it does not share code with the projector
    under test.
    """

    def __init__(self, dlr, *, reflection_symmetry=True):
        self.beta = float(dlr.beta)
        self.dB = dlr.dB
        self.x = np.asarray(dlr.dB.get_dlr_frequencies(), dtype=float)
        self.nodes = self.x / self.beta
        self.nu = np.asarray(dlr.nu, dtype=float)
        self.reflection_symmetry = bool(reflection_symmetry)
        if self.reflection_symmetry:
            self.kernel = 0.5 * (
                self.basis(1j * self.nu) + self.basis(-1j * self.nu)
            )
        else:
            self.kernel = self.basis(1j * self.nu)
        bose = np.tanh(0.5 * self.x)
        if self.reflection_symmetry:
            self.moment_rows = (bose * self.nodes).reshape(1, -1)
        else:
            self.moment_rows = np.vstack([bose, bose * self.nodes])

    def basis(self, z):
        z = np.asarray(z, dtype=np.complex128)
        rank = self.x.size
        basis = np.empty((z.size, rank), dtype=np.complex128)
        for j in range(rank):
            unit = np.zeros((rank, 1, 1), dtype=np.complex128)
            unit[j, 0, 0] = 1.0
            basis[:, j] = self.dB.eval_dlr_freq(unit, z, self.beta, xi=1)[:, 0, 0]
        return basis

    def fit(self, values):
        lhs = np.vstack((self.kernel.real, self.kernel.imag))
        rhs = np.concatenate((np.real(values), np.imag(values)))
        coeff, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        return np.asarray(coeff, dtype=float)

    def reconstruct(self, coefficients):
        return self.kernel @ np.asarray(coefficients, dtype=float)


def _boson(dlr, *, constraint_tol=1.0e-7, **kwargs):
    return CausalBosonProjector(
        d=dlr.dB,
        beta=dlr.beta,
        fit_omega=dlr.nu,
        constraint_tol=constraint_tol,
        **kwargs,
    )


def _causal_coefficients(verifier):
    return -0.2 * np.exp(-0.1 * verifier.nodes**2) - 0.05


def _causal_tail_coeffs(verifier, coeff):
    """Analytic physical tail ``[c0, c1, c2, c3]`` of a pure
    decaying bosonic channel ``kernel @ coeff``.

    Such channels have ``c0 = 0``.  The tanh-weighted moment rows give the
    projector's internal tail convention, so the physical tail is its negative:
    ``-moment_rows @ coeff`` is ``[c2]`` for the reflection-symmetrized kernel
    (only the even ``1/(inu)^2`` term survives) or ``[c1, c2]`` for the plain
    kernel.  Injecting these via ``tail_coeffs`` bypasses the 5-point lstsq tail
    fit, which on the coarse, sparse DLR grid is an unreliable estimator of the
    higher moments (mirrors the fermion ``_causal_tail_coeffs`` helper)."""
    mom = np.asarray(verifier.moment_rows @ np.asarray(coeff, float), dtype=float)
    if verifier.reflection_symmetry:
        # moment_rows row is [tanh(x/2)*omega] <-> c2; c1 is unconstrained
        c1, c2 = 0.0, -float(mom[0])
    else:
        c1, c2 = -float(mom[0]), -float(mom[1])
    return np.array([0.0, c1, c2, 0.0], dtype=float)


def _bad_coefficients(verifier, alpha_extra=0.2):
    moment_rows = verifier.moment_rows
    nrows = moment_rows.shape[0]
    _, _, vh = np.linalg.svd(moment_rows)
    null_space = vh[nrows:].T
    base = _causal_coefficients(verifier)
    # pick the null direction with the largest single component so a finite
    # step is guaranteed to flip that coefficient positive
    scores = np.max(np.abs(null_space), axis=0)
    direction = null_space[:, int(np.argmax(scores))]
    if direction[np.argmax(np.abs(direction))] < 0.0:
        direction = -direction
    idx = int(np.argmax(direction))
    alpha = abs(base[idx]) / direction[idx] + alpha_extra
    bad = base + alpha * direction
    assert np.max(bad) > 0.0
    np.testing.assert_allclose(moment_rows @ bad, moment_rows @ base, atol=1.0e-10)
    return bad


def test_moment_rows_use_dimensionless_tanh():
    dlr = _boson_dlr()
    x = np.asarray(dlr.dB.get_dlr_frequencies(), dtype=float)
    bose = np.tanh(0.5 * x)
    omega_l = x / dlr.beta

    reflected = _boson(dlr)
    assert reflected.reflection_symmetry is True  # causal_boson default
    assert reflected.moment_rows.shape == (1, x.size)
    np.testing.assert_allclose(reflected.moment_rows[0], bose * omega_l)
    np.testing.assert_allclose(reflected.bose_corr, bose)

    plain = _boson(dlr, reflection_symmetry=False)
    assert plain.moment_rows.shape == (2, x.size)
    np.testing.assert_allclose(plain.moment_rows[0], bose)
    np.testing.assert_allclose(plain.moment_rows[1], bose * omega_l)


def test_reflection_kernel_is_symmetrized():
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr, reflection_symmetry=False)
    nu = np.asarray(dlr.nu, dtype=float)

    reflected = _boson(dlr)
    expected = 0.5 * (verifier.basis(1j * nu) + verifier.basis(-1j * nu))
    np.testing.assert_allclose(reflected.kernel, expected, atol=1.0e-13)

    plain = _boson(dlr, reflection_symmetry=False)
    np.testing.assert_allclose(plain.kernel, verifier.kernel, atol=1.0e-13)


def test_causal_boson_projector_names_and_facade():
    dlr = _boson_dlr()

    direct = _boson(dlr)
    assert isinstance(direct, CausalBosonProjector)
    assert BosonPoleQPProjector is CausalBosonProjector

    facade = CausalProjector(
        statistic="B",
        d=dlr.dB,
        beta=dlr.beta,
        omega=dlr.nu,
        constraint_tol=1.0e-7,
    )
    assert facade.boson_backend == "pole_qp"
    assert isinstance(facade._backend, CausalBosonProjector)

    with pytest.raises(ValueError, match="legacy"):
        CausalProjector(
            statistic="B",
            d=dlr.dB,
            beta=dlr.beta,
            omega=dlr.nu,
            boson_backend="legacy",
        )


@pytest.mark.parametrize("reflection", [True, False])
def test_causal_boson_projects_and_roundtrips(reflection):
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr, reflection_symmetry=reflection)
    bad = _bad_coefficients(verifier)
    target = verifier.reconstruct(bad)
    tail = _causal_tail_coeffs(verifier, bad)  # analytic c0 = 0

    boson = _boson(dlr, reflection_symmetry=reflection)
    assert not boson.check(target, tail_coeffs=tail).causal

    projected = boson.project(target, tail_coeffs=tail)

    assert projected.shape == target.shape
    assert not boson.last_validation["skipped"]
    coeff = boson.last_coefficients
    assert np.max(coeff) <= 1.0e-6
    # the QP equality target is the data tail [c..] of the input, which for a
    # pure decaying channel equals moment_rows @ (its pole weights)
    np.testing.assert_allclose(
        boson.moment_rows @ coeff,
        verifier.moment_rows @ bad,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    # project -> check roundtrip.  With reflection symmetry the kernel is
    # rank-deficient, so check's least-squares representative of the output
    # need not be the causal coefficient vector project found — the causal
    # certificate is last_coefficients itself (asserted above).  Only the
    # full-rank plain kernel guarantees the check roundtrip.
    if not reflection:
        assert boson.check(projected, tail_coeffs=tail).causal

    relative = np.linalg.norm(projected - target) / np.linalg.norm(target)
    assert np.isfinite(relative)
    assert relative < 0.5


def test_boson_tail_coefficients_are_physical_sign():
    nu = np.array([0.0, 1.0, 2.0, 4.0, 7.0, 11.0])
    coeff = np.array([1.4, -0.6, 2.3, -1.1])
    target = _boson_tail_model(nu, coeff)

    tail = Fourier.BosonTailCoefficients(nu, target, tail_points=5)

    np.testing.assert_allclose(tail, coeff, atol=1.0e-12)


@pytest.mark.parametrize("reflection", [True, False])
def test_causal_boson_preserves_causal_input(reflection):
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr, reflection_symmetry=reflection)
    causal_coeff = _causal_coefficients(verifier)
    target = verifier.reconstruct(causal_coeff)
    tail = _causal_tail_coeffs(verifier, causal_coeff)  # analytic c0 = 0

    boson = _boson(dlr, reflection_symmetry=reflection)
    assert boson.check(target, tail_coeffs=tail).causal

    projected = boson.project(target, tail_coeffs=tail)
    assert boson.last_validation["skipped"] is False
    assert boson.last_validation["valid"] is True
    np.testing.assert_allclose(projected, target, atol=1.0e-6)


def test_causal_projector_boson_facade_projects():
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    causal_coeff = _causal_coefficients(verifier)
    target = verifier.reconstruct(causal_coeff)
    tail = _causal_tail_coeffs(verifier, causal_coeff)

    facade = CausalProjector(
        statistic="B",
        d=dlr.dB,
        beta=dlr.beta,
        omega=dlr.nu,
        constraint_tol=1.0e-7,
    )
    assert facade.check(target, tail_coeffs=tail).causal
    projected = facade.project(target, tail_coeffs=tail)

    assert facade.last_validation["valid"] is True
    np.testing.assert_allclose(projected, target, atol=1.0e-6)


def test_static_contamination_is_split_as_c0():
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    coeff = _causal_coefficients(verifier)
    target = verifier.reconstruct(coeff)
    offset = 0.5 * float(np.max(np.abs(target)))
    contaminated = target + offset

    boson = _boson(dlr)
    tail = _causal_tail_coeffs(verifier, coeff)
    tail[0] = offset

    assert boson.check(contaminated, tail_coeffs=tail).causal
    projected = boson.project(contaminated, tail_coeffs=tail)

    assert boson.last_validation["skipped"] is False
    assert boson.last_validation["valid"] is True
    assert boson.last_validation["c0"] == pytest.approx(offset, abs=1.0e-12)
    np.testing.assert_allclose(projected, contaminated, atol=1.0e-6)


def test_static_only_boson_is_split_as_c0():
    dlr = _boson_dlr()
    offset = 1.25
    target = np.full_like(dlr.nu, offset, dtype=np.complex128)
    tail = np.array([offset, 0.0, 0.0, 0.0], dtype=float)

    boson = _boson(dlr)
    assert boson.check(target, tail_coeffs=tail).causal
    projected = boson.project(target, tail_coeffs=tail)

    assert boson.last_validation["skipped"] is False
    assert boson.last_validation["valid"] is True
    assert boson.last_validation["c0"] == pytest.approx(offset, abs=1.0e-12)
    np.testing.assert_allclose(projected, target, atol=1.0e-6)


def test_clean_target_decays_within_tail_guard():
    # fixture sanity: the guard must not reject legitimate dynamic data
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    target = verifier.reconstruct(_causal_coefficients(verifier))

    nu = np.asarray(dlr.nu, dtype=float)
    tail = np.max(np.abs(target[np.argsort(np.abs(nu))[-2:]]))
    assert tail < 0.1 * np.max(np.abs(target))


def test_boson_loc_lat_moment_matches_physical_tail():
    crystal = _single_band_crystal()
    dlr = _boson_dlr()
    coeff = np.array([0.9, -0.4, 1.2, 0.7])
    values = _boson_tail_model(dlr.nu, coeff)

    local = BLocDyn(crystal, dlr, projector=None)
    local_values = np.zeros((1, 1, 1, 1, len(dlr.nu)), dtype=np.complex128, order="F")
    local_values[0, 0, 0, 0, :] = values
    moment, high = local.Moment(local_values)
    np.testing.assert_allclose(high[0, 0, 0, 0], coeff[0], atol=1.0e-12)
    np.testing.assert_allclose(moment[0, 0, 0, 0, :], coeff[1:], atol=1.0e-12)
    assert moment.shape == (1, 1, 1, 1, 3)
    assert high.shape == (1, 1, 1, 1)

    lattice = BLatDyn(crystal, dlr)
    lat_values = np.zeros(
        (1, 1, 1, 1, crystal.nk, len(dlr.nu)),
        dtype=np.complex128,
        order="F",
    )
    lat_values[0, 0, 0, 0, :, :] = values
    moment_lat, high_lat = lattice.Moment(lat_values)
    np.testing.assert_allclose(high_lat[0, 0, 0, 0, :], coeff[0], atol=1.0e-12)
    np.testing.assert_allclose(
        moment_lat[0, 0, 0, 0, :, :],
        np.tile(coeff[1:], (crystal.nk, 1)),
        atol=1.0e-12,
    )
    assert moment_lat.shape == (1, 1, 1, 1, crystal.nk, 3)
    assert high_lat.shape == (1, 1, 1, 1, crystal.nk)


def test_boson_moment_fits_offdiagonal_matrix_tail():
    crystal = _two_orbital_crystal()
    dlr = _boson_dlr()
    norb = len(crystal.bind)
    coeff = _hermitian_tail_coefficients(norb)
    values = _boson_matrix_tail_model(dlr.nu, coeff)

    local = BLocDyn(crystal, dlr, projector=None)
    local_values = np.zeros(
        (norb, norb, 1, 1, len(dlr.nu)),
        dtype=np.complex128,
        order="F",
    )
    local_values[:, :, 0, 0, :] = values
    moment, high = local.Moment(local_values)
    expected_moment = np.moveaxis(coeff[1:], 0, -1)[:, :, np.newaxis, np.newaxis, :]
    expected_high = coeff[0][:, :, np.newaxis, np.newaxis]
    np.testing.assert_allclose(high, expected_high, atol=1.0e-11)
    np.testing.assert_allclose(moment, expected_moment, atol=1.0e-11)
    assert abs(moment[0, 1, 0, 0, 0]) > 0.0
    np.testing.assert_allclose(moment[:, :, 0, 0, 0], moment[:, :, 0, 0, 0].T.conj())

    lattice = BLatDyn(crystal, dlr)
    lat_values = np.zeros(
        (norb, norb, 1, 1, crystal.nk, len(dlr.nu)),
        dtype=np.complex128,
        order="F",
    )
    lat_values[:, :, 0, 0, :, :] = values[:, :, np.newaxis, :]
    moment_lat, high_lat = lattice.Moment(lat_values)
    np.testing.assert_allclose(
        high_lat,
        np.repeat(expected_high[:, :, :, :, np.newaxis], crystal.nk, axis=4),
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        moment_lat,
        np.repeat(expected_moment[:, :, :, :, np.newaxis, :], crystal.nk, axis=4),
        atol=1.0e-11,
    )


def test_boson_loc_projects_offdiagonal_lat_preserves_offdiagonal():
    crystal = _two_orbital_crystal()
    dlr = _boson_dlr()
    norb = len(crystal.bind)
    nfreq = len(dlr.nu)
    offdiag = _boson_tail_model(dlr.nu, np.array([0.2, -0.1, 0.3, -0.05]))

    local = BLocDyn(crystal, dlr, projector=None)
    local_values = np.zeros((norb, norb, 1, 1, nfreq), dtype=np.complex128, order="F")
    local_values[0, 0, 0, 0, :] = 1.0
    local_values[1, 1, 0, 0, :] = 0.5
    local_values[0, 1, 0, 0, :] = offdiag
    local_values[1, 0, 0, 0, :] = np.conj(offdiag)

    local_out = local.CausalProjection(local_values)
    assert local_out.shape == local_values.shape
    assert np.isfortran(local_out)
    # Off-diagonal is now projected component-wise (the model is not causal
    # near nu=0, so values there legitimately change); the lower triangle is
    # the conjugate mirror and the constant diagonals are preserved.
    assert not np.allclose(local_out[0, 1, 0, 0, :], local_values[0, 1, 0, 0, :])
    np.testing.assert_allclose(
        local_out[1, 0, 0, 0, :],
        np.conj(local_out[0, 1, 0, 0, :]),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(local_out[0, 0, 0, 0, :], 1.0, atol=1.0e-5)
    np.testing.assert_allclose(local_out[1, 1, 0, 0, :], 0.5, atol=1.0e-5)

    lattice = BLatDyn(crystal, dlr)
    lat_values = np.zeros(
        (norb, norb, 1, 1, crystal.nk, nfreq),
        dtype=np.complex128,
        order="F",
    )
    lat_values[0, 0, 0, 0, :, :] = 1.0
    lat_values[1, 1, 0, 0, :, :] = 0.5
    lat_values[0, 1, 0, 0, :, :] = offdiag
    lat_values[1, 0, 0, 0, :, :] = np.conj(offdiag)

    lat_out = lattice.CausalProjection(lat_values)
    assert lat_out.shape == lat_values.shape
    assert np.isfortran(lat_out)
    np.testing.assert_allclose(lat_out[0, 1, 0, 0, :, :], lat_values[0, 1, 0, 0, :, :])
    np.testing.assert_allclose(lat_out[1, 0, 0, 0, :, :], lat_values[1, 0, 0, 0, :, :])


def test_boson_loc_offdiagonal_roundtrips_real_drops_imaginary_static():
    crystal = _two_orbital_crystal()
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    norb = len(crystal.bind)
    nfreq = len(dlr.nu)

    # Causal matrix model: sign-compatible real pole weights per component plus
    # a Hermitian static matrix with a complex off-diagonal c0.
    base = _causal_coefficients(verifier)
    c0_offdiag = 0.25 - 0.15j
    local_values = np.zeros((norb, norb, 1, 1, nfreq), dtype=np.complex128, order="F")
    local_values[0, 0, 0, 0, :] = verifier.reconstruct(base) + 0.3
    local_values[1, 1, 0, 0, :] = verifier.reconstruct(0.8 * base) + 0.1
    offdiag_dyn = verifier.reconstruct(0.5 * base)
    local_values[0, 1, 0, 0, :] = offdiag_dyn + c0_offdiag
    local_values[1, 0, 0, 0, :] = np.conj(local_values[0, 1, 0, 0, :])

    local = BLocDyn(crystal, dlr, projector=None)
    local_out = local.CausalProjection(local_values)

    # Hermiticity of the output orbital matrix at every frequency.
    np.testing.assert_allclose(
        local_out[1, 0, 0, 0, :],
        np.conj(local_out[0, 1, 0, 0, :]),
        atol=1.0e-13,
    )
    # The X-combination path treats off-diagonals as real-symmetric channels:
    # Im c0 and any dynamic imaginary part are dropped.
    np.testing.assert_allclose(
        np.imag(local_out[0, 1, 0, 0, :]), 0.0, atol=1.0e-10
    )
    # The real part of an already-causal input roundtrips up to the accuracy of
    # the 5-point tail fit on the sparse DLR grid.
    np.testing.assert_allclose(
        np.real(local_out[0, 1, 0, 0, :]),
        offdiag_dyn + np.real(c0_offdiag),
        atol=2.0e-2,
    )


def test_boson_loc_offdiagonal_x_identity_on_uniform_grid():
    crystal = _two_orbital_crystal()
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    norb = len(crystal.bind)
    nu_uniform = np.asarray(dlr.MatsubaraBosonUniform(), dtype=float)
    nfreq = len(nu_uniform)
    base = _causal_coefficients(verifier)
    kernel_uniform = 0.5 * (
        verifier.basis(1j * nu_uniform) + verifier.basis(-1j * nu_uniform)
    )
    dyn0 = kernel_uniform @ base
    dyn1 = kernel_uniform @ (0.8 * base)
    dyn01 = kernel_uniform @ (0.3 * base)

    local_values = np.zeros((norb, norb, 1, 1, nfreq), dtype=np.complex128, order="F")
    local_values[0, 0, 0, 0, :] = dyn0 + 0.3
    local_values[1, 1, 0, 0, :] = dyn1 + 0.1
    local_values[0, 1, 0, 0, :] = dyn01 + 0.05
    local_values[1, 0, 0, 0, :] = np.conj(local_values[0, 1, 0, 0, :])

    local = BLocDyn(crystal, dlr, projector=None)
    out = local.CausalProjection(local_values, grid="uniform")

    xarr = np.zeros_like(local_values, dtype=np.complex128, order="F")
    xarr[0, 1, 0, 0, :] = (
        local_values[0, 0, 0, 0, :]
        + local_values[0, 1, 0, 0, :]
        + local_values[1, 0, 0, 0, :]
        + local_values[1, 1, 0, 0, :]
    )
    xarr[1, 0, 0, 0, :] = xarr[0, 1, 0, 0, :]
    xmoment, xhigh, xsigma = local.Moment(
        xarr, grid="uniform", return_sigma=True
    )
    x_dlr = dlr.MatsubaraUniformGrid2DLR(xarr, omega=nu_uniform, sign=1)
    c0x = complex(xhigh[0, 1, 0, 0])
    tail = np.empty(4, dtype=float)
    tail[0] = 0.0
    tail[1:] = np.real(xmoment[0, 1, 0, 0, :])
    projector = CausalBosonProjector(
        d=dlr.dB,
        beta=dlr.beta,
        fit_omega=dlr.nu,
        coefficient_sign=-1,
        reflection_symmetry=True,
        max_iter=100000,
        constraint_tol=1.0e-8,
        fit_tol=1.0e-6,
        tail_tol=1.0e-1,
        raise_on_failure=True,
    )
    x_projected = ProjectBosonComponentWithFallback(
        projector,
        x_dlr[0, 1, 0, 0, :] - c0x,
        tail,
        tail_sigma=xsigma[0, 1, 0, 0, :],
    ) + c0x

    np.testing.assert_allclose(
        2.0 * out[0, 1, 0, 0, :] + out[0, 0, 0, 0, :] + out[1, 1, 0, 0, :],
        x_projected,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        out[1, 0, 0, 0, :], np.conj(out[0, 1, 0, 0, :]), atol=1.0e-13
    )


def test_boson_uniform_expansion_uses_transposed_conjugate():
    dlr = _boson_dlr()
    nu_uniform = dlr.MatsubaraBosonUniform()
    nu_dlr = np.asarray(dlr.nu, dtype=float)

    poles = np.array([0.7, -0.4])
    weights = np.zeros((2, 2, poles.size), dtype=np.complex128)
    weights[0, 0] = [-0.3, -0.2]
    weights[1, 1] = [-0.25, -0.15]
    weights[0, 1] = [0.1 + 0.05j, -0.08 + 0.02j]
    weights[1, 0] = np.conj(weights[0, 1])

    def model(freq):
        z = 1j * np.asarray(freq, dtype=float)
        out = np.zeros((2, 2, 1, 1, z.size), dtype=np.complex128)
        for il, pole in enumerate(poles):
            out[:, :, 0, 0, :] += weights[:, :, il, np.newaxis] / (z - pole)
        return out

    # Pole model with a Hermitian weight matrix satisfies
    # F_ij(-nu) = conj(F_ji(+nu)) analytically, so expanding the non-negative
    # grid must reproduce the direct evaluation on the signed DLR grid.
    converted = dlr.MatsubaraUniformGrid2DLR(
        model(nu_uniform), omega=nu_uniform, sign=1
    )
    np.testing.assert_allclose(converted, model(nu_dlr), atol=1.0e-8)


def test_boson_infeasible_moment_hard_raises_elastic_succeeds():
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    base = _causal_coefficients(verifier)
    target = verifier.reconstruct(base)

    boson = _boson(dlr)
    # Physical c2 < 0 maps to an internal m2 > 0, which the sign-constrained
    # weights (A_l <= 0) can never satisfy: the hard equality is infeasible.
    infeasible_tail = np.array([0.0, 0.0, -0.5, 0.0])
    # hard path (no sigma): still infeasible -> raises
    with pytest.raises(RuntimeError):
        boson.project(target, tail_coeffs=infeasible_tail)

    # elastic path: the tail sigmas turn the equality into an exact L1
    # penalty, so the QP always finds a causal solution.  A garbage moment
    # estimate comes with a comparably large sigma (that is what the noisy
    # tail fit reports — the data's own m2 is ~-6.5 here, so a fit claiming
    # -0.5 carries an O(5) uncertainty), keeping mu = 1/sigma moderate.
    tail_sigma = np.array([0.0, 0.1, 5.0, 0.0])
    projected = ProjectBosonComponentWithFallback(
        boson, target, infeasible_tail, tail_sigma=tail_sigma
    )
    assert projected.shape == target.shape
    assert np.all(np.isfinite(projected))
    validation = boson.last_validation
    assert validation["elastic"] is True
    assert validation["valid"] is True
    assert validation["clipped"] is False
    # output stays causal (sign -1: all pole weights nonpositive)
    assert validation["coefficient_violation"] <= validation["effective_tol"]
    # the slack records the (unreachable) distance to the causal moment cone:
    # m2_target = +0.5 while any causal solution has m2 <= 0
    assert validation["equality_slack"] is not None
    assert validation["equality_slack"][0] < 0.0
    # the causal data is still tracked far better than any clipped/raw
    # fallback would manage; exact statistical weighting is data-dependent
    relative = np.linalg.norm(projected - target) / np.linalg.norm(target)
    assert relative < 0.5


def test_boson_nondecaying_offset_is_split_and_dropped():
    # BWeiss scenario: the caller's highzero convention pins c0 = 0, but CTQMC
    # noise through the Dyson equation leaves a constant contamination.  The
    # projector must split the non-decaying offset off, warn, and DISCARD it
    # from the output instead of raising the old tail-decay gate.
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    base = _causal_coefficients(verifier)
    clean = verifier.reconstruct(base)
    # the offset must exceed tail_tol (default 0.1) relative to the data
    # magnitude (~13.6 here) for the non-decay detector to fire
    offset = 3.0
    contaminated = clean + offset

    boson = _boson(dlr)
    # honest decaying moments of the clean data, but c0 forced to 0
    m2_internal = float((boson.moment_rows @ base)[0])
    tail = np.array([0.0, 0.0, -m2_internal, 0.0])

    with pytest.warns(RuntimeWarning, match="DISCARDED"):
        out = boson.project(contaminated, tail_coeffs=tail)

    validation = boson.last_validation
    assert validation["tail_ratio"] > boson.tail_tol
    # the dropped offset is reported and matches the injected contamination
    assert abs(validation["c0_dropped"]) == pytest.approx(offset, abs=0.3)
    # the output is purely decaying: with highzero c0 = 0 nothing constant
    # is re-added, so the largest-|nu| values are small versus the maximum
    tail_idx = np.argsort(np.abs(boson.output_omega))[-2:]
    assert np.max(np.abs(out[tail_idx])) < boson.tail_tol * np.max(np.abs(out))
    # and the decaying part still tracks the clean causal data
    relative = np.linalg.norm(out - clean) / np.linalg.norm(clean)
    assert relative < 5.0e-2


def test_boson_loc_spin_offdiagonal_blocks_pass_through():
    dlr = _boson_dlr()
    crystal = Crystal(
        {
            "RVec": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "Basis": [[[0, 0, 0], 1]],
            "NSpin": 2,
            "NElec": 1.0,
            "KGrid": [2, 1, 1],
        }
    )
    nfreq = len(dlr.nu)
    spin_offdiag = _boson_tail_model(dlr.nu, np.array([0.05, 0.02, 0.1, 0.0]))

    local = BLocDyn(crystal, dlr, projector=None)
    values = np.zeros((1, 1, 2, 2, nfreq), dtype=np.complex128, order="F")
    values[0, 0, 0, 0, :] = 1.0
    values[0, 0, 1, 1, :] = 0.5
    values[0, 0, 0, 1, :] = spin_offdiag
    values[0, 0, 1, 0, :] = np.conj(spin_offdiag)

    out = local.CausalProjection(values)
    np.testing.assert_allclose(out[0, 0, 0, 1, :], values[0, 0, 0, 1, :])
    np.testing.assert_allclose(out[0, 0, 1, 0, :], values[0, 0, 1, 0, :])


def test_invalid_tail_tol_is_rejected():
    dlr = _boson_dlr()
    with pytest.raises(ValueError, match="tail_tol"):
        _boson(dlr, tail_tol=0.0)
    with pytest.raises(ValueError, match="tail_tol"):
        _boson(dlr, tail_tol=-1.0)


def test_boson_moment_restricted_fit_recovers_pure_c2():
    crystal = _single_band_crystal()
    dlr = _boson_dlr()
    c2 = 1.7
    values = _boson_tail_model(dlr.nu, np.array([0.0, 0.0, c2, 0.0]))

    local = BLocDyn(crystal, dlr, projector=None)
    local_values = np.zeros((1, 1, 1, 1, len(dlr.nu)), dtype=np.complex128, order="F")
    local_values[0, 0, 0, 0, :] = values
    moment, high = local.Moment(local_values, oddzero=True, highzero=True)

    np.testing.assert_allclose(high[0, 0, 0, 0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(moment[0, 0, 0, 0, 0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(moment[0, 0, 0, 0, 1], c2, atol=1.0e-12)
    np.testing.assert_allclose(moment[0, 0, 0, 0, 2], 0.0, atol=1.0e-14)

    # Single flags restrict only their own columns.
    moment_h, high_h = local.Moment(local_values, highzero=True)
    np.testing.assert_allclose(high_h[0, 0, 0, 0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(moment_h[0, 0, 0, 0, 1], c2, atol=1.0e-10)
    moment_o, high_o = local.Moment(local_values, oddzero=True)
    np.testing.assert_allclose(moment_o[0, 0, 0, 0, 0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(moment_o[0, 0, 0, 0, 2], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(moment_o[0, 0, 0, 0, 1], c2, atol=1.0e-10)


def test_boson_loc_positive_chi_projection_roundtrips():
    # chi convention: non-negative pole weights (coefficient_sign=+1) and a
    # strictly decaying tail (no constant, no odd moments).
    crystal = _single_band_crystal()
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    pos_coeff = -_causal_coefficients(verifier)
    assert np.all(pos_coeff >= 0.0)
    target = verifier.reconstruct(pos_coeff)

    local = BLocDyn(crystal, dlr, projector=None)
    local_values = np.zeros((1, 1, 1, 1, len(dlr.nu)), dtype=np.complex128, order="F")
    local_values[0, 0, 0, 0, :] = target

    out = local.CausalProjection(
        local_values, coefficient_sign=1, oddzero=True, highzero=True
    )

    np.testing.assert_allclose(out[0, 0, 0, 0, :], target, atol=2.0e-2)
    np.testing.assert_allclose(np.imag(out[0, 0, 0, 0, :]), 0.0, atol=1.0e-10)
    assert np.all(np.real(out[0, 0, 0, 0, :]) > 0.0)


def test_boson_loc_highzero_output_is_strictly_decaying():
    # A large constant fails the projector's tail-decay guard under highzero
    # (fallback returns the input), so contaminate with a noise-level offset
    # that stays below the guard and assert the structural property instead:
    # with highzero the output lies exactly in the decaying pole span, while
    # the default path carries the constant, which the span cannot represent.
    crystal = _single_band_crystal()
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    pos_coeff = -_causal_coefficients(verifier)
    target = verifier.reconstruct(pos_coeff)

    tail_idx = np.argsort(np.abs(np.asarray(dlr.nu, dtype=float)))[-2:]
    scale = float(np.max(np.abs(target)))
    tail_frac = float(np.max(np.abs(target[tail_idx]))) / scale
    offset = (0.08 - tail_frac) * scale
    assert offset > 0.0
    contaminated = target + offset

    local = BLocDyn(crystal, dlr, projector=None)
    local_values = np.zeros((1, 1, 1, 1, len(dlr.nu)), dtype=np.complex128, order="F")
    local_values[0, 0, 0, 0, :] = contaminated

    def pole_residual(values):
        coeff = verifier.fit(values)
        return float(np.linalg.norm(values - verifier.reconstruct(coeff)))

    # Default behavior splits and preserves the constant at the tail; the
    # resulting channel is NOT representable by decaying poles alone.
    kept = local.CausalProjection(local_values, coefficient_sign=1)
    np.testing.assert_allclose(
        np.real(kept[0, 0, 0, 0, tail_idx]), offset, atol=0.1 * offset
    )
    kept_residual = pole_residual(kept[0, 0, 0, 0, :])

    # highzero skips the c0 split entirely: the output is a pure pole sum,
    # hence strictly decaying beyond the grid.
    cleaned = local.CausalProjection(
        local_values, coefficient_sign=1, oddzero=True, highzero=True
    )
    cleaned_residual = pole_residual(cleaned[0, 0, 0, 0, :])
    assert cleaned_residual < 1.0e-6 * np.linalg.norm(cleaned)
    assert kept_residual > 100.0 * max(cleaned_residual, 1.0e-12)


def test_bweiss_like_projection_handles_nondecaying_contamination():
    # BWeiss.Cal path end-to-end: 5D cf on the uniform grid with
    # oddzero+highzero, coefficient_sign=-1 and constraint_tol="auto".  A
    # non-decaying contamination (CTQMC noise through the Dyson equation)
    # must be split off and DISCARDED with a warning, and every output
    # channel must be a causal decaying pole sum — the raw non-causal data
    # never passes through.
    crystal = _single_band_crystal()
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    base = _causal_coefficients(verifier)
    nu_uniform = np.asarray(dlr.MatsubaraBosonUniform(), dtype=float)
    kernel_uniform = 0.5 * (
        verifier.basis(1j * nu_uniform) + verifier.basis(-1j * nu_uniform)
    )
    clean_uniform = kernel_uniform @ base
    offset = 0.2 * float(np.max(np.abs(clean_uniform)))
    contaminated = clean_uniform + offset

    local = BLocDyn(crystal, dlr, projector=None)
    vals = np.zeros((1, 1, 1, 1, len(nu_uniform)), dtype=np.complex128, order="F")
    vals[0, 0, 0, 0, :] = contaminated

    with pytest.warns(RuntimeWarning, match="DISCARDED"):
        out = local.CausalProjection(
            vals,
            grid="uniform",
            coefficient_sign=-1,
            oddzero=True,
            highzero=True,
            constraint_tol="auto",
        )

    assert out.shape == (1, 1, 1, 1, len(dlr.nu))
    assert np.all(np.isfinite(out))
    # highzero: the output is a pure decaying pole sum — the discarded offset
    # never re-enters.  (Sign causality of the weights is covered by the
    # projector unit tests; the reflection-symmetric kernel is rank-deficient
    # across +-omega pairs, so a naive refit cannot resolve weight signs.)
    channel = out[0, 0, 0, 0, :]
    coeff = verifier.fit(channel)
    span_residual = float(
        np.linalg.norm(channel - verifier.reconstruct(coeff))
    )
    assert span_residual < 1.0e-6 * max(float(np.linalg.norm(channel)), 1.0)
    # the contamination is gone from the tail: largest-|nu| values are small
    tail_idx = np.argsort(np.abs(np.asarray(dlr.nu, dtype=float)))[-2:]
    assert np.max(np.abs(channel[tail_idx])) < 0.1 * np.max(np.abs(channel))
    # the projected channel tracks the clean data decisively better than the
    # unprojected contamination would (the strong offset also biases the
    # restricted c2 fit, so exact recovery is not attainable here — the
    # production-accuracy claim is validated on real glob.h5 data instead)
    clean_dlr = verifier.reconstruct(base)
    relative = np.linalg.norm(channel - clean_dlr) / np.linalg.norm(clean_dlr)
    unprojected = np.linalg.norm(
        (clean_dlr + offset) - clean_dlr
    ) / np.linalg.norm(clean_dlr)
    assert relative < 0.6 * unprojected


def test_boson_loc_positive_chi_offdiag_x_path_preserves_input():
    # The off-diagonal component alone has the wrong sign for A_l >= 0, but
    # the X combination has non-negative weight and can be projected directly.
    crystal = _two_orbital_crystal()
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    norb = len(crystal.bind)
    nfreq = len(dlr.nu)
    pos_coeff = -_causal_coefficients(verifier)

    local_values = np.zeros((norb, norb, 1, 1, nfreq), dtype=np.complex128, order="F")
    local_values[0, 0, 0, 0, :] = verifier.reconstruct(pos_coeff)
    local_values[1, 1, 0, 0, :] = verifier.reconstruct(0.8 * pos_coeff)
    # Negative weights give physical c2 > 0, infeasible under A_l >= 0.
    local_values[0, 1, 0, 0, :] = verifier.reconstruct(-0.5 * pos_coeff)
    local_values[1, 0, 0, 0, :] = np.conj(local_values[0, 1, 0, 0, :])

    local = BLocDyn(crystal, dlr, projector=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = local.CausalProjection(
            local_values, coefficient_sign=1, oddzero=True, highzero=True
        )

    runtime_warnings = [
        warn for warn in caught if issubclass(warn.category, RuntimeWarning)
    ]
    assert runtime_warnings == []
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(
        out[1, 0, 0, 0, :], np.conj(out[0, 1, 0, 0, :]), atol=1.0e-13
    )
    np.testing.assert_allclose(
        out[0, 0, 0, 0, :], local_values[0, 0, 0, 0, :], atol=2.0e-2
    )
    np.testing.assert_allclose(
        out[1, 1, 0, 0, :], local_values[1, 1, 0, 0, :], atol=2.0e-2
    )
    np.testing.assert_allclose(
        out[0, 1, 0, 0, :], local_values[0, 1, 0, 0, :], atol=2.0e-2
    )


def test_boson_loc_offdiagonal_fallback_uses_x_combination(monkeypatch):
    crystal = _two_orbital_crystal()
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    norb = len(crystal.bind)
    nfreq = len(dlr.nu)
    base = _causal_coefficients(verifier)

    values = np.zeros((norb, norb, 1, 1, nfreq), dtype=np.complex128, order="F")
    values[0, 0, 0, 0, :] = verifier.reconstruct(base) + 0.2
    values[1, 1, 0, 0, :] = verifier.reconstruct(0.8 * base) + 0.1
    values[0, 1, 0, 0, :] = verifier.reconstruct(0.2 * base) + 0.05
    values[1, 0, 0, 0, :] = np.conj(values[0, 1, 0, 0, :])

    fallback = np.zeros_like(values)
    fallback[0, 0, 0, 0, :] = verifier.reconstruct(0.4 * base) + 0.7
    fallback[1, 1, 0, 0, :] = verifier.reconstruct(0.6 * base) + 0.4
    fallback[0, 1, 0, 0, :] = verifier.reconstruct(0.1 * base) + (0.25 - 0.3j)
    fallback[1, 0, 0, 0, :] = np.conj(fallback[0, 1, 0, 0, :])

    monkeypatch.setattr(CausalBosonProjector, "project", _raise_runtime)
    local = BLocDyn(crystal, dlr, projector=None)
    with pytest.warns(RuntimeWarning, match="previous iteration"):
        out = local.CausalProjection(values, fallback_matrix=fallback)

    expected_offdiag = 0.5 * (
        fallback[0, 0, 0, 0, :]
        + fallback[0, 1, 0, 0, :]
        + fallback[1, 0, 0, 0, :]
        + fallback[1, 1, 0, 0, :]
        - fallback[0, 0, 0, 0, :]
        - fallback[1, 1, 0, 0, :]
    )
    np.testing.assert_allclose(out[0, 0, 0, 0, :], fallback[0, 0, 0, 0, :])
    np.testing.assert_allclose(out[1, 1, 0, 0, :], fallback[1, 1, 0, 0, :])
    np.testing.assert_allclose(out[0, 1, 0, 0, :], np.real(expected_offdiag))
    np.testing.assert_allclose(out[1, 0, 0, 0, :], np.conj(out[0, 1, 0, 0, :]))


def _raise_runtime(*args, **kwargs):
    raise RuntimeError("forced QP failure")


def test_component_fallback_prefers_previous_iteration_channel(monkeypatch):
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    projector = _boson(dlr, raise_on_failure=True)
    monkeypatch.setattr(projector, "project", _raise_runtime)

    target = verifier.reconstruct(_causal_coefficients(verifier))
    previous = verifier.reconstruct(0.5 * _causal_coefficients(verifier))

    with pytest.warns(RuntimeWarning, match="previous iteration"):
        out = ProjectBosonComponentWithFallback(
            projector,
            target,
            np.zeros(4),
            fallback_channel=previous,
        )
    np.testing.assert_allclose(out, previous)
    assert out is not previous


def test_component_fallback_clipped_when_no_previous(monkeypatch):
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    projector = _boson(dlr, raise_on_failure=True)
    monkeypatch.setattr(projector, "project", _raise_runtime)

    # Causal (negative-weight) target: the clipped regularized fit keeps most
    # of the weight, so the clipped fallback is the right answer.
    target = verifier.reconstruct(_causal_coefficients(verifier))
    with pytest.warns(RuntimeWarning, match="clipped fallback"):
        out = ProjectBosonComponentWithFallback(projector, target, np.zeros(4))
    assert np.linalg.norm(out) > 1.0e-3 * np.linalg.norm(target)
    assert not np.allclose(out, target)


def test_component_fallback_degenerate_clipping_returns_unprojected(monkeypatch):
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    projector = _boson(dlr, raise_on_failure=True)
    monkeypatch.setattr(projector, "project", _raise_runtime)

    # Positive-weight target under the A_l <= 0 constraint: clipping removes
    # every coefficient and the clipped output collapses to ~0.  The guard
    # must return the unprojected channel instead of zeros (the dyn.json
    # zero-output regression).
    target = verifier.reconstruct(-_causal_coefficients(verifier))
    with pytest.warns(RuntimeWarning, match="UNPROJECTED"):
        out = ProjectBosonComponentWithFallback(projector, target, np.zeros(4))
    np.testing.assert_allclose(out, target)


def test_causal_projection_uses_fallback_matrix_on_component_failure(monkeypatch):
    crystal = _single_band_crystal()
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    nfreq = len(dlr.nu)

    values = np.zeros((1, 1, 1, 1, nfreq), dtype=np.complex128, order="F")
    values[0, 0, 0, 0, :] = verifier.reconstruct(_causal_coefficients(verifier))
    fallback = np.zeros_like(values)
    fallback[0, 0, 0, 0, :] = verifier.reconstruct(
        0.3 * _causal_coefficients(verifier)
    ) + 0.7

    monkeypatch.setattr(CausalBosonProjector, "project", _raise_runtime)
    local = BLocDyn(crystal, dlr, projector=None)
    with pytest.warns(RuntimeWarning, match="previous iteration"):
        out = local.CausalProjection(values, fallback_matrix=fallback)

    # The c0 subtracted from the fallback channel is re-added afterwards, so
    # the full previous-iteration channel round-trips exactly.
    np.testing.assert_allclose(out, fallback)


def test_causal_projection_ignores_fallback_matrix_on_success():
    crystal = _single_band_crystal()
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    nfreq = len(dlr.nu)

    values = np.zeros((1, 1, 1, 1, nfreq), dtype=np.complex128, order="F")
    values[0, 0, 0, 0, :] = verifier.reconstruct(_causal_coefficients(verifier))
    fallback = np.full_like(values, 123.0)

    local = BLocDyn(crystal, dlr, projector=None)
    out = local.CausalProjection(values, fallback_matrix=fallback)

    assert not np.allclose(out, fallback)
    np.testing.assert_allclose(out, values, atol=2.0e-2)


def test_causal_projection_rejects_fallback_matrix_shape_mismatch():
    crystal = _single_band_crystal()
    dlr = _boson_dlr()
    nfreq = len(dlr.nu)

    values = np.zeros((1, 1, 1, 1, nfreq), dtype=np.complex128, order="F")
    bad = np.zeros((1, 1, 1, 1, nfreq + 1), dtype=np.complex128, order="F")

    local = BLocDyn(crystal, dlr, projector=None)
    with pytest.raises(ValueError, match="fallback_matrix shape"):
        local.CausalProjection(values, fallback_matrix=bad)
