import numpy as np
import pytest

from QAssemble.BLatDyn import BLatDyn
from QAssemble.BLocDyn import BLocDyn
from QAssemble.Crystal import Crystal
from QAssemble.utility.Causal import (
    BosonPoleQPProjector,
    CausalProjector,
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
    return CausalProjector(
        statistic="B",
        d=dlr.dB,
        beta=dlr.beta,
        omega=dlr.nu,
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


def test_causal_projector_boson_default_and_legacy_backends():
    dlr = _boson_dlr()

    default = _boson(dlr)
    assert default.boson_backend == "pole_qp"
    assert isinstance(default._backend, BosonPoleQPProjector)

    legacy = _boson(dlr, boson_backend="legacy")
    assert legacy.boson_backend == "legacy"
    assert not hasattr(legacy, "_backend")


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


def test_causal_boson_legacy_backend_keeps_skip_diagnostic():
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    causal_coeff = _causal_coefficients(verifier)
    target = verifier.reconstruct(causal_coeff)
    tail = _causal_tail_coeffs(verifier, causal_coeff)

    legacy = _boson(dlr, boson_backend="legacy")
    assert legacy.check(target, tail_coeffs=tail).causal
    projected = legacy.project(target, tail_coeffs=tail)

    assert legacy.last_validation["skipped"] is True
    assert legacy.last_status == "skipped"
    np.testing.assert_allclose(projected, target, atol=1.0e-8)


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


def test_boson_loc_lat_causal_projection_preserves_shape_order_and_offdiagonal():
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
    np.testing.assert_allclose(local_out[0, 1, 0, 0, :], local_values[0, 1, 0, 0, :])
    np.testing.assert_allclose(local_out[1, 0, 0, 0, :], local_values[1, 0, 0, 0, :])

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


def test_invalid_tail_tol_is_rejected():
    dlr = _boson_dlr()
    with pytest.raises(ValueError, match="tail_tol"):
        _boson(dlr, tail_tol=0.0)
    with pytest.raises(ValueError, match="tail_tol"):
        _boson(dlr, tail_tol=-1.0)
