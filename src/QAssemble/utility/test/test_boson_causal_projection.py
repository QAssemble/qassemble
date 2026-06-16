import numpy as np
import pytest

from QAssemble.utility.Causal import CausalProjector
from QAssemble.utility.DLR import DLR

# Bosonic causal projection is temporarily disabled: the projector now derives
# the QP equality target from the high-frequency data tail (fermion-only so
# far).  The bosonic tail needs the BLocDynM convention and tanh(x/2)-weighted
# moment rows, which are deferred to a follow-up.  project()/check() raise
# NotImplementedError for statistic='B' until then.
_BOSON_TAIL_DEFERRED = pytest.mark.xfail(
    reason="bosonic tail-based moments not implemented yet (fermion-only)",
    raises=NotImplementedError,
    strict=True,
)


def _boson_dlr():
    return DLR({"beta": 20.0, "cutoff": 8.0, "eps": 1.0e-12})


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


@_BOSON_TAIL_DEFERRED
@pytest.mark.parametrize("reflection", [True, False])
def test_causal_boson_projects_and_roundtrips(reflection):
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr, reflection_symmetry=reflection)
    bad = _bad_coefficients(verifier)
    target = verifier.reconstruct(bad)

    boson = _boson(dlr, reflection_symmetry=reflection)
    assert not boson.check(target).causal

    projected = boson.project(target)

    assert projected.shape == target.shape
    assert not boson.last_validation["skipped"]
    coeff = boson.last_coefficients
    assert np.max(coeff) <= 1.0e-6
    # the internally anchored moments equal those of the fit of the input
    np.testing.assert_allclose(
        boson.moment_rows @ coeff,
        verifier.moment_rows @ verifier.fit(target),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    # project -> check roundtrip.  With reflection symmetry the kernel is
    # rank-deficient, so check's least-squares representative of the output
    # need not be the causal coefficient vector project found — the causal
    # certificate is last_coefficients itself (asserted above).  Only the
    # full-rank plain kernel guarantees the check roundtrip.
    if not reflection:
        assert boson.check(projected).causal

    relative = np.linalg.norm(projected - target) / np.linalg.norm(target)
    assert np.isfinite(relative)
    assert relative < 0.5


@_BOSON_TAIL_DEFERRED
@pytest.mark.parametrize("reflection", [True, False])
def test_causal_boson_skips_qp_for_causal_input(reflection):
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr, reflection_symmetry=reflection)
    target = verifier.reconstruct(_causal_coefficients(verifier))

    boson = _boson(dlr, reflection_symmetry=reflection)
    assert boson.check(target).causal

    projected = boson.project(target)
    assert boson.last_validation["skipped"] is True
    assert boson.last_status == "skipped"
    np.testing.assert_allclose(projected, target, atol=1.0e-8)


def test_static_contamination_is_rejected():
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    target = verifier.reconstruct(_causal_coefficients(verifier))
    offset = 0.5 * float(np.max(np.abs(target)))
    contaminated = target + offset

    boson = _boson(dlr)
    with pytest.raises(RuntimeError, match="static"):
        boson.project(contaminated)
    with pytest.raises(RuntimeError, match="static"):
        boson.check(contaminated)

    # cleaning only the nu = 0 node must NOT silence the guard
    nu = np.asarray(dlr.nu, dtype=float)
    izero = int(np.argmin(np.abs(nu)))
    half_cleaned = contaminated.copy()
    half_cleaned[izero] = target[izero]
    with pytest.raises(RuntimeError, match="insufficient"):
        boson.project(half_cleaned)

    # the clean dynamic part passes the static guard; bosonic tail-based
    # moments are deferred (fermion-only), so check() then raises past the
    # guard.  This asserts the guard still fires first for contaminated data
    # above, and documents the deferred-boson boundary here.
    with pytest.raises(NotImplementedError):
        boson.check(target)


def test_clean_target_decays_within_tail_guard():
    # fixture sanity: the guard must not reject legitimate dynamic data
    dlr = _boson_dlr()
    verifier = _BosonVerifier(dlr)
    target = verifier.reconstruct(_causal_coefficients(verifier))

    nu = np.asarray(dlr.nu, dtype=float)
    tail = np.max(np.abs(target[np.argsort(np.abs(nu))[-2:]]))
    assert tail < 0.1 * np.max(np.abs(target))


def test_invalid_tail_tol_is_rejected():
    dlr = _boson_dlr()
    with pytest.raises(ValueError, match="tail_tol"):
        _boson(dlr, tail_tol=0.0)
    with pytest.raises(ValueError, match="tail_tol"):
        _boson(dlr, tail_tol=-1.0)
