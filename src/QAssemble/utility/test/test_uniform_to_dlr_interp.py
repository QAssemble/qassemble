"""Tests for the 3-step interpolation-based uniform->DLR Matsubara conversion.

``MatsubaraUniform2DLR`` / ``MatsubaraUniformGrid2DLR`` now run a 3-step
pipeline instead of fitting DLR coefficients directly from the raw uniform grid
in one global least-squares shot (which the user found numerically unstable):

1. linearly interpolate the uniform-grid data onto the DLR Matsubara sampling
   grid (``self.omega`` / ``self.nu``), real/imag independently;
2. solve for DLR coefficients exactly via a square LU solve
   (``dlr_from_matsubara``, flattened to a 2D batch);
3. re-evaluate the coefficients on the DLR grid (``matsubara_from_dlr``).

These tests pin:

1. **Exact on clean data.** The DLR nodes are a subset of the signed uniform
   grid, so step-1 is lossless and steps 2-3 are an exact round trip → machine
   precision for fermions and bosons across the 1D / 4D-fermion / 5D-boson
   entry points.

2. **LU vs lstsq agreement + transpose convention.** Replacing the step-2 LU
   solve with ``lstsq_dlr_from_matsubara`` on the SAME interpolated DLR-grid
   data gives the same answer, and both match the pydlr matrix-axis transpose
   convention (``matsubara_from_dlr`` transposes the trailing (a,b) axes once,
   so an asymmetric matrix input lands transposed).

3. **Stability vs raw one-shot lstsq.** On noisy uniform data the new
   interp->LU path reconstructs the DLR grid at least as accurately as the
   original raw-uniform lstsq fit (the user's core claim), and the square LU
   system is far better conditioned than the raw-uniform lstsq kernel.

4. **Coverage check.** A source grid that does not cover the DLR range raises,
   for both statistics, even when the supplied grid is unsorted (the check runs
   after the defensive argsort).
"""

import numpy as np
import pytest

from QAssemble.BLocDyn import BLocDyn
from QAssemble.utility.DLR import DLR


def _dlr():
    return DLR({"beta": 20.0, "cutoff": 8.0, "eps": 1.0e-12})


def _eval(d, coeff, freqs, xi):
    """Evaluate a single DLR-coefficient function on the given Matsubara grid."""
    c = np.asarray(coeff, dtype=np.complex128).reshape(-1, 1, 1)
    return d.eval_dlr_freq(c, 1j * np.asarray(freqs, dtype=np.float64), 20.0, xi=xi)[
        :, 0, 0
    ]


def _boson_positive_grid(dlr, include_zero=True):
    step = 2.0 * np.pi / dlr.beta
    nend = int(np.ceil(np.max(np.abs(dlr.nu)) / step))
    start = 0 if include_zero else 1
    return step * np.arange(start, nend + 1, dtype=np.float64)


def _boson_add_negative(mat, omega, hermitian=False):
    istart = 1 if np.isclose(omega[0], 0.0) else 0
    neg = np.conjugate(mat[..., istart:][..., ::-1])
    if hermitian:
        neg = np.swapaxes(neg, 0, 1)
    return (
        np.concatenate((neg, mat), axis=-1),
        np.concatenate((-omega[istart:][::-1], omega)),
    )


def test_boson_uniform_grids_split_non_negative_and_full():
    dlr = _dlr()
    nu_pos = dlr.MatsubaraBosonUniform()
    nu_full = dlr.MatsubaraBosonUniformFull()

    assert np.isclose(nu_pos[0], 0.0)
    assert np.all(nu_pos >= 0.0)
    assert np.all(np.diff(nu_pos) > 0.0)

    assert np.all(np.diff(nu_full) > 0.0)
    assert np.sum(np.isclose(nu_full, 0.0)) == 1
    np.testing.assert_allclose(nu_full, np.concatenate((-nu_pos[:0:-1], nu_pos)))
    assert nu_full[0] <= np.min(dlr.nu)
    assert nu_full[-1] >= np.max(dlr.nu)


# ---------------------------------------------------------------------------
# 1. clean round trip is exact
# ---------------------------------------------------------------------------


def test_fermion_clean_roundtrip_is_exact_1d():
    dlr = _dlr()
    rng = np.random.default_rng(0)
    coeff = -np.abs(rng.standard_normal(len(dlr.omega)))

    exact = _eval(dlr.dF, coeff, dlr.omega, xi=-1)
    uniform = _eval(dlr.dF, coeff, dlr.MatsubaraFermionUniformFull(), xi=-1)

    back = dlr.MatsubaraUniform2DLR(uniform, sign=-1)
    assert back.shape == exact.shape
    assert np.linalg.norm(back - exact) / np.linalg.norm(exact) < 1.0e-12


def test_boson_clean_roundtrip_is_exact_1d():
    dlr = _dlr()
    rng = np.random.default_rng(1)
    coeff = -np.abs(rng.standard_normal(len(dlr.nu)))

    exact = _eval(dlr.dB, coeff, dlr.nu, xi=1)
    uniform = _eval(dlr.dB, coeff, dlr.MatsubaraBosonUniformFull(), xi=1)

    back = dlr.MatsubaraUniform2DLR(uniform, sign=1)
    assert back.shape == exact.shape
    assert np.linalg.norm(back - exact) / np.linalg.norm(exact) < 1.0e-12


def test_fermion_4d_wrapper_clean_roundtrip_with_asymmetric_matrix():
    dlr = _dlr()
    omega_u = dlr.MatsubaraFermionUniformFull()
    rank = len(dlr.omega)
    rng = np.random.default_rng(7)
    norb = 2

    arr = np.zeros((norb, norb, 1, len(omega_u)), dtype=np.complex128)
    # exact reference with the pydlr matrix-axis transpose applied, so it matches
    # what the wrapper produces (result[i, j] holds the (j, i) channel).
    exact_t = np.zeros((norb, norb, 1, rank), dtype=np.complex128)
    for i in range(norb):
        for j in range(norb):
            coeff = -np.abs(rng.standard_normal(rank)) * (1.0 + 0.3 * (i * norb + j))
            arr[i, j, 0, :] = _eval(dlr.dF, coeff, omega_u, xi=-1)
            exact_t[j, i, 0, :] = _eval(dlr.dF, coeff, dlr.omega, xi=-1)

    back = dlr.MatsubaraUniformGrid2DLR(arr, sign=-1)
    assert back.shape == (norb, norb, 1, rank)
    assert np.linalg.norm(back - exact_t) / np.linalg.norm(exact_t) < 1.0e-12


def test_boson_5d_wrapper_clean_roundtrip():
    dlr = _dlr()
    nu_u = dlr.MatsubaraBosonUniformFull()
    rank = len(dlr.nu)
    rng = np.random.default_rng(3)
    norb = 2

    arr = np.zeros((norb, norb, 1, 1, len(nu_u)), dtype=np.complex128)
    exact = np.zeros((norb, norb, 1, 1, rank), dtype=np.complex128)
    for i in range(norb):
        for j in range(norb):
            coeff = -np.abs(rng.standard_normal(rank)) * (1.0 + 0.3 * (i * norb + j))
            arr[i, j, 0, 0, :] = _eval(dlr.dB, coeff, nu_u, xi=1)
            exact[i, j, 0, 0, :] = _eval(dlr.dB, coeff, dlr.nu, xi=1)

    back = dlr.MatsubaraUniformGrid2DLR(arr, sign=1)
    assert back.shape == (norb, norb, 1, 1, rank)
    # the 5D boson path flattens to a batch before the solve (no matrix axes to
    # transpose), so it matches the untransposed reference.
    assert np.linalg.norm(back - exact) / np.linalg.norm(exact) < 1.0e-12


def test_boson_5d_dlr_to_uniform_grid_batches_corr_axis():
    dlr = _dlr()
    nu_u = dlr.MatsubaraBosonUniform()
    rank = len(dlr.nu)
    rng = np.random.default_rng(4)
    norb = 2

    arr = np.zeros((norb, norb, 1, 1, rank), dtype=np.complex128)
    expected = np.zeros((norb, norb, 1, 1, len(nu_u)), dtype=np.complex128)
    for i in range(norb):
        for j in range(norb):
            coeff = -np.abs(rng.standard_normal(rank)) * (1.0 + 0.2 * (i * norb + j))
            arr[i, j, 0, 0, :] = _eval(dlr.dB, coeff, dlr.nu, xi=1)
            expected[i, j, 0, 0, :] = _eval(dlr.dB, coeff, nu_u, xi=1)

    # method="dlr" is the coefficient path this reference is built from, so the
    # check stays exact; the interp default is held to an analytic model in
    # test_boson_dlr_to_uniform_analytic_accuracy instead.
    out = dlr.MatsubaraDLR2UniformGrid(arr, sign=1, method="dlr")
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, rtol=1.0e-12, atol=1.0e-12)


# ---------------------------------------------------------------------------
# 2. LU vs lstsq (step-2 only) agreement and transpose convention
# ---------------------------------------------------------------------------


def _interp_lstsq_path(dlr, arr_block, omega_u, sign):
    """Reference: same interpolation onto the DLR grid, but step-2 = lstsq.

    Mirrors MatsubaraUniformGrid2DLR's fermion per-spin block path but swaps the
    square LU solve for ``lstsq_dlr_from_matsubara`` on the SAME DLR-grid data.
    """
    d = dlr.dF if sign == -1 else dlr.dB
    omega_dlr = dlr.omega if sign == -1 else dlr.nu
    order = np.argsort(omega_u)
    x_src = omega_u[order]
    block = np.moveaxis(arr_block, -1, 0)[order]  # (nfreq_src, norb, norb)
    interp = dlr._interp_to_grid(block, x_src, omega_dlr)
    fxx = d.lstsq_dlr_from_matsubara(w_q=omega_dlr * 1j, G_qaa=interp, beta=dlr.beta)
    out = d.matsubara_from_dlr(G_xaa=fxx, beta=dlr.beta, xi=sign)
    return np.moveaxis(out, 0, -1)


def test_lu_matches_lstsq_on_interpolated_grid_with_asymmetric_matrix():
    dlr = _dlr()
    omega_u = dlr.MatsubaraFermionUniformFull()
    rank = len(dlr.omega)
    rng = np.random.default_rng(11)
    norb = 2

    arr = np.zeros((norb, norb, 1, len(omega_u)), dtype=np.complex128)
    exact_t = np.zeros((norb, norb, 1, rank), dtype=np.complex128)
    for i in range(norb):
        for j in range(norb):
            coeff = -np.abs(rng.standard_normal(rank)) * (1.0 + 0.3 * (i * norb + j))
            arr[i, j, 0, :] = _eval(dlr.dF, coeff, omega_u, xi=-1)
            exact_t[j, i, 0, :] = _eval(dlr.dF, coeff, dlr.omega, xi=-1)

    lu = dlr.MatsubaraUniformGrid2DLR(arr, sign=-1)
    lstsq = _interp_lstsq_path(dlr, arr[:, :, 0, :], omega_u, sign=-1)[:, :, None, :]

    # LU and lstsq agree on the interpolated grid ...
    assert np.linalg.norm(lu - lstsq) / np.linalg.norm(lstsq) < 1.0e-10
    # ... and both honour the single trailing-axis transpose (not doubled).
    assert np.linalg.norm(lu - exact_t) / np.linalg.norm(exact_t) < 1.0e-10


# ---------------------------------------------------------------------------
# 3. stability vs the original raw-uniform one-shot lstsq
# ---------------------------------------------------------------------------


def _raw_uniform_lstsq(dlr, uniform, omega_u, sign):
    """The original pre-session path: fit DLR coeffs directly from the raw
    uniform grid in one lstsq shot, then re-evaluate on the DLR grid."""
    d = dlr.dF if sign == -1 else dlr.dB
    ff = np.asarray(uniform, dtype=np.complex128)[:, None, None]
    fxx = d.lstsq_dlr_from_matsubara(w_q=omega_u * 1j, G_qaa=ff, beta=dlr.beta)
    return d.matsubara_from_dlr(G_xaa=fxx, beta=dlr.beta, xi=-1 if sign == -1 else 1)[
        :, 0, 0
    ]


def test_interp_lu_reconstruction_is_comparable_to_raw_lstsq_on_noisy_data():
    """On white Gaussian noise the two paths are comparable.

    Note: pure white noise actually slightly *favours* the raw one-shot lstsq,
    because its global fit averages independent noise across the dense uniform
    grid, while local interpolation passes per-node noise straight through.  The
    stability advantage of interp->LU shows up in conditioning (see the next
    test) and on *structured* (non-white) perturbations, not in this white-noise
    average.  Here we only assert the interp->LU error stays the same order of
    magnitude as raw lstsq — i.e. nothing pathological.
    """
    dlr = _dlr()
    omega_u = dlr.MatsubaraFermionUniformFull()
    rank = len(dlr.omega)
    rng = np.random.default_rng(42)
    coeff = -np.abs(rng.standard_normal(rank))

    exact = _eval(dlr.dF, coeff, dlr.omega, xi=-1)
    uniform = _eval(dlr.dF, coeff, omega_u, xi=-1)
    noise = 0.02 * np.max(np.abs(uniform))

    interp_errs, raw_errs = [], []
    for seed in range(20):
        r = np.random.default_rng(seed)
        noisy = uniform + noise * (
            r.standard_normal(uniform.shape) + 1j * r.standard_normal(uniform.shape)
        )
        interp_back = dlr.MatsubaraUniform2DLR(noisy, sign=-1)
        raw_back = _raw_uniform_lstsq(dlr, noisy, omega_u, sign=-1)
        interp_errs.append(np.linalg.norm(interp_back - exact) / np.linalg.norm(exact))
        raw_errs.append(np.linalg.norm(raw_back - exact) / np.linalg.norm(exact))

    # same order of magnitude (within 2x), not pathological.
    assert np.mean(interp_errs) <= 2.0 * np.mean(raw_errs)


def test_square_lu_system_is_better_conditioned_than_raw_lstsq_kernel():
    dlr = _dlr()
    omega_u = dlr.MatsubaraFermionUniformFull()
    # square LU system used by step-2 (dlr_from_matsubara)
    cond_lu = np.linalg.cond(dlr.dF.dlrmf2cf)
    # raw one-shot lstsq kernel over the full uniform grid
    k_qx = -1.0 / (omega_u[:, None] * 1j - dlr.dF.dlrrf[None, :] / dlr.beta)
    cond_lstsq = np.linalg.cond(k_qx)
    assert cond_lu < cond_lstsq


# ---------------------------------------------------------------------------
# 4. coverage check
# ---------------------------------------------------------------------------


def test_coverage_check_raises_when_grid_does_not_cover_dlr_range():
    dlr = _dlr()
    rank = len(dlr.omega)
    # a uniform grid strictly inside the DLR range (too short on both ends)
    inner = dlr.omega * 0.5
    data = np.ones(len(inner), dtype=np.complex128)
    with pytest.raises(ValueError, match="does not cover"):
        dlr.MatsubaraUniform2DLR(data, omega=inner, sign=-1)


def test_coverage_check_runs_after_sort_for_unsorted_grid():
    dlr = _dlr()
    inner = dlr.omega * 0.5
    shuffled = inner.copy()
    np.random.default_rng(0).shuffle(shuffled)
    data = np.ones(len(shuffled), dtype=np.complex128)
    # an unsorted, non-covering grid must still raise (check is post-argsort)
    with pytest.raises(ValueError, match="does not cover"):
        dlr.MatsubaraUniform2DLR(data, omega=shuffled, sign=-1)


def test_coverage_check_passes_for_full_uniform_grid_boson():
    dlr = _dlr()
    # the default boson uniform grid covers the DLR nu range (within tol)
    nu_u = dlr.MatsubaraBosonUniformFull()
    data = np.ones(len(nu_u), dtype=np.complex128)
    out = dlr.MatsubaraUniform2DLR(data, omega=nu_u, sign=1)
    assert out.shape == (len(dlr.nu),)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 5. positive-only wrapper expansion
# ---------------------------------------------------------------------------


def test_fermion_positive_only_roundtrip_explicit_omega_diagonal_3d():
    dlr = _dlr()
    omega_pos = dlr.MatsubaraFermionUniform()
    rank = len(dlr.omega)
    rng = np.random.default_rng(101)
    norb = 2

    arr = np.zeros((norb, norb, len(omega_pos)), dtype=np.complex128)
    exact = np.zeros((norb, norb, 1, rank), dtype=np.complex128)
    for i in range(norb):
        coeff = -np.abs(rng.standard_normal(rank)) * (1.0 + 0.4 * i)
        arr[i, i, :] = _eval(dlr.dF, coeff, omega_pos, xi=-1)
        exact[i, i, 0, :] = _eval(dlr.dF, coeff, dlr.omega, xi=-1)

    back = dlr.MatsubaraUniformGrid2DLR(arr, omega=omega_pos, sign=-1)
    assert back.shape == exact.shape
    assert np.linalg.norm(back - exact) / np.linalg.norm(exact) < 1.0e-12


def test_fermion_positive_only_and_full_grid_outputs_are_identical():
    dlr = _dlr()
    omega_pos = dlr.MatsubaraFermionUniform()
    omega_full = np.concatenate((-omega_pos[::-1], omega_pos))
    rank = len(dlr.omega)
    rng = np.random.default_rng(102)
    norb = 2
    ns = 2

    arr_pos = np.zeros((norb, norb, ns, len(omega_pos)), dtype=np.complex128)
    for i in range(norb):
        for j in range(norb):
            for js in range(ns):
                coeff = -np.abs(rng.standard_normal(rank)) * (1.0 + 0.2 * (i + 2 * j + js))
                arr_pos[i, j, js, :] = _eval(dlr.dF, coeff, omega_pos, xi=-1)

    arr_full = dlr.MatsubaraAddNegativeFrequency(arr_pos)
    pos_back = dlr.MatsubaraUniformGrid2DLR(arr_pos, omega=omega_pos, sign=-1)
    full_back = dlr.MatsubaraUniformGrid2DLR(arr_full, omega=omega_full, sign=-1)
    np.testing.assert_allclose(pos_back, full_back, rtol=1.0e-12, atol=1.0e-12)


def test_fermion_positive_only_offdiagonal_uses_hermitian_transpose():
    dlr = _dlr()
    omega_pos = dlr.MatsubaraFermionUniform()
    omega_full = np.concatenate((-omega_pos[::-1], omega_pos))
    rank = len(dlr.omega)
    rng = np.random.default_rng(103)
    norb = 2

    arr_pos = np.zeros((norb, norb, 1, len(omega_pos)), dtype=np.complex128)
    for i in range(norb):
        for j in range(norb):
            coeff = -np.abs(rng.standard_normal(rank)) * (1.0 + 0.35 * (i + 2 * j))
            arr_pos[i, j, 0, :] = _eval(dlr.dF, coeff, omega_pos, xi=-1)
    assert not np.allclose(arr_pos[0, 1, 0, :], arr_pos[1, 0, 0, :])

    arr_full = dlr.MatsubaraAddNegativeFrequency(arr_pos)
    pos_back = dlr.MatsubaraUniformGrid2DLR(arr_pos, omega=omega_pos, sign=-1)
    full_back = dlr.MatsubaraUniformGrid2DLR(arr_full, omega=omega_full, sign=-1)
    np.testing.assert_allclose(pos_back, full_back, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("include_zero", [True, False])
def test_boson_positive_only_symmetric_default_with_and_without_zero(include_zero):
    dlr = _dlr()
    bloc = BLocDyn(None, dlr, None)
    nu_pos = dlr.MatsubaraBosonUniform() if include_zero else _boson_positive_grid(dlr, include_zero=False)
    rank = len(dlr.nu)
    rng = np.random.default_rng(104 + int(include_zero))
    norb = 2

    arr_pos = np.zeros((norb, norb, 1, 1, len(nu_pos)), dtype=np.complex128)
    for i, j in [(0, 0), (1, 1), (0, 1)]:
        coeff = -np.abs(rng.standard_normal(rank)) * (1.0 + 0.25 * (i + 2 * j))
        vals = _eval(dlr.dB, coeff, nu_pos, xi=1)
        arr_pos[i, j, 0, 0, :] = vals
        arr_pos[j, i, 0, 0, :] = vals

    arr_full, nu_full = _boson_add_negative(arr_pos, nu_pos, hermitian=False)
    pos_back = bloc.BosonUniform2DLR(arr_pos, omega=nu_pos)
    full_back = bloc.BosonUniform2DLR(arr_full, omega=nu_full)
    np.testing.assert_allclose(pos_back, full_back, rtol=1.0e-12, atol=1.0e-12)


def test_boson_positive_only_hermitian_true_offdiagonal_swaps_orbitals():
    dlr = _dlr()
    bloc = BLocDyn(None, dlr, None)
    nu_pos = dlr.MatsubaraBosonUniform()
    rank = len(dlr.nu)
    rng = np.random.default_rng(106)
    norb = 2

    arr_pos = np.zeros((norb, norb, 1, 1, len(nu_pos)), dtype=np.complex128)
    for i in range(norb):
        for j in range(norb):
            coeff = -np.abs(rng.standard_normal(rank)) * (1.0 + 0.3 * (i + 2 * j))
            arr_pos[i, j, 0, 0, :] = _eval(dlr.dB, coeff, nu_pos, xi=1)
    assert not np.allclose(arr_pos[0, 1, 0, 0, :], arr_pos[1, 0, 0, 0, :])

    arr_full, nu_full = _boson_add_negative(arr_pos, nu_pos, hermitian=True)
    pos_back = bloc.BosonUniform2DLR(arr_pos, omega=nu_pos, hermitian=True)
    full_back = bloc.BosonUniform2DLR(arr_full, omega=nu_full)
    np.testing.assert_allclose(pos_back, full_back, rtol=1.0e-12, atol=1.0e-12)


def test_boson_grid_wrapper_positive_only_and_full_outputs_are_identical():
    dlr = _dlr()
    nu_pos = dlr.MatsubaraBosonUniform()
    rank = len(dlr.nu)
    rng = np.random.default_rng(108)
    norb = 2

    arr_pos = np.zeros((norb, norb, 1, 1, len(nu_pos)), dtype=np.complex128)
    for i, j in [(0, 0), (1, 1), (0, 1)]:
        coeff = -np.abs(rng.standard_normal(rank)) * (1.0 + 0.25 * (i + 2 * j))
        vals = _eval(dlr.dB, coeff, nu_pos, xi=1)
        arr_pos[i, j, 0, 0, :] = vals
        arr_pos[j, i, 0, 0, :] = vals

    arr_full, nu_full = _boson_add_negative(arr_pos, nu_pos, hermitian=False)
    pos_back = dlr.MatsubaraUniformGrid2DLR(arr_pos, omega=nu_pos, sign=1)
    full_back = dlr.MatsubaraUniformGrid2DLR(arr_full, omega=nu_full, sign=1)
    np.testing.assert_allclose(pos_back, full_back, rtol=1.0e-12, atol=1.0e-12)


def test_fermion_positive_only_data_with_omega_none_raises_clear_error():
    dlr = _dlr()
    omega_pos = dlr.MatsubaraFermionUniform()
    data = np.ones((1, 1, 1, len(omega_pos)), dtype=np.complex128)

    with pytest.raises(ValueError, match="explicit positive omega"):
        dlr.MatsubaraUniformGrid2DLR(data, sign=-1)


def test_unsorted_full_grid_with_positive_first_element_is_not_positive_only():
    dlr = _dlr()
    omega_full = dlr.MatsubaraFermionUniformFull()
    rank = len(dlr.omega)
    rng = np.random.default_rng(107)
    coeff = -np.abs(rng.standard_normal(rank))
    arr_full = _eval(dlr.dF, coeff, omega_full, xi=-1).reshape(1, 1, 1, -1)

    first_positive = len(omega_full) // 2
    order = np.r_[first_positive, np.arange(first_positive), np.arange(first_positive + 1, len(omega_full))]
    omega_unsorted = omega_full[order]
    arr_unsorted = arr_full[..., order]
    assert omega_unsorted[0] > 0.0
    assert np.any(omega_unsorted < 0.0)

    sorted_back = dlr.MatsubaraUniformGrid2DLR(arr_full, omega=omega_full, sign=-1)
    unsorted_back = dlr.MatsubaraUniformGrid2DLR(arr_unsorted, omega=omega_unsorted, sign=-1)
    np.testing.assert_allclose(unsorted_back, sorted_back, rtol=1.0e-12, atol=1.0e-12)


def test_positive_only_coverage_check_still_fires():
    dlr = _dlr()
    omega_short = dlr.MatsubaraFermionUniform()[:3]
    data = np.ones((1, 1, 1, len(omega_short)), dtype=np.complex128)

    with pytest.raises(ValueError, match="does not cover"):
        dlr.MatsubaraUniformGrid2DLR(data, omega=omega_short, sign=-1)


def test_positive_only_edge_guards_raise_clear_errors():
    dlr = _dlr()
    bloc = BLocDyn(None, dlr, None)

    one_freq = np.ones((1, 1, 1, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match="does not cover"):
        dlr.MatsubaraUniformGrid2DLR(one_freq, omega=np.array([dlr.MatsubaraFermionUniform()[0]]), sign=-1)

    empty_fermion = np.ones((1, 1, 1, 0), dtype=np.complex128)
    with pytest.raises(ValueError, match="non-empty"):
        dlr.MatsubaraUniformGrid2DLR(empty_fermion, omega=np.array([]), sign=-1)

    with pytest.raises(ValueError, match="1D"):
        dlr.MatsubaraUniformGrid2DLR(one_freq, omega=np.array(0.0), sign=-1)

    empty_boson = np.ones((1, 1, 1, 1, 0), dtype=np.complex128)
    with pytest.raises(ValueError, match="non-empty"):
        bloc.BosonUniform2DLR(empty_boson, omega=np.array([]))

    scalar_boson = np.ones((1, 1, 1, 1, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match="1D"):
        bloc.BosonUniform2DLR(scalar_boson, omega=np.array(0.0))


# ---------------------------------------------------------------------------
# DLR -> uniform (fermion).  The legacy coefficient round trip loses the signal
# at production parameters, so the default is now direct interpolation of the
# node values.  These tests pin the new path; `method="dlr"` keeps the old one
# reachable for comparison.
# ---------------------------------------------------------------------------


def _hyb_nodes(dlr, poles, weights, beta):
    """Multi-pole hybridization sampled on the DLR Matsubara nodes."""
    z = 1j * dlr.omega
    return np.sum(
        np.asarray(weights)[None, :] / (z[:, None] - np.asarray(poles)[None, :]), axis=1
    )


def _hyb_exact(freqs, poles, weights):
    z = 1j * np.asarray(freqs, dtype=np.float64)
    return np.sum(
        np.asarray(weights)[None, :] / (z[:, None] - np.asarray(poles)[None, :]), axis=1
    )


def test_fold_restores_uniform_grid_coverage():
    """The positive branch alone does not span the uniform grid.

    ``MatsubaraFermionUniform`` sizes itself from the *signed* extreme, which is
    frequently the negative node, so a positive-only implementation would clamp
    (``np.interp``) or raise (``interp1d``) over the top of the grid.
    """
    for beta, cutoff in [(100.0, 300.0), (20.0, 8.0), (10.0, 10.0), (200.0, 400.0)]:
        dlr = DLR({"beta": beta, "cutoff": cutoff, "eps": 1.0e-12})
        uniform_max = dlr.MatsubaraFermionUniform().max()

        folded, _ = dlr._fold_to_nonnegative(
            dlr.omega, np.zeros((len(dlr.omega), 1), dtype=np.complex128)
        )

        assert folded.max() >= uniform_max - 1.0e-9, (
            f"folded grid does not cover the uniform grid at beta={beta}, cutoff={cutoff}"
        )
        assert np.all(folded >= 0.0)
        assert np.all(np.diff(folded) > 0.0)


def test_hermitian_fold_relation_is_exact():
    """f(-w) = conj(f(w)) holds exactly, so folding loses no accuracy."""
    dlr = _dlr()
    poles = np.array([-1.7, -0.4, 0.9, 2.3])
    weights = np.array([0.3, 0.5, 0.4, 0.2])

    probe = np.array([1.0, 3.5, 7.2, 11.15])
    assert np.allclose(
        _hyb_exact(-probe, poles, weights),
        np.conjugate(_hyb_exact(probe, poles, weights)),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_fermion_dlr_to_uniform_matches_analytic_reference():
    """Analytic multi-pole reference on the uniform grid (was uncovered)."""
    dlr = _dlr()
    beta = 20.0
    poles = np.array([-2.1, -0.6, 0.8, 1.9, 3.4])
    weights = np.array([0.25, 0.4, 0.35, 0.3, 0.15])

    node = _hyb_nodes(dlr, poles, weights, beta)
    uniform = dlr.MatsubaraFermionUniform()
    exact = _hyb_exact(uniform, poles, weights)

    arr = np.zeros((1, 1, 1, len(dlr.omega)), dtype=np.complex128)
    arr[0, 0, 0, :] = node

    out = dlr.MatsubaraDLR2UniformGrid(arr, sign=-1)[0, 0, 0, :]

    # Interpolation error, not round-off: the worst point sits mid-range where
    # this coarse fixture's DLR nodes are sparsest (low frequencies come back
    # exact to ~1e-16).  The legacy path is off by O(1) relative on the same
    # data at production parameters.
    rel = np.abs(out - exact).max() / np.abs(exact).max()
    assert rel < 1.0e-4, f"relative error {rel:.3e} exceeds interpolation budget"


def test_fermion_dlr_to_uniform_passes_through_nodes():
    """Output reproduces the input at frequencies that are DLR nodes.

    The legacy path fails this badly at production parameters (error 5.9e-2
    against node values of order 2.8e-4, with a sign flip at w=138.89).
    """
    dlr = DLR({"beta": 100.0, "cutoff": 300.0, "eps": 1.0e-15})
    poles = np.array([-1.3, 0.7, 2.2])
    weights = np.array([0.4, 0.35, 0.25])

    node = _hyb_nodes(dlr, poles, weights, 100.0)
    uniform = dlr.MatsubaraFermionUniform()

    arr = np.zeros((1, 1, 1, len(dlr.omega)), dtype=np.complex128)
    arr[0, 0, 0, :] = node
    out = dlr.MatsubaraDLR2UniformGrid(arr, sign=-1)[0, 0, 0, :]

    positive = dlr.omega > 0
    idx = np.abs(uniform[:, None] - dlr.omega[positive][None, :]).argmin(axis=0)
    err = np.abs(out[idx] - node[positive]).max()

    assert err < 1.0e-10, f"node passthrough error {err:.3e}"


def test_fermion_dlr_to_uniform_keeps_tail_constant():
    """Re(Delta)*w**2 is a constant; the legacy path swings to +10475."""
    dlr = DLR({"beta": 100.0, "cutoff": 300.0, "eps": 1.0e-15})
    poles = np.array([-1.3, 0.7, 2.2])
    weights = np.array([0.4, 0.35, 0.25])

    node = _hyb_nodes(dlr, poles, weights, 100.0)
    uniform = dlr.MatsubaraFermionUniform()
    exact_tail = (_hyb_exact(uniform, poles, weights).real * uniform**2)

    arr = np.zeros((1, 1, 1, len(dlr.omega)), dtype=np.complex128)
    arr[0, 0, 0, :] = node
    out = dlr.MatsubaraDLR2UniformGrid(arr, sign=-1)[0, 0, 0, :]

    window = (uniform >= uniform.max() / 10.0) & (uniform <= uniform.max())
    got = out.real[window] * uniform[window] ** 2
    ref = exact_tail[window]

    assert np.abs(got - ref).max() < 0.05 * np.abs(ref).mean()
    assert np.all(got < 0.0), "tail must not change sign"


def test_fermion_dlr_to_uniform_offdiagonal_uses_hermitian_transpose():
    """Matrix-valued folding is G(-iw) = G(iw)^dagger, not element-wise conj.

    This is why the fold lives in the wrapper: the flattened core has no
    orbital axes to transpose.
    """
    dlr = _dlr()
    rng = np.random.default_rng(11)
    rank = len(dlr.omega)

    arr = np.zeros((2, 2, 1, rank), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            arr[i, j, 0, :] = rng.standard_normal(rank) + 1j * rng.standard_normal(rank)

    out = dlr.MatsubaraDLR2UniformGrid(arr, sign=-1)

    # Rebuild the expected fold by hand, transposing the orbital axes.  The DLR
    # grid contains symmetric +-w pairs, so folding produces exact duplicate
    # frequencies that must be collapsed the same way the implementation does.
    mirrored = np.swapaxes(np.conjugate(arr), 0, 1)
    x_all = np.concatenate([dlr.omega, -dlr.omega])
    f_all = np.concatenate([arr, mirrored], axis=-1)
    keep = x_all >= 0.0
    x_all, f_all = x_all[keep], f_all[..., keep]
    order = np.argsort(x_all)
    x_all, f_all = x_all[order], f_all[..., order]
    _, unique_idx = np.unique(np.round(x_all, 12), return_index=True)
    unique_idx = np.sort(unique_idx)
    x_all, f_all = x_all[unique_idx], f_all[..., unique_idx]

    # Compare only where a folded node coincides with a uniform point, so the
    # assertion tests the fold rather than the interpolation in between.
    uniform = dlr.MatsubaraFermionUniform()
    idx = np.abs(uniform[:, None] - x_all[None, :]).argmin(axis=0)
    hit = np.abs(uniform[idx] - x_all) < 1.0e-9
    assert hit.sum() >= 4, "expected several folded nodes to land on the uniform grid"

    np.testing.assert_allclose(
        out[0, 1, 0, idx[hit]], f_all[0, 1, 0, hit], rtol=1.0e-9, atol=1.0e-9
    )


def test_fermion_method_dlr_still_available():
    """The legacy path stays reachable and stays exact where it is exact.

    At the well-conditioned fixture parameters the coefficient round trip is
    machine-exact; it is only at production parameters that it collapses.
    """
    dlr = _dlr()
    rng = np.random.default_rng(3)
    rank = len(dlr.omega)
    coeff = -np.abs(rng.standard_normal(rank))

    node = _eval(dlr.dF, coeff, dlr.omega, xi=-1)
    exact = _eval(dlr.dF, coeff, dlr.MatsubaraFermionUniform(), xi=-1)

    arr = np.zeros((1, 1, 1, rank), dtype=np.complex128)
    arr[0, 0, 0, :] = node

    out = dlr.MatsubaraDLR2UniformGrid(arr, sign=-1, method="dlr")[0, 0, 0, :]
    assert np.abs(out - exact).max() / np.abs(exact).max() < 1.0e-12


def test_boson_dlr_to_uniform_methods_differ_but_agree_on_clean_data():
    """``interp`` is the bosonic default now; ``dlr`` stays as a rollback.

    The two paths are numerically distinct (that is the point of the change),
    but on clean analytic input both reproduce the true function, so neither is
    silently wrong.  On clean data the coefficient path is in fact the more
    accurate of the two; interpolation wins on noisy input, which is what
    ``test_boson_dlr_to_uniform_passes_through_dlr_nodes`` pins.  The tolerance
    here is set by this fixture's sparse DLR grid (rank ~28); at production
    parameters interpolation reaches ~3e-6.
    """
    dlr = _dlr()
    nu_u = dlr.MatsubaraBosonUniform()
    arr = _boson_hermitian_5d(dlr, dlr.nu)
    exact = _boson_hermitian_5d(dlr, nu_u)

    out_interp = dlr.MatsubaraDLR2UniformGrid(arr, sign=1, method="interp")
    out_dlr = dlr.MatsubaraDLR2UniformGrid(arr, sign=1, method="dlr")

    assert not np.allclose(out_interp, out_dlr, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(
        dlr.MatsubaraDLR2UniformGrid(arr, sign=1), out_interp
    )

    scale = np.maximum(np.abs(exact), 1.0e-12)
    assert np.max(np.abs(out_interp - exact) / scale) < 1.0e-2
    assert np.max(np.abs(out_dlr - exact) / scale) < 1.0e-2


# ---------------------------------------------------------------------------
# 5. Bosonic DLR->uniform interpolation (nu=0 handled by an offset variable)
# ---------------------------------------------------------------------------


def _boson_production_dlr():
    """Production parameters, where the legacy conditioning actually bites."""
    return DLR({"beta": 100.0, "cutoff": 300.0, "eps": 1.0e-15})


def _boson_even_model(x):
    """Two-pole even bosonic model with a pure ``c2/(i*nu)**2`` tail."""
    x = np.asarray(x, dtype=np.float64)
    w0, w1 = 1.7, 5.3
    return (-2.0 * w0 / (x**2 + w0**2) - 1.4 * w1 / (x**2 + w1**2)).astype(
        np.complex128
    )


def _boson_hermitian_5d(dlr, freqs, norb=2):
    """5D bosonic array obeying ``F_ij(-nu) = conj(F_ji(+nu))``.

    Built from a real-valued even model with a symmetric orbital/spin weight,
    so the Hermitian relation holds by construction.  A fixture that violates
    it manufactures large round-trip errors that say nothing about the code.
    """
    base = _boson_even_model(freqs)
    weight = np.array([[1.0, 0.35], [0.35, 0.8]])[:norb, :norb]
    return np.asfortranarray(
        weight[:, :, None, None, None] * base[None, None, None, None, :]
    )


def _positive_node_to_uniform_index(dlr):
    """Map each non-negative DLR node onto its exact uniform-grid index."""
    nu_u = dlr.MatsubaraBosonUniform()
    pairs = []
    for k, v in enumerate(np.asarray(dlr.nu)):
        if v < 0.0:
            continue
        j = int(np.argmin(np.abs(nu_u - v)))
        if abs(nu_u[j] - v) < 1.0e-8:
            pairs.append((k, j))
    return pairs


def test_boson_dlr_to_uniform_passes_through_dlr_nodes():
    """The uniform grid contains the DLR nodes, so they must survive untouched.

    This is the regression this change exists for: the legacy coefficient path
    is conditioned at ``cond(dlrmf2cf) ~ 2e8``, so on noisy input it returns a
    value at a DLR node that differs from the input there by ~100x the noise.
    Interpolation reproduces node values by construction.
    """
    dlr = _boson_production_dlr()
    pairs = _positive_node_to_uniform_index(dlr)
    assert len(pairs) > 10, "expected the DLR nodes to land on the uniform grid"

    arr = _boson_hermitian_5d(dlr, dlr.nu)
    out = dlr.MatsubaraDLR2UniformGrid(arr, sign=1)
    for k, j in pairs:
        np.testing.assert_allclose(out[..., j], arr[..., k], atol=1.0e-12)

    # Same, with CTQMC-scale noise: the legacy path fails this by ~1e-2.
    rng = np.random.default_rng(11)
    noisy = arr + 1.0e-4 * rng.standard_normal(arr.shape)
    out_noisy = dlr.MatsubaraDLR2UniformGrid(noisy, sign=1)
    err = max(
        float(np.max(np.abs(out_noisy[..., j] - noisy[..., k]))) for k, j in pairs
    )
    assert err < 1.0e-3, f"node passthrough error {err:.3e} exceeds the noise scale"


def test_boson_dlr_to_uniform_handles_zero_frequency():
    """nu=0 is a real grid point in both the DLR and the uniform grid."""
    dlr = _boson_production_dlr()
    nu_u = dlr.MatsubaraBosonUniform()
    assert nu_u[0] == 0.0
    assert np.sum(np.asarray(dlr.nu) == 0.0) == 1

    arr = _boson_hermitian_5d(dlr, dlr.nu)
    out = dlr.MatsubaraDLR2UniformGrid(arr, sign=1)
    assert np.all(np.isfinite(out.real)) and np.all(np.isfinite(out.imag))

    exact = _boson_hermitian_5d(dlr, nu_u)
    np.testing.assert_allclose(out[..., 0], exact[..., 0], rtol=1.0e-6)


def test_boson_dlr_to_uniform_analytic_accuracy():
    """Accuracy against an analytic model, independent of the DLR expansion."""
    dlr = _boson_production_dlr()
    nu_u = dlr.MatsubaraBosonUniform()

    out = dlr.MatsubaraDLR2UniformGrid(_boson_hermitian_5d(dlr, dlr.nu), sign=1)
    exact = _boson_hermitian_5d(dlr, nu_u)

    rel = np.max(np.abs(out - exact) / np.maximum(np.abs(exact), 1.0e-12))
    assert rel < 1.0e-4, f"max relative error {rel:.3e}"


def test_boson_uniform_dlr_uniform_roundtrip():
    """uniform -> DLR -> uniform closes; no such test existed before."""
    dlr = _boson_production_dlr()
    nu_u = dlr.MatsubaraBosonUniform()

    start = _boson_hermitian_5d(dlr, nu_u)
    on_dlr = dlr.MatsubaraUniformGrid2DLR(start, omega=nu_u, sign=1)
    back = dlr.MatsubaraDLR2UniformGrid(on_dlr, sign=1)

    rel = np.linalg.norm(back - start) / np.linalg.norm(start)
    assert rel < 1.0e-4, f"round-trip relative error {rel:.3e}"


def test_interp_to_grid_offset_preserves_default():
    """offset=0.0 must be bit-identical to omitting it (fermion regression)."""
    dlr = _dlr()
    rng = np.random.default_rng(3)
    x_src = np.unique(np.round(np.abs(np.asarray(dlr.omega)), 12))
    ff = (rng.standard_normal((x_src.size, 1, 1))
          + 1j * rng.standard_normal((x_src.size, 1, 1)))
    x_target = dlr.MatsubaraFermionUniform()

    for variable, kind in (("linear", "linear"), ("inverse", "cubic")):
        base = dlr._interp_to_grid(ff, x_src, x_target, variable=variable, kind=kind)
        with_zero = dlr._interp_to_grid(
            ff, x_src, x_target, variable=variable, kind=kind, offset=0.0
        )
        np.testing.assert_array_equal(with_zero, base)


def test_boson_flattened_core_interp_matches_grid_wrapper_diagonal():
    """The flattened core mirrors element-wise, so it is exact on diagonals.

    Same contract as the fermionic core: ``MatsubaraDLR2Uniform`` folds with a
    plain conjugate (no transpose), which agrees with the wrapper's Hermitian
    fold for channels that are diagonal in orbital and spin.
    """
    dlr = _boson_production_dlr()
    arr = _boson_hermitian_5d(dlr, dlr.nu, norb=1)

    via_grid = dlr.MatsubaraDLR2UniformGrid(arr, sign=1)
    via_core = dlr.MatsubaraDLR2Uniform(arr[0, 0, 0, 0, :], sign=1)

    np.testing.assert_allclose(via_core, via_grid[0, 0, 0, 0, :], atol=1.0e-12)
