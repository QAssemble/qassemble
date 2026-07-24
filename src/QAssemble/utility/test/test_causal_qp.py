import inspect

import numpy as np
import pytest

from QAssemble.utility.Causal import (
    CausalCheckResult,
    CausalProjection,
    CausalQPResult,
)


def test_negative_coefficient_sign_projects_to_nonpositive_cone():
    result = CausalProjection(coefficient_sign=-1).project(np.array([-2.0, 1.0, -0.5]))

    assert isinstance(result, CausalQPResult)
    assert result.success
    assert result.max_inequality_violation <= 1.0e-8
    assert np.max(result.coefficients) <= 1.0e-7
    np.testing.assert_allclose(result.coefficients[[0, 2]], [-2.0, -0.5], atol=1.0e-7)


def test_positive_coefficient_sign_projects_to_nonnegative_cone():
    result = CausalProjection(coefficient_sign=1).project(np.array([-1.0, 2.0]))

    assert result.success
    assert result.max_inequality_violation <= 1.0e-8
    assert np.min(result.coefficients) >= -1.0e-7
    np.testing.assert_allclose(result.coefficients[1], 2.0, atol=1.0e-7)


def test_equality_constraint_is_satisfied():
    result = CausalProjection(coefficient_sign=-1).project(
        np.array([-2.0, 1.0, -0.5]),
        equality_matrix=np.ones(3),
        equality_target=np.array([-1.0]),
    )

    assert result.success
    assert result.max_equality_residual <= 1.0e-8
    assert result.max_inequality_violation <= 1.0e-8
    np.testing.assert_allclose(np.sum(result.coefficients), -1.0, atol=1.0e-8)


def test_tiny_imaginary_reference_is_accepted():
    result = CausalProjection(coefficient_sign=-1).project(
        np.array([-1.0 + 1.0e-12j, 0.5 - 1.0e-12j])
    )

    assert result.success
    assert result.coefficients.dtype == np.float64
    assert np.max(result.coefficients) <= 1.0e-7


def test_large_imaginary_reference_is_rejected():
    with pytest.raises(ValueError, match="reference has imaginary part"):
        CausalProjection(coefficient_sign=-1).project(np.array([-1.0 + 1.0e-3j]))


@pytest.mark.parametrize(
    "reference",
    [
        np.array([], dtype=float),
        np.zeros((1, 1), dtype=float),
        np.array([np.nan]),
        np.array([np.inf]),
    ],
)
def test_invalid_reference_is_rejected(reference):
    with pytest.raises(ValueError):
        CausalProjection().project(reference)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 0},
        {"max_iter": -1},
        {"constraint_tol": 0.0},
        {"constraint_tol": -1.0e-8},
        {"constraint_tol": np.inf},
        {"solvers": ()},
    ],
)
def test_invalid_constructor_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CausalProjection(**kwargs)


def test_complex_constraints_are_checked():
    qp = CausalProjection(coefficient_sign=-1)
    with pytest.raises(ValueError, match="equality_matrix has imaginary part"):
        qp.project(
            np.array([-1.0]),
            equality_matrix=np.array([1.0 + 1.0e-3j]),
            equality_target=np.array([-1.0]),
        )

    with pytest.raises(ValueError, match="equality_target has imaginary part"):
        qp.project(
            np.array([-1.0]),
            equality_matrix=np.array([1.0]),
            equality_target=np.array([-1.0 + 1.0e-3j]),
        )


def test_instance_api_only():
    import importlib

    causal_module = importlib.import_module("QAssemble.utility.Causal")
    signature = inspect.signature(CausalProjection)

    assert hasattr(causal_module, "CausalProjection")
    assert hasattr(causal_module, "CausalQPResult")
    assert hasattr(causal_module, "CausalCheckResult")
    assert not hasattr(causal_module, "Causal")
    assert not hasattr(causal_module, "Casual")
    assert not hasattr(causal_module, "project_coefficients")
    assert "weights" not in signature.parameters
    assert "scale" not in signature.parameters


def test_removed_weight_and_scale_options_are_rejected():
    with pytest.raises(TypeError):
        CausalProjection(weights=np.ones(1))
    with pytest.raises(TypeError):
        CausalProjection(scale=1.0)


def test_check_negative_vector_is_causal():
    result = CausalProjection(coefficient_sign=-1).check(np.array([-2.0, -0.5]))

    assert isinstance(result, CausalCheckResult)
    assert result.causal
    assert result.max_inequality_violation == 0.0
    assert result.max_equality_residual == 0.0
    assert result.violating_count == 0


def test_check_positive_entries_are_flagged():
    result = CausalProjection(coefficient_sign=-1).check(np.array([-1.0, 0.5, 2.0]))

    assert not result.causal
    assert result.violating_count == 2
    assert result.max_inequality_violation == pytest.approx(2.0)


def test_check_equality_residual():
    qp = CausalProjection(coefficient_sign=-1)

    satisfied = qp.check(
        np.array([-0.5, -0.5]),
        equality_matrix=np.ones(2),
        equality_target=np.array([-1.0]),
    )
    assert satisfied.causal
    assert satisfied.max_equality_residual <= 1.0e-12

    violated = qp.check(
        np.array([-0.5, -0.5]),
        equality_matrix=np.ones(2),
        equality_target=np.array([-2.0]),
    )
    assert not violated.causal
    assert violated.max_equality_residual == pytest.approx(1.0)
    assert violated.violating_count == 0


def test_project_then_check_roundtrip():
    qp = CausalProjection(coefficient_sign=-1)
    result = qp.project(np.array([-2.0, 1.0, -0.5]))

    assert result.success
    verdict = qp.check(result.coefficients, tol=1.0e-7)
    assert verdict.causal


def test_check_per_call_tol_overrides_constraint_tol():
    qp = CausalProjection(coefficient_sign=-1)
    coeff = np.array([-1.0, 1.0e-6])

    strict = qp.check(coeff)
    assert not strict.causal
    assert strict.violating_count == 1

    loose = qp.check(coeff, tol=1.0e-4)
    assert loose.causal
    assert loose.violating_count == 0


@pytest.mark.parametrize("bad_tol", [0.0, -1.0e-8, np.inf, np.nan])
def test_invalid_per_call_tol_is_rejected(bad_tol):
    qp = CausalProjection(coefficient_sign=-1)
    with pytest.raises(ValueError, match="tol"):
        qp.check(np.array([-1.0]), tol=bad_tol)
    with pytest.raises(ValueError, match="tol"):
        qp.project(np.array([-1.0]), tol=bad_tol)


def test_identity_weight_matrix_matches_default_metric():
    ref = np.array([-2.0, 1.0, -0.5, 0.3])
    qp = CausalProjection(coefficient_sign=-1)

    base = qp.project(ref)
    weighted = qp.project(ref, weight_matrix=np.eye(ref.size))

    np.testing.assert_allclose(weighted.coefficients, base.coefficients, atol=1.0e-7)
    assert weighted.objective == pytest.approx(base.objective, abs=1.0e-10)


def test_offdiagonal_weight_changes_sign_active_solution():
    # ref[0] > 0 activates the sign constraint; the W coupling then pulls the
    # free component: analytic optimum is x = [0, -0.1] (vs [0, -1] for W=I).
    ref = np.array([1.0, -1.0])
    weight = np.array([[1.0, 0.9], [0.9, 1.0]])
    qp = CausalProjection(coefficient_sign=-1)

    base = qp.project(ref)
    weighted = qp.project(ref, weight_matrix=weight)

    np.testing.assert_allclose(base.coefficients, [0.0, -1.0], atol=1.0e-6)
    np.testing.assert_allclose(weighted.coefficients, [0.0, -0.1], atol=1.0e-6)
    # objective is reported in the weighted metric: diff^T W diff = 0.19
    assert weighted.objective == pytest.approx(0.19, abs=1.0e-5)


def test_tiny_negative_eigenvalue_is_clipped_not_rejected():
    # boundary contract: lambda_min = -5e-11 > -1e-10 * lambda_max, so the
    # eigenvalue is numerical dust — clipped to zero, no ValueError
    ref = np.array([-1.0, -2.0])
    weight = np.diag([1.0, -5.0e-11])

    result = CausalProjection(coefficient_sign=-1).project(ref, weight_matrix=weight)

    assert result.success
    assert result.coefficients[0] == pytest.approx(-1.0, abs=1.0e-6)


def test_negative_eigenvalue_below_threshold_is_rejected():
    # lambda_min = -1e-9 < -1e-10 * lambda_max: a genuinely indefinite metric
    weight = np.diag([1.0, -1.0e-9])

    with pytest.raises(ValueError, match="positive semidefinite"):
        CausalProjection(coefficient_sign=-1).project(
            np.array([-1.0, -2.0]),
            weight_matrix=weight,
        )


def test_indefinite_weight_matrix_is_rejected():
    qp = CausalProjection(coefficient_sign=-1)
    with pytest.raises(ValueError, match="positive semidefinite"):
        qp.project(
            np.array([-1.0, 1.0]),
            weight_matrix=np.array([[1.0, 2.0], [2.0, 1.0]]),
        )


def test_weight_matrix_shape_is_validated():
    qp = CausalProjection(coefficient_sign=-1)
    with pytest.raises(ValueError, match="weight_matrix shape"):
        qp.project(np.array([-1.0, 1.0]), weight_matrix=np.eye(3))


# ---------------------------------------------------------------------------
# Elastic (slack) equality mode
# ---------------------------------------------------------------------------


def test_elastic_matches_hard_on_feasible_problem():
    ref = np.array([-2.0, 1.0, -0.5])
    eq_matrix = np.ones(3)
    eq_target = np.array([-1.0])
    qp = CausalProjection(coefficient_sign=-1)

    hard = qp.project(ref, equality_matrix=eq_matrix, equality_target=eq_target)
    elastic = qp.project(
        ref,
        equality_matrix=eq_matrix,
        equality_target=eq_target,
        equality_penalty=np.array([1.0e6]),
    )

    assert hard.success and elastic.success
    assert elastic.elastic and not hard.elastic
    np.testing.assert_allclose(elastic.coefficients, hard.coefficients, atol=1.0e-5)
    assert elastic.equality_slack is not None
    np.testing.assert_allclose(elastic.equality_slack, 0.0, atol=1.0e-5)


def test_elastic_succeeds_where_hard_equality_is_infeasible():
    # sum(x) = +1 with x <= 0 is impossible: hard mode fails, elastic mode
    # returns the minimal-violation causal solution (sum -> 0, slack -> -1).
    ref = np.array([-0.5, -0.25])
    eq_matrix = np.ones(2)
    eq_target = np.array([1.0])
    qp = CausalProjection(coefficient_sign=-1)

    hard = qp.project(ref, equality_matrix=eq_matrix, equality_target=eq_target)
    assert not hard.success

    elastic = qp.project(
        ref,
        equality_matrix=eq_matrix,
        equality_target=eq_target,
        equality_penalty=np.array([1.0e6]),
    )
    assert elastic.success
    assert elastic.max_inequality_violation <= 1.0e-7
    assert np.max(elastic.coefficients) <= 1.0e-6
    # |slack| equals the distance from the target to the feasible cone.
    np.testing.assert_allclose(elastic.equality_slack, [-1.0], atol=1.0e-4)
    assert elastic.max_equality_residual == pytest.approx(1.0, abs=1.0e-4)


def test_elastic_penalty_validation():
    qp = CausalProjection(coefficient_sign=-1)
    ref = np.array([-1.0, -2.0])

    with pytest.raises(ValueError, match="equality_penalty requires"):
        qp.project(ref, equality_penalty=np.array([1.0]))
    with pytest.raises(ValueError, match="equality_penalty shape"):
        qp.project(
            ref,
            equality_matrix=np.ones(2),
            equality_target=np.array([-1.0]),
            equality_penalty=np.array([1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="positive and finite"):
        qp.project(
            ref,
            equality_matrix=np.ones(2),
            equality_target=np.array([-1.0]),
            equality_penalty=np.array([0.0]),
        )


def test_elastic_extension_matrices_are_consistent():
    from scipy.sparse import csc_matrix

    rank = 3
    p = csc_matrix(np.eye(rank))
    q = np.zeros(rank)
    g = csc_matrix(-np.eye(rank))
    h = np.zeros(rank)
    a = csc_matrix(np.arange(1.0, 1.0 + rank).reshape(1, -1))
    b = np.array([-1.0])
    mu = np.array([2.5])

    p_e, q_e, g_e, h_e, a_e, b_e = CausalProjection._elastic_extend(
        p, q, g, h, a, b, mu, rank
    )

    n_ext = rank + 2
    assert p_e.shape == (n_ext, n_ext)
    # PSD: the slack block is zero, the coefficient block is the original P.
    eigenvalues = np.linalg.eigvalsh(p_e.toarray())
    assert eigenvalues.min() >= -1.0e-12
    np.testing.assert_allclose(q_e, np.concatenate([q, mu, mu]))
    assert g_e.shape == (rank + 2, n_ext)
    np.testing.assert_allclose(h_e, np.zeros(rank + 2))
    # A_ext = [A, -I, +I] encodes A x - b = s+ - s-.
    np.testing.assert_allclose(
        a_e.toarray(), np.hstack([a.toarray(), [[-1.0]], [[1.0]]])
    )
    np.testing.assert_allclose(b_e, b)


def _available_solvers() -> set:
    try:
        from qpsolvers import available_solvers
    except ImportError:  # pragma: no cover
        return set()
    return set(available_solvers)


@pytest.mark.parametrize("solver", CausalProjection.DEFAULT_SOLVERS)
def test_elastic_solver_smoke(solver):
    # Shape/plumbing smoke per solver: first-order solvers (osqp, scs) meet a
    # looser accuracy on the badly scaled mu=1e6 problem, so judge with a
    # relaxed per-call tol; precision is covered by the cascade tests above.
    if solver not in _available_solvers():
        pytest.skip(f"solver {solver!r} is not installed")
    result = CausalProjection(coefficient_sign=-1, solvers=(solver,)).project(
        np.array([-0.5, -0.25]),
        equality_matrix=np.ones(2),
        equality_target=np.array([1.0]),
        equality_penalty=np.array([1.0e6]),
        tol=1.0e-3,
    )
    assert result.success
    assert result.max_inequality_violation <= 1.0e-3
    np.testing.assert_allclose(result.equality_slack, [-1.0], atol=5.0e-3)
