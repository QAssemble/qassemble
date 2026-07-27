"""Quadratic programming utilities for coefficient-space causal projection."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qpsolvers import SolverNotFound, solve_qp
from scipy.sparse import csc_matrix

from .Common import Common
from .Fourier import Fourier


@dataclass(frozen=True)
class CausalQPResult:
    """Result returned by a causal coefficient QP."""

    coefficients: np.ndarray
    success: bool
    solver: str | None
    status: str
    attempts: tuple[str, ...]
    objective: float
    max_inequality_violation: float
    max_equality_residual: float
    relative_change: float
    equality_slack: np.ndarray | None = None
    elastic: bool = False


@dataclass(frozen=True)
class CausalCheckResult:
    """Result returned by a causal coefficient feasibility check.

    ``node_residual`` is filled by the kernel-aware projector ``check`` methods
    as a data-quality diagnostic; the pure-QP ``CausalProjection.check`` leaves
    it ``None``.
    """

    causal: bool
    max_inequality_violation: float
    max_equality_residual: float
    violating_count: int
    node_residual: float | None = None
    c0: float = 0.0


class CausalProjection:
    """Coefficient-space QP for scalar causal optimization.

    This class implements scalar causal optimization for one diagonal channel.
    It does not enforce full matrix-valued causality.  It is a pure QP solver:
    kernels, moments, and statistics live in the caller.  (Distinct from the
    ``CausalProjection`` *methods* on ``FLatDyn``/``FLocDyn`` — this is a
    module-level class, those are class attributes; no namespace collision.)

    The optimized variable is a real coefficient vector ``x``.  The solved
    problem is

        min_x (x - reference)^T W (x - reference)

    subject to a componentwise causal sign constraint and optional linear
    equality constraints.  ``W`` defaults to the identity metric.

    ``coefficient_sign=-1`` enforces ``x <= 0``.
    ``coefficient_sign=+1`` enforces ``x >= 0``.
    """

    DEFAULT_SOLVERS = ("clarabel", "osqp", "proxqp", "scs")

    def __init__(
        self,
        *,
        coefficient_sign: int = -1,
        solvers: Sequence[str] | None = None,
        max_iter: int = 100000,
        constraint_tol: float = 1.0e-8,
        imag_atol: float = 1.0e-10,
        imag_rtol: float = 1.0e-8,
        raise_on_failure: bool = False,
    ) -> None:
        if coefficient_sign not in (-1, 1):
            raise ValueError("coefficient_sign must be -1 or +1")
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
            raise ValueError("max_iter must be a positive integer")
        if int(max_iter) <= 0:
            raise ValueError("max_iter must be a positive integer")
        if constraint_tol <= 0.0 or not np.isfinite(constraint_tol):
            raise ValueError("constraint_tol must be a positive finite number")
        if imag_atol < 0.0 or not np.isfinite(imag_atol):
            raise ValueError("imag_atol must be a nonnegative finite number")
        if imag_rtol < 0.0 or not np.isfinite(imag_rtol):
            raise ValueError("imag_rtol must be a nonnegative finite number")

        if solvers is None:
            solver_tuple = self.DEFAULT_SOLVERS
        else:
            solver_tuple = tuple(solvers)
            if len(solver_tuple) == 0:
                raise ValueError("solvers must be nonempty")
            if not all(isinstance(solver, str) and solver for solver in solver_tuple):
                raise ValueError("solvers must contain nonempty solver names")

        self.coefficient_sign = int(coefficient_sign)
        self.solvers = solver_tuple
        self.max_iter = int(max_iter)
        self.constraint_tol = float(constraint_tol)
        self.imag_atol = float(imag_atol)
        self.imag_rtol = float(imag_rtol)
        self.raise_on_failure = bool(raise_on_failure)

    def project(
        self,
        reference: np.ndarray,
        equality_matrix: np.ndarray | None = None,
        equality_target: np.ndarray | None = None,
        weight_matrix: np.ndarray | None = None,
        tol: float | None = None,
        *,
        equality_penalty: np.ndarray | None = None,
    ) -> CausalQPResult:
        """Project a reference coefficient vector onto the causal QP set.

        Parameters
        ----------
        reference
            Reference DLR coefficient vector for one diagonal scalar channel.
            Complex input is accepted only when the imaginary part is within
            the configured numerical tolerance.
        equality_matrix, equality_target
            Optional constraints ``equality_matrix @ x = equality_target``.
        weight_matrix
            Optional PSD metric ``W`` for the objective
            ``(x - reference)^T W (x - reference)``; ``None`` keeps the
            identity metric.  The matrix is symmetrized and eigenvalue-checked;
            slightly negative eigenvalues are clipped to zero and the rewritten
            ``W`` is used consistently in the QP, the solver scoring, and the
            reported objective.
        tol
            Optional per-call constraint tolerance for solver acceptance and
            the success judgment; ``None`` uses ``constraint_tol``.
        equality_penalty
            Optional positive per-row penalties ``mu`` that soften the equality
            constraints into an exact L1 penalty (elastic mode).  The solved
            problem becomes ``min (x-ref)^T W (x-ref) + sum_p mu_p |A x - b|_p``
            via a positive slack pair ``s+/s-`` with ``A x - b = s+ - s-``.
            When the hard-equality problem is feasible and ``mu`` is large
            enough, the elastic solution coincides with the hard one (exact
            penalty); when it is infeasible, the minimal weighted violation is
            returned instead of failing.  In elastic mode ``success`` judges
            only the causal sign constraint, slack positivity, and the solver's
            accuracy on the extended system — the raw data equality residual
            ``|A x - b|`` (== ``|equality_slack|``) is *excluded* from the
            success judgment and reported via ``max_equality_residual`` /
            ``equality_slack`` for diagnostics only.
        """

        ref = Common.RealVector(reference, self.imag_atol, self.imag_rtol)
        rank = ref.size
        tol_val = self._resolve_tol(tol)
        weight = self._prepare_weight_matrix(weight_matrix, rank)

        p_unscaled = 2.0 * weight
        p_scale = max(float(np.linalg.norm(p_unscaled, ord=2)), 1.0)
        p = csc_matrix(p_unscaled / p_scale)
        q = -2.0 * (weight @ ref) / p_scale

        g = csc_matrix(-self.coefficient_sign * np.eye(rank))
        h = np.zeros(rank, dtype=float)
        a, b = self._equality_constraint(
            equality_matrix,
            equality_target,
            rank,
            self.imag_atol,
            self.imag_rtol,
        )

        elastic = equality_penalty is not None
        if elastic:
            if a is None:
                raise ValueError(
                    "equality_penalty requires equality_matrix/equality_target"
                )
            mu = np.asarray(equality_penalty, dtype=float).reshape(-1)
            if mu.shape != (b.size,):
                raise ValueError(
                    f"equality_penalty shape {mu.shape} does not match "
                    f"{(b.size,)}"
                )
            if not np.all(np.isfinite(mu)) or np.any(mu <= 0.0):
                raise ValueError("equality_penalty must be positive and finite")
            m_rows = b.size
            p_s, q_s, g_s, h_s, a_s, b_s = self._elastic_extend(
                p, q, g, h, a, b, mu / p_scale, rank
            )
            ref_s = np.concatenate([ref, np.zeros(2 * m_rows)])
            weight_s = np.zeros((rank + 2 * m_rows, rank + 2 * m_rows))
            weight_s[:rank, :rank] = weight
        else:
            p_s, q_s, g_s, h_s, a_s, b_s = p, q, g, h, a, b
            ref_s, weight_s = ref, weight

        sol, solver, status, attempts = self._solve_qp(
            p_s,
            q_s,
            g_s,
            h_s,
            a_s,
            b_s,
            ref_s,
            weight_s,
            tol_val,
        )

        if sol is None:
            if self.raise_on_failure:
                raise RuntimeError(
                    "causal QP failed for all configured solvers: "
                    + ", ".join(attempts)
                )
            failed_coeff = ref.copy()
            return CausalQPResult(
                coefficients=failed_coeff,
                success=False,
                solver=solver,
                status=status,
                attempts=tuple(attempts),
                objective=self._weighted_objective(failed_coeff, ref, weight),
                max_inequality_violation=self._max_inequality_violation(g, h, failed_coeff),
                max_equality_residual=self._max_equality_residual(a, b, failed_coeff),
                relative_change=0.0,
                equality_slack=None,
                elastic=elastic,
            )

        if elastic:
            coeff = np.asarray(sol[:rank], dtype=float)
            slack = np.asarray(
                sol[rank:rank + m_rows] - sol[rank + m_rows:], dtype=float
            )
            # Elastic success: causal sign + slack positivity + the solver's
            # accuracy on the extended equality system.  The raw data equality
            # residual |A x - b| is the *optimal* violation and is reported
            # only (max_equality_residual / equality_slack), never judged.
            ineq = self._max_inequality_violation(g_s, h_s, sol)
            eq_solver = self._max_equality_residual(a_s, b_s, sol)
            eq = self._max_equality_residual(a, b, coeff)
            success = bool(
                ineq <= tol_val
                and eq_solver <= tol_val
                and np.all(np.isfinite(coeff))
            )
            if self.raise_on_failure and not success:
                raise RuntimeError(
                    "elastic causal QP returned an inaccurate candidate: "
                    f"ineq={ineq:.3e}, eq_solver={eq_solver:.3e}"
                )
        else:
            coeff = np.asarray(sol, dtype=float)
            slack = None
            ineq = self._max_inequality_violation(g, h, coeff)
            eq = self._max_equality_residual(a, b, coeff)
            success = bool(ineq <= tol_val and eq <= tol_val)
            if self.raise_on_failure and not success:
                raise RuntimeError(
                    "causal QP returned a constraint-violating candidate: "
                    f"ineq={ineq:.3e}, eq={eq:.3e}"
                )

        return CausalQPResult(
            coefficients=coeff,
            success=success,
            solver=solver,
            status=status,
            attempts=tuple(attempts),
            objective=self._weighted_objective(coeff, ref, weight),
            max_inequality_violation=ineq,
            max_equality_residual=eq,
            relative_change=self._relative_change(coeff, ref),
            equality_slack=slack,
            elastic=elastic,
        )

    def check(
        self,
        coefficients: np.ndarray,
        *,
        equality_matrix: np.ndarray | None = None,
        equality_target: np.ndarray | None = None,
        tol: float | None = None,
    ) -> CausalCheckResult:
        """Check sign and equality feasibility of coefficients without projecting.

        Parameters
        ----------
        coefficients
            Real coefficient vector to test against the causal sign constraint.
        equality_matrix, equality_target
            Optional constraints ``equality_matrix @ x = equality_target``.
        tol
            Optional per-call tolerance for the causal judgment; ``None`` uses
            ``constraint_tol``.
        """

        coeff = Common.RealVector(coefficients, self.imag_atol, self.imag_rtol)
        rank = coeff.size
        tol_val = self._resolve_tol(tol)
        a, b = self._equality_constraint(
            equality_matrix,
            equality_target,
            rank,
            self.imag_atol,
            self.imag_rtol,
        )

        sign_residual = -self.coefficient_sign * coeff
        ineq = float(max(np.max(sign_residual), 0.0))
        violating_count = int(np.count_nonzero(sign_residual > tol_val))
        eq = self._max_equality_residual(a, b, coeff)

        return CausalCheckResult(
            causal=bool(ineq <= tol_val and eq <= tol_val),
            max_inequality_violation=ineq,
            max_equality_residual=eq,
            violating_count=violating_count,
            c0=0.0,
        )

    def _resolve_tol(self, tol: float | None) -> float:
        if tol is None:
            return self.constraint_tol
        tol_val = float(tol)
        if tol_val <= 0.0 or not np.isfinite(tol_val):
            raise ValueError("tol must be a positive finite number")
        return tol_val

    def _prepare_weight_matrix(
        self,
        weight_matrix: np.ndarray | None,
        rank: int,
    ) -> np.ndarray:
        if weight_matrix is None:
            return np.eye(rank, dtype=float)

        weight = Common.RealArray(
            weight_matrix,
            name="weight_matrix",
            imag_atol=self.imag_atol,
            imag_rtol=self.imag_rtol,
        )
        if weight.ndim != 2 or weight.shape != (rank, rank):
            raise ValueError(
                f"weight_matrix shape {weight.shape} does not match {(rank, rank)}"
            )
        weight = 0.5 * (weight + weight.T)
        eigenvalues, eigenvectors = np.linalg.eigh(weight)
        lam_min = float(eigenvalues[0])
        lam_max = float(eigenvalues[-1])
        if lam_min < -1.0e-10 * lam_max:
            raise ValueError(
                "weight_matrix must be positive semidefinite: "
                f"lambda_min={lam_min:.3e}, lambda_max={lam_max:.3e}"
            )
        clipped = (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
        return 0.5 * (clipped + clipped.T)

    def _solve_qp(
        self,
        p,
        q,
        g,
        h,
        a,
        b,
        reference: np.ndarray,
        weight: np.ndarray,
        tol_val: float,
    ):
        attempts = []
        best = None
        for solver in self.solvers:
            kwargs = self._solver_kwargs(solver, self.max_iter)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    sol = solve_qp(p, q, g, h, a, b, solver=solver, **kwargs)
            except SolverNotFound:
                attempts.append(f"{solver}:not_found")
                continue
            except Exception as exc:
                attempts.append(f"{solver}:exception={type(exc).__name__}")
                continue

            if sol is None or not np.all(np.isfinite(sol)):
                attempts.append(f"{solver}:none")
                continue

            sol = np.asarray(sol, dtype=float)
            ineq = self._max_inequality_violation(g, h, sol)
            eq = self._max_equality_residual(a, b, sol)
            objective = self._weighted_objective(sol, reference, weight)
            status = "|".join(str(item.message).replace(" ", "_") for item in caught) or "ok"
            attempts.append(f"{solver}:ineq={ineq:.3e}:eq={eq:.3e}")

            score = (max(ineq, eq), objective)
            if best is None or score < best[0]:
                best = (score, sol, solver, status, list(attempts))
            if ineq <= tol_val and eq <= tol_val:
                break

        if best is None:
            return None, None, "all_failed", attempts
        _, sol, solver, status, best_attempts = best
        return sol, solver, status, best_attempts

    @staticmethod
    def _elastic_extend(p, q, g, h, a, b, mu, rank):
        """Extend a hard-equality QP with a positive slack pair per row.

        The variable becomes ``x = [coeff(rank); s+(m); s-(m)]`` with

            P_ext = blockdiag(P, 0, 0)          (PSD: slack block is zero)
            q_ext = [q; mu; mu]                 (exact L1 penalty sum mu (s+ + s-))
            G_ext = [[G, 0, 0], [0, -I, 0], [0, 0, -I]]   (s+, s- >= 0)
            A_ext = [A, -I, +I]                 (A coeff - b = s+ - s-)

        ``mu`` must already carry the caller's objective scaling (the fermion
        path passes ``mu / p_scale`` so the penalty stays commensurate with the
        scaled quadratic term).  Returns ``(p, q, g, h, a, b)`` of the extended
        system as csc matrices / dense vectors.
        """
        a_dense = np.asarray(a.todense() if hasattr(a, "todense") else a, dtype=float)
        m = a_dense.shape[0]
        n = int(rank)
        eye_m = np.eye(m)

        p_dense = np.asarray(p.todense() if hasattr(p, "todense") else p, dtype=float)
        p_ext = np.zeros((n + 2 * m, n + 2 * m))
        p_ext[:n, :n] = p_dense

        q_ext = np.concatenate([np.asarray(q, dtype=float), mu, mu])

        g_dense = np.asarray(g.todense() if hasattr(g, "todense") else g, dtype=float)
        g_ext = np.zeros((g_dense.shape[0] + 2 * m, n + 2 * m))
        g_ext[: g_dense.shape[0], :n] = g_dense
        g_ext[g_dense.shape[0]: g_dense.shape[0] + m, n:n + m] = -eye_m
        g_ext[g_dense.shape[0] + m:, n + m:] = -eye_m
        h_ext = np.concatenate([np.asarray(h, dtype=float), np.zeros(2 * m)])

        a_ext = np.hstack([a_dense, -eye_m, eye_m])
        return (
            csc_matrix(p_ext),
            q_ext,
            csc_matrix(g_ext),
            h_ext,
            csc_matrix(a_ext),
            np.asarray(b, dtype=float),
        )

    @staticmethod
    def _equality_constraint(
        equality_matrix: np.ndarray | None,
        equality_target: np.ndarray | None,
        rank: int,
        imag_atol: float,
        imag_rtol: float,
    ) -> tuple[csc_matrix | None, np.ndarray | None]:
        if equality_matrix is None and equality_target is None:
            return None, None
        if equality_matrix is None or equality_target is None:
            raise ValueError("equality_matrix and equality_target must be provided together")

        matrix = Common.RealArray(
            equality_matrix,
            name="equality_matrix",
            imag_atol=imag_atol,
            imag_rtol=imag_rtol,
        )
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError(f"equality_matrix must be 1D or 2D, got {matrix.ndim}D")
        if matrix.shape[1] != rank:
            raise ValueError(
                f"equality_matrix has {matrix.shape[1]} columns, expected {rank}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("equality_matrix contains non-finite values")

        target = Common.RealArray(
            equality_target,
            name="equality_target",
            imag_atol=imag_atol,
            imag_rtol=imag_rtol,
        ).reshape(-1)
        if target.shape != (matrix.shape[0],):
            raise ValueError(
                f"equality_target shape {target.shape} does not match "
                f"{(matrix.shape[0],)}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("equality_target contains non-finite values")

        return csc_matrix(matrix), target

    @staticmethod
    def _solver_kwargs(solver: str, max_iter: int) -> dict:
        if solver == "scs":
            return {"max_iters": max_iter}
        return {"max_iter": max_iter}

    @staticmethod
    def _weighted_objective(
        coefficients: np.ndarray,
        reference: np.ndarray,
        weight: np.ndarray,
    ) -> float:
        diff = np.asarray(coefficients, dtype=float) - reference
        return float(diff @ (weight @ diff))

    @staticmethod
    def _max_inequality_violation(g, h, coefficients: np.ndarray) -> float:
        residual = np.asarray(g @ coefficients - h, dtype=float).reshape(-1)
        return float(max(np.max(residual), 0.0))

    @staticmethod
    def _max_equality_residual(a, b, coefficients: np.ndarray) -> float:
        if a is None:
            return 0.0
        residual = np.asarray(a @ coefficients - b, dtype=float).reshape(-1)
        return float(np.max(np.abs(residual)))

    @staticmethod
    def _relative_change(coefficients: np.ndarray, reference: np.ndarray) -> float:
        return float(
            np.linalg.norm(np.asarray(coefficients, dtype=float) - reference)
            / max(np.linalg.norm(reference), 1.0e-30)
        )


class CausalBosonProjector:
    """Bosonic DLR pole-weight causal projection using an existing DLR basis.

    This is the scalar bosonic projector used by ``CausalProjector``.  It mirrors
    the pole-QP construction from ``causal_boson.py`` but reuses the caller's
    existing bosonic DLR object instead of creating a new ``pydlr.dlr`` basis.
    """

    DEFAULT_SOLVERS = CausalProjection.DEFAULT_SOLVERS

    def __init__(
        self,
        *,
        d,
        beta: float,
        fit_omega: np.ndarray,
        output_omega: np.ndarray | None = None,
        coefficient_sign: int = -1,
        reflection_symmetry: bool = True,
        solvers: Sequence[str] | None = None,
        max_iter: int = 100000,
        constraint_tol: float | str = 1.0e-8,
        auto_safety: float = 10.0,
        auto_floor: float = 1.0e-8,
        fit_tol: float = 1.0e-6,
        tail_tol: float = 1.0e-1,
        tail_points: int = 5,
        regularization: float = 0.0,
        moment_sigma_safety: float = 3.0,
        raise_on_failure: bool = True,
    ) -> None:
        if coefficient_sign not in (-1, 1):
            raise ValueError("coefficient_sign must be -1 or +1")
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError("beta must be a positive finite number")
        if fit_tol <= 0.0 or not np.isfinite(fit_tol):
            raise ValueError("fit_tol must be a positive finite number")
        if tail_tol <= 0.0 or not np.isfinite(tail_tol):
            raise ValueError("tail_tol must be a positive finite number")
        if moment_sigma_safety <= 0.0 or not np.isfinite(moment_sigma_safety):
            raise ValueError("moment_sigma_safety must be a positive finite number")
        if regularization < 0.0 or not np.isfinite(regularization):
            raise ValueError("regularization must be a nonnegative finite number")
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
            raise ValueError("max_iter must be a positive integer")
        if int(max_iter) <= 0:
            raise ValueError("max_iter must be a positive integer")
        if isinstance(tail_points, bool) or not isinstance(
            tail_points, (int, np.integer)
        ):
            raise ValueError("tail_points must be an integer >= 4")
        if int(tail_points) < 4:
            raise ValueError("tail_points must be an integer >= 4")

        self._auto_tol = isinstance(constraint_tol, str)
        if self._auto_tol:
            if constraint_tol != "auto":
                raise ValueError("constraint_tol string must be 'auto'")
            if auto_safety <= 0.0 or not np.isfinite(auto_safety):
                raise ValueError("auto_safety must be a positive finite number")
            if auto_floor <= 0.0 or not np.isfinite(auto_floor):
                raise ValueError("auto_floor must be a positive finite number")
            qp_constraint_tol = float(auto_floor)
        else:
            qp_constraint_tol = float(constraint_tol)
            if qp_constraint_tol <= 0.0 or not np.isfinite(qp_constraint_tol):
                raise ValueError("constraint_tol must be a positive finite number")

        if solvers is None:
            solver_tuple = self.DEFAULT_SOLVERS
        else:
            solver_tuple = tuple(solvers)
            if len(solver_tuple) == 0:
                raise ValueError("solvers must be nonempty")
            if not all(isinstance(solver, str) and solver for solver in solver_tuple):
                raise ValueError("solvers must contain nonempty solver names")

        self.statistic = "B"
        self.d = d
        self.beta = float(beta)
        self.fit_omega = Common.RealFrequencyVector(fit_omega, name="fit_omega")
        self.output_omega = (
            self.fit_omega.copy()
            if output_omega is None
            else Common.RealFrequencyVector(output_omega, name="output_omega")
        )
        if self.fit_omega.size == 0 or self.output_omega.size == 0:
            raise ValueError("fit_omega and output_omega must be nonempty")

        x_nodes = np.asarray(d.get_dlr_frequencies(), dtype=float)
        if x_nodes.ndim != 1 or x_nodes.size == 0:
            raise ValueError("d must provide a nonempty one-dimensional DLR grid")
        if not np.all(np.isfinite(x_nodes)):
            raise ValueError("DLR real-frequency nodes contain non-finite values")

        self.x_nodes = x_nodes
        self.nodes = x_nodes / self.beta
        self.rank = int(x_nodes.size)
        self.omega = self.nodes
        self.bose_corr = np.tanh(0.5 * self.x_nodes)

        self.coefficient_sign = int(coefficient_sign)
        self.reflection_symmetry = bool(reflection_symmetry)
        self.solvers = solver_tuple
        self.max_iter = int(max_iter)
        self.constraint_tol = qp_constraint_tol
        self.auto_safety = float(auto_safety)
        self.auto_floor = float(auto_floor)
        self.fit_tol = float(fit_tol)
        self.tail_tol = float(tail_tol)
        self.tail_points = int(tail_points)
        self.regularization = float(regularization)
        self.moment_sigma_safety = float(moment_sigma_safety)
        self.raise_on_failure = bool(raise_on_failure)
        if self.tail_points > self.fit_omega.size:
            raise ValueError(
                f"tail_points {self.tail_points} exceeds the fit grid size "
                f"{self.fit_omega.size}"
            )

        self.kernel = self._frequency_kernel(self.fit_omega)
        self.output_kernel = self._frequency_kernel(self.output_omega)
        self.moment_rows = self._moment_rows()
        self.qp = CausalProjection(
            coefficient_sign=self.coefficient_sign,
            solvers=self.solvers,
            max_iter=self.max_iter,
            constraint_tol=self.constraint_tol,
            raise_on_failure=self.raise_on_failure,
        )

        self.last_coefficients: np.ndarray | None = None
        self.last_solver: str | None = None
        self.last_status: str | None = None
        self.last_solver_status: str | None = None
        self.last_attempts: tuple[str, ...] = ()
        self.last_solver_attempts: list[str] = []
        self.last_validation: dict[str, float | bool] = {}

    def project(
        self,
        target: np.ndarray,
        *,
        tail_coeffs: np.ndarray | None = None,
        moments: dict[str, float] | None = None,
        scale: float | None = None,
        enforce_moments: bool = True,
        moment_sigma: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project one bosonic scalar channel and return it on ``output_omega``.

        ``enforce_moments=False`` drops the moment equality rows from the QP
        (the sign constraint alone can be incompatible with an off-diagonal
        moment target); moment residuals are then recorded as diagnostics only.

        ``moment_sigma`` (unscaled one-sigma uncertainties ``[sigma_c1,
        sigma_c2]`` of the physical tail fit — the sign flip to the internal
        m1/m2 convention leaves sigma unchanged) enables the elastic retry:
        when the hard moment equalities fail, they are softened into an exact
        L1 penalty with ``mu = 1/sigma`` so the QP can never be infeasible.

        A non-decaying component surviving the ``c0`` subtraction (CTQMC noise
        entering through the Dyson equation) is measured against ``tail_tol``;
        instead of raising, its constant offset is estimated from the largest
        ``|nu|`` nodes, subtracted before the QP, and DISCARDED from the output
        (only the physical ``c0`` is re-added — with ``highzero`` that is 0, so
        the output always decays).  The dropped magnitude is reported via a
        ``RuntimeWarning`` and ``last_validation["c0_dropped"]``.

        Last resort: if every solver crashes (hard and elastic), the
        regularized-fit clipped fallback returns a guaranteed-causal channel
        instead of the raw non-causal data.
        """

        self._reset_diagnostics()
        target_vec = Common.ComplexVector(
            target,
            name="target",
            expected_size=self.fit_omega.size,
        )
        scale_val = self._scale(target_vec) if scale is None else float(scale)
        if scale_val <= 0.0 or not np.isfinite(scale_val):
            raise ValueError("scale must be a positive finite number")

        target_scaled = target_vec / scale_val
        c0, moment_target = self._prepare_moments(
            target_scaled,
            tail_coeffs=tail_coeffs,
            moments=moments,
            scale=scale_val,
        )
        target_dec = target_scaled - c0
        tail_ratio = self._validate_target(target_dec)
        c0_dropped = 0.0
        if tail_ratio > self.tail_tol:
            tail_indices = np.argsort(np.abs(self.fit_omega))[-2:]
            c0_eff = float(np.mean(np.real(target_dec[tail_indices])))
            target_dec = target_dec - c0_eff
            c0_dropped = float(c0_eff * scale_val)
            warnings.warn(
                "bosonic target minus c0 does not decay at the largest |nu| "
                f"nodes (|tail|/max = {tail_ratio:.3e} > tail_tol "
                f"{self.tail_tol:.1e}); a non-decaying offset of magnitude "
                f"{abs(c0_dropped):.3e} was subtracted and DISCARDED from the "
                "output",
                RuntimeWarning,
                stacklevel=2,
            )
        _, node_residual = self._fit_coefficients(target_dec)
        eff_tol, tol_scaled = self._effective_tol(node_residual, scale_val)

        p, q, p_scale = self._objective(target_dec)
        g = csc_matrix(-self.coefficient_sign * np.eye(self.rank))
        h = np.zeros(self.rank, dtype=float)
        if enforce_moments:
            a = csc_matrix(self.moment_rows)
            b = self._moment_vector(moment_target)
        else:
            a = None
            b = None

        sigma_scaled = None
        mu = None
        if moment_sigma is not None and a is not None:
            sigma_scaled, mu = self._moment_penalty(
                b, moment_sigma, scale_val, node_residual
            )

        equality_slack = None
        clipped = False
        elastic_used = False
        solution = self._solve_qp(p, q, g, h, a, b, tol_scaled)
        hard_ok = solution is not None and self._solution_ok(
            solution, g, h, a, b, tol_scaled
        )
        if not hard_ok and mu is not None:
            # hard moments infeasible (or solvers rejected them): elastic
            # exact-L1 retry — its feasible set is never empty.
            p_e, q_e, g_e, h_e, a_e, b_e = CausalProjection._elastic_extend(
                p, q, g, h, a, b, mu / p_scale, self.rank
            )
            sol_ext = self._solve_qp(p_e, q_e, g_e, h_e, a_e, b_e, tol_scaled)
            if sol_ext is not None:
                m_rows = b.size
                solution = np.asarray(sol_ext[: self.rank], dtype=float)
                equality_slack = (
                    sol_ext[self.rank : self.rank + m_rows]
                    - sol_ext[self.rank + m_rows :]
                ) * scale_val
                elastic_used = True
        if solution is None:
            # every solver crashed on every attempt.  With raise_on_failure the
            # error propagates (ProjectBosonComponentWithFallback catches it
            # and applies the clipped fallback there); without it the clipped
            # fallback replaces the historical raw non-causal return.
            if self.raise_on_failure:
                raise RuntimeError(
                    "bosonic causal QP failed for all configured solvers: "
                    + ", ".join(self.last_attempts)
                )
            solution = self._clipped_fallback_coefficients(target_dec)
            clipped = True
            distortion = float(
                np.linalg.norm(self.kernel @ solution - target_dec)
                / max(np.linalg.norm(target_dec), 1.0e-30)
            )
            warnings.warn(
                "bosonic causal QP failed for all configured solvers; "
                "returning the regularized-fit clipped fallback (guaranteed "
                f"causal, relative distortion {distortion:.3e})",
                RuntimeWarning,
                stacklevel=2,
            )

        coefficients = np.asarray(solution * scale_val, dtype=float)
        c0_unscaled = float(c0 * scale_val)
        self.last_coefficients = coefficients
        self.last_validation = self._validate(
            coefficients,
            target_vec - c0_unscaled - c0_dropped,
            self._moment_vector(moment_target) * scale_val,
            node_residual=node_residual,
            skipped=False,
            c0=c0_unscaled,
            constraint_tol=eff_tol,
            enforce_moments=enforce_moments,
            moment_sigma_unscaled=(
                sigma_scaled * scale_val if sigma_scaled is not None else None
            ),
            elastic=elastic_used,
            equality_slack=equality_slack,
            clipped=clipped,
            c0_dropped=c0_dropped,
            tail_ratio=tail_ratio,
        )
        if self.raise_on_failure and not bool(self.last_validation["valid"]):
            raise RuntimeError(
                "bosonic causal projection returned a constraint-violating "
                f"candidate: coeff={self.last_validation['coefficient_violation']:.3e}, "
                f"moment={self.last_validation['moment_residual']:.3e}"
            )
        # only the physical c0 is re-added; a dropped non-decaying offset never
        # re-enters the output
        return self.output_kernel @ coefficients + c0_unscaled

    def check(
        self,
        target: np.ndarray,
        *,
        enforce_gate: bool = True,
        tail_coeffs: np.ndarray | None = None,
        moments: dict[str, float] | None = None,
        moment_sigma: np.ndarray | None = None,
    ) -> CausalCheckResult:
        """Check the least-squares pole representative without projecting.

        Mirrors ``project``: a non-decaying offset surviving the ``c0``
        subtraction is estimated and removed before the fit (never raises),
        and ``moment_sigma`` switches the moment verdict to sigma units.
        """

        target_vec = Common.ComplexVector(
            target,
            name="target",
            expected_size=self.fit_omega.size,
        )
        scale = self._scale(target_vec)
        target_scaled = target_vec / scale
        c0, moment_target = self._prepare_moments(
            target_scaled,
            tail_coeffs=tail_coeffs,
            moments=moments,
            scale=scale,
        )
        target_dec = target_scaled - c0
        tail_ratio = self._validate_target(target_dec)
        if tail_ratio > self.tail_tol:
            tail_indices = np.argsort(np.abs(self.fit_omega))[-2:]
            target_dec = target_dec - float(
                np.mean(np.real(target_dec[tail_indices]))
            )
        reference, node_residual = self._fit_coefficients(target_dec)
        if enforce_gate:
            self._gate(node_residual)
        _, tol_scaled = self._effective_tol(node_residual, scale)
        moment_vector = self._moment_vector(moment_target)
        verdict = self.qp.check(
            reference,
            equality_matrix=self.moment_rows,
            equality_target=moment_vector,
            tol=tol_scaled,
        )
        causal = verdict.causal
        if moment_sigma is not None:
            sigma_scaled, _ = self._moment_penalty(
                moment_vector, moment_sigma, scale, node_residual
            )
            sign_residual = -self.coefficient_sign * reference
            sign_ok = float(max(np.max(sign_residual), 0.0)) <= tol_scaled
            ratio = (
                np.abs(self.moment_rows @ reference - moment_vector)
                / sigma_scaled
            )
            causal = bool(
                sign_ok and float(np.max(ratio)) <= self.moment_sigma_safety
            )
        return CausalCheckResult(
            causal=causal,
            max_inequality_violation=verdict.max_inequality_violation * scale,
            max_equality_residual=verdict.max_equality_residual * scale,
            violating_count=verdict.violating_count,
            node_residual=node_residual,
            c0=float(c0 * scale),
        )

    def reconstruct_frequency(self, coefficients: np.ndarray, nu: np.ndarray) -> np.ndarray:
        return self._frequency_kernel(Common.RealFrequencyVector(nu, name="nu")) @ np.asarray(
            coefficients, dtype=float
        )

    def _reset_diagnostics(self) -> None:
        self.last_coefficients = None
        self.last_solver = None
        self.last_status = None
        self.last_solver_status = None
        self.last_attempts = ()
        self.last_solver_attempts = []
        self.last_validation = {}

    @staticmethod
    def _scale(target: np.ndarray) -> float:
        return float(max(np.max(np.abs(target)), 1.0))

    def _basis_freq(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.complex128)
        basis = np.empty((z.size, self.rank), dtype=np.complex128)
        for j in range(self.rank):
            unit = np.zeros((self.rank, 1, 1), dtype=np.complex128)
            unit[j, 0, 0] = 1.0
            basis[:, j] = self.d.eval_dlr_freq(
                unit,
                z,
                self.beta,
                xi=1,
            )[:, 0, 0]
        return basis

    def _frequency_kernel(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        if self.reflection_symmetry:
            return 0.5 * (self._basis_freq(1j * omega) + self._basis_freq(-1j * omega))
        return self._basis_freq(1j * omega)

    def _moment_rows(self) -> np.ndarray:
        row_m2 = self.bose_corr * self.nodes
        if self.reflection_symmetry:
            return row_m2.reshape(1, -1)
        return np.vstack([self.bose_corr, row_m2])

    def _tail_coefficients(
        self,
        target_scaled: np.ndarray,
        *,
        tail_coeffs: np.ndarray | None,
        scale: float,
    ) -> np.ndarray:
        if tail_coeffs is not None:
            c = np.asarray(tail_coeffs, dtype=float).reshape(-1)
            if c.shape != (4,):
                raise ValueError(f"tail_coeffs must have shape (4,), got {c.shape}")
            if not np.all(np.isfinite(c)):
                raise ValueError("tail_coeffs contains non-finite values")
            return c / scale
        return Fourier.BosonTailCoefficients(
            self.fit_omega,
            target_scaled,
            self.tail_points,
        )

    def _prepare_moments(
        self,
        target_scaled: np.ndarray,
        *,
        tail_coeffs: np.ndarray | None,
        moments: dict[str, float] | None,
        scale: float,
    ) -> tuple[float, dict[str, float]]:
        if moments is not None:
            c0 = 0.0
            if tail_coeffs is not None:
                c0 = float(
                    self._tail_coefficients(
                        target_scaled,
                        tail_coeffs=tail_coeffs,
                        scale=scale,
                    )[0]
                )
            return c0, {
                "m1": float(moments.get("m1", 0.0)) / scale,
                "m2": float(moments.get("m2", 0.0)) / scale,
            }
        c = self._tail_coefficients(
            target_scaled,
            tail_coeffs=tail_coeffs,
            scale=scale,
        )
        c0 = float(c[0])
        # QAssemble tail coefficients are in the physical
        # c1/(i nu) + c2/(i nu)^2 convention.  The DLR pole rows encode the
        # internally signed moments used by the pole-weight constraints.
        return c0, {"m1": -float(c[1]), "m2": -float(c[2])}

    def _moment_vector(self, moments: dict[str, float]) -> np.ndarray:
        if self.reflection_symmetry:
            return np.array([moments["m2"]], dtype=float)
        return np.array([moments["m1"], moments["m2"]], dtype=float)

    def _effective_tol(self, node_residual: float, scale: float) -> tuple[float, float]:
        if not self._auto_tol:
            return self.constraint_tol, self.constraint_tol / scale
        eff_scaled = max(self.auto_floor / scale, self.auto_safety * node_residual)
        return eff_scaled * scale, eff_scaled

    def _objective(self, target_scaled: np.ndarray):
        p_dense = 2.0 * np.real(self.kernel.conj().T @ self.kernel)
        if self.regularization > 0.0:
            p_dense = p_dense + self.regularization * np.eye(self.rank)
        p_scale = max(float(np.linalg.norm(p_dense, ord=2)), 1.0)
        p_dense = p_dense / p_scale
        q = -2.0 * np.real(self.kernel.conj().T @ target_scaled) / p_scale
        # p_scale is returned so the elastic penalties mu can be rescaled
        # consistently with the scaled quadratic objective
        return csc_matrix(0.5 * (p_dense + p_dense.T)), q, p_scale

    def _solve_qp(self, p, q, g, h, a, b, tol_val: float) -> np.ndarray | None:
        best = None
        attempts: list[str] = []
        self.last_solver = None
        self.last_status = None
        self.last_solver_status = None
        for solver in self.solvers:
            kwargs = CausalProjection._solver_kwargs(solver, self.max_iter)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    sol = solve_qp(p, q, g, h, a, b, solver=solver, **kwargs)
            except SolverNotFound:
                attempts.append(f"{solver}:not_found")
                continue
            except Exception as exc:
                attempts.append(f"{solver}:exception={type(exc).__name__}")
                continue

            if sol is None or not np.all(np.isfinite(sol)):
                attempts.append(f"{solver}:none")
                continue

            sol = np.asarray(sol, dtype=float)
            coefficient_violation = float(max(np.max(g @ sol - h), 0.0))
            moment_violation = (
                0.0 if a is None else float(np.max(np.abs(a @ sol - b)))
            )
            status = "|".join(str(item.message).replace(" ", "_") for item in caught) or "ok"
            attempts.append(
                f"{solver}:coef={coefficient_violation:.3e}:moment={moment_violation:.3e}"
            )
            score = (max(coefficient_violation, moment_violation), float(q @ sol))
            if best is None or score < best[0]:
                best = (score, solver, status, sol)
            if coefficient_violation <= tol_val and moment_violation <= tol_val:
                break

        self.last_attempts = tuple(attempts)
        self.last_solver_attempts = list(attempts)
        if best is None:
            self.last_solver = "failed"
            self.last_status = "all_failed"
            self.last_solver_status = "all_failed"
            return None
        _, solver, status, solution = best
        self.last_solver = solver
        self.last_status = status
        self.last_solver_status = status
        return np.asarray(solution, dtype=float)

    def _fit_coefficients(self, target: np.ndarray) -> tuple[np.ndarray, float]:
        lhs = np.vstack((self.kernel.real, self.kernel.imag))
        rhs = np.concatenate((np.real(target), np.imag(target)))
        coeff, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        if not np.all(np.isfinite(coeff)):
            raise ValueError("unconstrained pole fit produced non-finite coefficients")
        coeff = np.asarray(coeff, dtype=float)
        residual = float(
            np.linalg.norm(self.kernel @ coeff - target)
            / max(np.linalg.norm(target), 1.0)
        )
        return coeff, residual

    def _validate_target(self, target_vec: np.ndarray) -> float:
        """Non-decay ratio ``|tail|/max`` of a c0-subtracted bosonic target.

        Returns 0.0 for (numerically) zero data.  The caller compares the
        ratio against ``tail_tol`` and splits off the non-decaying offset —
        this method never raises.
        """
        magnitude = float(np.max(np.abs(target_vec)))
        if magnitude <= 100.0 * np.finfo(float).eps:
            return 0.0
        tail_indices = np.argsort(np.abs(self.fit_omega))[-2:]
        tail_magnitude = float(np.max(np.abs(target_vec[tail_indices])))
        return tail_magnitude / magnitude

    def _gate(self, node_residual: float) -> None:
        if node_residual > self.fit_tol:
            raise RuntimeError(
                f"DLR node fit residual {node_residual:.3e} exceeds fit_tol "
                f"{self.fit_tol:.3e}; the target is not representable by the "
                "real-coefficient causal pole basis"
            )

    def _moment_penalty(
        self,
        moment_vector: np.ndarray,
        moment_sigma: np.ndarray,
        scale: float,
        node_residual: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scaled per-row moment sigmas and elastic penalties ``mu = 1/sigma``.

        ``moment_sigma`` is either the physical pair ``[sigma_c1, sigma_c2]``
        (mapped to the active rows: reflection symmetry keeps only the m2 row,
        whose sigma equals sigma_c2 — the internal sign flip leaves sigma
        unchanged) or already one value per moment row.  The same floors as
        the fermion path apply: node residual times the moment magnitude and
        the ``auto_floor`` knob.
        """
        n_rows = self.moment_rows.shape[0]
        sig = np.asarray(moment_sigma, dtype=float).reshape(-1)
        if sig.size == 2 and n_rows == 1:
            sigma_rows = sig[1:2]
        elif sig.size == n_rows:
            sigma_rows = sig
        else:
            raise ValueError(
                f"moment_sigma must have 2 entries [sigma_c1, sigma_c2] or "
                f"{n_rows} per-row entries, got {sig.size}"
            )
        if not np.all(np.isfinite(sigma_rows)) or np.any(sigma_rows < 0.0):
            raise ValueError("moment_sigma must be nonnegative and finite")
        abs_b = np.abs(np.asarray(moment_vector, dtype=float))
        sigma_scaled = np.maximum.reduce(
            [
                sigma_rows / scale,
                node_residual * abs_b,
                self.auto_floor * np.maximum(1.0, abs_b),
            ]
        )
        return sigma_scaled, 1.0 / sigma_scaled

    @staticmethod
    def _solution_ok(solution, g, h, a, b, tol_val: float) -> bool:
        coefficient_violation = float(max(np.max(g @ solution - h), 0.0))
        moment_violation = (
            0.0 if a is None else float(np.max(np.abs(a @ solution - b)))
        )
        return coefficient_violation <= tol_val and moment_violation <= tol_val

    def _clipped_fallback_coefficients(self, target_dec: np.ndarray) -> np.ndarray:
        """Guaranteed-causal last resort when every QP solver crashes.

        A truncated-SVD regularized fit (``rcond=1e-2``) keeps the pole
        coefficients tame, so clipping to the causal sign distorts the channel
        moderately (~11% measured on production data).  The *unregularized*
        fit must never be clipped: its huge cancelling coefficient pairs blow
        the reconstruction up by orders of magnitude once the wrong-sign half
        is removed.
        """
        lhs = np.vstack((self.kernel.real, self.kernel.imag))
        rhs = np.concatenate((np.real(target_dec), np.imag(target_dec)))
        coeff, *_ = np.linalg.lstsq(lhs, rhs, rcond=1.0e-2)
        coeff = np.asarray(coeff, dtype=float)
        if not np.all(np.isfinite(coeff)):
            raise ValueError(
                "regularized fallback fit produced non-finite coefficients"
            )
        if self.coefficient_sign < 0:
            return np.minimum(coeff, 0.0)
        return np.maximum(coeff, 0.0)

    def _validate(
        self,
        coefficients: np.ndarray,
        target: np.ndarray,
        moment_target: np.ndarray,
        *,
        node_residual: float,
        skipped: bool,
        c0: float,
        constraint_tol: float,
        enforce_moments: bool = True,
        moment_sigma_unscaled: np.ndarray | None = None,
        elastic: bool = False,
        equality_slack: np.ndarray | None = None,
        clipped: bool = False,
        c0_dropped: float = 0.0,
        tail_ratio: float = 0.0,
    ) -> dict[str, float | bool]:
        fit_projected = self.kernel @ coefficients
        relative_change = float(
            np.linalg.norm(fit_projected - target)
            / max(np.linalg.norm(target), 1.0e-30)
        )
        coefficient_violation = float(
            max(np.max(-self.coefficient_sign * coefficients), 0.0)
        )
        moment_residual_vec = np.asarray(
            self.moment_rows @ coefficients - moment_target,
            dtype=float,
        ).reshape(-1)
        moment_residual = float(np.max(np.abs(moment_residual_vec)))
        m1_residual = 0.0
        m2_residual = moment_residual_vec[0]
        if not self.reflection_symmetry:
            m1_residual = moment_residual_vec[0]
            m2_residual = moment_residual_vec[1]

        base_valid = bool(
            np.all(np.isfinite(coefficients))
            and coefficient_violation <= constraint_tol
            and np.isfinite(relative_change)
        )
        moment_residual_sigma = None
        judge_moments_hard = (
            enforce_moments
            and moment_sigma_unscaled is None
            and not elastic
            and not clipped
        )
        if judge_moments_hard:
            valid = base_valid and moment_residual <= constraint_tol
        else:
            # elastic/clipped path: the causal optimum must not be invalidated
            # by a moment excess (that would resurrect the raw non-causal
            # fallback); report the excess in sigma units and warn instead.
            valid = base_valid
            if moment_sigma_unscaled is not None:
                allowed = np.maximum(
                    constraint_tol,
                    self.moment_sigma_safety * moment_sigma_unscaled,
                )
                moment_residual_sigma = float(
                    np.max(
                        np.abs(moment_residual_vec)
                        / np.maximum(moment_sigma_unscaled, 1.0e-300)
                    )
                )
                if np.any(np.abs(moment_residual_vec) > allowed):
                    warnings.warn(
                        "elastic bosonic projection left a moment residual of "
                        f"{moment_residual_sigma:.2f} sigma (max allowed "
                        f"{self.moment_sigma_safety:.1f}); keeping the causal "
                        "optimum",
                        RuntimeWarning,
                        stacklevel=3,
                    )
        return {
            "valid": valid,
            "relative_change": relative_change,
            "coefficient_violation": coefficient_violation,
            "moment_residual": moment_residual,
            "moment_m1_residual": float(m1_residual),
            "moment_m2_residual": float(m2_residual),
            "moment_residual_sigma": moment_residual_sigma,
            "moment_sigma": (
                tuple(float(s) for s in moment_sigma_unscaled)
                if moment_sigma_unscaled is not None
                else None
            ),
            "elastic": bool(elastic),
            "equality_slack": (
                tuple(float(s) for s in equality_slack)
                if equality_slack is not None
                else None
            ),
            "clipped": bool(clipped),
            "c0_dropped": float(c0_dropped),
            "tail_ratio": float(tail_ratio),
            "node_residual": node_residual,
            "skipped": skipped,
            "c0": float(c0),
            "effective_tol": float(constraint_tol),
            "auto_tol": bool(self._auto_tol),
            "max_coefficient": float(np.max(coefficients)),
            "min_coefficient": float(np.min(coefficients)),
            "positive_count": int(np.sum(coefficients > 0.0)),
            "fit_grid_size": int(self.fit_omega.size),
        }


def ProjectBosonComponentWithFallback(
    projector: CausalBosonProjector,
    target: np.ndarray,
    tail_coeffs: np.ndarray,
    tail_sigma: np.ndarray | None = None,
    *,
    fallback_channel: np.ndarray | None = None,
    zero_ratio: float = 1.0e-3,
) -> np.ndarray:
    """Project one bosonic channel through the single elastic path.

    ``tail_sigma`` (the 4-vector of tail fit sigmas ``[sigma_c0 .. sigma_c3]``)
    supplies the elastic moment penalties; with it the QP is never infeasible,
    so the historical ``enforce_moments=False`` retry cascade is gone.

    If the projector still fails (every solver crashed and even its internal
    clipped fallback raised), the failure priority is:

    1. ``fallback_channel`` — the previous iteration's projected channel on
       ``output_omega``, when the caller has one.
    2. The regularized-fit clipped fallback, unless it is degenerate: when the
       causal sign clipping removes (almost) all spectral weight the clipped
       output collapses toward zero, and a zero channel (e.g. a vanishing
       retarded interaction fed to CTQMC) is physically worse than an
       unprojected one.  ``zero_ratio`` is that degeneracy threshold on
       ``|clipped| / |target|``.
    3. The raw unprojected channel, with a loud warning.
    """
    try:
        return projector.project(
            target,
            tail_coeffs=tail_coeffs,
            moment_sigma=tail_sigma[1:3] if tail_sigma is not None else None,
        )
    except RuntimeError as exc:
        if fallback_channel is not None:
            warnings.warn(
                "bosonic causal projection failed; returning the previous "
                f"iteration's projected channel: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.asarray(fallback_channel, dtype=np.complex128).copy()
        target_arr = np.asarray(target, dtype=np.complex128)
        coeff = projector._clipped_fallback_coefficients(target_arr)
        clipped = projector.output_kernel @ coeff
        target_norm = float(np.linalg.norm(target_arr))
        if float(np.linalg.norm(clipped)) >= zero_ratio * target_norm:
            warnings.warn(
                "bosonic causal projection failed; returning the "
                f"regularized-fit clipped fallback instead of the raw channel: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return clipped
        # Degenerate clipping: the causal cone holds (almost) none of the
        # channel's weight, so clipping would zero it out entirely.
        warnings.warn(
            "bosonic causal projection failed and the clipped fallback is "
            f"degenerate (|clipped|/|target| < {zero_ratio:.1e}); returning "
            f"the UNPROJECTED channel to preserve the physics: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return target_arr.copy()


BosonPoleQPProjector = CausalBosonProjector


class CausalFermionProjector:
    """Fermionic DLR scalar causal projection.

    This class keeps the kernel-aware fermionic path: fit one scalar channel to
    real DLR pole coefficients, enforce the fermionic pole-weight sign
    constraint, and preserve the first three high-frequency moments.
    """

    DEFAULT_SOLVERS = CausalProjection.DEFAULT_SOLVERS

    def __init__(
        self,
        *,
        d,
        beta: float,
        omega: np.ndarray,
        coefficient_sign: int = -1,
        solvers: Sequence[str] | None = None,
        max_iter: int = 100000,
        constraint_tol: float | str = 1.0e-8,
        auto_safety: float = 10.0,
        auto_floor: float = 1.0e-8,
        fit_tol: float = 1.0e-6,
        tail_points: int = 5,
        moment_sigma_safety: float = 3.0,
        raise_on_failure: bool = True,
    ) -> None:
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError("beta must be a positive finite number")
        if fit_tol <= 0.0 or not np.isfinite(fit_tol):
            raise ValueError("fit_tol must be a positive finite number")
        if moment_sigma_safety <= 0.0 or not np.isfinite(moment_sigma_safety):
            raise ValueError("moment_sigma_safety must be a positive finite number")
        # constraint_tol may be a positive number, or the string "auto": in auto
        # mode the per-call acceptance tolerance is set to the data's noise floor
        # (auto_safety * node_residual, lower-bounded by auto_floor), so noisy
        # data is not rejected for failing an unattainable fixed tolerance.
        self._auto_tol = isinstance(constraint_tol, str)
        if self._auto_tol:
            if constraint_tol != "auto":
                raise ValueError("constraint_tol string must be 'auto'")
            if auto_safety <= 0.0 or not np.isfinite(auto_safety):
                raise ValueError("auto_safety must be a positive finite number")
            if auto_floor <= 0.0 or not np.isfinite(auto_floor):
                raise ValueError("auto_floor must be a positive finite number")
        self.auto_safety = float(auto_safety)
        self.auto_floor = float(auto_floor)
        if isinstance(tail_points, bool) or not isinstance(
            tail_points, (int, np.integer)
        ):
            raise ValueError("tail_points must be an integer >= 4")
        if int(tail_points) < 4:
            # the tail model has four coefficients [c0, c1, c2, c3]; at least
            # four sampling points are needed for the lstsq fit.
            raise ValueError("tail_points must be an integer >= 4")

        self.statistic = "F"
        # the inner QP needs a numeric default tolerance; in auto mode the real
        # per-call tolerance is always passed explicitly via tol=, so this is a
        # never-used fallback.  The inner QP never raises: this projector must
        # distinguish "all solvers crashed" (-> clipped fallback) from "the
        # best candidate violates the tolerance" (-> judged by _validate), so
        # it inspects the returned CausalQPResult instead of catching errors.
        qp_constraint_tol = self.auto_floor if self._auto_tol else constraint_tol
        self.qp = CausalProjection(
            coefficient_sign=coefficient_sign,
            solvers=solvers,
            max_iter=max_iter,
            constraint_tol=qp_constraint_tol,
            raise_on_failure=False,
        )

        self.d = d
        self.beta = float(beta)
        self.omega = Common.RealFrequencyVector(omega, name="omega")

        x_nodes = np.asarray(d.get_dlr_frequencies(), dtype=float)
        if x_nodes.ndim != 1 or x_nodes.size == 0:
            raise ValueError("d must provide a nonempty one-dimensional DLR grid")
        if not np.all(np.isfinite(x_nodes)):
            raise ValueError("DLR real-frequency nodes contain non-finite values")
        self.x_nodes = x_nodes
        self.nodes = x_nodes / self.beta
        self.rank = int(x_nodes.size)

        self.coefficient_sign = self.qp.coefficient_sign
        self.solvers = self.qp.solvers
        self.max_iter = self.qp.max_iter
        self.constraint_tol = self.qp.constraint_tol
        self.fit_tol = float(fit_tol)
        self.tail_points = int(tail_points)
        self.moment_sigma_safety = float(moment_sigma_safety)
        self.raise_on_failure = bool(raise_on_failure)
        if self.tail_points > self.omega.size:
            raise ValueError(
                f"tail_points {self.tail_points} exceeds the sampling grid size "
                f"{self.omega.size}"
            )

        self.kernel = self._build_kernel(self.omega)
        weight = np.real(self.kernel.conj().T @ self.kernel)
        self.weight = 0.5 * (weight + weight.T)
        self.moment_rows = self._moment_rows()

        self.last_coefficients: np.ndarray | None = None
        self.last_solver: str | None = None
        self.last_status: str | None = None
        self.last_attempts: tuple[str, ...] = ()
        self.last_validation: dict[str, float | bool] = {}

    def project(
        self,
        target: np.ndarray,
        *,
        tail_coeffs: np.ndarray | None = None,
        moment_sigma: np.ndarray | None = None,
    ) -> np.ndarray:
        """Causal-project one scalar Matsubara channel onto the node grid.

        The high-frequency tail ``c0 + c1/(iw) + c2/(iw)^2 + c3/(iw)^3`` of the
        input is estimated by ``_tail_coefficients``.  The constant term ``c0``
        is not representable by the decaying DLR pole basis, so it is subtracted
        before the QP and re-added to the returned channel (this mirrors
        ``Fourier.FLocDynM``, which separates ``high`` = c0 from the decaying
        moments).  For a Green's function ``c0 ~ 0`` (a no-op); for a
        self-energy ``c0 = Sigma_inf`` is preserved.  ``[c1, c2, c3]`` become
        the QP equality target.

        ``tail_coeffs`` lets the caller inject ``[c0, c1, c2, c3]`` computed on
        a different (e.g. native uniform) grid; when ``None`` they are estimated
        on ``self.omega``.

        ``moment_sigma`` (shape (3,), unscaled physical one-sigma uncertainties
        of ``[c1, c2, c3]``) switches the moment equalities to *elastic* mode:
        the QP softens them into an exact L1 penalty with ``mu_p = 1/sigma_p``
        (floored by the node residual and ``auto_floor``), so noisy moment
        estimates can never make the QP infeasible — the causal sign constraint
        stays hard and the moments are matched as well as the noise allows.

        Last-resort fallback: if every configured QP solver crashes, a
        regularized-fit clipping (see ``_clipped_fallback_coefficients``)
        returns a guaranteed-causal channel instead of the raw non-causal data.
        """

        self._reset_diagnostics()
        target_vec = Common.ComplexVector(
            target,
            name="target",
            expected_size=self.omega.size,
        )
        scale = float(max(np.max(np.abs(target_vec)), 1.0))
        target_scaled = target_vec / scale

        c0, c1_phys, c2_phys, c3_phys = self._tail_coefficients(
            target_scaled, tail_coeffs=tail_coeffs, scale=scale
        )
        # remove the constant tail; only the decaying part is pole-representable
        target_dec = target_scaled - c0
        self._validate_target(target_dec)

        reference, node_residual = self._fit_coefficients(target_dec)
        normal_eq_residual = float(
            np.linalg.norm(
                self.weight @ reference
                - np.real(self.kernel.conj().T @ target_dec)
            )
        )

        c1, c2, c3 = self._internal_tail_moments(c1_phys, c2_phys, c3_phys)
        equality_target = self._equality_target(c1, c2, c3)
        eff_tol, tol_scaled = self._effective_tol(node_residual, scale)

        elastic = moment_sigma is not None
        if elastic:
            sigma_scaled, mu = self._moment_penalty(
                equality_target, moment_sigma, scale, node_residual
            )
        else:
            sigma_scaled, mu = None, None

        equality_slack = None
        clipped = False
        used_elastic = False
        if self._reference_is_causal(
            reference, equality_target, tol_scaled, sigma_scaled
        ):
            coeff_scaled = reference
            skipped = True
            self.last_solver = None
            self.last_status = "skipped"
            self.last_attempts = ()
        else:
            skipped = False
            # hard equalities first: on feasible (clean) data this is exact
            # and cheap.  Only when the hard problem fails (garbage moment
            # estimates make the moment cone infeasible) fall back to the
            # elastic penalties, whose feasible set is never empty — the
            # huge mu = 1/sigma values make the extended QP numerically
            # stiff, so it is not used when the hard solve already works.
            result = self.qp.project(
                reference,
                equality_matrix=self.moment_rows,
                equality_target=equality_target,
                weight_matrix=self.weight,
                tol=tol_scaled,
            )
            if elastic and not result.success:
                result = self.qp.project(
                    reference,
                    equality_matrix=self.moment_rows,
                    equality_target=equality_target,
                    weight_matrix=self.weight,
                    tol=tol_scaled,
                    equality_penalty=mu,
                )
                used_elastic = True
            if not result.success and result.status == "all_failed":
                # every solver crashed: guaranteed-causal clipped fallback
                # instead of passing raw non-causal data downstream.
                coeff_scaled = self._clipped_fallback_coefficients(target_dec)
                clipped = True
                self.last_solver = None
                self.last_status = "clipped_fallback"
                self.last_attempts = result.attempts
                distortion = float(
                    np.linalg.norm(self.kernel @ coeff_scaled - target_dec)
                    / max(np.linalg.norm(target_dec), 1.0e-30)
                )
                warnings.warn(
                    "causal QP failed for all configured solvers; returning the "
                    "regularized-fit clipped fallback (guaranteed causal, "
                    f"relative distortion {distortion:.3e})",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                coeff_scaled = result.coefficients
                self.last_solver = result.solver
                self.last_status = result.status
                self.last_attempts = result.attempts
                if result.equality_slack is not None:
                    equality_slack = result.equality_slack * scale

        coefficients = np.asarray(coeff_scaled * scale, dtype=float)
        c0_unscaled = float(c0 * scale)
        self.last_coefficients = coefficients
        # the decaying target the QP actually saw, in unscaled units
        target_dec_unscaled = target_vec - c0_unscaled
        self.last_validation = self._validate(
            coefficients,
            target_dec_unscaled,
            equality_target * scale,
            node_residual=node_residual,
            normal_eq_residual=normal_eq_residual,
            skipped=skipped,
            c0=c0_unscaled,
            constraint_tol=eff_tol,
            moment_sigma_unscaled=(
                sigma_scaled * scale if sigma_scaled is not None else None
            ),
            elastic=used_elastic,
            equality_slack=equality_slack,
            clipped=clipped,
        )
        if self.raise_on_failure and not bool(self.last_validation["valid"]):
            raise RuntimeError(
                "causal projection returned a constraint-violating candidate: "
                f"coeff={self.last_validation['coefficient_violation']:.3e}, "
                f"moment={self.last_validation['moment_residual']:.3e}"
            )
        # re-add the constant tail removed before the QP
        return self.kernel @ coefficients + c0_unscaled

    def check(
        self,
        target: np.ndarray,
        *,
        enforce_gate: bool = True,
        tail_coeffs: np.ndarray | None = None,
        moment_sigma: np.ndarray | None = None,
    ) -> CausalCheckResult:
        """Gate and feasibility-check one scalar channel without projecting.

        Violations are reported in the unscaled units of the input data.
        With ``enforce_gate=False`` a node fit residual above ``fit_tol``
        does not raise; it is only reported in ``node_residual`` (the sign
        verdict of such a channel describes a garbage fit — use the residual
        to judge data quality first).  The ``last_*`` diagnostics are not
        modified; they always describe the most recent ``project`` call.

        Mirrors ``project``: the constant tail ``c0`` is subtracted before the
        decaying fit, and ``[c1, c2, c3]`` (estimated on ``self.omega`` or
        injected via ``tail_coeffs``) are the equality target.  With
        ``moment_sigma`` the moment part of the verdict is judged in sigma
        units (``<= moment_sigma_safety``) instead of the hard tolerance,
        matching the elastic ``project`` skip criterion.
        """

        target_vec = Common.ComplexVector(
            target,
            name="target",
            expected_size=self.omega.size,
        )
        scale = float(max(np.max(np.abs(target_vec)), 1.0))
        target_scaled = target_vec / scale

        c0, c1_phys, c2_phys, c3_phys = self._tail_coefficients(
            target_scaled, tail_coeffs=tail_coeffs, scale=scale
        )
        target_dec = target_scaled - c0
        self._validate_target(target_dec)

        reference, node_residual = self._fit_coefficients(target_dec)
        if enforce_gate:
            self._gate(node_residual)

        c1, c2, c3 = self._internal_tail_moments(c1_phys, c2_phys, c3_phys)
        equality_target = self._equality_target(c1, c2, c3)
        _, tol_scaled = self._effective_tol(node_residual, scale)
        verdict = self.qp.check(
            reference,
            equality_matrix=self.moment_rows,
            equality_target=equality_target,
            tol=tol_scaled,
        )
        causal = verdict.causal
        if moment_sigma is not None:
            sigma_scaled, _ = self._moment_penalty(
                equality_target, moment_sigma, scale, node_residual
            )
            causal = self._reference_is_causal(
                reference, equality_target, tol_scaled, sigma_scaled
            )
        return CausalCheckResult(
            causal=causal,
            max_inequality_violation=verdict.max_inequality_violation * scale,
            max_equality_residual=verdict.max_equality_residual * scale,
            violating_count=verdict.violating_count,
            node_residual=node_residual,
            c0=float(c0 * scale),
        )

    def _tail_coefficients(
        self,
        target_scaled: np.ndarray,
        *,
        tail_coeffs: np.ndarray | None = None,
        scale: float,
    ) -> np.ndarray:
        """Estimate physical ``[c0, c1, c2, c3]`` of the high-frequency tail.

        The model is ``G(iw) ~ c0 + c1/(iw) + c2/(iw)^2 + c3/(iw)^3``, fit by a
        real (real/imag-stacked) least squares over the ``tail_points`` largest
        ``|omega|`` sampling points.  This generalizes the two-point formula of
        ``Fourier.FLocDynM`` (its exactly-determined special case) and is robust
        to high-frequency noise.  Returned coefficients are in the same scaled
        units as ``target_scaled`` so the QP equality target and the validation
        ``* scale`` rescaling stay consistent.

        ``tail_coeffs`` (unscaled ``[c0, c1, c2, c3]`` computed on a different
        grid by the caller) bypasses the local fit; it is rescaled by ``scale``.

        The returned tail is in physical sign.  The QP-only sign convention is
        applied later by :meth:`_internal_tail_moments`.
        """

        if tail_coeffs is not None:
            c = np.asarray(tail_coeffs, dtype=float).reshape(-1)
            if c.shape != (4,):
                raise ValueError(
                    f"tail_coeffs must have shape (4,), got {c.shape}"
                )
            if not np.all(np.isfinite(c)):
                raise ValueError("tail_coeffs contains non-finite values")
            # injected coeffs are physical-sign tail coefficients (e.g. from
            # FermionTailCoefficients/BosonTailCoefficients on the native grid);
            # only rescale.
            c = c / scale
            return c

        return Fourier.FermionTailCoefficients(
            self.omega, target_scaled, self.tail_points
        )

    @staticmethod
    def _internal_tail_moments(
        c1_phys: float,
        c2_phys: float,
        c3_phys: float,
    ) -> tuple[float, float, float]:
        """Convert physical tail moments to the projector's pole convention."""
        return -float(c1_phys), -float(c2_phys), -float(c3_phys)

    def _equality_target(self, c1: float, c2: float, c3: float) -> np.ndarray:
        """QP equality target (scaled) matching fermionic ``moment_rows``.

        The 3 rows ``nodes**p`` (p=0,1,2) <-> the internally signed data
        tail ``[-c1_phys, -c2_phys, -c3_phys]``.
        """
        return np.array([c1, c2, c3], dtype=float)

    def _effective_tol(
        self, node_residual: float, scale: float
    ) -> tuple[float, float]:
        """Per-call acceptance tolerance as (unscaled, scaled).

        Fixed mode returns ``constraint_tol`` unchanged.  In ``"auto"`` mode the
        tolerance is set to the data's noise floor: ``node_residual`` (the
        relative pole-fit residual in scaled space, a per-channel noise proxy)
        times ``auto_safety``, lower-bounded by ``auto_floor``.  The scaled value
        feeds the QP ``tol`` (the QP works in scaled space); the unscaled value
        feeds ``_validate`` (it compares unscaled violations).
        """
        if not self._auto_tol:
            return self.constraint_tol, self.constraint_tol / scale
        eff_scaled = max(self.auto_floor / scale, self.auto_safety * node_residual)
        return eff_scaled * scale, eff_scaled

    def _reset_diagnostics(self) -> None:
        self.last_coefficients = None
        self.last_solver = None
        self.last_status = None
        self.last_attempts = ()
        self.last_validation = {}

    def _basis_freq(self, z: np.ndarray) -> np.ndarray:
        # pydlr applies a hidden .conj() inside eval_dlr_freq, so the kernel
        # must be assembled from unit coefficient vectors, never by hand.
        z = np.asarray(z, dtype=np.complex128)
        basis = np.empty((z.size, self.rank), dtype=np.complex128)
        for j in range(self.rank):
            unit = np.zeros((self.rank, 1, 1), dtype=np.complex128)
            unit[j, 0, 0] = 1.0
            basis[:, j] = self.d.eval_dlr_freq(
                unit,
                z,
                self.beta,
                xi=-1,
            )[:, 0, 0]
        return basis

    def _build_kernel(self, omega: np.ndarray) -> np.ndarray:
        return self._basis_freq(1j * omega)

    def _moment_rows(self) -> np.ndarray:
        # preserve the p = 0, 1, 2 frequency moments of the fitted reference
        # coefficients
        return np.vstack([self.nodes**power for power in range(3)])

    def _validate_target(self, target_vec: np.ndarray) -> None:
        return

    def _fit_coefficients(self, target: np.ndarray) -> tuple[np.ndarray, float]:
        lhs = np.vstack((self.kernel.real, self.kernel.imag))
        rhs = np.concatenate((np.real(target), np.imag(target)))
        coeff, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        if not np.all(np.isfinite(coeff)):
            raise ValueError("unconstrained pole fit produced non-finite coefficients")
        coeff = np.asarray(coeff, dtype=float)
        residual = float(
            np.linalg.norm(self.kernel @ coeff - target)
            / max(np.linalg.norm(target), 1.0)
        )
        return coeff, residual

    def _gate(self, node_residual: float) -> None:
        if node_residual > self.fit_tol:
            raise RuntimeError(
                f"DLR node fit residual {node_residual:.3e} exceeds fit_tol "
                f"{self.fit_tol:.3e}; the target is not representable by the "
                "real-coefficient causal pole basis (non-Hermitian noise or a "
                "mismatched grid?)"
            )

    def _reference_is_causal(
        self,
        reference: np.ndarray,
        equality_target: np.ndarray,
        tol_scaled: float,
        sigma_scaled: np.ndarray | None,
    ) -> bool:
        """Skip fast-path judgment for an unconstrained-fit reference.

        The sign verdict always uses the hard per-call tolerance.  Without
        ``sigma_scaled`` the moment verdict is the historical hard-equality
        one; with it the moments only need to sit within
        ``moment_sigma_safety`` sigma of the (noisy) tail estimate.
        """
        if sigma_scaled is None:
            verdict = self.qp.check(
                reference,
                equality_matrix=self.moment_rows,
                equality_target=equality_target,
                tol=tol_scaled,
            )
            return bool(verdict.causal)
        sign_residual = -self.coefficient_sign * reference
        if float(max(np.max(sign_residual), 0.0)) > tol_scaled:
            return False
        ratio = np.abs(self.moment_rows @ reference - equality_target) / sigma_scaled
        return bool(np.max(ratio) <= self.moment_sigma_safety)

    def _moment_penalty(
        self,
        equality_target: np.ndarray,
        moment_sigma: np.ndarray,
        scale: float,
        node_residual: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scaled moment sigmas and the elastic penalties ``mu = 1/sigma``.

        ``moment_sigma`` carries the unscaled physical one-sigma uncertainties
        of the tail moments.  Each scaled sigma is floored by the node fit
        residual (relative noise proxy) times the moment magnitude, and by the
        existing ``auto_floor`` knob, so ``mu`` can never blow up on a
        spuriously tiny fit covariance.
        """
        n_rows = self.moment_rows.shape[0]
        sigma_phys = np.asarray(moment_sigma, dtype=float).reshape(-1)
        if sigma_phys.shape != (n_rows,):
            raise ValueError(
                f"moment_sigma must have shape ({n_rows},), got {sigma_phys.shape}"
            )
        if not np.all(np.isfinite(sigma_phys)) or np.any(sigma_phys < 0.0):
            raise ValueError("moment_sigma must be nonnegative and finite")
        abs_b = np.abs(np.asarray(equality_target, dtype=float))
        sigma_scaled = np.maximum.reduce(
            [
                sigma_phys / scale,
                node_residual * abs_b,
                self.auto_floor * np.maximum(1.0, abs_b),
            ]
        )
        return sigma_scaled, 1.0 / sigma_scaled

    def _clipped_fallback_coefficients(self, target_dec: np.ndarray) -> np.ndarray:
        """Guaranteed-causal last resort when every QP solver crashes.

        A truncated-SVD regularized fit (``rcond=1e-2``) keeps the pole
        coefficients tame (measured ~+-0.3 on production glob.h5 data), so
        clipping to the causal sign distorts the channel by only ~11%.  The
        *unregularized* fit must never be clipped: its huge cancelling
        coefficient pairs (~+-1e8) blow the reconstruction up by ~4e8x once
        the wrong-sign half is removed.
        """
        lhs = np.vstack((self.kernel.real, self.kernel.imag))
        rhs = np.concatenate((np.real(target_dec), np.imag(target_dec)))
        coeff, *_ = np.linalg.lstsq(lhs, rhs, rcond=1.0e-2)
        coeff = np.asarray(coeff, dtype=float)
        if not np.all(np.isfinite(coeff)):
            raise ValueError(
                "regularized fallback fit produced non-finite coefficients"
            )
        if self.coefficient_sign < 0:
            return np.minimum(coeff, 0.0)
        return np.maximum(coeff, 0.0)

    def _validate(
        self,
        coefficients: np.ndarray,
        target: np.ndarray,
        equality_target: np.ndarray,
        *,
        node_residual: float,
        normal_eq_residual: float,
        skipped: bool,
        c0: float,
        constraint_tol: float,
        moment_sigma_unscaled: np.ndarray | None = None,
        elastic: bool = False,
        equality_slack: np.ndarray | None = None,
        clipped: bool = False,
    ) -> dict[str, float | bool]:
        fit_projected = self.kernel @ coefficients
        relative_change = float(
            np.linalg.norm(fit_projected - target)
            / max(np.linalg.norm(target), 1.0e-30)
        )
        coefficient_violation = float(
            max(np.max(-self.coefficient_sign * coefficients), 0.0)
        )
        moment_residual_vec = np.abs(
            self.moment_rows @ coefficients - equality_target
        )
        moment_residual = float(np.max(moment_residual_vec))

        base_valid = bool(
            np.all(np.isfinite(coefficients))
            and coefficient_violation <= constraint_tol
            and np.isfinite(relative_change)
        )
        moment_residual_sigma = None
        if moment_sigma_unscaled is None and not elastic and not clipped:
            # historical hard path: the moment equalities are part of validity
            valid = base_valid and moment_residual <= constraint_tol
        else:
            # elastic/clipped path: the returned coefficients are the best
            # causally attainable answer — a moment excess must not flip
            # validity (that would resurrect the raw non-causal fallback);
            # report it in sigma units and warn instead.
            valid = base_valid
            if moment_sigma_unscaled is not None:
                allowed = np.maximum(
                    constraint_tol,
                    self.moment_sigma_safety * moment_sigma_unscaled,
                )
                moment_residual_sigma = float(
                    np.max(
                        moment_residual_vec
                        / np.maximum(moment_sigma_unscaled, 1.0e-300)
                    )
                )
                if np.any(moment_residual_vec > allowed):
                    warnings.warn(
                        "elastic causal projection left a moment residual of "
                        f"{moment_residual_sigma:.2f} sigma (max allowed "
                        f"{self.moment_sigma_safety:.1f}); keeping the causal "
                        "optimum",
                        RuntimeWarning,
                        stacklevel=3,
                    )
        return {
            "valid": valid,
            "relative_change": relative_change,
            "coefficient_violation": coefficient_violation,
            "moment_residual": moment_residual,
            "moment_residual_sigma": moment_residual_sigma,
            "moment_sigma": (
                tuple(float(s) for s in moment_sigma_unscaled)
                if moment_sigma_unscaled is not None
                else None
            ),
            "elastic": bool(elastic),
            "equality_slack": (
                tuple(float(s) for s in equality_slack)
                if equality_slack is not None
                else None
            ),
            "clipped": bool(clipped),
            "node_residual": node_residual,
            "normal_eq_residual": normal_eq_residual,
            "skipped": skipped,
            "c0": float(c0),
            "effective_tol": float(constraint_tol),
            "auto_tol": bool(self._auto_tol),
            "max_coefficient": float(np.max(coefficients)),
            "min_coefficient": float(np.min(coefficients)),
        }


class CausalProjector:
    """Compatibility facade for scalar causal projection.

    New code should instantiate ``CausalFermionProjector`` or
    ``CausalBosonProjector`` directly.  This facade keeps the historical
    ``statistic=`` constructor available and forwards all diagnostics and public
    methods to the selected backend.
    """

    DEFAULT_SOLVERS = CausalProjection.DEFAULT_SOLVERS

    def __init__(
        self,
        *,
        statistic: str,
        d,
        beta: float,
        omega: np.ndarray,
        coefficient_sign: int = -1,
        reflection_symmetry: bool = True,
        boson_backend: str = "pole_qp",
        fit_omega: np.ndarray | None = None,
        output_omega: np.ndarray | None = None,
        solvers: Sequence[str] | None = None,
        max_iter: int = 100000,
        constraint_tol: float | str = 1.0e-8,
        auto_safety: float = 10.0,
        auto_floor: float = 1.0e-8,
        fit_tol: float = 1.0e-6,
        tail_tol: float = 1.0e-1,
        tail_points: int = 5,
        regularization: float = 0.0,
        raise_on_failure: bool = True,
    ) -> None:
        if statistic not in ("F", "B"):
            raise ValueError("statistic must be 'F' or 'B'")
        if boson_backend != "pole_qp":
            raise ValueError("boson_backend='legacy' is no longer supported")

        self.statistic = statistic
        self.boson_backend = "pole_qp"
        if statistic == "F":
            self._backend = CausalFermionProjector(
                d=d,
                beta=beta,
                omega=omega,
                coefficient_sign=coefficient_sign,
                solvers=solvers,
                max_iter=max_iter,
                constraint_tol=constraint_tol,
                auto_safety=auto_safety,
                auto_floor=auto_floor,
                fit_tol=fit_tol,
                tail_points=tail_points,
                raise_on_failure=raise_on_failure,
            )
            return

        self._backend = CausalBosonProjector(
            d=d,
            beta=beta,
            fit_omega=omega if fit_omega is None else fit_omega,
            output_omega=omega if output_omega is None else output_omega,
            coefficient_sign=coefficient_sign,
            reflection_symmetry=reflection_symmetry,
            solvers=solvers,
            max_iter=max_iter,
            constraint_tol=constraint_tol,
            auto_safety=auto_safety,
            auto_floor=auto_floor,
            fit_tol=fit_tol,
            tail_tol=tail_tol,
            tail_points=tail_points,
            regularization=regularization,
            raise_on_failure=raise_on_failure,
        )

    def __getattr__(self, name: str):
        return getattr(self._backend, name)

    def project(
        self,
        target: np.ndarray,
        *,
        tail_coeffs: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        return self._backend.project(target, tail_coeffs=tail_coeffs, **kwargs)

    def check(
        self,
        target: np.ndarray,
        *,
        enforce_gate: bool = True,
        tail_coeffs: np.ndarray | None = None,
        **kwargs,
    ) -> CausalCheckResult:
        return self._backend.check(
            target,
            enforce_gate=enforce_gate,
            tail_coeffs=tail_coeffs,
            **kwargs,
        )
