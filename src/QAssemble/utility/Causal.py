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


@dataclass(frozen=True)
class CausalCheckResult:
    """Result returned by a causal coefficient feasibility check.

    ``node_residual`` is filled by the kernel-aware ``CausalProjector.check``
    as a data-quality diagnostic; the pure-QP ``CausalProjection.check``
    leaves it ``None``.
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

    DEFAULT_SOLVERS = ("clarabel", "proxqp", "scs")

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

        coeff, solver, status, attempts = self._solve_qp(
            p,
            q,
            g,
            h,
            a,
            b,
            ref,
            weight,
            tol_val,
        )

        if coeff is None:
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
            )

        ineq = self._max_inequality_violation(g, h, coeff)
        eq = self._max_equality_residual(a, b, coeff)
        success = bool(ineq <= tol_val and eq <= tol_val)
        if self.raise_on_failure and not success:
            raise RuntimeError(
                "causal QP returned a constraint-violating candidate: "
                f"ineq={ineq:.3e}, eq={eq:.3e}"
            )

        return CausalQPResult(
            coefficients=np.asarray(coeff, dtype=float),
            success=success,
            solver=solver,
            status=status,
            attempts=tuple(attempts),
            objective=self._weighted_objective(coeff, ref, weight),
            max_inequality_violation=ineq,
            max_equality_residual=eq,
            relative_change=self._relative_change(coeff, ref),
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


class CausalProjector:
    """Decompose -> gate -> check -> QP -> revalidate causal projection.

    A single class handles both statistics, selected by ``statistic``
    (``'F'`` fermion / ``'B'`` boson).  The statistic only changes four
    things: the DLR kernel sign ``xi``, the kernel construction (bosonic
    reflection symmetrization), the moment rows, and the target validation.
    Everything else — the scale/fit/gate/moment/QP/revalidate pipeline — is
    shared.  The pole basis is always the DLR real-frequency grid of ``d``;
    the ``omega`` constructor argument is the (arbitrary) Matsubara
    *sampling* grid the input data lives on, so the same projector works on
    a sparse DLR grid or a dense uniform grid (see ``resolve_causal_grid``).
    The QP itself is delegated to a composed ``CausalProjection`` instance
    (``self.qp``).

    The ``reflection_symmetry`` and ``tail_tol`` arguments are bosonic-only;
    for ``statistic='F'`` they are silently ignored.

    The pipeline for one scalar Matsubara channel is:

    1. scale the target by ``max(|target|, 1)``;
    2. estimate the physical tail ``[c0, c1, c2, c3]`` by a robust real least squares
       over the ``tail_points`` largest ``|omega|`` points
       (``_tail_coefficients``); the constant ``c0`` is not representable by
       the decaying pole basis, so subtract it (``target_dec = target - c0``)
       and re-add it to the final channel — Green's functions have ``c0 ~ 0``
       (no-op), self-energies keep ``c0 = Sigma_inf``;
    3. decompose ``target_dec`` into real DLR pole weights via stacked
       real/imag least squares on the native node kernel; a node fit residual
       above ``fit_tol`` (gate) -> ``RuntimeError``;
    4. the QP equality target is the internally signed data tail
       ``[-c1, -c2, -c3]`` (not the moments of the fitted coefficients);
       ``moment_rows = nodes**p`` maps coefficients to those internal moments;
    5. if the reference is already causal, skip the QP (the output is still
       the refit ``kernel @ reference + c0``);
    6. otherwise solve the W = Re(K^H K) weighted sign-constrained QP, which
       is the frequency-norm optimal causal fit;
    7. unscale, revalidate against ``constraint_tol``, and return
       ``kernel @ coefficients + c0``.

    Diagnostics from the most recent ``project`` call are exposed as
    ``last_coefficients``, ``last_solver``, ``last_status``,
    ``last_attempts``, ``last_validation``.  They are reset at the start of
    every ``project`` call, so they never describe an earlier call; a gate
    rejection leaves ``last_status = "gate_failed"`` and a minimal
    ``last_validation``.  ``check`` does not touch them.

    Bosonic caveat (``statistic='B'`` with ``reflection_symmetry=True``):
    the symmetrized kernel is numerically rank-deficient (effective rank ~
    half), so the coefficient representation of a given target is not
    unique.  ``check`` evaluates the sufficient sign condition on the
    least-squares representative and can therefore report non-causal for
    data that admits a causal representation (e.g. the output of
    ``project``, whose causal coefficients are in ``last_coefficients``).
    ``project`` itself is unaffected.  A constant offset is estimated as
    ``c0`` and subtracted before the decaying pole fit; targets whose
    ``target - c0`` does not decay at the largest ``|nu|`` nodes are rejected
    with a ``RuntimeError`` (``tail_tol`` controls this guard).
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
        solvers: Sequence[str] | None = None,
        max_iter: int = 100000,
        constraint_tol: float | str = 1.0e-8,
        auto_safety: float = 10.0,
        auto_floor: float = 1.0e-8,
        fit_tol: float = 1.0e-6,
        tail_tol: float = 1.0e-1,
        tail_points: int = 5,
        raise_on_failure: bool = True,
    ) -> None:
        if statistic not in ("F", "B"):
            raise ValueError("statistic must be 'F' or 'B'")
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError("beta must be a positive finite number")
        if fit_tol <= 0.0 or not np.isfinite(fit_tol):
            raise ValueError("fit_tol must be a positive finite number")
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

        self.statistic = statistic
        self._xi = -1 if statistic == "F" else 1
        # reflection_symmetry / tail_tol are bosonic-only; for fermions they
        # are silently ignored (not validated, not stored).
        if statistic == "B":
            if tail_tol <= 0.0 or not np.isfinite(tail_tol):
                raise ValueError("tail_tol must be a positive finite number")
            self.reflection_symmetry = bool(reflection_symmetry)
            self.tail_tol = float(tail_tol)

        # the inner QP needs a numeric default tolerance; in auto mode the real
        # per-call tolerance is always passed explicitly via tol=, so this is a
        # never-used fallback.
        qp_constraint_tol = self.auto_floor if self._auto_tol else constraint_tol
        self.qp = CausalProjection(
            coefficient_sign=coefficient_sign,
            solvers=solvers,
            max_iter=max_iter,
            constraint_tol=qp_constraint_tol,
            raise_on_failure=raise_on_failure,
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

        verdict = self.qp.check(
            reference,
            equality_matrix=self.moment_rows,
            equality_target=equality_target,
            tol=tol_scaled,
        )
        if verdict.causal:
            coeff_scaled = reference
            skipped = True
            self.last_solver = None
            self.last_status = "skipped"
            self.last_attempts = ()
        else:
            result = self.qp.project(
                reference,
                equality_matrix=self.moment_rows,
                equality_target=equality_target,
                weight_matrix=self.weight,
                tol=tol_scaled,
            )
            coeff_scaled = result.coefficients
            skipped = False
            self.last_solver = result.solver
            self.last_status = result.status
            self.last_attempts = result.attempts

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
        injected via ``tail_coeffs``) are the equality target.
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
        return CausalCheckResult(
            causal=verdict.causal,
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

        For both fermions and bosons the returned tail is in physical sign.
        The QP-only sign convention is applied later by
        :meth:`_internal_tail_moments`.
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

        if self.statistic == "F":
            return Fourier.FermionTailCoefficients(
                self.omega, target_scaled, self.tail_points
            )
        return Fourier.BosonTailCoefficients(
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
        """QP equality target (scaled) matching ``moment_rows`` per statistic.

        Fermion: 3 rows ``nodes**p`` (p=0,1,2) <-> the internally signed data
        tail ``[-c1_phys, -c2_phys, -c3_phys]``.

        Boson: the tanh-weighted ``moment_rows`` encode
        the internally signed physical tail.  The reflection-symmetrized kernel
        keeps only even powers, so the single row ``tanh(x/2)*omega`` anchors
        ``-c2_phys``; the plain kernel's two rows
        ``[tanh(x/2), tanh(x/2)*omega]`` anchor ``[-c1_phys, -c2_phys]``.
        """
        if self.statistic == "F":
            return np.array([c1, c2, c3], dtype=float)
        if self.reflection_symmetry:
            return np.array([c2], dtype=float)
        return np.array([c1, c2], dtype=float)

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
                xi=self._xi,
            )[:, 0, 0]
        return basis

    def _build_kernel(self, omega: np.ndarray) -> np.ndarray:
        if self.statistic == "F":
            return self._basis_freq(1j * omega)
        # Boson: reflection-symmetrize K_sym = (K(+i nu) + K(-i nu)) / 2 when
        # requested (matches causal_boson.BosonPoleQPProjector), else plain.
        if self.reflection_symmetry:
            return 0.5 * (
                self._basis_freq(1j * omega) + self._basis_freq(-1j * omega)
            )
        return self._basis_freq(1j * omega)

    def _moment_rows(self) -> np.ndarray:
        if self.statistic == "F":
            # preserve the p = 0, 1, 2 frequency moments of the fitted
            # reference coefficients
            return np.vstack([self.nodes**power for power in range(3)])
        # Boson: dimensionless x in the tanh — tanh(0.5 * beta * omega_l) in
        # disguise; using omega_l directly would be off by a factor beta.
        # The tanh factor of the bosonic kernel absorbs the sign structure of
        # the odd spectral function, so a uniform coefficient sign suffices.
        self.bose_corr = np.tanh(0.5 * self.x_nodes)
        row_m2 = self.bose_corr * self.nodes
        if self.reflection_symmetry:
            # only the M2 row tanh(x/2) * omega_l is enforced
            return row_m2.reshape(1, -1)
        # plain kernel: enforce both M1 = tanh(x/2) and M2 rows
        return np.vstack([self.bose_corr, row_m2])

    def _validate_target(self, target_vec: np.ndarray) -> None:
        if self.statistic == "F":
            return
        # Boson: after subtracting c0, the decaying target must be small at the
        # largest |nu| nodes.
        magnitude = float(np.max(np.abs(target_vec)))
        if magnitude <= 100.0 * np.finfo(float).eps:
            return
        tail_indices = np.argsort(np.abs(self.omega))[-2:]
        tail_magnitude = float(np.max(np.abs(target_vec[tail_indices])))
        if tail_magnitude > self.tail_tol * magnitude:
            raise RuntimeError(
                "bosonic target minus c0 does not decay at the largest |nu| nodes "
                f"(|tail|/max = {tail_magnitude / magnitude:.3e} > tail_tol "
                f"{self.tail_tol:.1e}); the fitted c0 does not isolate a "
                "decaying bosonic pole contribution"
            )

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
    ) -> dict[str, float | bool]:
        fit_projected = self.kernel @ coefficients
        relative_change = float(
            np.linalg.norm(fit_projected - target)
            / max(np.linalg.norm(target), 1.0e-30)
        )
        coefficient_violation = float(
            max(np.max(-self.coefficient_sign * coefficients), 0.0)
        )
        moment_residual = float(
            np.max(np.abs(self.moment_rows @ coefficients - equality_target))
        )
        return {
            "valid": bool(
                np.all(np.isfinite(coefficients))
                and coefficient_violation <= constraint_tol
                and moment_residual <= constraint_tol
                and np.isfinite(relative_change)
            ),
            "relative_change": relative_change,
            "coefficient_violation": coefficient_violation,
            "moment_residual": moment_residual,
            "node_residual": node_residual,
            "normal_eq_residual": normal_eq_residual,
            "skipped": skipped,
            "c0": float(c0),
            "effective_tol": float(constraint_tol),
            "auto_tol": bool(self._auto_tol),
            "max_coefficient": float(np.max(coefficients)),
            "min_coefficient": float(np.min(coefficients)),
        }
