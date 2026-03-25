"""
This module provides a set of static methods for solving the Dyson equation to compute
renormalized Green's functions from bare Green's functions and self-energies.
"""
import numpy as np


class Dyson:
    """Lightweight Dyson solvers with a focus on peak-memory efficiency."""

    @staticmethod
    def _allocate_output(reference, out):
        """Return an output array matching *reference*, allocating if needed."""

        dtype = np.result_type(reference.dtype, np.complex128)
        order = "F" if np.isfortran(reference) else "C"

        if out is None:
            return np.empty(reference.shape, dtype=dtype, order=order)

        if out.shape != reference.shape:
            raise ValueError("Output array has incorrect shape.")

        if out.dtype != dtype:
            return out.astype(dtype, copy=False)

        return out

    @staticmethod
    def _solve_fermionic(g0, sigma, out):
        """Solve Dyson blocks for fermionic objects in-place on *out*."""

        # g0, sigma, out: (norb, norb, ...)
        norb = g0.shape[0]
        extra_shape = g0.shape[2:]

        # Reshape to (N, norb, norb) batch
        g0_batch = np.moveaxis(g0, (0, 1), (-2, -1)).reshape(-1, norb, norb)
        sigma_batch = np.moveaxis(sigma, (0, 1), (-2, -1)).reshape(-1, norb, norb)

        # Batched Dyson: G = G0 (I - Sigma G0)^{-1}
        temp = -sigma_batch @ g0_batch
        idx = np.arange(norb)
        temp[:, idx, idx] += 1.0

        # np.linalg.solve batched: solve temp^T x = g0^T for each block
        temp_T = np.ascontiguousarray(temp.transpose(0, 2, 1))
        g0_T = np.ascontiguousarray(g0_batch.transpose(0, 2, 1))
        solved_T = np.linalg.solve(temp_T, g0_T)
        result_batch = solved_T.transpose(0, 2, 1)  # (N, norb, norb)

        result = result_batch.reshape(extra_shape + (norb, norb))
        out[...] = np.moveaxis(result, (-2, -1), (0, 1))

    @staticmethod
    def _solve_bosonic(g0, sigma, out):
        """Solve Dyson blocks for bosonic objects in-place on *out*."""

        norb = g0.shape[0]
        ns = g0.shape[2]
        dim = norb * ns
        extra_shape = g0.shape[4:]

        if not extra_shape:
            # Single block fallback
            g0_comp = g0.transpose(0, 2, 1, 3).reshape(dim, dim)
            sigma_comp = sigma.transpose(0, 2, 1, 3).reshape(dim, dim)
            temp = -sigma_comp @ g0_comp
            didx = np.arange(dim)
            temp[didx, didx] += 1.0
            solved_t = np.linalg.solve(temp.T, g0_comp.T)
            out[...] = solved_t.T.reshape(norb, ns, norb, ns).transpose(0, 2, 1, 3)
            return

        N = 1
        for s in extra_shape:
            N *= s

        # (norb, norb, ns, ns, ...) -> (..., norb, ns, norb, ns) -> (N, dim, dim)
        g0_perm = np.moveaxis(g0, (0, 1, 2, 3), (-4, -2, -3, -1))
        sigma_perm = np.moveaxis(sigma, (0, 1, 2, 3), (-4, -2, -3, -1))
        g0_batch = g0_perm.reshape(N, dim, dim)
        sigma_batch = sigma_perm.reshape(N, dim, dim)

        # Batched Dyson
        temp = -sigma_batch @ g0_batch
        didx = np.arange(dim)
        temp[:, didx, didx] += 1.0

        temp_T = np.ascontiguousarray(temp.transpose(0, 2, 1))
        g0_T = np.ascontiguousarray(g0_batch.transpose(0, 2, 1))
        solved_T = np.linalg.solve(temp_T, g0_T)
        result_batch = solved_T.transpose(0, 2, 1)  # (N, dim, dim)

        # (N, dim, dim) -> (..., norb, ns, norb, ns) -> (norb, norb, ns, ns, ...)
        result = result_batch.reshape(extra_shape + (norb, ns, norb, ns))
        out[...] = np.moveaxis(result, (-4, -2, -3, -1), (0, 1, 2, 3))

    # Public API ---------------------------------------------------------

    @staticmethod
    def FLocStc(ffin, sig, out=None):
        """Static fermionic Dyson on local blocks."""

        result = Dyson._allocate_output(ffin, out)
        Dyson._solve_fermionic(ffin, sig, result)
        return result

    @staticmethod
    def FLatStc(ffin, sig, out=None):
        """Static fermionic Dyson on lattice blocks."""

        result = Dyson._allocate_output(ffin, out)
        Dyson._solve_fermionic(ffin, sig, result)
        return result

    @staticmethod
    def FLocDyn(ffin, sig, out=None):
        """Dynamic fermionic Dyson on local blocks."""

        result = Dyson._allocate_output(ffin, out)
        Dyson._solve_fermionic(ffin, sig, result)
        return result

    @staticmethod
    def FLatDyn(ffin, sig, out=None):
        """Dynamic fermionic Dyson on lattice blocks."""

        result = Dyson._allocate_output(ffin, out)
        Dyson._solve_fermionic(ffin, sig, result)
        return result

    @staticmethod
    def BLocStc(ffin, sig, out=None):
        """Static bosonic Dyson on local blocks."""

        result = Dyson._allocate_output(ffin, out)
        Dyson._solve_bosonic(ffin, sig, result)
        return result

    @staticmethod
    def BLocDyn(ffin, sig, out=None):
        """Dynamic bosonic Dyson on local blocks."""

        result = Dyson._allocate_output(ffin, out)
        Dyson._solve_bosonic(ffin, sig, result)
        return result

    @staticmethod
    def BLatStc(ffin, sig, out=None):
        """Static bosonic Dyson on lattice blocks."""

        result = Dyson._allocate_output(ffin, out)
        Dyson._solve_bosonic(ffin, sig, result)
        return result

    @staticmethod
    def BLatDyn(ffin, sig, out=None):
        """Dynamic bosonic Dyson on lattice blocks."""

        result = Dyson._allocate_output(ffin, out)
        Dyson._solve_bosonic(ffin, sig, result)
        return result
