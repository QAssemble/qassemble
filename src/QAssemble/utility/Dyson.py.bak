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

        g0_blocks = np.moveaxis(g0, (0, 1), (-2, -1))
        sigma_blocks = np.moveaxis(sigma, (0, 1), (-2, -1))
        out_blocks = np.moveaxis(out, (0, 1), (-2, -1))

        Dyson._solve_blockwise(g0_blocks, sigma_blocks, out_blocks)

    @staticmethod
    def _solve_blockwise(g0_blocks, sigma_blocks, out_blocks):
        """Iterate over stacked matrices and solve the Dyson equation."""

        norb = g0_blocks.shape[-1]
        diag_idx = np.diag_indices(norb)

        for multi_idx in np.ndindex(g0_blocks.shape[:-2]):
            g0_block = g0_blocks[multi_idx]
            sigma_block = sigma_blocks[multi_idx]

            temp = -sigma_block @ g0_block
            temp[diag_idx] += 1.0

            solved_t = np.linalg.solve(temp.T, g0_block.T)
            out_blocks[multi_idx] = solved_t.T

    @staticmethod
    def _solve_boson_block(g0_block, sigma_block, out_block, diag_idx):
        """Solve a single bosonic Dyson block."""

        norb = g0_block.shape[0]
        ns = g0_block.shape[2]
        dim = norb * ns

        g0_comp = g0_block.transpose(0, 2, 1, 3).reshape(dim, dim)
        sigma_comp = sigma_block.transpose(0, 2, 1, 3).reshape(dim, dim)

        temp = -sigma_comp @ g0_comp
        temp[diag_idx] += 1.0

        solved_t = np.linalg.solve(temp.T, g0_comp.T)
        result = solved_t.T.reshape(norb, ns, norb, ns).transpose(0, 2, 1, 3)
        out_block[...] = result

    @staticmethod
    def _solve_bosonic(g0, sigma, out):
        """Solve Dyson blocks for bosonic objects in-place on *out*."""

        norb = g0.shape[0]
        ns = g0.shape[2]
        dim = norb * ns
        diag_idx = np.diag_indices(dim)

        # Remaining axes enumerate k-points, frequencies, etc.
        extra_shape = g0.shape[4:]

        if not extra_shape:
            Dyson._solve_boson_block(g0, sigma, out, diag_idx)
            return

        leading = (slice(None),) * 4  # preserve (norb, norb, ns, ns) layout

        for multi_idx in np.ndindex(extra_shape):
            idx = leading + multi_idx
            Dyson._solve_boson_block(
                g0[idx],
                sigma[idx],
                out[idx],
                diag_idx,
            )

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
