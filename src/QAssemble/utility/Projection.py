"""
Projection utilities translated from `modules/Projection.f90`.

Each method performs basis projection with spin-resolved projectors:
    out = P^dagger * in * P
"""

from typing import Optional, Tuple

import numpy as np


class Projection:
    """Static projection kernels for fermionic and bosonic objects."""

    @staticmethod
    def _allocate_output(shape: Tuple[int, ...], reference: np.ndarray, out: Optional[np.ndarray]) -> np.ndarray:
        dtype = np.result_type(reference.dtype, np.complex128)
        order = "F" if np.isfortran(reference) else "C"

        if out is None:
            return np.zeros(shape, dtype=dtype, order=order)

        if out.shape != shape:
            raise ValueError(f"Output shape mismatch: expected {shape}, got {out.shape}")

        if out.dtype != dtype:
            return out.astype(dtype, copy=False)

        out.fill(0.0)
        return out

    @staticmethod
    def _validate_fp(ff: np.ndarray, projector: np.ndarray, ff_ndim: int) -> Tuple[int, int, int]:
        if ff.ndim != ff_ndim:
            raise ValueError(f"ff must be {ff_ndim}D, got {ff.ndim}D")
        if projector.ndim != 3:
            raise ValueError(f"projector must be 3D, got {projector.ndim}D")

        norb, norb2, ns = ff.shape[:3]
        if norb != norb2:
            raise ValueError("ff first two dimensions must be square (norb, norb)")

        if projector.shape[0] != norb:
            raise ValueError(
                f"projector first dimension mismatch: expected {norb}, got {projector.shape[0]}"
            )
        if projector.shape[2] != ns:
            raise ValueError(
                f"projector spin dimension mismatch: expected {ns}, got {projector.shape[2]}"
            )

        norbc = projector.shape[1]
        return norb, ns, norbc

    @staticmethod
    def _validate_bp(ff: np.ndarray, projector: np.ndarray, ff_ndim: int) -> Tuple[int, int, int]:
        if ff.ndim != ff_ndim:
            raise ValueError(f"ff must be {ff_ndim}D, got {ff.ndim}D")
        if projector.ndim != 3:
            raise ValueError(f"projector must be 3D, got {projector.ndim}D")

        norb, norb2, ns, ns2 = ff.shape[:4]
        if norb != norb2:
            raise ValueError("ff first two dimensions must be square (norb, norb)")
        if ns != ns2:
            raise ValueError("ff spin block must be square (ns, ns)")

        if projector.shape[0] != norb:
            raise ValueError(
                f"projector first dimension mismatch: expected {norb}, got {projector.shape[0]}"
            )
        if projector.shape[2] != ns:
            raise ValueError(
                f"projector spin dimension mismatch: expected {ns}, got {projector.shape[2]}"
            )

        norbc = projector.shape[1]
        return norb, ns, norbc

    @staticmethod
    def FLocStc(ff: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Project local static fermionic quantity.

        Parameters
        ----------
        ff : ndarray[norb, norb, ns]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norbc, norbc, ns]
        """
        _, ns, norbc = Projection._validate_fp(ff, projector, ff_ndim=3)
        result = Projection._allocate_output((norbc, norbc, ns), ff, out)

        for is_ in range(ns):
            p = projector[:, :, is_]
            result[:, :, is_] = p.conj().T @ ff[:, :, is_] @ p

        return result

    @staticmethod
    def FLatStc(ff: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Project lattice static fermionic quantity and k-average.

        Parameters
        ----------
        ff : ndarray[norb, norb, ns, nk]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norbc, norbc, ns]
        """
        _, ns, norbc = Projection._validate_fp(ff, projector, ff_ndim=4)
        nk = ff.shape[3]
        result = Projection._allocate_output((norbc, norbc, ns), ff, out)

        for ik in range(nk):
            result += Projection.FLocStc(ff[:, :, :, ik], projector)

        result /= float(nk)
        return result

    @staticmethod
    def FLocDyn(ff: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Project local dynamic fermionic quantity.

        Parameters
        ----------
        ff : ndarray[norb, norb, ns, nf]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norbc, norbc, ns, nf]
        """
        _, ns, norbc = Projection._validate_fp(ff, projector, ff_ndim=4)
        nf = ff.shape[3]
        result = Projection._allocate_output((norbc, norbc, ns, nf), ff, out)

        for ifreq in range(nf):
            result[:, :, :, ifreq] = Projection.FLocStc(ff[:, :, :, ifreq], projector)

        return result

    @staticmethod
    def FLatDyn(ff: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Project lattice dynamic fermionic quantity and k-average.

        Parameters
        ----------
        ff : ndarray[norb, norb, ns, nk, nf]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norbc, norbc, ns, nf]
        """
        _, ns, norbc = Projection._validate_fp(ff, projector, ff_ndim=5)
        nf = ff.shape[4]
        result = Projection._allocate_output((norbc, norbc, ns, nf), ff, out)

        for ifreq in range(nf):
            result[:, :, :, ifreq] = Projection.FLatStc(ff[:, :, :, :, ifreq], projector)

        return result

    @staticmethod
    def BLocStc(ff: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Project local static bosonic quantity.

        Parameters
        ----------
        ff : ndarray[norb, norb, ns, ns]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norbc, norbc, ns, ns]
        """
        _, ns, norbc = Projection._validate_bp(ff, projector, ff_ndim=4)
        result = Projection._allocate_output((norbc, norbc, ns, ns), ff, out)

        for is_ in range(ns):
            pl = projector[:, :, is_]
            for js in range(ns):
                pr = projector[:, :, js]
                result[:, :, is_, js] = pl.conj().T @ ff[:, :, is_, js] @ pr

        return result

    @staticmethod
    def BLatStc(ff: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Project lattice static bosonic quantity and k-average.

        Parameters
        ----------
        ff : ndarray[norb, norb, ns, ns, nk]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norbc, norbc, ns, ns]
        """
        _, ns, norbc = Projection._validate_bp(ff, projector, ff_ndim=5)
        nk = ff.shape[4]
        result = Projection._allocate_output((norbc, norbc, ns, ns), ff, out)

        for ik in range(nk):
            result += Projection.BLocStc(ff[:, :, :, :, ik], projector)

        result /= float(nk)
        return result

    @staticmethod
    def BLocDyn(ff: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Project local dynamic bosonic quantity.

        Parameters
        ----------
        ff : ndarray[norb, norb, ns, ns, nf]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norbc, norbc, ns, ns, nf]
        """
        _, ns, norbc = Projection._validate_bp(ff, projector, ff_ndim=5)
        nf = ff.shape[4]
        result = Projection._allocate_output((norbc, norbc, ns, ns, nf), ff, out)

        for ifreq in range(nf):
            result[:, :, :, :, ifreq] = Projection.BLocStc(ff[:, :, :, :, ifreq], projector)

        return result

    @staticmethod
    def BLatDyn(ff: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Project lattice dynamic bosonic quantity and k-average.

        Parameters
        ----------
        ff : ndarray[norb, norb, ns, ns, nk, nf]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norbc, norbc, ns, ns, nf]
        """
        _, ns, norbc = Projection._validate_bp(ff, projector, ff_ndim=6)
        nf = ff.shape[5]
        result = Projection._allocate_output((norbc, norbc, ns, ns, nf), ff, out)

        for ifreq in range(nf):
            result[:, :, :, :, ifreq] = Projection.BLatStc(ff[:, :, :, :, :, ifreq], projector)

        return result
