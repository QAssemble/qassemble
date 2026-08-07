"""
Embedding utilities translated from `modules/Embedding.f90`.

Each method performs basis embedding with spin-resolved projectors:
    out = P * in * P^dagger
"""

from typing import Optional, Tuple

import numpy as np


class Embedding:
    """Static embedding kernels for fermionic and bosonic objects."""

    @staticmethod
    def _allocate_output(
        shape: Tuple[int, ...], reference: np.ndarray, out: Optional[np.ndarray]
    ) -> np.ndarray:
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
    def _validate_fe(
        ffc: np.ndarray, projector: np.ndarray, ffc_ndim: int
    ) -> Tuple[int, int, int]:
        if ffc.ndim != ffc_ndim:
            raise ValueError(f"ffc must be {ffc_ndim}D, got {ffc.ndim}D")
        if projector.ndim != 3:
            raise ValueError(f"projector must be 3D, got {projector.ndim}D")

        norbc, norbc2, ns = ffc.shape[:3]
        if norbc != norbc2:
            raise ValueError("ffc first two dimensions must be square (norbc, norbc)")

        norb, norbc_p, ns_p = projector.shape
        if norbc_p != norbc:
            raise ValueError(
                f"projector norbc mismatch: expected {norbc}, got {norbc_p}"
            )
        if ns_p != ns:
            raise ValueError(f"projector spin mismatch: expected {ns}, got {ns_p}")

        return norb, ns, norbc

    @staticmethod
    def _validate_be(
        ffc: np.ndarray, projector: np.ndarray, ffc_ndim: int
    ) -> Tuple[int, int, int]:
        if ffc.ndim != ffc_ndim:
            raise ValueError(f"ffc must be {ffc_ndim}D, got {ffc.ndim}D")
        if projector.ndim != 3:
            raise ValueError(f"projector must be 3D, got {projector.ndim}D")

        norbc, norbc2, ns, ns2 = ffc.shape[:4]
        if norbc != norbc2:
            raise ValueError("ffc first two dimensions must be square (norbc, norbc)")
        if ns != ns2:
            raise ValueError("ffc spin block must be square (ns, ns)")

        norb, norbc_p, ns_p = projector.shape
        if norbc_p != norbc:
            raise ValueError(
                f"projector norbc mismatch: expected {norbc}, got {norbc_p}"
            )
        if ns_p != ns:
            raise ValueError(f"projector spin mismatch: expected {ns}, got {ns_p}")

        return norb, ns, norbc

    @staticmethod
    def FLocStc(
        ffc: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Embed local static fermionic quantity.

        Parameters
        ----------
        ffc : ndarray[norbc, norbc, ns]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norb, norb, ns]
        """
        norb, ns, _ = Embedding._validate_fe(ffc, projector, ffc_ndim=3)
        result = Embedding._allocate_output((norb, norb, ns), ffc, out)

        for is_ in range(ns):
            p = projector[:, :, is_]
            result[:, :, is_] = p @ ffc[:, :, is_] @ p.conj().T

        return result

    @staticmethod
    def FLatStc(
        ffc: np.ndarray,
        projector: np.ndarray,
        nk: int,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Embed static fermionic quantity onto all k-points.

        Parameters
        ----------
        ffc : ndarray[norbc, norbc, ns]
        projector : ndarray[norb, norbc, ns]
        nk : int
            Number of k-points.
        out : optional ndarray[norb, norb, ns, nk]
        """
        norb, ns, _ = Embedding._validate_fe(ffc, projector, ffc_ndim=3)
        if nk <= 0:
            raise ValueError("nk must be a positive integer")

        result = Embedding._allocate_output((norb, norb, ns, nk), ffc, out)
        local = Embedding.FLocStc(ffc, projector)

        for ik in range(nk):
            result[:, :, :, ik] = local

        return result

    @staticmethod
    def FLocDyn(
        ffc: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Embed local dynamic fermionic quantity.

        Parameters
        ----------
        ffc : ndarray[norbc, norbc, ns, nf]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norb, norb, ns, nf]
        """
        norb, ns, _ = Embedding._validate_fe(ffc, projector, ffc_ndim=4)
        nf = ffc.shape[3]
        result = Embedding._allocate_output((norb, norb, ns, nf), ffc, out)

        for ifreq in range(nf):
            result[:, :, :, ifreq] = Embedding.FLocStc(ffc[:, :, :, ifreq], projector)

        return result

    @staticmethod
    def FLatDyn(
        ffc: np.ndarray,
        projector: np.ndarray,
        nk: int,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Embed dynamic fermionic quantity onto all k-points.

        Parameters
        ----------
        ffc : ndarray[norbc, norbc, ns, nf]
        projector : ndarray[norb, norbc, ns]
        nk : int
            Number of k-points.
        out : optional ndarray[norb, norb, ns, nk, nf]
        """
        norb, ns, _ = Embedding._validate_fe(ffc, projector, ffc_ndim=4)
        if nk <= 0:
            raise ValueError("nk must be a positive integer")

        nf = ffc.shape[3]
        result = Embedding._allocate_output((norb, norb, ns, nk, nf), ffc, out)

        for ifreq in range(nf):
            result[:, :, :, :, ifreq] = Embedding.FLatStc(
                ffc[:, :, :, ifreq], projector, nk
            )

        return result

    @staticmethod
    def BLocStc(
        ffc: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Embed local static bosonic quantity.

        Parameters
        ----------
        ffc : ndarray[norbc, norbc, ns, ns]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norb, norb, ns, ns]
        """
        norb, ns, _ = Embedding._validate_be(ffc, projector, ffc_ndim=4)
        result = Embedding._allocate_output((norb, norb, ns, ns), ffc, out)

        for is_ in range(ns):
            pl = projector[:, :, is_]
            for js in range(ns):
                pr = projector[:, :, js]
                result[:, :, is_, js] = pl @ ffc[:, :, is_, js] @ pr.conj().T

        return result

    @staticmethod
    def BLatStc(
        ffc: np.ndarray,
        projector: np.ndarray,
        nk: int,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Embed static bosonic quantity onto all k-points.

        Parameters
        ----------
        ffc : ndarray[norbc, norbc, ns, ns]
        projector : ndarray[norb, norbc, ns]
        nk : int
            Number of k-points.
        out : optional ndarray[norb, norb, ns, ns, nk]
        """
        norb, ns, _ = Embedding._validate_be(ffc, projector, ffc_ndim=4)
        if nk <= 0:
            raise ValueError("nk must be a positive integer")

        result = Embedding._allocate_output((norb, norb, ns, ns, nk), ffc, out)
        local = Embedding.BLocStc(ffc, projector)

        for ik in range(nk):
            result[:, :, :, :, ik] = local

        return result

    @staticmethod
    def BLocDyn(
        ffc: np.ndarray, projector: np.ndarray, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Embed local dynamic bosonic quantity.

        Parameters
        ----------
        ffc : ndarray[norbc, norbc, ns, ns, nf]
        projector : ndarray[norb, norbc, ns]
        out : optional ndarray[norb, norb, ns, ns, nf]
        """
        norb, ns, _ = Embedding._validate_be(ffc, projector, ffc_ndim=5)
        nf = ffc.shape[4]
        result = Embedding._allocate_output((norb, norb, ns, ns, nf), ffc, out)

        for ifreq in range(nf):
            result[:, :, :, :, ifreq] = Embedding.BLocStc(
                ffc[:, :, :, :, ifreq], projector
            )

        return result

    @staticmethod
    def BLatDyn(
        ffc: np.ndarray,
        projector: np.ndarray,
        nk: int,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Embed dynamic bosonic quantity onto all k-points.

        Parameters
        ----------
        ffc : ndarray[norbc, norbc, ns, ns, nf]
        projector : ndarray[norb, norbc, ns]
        nk : int
            Number of k-points.
        out : optional ndarray[norb, norb, ns, ns, nk, nf]
        """
        norb, ns, _ = Embedding._validate_be(ffc, projector, ffc_ndim=5)
        if nk <= 0:
            raise ValueError("nk must be a positive integer")

        nf = ffc.shape[4]
        result = Embedding._allocate_output((norb, norb, ns, ns, nk, nf), ffc, out)

        for ifreq in range(nf):
            result[:, :, :, :, :, ifreq] = Embedding.BLatStc(
                ffc[:, :, :, :, ifreq], projector, nk
            )

        return result
