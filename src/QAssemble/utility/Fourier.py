
"""
This module provides a collection of static methods for performing Fourier transforms
between the imaginary-time and Matsubara frequency domains for Green's functions and
self-energies.

Classes
-------
Fourier
    Serial (CPU-only) Fourier transforms.
FourierMPI
    MPI-parallel Fourier transforms using mpi4py_fft (PFFT).
"""
import numpy as np
from Common import Common
from numba import jit
import finufft
from scipy.linalg import solve
from scipy.fftpack import fftn, ifftn


class Fourier:
    """
    Serial Fourier transforms on Green's functions.

    Covers:
    - Local / lattice, fermionic / bosonic dynamic transforms (T↔F via NUFFT)
    - High-frequency moment extraction
    - Static K↔R transforms via scipy FFT
    """

    # ---------------------------------------------------------------------------
    # Local dynamic transforms  (NUFFT-based)
    # ---------------------------------------------------------------------------

    @staticmethod
    def FLocDynT2F(tau: np.ndarray, ftau: np.ndarray, freq: np.ndarray) -> np.ndarray:
        """τ → iω  for a local fermionic Green's function  (norb, norb, ns, ntau)."""

        norb, _, ns, ntau = ftau.shape
        nfreq = len(freq)
        pi = np.pi
        beta = pi / freq[0]

        ntau_finu  = 2 * ntau
        nfreq_finu = 4 * nfreq - 1

        ff = np.zeros((norb, norb, ns, nfreq), dtype=np.complex128, order='F')

        taurad_finu = np.zeros((ntau_finu), dtype=np.float64, order='F')
        for itau in range(ntau):
            taurad_finu[itau + ntau]     =  tau[itau] / beta * pi
            taurad_finu[ntau - itau - 1] = -taurad_finu[itau + ntau]

        for iorb in range(norb):
            for jorb in range(norb):
                for js in range(ns):
                    ftau_finu = np.zeros((ntau_finu), dtype=np.complex128, order='F')
                    for itau in range(ntau):
                        ftau_finu[itau + ntau] = ftau[iorb, jorb, js, itau] * \
                            np.sqrt(tau[itau] * (beta - tau[itau])) * pi / ntau
                        ftau_finu[itau] = -ftau_finu[itau + ntau]

                    ff_finu  = finufft.nufft1d1(taurad_finu, ftau_finu, nfreq_finu,
                                                 isign=1, eps=1e-12, nthreads=1)
                    k0_index = (nfreq_finu - 1) // 2
                    for ifreq in range(nfreq * 2):
                        if ifreq % 2 == 1:
                            ff[iorb, jorb, js, (ifreq - 1) // 2] = ff_finu[k0_index + ifreq] / 2.0

        return ff

    @staticmethod
    def FLatDynT2F(tau: np.ndarray, ftau: np.ndarray, freq: np.ndarray) -> np.ndarray:
        """τ → iω  for a lattice fermionic Green's function  (norb, norb, ns, nk, ntau)."""

        norb, _, ns, nk, _ = ftau.shape
        nfreq = len(freq)
        ff = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')

        for ik in range(nk):
            ff[..., ik, :] = Fourier.FLocDynT2F(tau, ftau[..., ik, :], freq)

        return ff

    @staticmethod
    def FLocDynF2T(freq: np.ndarray, ff: np.ndarray,
                   moment: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """iω → τ  for a local fermionic Green's function  (norb, norb, ns, nfreq)."""

        pi   = np.pi
        beta = pi / freq[0]

        norb, _, ns, nfreq = ff.shape
        ntau       = len(tau)
        ftau       = np.zeros((norb, norb, ns, ntau), dtype=np.complex128, order='F')
        ntau_finu  = ntau
        nfreq_finu = nfreq * 4 - 1

        momega_finu = np.zeros((nfreq_finu, 3), dtype=np.complex128, order='F')
        for ifreq in range(-2 * nfreq + 1, 2 * nfreq):
            if ifreq % 2 == 1:
                momega_finu[ifreq + 2 * nfreq - 1, 0] = 1.0 / (pi / beta * ifreq * 1j)
                momega_finu[ifreq + 2 * nfreq - 1, 1] = 1.0 / (pi / beta * ifreq * 1j) ** 2
                momega_finu[ifreq + 2 * nfreq - 1, 2] = 1.0 / (pi / beta * ifreq * 1j) ** 3

        taurad_finu = tau / beta * pi
        mtau_finu   = np.zeros((ntau_finu, 3), dtype=np.complex128, order='F')
        for ii in range(3):
            mtau_finu[:, ii] = finufft.nufft1d2(taurad_finu, momega_finu[:, ii],
                                                  isign=-1, eps=1e-12, nthreads=1)

        for js in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    ff_finu = np.zeros((nfreq_finu), dtype=np.complex128, order='F')
                    for ifreq in range(-2 * nfreq + 1, 2 * nfreq):
                        if ifreq % 2 == 1:
                            if ifreq > 0:
                                ff_finu[ifreq + 2 * nfreq - 1] = ff[iorb, jorb, js, (ifreq - 1) // 2]
                            else:
                                ff_finu[ifreq + 2 * nfreq - 1] = np.conjugate(ff[jorb, iorb, js, (-ifreq - 1) // 2])

                    ftau_finu = finufft.nufft1d2(taurad_finu, ff_finu, isign=-1, eps=1e-12, nthreads=1)

                    for itau in range(ntau):
                        xx = tau[itau] / beta
                        ftau[iorb, jorb, js, itau] = ftau_finu[itau] / beta
                        for ii in range(3):
                            ftau[iorb, jorb, js, itau] -= (moment[iorb, jorb, js, ii] *
                                                            mtau_finu[itau, ii] / beta)
                            ftau[iorb, jorb, js, itau] += (0.5 * beta ** ii /
                                                            Common.FactorialInt(ii) *
                                                            (-1) ** (ii + 1) *
                                                            Common.EulerPolynomial(xx, ii) *
                                                            moment[iorb, jorb, js, ii])

        return ftau

    @staticmethod
    def FLatDynF2T(freq: np.ndarray, ff: np.ndarray,
                   moment: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """iω → τ  for a lattice fermionic Green's function  (norb, norb, ns, nk, nfreq)."""

        norb, _, ns, nk, _ = ff.shape
        ntau = len(tau)
        ftau = np.zeros((norb, norb, ns, nk, ntau), dtype=np.complex128, order='F')

        for ik in range(nk):
            ftau[..., ik, :] = Fourier.FLocDynF2T(freq, ff[..., ik, :], moment[..., ik, :], tau)

        return ftau

    # ---------------------------------------------------------------------------
    # High-frequency moment extraction
    # ---------------------------------------------------------------------------

    @staticmethod
    def FLocDynM(freq: np.ndarray, ff1: np.ndarray, ff2: np.ndarray,
                 isgreen: bool, highzero: bool) -> tuple:
        """High-frequency moments for a local fermionic Green's function."""

        norb, _, ns = ff1.shape
        nfreq  = len(freq)
        moment = np.zeros((norb, norb, ns, 3), dtype=np.complex128, order='F')
        high   = np.zeros((norb, norb, ns),    dtype=np.complex128, order='F')
        ai = 1j

        if isgreen:
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        moment[iorb, jorb, js, 0] = 1.0 if iorb == jorb else 0.0
                        moment[iorb, jorb, js, 1] = ((ff1[iorb, jorb, js] +
                                                       np.conj(ff1[jorb, iorb, js])) / 2.0 *
                                                      (freq[nfreq - 1] * ai) ** 2)
                        moment[iorb, jorb, js, 2] = ((ff1[iorb, jorb, js] -
                                                       np.conj(ff1[jorb, iorb, js]) -
                                                       moment[iorb, jorb, js, 0] * 2.0 /
                                                       (freq[nfreq - 1] * ai)) / 2.0 *
                                                      (freq[nfreq - 1] * ai) ** 3)
        else:
            if highzero:
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            moment[iorb, jorb, js, 0] = ((ff1[iorb, jorb, js] -
                                                           np.conjugate(ff1[jorb, iorb, js])) / 2.0 *
                                                          (freq[nfreq - 1] * ai))
                            moment[iorb, jorb, js, 1] = ((ff1[iorb, jorb, js] +
                                                           np.conjugate(ff1[jorb, iorb, js])) / 2.0 *
                                                          (freq[nfreq - 1] * ai) ** 2)
            else:
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            amat = np.zeros((4, 4), dtype=np.complex128, order='F')
                            bmat = np.zeros((4, 1), dtype=np.complex128, order='F')
                            amat[0, :] = [1.0,  1.0 / (freq[nfreq-1]*ai), 1.0 / (freq[nfreq-1]*ai)**2,  1.0 / (freq[nfreq-1]*ai)**3]
                            amat[1, :] = [1.0, -1.0 / (freq[nfreq-1]*ai), 1.0 / (freq[nfreq-1]*ai)**2, -1.0 / (freq[nfreq-1]*ai)**3]
                            amat[2, :] = [1.0,  1.0 / (freq[nfreq-2]*ai), 1.0 / (freq[nfreq-2]*ai)**2,  1.0 / (freq[nfreq-2]*ai)**3]
                            amat[3, :] = [1.0, -1.0 / (freq[nfreq-2]*ai), 1.0 / (freq[nfreq-2]*ai)**2, -1.0 / (freq[nfreq-2]*ai)**3]
                            bmat[0, 0] = ff1[iorb, jorb, js]
                            bmat[1, 0] = np.conjugate(ff1[jorb, iorb, js])
                            bmat[2, 0] = ff2[iorb, jorb, js]
                            bmat[3, 0] = np.conjugate(ff2[jorb, iorb, js])
                            sol = solve(amat, bmat)
                            high[iorb, jorb, js]    = sol[0, 0]
                            moment[iorb, jorb, js, 0] = sol[1, 0]
                            moment[iorb, jorb, js, 1] = sol[2, 0]
                            moment[iorb, jorb, js, 2] = sol[3, 0]

        for js in range(ns):
            high[:, :, js] = (high[:, :, js].T.conj() + high[:, :, js]) / 2.0
            for ii in range(3):
                moment[:, :, js, ii] = (moment[:, :, js, ii].T.conj() + moment[:, :, js, ii]) / 2.0

        return moment, high

    @staticmethod
    def FLatDynM(freq: np.ndarray, ff1: np.ndarray, ff2: np.ndarray,
                 isgreen: bool, highzero: bool) -> tuple:
        """High-frequency moments for a lattice fermionic Green's function."""

        norb, _, ns, nk = ff1.shape
        moment = np.zeros((norb, norb, ns, nk, 3), dtype=np.complex128, order='F')
        high   = np.zeros((norb, norb, ns, nk),    dtype=np.complex128, order='F')

        for ik in range(nk):
            moment[..., ik, :], high[..., ik] = Fourier.FLocDynM(
                freq, ff1[..., ik], ff2[..., ik], isgreen, highzero)

        return moment, high

    @staticmethod
    def BLocDynM(freq: np.ndarray, ff: np.ndarray, oddzero: bool, highzero: bool) -> tuple:
        """High-frequency moments for a local bosonic Green's function."""

        ai = 1j
        norb, _, ns, _, _ = ff.shape
        moment = np.zeros((norb, norb, ns, ns, 3), dtype=np.complex128, order='F')
        high   = np.zeros((norb, norb, ns, ns),    dtype=np.complex128, order='F')

        if oddzero:
            if highzero:
                moment[..., 1] = ff[..., -1] * (freq[-1] * ai) ** 2
            else:
                moment[..., 1] = ((ff[..., -1] - ff[..., -2]) * -1.0 *
                                   (freq[-1] * ai * freq[-2] * ai) ** 2 /
                                   ((freq[-1] * ai + freq[-2] * ai) *
                                    (freq[-1] * ai - freq[-2] * ai)))
                high = ff[..., -1] - moment[..., 1] / (freq[-1] * ai) ** 2
        else:
            if highzero:
                for is_ in range(ns):
                    for js in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                moment[iorb, jorb, is_, js, 0] += ((ff[iorb, jorb, is_, js, -1] -
                                                                     np.conj(ff[jorb, iorb, js, is_, -1])) /
                                                                    (2.0 * (freq[-1] * ai)))
                                moment[iorb, jorb, is_, js, 1] += ((ff[iorb, jorb, is_, js, -1] +
                                                                     np.conj(ff[jorb, iorb, js, is_, -1])) /
                                                                    (2.0 * (freq[-1] * ai) ** 2))
            else:
                amat = np.zeros((4, 4), dtype=np.complex128)
                bmat = np.zeros((4, 1), dtype=np.complex128)
                for is_ in range(ns):
                    for js in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                amat[0, :] = [1.0,  1.0 / (freq[-1]*ai), 1.0 / (freq[-1]*ai)**2,  1.0 / (freq[-1]*ai)**3]
                                amat[1, :] = [1.0, -1.0 / (freq[-1]*ai), 1.0 / (freq[-1]*ai)**2, -1.0 / (freq[-1]*ai)**3]
                                amat[2, :] = [1.0,  1.0 / (freq[-2]*ai), 1.0 / (freq[-2]*ai)**2,  1.0 / (freq[-2]*ai)**3]
                                amat[3, :] = [1.0, -1.0 / (freq[-2]*ai), 1.0 / (freq[-2]*ai)**2, -1.0 / (freq[-2]*ai)**3]
                                bmat[0, 0] = ff[iorb, jorb, is_, js, -1]
                                bmat[1, 0] = np.conj(ff[jorb, iorb, js, is_, -1])
                                bmat[2, 0] = ff[iorb, jorb, is_, js, -2]
                                bmat[3, 0] = np.conj(ff[jorb, iorb, js, is_, -2])
                                x = np.linalg.solve(amat, bmat)
                                high[iorb, jorb, is_, js]    = x[0, 0]
                                moment[iorb, jorb, is_, js, 0] = x[1, 0]
                                moment[iorb, jorb, is_, js, 1] = x[2, 0]
                                moment[iorb, jorb, is_, js, 2] = x[3, 0]

        for iorb in range(norb):
            for jorb in range(norb):
                for is_ in range(ns):
                    for js in range(ns):
                        high[iorb, jorb, is_, js] = (np.conj(high[jorb, iorb, js, is_]) +
                                                      high[iorb, jorb, is_, js]) / 2.0
                        for ii in range(3):
                            moment[iorb, jorb, is_, js, ii] = (np.conj(moment[jorb, iorb, js, is_, ii]) +
                                                                moment[iorb, jorb, is_, js, ii]) / 2.0

        return moment, high

    @staticmethod
    def BLatDynM(freq: np.ndarray, ff: np.ndarray, oddzero: bool, highzero: bool) -> tuple:
        """High-frequency moments for a lattice bosonic Green's function."""

        norb, _, ns, _, nk, _ = ff.shape
        moment = np.zeros((norb, norb, ns, ns, nk, 3), dtype=np.complex128, order='F')
        high   = np.zeros((norb, norb, ns, ns, nk),    dtype=np.complex128, order='F')

        for ik in range(nk):
            moment[..., ik, :], high[..., ik] = Fourier.BLocDynM(freq, ff[..., ik, :], oddzero, highzero)

        return moment, high

    # ---------------------------------------------------------------------------
    # Serial K↔R Fourier transforms  (scipy FFT)
    # ---------------------------------------------------------------------------

    @staticmethod
    def FLatStcK2R(fin: np.ndarray, rkgrid: list) -> np.ndarray:
        """K→R  (serial, fermionic)  shape: (norb, norb, ns, nk)."""
        norb, _, ns, nk = fin.shape
        fout = np.zeros((norb, norb, ns, nk), dtype=np.complex128, order='F')

        for js in range(ns):
            for jorb in range(norb):
                for iorb in range(norb):
                    temp3d = fin[iorb, jorb, js, :].reshape(rkgrid, order='F')
                    fout[iorb, jorb, js, :] = ifftn(temp3d).reshape(nk, order='F') * nk

        return fout

    @staticmethod
    def FLatStcR2K(fin: np.ndarray, rkgrid: list) -> np.ndarray:
        """R→K  (serial, fermionic)  shape: (norb, norb, ns, nr)."""
        norb, _, ns, nr = fin.shape
        fout = np.zeros((norb, norb, ns, nr), dtype=np.complex128, order='F')

        for js in range(ns):
            for jorb in range(norb):
                for iorb in range(norb):
                    temp3d = fin[iorb, jorb, js, :].reshape(rkgrid, order='F')
                    fout[iorb, jorb, js, :] = fftn(temp3d).reshape(nr, order='F') / nr

        return fout

    @staticmethod
    def BLatStcK2R(fin: np.ndarray, rkgrid: list) -> np.ndarray:
        """K→R  (serial, bosonic)  shape: (norb, norb, ns, ns, nk)."""
        norb, _, ns, _, nk = fin.shape
        fout = np.zeros((norb, norb, ns, ns, nk), dtype=np.complex128, order='F')

        for ks in range(ns):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        temp3d = fin[iorb, jorb, js, ks, :].reshape(rkgrid, order='F')
                        fout[iorb, jorb, js, ks, :] = ifftn(temp3d).reshape(nk, order='F') * nk

        return fout

    @staticmethod
    def BLatStcR2K(fin: np.ndarray, rkgrid: list) -> np.ndarray:
        """R→K  (serial, bosonic)  shape: (norb, norb, ns, ns, nr)."""
        norb, _, ns, _, nr = fin.shape
        fout = np.zeros((norb, norb, ns, ns, nr), dtype=np.complex128, order='F')

        for ks in range(ns):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        temp3d = fin[iorb, jorb, js, ks, :].reshape(rkgrid, order='F')
                        fout[iorb, jorb, js, ks, :] = fftn(temp3d).reshape(nr, order='F') / nr

        return fout

    # ---------------------------------------------------------------------------
    # MPI K↔R delegates  (forward to FourierMPI, keeping _MPI naming convention)
    # ---------------------------------------------------------------------------

    @staticmethod
    def FLatStcK2R_MPI(fin: np.ndarray, nodedict: dict) -> np.ndarray:
        """K→R  (MPI, fermionic) — delegates to FourierMPI.FLatStcK2R."""
        return FourierMPI.FLatStcK2R(fin, nodedict)

    @staticmethod
    def FLatStcR2K_MPI(fin: np.ndarray, nodedict: dict) -> np.ndarray:
        """R→K  (MPI, fermionic) — delegates to FourierMPI.FLatStcR2K."""
        return FourierMPI.FLatStcR2K(fin, nodedict)

    @staticmethod
    def BLatStcK2R_MPI(fin: np.ndarray, nodedict: dict) -> np.ndarray:
        """K→R  (MPI, bosonic) — delegates to FourierMPI.BLatStcK2R."""
        return FourierMPI.BLatStcK2R(fin, nodedict)

    @staticmethod
    def BLatStcR2K_MPI(fin: np.ndarray, nodedict: dict) -> np.ndarray:
        """R→K  (MPI, bosonic) — delegates to FourierMPI.BLatStcR2K."""
        return FourierMPI.BLatStcR2K(fin, nodedict)


# =============================================================================
class FourierMPI:
    """
    MPI-parallel K↔R Fourier transforms using mpi4py_fft (PFFT).

    All methods receive a *nodedict* produced by ``MPIManager.Query()``.
    Each rank operates on its local k/r slice; the full global result is
    assembled via ``MPI.Allreduce``.
    """

    @staticmethod
    def FLatStcK2R(fin: np.ndarray, nodedict: dict) -> np.ndarray:
        """K→R  (MPI, fermionic)  shape: (norb, norb, ns, nk)."""
        from mpi4py import MPI

        commk     = nodedict['commk']
        fft_obj   = nodedict['fft']
        rkgrid    = nodedict['grid']
        rank      = commk.Get_rank()
        nk_global = rkgrid[0] * rkgrid[1] * rkgrid[2]

        norb, _, ns, _ = fin.shape
        fout_local  = np.zeros((norb, norb, ns, nk_global), dtype=np.complex128, order='F')
        fout_global = np.zeros((norb, norb, ns, nk_global), dtype=np.complex128, order='F')

        kloc2glob = nodedict['kloc2glob']
        rloc2glob = nodedict['rloc2glob']

        for js in range(ns):
            for jorb in range(norb):
                for iorb in range(norb):
                    arr_fwd = fft_obj.arr.copy()
                    arr_fwd[:] = 0.0
                    for loc_idx, glob_idx in kloc2glob[rank].items():
                        arr_fwd.flat[loc_idx] = fin[iorb, jorb, js, glob_idx]

                    arr_bwd = fft_obj.Backward(arr_fwd)

                    for loc_idx, glob_idx in rloc2glob[rank].items():
                        fout_local[iorb, jorb, js, glob_idx] = arr_bwd.flat[loc_idx]

        commk.Allreduce(fout_local, fout_global, op=MPI.SUM)
        return fout_global

    @staticmethod
    def FLatStcR2K(fin: np.ndarray, nodedict: dict) -> np.ndarray:
        """R→K  (MPI, fermionic)  shape: (norb, norb, ns, nr)."""
        from mpi4py import MPI

        commk     = nodedict['commk']
        fft_obj   = nodedict['fft']
        rkgrid    = nodedict['grid']
        rank      = commk.Get_rank()
        nk_global = rkgrid[0] * rkgrid[1] * rkgrid[2]

        norb, _, ns, _ = fin.shape
        fout_local  = np.zeros((norb, norb, ns, nk_global), dtype=np.complex128, order='F')
        fout_global = np.zeros((norb, norb, ns, nk_global), dtype=np.complex128, order='F')

        rloc2glob = nodedict['rloc2glob']
        kloc2glob = nodedict['kloc2glob']

        for js in range(ns):
            for jorb in range(norb):
                for iorb in range(norb):
                    arr_bwd = fft_obj.arrT.copy()
                    arr_bwd[:] = 0.0
                    for loc_idx, glob_idx in rloc2glob[rank].items():
                        arr_bwd.flat[loc_idx] = fin[iorb, jorb, js, glob_idx]

                    arr_fwd = fft_obj.Forward(arr_bwd)

                    for loc_idx, glob_idx in kloc2glob[rank].items():
                        fout_local[iorb, jorb, js, glob_idx] = arr_fwd.flat[loc_idx] / nk_global

        commk.Allreduce(fout_local, fout_global, op=MPI.SUM)
        return fout_global

    @staticmethod
    def BLatStcK2R(fin: np.ndarray, nodedict: dict) -> np.ndarray:
        """K→R  (MPI, bosonic)  shape: (norb, norb, ns, ns, nk)."""
        from mpi4py import MPI

        commk     = nodedict['commk']
        fft_obj   = nodedict['fft']
        rkgrid    = nodedict['grid']
        rank      = commk.Get_rank()
        nk_global = rkgrid[0] * rkgrid[1] * rkgrid[2]

        norb, _, ns, _, _ = fin.shape
        fout_local  = np.zeros((norb, norb, ns, ns, nk_global), dtype=np.complex128, order='F')
        fout_global = np.zeros((norb, norb, ns, ns, nk_global), dtype=np.complex128, order='F')

        kloc2glob = nodedict['kloc2glob']
        rloc2glob = nodedict['rloc2glob']

        for ks in range(ns):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        arr_fwd = fft_obj.arr.copy()
                        arr_fwd[:] = 0.0
                        for loc_idx, glob_idx in kloc2glob[rank].items():
                            arr_fwd.flat[loc_idx] = fin[iorb, jorb, js, ks, glob_idx]

                        arr_bwd = fft_obj.Backward(arr_fwd)

                        for loc_idx, glob_idx in rloc2glob[rank].items():
                            fout_local[iorb, jorb, js, ks, glob_idx] = arr_bwd.flat[loc_idx]

        commk.Allreduce(fout_local, fout_global, op=MPI.SUM)
        return fout_global

    @staticmethod
    def BLatStcR2K(fin: np.ndarray, nodedict: dict) -> np.ndarray:
        """R→K  (MPI, bosonic)  shape: (norb, norb, ns, ns, nr)."""
        from mpi4py import MPI

        commk     = nodedict['commk']
        fft_obj   = nodedict['fft']
        rkgrid    = nodedict['grid']
        rank      = commk.Get_rank()
        nk_global = rkgrid[0] * rkgrid[1] * rkgrid[2]

        norb, _, ns, _, _ = fin.shape
        fout_local  = np.zeros((norb, norb, ns, ns, nk_global), dtype=np.complex128, order='F')
        fout_global = np.zeros((norb, norb, ns, ns, nk_global), dtype=np.complex128, order='F')

        rloc2glob = nodedict['rloc2glob']
        kloc2glob = nodedict['kloc2glob']

        for ks in range(ns):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        arr_bwd = fft_obj.arrT.copy()
                        arr_bwd[:] = 0.0
                        for loc_idx, glob_idx in rloc2glob[rank].items():
                            arr_bwd.flat[loc_idx] = fin[iorb, jorb, js, ks, glob_idx]

                        arr_fwd = fft_obj.Forward(arr_bwd)

                        for loc_idx, glob_idx in kloc2glob[rank].items():
                            fout_local[iorb, jorb, js, ks, glob_idx] = arr_fwd.flat[loc_idx] / nk_global

        commk.Allreduce(fout_local, fout_global, op=MPI.SUM)
        return fout_global

    @staticmethod
    def FPathStcR2K(fin: np.ndarray, nodedict: dict, nodedict2: dict,
                    k: np.ndarray, rvec: np.ndarray) -> np.ndarray:

        pi = np.pi
        ai = 1j

        norb, _, ns, nr = fin.shape
        rank = nodedict['commkrank']
        nk   = len(nodedict['klocal'][rank])

        fout = np.zeros((norb, norb, ns, nk), dtype=np.complex128, order='F')

        for ik in range(nk):
            tempval = 0.0
            for ir in range(nr):
                kidx = nodedict['KLocal2Global']([rank, ik],  nodedict['klocal2global'])
                ridx = nodedict2['RLocal2Global']([rank, ir], nodedict2['rlocal2global'])
                tempval = tempval + fin[..., ir] * np.exp(-2.0 * ai * pi * np.dot(k[kidx], rvec[ridx]))
            fout[..., ik] = tempval

        return fout

    @staticmethod
    def FPathDynR2K(fin: np.ndarray, nodedict: dict, nodedict2: dict,
                    k: np.ndarray, rvec: np.ndarray) -> np.ndarray:

        norb, _, ns, _, nfreq = fin.shape
        rank = nodedict['commkrank']
        nk   = len(nodedict['klocal'][rank])
        fout = np.zeros((norb, norb, ns, nk, nfreq), dtype=np.complex128, order='F')

        for ifreq in range(nfreq):
            fout[..., ifreq] = FourierMPI.FPathStcR2K(fin[..., ifreq], nodedict, nodedict2, k, rvec)

        return fout
