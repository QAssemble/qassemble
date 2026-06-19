"""Boson Matsubara<->tau round-trip tests (F2T / T2F).

`BLocDyn.F2T` was missing the `bosonic_corr_x` divide that `BLatDyn.F2T`,
`DLR.BF2T`, and pydlr's `dlr_from_matsubara` (xi=1) all apply — so a boson
Matsubara->tau transform on the local path was off by `1/bosonic_corr_x` per
DLR pole, silently corrupting the tau representation of Chi / PImp / BWeiss.
The four near-duplicate F2T/T2F methods (BLocDyn, BLatDyn) now share the
`DLR.BatchBF2T` / `DLR.BatchBT2F` cores, so the correction can no longer
diverge between them.

These tests pin:

A. `BatchBF2T` / `BatchBT2F` are exact inverses (T2F∘F2T = identity at machine
   precision).  Before the fix this fails — F2T lacked the `corr_x` divide that
   T2F multiplies back in, so the round trip was off by `corr_x**2` per pole.
B. `BLocDyn.F2T` matches the trusted 1D `DLR.BF2T` channel-by-channel (and now
   equals `BLatDyn.F2T` on the same data), directly proving the corr_x fix.
"""

import numpy as np

from QAssemble.Crystal import Crystal
from QAssemble.utility.DLR import DLR
from QAssemble.FLocDyn import FLocDyn  # noqa: F401  (ensures package import side effects)
from QAssemble.BLocDyn import BLocDyn
from QAssemble.BLatDyn import BLatDyn


def _dlr():
    return DLR({"beta": 20.0, "cutoff": 8.0, "eps": 1.0e-12})


def _crystal():
    return Crystal(
        {
            "RVec": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "Basis": [[[0, 0, 0], 1]],
            "NSpin": 1,
            "NElec": 1.0,
            "KGrid": [2, 2, 2],
        }
    )


def _known_bf_2d(dlr, nbatch, seed):
    """Build boson Matsubara data on the DLR nu grid from decaying DLR coeffs."""
    rank = dlr.dB.rank
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((rank, nbatch)) * np.exp(-np.arange(rank)[:, None])
    # evaluate on the DLR nu grid: bf = beta * (T_qx * corr_x) @ G
    return dlr.beta * (dlr.dB.T_qx * dlr.dB.bosonic_corr_x[None, :]) @ G


# ---------------------------------------------------------------------------
# A. helper round trip is identity
# ---------------------------------------------------------------------------


def test_batch_boson_matsubara_tau_roundtrip_is_identity():
    dlr = _dlr()
    bf_2d = _known_bf_2d(dlr, nbatch=4, seed=0)

    btau_2d = dlr.BatchBF2T(bf_2d)
    bf_back = dlr.BatchBT2F(btau_2d)

    np.testing.assert_allclose(bf_back, bf_2d, atol=1.0e-10, rtol=1.0e-10)


def test_batch_bf2t_matches_1d_bf2t_per_column():
    dlr = _dlr()
    bf_2d = _known_bf_2d(dlr, nbatch=3, seed=1)

    batched = dlr.BatchBF2T(bf_2d)
    for k in range(bf_2d.shape[1]):
        ref = dlr.BF2T(bf_2d[:, k])
        np.testing.assert_allclose(batched[:, k], ref, atol=1.0e-12, rtol=1.0e-12)


def test_batch_bt2f_matches_1d_bt2f_per_column():
    dlr = _dlr()
    bf_2d = _known_bf_2d(dlr, nbatch=3, seed=4)
    btau_2d = dlr.BatchBF2T(bf_2d)  # known tau data on the DLR tau grid

    batched = dlr.BatchBT2F(btau_2d)
    for k in range(btau_2d.shape[1]):
        ref = dlr.BT2F(btau_2d[:, k])
        np.testing.assert_allclose(batched[:, k], ref, atol=1.0e-12, rtol=1.0e-12)


# ---------------------------------------------------------------------------
# B. BLocDyn / BLatDyn F2T match the trusted 1D reference (proves corr_x fix)
# ---------------------------------------------------------------------------


def test_blocdyn_f2t_matches_dlr_bf2t_per_channel():
    dlr = _dlr()
    crystal = _crystal()
    bloc = BLocDyn(crystal, dlr, projector=None)

    norb, ns = 2, 1
    nfreq = len(dlr.nu)
    rng = np.random.default_rng(2)
    bf = np.zeros((norb, norb, ns, ns, nfreq), dtype=np.complex128, order="F")
    refs = {}
    for i in range(norb):
        for j in range(norb):
            col = _known_bf_2d(dlr, nbatch=1, seed=10 * i + j)[:, 0]
            bf[i, j, 0, 0, :] = col
            refs[(i, j)] = dlr.BF2T(col)

    btau = bloc.F2T(bf)
    assert btau.shape[-1] == len(dlr.tauB)
    for (i, j), ref in refs.items():
        np.testing.assert_allclose(btau[i, j, 0, 0, :], ref, atol=1.0e-12, rtol=1.0e-12)


def test_blocdyn_f2t_t2f_roundtrip_is_identity():
    dlr = _dlr()
    crystal = _crystal()
    bloc = BLocDyn(crystal, dlr, projector=None)

    norb, ns = 2, 1
    nfreq = len(dlr.nu)
    bf = np.zeros((norb, norb, ns, ns, nfreq), dtype=np.complex128, order="F")
    for i in range(norb):
        for j in range(norb):
            bf[i, j, 0, 0, :] = _known_bf_2d(dlr, nbatch=1, seed=100 * i + j)[:, 0]

    bf_back = bloc.T2F(bloc.F2T(bf))
    np.testing.assert_allclose(bf_back, bf, atol=1.0e-10, rtol=1.0e-10)


def test_blocdyn_and_blatdyn_f2t_agree_on_shared_channel():
    dlr = _dlr()
    crystal = _crystal()
    bloc = BLocDyn(crystal, dlr, projector=None)
    blat = BLatDyn(crystal, dlr)

    norb, ns, nrk = 2, 1, 1
    nfreq = len(dlr.nu)
    rng = np.random.default_rng(3)
    bf_loc = np.zeros((norb, norb, ns, ns, nfreq), dtype=np.complex128, order="F")
    for i in range(norb):
        for j in range(norb):
            bf_loc[i, j, 0, 0, :] = _known_bf_2d(dlr, nbatch=1, seed=5 * i + j)[:, 0]

    # same data with a trivial k-axis for the lattice method
    bf_lat = bf_loc[:, :, :, :, np.newaxis, :]

    btau_loc = bloc.F2T(bf_loc)
    btau_lat = blat.F2T(np.asfortranarray(bf_lat))
    np.testing.assert_allclose(
        btau_lat[:, :, :, :, 0, :], btau_loc, atol=1.0e-12, rtol=1.0e-12
    )


def test_blatdyn_f2t_t2f_roundtrip_multi_k_multi_spin():
    """Stress the BLatDyn flatten/reshape with nrk>1 and ns>1.

    BLatDyn.F2T/T2F read their dimensions from the input shape (not crystal.ns),
    so a (norb,norb,ns,ns,nrk,nfreq) array with ns=2, nrk=3 exercises the full
    batch path; T2F∘F2T must be the identity.
    """
    dlr = _dlr()
    crystal = _crystal()
    blat = BLatDyn(crystal, dlr)

    norb, ns, nrk = 2, 2, 3
    nfreq = len(dlr.nu)
    nbatch = norb * norb * ns * ns * nrk
    flat = _known_bf_2d(dlr, nbatch=nbatch, seed=7)  # (nfreq, batch)
    bf = np.moveaxis(flat.reshape(nfreq, norb, norb, ns, ns, nrk), 0, -1)
    bf = np.asfortranarray(bf)

    bf_back = blat.T2F(blat.F2T(bf))
    assert bf_back.shape == bf.shape
    np.testing.assert_allclose(bf_back, bf, atol=1.0e-10, rtol=1.0e-10)
