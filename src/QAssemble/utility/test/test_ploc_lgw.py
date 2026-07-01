from types import SimpleNamespace

import numpy as np

from QAssemble.BLocDyn import PLoc


class _FakeDLR:
    def __init__(self, ntau=4):
        self.tauF = np.arange(ntau, dtype=float)
        self.tauB = np.arange(ntau, dtype=float)
        self.nu = np.arange(ntau, dtype=float)

    def TauF2TauB(self, ftau):
        return np.asarray(ftau, dtype=np.complex128)

    def T2mT(self, ftau):
        return -np.asarray(ftau, dtype=np.complex128)[::-1]

    def BatchBT2F(self, btau_2d):
        return np.asarray(btau_2d, dtype=np.complex128)


class _FakeProjector:
    def __init__(self, ns):
        self.fprojector = {"1": np.zeros((2, 2, ns), dtype=float)}
        self.bprojector = {"1": np.zeros((4, 4, ns), dtype=float)}
        self.equiv = {"1": np.eye(4, dtype=int)}
        self._pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def ProbBorb2FPair(self, key, iorbc, ispace=0):
        assert key == "1"
        assert ispace == 0
        return self._pairs[int(iorbc)]


def _crystal(ns, soc=False):
    return SimpleNamespace(ns=ns, soc=soc)


def _green_tau(ns, ntau=4):
    gtau = np.zeros((2, 2, ns, ntau), dtype=np.complex128, order="F")
    for ispin in range(ns):
        for jorb in range(2):
            for iorb in range(2):
                base = 100 * iorb + 10 * jorb + ispin
                gtau[iorb, jorb, ispin, :] = base + np.arange(ntau) + 1j * (base + 2 * np.arange(ntau))
    return gtau


def _expected_pol_tau(gtau, pairs, ns, soc):
    ntau = gtau.shape[-1]
    nborb = len(pairs)
    left = np.array([pair[0] for pair in pairs], dtype=np.int64)
    right = np.array([pair[1] for pair in pairs], dtype=np.int64)
    gminus = -gtau[..., ::-1]

    out = np.zeros((nborb, nborb, ns, ns, ntau), dtype=np.complex128, order="F")
    if ns == 2:
        for ispin in range(ns):
            out[:, :, ispin, ispin, :] = (
                gminus[right[np.newaxis, :], left[:, np.newaxis], ispin, :]
                * gtau[right[:, np.newaxis], left[np.newaxis, :], ispin, :]
            )
    else:
        spin_factor = 1.0 if soc else 2.0
        out[:, :, 0, 0, :] = spin_factor * (
            gminus[right[np.newaxis, :], left[:, np.newaxis], 0, :]
            * gtau[right[:, np.newaxis], left[np.newaxis, :], 0, :]
        )
    return out


def test_ploc_lgw_matches_local_bubble_for_two_spin_channels():
    ns = 2
    dlr = _FakeDLR(ntau=4)
    projector = _FakeProjector(ns=ns)
    gtau = _green_tau(ns=ns, ntau=4)

    ploc = PLoc(
        crystal=_crystal(ns=ns),
        dlr=dlr,
        projector=projector,
        key="1",
        gloc=gtau,
    )

    expected = _expected_pol_tau(gtau, projector._pairs, ns=ns, soc=False)
    np.testing.assert_allclose(ploc.t, expected)
    np.testing.assert_allclose(ploc.f, expected)
    np.testing.assert_allclose(ploc.t[:, :, 0, 1, :], 0.0)
    np.testing.assert_allclose(ploc.t[:, :, 1, 0, :], 0.0)


def test_ploc_lgw_applies_non_soc_spin_factor_for_single_spin_channel():
    ns = 1
    dlr = _FakeDLR(ntau=4)
    projector = _FakeProjector(ns=ns)
    gtau = _green_tau(ns=ns, ntau=4)

    ploc_soc = PLoc(
        crystal=_crystal(ns=ns, soc=True),
        dlr=dlr,
        projector=projector,
        key="1",
        gloc=gtau,
    )
    ploc_non_soc = PLoc(
        crystal=_crystal(ns=ns, soc=False),
        dlr=dlr,
        projector=projector,
        key="1",
        gloc=gtau,
    )

    np.testing.assert_allclose(ploc_soc.t, _expected_pol_tau(gtau, projector._pairs, ns=ns, soc=True))
    np.testing.assert_allclose(ploc_non_soc.t, 2.0 * ploc_soc.t)
