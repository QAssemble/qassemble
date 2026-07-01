from types import SimpleNamespace

import numpy as np

from QAssemble.BLocDyn import WLoc


class _FakeDLR:
    def __init__(self, nfreq=3):
        self.nu = np.arange(nfreq, dtype=float)
        self.tauB = np.arange(nfreq, dtype=float)

    def BatchBF2T(self, bf_2d):
        return 2.0 * np.asarray(bf_2d, dtype=np.complex128)


class _FakeProjector:
    def __init__(self):
        self.bprojector = {"1": np.zeros((1, 1, 1), dtype=float)}
        self.equiv = {"1": np.eye(1, dtype=int)}


def test_wloc_scalar_screened_interaction_and_correlation_part():
    dlr = _FakeDLR(nfreq=3)
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()

    vloc = np.zeros((1, 1, 1, 1), dtype=np.complex128, order="F")
    vloc[0, 0, 0, 0] = 2.0
    pol = np.zeros((1, 1, 1, 1, 3), dtype=np.complex128, order="F")
    pol[0, 0, 0, 0, :] = np.array([0.1, -0.2, 0.05])

    wloc = WLoc(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        pol=pol,
        vloc=vloc,
    )

    expected = vloc[..., np.newaxis] / (1.0 - pol * vloc[..., np.newaxis])
    np.testing.assert_allclose(wloc.f, expected)
    np.testing.assert_allclose(wloc.cf, expected - vloc[..., np.newaxis])


def test_wloc_builds_tau_quantities_through_f2t():
    dlr = _FakeDLR(nfreq=3)
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()

    vloc = np.ones((1, 1, 1, 1), dtype=np.complex128, order="F")
    pol = np.zeros((1, 1, 1, 1, 3), dtype=np.complex128, order="F")
    pol[0, 0, 0, 0, :] = np.array([0.0, 0.25, -0.5])

    wloc = WLoc(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        pol=pol,
        vloc=vloc,
    )

    np.testing.assert_allclose(wloc.t, 2.0 * wloc.f)
    np.testing.assert_allclose(wloc.ct, 2.0 * wloc.cf)
