from types import SimpleNamespace

import h5py
import numpy as np

from QAssemble.FLocDyn import SigGWCLoc


class _IdentityBosonDLR:
    def dlr_from_tau(self, arr):
        return np.asarray(arr, dtype=np.complex128)

    def eval_dlr_tau(self, arr, tau, beta):
        return np.asarray(arr, dtype=np.complex128)


class _IdentityFermionDLR:
    def dlr_from_tau(self, arr):
        return np.asarray(arr, dtype=np.complex128)

    def matsubara_from_dlr(self, arr, beta, xi):
        return np.asarray(arr, dtype=np.complex128)


class _FakeDLR:
    def __init__(self, ntau):
        self.beta = 1.0
        self.tauB = np.arange(ntau, dtype=float)
        self.tauF = np.arange(ntau, dtype=float)
        self.dB = _IdentityBosonDLR()
        self.dF = _IdentityFermionDLR()


class _FakeProjector:
    def __init__(self):
        self.fprojector = {"1": np.zeros((2, 2, 1), dtype=float)}

    def ProbFPair2Borb(self, key, iorb, jorb, ispace=0):
        return 2 * int(iorb) + int(jorb)


def _expected_siggwc_loc(green, wloc, projector, key):
    norb = green.shape[0]
    ns = green.shape[2]
    ntau = green.shape[3]
    out = np.zeros_like(green, dtype=np.complex128)

    for js in range(ns):
        for korb in range(norb):
            for porb in range(norb):
                for iorb in range(norb):
                    ib = projector.ProbFPair2Borb(key, korb, iorb)
                    for jorb in range(norb):
                        jb = projector.ProbFPair2Borb(key, jorb, porb)
                        out[korb, porb, js, :] -= (
                            wloc[ib, jb, js, js, :] * green[iorb, jorb, js, :]
                        )

    return out


def test_siggwc_loc_matches_local_contraction():
    ns = 2
    crystal = SimpleNamespace(ns=ns)
    projector = _FakeProjector()
    dlr = _FakeDLR(ntau=3)

    green = np.zeros((2, 2, ns, 3), dtype=np.complex128, order="F")
    for iorb in range(2):
        for jorb in range(2):
            for js in range(ns):
                green[iorb, jorb, js, :] = (
                    (js + 1) * 100 + (iorb + 1) * 10 + (jorb + 1) + np.arange(3)
                )

    wloc = np.zeros((4, 4, ns, ns, 3), dtype=np.complex128, order="F")
    for ib in range(4):
        for jb in range(4):
            for js in range(ns):
                wloc[ib, jb, js, js, :] = (
                    (js + 1) * 1000 + (ib + 1) * 100 + (jb + 1) * 10 + np.arange(3)
                )
    wloc[:, :, 0, 1, :] = 1.0e9
    wloc[:, :, 1, 0, :] = -1.0e9

    sig = SigGWCLoc(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        green=green,
        wloc=wloc,
    )

    expected = _expected_siggwc_loc(green, wloc, projector, "1")
    np.testing.assert_allclose(sig.t, expected)
    np.testing.assert_allclose(sig.f, expected)


def test_siggwc_loc_save_writes_frequency_data(tmp_path):
    crystal = SimpleNamespace(ns=1)
    projector = _FakeProjector()
    dlr = _FakeDLR(ntau=2)
    green = np.ones((2, 2, 1, 2), dtype=np.complex128, order="F")
    wloc = np.ones((4, 4, 1, 1, 2), dtype=np.complex128, order="F")
    h5_path = tmp_path / "siggwc_loc.h5"

    sig = SigGWCLoc(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        green=green,
        wloc=wloc,
        hdf5file=str(h5_path),
        group="gwloc",
    )
    sig.Save("siggwc.loc", scf=False)

    with h5py.File(h5_path, "r") as handle:
        data = handle["gwloc"]["SigGWCLoc"]["siggwc.loc"][:]
    np.testing.assert_allclose(data, sig.f)
