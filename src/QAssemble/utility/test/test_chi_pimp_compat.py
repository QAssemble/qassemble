from types import SimpleNamespace

import numpy as np
import pytest

from QAssemble.BLocDyn import Chi, PImp


class _FakeDLR:
    def __init__(self, nfreq=2):
        self.nu = np.arange(nfreq, dtype=float)
        self.tauB = np.arange(nfreq, dtype=float)

    def BatchBF2T(self, bf_2d):
        return np.asarray(bf_2d, dtype=np.complex128)


class _FakeProjector:
    def __init__(self):
        self.bprojector = {"1": np.ones((1, 1, 1), dtype=float)}
        self.equiv = {"1": np.eye(1, dtype=int)}

    def ProbBorb2FPair(self, key, iorbc, ispace=0):
        assert key == "1"
        assert int(iorbc) == 0
        assert ispace == 0
        return (0, 0)


def _chi_reader():
    obj = Chi.__new__(Chi)
    obj.crystal = SimpleNamespace(ns=1)
    obj.dlr = _FakeDLR(nfreq=2)
    obj.projector = _FakeProjector()
    obj.key = "1"
    return obj


def test_chi_quad_susceptibility_converts_ctqmc_7d_to_bosonic_5d():
    reader = _chi_reader()
    raw = np.zeros((1, 1, 1, 1, 2, 2, 2), dtype=np.complex128, order="F")
    raw[0, 0, 0, 0, 0, 0, :] = [1.0, 2.0]
    raw[0, 0, 0, 0, 0, 1, :] = [3.0, 4.0]
    raw[0, 0, 0, 0, 1, 0, :] = [5.0, 6.0]
    raw[0, 0, 0, 0, 1, 1, :] = [7.0, 8.0]

    out = reader.QuadSusceptibility2Boson(raw)

    expected = 0.5 * np.array([16.0, 20.0], dtype=np.complex128)
    assert out.shape == (1, 1, 1, 1, 2)
    np.testing.assert_allclose(out[0, 0, 0, 0, :], expected)


def test_chi_quad_susceptibility_rejects_non_ctqmc_shape():
    reader = _chi_reader()
    chi = np.zeros((1, 1, 1, 1, 2), dtype=np.complex128)

    with pytest.raises(ValueError, match="chi must be 7D"):
        reader.QuadSusceptibility2Boson(chi)


def test_pimp_accepts_chi_bosonic_output_and_applies_inverse_dyson():
    crystal = SimpleNamespace(ns=1)
    dlr = _FakeDLR(nfreq=2)
    projector = _FakeProjector()
    reader = _chi_reader()
    raw = np.zeros((1, 1, 1, 1, 2, 2, 2), dtype=np.complex128, order="F")
    raw[0, 0, 0, 0, :, :, 0] = 0.1
    raw[0, 0, 0, 0, :, :, 1] = 0.15
    chi = reader.QuadSusceptibility2Boson(raw)
    utilde = np.zeros_like(chi)
    utilde[0, 0, 0, 0, :] = [1.0, 2.0]

    pimp = PImp(
        crystal=crystal,
        dlr=dlr,
        projector=projector,
        key="1",
        chi=SimpleNamespace(f=chi),
        utilde=utilde,
    )

    expected = chi / (1.0 + utilde * chi)
    np.testing.assert_allclose(pimp.chi_boson, chi)
    np.testing.assert_allclose(pimp.f, expected)
    np.testing.assert_allclose(pimp.t, expected)
