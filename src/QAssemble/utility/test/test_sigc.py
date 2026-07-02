import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from QAssemble.FLatDyn import SigC

flatdyn_mod = importlib.import_module("QAssemble.FLatDyn")


def _crystal(norb=2, ns=2, nk=3):
    return SimpleNamespace(
        find=list(range(norb)),
        ns=ns,
        kpoint=np.zeros((nk, 3), dtype=float),
        rkgrid=[nk, 1, 1],
        basisf=np.zeros((norb, 3), dtype=float),
        forb2atom=np.arange(norb),
    )


def _dlr(nfreq=4):
    return SimpleNamespace(
        omega=np.arange(nfreq, dtype=float),
        tauF=np.arange(3, dtype=float),
    )


def test_sigc_constructor_validates_lattice_input_shapes():
    crystal = _crystal(norb=2, ns=2, nk=3)
    dlr = _dlr(nfreq=4)

    SigC(
        crystal=crystal,
        dlr=dlr,
        sigh=np.zeros((2, 2, 2, 3)),
        sigf=np.zeros((2, 2, 2, 3)),
        siggwc=np.zeros((2, 2, 2, 3, 4)),
    )

    with pytest.raises(ValueError, match="sigh shape mismatch"):
        SigC(crystal=crystal, dlr=dlr, sigh=np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="siggwc shape mismatch"):
        SigC(crystal=crystal, dlr=dlr, siggwc=np.zeros((2, 2, 2, 3)))


def test_sigc_call_gathers_available_components_without_returning():
    crystal = _crystal(norb=1, ns=1, nk=2)
    dlr = _dlr(nfreq=3)
    sigh = np.full((1, 1, 1, 2), 2.0)
    siggwc = np.full((1, 1, 1, 2, 3), 5.0)
    sigimp = np.full((1, 1, 1, 2, 3), 7.0)

    sigc = SigC(crystal=crystal, dlr=dlr, sigh=sigh, siggwc=siggwc)
    sigc.sigimp = sigimp

    assert sigc() is None
    np.testing.assert_allclose(sigc.kf, 14.0)

    empty = SigC(crystal=crystal, dlr=dlr)
    empty()
    np.testing.assert_allclose(empty.kf, 0.0)


def test_sigc_impembedding_accumulates_multiple_keys(monkeypatch):
    crystal = _crystal(norb=1, ns=1, nk=2)
    dlr = _dlr(nfreq=3)
    projector = object()

    def fake_dynamic_embedding(self, matin, projector=None, key=None):
        value = np.asarray(matin, dtype=np.complex128).item()
        return np.full(self._dynamic_shape, value, dtype=np.complex128, order="F")

    def fake_static_embedding(self, matin, projector=None, key=None):
        value = np.asarray(matin, dtype=np.complex128).item()
        shape = (
            len(self.crystal.find),
            len(self.crystal.find),
            self.crystal.ns,
            len(self.crystal.kpoint),
        )
        return np.full(shape, value, dtype=np.complex128, order="F")

    monkeypatch.setattr(flatdyn_mod.FLatDyn, "Embedding", fake_dynamic_embedding)
    monkeypatch.setattr(flatdyn_mod.FLatStc, "Embedding", fake_static_embedding)

    sigc = SigC(crystal=crystal, dlr=dlr)
    sigc.ImpEmbedding(
        sigimp=np.array(1.0),
        sighimp=np.array(10.0),
        sigfimp=np.array(100.0),
        projector=projector,
        key="a",
    )
    sigc.ImpEmbedding(
        sigimp=np.array(2.0),
        sighimp=np.array(20.0),
        sigfimp=np.array(200.0),
        projector=projector,
        key="b",
    )

    np.testing.assert_allclose(sigc.sigimp, 3.0)
    np.testing.assert_allclose(sigc.sigh, 30.0)
    np.testing.assert_allclose(sigc.sigf, 300.0)

    sigc()
    np.testing.assert_allclose(sigc.kf, 333.0)


def test_sigc_impembedding_requires_projector_key_and_component():
    sigc = SigC(crystal=_crystal(), dlr=_dlr())

    with pytest.raises(ValueError, match="projector is required"):
        sigc.ImpEmbedding(sigimp=np.array(1.0), key="a")

    with pytest.raises(ValueError, match="key is required"):
        sigc.ImpEmbedding(sigimp=np.array(1.0), projector=object())

    with pytest.raises(ValueError, match="at least one impurity self-energy"):
        sigc.ImpEmbedding(projector=object(), key="a")
