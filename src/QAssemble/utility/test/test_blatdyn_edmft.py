from types import SimpleNamespace

import numpy as np

from QAssemble.BLatDyn import BLatDyn, PolC, W


def _patch_lightweight_lattice(monkeypatch):
    def base_init(self, crystal, dlr, **kwargs):
        self.crystal = crystal
        self.dlr = dlr

    monkeypatch.setattr(BLatDyn, "__init__", base_init)
    monkeypatch.setattr(BLatDyn, "F2T", lambda self, value: np.asarray(value))
    monkeypatch.setattr(BLatDyn, "K2R", lambda self, value: np.asarray(value))


def _inputs():
    crystal = SimpleNamespace(
        bind=[0], find=[0], ns=1, nk=2, kpoint=[0, 1]
    )
    dlr = SimpleNamespace(nu=np.asarray([0.0, 1.0]), tauB=np.asarray([0.0, 1.0]))
    vstatic = np.asarray([[[[[2.0, 3.0]]]]], dtype=np.complex128)
    vdynamic = np.repeat(vstatic[..., np.newaxis], len(dlr.nu), axis=-1)
    return crystal, dlr, SimpleNamespace(k=vstatic), vdynamic


def test_w_without_polarization_is_bare(monkeypatch):
    _patch_lightweight_lattice(monkeypatch)
    crystal, dlr, vbare, vdynamic = _inputs()
    monkeypatch.setattr(W, "StcEmbedding", lambda self, value: vdynamic)

    w = W(crystal=crystal, dlr=dlr, vbare=vbare, pol=None)

    assert w.pol is None
    assert w.is_bare
    np.testing.assert_allclose(w.kf, vdynamic)
    np.testing.assert_allclose(w.ckf, 0.0)


def test_explicit_zero_polarization_matches_bare_but_is_not_initialization(monkeypatch):
    _patch_lightweight_lattice(monkeypatch)
    crystal, dlr, vbare, vdynamic = _inputs()
    monkeypatch.setattr(W, "StcEmbedding", lambda self, value: vdynamic)
    monkeypatch.setattr(W, "Double2Full", lambda self, value: value)
    monkeypatch.setattr(W, "Full2Double", lambda self, value: value)
    monkeypatch.setattr(W, "Dyson", lambda self, v, p: v / (1.0 - p * v))

    zero = np.zeros_like(vdynamic)
    w = W(crystal=crystal, dlr=dlr, vbare=vbare, pol=zero)

    assert not w.is_bare
    np.testing.assert_allclose(w.kf, vdynamic)
    np.testing.assert_allclose(w.ckf, 0.0)


def test_polc_combines_gw_double_counting_and_impurity(monkeypatch):
    _patch_lightweight_lattice(monkeypatch)
    crystal, dlr, _, vdynamic = _inputs()

    def embed(self, local, projector, key):
        return np.repeat(np.asarray(local)[..., np.newaxis, :], crystal.nk, axis=-2)

    monkeypatch.setattr(PolC, "Embedding", embed)
    projector = SimpleNamespace()
    local = np.ones((1, 1, 1, 1, 2), dtype=np.complex128)
    pgw = 5.0 * np.ones_like(vdynamic)

    pol = PolC(crystal, dlr)
    pol.GWContribution(pgw)
    pol.GWDoubleCounting(2.0 * local, projector, "1")
    pol.ImpEmbedding(3.0 * local, projector, "1")

    np.testing.assert_allclose(pol(), 6.0)

