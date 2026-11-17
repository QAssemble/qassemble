from types import SimpleNamespace

import numpy as np
import pytest

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


@pytest.mark.parametrize("as_quantity", [False, True])
def test_polc_density_only_double_counting_preserves_density_block(monkeypatch, as_quantity):
    _patch_lightweight_lattice(monkeypatch)
    crystal, dlr, _, _ = _inputs()
    crystal.bind = list(range(9))
    crystal.ns = 2
    shape = (9, 9, 2, 2, len(dlr.nu))
    values = np.arange(np.prod(shape)).reshape(shape) + 1.0
    local = values + 0.1j * values[..., ::-1]
    original = local.copy()
    source = SimpleNamespace(f=local) if as_quantity else local
    density = [0, 4, 8]

    def embed(self, local, projector, key):
        return np.repeat(np.asarray(local)[..., np.newaxis, :], crystal.nk, axis=-2)

    monkeypatch.setattr(PolC, "Embedding", embed)
    projector = SimpleNamespace(DensityBorbs=lambda key: density)
    default = PolC(crystal, dlr)
    default.GWDoubleCounting(source, projector, "1")
    expected_default = embed(default, original, projector, "1")
    assert np.array_equal(default.pgw_dc, expected_default)
    assert default.pgw_dc.tobytes() == expected_default.tobytes()

    explicit_false = PolC(crystal, dlr)
    explicit_false.GWDoubleCounting(source, projector, "1", density_only=False)
    assert np.array_equal(explicit_false.pgw_dc, default.pgw_dc)

    masked = PolC(crystal, dlr)
    masked.GWDoubleCounting(source, projector, "1", density_only=True)

    keep = np.zeros((9, 9), dtype=bool)
    keep[np.ix_(density, density)] = True
    assert np.array_equal(masked.pgw_dc[keep], default.pgw_dc[keep])
    assert np.count_nonzero(masked.pgw_dc[~keep]) == 0
    assert np.all(default.pgw_dc[~keep] != 0.0)
    assert np.array_equal(local, original)

