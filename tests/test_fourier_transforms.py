import numpy as np
import pytest

from QAssemble import BLatStc, FLatStc


def _complex_random(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_flatstc_k2r_r2k_round_trip(minimal_crystal, rng):
    obj = FLatStc(minimal_crystal)
    norb = len(minimal_crystal.find)
    ns = minimal_crystal.ns
    nk = len(minimal_crystal.kpoint)
    matk = _complex_random(rng, (norb, norb, ns, nk))

    matr = obj.K2R(matk)
    round_trip = obj.R2K(matr)

    np.testing.assert_allclose(round_trip, matk, atol=1e-12, rtol=1e-12)


def test_blatstc_k2r_r2k_round_trip(minimal_crystal, rng):
    obj = BLatStc(minimal_crystal)
    norb = len(minimal_crystal.bind)
    ns = minimal_crystal.ns
    nk = len(minimal_crystal.kpoint)
    matk = _complex_random(rng, (norb, norb, ns, ns, nk))

    matr = obj.K2R(matk)
    round_trip = obj.R2K(matr)

    np.testing.assert_allclose(round_trip, matk, atol=1e-12, rtol=1e-12)


def test_blatstc_k2r_rejects_wrong_k_axis(minimal_crystal, rng):
    obj = BLatStc(minimal_crystal)
    norb = len(minimal_crystal.bind)
    ns = minimal_crystal.ns
    matk = _complex_random(rng, (norb, norb, ns, ns, 3))

    with pytest.raises(ValueError, match="Incompatible k-space axis"):
        obj.K2R(matk)
