import numpy as np
import pytest

from QAssemble.utility.Embedding import Embedding
from QAssemble.utility.Projection import Projection

NORB = 4
NORBC = 2
NS = 2


def _random_complex(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _random_projector(rng, orthonormal=False):
    projector = _random_complex(rng, (NORB, NORBC, NS))
    if orthonormal:
        for is_ in range(NS):
            projector[:, :, is_], _ = np.linalg.qr(projector[:, :, is_])
    return projector


def test_projection_flocstc_matches_manual():
    rng = np.random.default_rng(11)
    ff = _random_complex(rng, (NORB, NORB, NS))
    projector = _random_projector(rng)

    result = Projection.FLocStc(ff, projector)

    assert result.shape == (NORBC, NORBC, NS)
    for is_ in range(NS):
        p = projector[:, :, is_]
        expected = p.conj().T @ ff[:, :, is_] @ p
        np.testing.assert_allclose(result[:, :, is_], expected, atol=1e-12, rtol=1e-12)


def test_embedding_flocstc_matches_manual():
    rng = np.random.default_rng(12)
    ffc = _random_complex(rng, (NORBC, NORBC, NS))
    projector = _random_projector(rng)

    result = Embedding.FLocStc(ffc, projector)

    assert result.shape == (NORB, NORB, NS)
    for is_ in range(NS):
        p = projector[:, :, is_]
        expected = p @ ffc[:, :, is_] @ p.conj().T
        np.testing.assert_allclose(result[:, :, is_], expected, atol=1e-12, rtol=1e-12)


def test_projection_flocdyn_matches_per_frequency_static():
    rng = np.random.default_rng(13)
    nf = 3
    ff = _random_complex(rng, (NORB, NORB, NS, nf))
    projector = _random_projector(rng)

    result = Projection.FLocDyn(ff, projector)

    assert result.shape == (NORBC, NORBC, NS, nf)
    for ifreq in range(nf):
        expected = Projection.FLocStc(ff[:, :, :, ifreq], projector)
        np.testing.assert_allclose(
            result[:, :, :, ifreq], expected, atol=1e-12, rtol=1e-12
        )


def test_embedding_blocstc_matches_manual_spin_blocks():
    rng = np.random.default_rng(14)
    ffc = _random_complex(rng, (NORBC, NORBC, NS, NS))
    projector = _random_projector(rng)

    result = Embedding.BLocStc(ffc, projector)

    assert result.shape == (NORB, NORB, NS, NS)
    for is_ in range(NS):
        for js in range(NS):
            pl = projector[:, :, is_]
            pr = projector[:, :, js]
            expected = pl @ ffc[:, :, is_, js] @ pr.conj().T
            np.testing.assert_allclose(
                result[:, :, is_, js], expected, atol=1e-12, rtol=1e-12
            )


def test_projection_flatstc_averages_over_k():
    rng = np.random.default_rng(15)
    nk = 3
    ff = _random_complex(rng, (NORB, NORB, NS, nk))
    projector = _random_projector(rng)

    result = Projection.FLatStc(ff, projector)

    expected = np.zeros((NORBC, NORBC, NS), dtype=np.complex128)
    for ik in range(nk):
        expected += Projection.FLocStc(ff[:, :, :, ik], projector)
    expected /= nk
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)


def test_projection_of_embedding_roundtrips_with_orthonormal_projector():
    rng = np.random.default_rng(16)
    ffc = _random_complex(rng, (NORBC, NORBC, NS))
    projector = _random_projector(rng, orthonormal=True)

    embedded = Embedding.FLocStc(ffc, projector)
    recovered = Projection.FLocStc(embedded, projector)

    np.testing.assert_allclose(recovered, ffc, atol=1e-12, rtol=1e-12)


def test_shape_mismatch_raises():
    rng = np.random.default_rng(17)
    ff = _random_complex(rng, (NORB, NORB, NS))
    bad_projector = _random_complex(rng, (NORB + 1, NORBC, NS))

    with pytest.raises(ValueError):
        Projection.FLocStc(ff, bad_projector)

    ffc = _random_complex(rng, (NORBC, NORBC, NS))
    bad_norbc_projector = _random_complex(rng, (NORB, NORBC + 1, NS))
    with pytest.raises(ValueError):
        Embedding.FLocStc(ffc, bad_norbc_projector)
