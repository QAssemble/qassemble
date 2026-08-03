import numpy as np

from QAssemble.utility.Dyson import Dyson


def _direct_dyson(g0, sigma):
    eye = np.eye(g0.shape[0], dtype=np.complex128)
    return np.linalg.solve((eye - sigma @ g0).T, g0.T).T


def test_fermionic_static_dyson_matches_direct_solve():
    g0_block = np.array([[1.0 + 0.2j, 0.1 - 0.3j], [0.2 + 0.1j, 0.8 - 0.1j]])
    sigma_block = np.array([[0.05, 0.02j], [-0.01j, -0.03]])
    g0 = np.asfortranarray(g0_block[:, :, None, None])
    sigma = np.asfortranarray(sigma_block[:, :, None, None])

    result = Dyson.FLatStc(g0, sigma)
    expected = _direct_dyson(g0_block, sigma_block)[:, :, None, None]

    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)


def test_bosonic_static_dyson_matches_direct_solve():
    nk = 3
    g0 = np.zeros((2, 2, 1, 1, nk), dtype=np.complex128, order="F")
    sigma = np.zeros_like(g0)

    for ik in range(nk):
        g0[:, :, 0, 0, ik] = np.array(
            [[1.0 + 0.1j * ik, 0.05], [0.02j, 0.7 - 0.03j * ik]]
        )
        sigma[:, :, 0, 0, ik] = np.array([[0.02, 0.01j], [-0.02j, -0.01]])

    result = Dyson.BLatStc(g0, sigma)

    for ik in range(nk):
        expected = _direct_dyson(g0[:, :, 0, 0, ik], sigma[:, :, 0, 0, ik])
        np.testing.assert_allclose(result[:, :, 0, 0, ik], expected, atol=1e-12, rtol=1e-12)
    assert np.isfinite(result).all()
