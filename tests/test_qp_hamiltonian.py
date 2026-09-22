"""Quasiparticle Hamiltonian and chemical-potential conventions."""

import numpy as np
import pytest

from QAssemble import H, SigStc, Z


def test_external_mu_reproduces_searched_mu(minimal_crystal):
    shape = (2, 2, minimal_crystal.ns, len(minimal_crystal.kpoint))
    h0 = np.zeros(shape, dtype=complex, order="F")
    h0[0, 0] = -1
    h0[1, 1] = 1

    searched = H(minimal_crystal, h0=h0, beta=10)
    external = H(minimal_crystal, h0=h0, beta=10, mu=searched.mu)

    assert external.mu == searched.mu
    np.testing.assert_allclose(external.k, searched.k)


def test_dressed_hqp_matches_explicit_formula(minimal_crystal, rng):
    shape = (2, 2, minimal_crystal.ns, len(minimal_crystal.kpoint))
    h0, sigh, sigf = [np.empty(shape, dtype=complex, order="F") for _ in range(3)]
    sigmac = np.empty((*shape, 1), dtype=complex, order="F")
    beta = 10.0
    mu = 0.37

    for ik in range(shape[3]):
        for js in range(shape[2]):
            for array in (h0, sigh, sigf):
                a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
                array[:, :, js, ik] = (a + a.conj().T) / 2
            a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            gamma = 0.02 * a @ a.conj().T
            sigmac[:, :, js, ik, 0] = 0.1 * np.eye(2) - 1j * gamma

    z = Z(minimal_crystal, sigmac=sigmac, beta=beta)
    sig_stc = SigStc(minimal_crystal, sigmac=sigmac, beta=beta)
    h = H(minimal_crystal, h0=h0, beta=beta, mu=mu,
          sigh=sigh, sigf=sigf, sigmac=sig_stc.k, z=z.k)

    for ik in range(shape[3]):
        for js in range(shape[2]):
            evals, evecs = np.linalg.eigh(z.k[:, :, js, ik])
            zs = (evecs * np.sqrt(evals)) @ evecs.conj().T
            static = h0[:, :, js, ik] + sigh[:, :, js, ik] + sigf[:, :, js, ik]
            static += sig_stc.k[:, :, js, ik] - mu * np.eye(2)
            np.testing.assert_allclose(h.k[:, :, js, ik], zs @ static @ zs, atol=1e-12)


def test_z_eigenvalue_out_of_range_raises(minimal_crystal):
    shape = (2, 2, minimal_crystal.ns, len(minimal_crystal.kpoint))
    h0 = np.zeros(shape, dtype=complex, order="F")
    z = np.zeros(shape, dtype=complex, order="F")
    z[0, 0] = 1.1
    z[1, 1] = 1.0

    with pytest.raises(ValueError, match="Z-factor eigenvalues outside"):
        H(minimal_crystal, h0=h0, beta=10, mu=0, sigmac=h0, z=z)
