"""Sign-convention and operand-order guards for the impurity polarization.

``PImp`` must reproduce the reference definition

    P = -(1 - chi U)^-1 chi          FullGWEDMFT qft_comlocal.F:406-413
                                     gw_edmft_v208 edmft.py:2676

with ``chi`` the raw (positive) solver susceptibility and ``U`` the full
bosonic Weiss interaction.  ``P`` is therefore *negative* for a physical
susceptibility, which is what the ``coefficient_sign=-1`` causal projection in
``PImp.Cal`` and the negative-``P`` convention in ``WImp``/``BWeiss``/``PolC``/
``W`` all assume.

These tests exist because the formula was previously written with two
compensating sign errors that produced a positive ``P``; the causal projection
masked the sign while corrupting the magnitude.
"""

from types import SimpleNamespace

import numpy as np

from QAssemble.BLocDyn import PImp
from QAssemble.utility.Dyson import Dyson


class _FakeDLR:
    """Minimal DLR stub; the uniform and DLR grids are the same size here so
    ``Cal`` never needs a real interpolation."""

    def __init__(self, nfreq):
        self.beta = 1.0
        self.nu = np.arange(nfreq, dtype=float)
        self.tauB = np.arange(nfreq, dtype=float)

    def MatsubaraBosonUniform(self):
        return self.nu

    def MatsubaraDLR2UniformGrid(self, value, sign=1):
        return np.asarray(value, dtype=np.complex128)

    def BatchBF2T(self, bf_2d):
        return np.asarray(bf_2d, dtype=np.complex128)


class _FakeProjector:
    def __init__(self, norbb):
        self.bprojector = {"1": np.ones((1, norbb, norbb), dtype=float)}
        self.equiv = {"1": np.eye(norbb, dtype=int)}
        self.blocal2pair = {"1": [{i: (i, i) for i in range(norbb)}]}

    def ProbBorb2FPair(self, key, iorbc, ispace=0):
        return (int(iorbc), int(iorbc))


def _build_pimp(chi, utilde, ns=1):
    """Construct PImp with the causal projection stubbed to the identity so the
    raw algebra of ``Cal`` is what gets asserted."""
    norbb = chi.shape[0]
    nfreq = chi.shape[-1]
    obj = PImp.__new__(PImp)
    obj.crystal = SimpleNamespace(ns=ns)
    obj.dlr = _FakeDLR(nfreq)
    obj.projector = _FakeProjector(norbb)
    obj.key = "1"
    obj.chi = SimpleNamespace(f_uniform=chi)
    obj.utilde = utilde
    obj.control = {}
    obj.hdf5file = None
    obj.group = None
    obj.subgroup = "PImp"
    obj.iteration = 1
    obj.chi_boson = None
    obj.chi_boson_uniform = None
    obj.utilde_uniform = None
    obj.f = None
    obj.f_uniform = None
    obj.t = None
    # Identity projection + no fallback cache: isolate Cal's algebra.
    obj.CausalProjection = lambda value, **kwargs: np.asfortranarray(value)
    obj.ReadBrdPrev = lambda stem, shape: None
    obj.Cal()
    return obj


def _blocks(mat, norbb, ns, nfreq):
    """Flatten (norb, norb, ns, ns, nfreq) into one (ns*norb) matrix per
    frequency, matching Dyson's composite ordering (spin fastest)."""
    dim = norbb * ns
    return [
        mat[..., ifreq].transpose(0, 2, 1, 3).reshape(dim, dim)
        for ifreq in range(nfreq)
    ]


def _physical_inputs(norbb=3, ns=2, nfreq=4, seed=0):
    """Positive-definite chi decaying in frequency, and a U_weiss chosen so the
    system stays in the stable regime (all eigenvalues of 1 - chi*U positive)."""
    rng = np.random.default_rng(seed)
    dim = norbb * ns
    a = rng.normal(size=(dim, dim))
    chi_static = (a @ a.T) * (0.05 / dim)
    b = rng.normal(size=(dim, dim))
    u_static = (b @ b.T) * (0.2 / dim) + 2.0 * np.eye(dim)

    chi = np.zeros((norbb, norbb, ns, ns, nfreq), dtype=np.complex128, order="F")
    utilde = np.zeros_like(chi)
    for ifreq in range(nfreq):
        decay = 1.0 / (1.0 + ifreq)
        chi[..., ifreq] = (
            (chi_static * decay).reshape(norbb, ns, norbb, ns).transpose(0, 2, 1, 3)
        )
        utilde[..., ifreq] = (
            u_static.reshape(norbb, ns, norbb, ns).transpose(0, 2, 1, 3)
        )
    return chi, utilde, norbb, ns, nfreq


def test_pimp_is_negative_for_positive_susceptibility():
    """The regression guard for the original bug: a positive chi must yield a
    negative polarization on every diagonal channel."""
    chi, utilde, norbb, ns, nfreq = _physical_inputs()
    pimp = _build_pimp(chi, utilde, ns=ns)

    for ifreq in range(nfreq):
        for iorb in range(norbb):
            for ispin in range(ns):
                value = pimp.f_uniform[iorb, iorb, ispin, ispin, ifreq]
                assert value.real <= 1e-12, (
                    f"P must be non-positive on diagonal channels; got "
                    f"{value.real} at orb={iorb}, spin={ispin}, freq={ifreq}"
                )

    # Guard against a trivially-zero pass: the static channel is strictly negative.
    for iorb in range(norbb):
        for ispin in range(ns):
            assert pimp.f_uniform[iorb, iorb, ispin, ispin, 0].real < -1e-6


def test_pimp_matches_reference_form_for_noncommuting_blocks():
    """Pins the operand order.  Dyson returns chi (1 - U chi)^-1, which equals
    (1 - chi U)^-1 chi by push-through; the *other* ordering,
    (1 - U chi)^-1 chi, is a different matrix and must not be what we compute."""
    chi, utilde, norbb, ns, nfreq = _physical_inputs(seed=3)
    pimp = _build_pimp(chi, utilde, ns=ns)

    chi_blocks = _blocks(chi, norbb, ns, nfreq)
    u_blocks = _blocks(utilde, norbb, ns, nfreq)
    got_blocks = _blocks(pimp.f_uniform, norbb, ns, nfreq)
    dim = norbb * ns
    eye = np.eye(dim)

    commutators = []
    discriminations = []
    for x, u, got in zip(chi_blocks, u_blocks, got_blocks):
        reference = -np.linalg.solve(eye - x @ u, x)
        wrong_order = -np.linalg.solve(eye - u @ x, x)
        np.testing.assert_allclose(got, reference, rtol=1e-10, atol=1e-12)
        commutators.append(np.linalg.norm(x @ u - u @ x))
        discriminations.append(np.abs(reference - wrong_order).max())

    # The test would pass vacuously on commuting data, so assert the blocks
    # genuinely do not commute and the two orderings are far apart.
    assert min(commutators) > 1e-6
    assert max(discriminations) > 1e-3


def test_pimp_and_wimp_are_mutually_consistent():
    """With the corrected sign, WImp's Dyson(U, P) reproduces both FullGWEDMFT
    closed forms: W = U - U chi U and P = U^-1 - W^-1."""
    chi, utilde, norbb, ns, nfreq = _physical_inputs(seed=5)
    pimp = _build_pimp(chi, utilde, ns=ns)

    # WImp.Cal's algebra, without constructing the object (which would need a
    # real DLR for its uniform-grid conversion).
    wimp = Dyson.BLocDyn(np.asfortranarray(utilde), np.asfortranarray(pimp.f_uniform))

    chi_blocks = _blocks(chi, norbb, ns, nfreq)
    u_blocks = _blocks(utilde, norbb, ns, nfreq)
    p_blocks = _blocks(pimp.f_uniform, norbb, ns, nfreq)
    w_blocks = _blocks(wimp, norbb, ns, nfreq)

    for x, u, p, w in zip(chi_blocks, u_blocks, p_blocks, w_blocks):
        np.testing.assert_allclose(w, u - u @ x @ u, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(
            p, np.linalg.inv(u) - np.linalg.inv(w), rtol=1e-7, atol=1e-10
        )
