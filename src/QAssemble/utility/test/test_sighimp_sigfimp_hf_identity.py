"""SigH + SigF == hf, exactly and at every iteration.

SigF is not an independent quantity.  ``SigFImp.Cal`` defines it as the
complement of SigH with respect to the solver's static moment,

    s = hf - sigh                       FLocStc.py:688-703

``SigCImp`` then subtracts the *unmixed* ``hf`` from the dynamic self-energy
(FLocDyn.py:1212) and the embedding re-adds ``SigH + SigF``
(FLatDyn.py:1337-1340, FLocDyn.py:1336-1338), so the total

    Sigma_tot = (Sigma_ctqmc - hf) + SigH~ + SigF~

reconstructs the solver self-energy only while the identity holds.  The ``hf``
that is subtracted and the ``hf`` that is re-added must be the same object.

Mixing SigH and SigF independently -- each with its own HDF5 history -- breaks
it by one iteration of lag.  With linear mixing at ratio ``m``:

    (SigH~ + SigF~) - hf = (1 - m) * [SigF~(n-1) - (hf(n) - SigH~(n))]

This is a complex identity: it applies to the real and imaginary parts alike.
The imaginary residual was the visible symptom -- it reached ~0.03 against
w0 = pi/beta = 0.0314, driving Im G^-1 = w0 - Im Sigma to ~9e-4 and diverging
GLoc -- but forcing ``.real`` would only have silenced that alarm while leaving
the real part wrong, converging quietly to the wrong fixed point.
``test_hf_identity_fails_under_independent_mixing`` pins that specific claim.

The reference implementation avoids this structurally: it never splits the pair
(FullGWEDMFT bin/classes/cimpurity.py:322,408 carry a single combined ``Shf``)
and mixes only the dynamic self-energy (bin/comfull.py:1451).  It also forces
realness at read time (bin/classes/old.py:980, ``complex(mom1, 0.0)``).

The lattice HF path (Method.py:105-115, FLatStc SigH/SigF) is deliberately NOT
covered here and must not be "fixed" the same way: lattice SigF is computed
independently from ``occr`` and ``vbare`` (FLatStc.py:1087+), not as
``hf - sigh``, so no such identity exists and independent mixing is correct
there.
"""

import os
from types import SimpleNamespace

import h5py
import numpy as np

from QAssemble.FLocStc import SigFImp, SigHImp


class _FakeProjector:
    """Boson indices enumerate every (iorb, jorb) fermion pair, matching
    Projector.py:329-347."""

    def __init__(self, norb, ns):
        self.norb = norb
        self._pairs = [(i, j) for i in range(norb) for j in range(norb)]
        norbb = len(self._pairs)
        self.fprojector = {"1": np.ones((1, norb, ns), dtype=float)}
        self.bprojector = {"1": np.ones((1, norbb, ns), dtype=float)}
        self.equiv = {"1": np.eye(norb, dtype=int)}

    def ProbBorb2FPair(self, key, iorbc, ispace=0):
        return self._pairs[int(iorbc)]


def _physical_vloc(norb, ns, seed):
    """A real Coulomb tensor carrying its physical exchange symmetry.

    Built as V[(i,l),(j,k)] = M[i,l] M[j,k] with M real symmetric, which is the
    property that makes the Hartree contraction Hermitian.  Without it the
    diagonal of h picks up an imaginary part even from an exactly Hermitian
    occupation, so this is what a physical vloc must look like.
    """
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(norb, norb))
    m = (m + m.T) / 2
    pairs = [(i, j) for i in range(norb) for j in range(norb)]
    norbb = len(pairs)
    v = np.zeros((norbb, norbb, ns, ns), dtype=np.complex128, order="F")
    for ii, (i, l) in enumerate(pairs):
        for jj, (j, k) in enumerate(pairs):
            v[ii, jj, :, :] = m[i, l] * m[j, k]
    return v


def _hermitian_occ(norb, ns, seed, imag_scale=0.0):
    """A Hermitian density matrix, optionally with a symmetry-violating
    imaginary contamination of the size measured in the CTQMC output."""
    rng = np.random.default_rng(seed)
    occ = np.zeros((norb, norb, ns), dtype=np.complex128, order="F")
    for js in range(ns):
        a = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
        block = (a + a.conj().T) / 2
        if imag_scale:
            block = block + 1j * imag_scale * rng.normal(size=(norb, norb))
        occ[:, :, js] = block
    return occ


def _build_sighimp(occ, vloc, norb, ns, soc=False):
    obj = SigHImp.__new__(SigHImp)
    obj.crystal = SimpleNamespace(ns=ns, soc=soc)
    obj.projector = _FakeProjector(norb, ns)
    obj.key = "1"
    obj.occ = occ
    obj.vloc = vloc
    obj.control = {}
    obj.hdf5file = None
    obj.group = None
    obj.subgroup = "SigHImp"
    obj.iteration = 1
    obj.s = None
    obj.h = None
    obj.Cal()
    return obj


def _build_sigfimp(hf_moment, sigh, norb, ns):
    """Construct SigFImp with a static moment of ``hf_moment``.

    For ns == 1 the CTQMC-shaped dict path is exercised (moments[0] is a scalar
    float, as the solver emits it).  For ns != 1 Dict2Arr requires a per-spin
    array, which _read_ctqmc_sigma_hf cannot produce, so the already-projected
    array path (FLocStc.py:695) is used instead.
    """
    obj = SigFImp.__new__(SigFImp)
    obj.crystal = SimpleNamespace(ns=ns)
    obj.projector = _FakeProjector(norb, ns)
    obj.key = "1"
    if ns == 1:
        obj.sigma_in = {
            str(ind): {"moments": [hf_moment]} for ind in range(1, norb + 1)
        }
    else:
        arr = np.zeros((norb, norb, ns), dtype=np.complex128, order="F")
        for iorb in range(norb):
            arr[iorb, iorb, :] = hf_moment
        obj.sigma_in = arr
    obj.sigh = sigh
    obj.control = {}
    obj.hdf5file = None
    obj.group = None
    obj.subgroup = "SigFImp"
    obj.iteration = 1
    obj.hf = None
    obj.s = None
    obj.Cal()
    return obj


def _linear(mix, fnew, fold):
    return mix * fnew + (1.0 - mix) * fold


def test_hf_identity_holds_across_iterations():
    """SigH + SigF == hf at every iteration, for both spin branches.

    Drives five iterations with a changing occupation and a changing hf, mixing
    SigH the way CTQMC.PostProcessing does, and rebuilding SigF from the mixed
    SigH.  The ns == 1 and ns != 1 branches of SigHImp.Cal (FLocStc.py:506-530)
    index differently, so both are exercised.
    """
    norb, mix = 2, 0.1

    for ns in (1, 2):
        vloc = _physical_vloc(norb, ns, seed=11)
        h_mixed = None
        hf_seen, damping_seen = [], []

        for iteration in range(1, 6):
            occ = _hermitian_occ(norb, ns, seed=100 + iteration)
            hf_moment = 0.5 + 0.11 * iteration  # genuinely moves

            sigh = _build_sighimp(occ, vloc, norb, ns)
            raw = sigh.h.copy()
            h_mixed = raw if h_mixed is None else _linear(mix, raw, h_mixed)
            sigh.h = np.asfortranarray(h_mixed)
            sigh.s = sigh.h

            sigf = _build_sigfimp(hf_moment, sigh, norb, ns)

            total = sigh.h + sigf.s
            np.testing.assert_allclose(
                total, sigf.hf, atol=1e-14,
                err_msg=f"identity broken at ns={ns}, iter={iteration}",
            )

            hf_seen.append(sigf.hf.copy())
            damping_seen.append(np.abs(h_mixed - raw).max())

        # Anti-vacuity: hf must genuinely move, and mixing must genuinely be
        # damping, or the identity would hold trivially.
        hf_drift = np.abs(hf_seen[-1] - hf_seen[0]).max()
        assert hf_drift > 1e-3, f"hf did not move (ns={ns}): {hf_drift}"
        assert max(damping_seen) > 1e-3, (
            f"mixing was inactive (ns={ns}): {max(damping_seen)}"
        )


def test_hf_identity_fails_under_independent_mixing():
    """The pre-fix sequence breaks the identity, and not only in Im.

    This is the refutation of "forcing .real is enough": the real part carries a
    residual of the same order as the imaginary one, so zeroing Im would leave a
    wrong static self-energy behind while removing the divergence that made it
    visible.
    """
    norb, ns, mix = 2, 1, 0.1
    vloc = _physical_vloc(norb, ns, seed=11)

    h_mixed = f_mixed = None
    worst_real = 0.0

    for iteration in range(1, 6):
        occ = _hermitian_occ(norb, ns, seed=100 + iteration)
        hf_moment = 0.5 + 0.11 * iteration

        sigh = _build_sighimp(occ, vloc, norb, ns)
        raw = sigh.h.copy()
        h_mixed = raw if h_mixed is None else _linear(mix, raw, h_mixed)
        sigh.h = np.asfortranarray(h_mixed)
        sigh.s = sigh.h

        sigf = _build_sigfimp(hf_moment, sigh, norb, ns)
        # The bug: SigF is mixed again, against its own history.
        f_raw = sigf.s.copy()
        f_mixed = f_raw if f_mixed is None else _linear(mix, f_raw, f_mixed)

        residual = (h_mixed + f_mixed) - sigf.hf
        worst_real = max(worst_real, np.abs(residual.real).max())

    assert worst_real > 1e-3, (
        "independent mixing should leave a real-part residual; "
        f"got {worst_real}. If this fails the test no longer refutes the "
        "'.real is sufficient' position."
    )


def test_sighimp_cal_discards_spurious_imaginary_occupation():
    """A symmetry-violating occupation must not leak into the Hartree term.

    The CTQMC Green's function violates G(-iw) = G(iw)* at the ~1e-2 level, so
    occ acquires an imaginary part that a physical density matrix cannot have.
    """
    norb, ns = 2, 1
    vloc = _physical_vloc(norb, ns, seed=11)
    occ = _hermitian_occ(norb, ns, seed=7, imag_scale=3e-2)

    sigh = _build_sighimp(occ, vloc, norb, ns)

    np.testing.assert_allclose(sigh.h.imag, 0.0, atol=1e-14)
    # Anti-vacuity: a zeroed-out h would satisfy the assertion above.
    assert np.abs(sigh.h.real).max() > 1e-6, "h collapsed to zero"

    # And the contamination was real: without the fix Im(h) would be visible.
    unfixed = np.zeros_like(sigh.h)
    pairs = [(i, j) for i in range(norb) for j in range(norb)]
    for ii, (a, b) in enumerate(pairs):
        for jj, (c, d) in enumerate(pairs):
            unfixed[a, b, 0] += vloc[ii, jj, 0, 0] * occ[d, c, 0] * 2
    assert np.abs(unfixed.imag).max() > 1e-4, (
        "the test input carries no imaginary contamination, so it cannot "
        "demonstrate that the fix does anything"
    )


def test_sigfimp_mixing_is_a_rederivation(tmp_path):
    """Mixing() must recompute s = hf - sigh, not blend with a history.

    Runs against a real HDF5 file and a fully-populated control dict, so an
    independent mix would succeed rather than error out -- the assertions have
    to be what rejects it, not a missing-file accident.
    """
    norb, ns = 2, 1
    vloc = _physical_vloc(norb, ns, seed=11)
    occ = _hermitian_occ(norb, ns, seed=3)

    sigh = _build_sighimp(occ, vloc, norb, ns)
    sigf = _build_sigfimp(0.75, sigh, norb, ns)
    sigf.hdf5file = str(tmp_path / "mix.h5")
    sigf.group = "calc"
    sigf.control = {"mix": 0.1, "mixing_method": "linear", "npulay": 2}

    first = sigf.s.copy()
    sigf.Mixing()
    np.testing.assert_allclose(sigf.s, first, atol=1e-15)

    # At iter > 1 an independent mix would blend against stored history; the
    # re-derivation must stay pinned to hf - sigh.
    sigf.iteration = 4
    sigf.Mixing()
    np.testing.assert_allclose(sigf.s, first, atol=1e-15)
    np.testing.assert_allclose(sigh.h + sigf.s, sigf.hf, atol=1e-14)

    # A change in the mixed SigH propagates in full, undamped.  Under a 0.1
    # linear mix only a tenth of it would land, so this rejects the bug.
    sigh.h = np.asfortranarray(sigh.h * 1.5)
    sigh.s = sigh.h
    sigf.Mixing()
    np.testing.assert_allclose(sigh.h + sigf.s, sigf.hf, atol=1e-14)
    assert np.abs(sigf.s - first).max() > 1e-6, "SigF did not follow SigH"

    # No sigfimp mixing history exists, whatever else happened.
    if os.path.exists(sigf.hdf5file):
        with h5py.File(sigf.hdf5file, "r") as handle:
            assert "calc/Mixing/1/sigfimp" not in handle


def test_hf_moment_is_read_as_a_real_quantity():
    """A complex-valued static moment is coerced to its real part.

    Mirrors FullGWEDMFT bin/classes/old.py:980, complex(mom1, 0.0).
    """
    norb, ns = 2, 1
    vloc = _physical_vloc(norb, ns, seed=11)
    occ = _hermitian_occ(norb, ns, seed=5)
    sigh = _build_sighimp(occ, vloc, norb, ns)

    sigf = SigFImp.__new__(SigFImp)
    sigf.crystal = SimpleNamespace(ns=ns)
    sigf.projector = _FakeProjector(norb, ns)
    sigf.key = "1"
    sigf.sigma_in = {
        str(ind): {"moments": [complex(0.75, 0.4)]} for ind in range(1, norb + 1)
    }
    sigf.sigh = sigh
    sigf.control = {}
    sigf.hdf5file = None
    sigf.group = None
    sigf.subgroup = "SigFImp"
    sigf.iteration = 1
    sigf.hf = None
    sigf.s = None
    sigf.Cal()

    np.testing.assert_allclose(sigf.hf.imag, 0.0, atol=1e-15)
    assert np.abs(sigf.hf.real).max() > 1e-6, "hf collapsed to zero"
