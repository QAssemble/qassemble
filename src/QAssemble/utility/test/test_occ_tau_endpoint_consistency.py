"""Every occupation is evaluated at the same imaginary-time endpoint.

``occ = -G(tau -> beta^-)`` is read off a DLR expansion, and the tau point it is
read at is a free choice.  The lattice path fixes it to the last DLR node:

    FLatDyn.py:990    self._tau_beta = self.dlr.tauF[-1]

and that choice is load-bearing there, because the charge-neutrality root find
evaluates the electron count at the *same* point --

    FLatDyn.py:1134   eval_dlr_tau(..., self._tau_beta_cache, ...)
    FLatDyn.py:1190   brentq(self.NumOfE, ...)      -> Ne(mu) == crystal.nume
    FLatDyn.py:1199   self.UpdateMu()               -> self.Occ()

-- so if ``Occ`` read a different point than ``NumOfE`` the occupation handed to
SigH/SigF would not carry the charge the chemical potential was solved for.

The local paths (``GLoc.Occ``, ``GImp.Occ``) previously used ``dlr.beta``
instead, leaving the code with two different endpoints.  These tests pin all
three to ``dlr.tauF[-1]``.

Note the trade-off this locks in: ``tauF[-1]`` is strictly inside the interval
(beta - tauF[-1] is ~2e-4 at the production lambF), so occupations carry a
truncation error of that order rather than the ~1e-15 that evaluating exactly at
beta would give.  ``test_occupation_error_stays_within_the_expected_band``
records the size of that error so a regression is visible rather than silent.
"""

import inspect

import numpy as np
from pydlr import dlr

from QAssemble.FLatDyn import G
from QAssemble.FLocDyn import GImp, GLoc


def _source(func):
    return inspect.getsource(func)


def test_local_occ_reads_the_last_dlr_tau_node():
    """GLoc.Occ and GImp.Occ evaluate at dlr.tauF[-1], not dlr.beta."""
    for cls in (GLoc, GImp):
        src = _source(cls.Occ)
        assert "self.dlr.tauF[-1]" in src, (
            f"{cls.__name__}.Occ no longer reads the last DLR tau node"
        )
        assert "np.array([self.dlr.beta]" not in src, (
            f"{cls.__name__}.Occ evaluates at beta again, diverging from the "
            f"lattice path (FLatDyn.py:990)"
        )


def test_lattice_occ_and_charge_root_find_share_the_endpoint():
    """G.Occ and G.NumOfE must not drift apart.

    SearchMu solves Ne(mu) == nume using NumOfE and then hands Occ's result
    downstream; both have to read the same tau point for that to be consistent.
    """
    assert "self._tau_beta = self.dlr.tauF[-1]" in _source(G.__init__)
    assert "self._tau_beta" in _source(G.Occ)
    assert "self._tau_beta_cache" in _source(G.NumOfE)


def test_occupation_error_stays_within_the_expected_band():
    """Quantify what reading at tauF[-1] costs, on an exactly solvable case.

    A single pole at energy e has G(tau) = -exp(-tau e) / (1 + exp(-beta e)) and
    occupation n = 1 / (1 + exp(beta e)).  Reading the DLR expansion at
    tauF[-1] instead of beta introduces a truncation error; this pins its
    magnitude so a change in the DLR grid or the endpoint choice shows up.
    """
    beta, cutoff = 100.0, 50.0
    lambF = (beta / np.pi * cutoff - 1) / 2
    dF = dlr(lamb=lambF, eps=1e-10, dense_imfreq=False)
    tau = dF.get_tau(beta)

    gap = beta - tau[-1]
    assert 0.0 < gap < 1e-3, f"unexpected endpoint gap {gap}"

    worst = 0.0
    for e in (-0.3, -0.05, 0.05, 0.3):
        green = -np.exp(-tau * e) / (1.0 + np.exp(-beta * e))
        exact = 1.0 / (1.0 + np.exp(beta * e))
        coeff = dF.dlr_from_tau(green.reshape(-1, 1))
        got = -dF.eval_dlr_tau(
            coeff[:, :, None], np.array([tau[-1]]), beta=beta
        )[0, 0, 0].real
        worst = max(worst, abs(got - exact))

    # Anti-vacuity: the error is real, not a rounding artefact, so this test
    # genuinely tracks the endpoint choice rather than passing for free.
    assert worst > 1e-8, (
        f"error {worst} is too small to be the tauF[-1] truncation; the "
        f"endpoint may have silently reverted to beta"
    )
    assert worst < 1e-3, f"occupation error grew beyond the known band: {worst}"
