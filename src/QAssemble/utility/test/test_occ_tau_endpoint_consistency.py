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
import logging
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from QAssemble.FLatDyn import G
from QAssemble.FLocDyn import GImp, GLoc
from QAssemble.utility.DLR import CppdlrDLR


class _IdentityFermionDLR:
    def dlr_from_tau(self, block):
        return block

    def eval_dlr_tau(self, coefficients, tau, beta):
        return coefficients


def _source(func):
    return inspect.getsource(func)


def test_gimp_cal_passes_identity_first_moment():
    recorded = {}

    def convert(value, **kwargs):
        recorded.update(kwargs)
        return value

    green = np.zeros((2,2,2,8), complex)
    owner = object.__new__(GImp)
    owner.projector = SimpleNamespace(equiv={"imp": [0,1]})
    owner.key = "imp"
    owner.green = {"0": {"real": np.zeros(8)}}
    owner.ReadDict = lambda equiv, values: green
    owner.dlr = SimpleNamespace(MatsubaraAddNegativeFrequency=lambda value: value,
                                MatsubaraUniformGrid2DLR=convert)
    owner.F2T = lambda value: value
    owner.Occ = lambda value: None
    owner.Cal()
    assert recorded["highzero"] is True
    assert recorded["m1"].shape == (2,2,2)
    np.testing.assert_array_equal(recorded["m1"], np.repeat(np.eye(2)[:,:,None], 2, axis=2))


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


def test_local_occ_preserves_orbital_axis_order():
    """Moving tau first must not transpose complex orbital off-diagonals."""
    expected = np.array(
        [
            [0.8, 0.2 + 0.3j, -0.1 + 0.4j],
            [0.2 - 0.3j, 0.5, 0.15 - 0.25j],
            [-0.1 - 0.4j, 0.15 + 0.25j, 0.1],
        ],
        dtype=np.complex128,
    )
    mat = np.zeros((3, 3, 1, 1), dtype=np.complex128)
    mat[:, :, 0, 0] = -expected
    owner = SimpleNamespace(
        dlr=SimpleNamespace(
            tauF=np.array([0.5]), beta=1.0, dF=_IdentityFermionDLR()
        )
    )

    for cls in (GLoc, GImp):
        got = cls.Occ(owner, mat)[:, :, 0]
        np.testing.assert_allclose(got, expected)
        assert not np.allclose(got, expected.T)


def test_occupation_error_stays_within_the_expected_band():
    """Quantify what reading at tauF[-1] costs, on an exactly solvable case.

    A single pole at energy e has G(tau) = -exp(-tau e) / (1 + exp(-beta e)) and
    occupation n = 1 / (1 + exp(beta e)).  Reading the DLR expansion at
    tauF[-1] instead of beta introduces a truncation error; this pins its
    magnitude so a change in the DLR grid or the endpoint choice shows up.
    """
    beta, cutoff = 100.0, 50.0
    lambF = beta * cutoff
    dF = CppdlrDLR(lamb=lambF, eps=1e-10)
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


@pytest.mark.parametrize("scf", [False, True])
def test_lattice_occ_records_antihermiticity_without_changing_data(tmp_path, caplog, scf):
    matrix = np.array([[.8 + .1j, .2 + .3j], [.05 - .4j, .4]])
    expected = np.stack((matrix, matrix.conj().T), axis=-1)[:, :, None, :]
    kt = -expected[..., None].copy()
    original = kt.copy()
    owner = SimpleNamespace(
        crystal=SimpleNamespace(find=[0, 1], ns=1, kpoint=[0, 1]),
        dlr=SimpleNamespace(beta=1., dF=_IdentityFermionDLR()),
        kt=kt, _tau_beta=.5, flatstc=SimpleNamespace(K2R=lambda value: value.copy()),
        mu_search_diagnostics={"occupation_at_mu": 1.2},
        hdf5file=str(tmp_path / "green.h5"), group="gwedmft", subgroup="G",
        iteration=3, mu=2., kf=np.zeros_like(kt), CheckGroup=lambda *args: False,
    )
    with caplog.at_level(logging.INFO, logger="QAssemble"):
        G.Occ(owner)
    np.testing.assert_array_equal(owner.kt, original)
    np.testing.assert_array_equal(owner.occk, expected)
    np.testing.assert_array_equal(owner.occ, expected.mean(axis=-1))
    np.testing.assert_array_equal(owner.occr, expected)
    # The two k points cancel after averaging; diagnostics must precede that average.
    np.testing.assert_array_equal(owner.occ, owner.occ.swapaxes(0, 1).conj())
    diag = owner.mu_search_diagnostics
    assert diag["tau_antiherm_max"] == pytest.approx(.2)
    assert diag["occ_antiherm_max"] == pytest.approx(.2)
    assert diag["occupation_at_mu"] == 1.2
    assert "[G.Occ] tau_antiherm_max=2.000000e-01 occ_antiherm_max=2.000000e-01" in caplog.text
    G.Save(owner, "gkf", scf=scf)
    suffix = ".3" if scf else ""
    with h5py.File(owner.hdf5file, "r") as handle:
        for name in ("tau_antiherm_max", "occ_antiherm_max"):
            assert handle[f"gwedmft/G/{name}{suffix}"][()] == pytest.approx(.2)
        assert handle[f"gwedmft/G/occupation_at_mu{suffix}"][()] == 1.2


@pytest.mark.parametrize("cls", [GLoc, GImp])
def test_local_occ_logs_antihermiticity_without_projection(caplog, cls):
    expected = np.array([[.8 + .1j, .2 + .3j], [.05 - .4j, .4]])[:, :, None]
    mat = -expected[..., None].copy()
    original = mat.copy()
    owner = SimpleNamespace(dlr=SimpleNamespace(
        tauF=np.array([.5]), beta=1., dF=_IdentityFermionDLR()))
    with caplog.at_level(logging.INFO, logger="QAssemble"):
        actual = cls.Occ(owner, mat)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mat, original)
    assert (f"[{cls.__name__}.Occ] tau_antiherm_max=2.000000e-01 "
            "occ_antiherm_max=2.000000e-01") in caplog.text
