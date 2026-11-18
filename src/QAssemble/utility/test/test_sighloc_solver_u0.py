from types import SimpleNamespace

import numpy as np
import pytest

from QAssemble.BLocDyn import BWeiss
from QAssemble.FLocStc import SigHImp, SigHLoc

from test_sighimp_sigfimp_hf_identity import (
    _FakeProjector,
    _hermitian_occ,
    _physical_vloc,
)


def test_static_bweiss_returns_bare_interaction():
    bare = _physical_vloc(norb=2, ns=1, seed=17)
    bath = BWeiss(
        SimpleNamespace(soc=False), None, _FakeProjector(2, 1), "1",
        vloc=SimpleNamespace(vproj={"1": bare}),
    )

    interaction, source = bath.SolverStaticInteraction("1")

    assert interaction is bare
    assert source == "bare_vloc"


@pytest.mark.parametrize("ns,soc", [(1, False), (1, True), (2, False)])
def test_hartree_dc_matches_impurity_contraction_with_solver_zero_mode(ns, soc):
    crystal = SimpleNamespace(soc=soc)
    projector = _FakeProjector(2, ns)
    occ = _hermitian_occ(norb=2, ns=ns, seed=41)
    bare = _physical_vloc(norb=2, ns=ns, seed=17)
    solver_u0 = 0.75 * bare
    bath = BWeiss(crystal, None, projector, "1", vloc=SimpleNamespace(vproj={"1": bare}))
    bath.f = bare[..., None]
    bath.f_uniform = np.stack((0.5 * bare, bare), axis=-1)
    bath.f_to_solver_uniform = np.stack((solver_u0, bare), axis=-1)

    interaction, source = bath.SolverStaticInteraction("1")
    dc = SigHLoc(crystal, projector, "1", occ=occ, vloc=interaction)
    impurity = SigHImp(crystal, projector, "1", occ=occ, vloc=solver_u0)

    assert source == "bweiss_solver_uniform_nu0"
    np.testing.assert_array_equal(interaction, solver_u0)
    np.testing.assert_allclose(dc.hloc, impurity.h, rtol=0, atol=1e-14)
