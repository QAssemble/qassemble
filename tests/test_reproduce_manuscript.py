"""Manuscript-reproduction pipeline: run the graphene GW calculation of the paper.

Runs `examples/graphene/qassemble.in` (KGrid 25x25x1) end to end and checks the
converged chemical potential against the reference value recorded by the
original manuscript calculation (chemicalpotential in its result.log).

Optionally, set QASSEMBLE_REFERENCE_H5 to the path of a reference output file
(legacy pre-0.2 group names are handled) to also compare the converged Fock
self-energy dataset.
"""

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from QAssemble.Run import Run

EXAMPLE_INPUT = Path(__file__).resolve().parent.parent / "examples" / "graphene" / "qassemble.in"

# From the original manuscript run (result.log: chemicalpotential : 1.6000003040674478).
REFERENCE_MU = 1.6000003040674478


@pytest.mark.slow
def test_manuscript_graphene_gw_reproduces_reference_mu(tmp_path, monkeypatch):
    shutil.copy(EXAMPLE_INPUT, tmp_path / "qassemble.in")
    monkeypatch.chdir(tmp_path)

    Run(input_file=tmp_path / "qassemble.in")

    with h5py.File(tmp_path / "graphene.h5", "r") as h5:
        mu = h5["/gw/G/mu"][()]
        n_iterations = max(
            int(name.split(".")[1]) for name in h5["/gw/G"] if name.startswith("gkf.")
        )
        sigf = h5["/gw/SigF/sigmaf"][()]

    assert mu == pytest.approx(REFERENCE_MU, abs=1e-5)
    assert n_iterations < 20

    reference_file = os.environ.get("QASSEMBLE_REFERENCE_H5")
    if reference_file:
        with h5py.File(reference_file, "r") as ref:
            group = "gw/SigF" if "gw/SigF" in ref else "gw/SigmaFock"
            ref_sigf = ref[f"{group}/sigmaf"][()]
        np.testing.assert_allclose(sigf, ref_sigf, atol=1e-4, rtol=0)
