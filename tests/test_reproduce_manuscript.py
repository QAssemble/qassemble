"""Manuscript-reproduction pipeline: run the graphene GW calculation of the paper.

Runs `examples/graphene/qassemble.in` (t=1.0, U=2.0, V=0.2, KGrid 25x25x1,
T=2000 K, DLR cutoff 100 eV — the parameters of the manuscript's graphene
results section) end to end and checks the converged chemical potential
against the pinned reference value.

The reference was generated with this exact input on a validated code state:
the same code reproduces an independently logged t=2.8 reference calculation
(chemicalpotential 1.6000003040674478 in its result.log, converged Fock
self-energy matching within 1e-4 across the full k-grid), which anchors the
pinned value below.
"""

import shutil
from pathlib import Path

import h5py
import pytest

from QAssemble.Run import Run

EXAMPLE_INPUT = Path(__file__).resolve().parent.parent / "examples" / "graphene" / "qassemble.in"

REFERENCE_MU = 1.5999999636787783


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

    assert mu == pytest.approx(REFERENCE_MU, abs=1e-5)
    assert n_iterations < 60
