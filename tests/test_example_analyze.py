"""Regression test for the tutorial post-processing script examples/graphene/analyze.py.

Runs a small GW calculation and the full analysis on top of it, so API changes
that would break the documented workflow fail the fast suite.
"""

import sys
from pathlib import Path

import pytest

from QAssemble.Run import Run

from conftest import graphene_sections, write_qassemble_input

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "graphene"


@pytest.fixture
def analyze():
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        import analyze

        yield analyze
    finally:
        sys.path.remove(str(EXAMPLES_DIR))
        sys.modules.pop("analyze", None)


def test_analyze_produces_figures_and_sane_summary(tmp_path, monkeypatch, analyze):
    prefix = tmp_path / "calc"
    write_qassemble_input(tmp_path / "qassemble.in", graphene_sections(prefix))
    monkeypatch.chdir(tmp_path)
    Run(input_file=tmp_path / "qassemble.in")

    summary = analyze.main(h5file=f"{prefix}.h5", outdir=str(tmp_path))

    for name in ("band_comparison.png", "dos.png", "green_matsubara.png"):
        assert (tmp_path / name).stat().st_size > 0

    assert summary["mu"] == pytest.approx(1.6, abs=1e-3)
    # Weakly correlated regime: quasiparticle weight close to, and below, one.
    assert 0.9 < summary["z_min"] <= summary["z_max"] <= 1.0
