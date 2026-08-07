"""End-to-end integration tests: run TB/HF/GW through `Run` and inspect the HDF5 output.

Golden chemical potentials were generated with this test's exact inputs on the
manuscript graphene model and cross-checked against the reference calculation
(mu = 1.6000003... at KGrid 25x25x1); reruns on the same platform are bit-identical.
"""

import h5py
import numpy as np
import pytest

from QAssemble.Run import Run

from conftest import graphene_sections, write_qassemble_input

HF_MU_GOLDEN = 1.6000000000000227
GW_MU_GOLDEN = 1.6000014233043185


def _run(tmp_path, monkeypatch, **kwargs):
    prefix = tmp_path / "calc"
    input_path = tmp_path / "qassemble.in"
    write_qassemble_input(input_path, graphene_sections(prefix, **kwargs))
    monkeypatch.chdir(tmp_path)
    Run(input_file=input_path)
    return h5py.File(f"{prefix}.h5", "r")


def test_tb_h0k_matches_analytic_graphene_dispersion(tmp_path, monkeypatch):
    nk_lin = 6
    with _run(tmp_path, monkeypatch, method="tb", kgrid=(nk_lin, nk_lin, 1)) as h5:
        h0k = h5["/tb/H0/h0k"][()]

    assert h0k.shape == (2, 2, 1, nk_lin * nk_lin)

    # Hermiticity at every k
    np.testing.assert_allclose(
        h0k, np.conj(h0k.transpose(1, 0, 2, 3)), atol=1e-12, rtol=0
    )

    # |H01(k)| = t |1 + e^{-i k.a1} + e^{-i k.a2}| is basis-phase independent,
    # so compare the sorted values over the full grid.
    expected = sorted(
        2.8 * abs(1 + np.exp(-2j * np.pi * m1 / nk_lin) + np.exp(-2j * np.pi * m2 / nk_lin))
        for m1 in range(nk_lin)
        for m2 in range(nk_lin)
    )
    np.testing.assert_allclose(sorted(np.abs(h0k[0, 1, 0, :])), expected, atol=1e-10)

    # Bipartite lattice with zero onsite energy: eigenvalues come in +/- pairs,
    # and a 6x6 grid contains both Dirac points (two zero modes).
    eig = np.linalg.eigvalsh(h0k[:, :, 0, :].transpose(2, 0, 1))
    np.testing.assert_allclose(eig[:, 0], -eig[:, 1], atol=1e-10)
    assert int(np.sum(np.abs(h0k[0, 1, 0, :]) < 1e-10)) == 2


def test_hf_converges_to_reference_chemical_potential(tmp_path, monkeypatch):
    with _run(tmp_path, monkeypatch, method="hf", nscf=500) as h5:
        mu = h5["/hf/H/mu"][()]
        hk = h5["/hf/H/hk"][()]
        sigh = h5["/hf/SigH/sigh"][()]
        sigf = h5["/hf/SigF/sigf"][()]

    assert mu == pytest.approx(HF_MU_GOLDEN, abs=1e-7)
    np.testing.assert_allclose(
        hk, np.conj(hk.transpose(1, 0, 2, 3)), atol=1e-10, rtol=0
    )
    assert np.isfinite(sigh).all()
    assert np.isfinite(sigf).all()


def test_gw_converges_to_reference_chemical_potential(tmp_path, monkeypatch):
    with _run(tmp_path, monkeypatch, method="gw", nscf=100) as h5:
        assert set(h5["/gw"].keys()) >= {
            "H0", "G0", "G", "SigH", "SigF", "SigGWC", "P", "W", "V",
        }
        mu = h5["/gw/G/mu"][()]
        gkf = h5["/gw/G/gkf"][()]
        n_iterations = max(
            int(name.split(".")[1]) for name in h5["/gw/G"] if name.startswith("gkf.")
        )

    # abs=1e-5: the converged SCF fixed point drifts by O(1e-6) across
    # platforms/BLAS builds (observed 2.4e-6 between macOS and ubuntu CI).
    assert mu == pytest.approx(GW_MU_GOLDEN, abs=1e-5)
    assert np.isfinite(gkf).all()
    # Early convergence exit must trigger well before itermax (6 iterations when pinned)
    assert n_iterations < 50
