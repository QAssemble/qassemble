"""Post-process the graphene GW calculation.

Run `qassemble` in this directory first (produces `graphene.h5`), then

    python analyze.py

This reproduces the manuscript-style analysis from the converged HDF5 output:

1. Tight-binding vs GW quasiparticle band structure along Gamma-K-M-Gamma
   (`band_comparison.png`). The quasiparticle Hamiltonian uses the static
   self-energy estimate at the first Matsubara frequency and the Z-factor
   from its anti-Hermitian part.
2. Orbital-resolved density of states (`dos.png`).
3. The interacting Matsubara Green's function at the Gamma point
   (`green_matsubara.png`), evaluated from the DLR nodes onto a uniform grid.

Real-frequency spectral functions require analytic continuation and are not
part of this example.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np

from QAssemble import FPathDyn, FPathStc

KPATH = [[0, 0, 0], [2 / 3, 1 / 3, 0], [1 / 2, 1 / 2, 0], [0, 0, 0]]
KLABELS = [r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"]
NKPATH = 121


def load_results(h5file):
    """Read the converged GW datasets from `<prefix>.h5`."""
    with h5py.File(h5file, "r") as h5:
        return {
            "h0k": h5["/gw/H0/h0k"][()],
            "sigh": h5["/gw/SigH/sigmah"][()],
            "sigf": h5["/gw/SigF/sigmaf"][()],
            "siggwc": h5["/gw/SigGWC/sigmagwckf"][()],
            "mu": h5["/gw/G/mu"][()],
            "gkf": h5["/gw/G/gkf"][()],
        }


def qp_hamiltonian(flatstc, h0k, sigh, sigf, siggwc, mu, omega):
    """Quasiparticle Hamiltonian from the first-Matsubara static estimate.

    The dynamic self-energy Sigma(i omega_0) at the lowest positive Matsubara
    node is split into a Hermitian part (static level shift) and an
    anti-Hermitian part, from which Z^-1 = 1 - Im Sigma(i omega_0)/omega_0.
    Returns (hqp, z_eigval) with hqp = Z^1/2 (H_HF + Sigma_stc - mu) Z^1/2.
    """
    norb, _, ns, nk = h0k.shape
    i0 = int(np.argmin(np.where(omega > 0, omega, np.inf)))
    sig0 = siggwc[:, :, :, :, i0]

    sig_stc = np.zeros_like(h0k)
    zinv = np.zeros_like(h0k)
    for ik in range(nk):
        for js in range(ns):
            s = sig0[:, :, js, ik]
            sig_stc[:, :, js, ik] = (s + s.conj().T) / 2
            zinv[:, :, js, ik] = np.eye(norb) + 1j / (2 * omega[i0]) * (s - s.conj().T)

    z = flatstc.Inverse(zinv)
    eigval, eigvec = flatstc.Diagonalize(z, True)
    z_eig = np.array([np.diag(eigval[:, :, js, ik]).real
                      for ik in range(nk) for js in range(ns)])
    if not ((z_eig >= 0) & (z_eig <= 1)).all():
        raise ValueError("Z-factor eigenvalues outside [0, 1]; check the input data.")

    hqp = np.zeros_like(h0k)
    h_temp = h0k + sigh + sigf + sig_stc
    for ik in range(nk):
        for js in range(ns):
            zs = eigvec[:, :, js, ik] @ (
                np.sqrt(eigval[:, :, js, ik]) @ np.linalg.inv(eigvec[:, :, js, ik])
            )
            hqp[:, :, js, ik] = zs @ ((h_temp[:, :, js, ik] - np.eye(norb) * mu) @ zs)
    return hqp, z_eig


def band_on_path(fpathstc, mat_k):
    """Diagonalize a k-space matrix along the high-symmetry path."""
    mat_r = fpathstc.flatstc.K2R(mat_k)
    mat_path = fpathstc.R2K(matr=mat_r, kpoint=fpathstc.crystal.kpath)
    return fpathstc.flatstc.Diagonalize(mat_path)


def main(h5file="graphene.h5", outdir="."):
    fpathstc = FPathStc(hdf5file=h5file)
    fpathstc.crystal.Kpath(kpath=KPATH, nk=NKPATH)
    fpathdyn = FPathDyn(hdf5file=h5file)

    data = load_results(h5file)
    mu = data["mu"]
    print(f"Chemical potential mu = {mu:.6f} eV")

    hqp, z_eig = qp_hamiltonian(
        fpathstc.flatstc, data["h0k"], data["sigh"], data["sigf"],
        data["siggwc"], mu, fpathdyn.dlr.omega,
    )
    print(f"Z-factor eigenvalues: min {z_eig.min():.4f}, max {z_eig.max():.4f}")

    # 1. Band structure: TB vs GW quasiparticle
    tb_band = band_on_path(fpathstc, data["h0k"])
    qp_band = band_on_path(fpathstc, hqp)
    crystal = fpathstc.crystal

    # The bare TB bands are particle-hole symmetric around zero, which is their
    # own chemical potential at half filling; the QP bands are already
    # referenced to the interacting mu inside qp_hamiltonian.
    fig, ax = plt.subplots()
    for iorb in range(tb_band.shape[0]):
        ax.plot(crystal.kdist, tb_band[iorb, iorb, 0, :].real, "k-",
                linewidth=2.0, label="Tight-binding" if iorb == 0 else None)
        ax.plot(crystal.kdist, qp_band[iorb, iorb, 0, :].real, "b--",
                linewidth=1.5, label="GW quasiparticle" if iorb == 0 else None)
    ax.hlines(0, crystal.knode[0], crystal.knode[-1], colors="red",
              linestyles="dotted", linewidth=1)
    ax.set_xlim(crystal.knode[0], crystal.knode[-1])
    ax.set_xticks(crystal.knode)
    ax.set_xticklabels(KLABELS)
    ax.set_ylabel(r"$E - \mu$ (eV)")
    ax.legend()
    fig.savefig(f"{outdir}/band_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Density of states (orbital-resolved, Gaussian broadening)
    hqp_r = fpathstc.flatstc.K2R(hqp)
    energy, dos = fpathstc.Dos(
        matr=hqp_r, kgrid=[90, 90, 1], sigma=0.12, energyrange=[-5, 5]
    )

    fig, ax = plt.subplots()
    for iorb in range(dos.shape[0]):
        ax.plot(energy, dos[iorb, 0, :].real, label=f"orbital {iorb}")
    ax.set_xlabel(r"$E - \mu$ (eV)")
    ax.set_ylabel("DOS (states/eV)")
    ax.legend()
    fig.savefig(f"{outdir}/dos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Interacting Green's function at Gamma on the Matsubara axis
    ik_gamma = int(np.argmin(np.linalg.norm(crystal.kpoint, axis=-1)))
    omega_uniform = fpathdyn.dlr.MatsubaraFermionUniform()

    fig, ax = plt.subplots()
    for iorb in range(data["gkf"].shape[0]):
        g_dlr = data["gkf"][iorb, iorb, 0, ik_gamma, :]
        g_uniform = fpathdyn.dlr.MatsubaraDLR2Uniform(g_dlr)[:, 0, 0]
        ax.plot(omega_uniform, g_uniform.imag, "-",
                label=rf"orbital {iorb}")
        ax.plot(fpathdyn.dlr.omega, g_dlr.imag, "k.", markersize=4,
                label="DLR nodes" if iorb == 0 else None)
    ax.set_xlim(0, omega_uniform[-1] / 4)
    ax.set_xlabel(r"$\omega_n$ (eV)")
    ax.set_ylabel(r"Im $G(\Gamma, i\omega_n)$")
    ax.legend()
    fig.savefig(f"{outdir}/green_matsubara.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote band_comparison.png, dos.png, green_matsubara.png to {outdir}")
    return {"mu": mu, "z_min": z_eig.min(), "z_max": z_eig.max()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5file", default="graphene.h5")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()
    main(h5file=args.h5file, outdir=args.outdir)
