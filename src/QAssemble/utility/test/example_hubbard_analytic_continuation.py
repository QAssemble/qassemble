"""Example: analytic continuation of the bare Green's function G0(i w_n)
for a single-band Hubbard model on a simple-cubic lattice.

Model
-----
Single orbital per site, nearest-neighbour hopping t = 1 on a cubic lattice
with primitive vectors RVec = [[1,0,0],[0,1,0],[0,0,1]]. The tight-binding
dispersion is

    eps(k) = -2 t [ cos(kx) + cos(ky) + cos(kz) ],     bandwidth = 12 t,

and the bare lattice Green's function is

    G0(k, i w_n) = 1 / ( i w_n - eps(k) ).

We build G0 on the DLR Matsubara grid (beta = 100, cutoff = 50 eV), then use
FLatDyn.AnalyticContinuation to continue it to the real axis and obtain the
k-resolved spectral function A(k, w) and the k-summed density of states.

Run with:
    export QAssemble=/path/to/DiagE
    python example_hubbard_analytic_continuation.py
"""
import numpy as np

from QAssemble.Crystal import Crystal
from QAssemble.utility.DLR import DLR
from QAssemble.FLatDyn import G0


# ----------------------------------------------------------------------
# Model / numerics parameters
# ----------------------------------------------------------------------
T_HOP = 1.0                       # nearest-neighbour hopping t
RVEC = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
KGRID = [10, 10, 10]              # 10^3 = 1000 k-points
BETA = 100.0                      # inverse temperature [eV^-1]
CUTOFF = 50.0                     # DLR Matsubara cutoff [eV]
ETA = 0.05                        # real-axis broadening
WMIN, WMAX, NW = -8.0, 8.0, 801   # real-frequency grid [eV]


def build_hamtb(crystal: Crystal, t: float) -> np.ndarray:
    """Simple-cubic tight-binding H0(k), shape (norb, norb, ns, nk).

    For one orbital per site this is just the diagonal dispersion
    eps(k) = -2 t [cos(kx) + cos(ky) + cos(kz)]. crystal.kpoint holds
    fractional coordinates in [0, 1), so k.r = 2*pi*frac.
    """
    norb = len(crystal.find)
    ns = crystal.ns
    nk = crystal.kpoint.shape[0]

    kx, ky, kz = crystal.kpoint[:, 0], crystal.kpoint[:, 1], crystal.kpoint[:, 2]
    eps = -2.0 * t * (
        np.cos(2.0 * np.pi * kx)
        + np.cos(2.0 * np.pi * ky)
        + np.cos(2.0 * np.pi * kz)
    )

    hamtb = np.zeros((norb, norb, ns, nk), dtype=np.complex128, order="F")
    hamtb[0, 0, 0, :] = eps
    return hamtb


def main() -> None:
    # 1. Crystal: 1 orbital per site, cubic lattice, 10^3 k-mesh.
    crystal = Crystal({
        "RVec": RVEC,
        "Basis": [[[0, 0, 0], 1]],   # one site, one orbital
        "NSpin": 1,                  # paramagnetic
        "NElec": 1.0,
        "KGrid": KGRID,
    })
    nk = crystal.kpoint.shape[0]
    print(f"Crystal: norb={len(crystal.find)}, ns={crystal.ns}, nk={nk}")

    # 2. Tight-binding Hamiltonian H0(k).
    hamtb = build_hamtb(crystal, T_HOP)
    eps = hamtb[0, 0, 0, :].real
    print(f"Dispersion eps(k): min={eps.min():.3f}, max={eps.max():.3f} "
          f"(expected -6t..+6t = {-6*T_HOP:.1f}..{6*T_HOP:.1f})")

    # 3. DLR grid (beta = 100, cutoff = 50 eV).
    dlr = DLR({"beta": BETA, "cutoff": CUTOFF, "eps": 1e-12})
    print(f"DLR: beta={dlr.beta}, cutoff={CUTOFF} eV, "
          f"n_matsubara={len(dlr.omega)}, lambF={dlr.lambF:.1f}")

    # 4. Bare Green's function G0(k, i w_n) on the DLR Matsubara grid.
    #    G0.kf has shape (norb, norb, ns, nk, nfreq_dlr).
    g0 = G0(crystal=crystal, dlr=dlr, hamtb=hamtb, hdf5file=None)
    print(f"G0(i w_n) shape: {g0.kf.shape}")

    # 5. Analytic continuation to the real axis.
    #    gret = retarded G^R(k, w), akf = A(k, w) = -Im G^R / pi.
    wreal = np.linspace(WMIN, WMAX, NW)
    gret, akf = g0.AnalyticContinuation(g0.kf, wreal, eta=ETA)
    print(f"Continued: gret {gret.shape}, akf {akf.shape}")

    # 6. Physics checks / observables.
    a_kw = akf[0, 0, 0, :, :].real           # (nk, nw)
    dos = a_kw.mean(axis=0)                   # k-summed density of states
    trapz = getattr(np, "trapezoid", np.trapz)

    # Gamma point (k=0) has eps = -6t; its spectral peak should sit there.
    a_gamma = akf[0, 0, 0, 0, :].real
    w_peak_gamma = wreal[np.argmax(a_gamma)]
    print(f"Gamma point: eps={eps[0]:.3f}, A(k=0,w) peak at w={w_peak_gamma:.3f}")
    print(f"DOS: peak at w={wreal[np.argmax(dos)]:.3f}, "
          f"integral(DOS) = {trapz(dos, wreal):.4f} (expect ~1)")

    # 7. Optional plot (skipped silently if matplotlib is unavailable).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(wreal, dos, lw=1.5)
        ax.set_xlabel(r"$\omega$ [eV]")
        ax.set_ylabel(r"DOS  $\frac{1}{N_k}\sum_k A(k,\omega)$")
        ax.set_title("Single-band cubic Hubbard $G_0$: DLR analytic continuation")
        ax.axvline(0.0, color="k", lw=0.5, ls=":")
        fig.tight_layout()
        out = "hubbard_dos.png"
        fig.savefig(out, dpi=150)
        print(f"Saved DOS plot to {out}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
