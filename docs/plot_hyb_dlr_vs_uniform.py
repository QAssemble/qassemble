#!/usr/bin/env python3
"""Plot DLR->uniform hybridization: current code vs the stored data.

Reads iterations 27-32 from ``glob.h5(.bak)`` and, for each, converts the
DLR-grid hybridization onto the uniform Matsubara grid with the current
implementation (``method="interp"``) and with the legacy path
(``method="dlr"``), overlaying the DLR nodes that both are built from.

Real and imaginary parts are both plotted twice: over the full frequency range,
and zoomed into the tail (``--zoom``, default w >= 10).  The zoomed axes are
scaled to the corrected curve, so the legacy excursions leave the panel rather
than compressing the signal into a flat line -- past w ~ 20 the legacy error
(Re 2.7e-2) is larger than Re(Delta) itself (1.8e-2).

``edmft/Hyb`` stores **no** ``_uniform`` datasets, so the honest reference is
the DLR nodes themselves -- the input both paths must reproduce. Where a stored
uniform copy does exist (``edmft/SigCImp/sigimp.<it>.1_uniform``, written by the
old code through the same conversion), ``--with-sigma`` adds a second figure
comparing it against a fresh conversion of the same DLR data.

Usage
-----
    python docs/plot_hyb_dlr_vs_uniform.py
    python docs/plot_hyb_dlr_vs_uniform.py --iters 30-32 --orbital 1
    python docs/plot_hyb_dlr_vs_uniform.py --with-sigma

Read-only; nothing in QAssemble is modified.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))


def parse_iters(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def convert(dlr, arr4, method):
    """DLR grid -> uniform grid for one 4D fermionic array."""
    return dlr.MatsubaraDLR2UniformGrid(np.asfortranarray(arr4), sign=-1, method=method)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--h5", default=os.path.expanduser("~/Downloads/glob.h5.bak"))
    ap.add_argument("--iters", default="27-32")
    ap.add_argument("--imp", default="1")
    ap.add_argument("--orbital", type=int, default=0)
    ap.add_argument("--spin", type=int, default=0)
    ap.add_argument("--with-sigma", action="store_true",
                    help="also compare against stored sigimp *_uniform data")
    ap.add_argument("--zoom", type=float, default=10.0,
                    help="lower frequency bound for the zoomed Re/Im panels")
    ap.add_argument("--out", default=os.path.join(_REPO, "docs", "hyb_dlr_vs_uniform.png"))
    ap.add_argument("--out-sigma",
                    default=os.path.join(_REPO, "docs", "sigma_stored_vs_current.png"))
    args = ap.parse_args()

    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from QAssemble.utility.DLR import DLR

    iters = parse_iters(args.iters)
    io, so = args.orbital, args.spin

    with h5py.File(args.h5, "r") as f:
        beta = float(f["input/Control"]["beta"][()])
        cutoff = float(f["input/Control"]["MatsubaraCutOff"][()])
        dlr = DLR({"beta": beta, "cutoff": cutoff, "eps": 1.0e-15})
        w_node = np.asarray(dlr.omega, dtype=np.float64)
        w_uni = dlr.MatsubaraFermionUniform()
        pos = w_node > 0

        print(f"file  : {args.h5}")
        print(f"beta={beta}  MatsubaraCutOff={cutoff}  DLR rank={len(w_node)}  "
              f"uniform={len(w_uni)}")
        print(f"orbital ({io},{io}) spin {so}\n")

        rows = []
        for it in iters:
            key = f"edmft/Hyb/hyb.{it}.{args.imp}"
            if key not in f:
                print(f"[iter {it}] missing {key}")
                continue
            arr = np.asarray(f[key][()], dtype=np.complex128)
            node = arr[io, io, so, :]
            rows.append((it, node,
                         convert(dlr, arr, "interp")[io, io, so, :],
                         convert(dlr, arr, "dlr")[io, io, so, :]))

        # ---- figure 1: hybridization ------------------------------------
        n = len(rows)
        fig, axes = plt.subplots(n, 4, figsize=(21, 3.0 * n), squeeze=False)

        # Columns 1-2 are the full range; 3-4 zoom the same Re/Im into the high
        # frequency region.  Beyond w~20 the legacy error (Re 2.7e-2) exceeds the
        # signal itself (Re 1.8e-2), which the full-range axes cannot show.
        for r, (it, node, new, old) in enumerate(rows):
            panels = [
                ("Re $\\Delta$", new.real, old.real, node.real, None),
                ("Im $\\Delta$", new.imag, old.imag, node.imag, None),
                (f"Re $\\Delta$  ($\\omega_n \\geq {args.zoom:g}$)",
                 new.real, old.real, node.real, "zoom"),
                (f"Im $\\Delta$  ($\\omega_n \\geq {args.zoom:g}$)",
                 new.imag, old.imag, node.imag, "zoom"),
            ]
            for c, (title, yn, yo, ynode, mode) in enumerate(panels):
                ax = axes[r][c]
                ax.plot(w_uni, yo, color="tab:blue", lw=1.0, alpha=0.8,
                        label='stored path (method="dlr")')
                ax.plot(w_uni, yn, color="tab:red", lw=1.4,
                        label='current (method="interp")')
                ax.plot(w_node[pos], ynode[pos], "ko", ms=3.5,
                        label="DLR nodes (input)")
                ax.set_xscale("log")
                if r == 0:
                    ax.set_title(title)
                if c == 0:
                    ax.set_ylabel(f"iter {it}")
                if mode == "zoom":
                    ax.set_xlim(args.zoom, w_uni.max() * 1.05)
                    # Scale to the *correct* curve plus its nodes, so the legacy
                    # excursions run off-panel instead of flattening the signal.
                    m = w_uni >= args.zoom
                    mn = (w_node >= args.zoom) & pos
                    ref = np.concatenate([yn[m], ynode[mn]])
                    lo, hi = float(ref.min()), float(ref.max())
                    pad = 0.35 * max(hi - lo, 1.0e-12)
                    ax.set_ylim(lo - pad, hi + pad)
                    ax.axhline(0.0, color="k", ls=":", lw=0.7)
                if r == n - 1:
                    ax.set_xlabel(r"$\omega_n$")
                if r == 0 and c == 0:
                    ax.legend(fontsize=7)

        fig.suptitle(
            "Hybridization DLR -> uniform: current vs legacy conversion "
            f"(orbital {io}, spin {so})\n"
            "right two columns zoom the same Re/Im into the tail, scaled to the "
            "correct curve; edmft/Hyb stores no uniform copy, so the DLR nodes "
            "are the reference",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(args.out, dpi=130)
        print(f"wrote {args.out}")

        # ---- figure 2: self-energy against genuinely stored uniform -----
        if args.with_sigma:
            srows = []
            with h5py.File(args.h5, "r") as f2:
                for it in iters:
                    kd = f"edmft/SigCImp/sigimp.{it}.{args.imp}"
                    ku = kd + "_uniform"
                    if kd not in f2 or ku not in f2:
                        continue
                    a = np.asarray(f2[kd][()], dtype=np.complex128)
                    stored = np.asarray(f2[ku][()], dtype=np.complex128)[io, io, so, :]
                    srows.append((it, a[io, io, so, :],
                                  convert(dlr, a, "interp")[io, io, so, :],
                                  stored))

            if not srows:
                print("no stored sigimp *_uniform data found; skipping sigma figure")
                return

            # The stored uniform grid predates the current one and has its own
            # length; rebuild the matching grid from its size rather than
            # assuming it lines up.
            n_stored = srows[0][3].size
            w_stored = np.pi / beta * (2 * np.arange(n_stored) + 1)

            m = len(srows)
            fig2, ax2 = plt.subplots(m, 2, figsize=(12, 3.0 * m), squeeze=False)
            for r, (it, node, new, stored) in enumerate(srows):
                for c, (title, yn, ys, ynode) in enumerate([
                    ("Re $\\Sigma$", new.real, stored.real, node.real),
                    ("Im $\\Sigma$", new.imag, stored.imag, node.imag),
                ]):
                    ax = ax2[r][c]
                    ax.plot(w_stored, ys, color="tab:blue", lw=1.0, alpha=0.8,
                            label="stored *_uniform (old code)")
                    ax.plot(w_uni, yn, color="tab:red", lw=1.4,
                            label="current conversion")
                    ax.plot(w_node[pos], ynode[pos], "ko", ms=3.5, label="DLR nodes")
                    ax.set_xscale("log")
                    if r == 0:
                        ax.set_title(title)
                    if c == 0:
                        ax.set_ylabel(f"iter {it}")
                    if r == m - 1:
                        ax.set_xlabel(r"$\omega_n$")
                    if r == 0 and c == 0:
                        ax.legend(fontsize=7)

            fig2.suptitle(
                "Self-energy: stored uniform copy (written by the old code) vs a "
                f"fresh conversion of the same DLR data (orbital {io}, spin {so})",
                fontsize=11,
            )
            fig2.tight_layout(rect=(0, 0, 1, 0.96))
            fig2.savefig(args.out_sigma, dpi=130)
            print(f"wrote {args.out_sigma}")


if __name__ == "__main__":
    main()
