#!/usr/bin/env python3
"""Compare DLR->uniform Matsubara reconstruction schemes on real hyb data.

Reads DLR-node hybridization from glob.h5(.bak) and rebuilds it on the uniform
Matsubara grid three ways, then scores each by how flat ``Re D * w**2`` is in
the tail (the exact value is a constant, so any curvature is reconstruction
error).

Schemes
-------
current
    What ``DLR.MatsubaraDLR2Uniform`` does today: square LU solve for the DLR
    coefficients, then ``eval_dlr_freq``.  This is the path that blows up.
(a) regularised
    Same DLR evaluation, but the coefficients come from a truncated-SVD least
    squares solve instead of the square LU.  Kills the catastrophic
    cancellation without touching ``eps`` or ``MatsubaraCutOff``.
(b) interpolation
    Skip the coefficients entirely and interpolate the DLR node values onto the
    uniform grid.  Three variants are scored: linear in w, linear in 1/w, and
    cubic in log(w).  The 1/w variant is the interesting one -- a 1/w**2 tail is
    nearly straight in that variable.

Usage
-----
    python docs/compare_dlr2uniform.py
    python docs/compare_dlr2uniform.py --h5 ~/Downloads/glob.h5.bak --iters 27-32
    python docs/compare_dlr2uniform.py --orbital 1 --no-plot

Nothing in QAssemble is modified; this is a read-only diagnostic.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Import QAssemble from the repo checkout this script lives in.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))


# --------------------------------------------------------------------------
# reconstruction schemes
# --------------------------------------------------------------------------
def coeffs_lu(d, fnode, beta):
    """Current path: square LU solve, as pydlr ``dlr_from_matsubara`` does it.

    Called with a 2D RHS directly (pydlr's wrapper rejects a 3D one, and
    ``DLR.MatsubaraUniform2DLR`` flattens for the same reason).
    """
    from scipy.linalg import lu_solve

    c = lu_solve((d.dF.dlrmf2cf, d.dF.mf2cfpiv), fnode[:, None] / beta)
    return np.asarray(c).reshape(-1)


def design_matrix(d, w, beta):
    """DLR design matrix in pydlr's own convention.

    ``eval_dlr_freq`` computes ``conj( sum_x c_x / (z + dlrrf_x/beta) )`` -- note
    the plus sign, the beta-scaled poles, and the trailing conjugate.  Matching
    it exactly is what makes the (a) coefficients drop into the existing
    evaluator unchanged.
    """
    z = 1j * np.asarray(w, dtype=np.float64)
    return 1.0 / (z[:, None] + (d.dF.dlrrf / beta)[None, :])


def coeffs_svd(d, fnode, beta, rcond):
    """(a) Truncated-SVD least squares in place of the square LU solve.

    ``dlrmf2cf`` is already LU-factored, so the design matrix is rebuilt from
    the DLR poles and solved in a rank-truncated sense.  Modes below ``rcond``
    are exactly the ones whose coefficients blow up under the square solve.
    """
    A = design_matrix(d, d.omega, beta)
    c, *_ = np.linalg.lstsq(A, np.conj(fnode), rcond=rcond)
    return c


def eval_from_dlrrf(d, c, w_target, beta):
    """Evaluate the DLR expansion on an arbitrary frequency grid."""
    return np.conj(design_matrix(d, w_target, beta) @ c)


def interp_variants(w_node, f_node, w_target):
    """(b) Coefficient-free interpolation, three variants.

    The DLR Matsubara grid straddles zero (48 negative / 47 positive nodes here)
    while the uniform target grid is positive-only, so the positive branch is
    selected first.  Real and imaginary parts are then interpolated
    independently, mirroring ``DLR._interp_to_grid``.
    """
    from scipy.interpolate import interp1d

    pos = w_node > 0
    xs_w, fs = w_node[pos], f_node[pos]
    order = np.argsort(xs_w)
    xs_w, fs = xs_w[order], fs[order]

    # Clip the target to the node span; interp1d refuses to extrapolate and the
    # uniform grid can reach marginally past the outermost node.
    xt = np.clip(w_target, xs_w[0], xs_w[-1])

    out = {}

    def _both(xs, xtt, kind, label):
        re = interp1d(xs, fs.real, kind=kind, assume_sorted=True)(xtt)
        im = interp1d(xs, fs.imag, kind=kind, assume_sorted=True)(xtt)
        out[label] = re + 1j * im

    # linear in w -- the naive reading of "use interp1d"
    _both(xs_w, xt, "linear", "(b1) interp linear in w")

    # linear in 1/w -- a c/w**2 tail is nearly straight in this variable, so the
    # systematic bias of linear interpolation across sparse log-spaced nodes
    # largely cancels.  1/w descends, so re-sort to keep the source ascending.
    inv_s, inv_t = 1.0 / xs_w, 1.0 / xt
    o2 = np.argsort(inv_s)
    re = interp1d(inv_s[o2], fs.real[o2], kind="linear", assume_sorted=True)(inv_t)
    im = interp1d(inv_s[o2], fs.imag[o2], kind="linear", assume_sorted=True)(inv_t)
    out["(b2) interp linear in 1/w"] = re + 1j * im

    # cubic in log(w) -- positive DLR nodes are ~log-spaced, so this is the
    # variable in which they are closest to uniformly distributed.
    _both(np.log(xs_w), np.log(xt), "cubic", "(b3) interp cubic in log w")

    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def tail_report(w, f, wlo, whi):
    """Summarise ``Re f * w**2`` over the tail window.

    The exact hybridization tail is Re D -> -V2*e / w**2, so this product is a
    constant.  Spread is therefore pure reconstruction error.
    """
    m = (w >= wlo) & (w <= whi)
    if not m.any():
        return None
    p = f.real[m] * w[m] ** 2
    return {
        "n": int(m.sum()),
        "min": float(p.min()),
        "max": float(p.max()),
        "median": float(np.median(p)),
        "spread": float(p.max() - p.min()),
    }


def node_roundtrip_error(w_node, f_node, w_uni, f_uni, wlo=None):
    """Max |error| where the uniform grid lands closest to each DLR node.

    A scheme that cannot reproduce its own input nodes is broken; this is the
    check that exposed the current path (sign flip at w=137.88).  Only positive
    nodes are scored -- the uniform grid is positive-only -- and the comparison
    is restricted to the tail when ``wlo`` is given, since that is the window
    the downstream tail fit actually consumes.
    """
    m = w_node > 0 if wlo is None else (w_node >= wlo)
    wn, fn = w_node[m], f_node[m]
    if wn.size == 0:
        return float("nan"), float("nan"), 0j, 0j
    idx = np.abs(w_uni[:, None] - wn[None, :]).argmin(axis=0)
    err = np.abs(f_uni[idx] - fn)
    j = int(np.argmax(err))
    return float(err.max()), float(wn[j]), complex(fn[j]), complex(f_uni[idx[j]])


# --------------------------------------------------------------------------
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=os.path.expanduser("~/Downloads/glob.h5.bak"))
    ap.add_argument("--iters", default="27-32", help="e.g. 32 or 27-32 or 27,30,32")
    ap.add_argument("--imp", default="1", help="impurity index in the hyb.<it>.<imp> key")
    ap.add_argument("--orbital", type=int, default=0, help="diagonal orbital to analyse")
    ap.add_argument("--spin", type=int, default=0)
    ap.add_argument("--rcond", type=float, default=1e-8, help="truncated-SVD cutoff for (a)")
    ap.add_argument("--tail", default="46,461", help="tail fit window 'wlo,whi'")
    ap.add_argument("--plot", dest="plot", action="store_true", default=True)
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    ap.add_argument("--out", default=os.path.join(_REPO, "docs", "dlr2uniform_compare.png"))
    args = ap.parse_args()

    import h5py
    from QAssemble.utility.DLR import DLR

    wlo, whi = (float(x) for x in args.tail.split(","))
    iters = parse_iters(args.iters)

    with h5py.File(args.h5, "r") as f:
        ctrl = f["input/Control"]
        beta = float(ctrl["beta"][()])
        cutoff = float(ctrl["MatsubaraCutOff"][()])

        d = DLR({"beta": beta, "cutoff": cutoff, "eps": 1e-15})
        w_node = np.asarray(d.omega, dtype=np.float64)
        w_uni = d.MatsubaraFermionUniform()

        print(f"file    : {args.h5}")
        print(f"beta={beta}  MatsubaraCutOff={cutoff}  lambF={d.lambF:.1f}  "
              f"DLR rank={len(w_node)}")
        print(f"uniform : {len(w_uni)} points, dw={w_uni[1] - w_uni[0]:.6f}, "
              f"wmax={w_uni[-1]:.2f}")
        print(f"tail    : w in [{wlo}, {whi}]  "
              f"({int(((w_node >= wlo) & (w_node <= whi)).sum())} DLR nodes inside)")
        print(f"orbital : ({args.orbital},{args.orbital}) spin {args.spin}, "
              f"rcond={args.rcond:g}")

        cond = np.linalg.cond(design_matrix(d, w_node, beta))
        print(f"cond(A) : {cond:.2e}   (double-precision limit 1/eps = 4.5e+15"
              f" -> {'EXCEEDED' if cond > 4.5e15 else 'ok'})\n")

        curves_for_plot = None

        for it in iters:
            key = f"edmft/Hyb/hyb.{it}.{args.imp}"
            if key not in f:
                print(f"[iter {it}] missing {key}, skipped")
                continue
            raw = f[key][()]
            f_node = np.asarray(
                raw[args.orbital, args.orbital, args.spin, :], dtype=np.complex128
            )

            if f_node.shape[0] != w_node.shape[0]:
                print(f"[iter {it}] rank mismatch: data {f_node.shape[0]} "
                      f"vs DLR {w_node.shape[0]}, skipped")
                continue

            # Reference tail constant, taken from the input nodes themselves so
            # the score never depends on a reconstruction.  Averaged over the
            # positive nodes inside the tail window, where Re D ~ c/w**2 holds.
            tailmask = (w_node >= wlo) & (w_node <= whi)
            ref = float(np.median(f_node.real[tailmask] * w_node[tailmask] ** 2))

            c_lu = coeffs_lu(d, f_node, beta)
            c_sv = coeffs_svd(d, f_node, beta, args.rcond)

            curves = {}
            curves["current (square LU)"] = eval_from_dlrrf(d, c_lu, w_uni, beta)
            curves[f"(a) SVD rcond={args.rcond:g}"] = eval_from_dlrrf(
                d, c_sv, w_uni, beta
            )
            curves.update(interp_variants(w_node, f_node, w_uni))

            # The shipped implementation: Hermitian fold + cubic in 1/w.  Scored
            # alongside the hand-rolled variants so the diagnostic tracks what
            # production actually runs.
            arr = np.zeros((1, 1, 1, w_node.size), dtype=np.complex128)
            arr[0, 0, 0, :] = f_node
            curves["(*) shipped: fold + cubic 1/w"] = d.MatsubaraDLR2UniformGrid(
                np.asfortranarray(arr), sign=-1
            )[0, 0, 0, :]

            print(f"=== iteration {it} "
                  f"(reference Re D*w^2 at largest node = {ref:.3f}) ===")
            print(f"    coefficient norm  sum|c|:  LU {np.abs(c_lu).sum():.2e}   "
                  f"SVD {np.abs(c_sv).sum():.2e}")
            header = (f"{'scheme':<28} {'median':>9} {'min':>9} {'max':>9} "
                      f"{'spread':>9} {'node err':>10}")
            print(header)
            print("-" * len(header))
            for name, cur in curves.items():
                r = tail_report(w_uni, cur, wlo, whi)
                nerr, nw, fn, fu = node_roundtrip_error(
                    w_node, f_node, w_uni, cur, wlo=wlo
                )
                if r is None:
                    print(f"{name:<28} (tail window empty)")
                    continue
                print(f"{name:<28} {r['median']:9.3f} {r['min']:9.3f} "
                      f"{r['max']:9.3f} {r['spread']:9.3f} {nerr:10.2e}")
                if nerr > 1e-5:
                    print(f"{'':<28}   worst node w={nw:.2f}: "
                          f"{fn.real:+.3e} -> {fu.real:+.3e}"
                          f"{'  SIGN FLIP' if fn.real * fu.real < 0 else ''}")
            print()

            if curves_for_plot is None:
                curves_for_plot = (it, curves, f_node, ref)

    if args.plot and curves_for_plot is not None:
        it, curves, f_node, ref = curves_for_plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib unavailable; skipping plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
        for ax, (lo, hi) in zip(axes, [(w_uni[1], w_uni[-1]), (wlo, whi)]):
            m = (w_uni >= lo) & (w_uni <= hi)
            for name, cur in curves.items():
                ax.plot(w_uni[m], cur.real[m] * w_uni[m] ** 2, lw=1.2, label=name)
            nm = (w_node >= lo) & (w_node <= hi)
            ax.plot(w_node[nm], f_node.real[nm] * w_node[nm] ** 2, "ko",
                    ms=5, label="DLR nodes (input)")
            ax.axhline(ref, color="k", ls="--", lw=1, label=f"reference {ref:.2f}")
            ax.set_xlabel(r"$\omega_n$")
            ax.set_ylabel(r"Re $\Delta \cdot \omega_n^2$")
            ax.set_xscale("log")
        axes[0].set_title(f"iteration {it}: full range")
        axes[1].set_title(f"tail window [{wlo}, {whi}] (exact = flat line)")
        axes[1].set_ylim(ref - 3 * abs(ref), ref + 3 * abs(ref))
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out, dpi=140)
        print(f"plot written: {args.out}")


if __name__ == "__main__":
    main()
