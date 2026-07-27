"""Validate the elastic causal projection against stored glob.h5 data.

Usage:
    python validate_glob_causal.py <glob.h5> --beta <beta> --cutoff <cutoff> [--eps 1e-12]

Re-applies the new projection pipeline (log-spaced sigma tail fit + elastic
moment penalties + offset split + clipped fallback) to every stored

    <...>/Hyb/hyb.<it>.<key>                              (fermion, DLR grid)
    <...>/BWeiss/bweiss.<it>.<key>_correlated_uniform     (boson zero-static)
    <...>/PImp/pimp.<it>.<key>                            (boson zero-static)
    <...>/WLoc/wloc.<it>.<key>                            (boson static-fit)

channel (the bosonic sampling grid is detected from the frequency dimension)
and prints a per-dataset summary table: projection success rate, skip
(fast-path) / elastic / clipped counts, offset-drop frequency, maximum causal
sign violation, maximum moment deviation in sigma units, and the relative
change of each channel.  The h5 file is opened read-only and never written.

Expected on the production data that motivated this pipeline: ~100% success
(the historical hard-equality path succeeded 0/15 times on the fermion side)
with relative changes in the 0.3--0.7% band.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from types import SimpleNamespace

import h5py
import numpy as np

from QAssemble.BLocDyn import BLocDyn
from QAssemble.FLocDyn import FLocDyn
from QAssemble.utility.Causal import CausalBosonProjector, CausalFermionProjector
from QAssemble.utility.DLR import DLR

_HYB_RE = re.compile(r"^hyb\.\d+\.[^.]+$")
_BWEISS_RE = re.compile(r"^bweiss\.\d+\..+_correlated_uniform$")
# The digit requirement after the stem dot keeps the *_brd_prev.<key> fallback
# caches out of the scan.
_PIMP_RE = re.compile(r"^pimp\.\d+\.[^.]+$")
_WLOC_RE = re.compile(r"^wloc\.\d+\.[^.]+$")


def _collect_datasets(h5file):
    """All (path, kind) pairs for stored projection inputs.

    Kinds: ``fermion`` (Hyb), ``boson_zero`` (BWeiss correlated bath and Pimp,
    zero high-frequency limit), ``boson_fit`` (Wloc, static split off by a
    tail fit before validation).
    """
    found = []

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        parts = name.split("/")
        base = parts[-1]
        parent = parts[-2] if len(parts) > 1 else ""
        if parent == "Hyb" and _HYB_RE.match(base):
            found.append((name, "fermion"))
        elif parent == "BWeiss" and _BWEISS_RE.match(base):
            found.append((name, "boson_zero"))
        elif parent == "PImp" and _PIMP_RE.match(base):
            found.append((name, "boson_zero"))
        elif parent == "WLoc" and _WLOC_RE.match(base):
            found.append((name, "boson_fit"))

    h5file.visititems(visit)
    return sorted(found)


def _scan_dyn_json_zeros(run_dir):
    """Report all-zero channels in the run's dyn.json files.

    A zero retarded interaction handed to CTQMC is the symptom this pipeline
    exists to prevent; scanning the actual dyn.*.json files catches it at the
    consumer.  Returns the number of all-zero channels found.
    """
    import json
    import os

    zero_channels = 0
    dyn_files = []
    for root, _dirs, files in os.walk(run_dir):
        for fname in files:
            if fname == "dyn.json" or (
                fname.startswith("dyn.") and fname.endswith(".json")
            ):
                dyn_files.append(os.path.join(root, fname))
    for path in sorted(dyn_files):
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"dyn-scan {path}: ERROR {exc}")
            continue
        for channel, values in payload.items():
            arr = np.asarray(values, dtype=float)
            if arr.size and np.all(np.abs(arr) < 1.0e-14):
                zero_channels += 1
                print(
                    f"dyn-scan WARNING: {path} channel '{channel}' is "
                    f"ALL ZERO ({arr.size} points)"
                )
    print(
        f"dyn-scan: {len(dyn_files)} file(s), "
        f"{zero_channels} all-zero channel(s)"
    )
    return zero_channels


class _Stats:
    def __init__(self):
        self.channels = 0
        self.success = 0
        self.skipped = 0
        self.elastic = 0
        self.clipped = 0
        self.offset_drops = 0
        self.max_sign = 0.0
        self.max_moment_sigma = 0.0
        self.rel_changes = []

    def record(self, validation):
        self.channels += 1
        if validation.get("valid", False):
            self.success += 1
        if validation.get("skipped", False):
            self.skipped += 1
        if validation.get("elastic", False):
            self.elastic += 1
        if validation.get("clipped", False):
            self.clipped += 1
        if abs(validation.get("c0_dropped", 0.0) or 0.0) > 0.0:
            self.offset_drops += 1
        self.max_sign = max(self.max_sign, validation.get("coefficient_violation", 0.0))
        ratio = validation.get("moment_residual_sigma")
        if ratio is not None and np.isfinite(ratio):
            self.max_moment_sigma = max(self.max_moment_sigma, ratio)
        rel = validation.get("relative_change", np.nan)
        if np.isfinite(rel):
            self.rel_changes.append(rel)

    def row(self, name):
        rel = np.asarray(self.rel_changes) if self.rel_changes else np.array([np.nan])
        return (
            f"{name:<44s} {self.channels:>4d} {self.success:>4d} "
            f"{self.skipped:>4d} {self.elastic:>4d} {self.clipped:>4d} "
            f"{self.offset_drops:>4d} {self.max_sign:>9.2e} "
            f"{self.max_moment_sigma:>8.2f} "
            f"{np.nanmean(rel):>9.3e} {np.nanmax(rel):>9.3e}"
        )


def _validate_fermion(data, dlr, stats):
    arr = np.asarray(data, dtype=np.complex128)
    if arr.ndim == 3:
        arr = arr[:, :, np.newaxis, :]
    norb, _, ns, nfreq = arr.shape
    loc = FLocDyn(SimpleNamespace(ns=ns), dlr, None)
    if nfreq == len(dlr.omega):
        uniform = dlr.MatsubaraDLR2UniformGrid(arr, sign=-1)
    else:
        uniform = arr
    # DLR2Uniform returns the positive-only grid; expand to the full signed
    # grid (no-op if already full) for the tail fit and the DLR conversion.
    uniform = loc._ExpandPositiveUniform(uniform)
    moment, high, sigma = loc.Moment(uniform, grid="uniform", return_sigma=True)
    arr_dlr = dlr.MatsubaraUniformGrid2DLR(uniform, sign=-1)

    projector = CausalFermionProjector(
        d=dlr.dF,
        beta=dlr.beta,
        omega=np.asarray(dlr.omega, dtype=float),
        constraint_tol="auto",
        raise_on_failure=False,
    )
    for js in range(ns):
        for iorb in range(norb):
            tail_coeffs = np.empty(4, dtype=float)
            tail_coeffs[0] = float(np.real(high[iorb, iorb, js]))
            tail_coeffs[1:] = np.real(moment[iorb, iorb, js, :])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                projector.project(
                    arr_dlr[iorb, iorb, js, :],
                    tail_coeffs=tail_coeffs,
                    moment_sigma=sigma[iorb, iorb, js, 1:],
                )
            stats.record(projector.last_validation)


def _validate_boson(data, dlr, stats, mode="zero"):
    """Validate one stored bosonic dataset.

    ``mode="zero"``: zero high-frequency limit (BWeiss correlated bath, Pimp).
    ``mode="fit"``:  the static part is estimated by the c0 tail fit and split
    off before validation (Wloc, which stores the full W including v).
    The sampling grid is detected from the frequency dimension: DLR-grid data
    (Pimp/Wloc `Save` outputs) skips the uniform interpolation.
    """
    arr = np.asarray(data, dtype=np.complex128)
    if arr.ndim != 5:
        raise ValueError(f"expected 5D bosonic data, got {arr.ndim}D")
    norb, _, ns, _, nfreq = arr.shape
    bl = BLocDyn(SimpleNamespace(ns=ns), dlr, None)
    grid = "dlr" if nfreq == len(dlr.nu) else "uniform"
    highzero = mode == "zero"
    moment, high, sigma = bl.Moment(
        arr, grid=grid, oddzero=highzero, highzero=highzero, return_sigma=True
    )
    if grid == "dlr":
        arr_dlr = arr
    else:
        nu_uniform = np.asarray(dlr.MatsubaraBosonUniform(), dtype=float)
        arr_dlr = dlr.MatsubaraUniformGrid2DLR(arr, omega=nu_uniform, sign=1)

    projector = CausalBosonProjector(
        d=dlr.dB,
        beta=dlr.beta,
        fit_omega=np.asarray(dlr.nu, dtype=float),
        coefficient_sign=-1,
        reflection_symmetry=True,
        constraint_tol="auto",
        raise_on_failure=False,
    )
    for is_ in range(ns):
        for iorb in range(norb):
            for jorb in range(iorb, norb):
                c0 = complex(high[iorb, jorb, is_, is_])
                tail_coeffs = np.empty(4, dtype=float)
                tail_coeffs[0] = 0.0
                tail_coeffs[1:] = np.real(moment[iorb, jorb, is_, is_, :])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    projector.project(
                        arr_dlr[iorb, jorb, is_, is_, :] - c0,
                        tail_coeffs=tail_coeffs,
                        moment_sigma=sigma[iorb, jorb, is_, is_, 1:3],
                    )
                stats.record(projector.last_validation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("h5file", help="path to glob.h5 (opened read-only)")
    parser.add_argument("--beta", type=float, required=True, help="inverse temperature")
    parser.add_argument("--cutoff", type=float, required=True, help="DLR cutoff Lambda/beta")
    parser.add_argument("--eps", type=float, default=1.0e-12, help="DLR accuracy")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="optionally scan this run directory's dyn.*.json files for "
        "all-zero retarded-interaction channels",
    )
    args = parser.parse_args(argv)

    if args.run_dir is not None:
        _scan_dyn_json_zeros(args.run_dir)

    dlr = DLR({"beta": args.beta, "cutoff": args.cutoff, "eps": args.eps})

    header = (
        f"{'dataset':<44s} {'chan':>4s} {'ok':>4s} {'skip':>4s} {'elas':>4s} "
        f"{'clip':>4s} {'drop':>4s} {'signviol':>9s} {'mom/sig':>8s} "
        f"{'rel_avg':>9s} {'rel_max':>9s}"
    )
    total = {"fermion": _Stats(), "boson": _Stats()}
    with h5py.File(args.h5file, "r") as h5:
        datasets = _collect_datasets(h5)
        if not datasets:
            print(
                "no hyb.<it>.<key> / bweiss.<it>.<key>_correlated_uniform / "
                "pimp.<it>.<key> / wloc.<it>.<key> datasets found"
            )
            return 1
        print(header)
        print("-" * len(header))
        for name, kind in datasets:
            stats = _Stats()
            try:
                if kind == "fermion":
                    _validate_fermion(h5[name][...], dlr, stats)
                else:
                    _validate_boson(
                        h5[name][...], dlr, stats,
                        mode="fit" if kind == "boson_fit" else "zero",
                    )
            except Exception as exc:  # keep scanning the rest of the file
                print(f"{name:<44s} ERROR: {exc}")
                continue
            print(stats.row(name))
            agg = total["fermion" if kind == "fermion" else "boson"]
            agg.channels += stats.channels
            agg.success += stats.success
            agg.skipped += stats.skipped
            agg.elastic += stats.elastic
            agg.clipped += stats.clipped
            agg.offset_drops += stats.offset_drops
            agg.max_sign = max(agg.max_sign, stats.max_sign)
            agg.max_moment_sigma = max(agg.max_moment_sigma, stats.max_moment_sigma)
            agg.rel_changes.extend(stats.rel_changes)
        print("-" * len(header))
        for kind in ("fermion", "boson"):
            if total[kind].channels:
                print(total[kind].row(f"TOTAL ({kind})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
