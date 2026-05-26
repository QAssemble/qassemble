import json
import logging
import os
import h5py
import numpy as np

logger = logging.getLogger("QAssemble")


class Convergence:
    """Method-agnostic convergence checker.

    Construction mirrors CorrelationFunction.__init__: takes `control`,
    inspects control['run']['method'], and configures itself accordingly.

    Two modes, exposed as two methods:
      - Check(...)       for hf/gw           (single-matrix + mu criterion)
      - CheckDelta(...)  for dmft/edmft/gw+edmft (abs+rel deltas, causality, diag)

    Returns (converged, info) tuples. Never raises for convergence
    decisions; driver retains all flow control (break, raise, gc.collect).

    DMFT-family path additionally exposes:
      - Start()                   to truncate convergence.jsonl
      - RecordCausalityFail(...)  to write a FAIL row before driver raises
      - self.causality_tol_hyb / self.causality_tol_sig / self.min_iter
        for driver-side checks
    """

    _ROW_COLS = [
        "iter", "causality_hyb", "causality_sig",
        "gcheck", "dG_iter", "dG_rel",
        "dSig_iter", "dSig_rel", "dmu",
        "n_imp", "ctqmc_sign", "histo_1", "histo_2", "mu",
    ]

    def __init__(self, control: dict):
        self.control = control
        self.method = control["run"]["method"]
        self.hdf5path = control["run"]["fn"]

        if self.method == "hf":
            self._init_hf()
        elif self.method == "gw":
            self._init_gw()
        elif self.method in ("dmft", "edmft", "gw+edmft"):
            self._init_delta()
        elif self.method == "tb":
            pass
        else:
            raise ValueError(f"Convergence: unknown method '{self.method}'")

    def _init_hf(self):
        run = self.control["run"]
        self.tol_f = float(run.get("tol_f", 1e-7))
        self.tol_mu = float(run.get("tol_mu", 0.01))

    def _init_gw(self):
        run = self.control["run"]
        self.tol_f = float(run.get("tol_f", 1e-6))
        self.tol_b = float(run.get("tol_b", 1e-4))
        self.tol_mu = float(run.get("tol_mu", 0.01))

    def _init_delta(self):
        run = self.control["run"]

        removed_keys = ("dmft_tol", "convergence_mode", "tol_G")
        present_removed = [k for k in removed_keys if k in run]
        if present_removed:
            raise ValueError(
                f"control['run'] contains removed keys {present_removed}. "
                f"Legacy convergence mode was removed (CTQMC noise floor made "
                f"|G_loc-G_imp|<=dmft_tol unreachable). "
                f"Convergence now uses delta criteria only: "
                f"tol_dG_abs, tol_dG_rel, tol_dsig_abs, tol_dsig_rel, tol_dmu. "
                f"Remove the listed keys from your input.ini to proceed."
            )

        self.tol_dG_abs = float(run.get("tol_dG_abs", 1e-5))
        self.tol_dG_rel = float(run.get("tol_dG_rel", 1e-3))
        self.tol_dsig_abs = float(run.get("tol_dsig_abs", 3e-4))
        self.tol_dsig_rel = float(run.get("tol_dsig_rel", 3e-3))
        self.tol_dmu = float(run.get("tol_dmu", 1e-5))
        self.causality_tol_hyb = float(run.get("causality_tol_hyb", 1e-8))
        self.causality_tol_sig = float(run.get("causality_tol_sig", 1e-8))
        self.min_iter = int(run.get("min_iter", 3))

        self._conv_table = []
        top_dir = os.path.dirname(os.path.abspath(self.hdf5path)) or "."
        self._conv_log_path = os.path.join(top_dir, "convergence.log")
        self._conv_jsonl_path = os.path.join(top_dir, "convergence.jsonl")

        self.prev = {"sigh": None, "sigf": None, "sigc": None,
                     "gloc_f": None, "mu": None}

    def Start(self):
        """Truncate convergence.jsonl. Called by the DMFT driver just before
        the iteration loop."""
        self._conv_table = []
        try:
            with open(self._conv_jsonl_path, "w"):
                pass
        except Exception as e:
            logger.warning(f"convergence.jsonl truncate failed: {e}")

    @staticmethod
    def SCFCheck(a, b) -> float:
        return np.abs(a - b).max()

    @staticmethod
    def SCFCheckDict(gloc: dict, gimp: dict) -> float:
        check = 0.0
        for key in gimp.keys():
            if key not in gloc:
                raise KeyError(f"Missing lattice Green's function for problem key '{key}'")
            check = max(check, Convergence.SCFCheck(gloc[key], gimp[key]))
        return check

    def Check(self, iter, *, fnew, fold, mu_new, mu_old,
              bnew=None, bold=None, npulay=0):
        """HF when bnew/bold are None; GW when provided.
        Returns (converged, info) where info contains 'fcheck', 'mucheck',
        and (GW only) 'bcheck'."""
        fcheck = self.SCFCheck(fnew, fold)
        mucheck = abs(mu_new - mu_old)
        info = {"fcheck": fcheck, "mucheck": mucheck}
        if bnew is None and bold is None:
            converged = (fcheck <= self.tol_f) and (mucheck <= self.tol_mu)
        else:
            bcheck = self.SCFCheck(bnew, bold)
            info["bcheck"] = bcheck
            converged = (
                (iter > npulay)
                and (fcheck <= self.tol_f)
                and (mucheck <= self.tol_mu)
                and (bcheck <= self.tol_b)
            )
        return converged, info

    def RecordCausalityFail(self, iter, *, kind, key, max_im, mu,
                            subreason=None, diag=None):
        """Append a FAIL row to convergence.log + JSONL + HDF5 mirror.
        kind ∈ {'hyb','sig'}. Driver raises immediately afterwards."""
        diag = diag or {}
        if kind == "hyb":
            causality_hyb = f"FAIL@{key}(Im={max_im:.3e})"
            causality_sig = "-"
            n_imp = "-"
            ctqmc_sign = "-"
            histo_1 = "-"
            histo_2 = "-"
        elif kind == "sig":
            tag = f"Im={max_im:.3e}"
            if subreason:
                tag = f"{tag},{subreason}"
            causality_hyb = "OK"
            causality_sig = f"FAIL@{key}({tag})"
            n_imp = diag.get("nimp")
            ctqmc_sign = diag.get("sign")
            histo_1 = diag.get("histo_m1")
            histo_2 = diag.get("histo_m2")
        else:
            raise ValueError(f"RecordCausalityFail: unknown kind '{kind}'")

        row = {
            "iter": iter,
            "causality_hyb": causality_hyb,
            "causality_sig": causality_sig,
            "gcheck": "-", "dG_iter": "-", "dG_rel": "-",
            "dSig_iter": "-", "dSig_rel": "-", "dmu": "-",
            "n_imp": n_imp, "ctqmc_sign": ctqmc_sign,
            "histo_1": histo_1, "histo_2": histo_2,
            "mu": float(mu),
        }
        self._write_convergence_log(row)

    def CheckDelta(self, iter, *,
                   gloc_next, sigmah_next, sigmaf_next, sigc_next,
                   mu_next,
                   gimp_by_key, diag_by_key, sig_rolled_back,
                   will_continue=True):
        """Per-iter DMFT convergence step. See plan for sequence."""
        gcheck = self.SCFCheckDict(gloc_next.f, gimp_by_key)

        dG_iter = dSig_iter = dmu = float("nan")
        dG_rel = dSig_rel = float("nan")
        delta_ready = (iter > self.min_iter) and (self.prev["gloc_f"] is not None)
        if delta_ready:
            dG_iter = 0.0
            dG_rel = 0.0
            matched_keys = 0
            for k in gloc_next.f.keys():
                if k not in self.prev["gloc_f"]:
                    continue
                check_k = self.SCFCheck(gloc_next.f[k], self.prev["gloc_f"][k])
                scale_k = max(float(np.abs(gloc_next.f[k]).max()), 1e-30)
                dG_iter = max(dG_iter, check_k)
                dG_rel = max(dG_rel, check_k / scale_k)
                matched_keys += 1
            if matched_keys == 0:
                logger.warning(
                    f"iter {iter}: gloc key-set fully changed; "
                    f"delta convergence forced to non-convergence (dG=inf)."
                )
                dG_iter = float("inf")
                dG_rel = float("inf")
            dSig_iter = max(
                self.SCFCheck(sigmah_next, self.prev["sigh"]),
                self.SCFCheck(sigmaf_next, self.prev["sigf"]),
                self.SCFCheck(sigc_next, self.prev["sigc"]),
            )
            s_scale = max(
                float(np.abs(sigc_next).max()),
                float(np.abs(sigmah_next).max()),
                float(np.abs(sigmaf_next).max()),
                1e-30,
            )
            dSig_rel = dSig_iter / s_scale
            dmu = abs(float(mu_next) - float(self.prev["mu"]))

        def _agg_diag(name, fn=np.mean):
            vals = [d[name] for d in diag_by_key.values()
                    if d.get(name) is not None]
            return float(fn(vals)) if vals else float("nan")

        row = {
            "iter": iter,
            "causality_hyb": "OK",
            "causality_sig": ",".join(sig_rolled_back) if sig_rolled_back else "OK",
            "gcheck": gcheck,
            "dG_iter": dG_iter,
            "dG_rel": dG_rel,
            "dSig_iter": dSig_iter,
            "dSig_rel": dSig_rel,
            "dmu": dmu,
            "n_imp": _agg_diag("nimp", np.sum),
            "ctqmc_sign": _agg_diag("sign", np.min),
            "histo_1": _agg_diag("histo_m1"),
            "histo_2": _agg_diag("histo_m2"),
            "mu": float(mu_next),
        }

        if delta_ready:
            converged = (
                dG_iter <= self.tol_dG_abs
                and dG_rel <= self.tol_dG_rel
                and dSig_iter <= self.tol_dsig_abs
                and dSig_rel <= self.tol_dsig_rel
                and dmu <= self.tol_dmu
            )
        else:
            converged = False

        gcheck_warn_threshold = max(self.tol_dG_abs * 1000, 1e-2)
        gcheck_warn_triggered = bool(converged and gcheck > gcheck_warn_threshold)

        self._write_convergence_log(row)

        if will_continue and not converged:
            self.prev = {
                "sigh": sigmah_next.copy(),
                "sigf": sigmaf_next.copy(),
                "sigc": sigc_next.copy(),
                "gloc_f": {k: v.copy() for k, v in gloc_next.f.items()},
                "mu": float(mu_next),
            }

        info = {
            "gcheck": gcheck,
            "dG_iter": dG_iter,
            "dG_rel": dG_rel,
            "dSig_iter": dSig_iter,
            "dSig_rel": dSig_rel,
            "dmu": dmu,
            "gcheck_warn_threshold": gcheck_warn_threshold,
            "gcheck_warn_triggered": gcheck_warn_triggered,
        }
        return converged, info

    def _write_convergence_log(self, row):
        self._conv_table.append(row)
        cols = self._ROW_COLS
        str_table = [[self._fmt_cell(r.get(c, "")) for c in cols]
                     for r in self._conv_table]
        widths = {c: max(len(c), max((len(s[i]) for s in str_table), default=0))
                  for i, c in enumerate(cols)}
        lines = ["  ".join(c.rjust(widths[c]) for c in cols)]
        for s in str_table:
            lines.append("  ".join(s[i].rjust(widths[c]) for i, c in enumerate(cols)))
        tmp = self._conv_log_path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp, self._conv_log_path)

        try:
            sanitized = {k: self._json_sanitize(v) for k, v in row.items()}
            with open(self._conv_jsonl_path, "a") as fh:
                fh.write(json.dumps(sanitized, default=self._json_default,
                                    allow_nan=False) + "\n")
        except Exception as e:
            logger.warning(f"convergence.jsonl append failed: {e}")

        try:
            with h5py.File(self.hdf5path + ".h5", "a") as h:
                grp = h.require_group("impurity_solver/convergence")
                if "table" in grp:
                    del grp["table"]
                grp.create_dataset("table", data=np.array(lines, dtype="S256"))
        except Exception as e:
            logger.warning(f"convergence.log HDF5 mirror failed: {e}")

    @staticmethod
    def _json_sanitize(v):
        if isinstance(v, float):
            return v if np.isfinite(v) else None
        if isinstance(v, (np.floating,)):
            f = float(v)
            return f if np.isfinite(f) else None
        if isinstance(v, (np.ndarray, list, tuple)):
            arr = np.asarray(v).ravel()
            if arr.size > 32:
                return {"summary": "truncated", "n": int(arr.size),
                        "head": [Convergence._json_sanitize(x.item()
                                 if hasattr(x, "item") else x) for x in arr[:8]]}
            return [Convergence._json_sanitize(x.item()
                    if hasattr(x, "item") else x) for x in arr]
        return v

    @staticmethod
    def _json_default(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            f = float(v)
            return f if np.isfinite(f) else None
        if isinstance(v, (np.complexfloating, complex)):
            c = complex(v)
            re = c.real if np.isfinite(c.real) else None
            im = c.imag if np.isfinite(c.imag) else None
            return {"re": re, "im": im}
        if isinstance(v, (np.ndarray, list, tuple)):
            return list(np.asarray(v).tolist())
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return str(v)

    @staticmethod
    def _fmt_cell(v):
        if v is None:
            return "-"
        if isinstance(v, str):
            return v if v else "-"
        if isinstance(v, (bool, np.bool_)):
            return "T" if bool(v) else "F"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):d}"
        if isinstance(v, (np.floating, float)):
            f = float(v)
            if not np.isfinite(f):
                return "nan" if np.isnan(f) else ("inf" if f > 0 else "-inf")
            return f"{f:.6e}"
        if isinstance(v, (complex, np.complexfloating)):
            c = complex(v)
            return f"({c.real:.4e}{'+' if c.imag >= 0 else ''}{c.imag:.4e}j)"
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v).ravel()
            if arr.size == 0:
                return "[]"
            head_val = arr[0].item() if hasattr(arr[0], "item") else arr[0]
            head = Convergence._fmt_cell(head_val)
            return head if arr.size == 1 else f"{head}|n={arr.size}"
        return str(v)
