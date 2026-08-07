import json
import logging
import os
import h5py
import numpy as np

logger = logging.getLogger("QAssemble")


class Convergence:
    """Method-agnostic convergence checker.

    All tolerance values are read from `control['run']` by canonical name:
      - Self tests: `tol_d<Name>_abs` AND `tol_d<Name>_rel`
      - Cross tests: `tol_<NameA>_<NameB>_abs`

    Missing key → the test is registered as `gating=False, passed=None`
    (observational only, not part of the AND verdict). A one-time
    `logger.info` per name names the missing key explicitly.

    Per-iteration usage:
        conv.StartIter(iter)
        conv.CheckSelfHDF5('GLoc', group='dmft', subgroup='GLoc',
                           current='gloc.2', previous='gloc.1', keys=...)
        conv.CheckSelf('mu',   value=..., kind='scalar')
        conv.CheckCrossHDF5('GLoc', name_b='GImp', group='dmft',
                            subgroup_a='GLoc', subgroup_b='GImp',
                            stem_a='gloc.2', stem_b='gimp.2', keys=...)
        conv.RecordDiagnostics(diag_by_key)
        converged, info = conv.Commit(iter, will_continue=...)

    HDF5 mirror group is `control['run'].get('convergence_hdf5_group',
    'convergence')`. Drivers may `setdefault` it before constructing
    `Convergence(control)`.
    """

    _MIN_DEFENSE_REMOVED_KEYS = ("dmft_tol", "convergence_mode", "tol_G")

    def __init__(self, control: dict):
        self.control = control
        self.method = control["run"].get("method")
        self.hdf5path = control["run"]["fn"]

        if self.method == "tb":
            self._init_tb_sentinels()
            return
        self._init_common()

    def _init_tb_sentinels(self):
        """Sentinel state for method=='tb'. Any convergence-API call on
        a tb instance raises RuntimeError with an actionable message."""
        self._tb_no_op = True
        self.min_iter = 0

    def _require_not_tb(self, method_name):
        if getattr(self, "_tb_no_op", False):
            raise RuntimeError(
                f"Convergence: method='tb' has no convergence loop; "
                f"driver should not call {method_name}() for tb runs."
            )

    def _init_common(self):
        run = self.control["run"]

        # Thin defense for callers that bypass Run.py (test scripts,
        # notebooks). Run.py also warns on a broader legacy key set.
        present_removed = [k for k in self._MIN_DEFENSE_REMOVED_KEYS
                           if k in run]
        if present_removed:
            raise ValueError(
                f"control['run'] contains removed keys {present_removed}. "
                f"These keys were eliminated when the legacy convergence "
                f"mode was deleted. Convergence now uses delta criteria "
                f"only (tol_d<Name>_abs/_rel, tol_<A>_<B>_abs). "
                f"Remove the listed keys from control['run']."
            )

        self._tol_run = run
        self.min_iter = int(run.get("min_iter", 0))
        self._conv_hdf5_group = run.get(
            "convergence_hdf5_group", "convergence"
        )
        self._self_prev = {}
        self._pending_prev = {}
        self._iter_buf = []
        self._iter_diag = {}
        self._iter_no = None
        self._ready_after = self.min_iter
        self._schema_cols = None
        self._schema_warned = False
        self._observational_warned = set()
        self._zero_gating_warned = False
        self._conv_table = []
        top_dir = os.path.dirname(os.path.abspath(self.hdf5path)) or "."
        self._conv_log_path = os.path.join(top_dir, "convergence.log")
        self._conv_jsonl_path = os.path.join(top_dir, "convergence.jsonl")

    def Start(self):
        """Reset convergence state and truncate convergence.jsonl."""
        self._require_not_tb("Start")
        self._self_prev = {}
        self._pending_prev = {}
        self._iter_buf = []
        self._iter_diag = {}
        self._iter_no = None
        self._schema_cols = None
        self._schema_warned = False
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
    def _max_diff_and_scale(a, b, kind):
        if kind == "array":
            abs_d = float(np.abs(a - b).max())
            scale = max(float(np.abs(a).max()), 1e-30)
            return abs_d, scale
        if kind == "dict":
            matched = a.keys() & b.keys()
            if not matched:
                return float("inf"), 1.0
            abs_d = 0.0
            scale = 1e-30
            for k in matched:
                abs_d = max(abs_d, float(np.abs(a[k] - b[k]).max()))
                scale = max(scale, float(np.abs(a[k]).max()))
            if a.keys() != b.keys():
                logger.warning(
                    "Convergence: dict key-set differs between old and new; "
                    f"comparing intersection of size {len(matched)}."
                )
            return abs_d, max(scale, 1e-30)
        if kind == "scalar":
            abs_d = abs(float(a) - float(b))
            scale = max(abs(float(a)), abs(float(b)), 1e-30)
            return abs_d, scale
        raise ValueError(f"_max_diff_and_scale: unknown kind {kind!r}")

    @staticmethod
    def _require_hdf5_dataset(handle, path):
        if path not in handle:
            raise KeyError(f"Convergence HDF5 dataset not found: {path}")
        return np.asarray(handle[path])

    @classmethod
    def _max_diff_and_scale_hdf5_keyed(cls, handle, paths_a, paths_b):
        if len(paths_a) != len(paths_b):
            raise ValueError("HDF5 convergence path lists must have same length")
        if not paths_a:
            raise ValueError("HDF5 convergence requires at least one key")

        abs_d = 0.0
        scale = 1e-30
        for path_a, path_b in zip(paths_a, paths_b):
            a = cls._require_hdf5_dataset(handle, path_a)
            b = cls._require_hdf5_dataset(handle, path_b)
            if a.shape != b.shape:
                raise ValueError(
                    "Convergence HDF5 shape mismatch: "
                    f"{path_a} has {a.shape}, {path_b} has {b.shape}"
                )
            abs_d = max(abs_d, float(np.abs(a - b).max()))
            scale = max(scale, float(np.abs(a).max()))
        return abs_d, max(scale, 1e-30)

    @staticmethod
    def _hdf5_keyed_paths(group, subgroup, stem, keys):
        return [
            f"{group}/{subgroup}/{stem}.{str(key)}"
            for key in keys
        ]

    @staticmethod
    def _copy_value(value, kind):
        if kind == "array":
            return np.asarray(value).copy()
        if kind == "dict":
            return {k: v.copy() for k, v in value.items()}
        if kind == "scalar":
            return float(value)
        raise ValueError(f"_copy_value: unknown kind {kind!r}")

    def _require_iter(self, method_name):
        if self._iter_no is None:
            raise RuntimeError(
                f"Convergence: call StartIter(iter) before {method_name}"
            )

    def seed_prev(self, name, value, kind):
        """Pre-seed a self-test previous value.

        May be called before or after Start(), but Start() resets seeded
        values. Call it before StartIter() for the iteration whose
        CheckSelf() comparison should use the seeded value.
        """
        self._require_not_tb("seed_prev")
        self._self_prev[name] = self._copy_value(value, kind)

    def StartIter(self, iter, *, ready_after=None):
        """Reset per-iter buffers. Must be called once per iter before
        CheckSelf/CheckCross/RecordDiagnostics/Commit."""
        self._require_not_tb("StartIter")
        self._iter_buf = []
        self._pending_prev = {}
        self._iter_diag = {}
        self._iter_no = int(iter)
        self._ready_after = (
            ready_after if ready_after is not None else self.min_iter
        )

    def _warn_observational(self, name, missing_keys):
        if name in self._observational_warned:
            return
        self._observational_warned.add(name)
        logger.info(
            f"Convergence: '{name}' test is observational; missing "
            f"{missing_keys} in control['run']. Add these keys to make "
            f"the test gate the converged decision."
        )

    def CheckSelf(self, name, *, value, kind, gating=True):
        """Register a self-test (old vs new) for `name`.

        Tolerances are read lazily from `self._tol_run`:
          - tol_abs: control['run']['tol_d<name>_abs']
          - tol_rel: control['run']['tol_d<name>_rel']

        If either tol is missing, the test is registered with
        gating=False, passed=None — observational only. Both abs and
        rel must be present for the test to gate.

        Gated by the active ready_after threshold: until iter > threshold
        (and a prev exists), passed=False (or None if observational).
        """
        self._require_not_tb("CheckSelf")
        self._require_iter("CheckSelf")
        abs_key = f"tol_d{name}_abs"
        rel_key = f"tol_d{name}_rel"
        tol_abs_raw = self._tol_run.get(abs_key)
        tol_rel_raw = self._tol_run.get(rel_key)
        observational = (tol_abs_raw is None) or (tol_rel_raw is None)

        prev = self._self_prev.get(name)
        ready = (prev is not None) and (self._iter_no > self._ready_after)
        if ready:
            abs_d, scale = self._max_diff_and_scale(value, prev, kind)
            rel_d = abs_d / scale
        else:
            abs_d = float("nan")
            rel_d = float("nan")

        if observational:
            missing = [k for k, v in ((abs_key, tol_abs_raw),
                                       (rel_key, tol_rel_raw)) if v is None]
            self._warn_observational(name, missing)
            tol_abs = None
            tol_rel = None
            passed = None
            effective_gating = False
        else:
            tol_abs = float(tol_abs_raw)
            tol_rel = float(tol_rel_raw)
            passed = bool(ready and (abs_d <= tol_abs) and (rel_d <= tol_rel))
            effective_gating = gating

        self._pending_prev[name] = (self._copy_value(value, kind), kind)
        self._iter_buf.append({
            "name": name,
            "kind_test": "self",
            "kind_value": kind,
            "abs": abs_d,
            "rel": rel_d,
            "tol_abs": tol_abs,
            "tol_rel": tol_rel,
            "passed": passed,
            "gating": effective_gating,
        })

    def CheckSelfHDF5(
        self,
        name,
        *,
        group,
        subgroup,
        current,
        previous,
        keys,
        gating=True,
    ):
        """Register a self-test by reading keyed datasets from HDF5."""
        self._require_not_tb("CheckSelfHDF5")
        self._require_iter("CheckSelfHDF5")
        key_list = list(keys)
        abs_key = f"tol_d{name}_abs"
        rel_key = f"tol_d{name}_rel"
        tol_abs_raw = self._tol_run.get(abs_key)
        tol_rel_raw = self._tol_run.get(rel_key)
        observational = (tol_abs_raw is None) or (tol_rel_raw is None)

        ready = self._iter_no > self._ready_after
        if ready:
            paths_current = self._hdf5_keyed_paths(
                group,
                subgroup,
                current,
                key_list,
            )
            paths_previous = self._hdf5_keyed_paths(
                group,
                subgroup,
                previous,
                key_list,
            )
            with h5py.File(self.hdf5path + ".h5", "r") as handle:
                abs_d, scale = self._max_diff_and_scale_hdf5_keyed(
                    handle,
                    paths_current,
                    paths_previous,
                )
            rel_d = abs_d / scale
        else:
            abs_d = float("nan")
            rel_d = float("nan")

        if observational:
            missing = [k for k, v in ((abs_key, tol_abs_raw),
                                       (rel_key, tol_rel_raw)) if v is None]
            self._warn_observational(name, missing)
            tol_abs = None
            tol_rel = None
            passed = None
            effective_gating = False
        else:
            tol_abs = float(tol_abs_raw)
            tol_rel = float(tol_rel_raw)
            passed = bool(ready and (abs_d <= tol_abs) and (rel_d <= tol_rel))
            effective_gating = gating

        self._iter_buf.append({
            "name": name,
            "kind_test": "self",
            "kind_value": "hdf5",
            "abs": abs_d,
            "rel": rel_d,
            "tol_abs": tol_abs,
            "tol_rel": tol_rel,
            "passed": passed,
            "gating": effective_gating,
        })

    def CheckCross(self, name_a, *, a, name_b, b, kind, gating=True):
        """Register a cross-test (same iter, different quantities).

        Tolerance: control['run']['tol_<name_a>_<name_b>_abs'].
        If missing, the test is registered with gating=False,
        passed=None — observational only. No min_iter gate.
        """
        self._require_not_tb("CheckCross")
        self._require_iter("CheckCross")
        key = f"tol_{name_a}_{name_b}_abs"
        tol_abs_raw = self._tol_run.get(key)
        observational = tol_abs_raw is None

        abs_d, _ = self._max_diff_and_scale(a, b, kind)

        if observational:
            self._warn_observational(f"{name_a}-{name_b}", [key])
            tol_abs = None
            passed = None
            effective_gating = False
        else:
            tol_abs = float(tol_abs_raw)
            passed = bool(abs_d <= tol_abs)
            effective_gating = gating

        self._iter_buf.append({
            "name": f"{name_a}-{name_b}",
            "kind_test": "cross",
            "kind_value": kind,
            "abs": abs_d,
            "rel": None,
            "tol_abs": tol_abs,
            "tol_rel": None,
            "passed": passed,
            "gating": effective_gating,
        })

    def CheckCrossHDF5(
        self,
        name_a,
        *,
        name_b,
        group,
        subgroup_a,
        subgroup_b,
        stem_a,
        stem_b,
        keys,
        gating=True,
    ):
        """Register a cross-test by reading keyed datasets from HDF5."""
        self._require_not_tb("CheckCrossHDF5")
        self._require_iter("CheckCrossHDF5")
        key_list = list(keys)
        key = f"tol_{name_a}_{name_b}_abs"
        tol_abs_raw = self._tol_run.get(key)
        observational = tol_abs_raw is None

        paths_a = self._hdf5_keyed_paths(group, subgroup_a, stem_a, key_list)
        paths_b = self._hdf5_keyed_paths(group, subgroup_b, stem_b, key_list)
        with h5py.File(self.hdf5path + ".h5", "r") as handle:
            abs_d, _ = self._max_diff_and_scale_hdf5_keyed(
                handle,
                paths_a,
                paths_b,
            )

        if observational:
            self._warn_observational(f"{name_a}-{name_b}", [key])
            tol_abs = None
            passed = None
            effective_gating = False
        else:
            tol_abs = float(tol_abs_raw)
            passed = bool(abs_d <= tol_abs)
            effective_gating = gating

        self._iter_buf.append({
            "name": f"{name_a}-{name_b}",
            "kind_test": "cross",
            "kind_value": "hdf5",
            "abs": abs_d,
            "rel": None,
            "tol_abs": tol_abs,
            "tol_rel": None,
            "passed": passed,
            "gating": effective_gating,
        })

    def RecordDiagnostics(self, diag_by_key=None):
        """Stash per-key impurity diagnostics for Commit() to aggregate
        into the convergence row (n_imp, sign, histo_1, histo_2)."""
        self._require_not_tb("RecordDiagnostics")
        self._require_iter("RecordDiagnostics")
        self._iter_diag = diag_by_key or {}

    def _agg_diag(self, name, fn=np.mean):
        vals = [d[name] for d in self._iter_diag.values()
                if d.get(name) is not None]
        return float(fn(vals)) if vals else float("nan")

    def Commit(self, iter, *, will_continue):
        """Finalize the iter: compute gating verdict, write the log row,
        promote staged prev values when continuing, return (converged, info).

        Zero-gating safety: if no gating=True records exist this iter,
        emit a one-time warning and force converged=False (avoids
        trivially-True `all([])`).
        """
        self._require_not_tb("Commit")
        self._require_iter("Commit")
        if int(iter) != self._iter_no:
            raise RuntimeError(
                f"Convergence: Commit(iter={iter}) does not match "
                f"StartIter(iter={self._iter_no})"
            )

        gating_records = [r for r in self._iter_buf if r["gating"]]
        if not gating_records:
            if not self._zero_gating_warned:
                logger.warning(
                    f"Convergence: no gating tests registered "
                    f"(iter={self._iter_no}); reporting converged=False "
                    f"to avoid trivially-True all([]). Add at least one "
                    f"matched tol_d<Name>_abs+_rel pair (or "
                    f"tol_<A>_<B>_abs) in control['run'] to enable gating."
                )
                self._zero_gating_warned = True
            converged = False
        else:
            converged = all(r["passed"] for r in gating_records)

        row = {"iter": self._iter_no}
        info_self = {}
        info_cross = {}
        mu_value = None
        for r in self._iter_buf:
            if r["kind_test"] == "self":
                row[f"d{r['name']}_abs"] = r["abs"]
                row[f"d{r['name']}_rel"] = r["rel"]
                info_self[r["name"]] = {
                    "abs": r["abs"], "rel": r["rel"], "passed": r["passed"],
                    "tol_abs": r["tol_abs"], "tol_rel": r["tol_rel"],
                    "gating": r["gating"],
                }
                if r["name"] == "mu":
                    staged = self._pending_prev.get("mu")
                    if staged is not None:
                        mu_value = float(staged[0])
            else:
                row[f"{r['name']}_abs"] = r["abs"]
                info_cross[r["name"]] = {
                    "abs": r["abs"], "passed": r["passed"],
                    "tol_abs": r["tol_abs"], "gating": r["gating"],
                }
        if self._iter_diag:
            row["n_imp"] = self._agg_diag("nimp", np.sum)
            row["ctqmc_sign"] = self._agg_diag("sign", np.min)
            row["histo_1"] = self._agg_diag("histo_m1")
            row["histo_2"] = self._agg_diag("histo_m2")
            guard_aggs = {
                "sigimp_guard_used_fallback": np.max,
                "sigimp_guard_failed": np.max,
                "sigimp_raw_roughness": np.max,
                "sigimp_err_mean": np.mean,
                "sigimp_err_max": np.max,
                "sigimp_max_positive_imag": np.max,
            }
            for name, fn in guard_aggs.items():
                vals = []
                for diag in self._iter_diag.values():
                    value = diag.get(name)
                    if value is None:
                        continue
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        vals.append(value)
                if vals:
                    row[name] = float(fn(vals))
        if mu_value is not None:
            row["mu"] = mu_value

        self._write_convergence_log(row)
        deltas = ", ".join(
            f"{k}={v:.3e}"
            for k, v in row.items()
            if k != "iter" and isinstance(v, (int, float, np.floating))
        )
        logger.info(f"[conv][iter {row['iter']}] {deltas}")

        if will_continue and not converged:
            for name, (val, kind) in self._pending_prev.items():
                self._self_prev[name] = val

        info = {
            "self": info_self,
            "cross": info_cross,
            "converged": converged,
        }

        self._iter_no = None
        return converged, info

    def _write_convergence_log(self, row):
        self._conv_table.append(row)

        if self._schema_cols is None:
            self._schema_cols = list(row.keys())
        else:
            extra = [k for k in row.keys() if k not in self._schema_cols]
            if extra and not self._schema_warned:
                logger.warning(
                    f"convergence.log: ignoring unexpected column(s) {extra} "
                    f"not in locked schema {self._schema_cols}. "
                    f"(Subsequent occurrences silently dropped from text log; "
                    f"JSONL still receives the full row.)"
                )
                self._schema_warned = True

        cols = self._schema_cols
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
                grp = h.require_group(self._conv_hdf5_group)
                if "table" in grp:
                    del grp["table"]
                grp.create_dataset("table", data=np.array(lines, dtype="S256"))
        except Exception as e:
            logger.warning(f"convergence.log HDF5 mirror failed: {e}")

    @staticmethod
    def _json_sanitize(v):
        if v is None:
            return None
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
