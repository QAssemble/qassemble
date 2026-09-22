import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from QAssemble.CorrelationFunction import CorrelationFunction
from QAssemble.FLatDyn import G
from QAssemble.Run import Run
from QAssemble.utility.Convergence import Convergence
from QAssemble.utility.HDF5 import IO
from QAssemble.utility.Mixing import Mixing


def _complete(handle, iteration, keys):
    paths = [f"gwedmft/G/gkf.{iteration}", f"gwedmft/G/mu.{iteration}",
             f"gwedmft/W/wkf.{iteration}"]
    for key in keys:
        paths += [f"gwedmft/{group}/{stem}.{iteration}.{key}" for group, stem in (
            ("GLoc", "gloc"), ("WLoc", "wloc"), ("GImp", "gimp"),
            ("SigHImp", "sighimp"), ("SigFImp", "sigfimp"),
            ("SigCImp", "sigimp"), ("Chi", "chi"), ("PImp", "pimp"),
            ("WImp", "wimp"),
        )]
    for path in paths:
        parent, name = path.rsplit("/", 1)
        handle.require_group(parent).create_dataset(name, data=1)


def test_last_complete_iteration_requires_every_key_and_contiguous_history(tmp_path):
    path = tmp_path / "glob.h5"
    assert IO.LastCompleteIteration(path, "gwedmft", ["a"], 4) == 0
    with h5py.File(path, "w") as handle:
        handle.create_group("gwedmft")
    assert IO.LastCompleteIteration(path, "gwedmft", ["a"], 4) == 0
    with h5py.File(path, "a") as handle:
        for iteration in (1, 2):
            _complete(handle, iteration, ["a", "b"])
        _complete(handle, 3, ["a"])
        _complete(handle, 4, ["a", "b"])
    assert IO.LastCompleteIteration(path, "gwedmft", ["a", "b"], 4) == 2
    assert IO.LastCompleteIteration(path, "gwedmft", ["a"], 4) == 4
    with h5py.File(path, "a") as handle:
        del handle["gwedmft/GImp/gimp.2.b"]
    assert IO.LastCompleteIteration(path, "gwedmft", ["a", "b"], 4) == 1


def test_torn_ctqmc_output_does_not_count(tmp_path):
    path = tmp_path / "glob.h5"
    with h5py.File(path, "w") as handle:
        _complete(handle, 1, ["a"])
        _complete(handle, 2, ["a"])
        del handle["gwedmft/GImp/gimp.2.a"]
    assert IO.LastCompleteIteration(path, "gwedmft", ["a"], 2) == 1


def _mix(path, component, iteration, value):
    return IO.MixComponent(
        str(path), "gwedmft", "global", component, np.asarray(value),
        iter=iteration, mix=0.5, method="pulay", npulay=3, mixer=Mixing(),
    )


def test_align_mixing_keeps_current_and_resets_legacy_component(tmp_path, caplog):
    path = tmp_path / "glob.h5"
    for name in ("sigh", "sigf"):
        _mix(path, name, 1, [0.0, 0.0])
        _mix(path, name, 2, [1.0, 0.0])
    _mix(path, "sigf", 3, [2.0, 0.0])
    with h5py.File(path, "a") as handle:
        component = handle["gwedmft/Mixing/global/sigf"]
        for name in list(component):
            if name.startswith("last."):
                del component[name]
    with h5py.File(path, "r") as handle:
        kept_before = handle["gwedmft/Mixing/global/sigh/last"][()].copy()
    actions = IO.AlignMixingHistory(path, "gwedmft", ["a"], 2)
    assert actions["global/sigh"] == "kept"
    assert actions["global/sigf"] == "reset"
    assert "next iteration will pass UNMIXED" in caplog.text
    assert actions["global/pol"] == "absent"
    with h5py.File(path, "r") as handle:
        kept = handle["gwedmft/Mixing/global/sigh"]
        np.testing.assert_allclose(kept["last"][()], kept_before)
        assert int(kept.attrs["num_history"]) == 1
        reset = handle["gwedmft/Mixing/global/sigf"]
        assert "last" not in reset and "last_iter" not in reset.attrs
    new = np.asarray([3.0, 1.0])
    np.testing.assert_allclose(_mix(path, "sigf", 3, new), new)
    assert not np.allclose(_mix(path, "sigh", 3, new), new)


def test_align_mixing_rewinds_torn_component_and_next_iteration_mixes(tmp_path):
    path = tmp_path / "glob.h5"
    _mix(path, "sigf", 1, [0.0, 0.0])
    _mix(path, "sigf", 2, [1.0, 0.0])
    with h5py.File(path, "r") as handle:
        expected = handle["gwedmft/Mixing/global/sigf/last.2"][()].copy()
    _mix(path, "sigf", 3, [2.0, 0.0])

    actions = IO.AlignMixingHistory(path, "gwedmft", ["a"], 2)
    assert actions["global/sigf"] == "rewound"
    with h5py.File(path, "r") as handle:
        component = handle["gwedmft/Mixing/global/sigf"]
        np.testing.assert_array_equal(component["last"][()], expected)
        assert "last.3" not in component
        assert int(component.attrs["last_iter"]) == 2
        assert int(component.attrs["num_history"]) == 0
        assert int(component.attrs["next_slot"]) == 0
        assert list(component["input_history"]) == []
        assert list(component["residual_history"]) == []
    new = np.asarray([3.0, 1.0])
    assert not np.allclose(_mix(path, "sigf", 3, new), new)


def test_projected_last_snapshot_is_the_rewind_value(tmp_path):
    path = tmp_path / "glob.h5"
    _mix(path, "sigf", 1, [0.0, 0.0])
    _mix(path, "sigf", 2, [1.0, 0.0])
    projected = np.asarray([0.25, 0.75])
    IO.OverwriteMixingLast(path, "gwedmft", "global", "sigf", projected)
    _mix(path, "sigf", 3, [2.0, 0.0])
    assert IO.AlignMixingHistory(path, "gwedmft", ["a"], 2)["global/sigf"] == "rewound"
    with h5py.File(path, "r") as handle:
        component = handle["gwedmft/Mixing/global/sigf"]
        np.testing.assert_array_equal(component["last.2"][()], projected)
        np.testing.assert_array_equal(component["last"][()], projected)


def test_snapshot_pruning_keeps_at_least_two_iterations(tmp_path):
    path = tmp_path / "glob.h5"
    for iteration in range(1, 5):
        IO.MixComponent(
            str(path), "gwedmft", "global", "sigh", np.asarray([iteration]),
            iter=iteration, mix=0.5, method="linear", npulay=1, mixer=Mixing(),
        )
    with h5py.File(path, "r") as handle:
        component = handle["gwedmft/Mixing/global/sigh"]
        assert sorted(name for name in component if name.startswith("last.")) == ["last.3", "last.4"]


def test_rewound_linear_mixing_matches_uninterrupted_run(tmp_path):
    def mix(path, iteration, value):
        return IO.MixComponent(
            str(path), "gwedmft", "global", "sigh", np.asarray(value),
            iter=iteration, mix=0.1, method="linear", npulay=2, mixer=Mixing(),
        )

    clean = tmp_path / "clean.h5"
    resumed = tmp_path / "resumed.h5"
    for path in (clean, resumed):
        mix(path, 1, [0.0, 0.0])
        mix(path, 2, [1.0, 0.0])
    expected = mix(clean, 3, [3.0, 1.0])
    mix(resumed, 3, [2.0, 0.0])
    assert IO.AlignMixingHistory(resumed, "gwedmft", ["a"], 2)["global/sigh"] == "rewound"
    np.testing.assert_array_equal(mix(resumed, 3, [3.0, 1.0]), expected)


def test_rewind_when_last_changed_before_last_iter_was_updated(tmp_path):
    path = tmp_path / "glob.h5"
    _mix(path, "sigh", 1, [0.0, 0.0])
    _mix(path, "sigh", 2, [1.0, 0.0])
    with h5py.File(path, "a") as handle:
        component = handle["gwedmft/Mixing/global/sigh"]
        expected = component["last.2"][()].copy()
        IO.CreateDataset(component, "last", np.asarray([9.0, 9.0]))
    assert IO.AlignMixingHistory(path, "gwedmft", ["a"], 2)["global/sigh"] == "rewound"
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["gwedmft/Mixing/global/sigh/last"][()], expected)


def test_convergence_resume_preserves_rows_and_next_commit(tmp_path):
    prefix = tmp_path / "glob"
    jsonl = tmp_path / "convergence.jsonl"
    rows = [{"iter": 1, "converged": False}, {"iter": 2, "converged": False},
            {"iter": 3, "converged": False}]
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows))
    conv = Convergence({"run": {"method": "gw+edmft", "fn": str(prefix)}})
    conv.Resume(2)
    assert len(conv._conv_table) == 2
    assert conv._schema_cols == list(rows[0])
    assert len(jsonl.read_text().splitlines()) == 3
    conv.StartIter(3)
    conv.Commit(3, will_continue=True)
    log = (tmp_path / "convergence.log").read_text()
    assert len(conv._conv_table) == 3
    assert all(str(i) in log for i in (1, 2, 3))


def test_run_input_allows_new_mode(tmp_path):
    run = Run.__new__(Run)
    assert run.CheckInput({"Control": {"Method": "gw+edmft"}},
                          {"Control": {"Method": "gw+edmft", "Mode": "Auto"}})


def test_restore_green_reconstructs_saved_kf_and_derived_state(tmp_path):
    path = tmp_path / "glob.h5"
    saved = np.asarray([[[[[0.3 - 0.4j]]]]], order="F")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("gwedmft/G/gkf.2", data=saved)
        handle.create_dataset("gwedmft/G/mu.2", data=0.2)
    corr = CorrelationFunction.__new__(CorrelationFunction)
    corr.hdf5path = str(path)
    corr.greenbare = SimpleNamespace(kf=np.empty_like(saved))
    green = object.__new__(G)
    green.crystal = SimpleNamespace(find=[0], ns=1, kpoint=[0])
    green.dlr = SimpleNamespace(omega=[1])
    green.F2T = lambda value: value.copy()
    green.K2R = lambda value: value.copy()
    green.Occ = lambda: setattr(green, "occ", green.kf.copy())
    corr._restore_green(green, "gwedmft", 2)
    np.testing.assert_allclose(green.kf, saved, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(green.kt, saved)
    np.testing.assert_allclose(green.occ, saved)

    corr.greenbare.kf = np.empty((2, 2, 1, 1, 1))
    with pytest.raises(ValueError, match="gwedmft/G/gkf.2"):
        corr._restore_green(green, "gwedmft", 2)


def test_restore_wlat_rebuilds_connected_real_time_part(tmp_path):
    path = tmp_path / "glob.h5"
    saved = np.asarray([[[[[[3.0 + 2.0j]]]]]])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("gwedmft/W/wkf.2", data=saved)
    corr = CorrelationFunction.__new__(CorrelationFunction)
    corr.hdf5path = str(path)
    corr.vbare = SimpleNamespace(k=np.asarray([1.0]))
    wlat = SimpleNamespace(
        kf=np.zeros_like(saved),
        StcEmbedding=lambda bare: np.ones_like(saved) * bare[0],
        F2T=lambda value: value * 2,
        K2R=lambda value: value * 3,
    )
    corr._restore_wlat(wlat, "gwedmft", 2)
    np.testing.assert_allclose(wlat.kf, saved)
    np.testing.assert_allclose(wlat.ckf, saved - 1)
    np.testing.assert_allclose(wlat.crt, (saved - 1) * 6)
