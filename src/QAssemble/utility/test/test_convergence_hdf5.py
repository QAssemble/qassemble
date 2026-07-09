import h5py
import numpy as np
import pytest

from QAssemble.utility.Convergence import Convergence


def _conv(tmp_path, *, method="dmft"):
    return Convergence({
        "run": {
            "method": method,
            "fn": str(tmp_path / "calc"),
            "tol_dGLoc_abs": 10.0,
            "tol_dGLoc_rel": 10.0,
            "tol_dWLoc_abs": 10.0,
            "tol_dWLoc_rel": 10.0,
            "tol_GLoc_GImp_abs": 10.0,
        }
    })


def _write(handle, path, value):
    handle.create_dataset(path, data=np.asarray(value, dtype=np.complex128))


def test_hdf5_self_and_cross_checks_use_keyed_max_diff(tmp_path):
    with h5py.File(tmp_path / "calc.h5", "w") as handle:
        _write(handle, "dmft/GLoc/gloc.0.1", [0.0, 0.0])
        _write(handle, "dmft/GLoc/gloc.1.1", [2.0, 0.0])
        _write(handle, "dmft/GLoc/gloc.0.2", [0.0, 0.0])
        _write(handle, "dmft/GLoc/gloc.1.2", [0.0, 3.0])
        _write(handle, "dmft/GImp/gimp.1.1", [1.0, 0.0])
        _write(handle, "dmft/GImp/gimp.1.2", [0.0, 7.0])

    conv = _conv(tmp_path)
    conv.Start()
    conv.StartIter(1)
    conv.CheckSelfHDF5(
        "GLoc",
        group="dmft",
        subgroup="GLoc",
        current="gloc.1",
        previous="gloc.0",
        keys=["1", "2"],
    )
    conv.CheckCrossHDF5(
        "GLoc",
        name_b="GImp",
        group="dmft",
        subgroup_a="GLoc",
        subgroup_b="GImp",
        stem_a="gloc.1",
        stem_b="gimp.1",
        keys=["1", "2"],
    )
    converged, info = conv.Commit(1, will_continue=False)

    assert converged is True
    assert info["self"]["GLoc"]["abs"] == pytest.approx(3.0)
    assert info["cross"]["GLoc-GImp"]["abs"] == pytest.approx(4.0)


def test_hdf5_check_reports_missing_dataset_path(tmp_path):
    with h5py.File(tmp_path / "calc.h5", "w") as handle:
        _write(handle, "dmft/GLoc/gloc.1.1", [1.0])

    conv = _conv(tmp_path)
    conv.Start()
    conv.StartIter(1)

    with pytest.raises(KeyError, match="dmft/GLoc/gloc.0.1"):
        conv.CheckSelfHDF5(
            "GLoc",
            group="dmft",
            subgroup="GLoc",
            current="gloc.1",
            previous="gloc.0",
            keys=["1"],
        )


def test_hdf5_self_check_supports_edmft_wloc_paths(tmp_path):
    with h5py.File(tmp_path / "calc.h5", "w") as handle:
        _write(handle, "edmft/WLoc/wloc.0.1", [1.0, 0.0])
        _write(handle, "edmft/WLoc/wloc.1.1", [4.0, 0.0])
        _write(handle, "edmft/WLoc/wloc.0.2", [0.0, 2.0])
        _write(handle, "edmft/WLoc/wloc.1.2", [0.0, 6.0])

    conv = _conv(tmp_path, method="edmft")
    conv.Start()
    conv.StartIter(1)
    conv.CheckSelfHDF5(
        "WLoc",
        group="edmft",
        subgroup="WLoc",
        current="wloc.1",
        previous="wloc.0",
        keys=["1", "2"],
    )
    converged, info = conv.Commit(1, will_continue=False)

    assert converged is True
    assert info["self"]["WLoc"]["abs"] == pytest.approx(4.0)
