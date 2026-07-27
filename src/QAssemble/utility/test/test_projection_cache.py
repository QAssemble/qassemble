from types import SimpleNamespace

import h5py
import numpy as np

from QAssemble.BLocDyn import BLocDyn
from QAssemble.utility.HDF5 import IO
from QAssemble.utility.Mixing import Mixing as MixingKernel


def test_read_projection_cache_missing_file_returns_none(tmp_path):
    assert (
        IO.ReadProjectionCache(str(tmp_path / "absent.h5"), "calc", "BWeiss", "x")
        is None
    )


def test_read_projection_cache_none_target_returns_none():
    assert IO.ReadProjectionCache(None, "calc", "BWeiss", "x") is None
    assert IO.ReadProjectionCache("f.h5", None, "BWeiss", "x") is None


def test_projection_cache_roundtrip(tmp_path):
    path = str(tmp_path / "cache.h5")
    value = np.arange(6, dtype=np.complex128).reshape(2, 3) + 1j

    IO.WriteProjectionCache(path, "calc", "BWeiss", "bweiss_brd_prev.1", value)
    out = IO.ReadProjectionCache(
        path, "calc", "BWeiss", "bweiss_brd_prev.1", expected_shape=(2, 3)
    )

    np.testing.assert_allclose(out, value)
    assert np.isfortran(out)

    # Overwrite replaces in place.
    IO.WriteProjectionCache(path, "calc", "BWeiss", "bweiss_brd_prev.1", 2 * value)
    out2 = IO.ReadProjectionCache(path, "calc", "BWeiss", "bweiss_brd_prev.1")
    np.testing.assert_allclose(out2, 2 * value)


def test_read_projection_cache_shape_mismatch_returns_none(tmp_path):
    path = str(tmp_path / "cache.h5")
    IO.WriteProjectionCache(path, "calc", "BWeiss", "x", np.zeros((2, 3)))

    assert (
        IO.ReadProjectionCache(path, "calc", "BWeiss", "x", expected_shape=(2, 4))
        is None
    )


def test_write_projection_cache_none_target_is_noop(tmp_path):
    IO.WriteProjectionCache(None, "calc", "BWeiss", "x", np.zeros(2))
    IO.WriteProjectionCache(str(tmp_path / "c.h5"), None, "BWeiss", "x", np.zeros(2))
    assert not (tmp_path / "c.h5").exists()


def test_overwrite_mixing_last_replaces_only_last(tmp_path):
    path = str(tmp_path / "mix.h5")
    mixer = MixingKernel()
    common = dict(
        hdf5file=path, group="calc", key="1", component="pimp",
        mix=0.5, method="pulay", npulay=2, mixer=mixer,
    )
    IO.MixComponent(value=np.asarray([1.0], dtype=np.complex128), iter=1, **common)
    IO.MixComponent(value=np.asarray([3.0], dtype=np.complex128), iter=2, **common)

    with h5py.File(path, "r") as handle:
        comp = handle["calc/Mixing/1/pimp"]
        history_before = {
            f"{kind}/{slot}": np.asarray(comp[kind][slot][()])
            for kind in ("input_history", "residual_history")
            for slot in comp[kind]
        }
        attrs_before = dict(comp.attrs)

    IO.OverwriteMixingLast(path, "calc", "1", "pimp", np.asarray([99.0]))

    with h5py.File(path, "r") as handle:
        comp = handle["calc/Mixing/1/pimp"]
        np.testing.assert_allclose(comp["last"][()], [99.0])
        for name, before in history_before.items():
            np.testing.assert_array_equal(comp[name][()], before)
        for attr, before in attrs_before.items():
            np.testing.assert_array_equal(comp.attrs[attr], before)


def test_overwrite_mixing_last_none_target_is_noop(tmp_path):
    IO.OverwriteMixingLast(None, "calc", "1", "pimp", np.asarray([1.0]))
    path = tmp_path / "m.h5"
    IO.OverwriteMixingLast(str(path), "calc", None, "pimp", np.asarray([1.0]))
    assert not path.exists()


def _bloc_with_cache(path):
    obj = BLocDyn(
        crystal=SimpleNamespace(ns=1),
        dlr=SimpleNamespace(nu=np.arange(3, dtype=float)),
        projector=None,
    )
    obj.hdf5file = path
    obj.group = "calc"
    obj.subgroup = "BWeiss"
    obj.key = "1"
    return obj


def test_brd_prev_roundtrip_through_blocdyn(tmp_path):
    path = str(tmp_path / "glob.h5")
    obj = _bloc_with_cache(path)
    value = np.arange(3, dtype=np.complex128)

    assert obj.ReadBrdPrev("bweiss", (3,)) is None
    obj.WriteBrdPrev("bweiss", value)
    np.testing.assert_allclose(obj.ReadBrdPrev("bweiss", (3,)), value)
    assert obj.ReadBrdPrev("bweiss", (4,)) is None

    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(
            handle["calc/BWeiss/bweiss_brd_prev.1"][()], value
        )


def test_brd_prev_noop_without_cache_target(tmp_path):
    obj = BLocDyn(
        crystal=SimpleNamespace(ns=1),
        dlr=SimpleNamespace(nu=np.arange(3, dtype=float)),
        projector=None,
    )
    # No hdf5file/group/key configured (the common test-fixture situation).
    assert obj.ReadBrdPrev("bweiss", (3,)) is None
    obj.WriteBrdPrev("bweiss", np.zeros(3))
