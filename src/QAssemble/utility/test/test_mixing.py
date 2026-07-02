import h5py
import numpy as np
import pytest

from QAssemble.utility.Mixing import Mixing


def _mixer(path, *, method="linear", npulay=5):
    return Mixing(
        method=method,
        npulay=npulay,
        hdf5file=str(path),
        group="calc",
    )


def test_mixing_rejects_calls_without_quantities(tmp_path):
    mixer = _mixer(tmp_path / "mix.h5")

    with pytest.raises(ValueError, match="quantities"):
        mixer(iter=1, mix=0.5, key="global")


def test_hdf5_linear_mixing_restarts_from_last(tmp_path):
    path = tmp_path / "mix.h5"

    mixed = _mixer(path, method="linear")(
        iter=1,
        mix=0.25,
        key="global",
        quantities={
            "a": np.asarray([10.0]),
            "b": np.asarray([20.0, 30.0]),
        },
    )
    np.testing.assert_allclose(mixed["a"], [10.0])
    np.testing.assert_allclose(mixed["b"], [20.0, 30.0])

    mixed = _mixer(path, method="linear")(
        iter=2,
        mix=0.25,
        key="global",
        quantities={
            "a": np.asarray([14.0]),
            "b": np.asarray([24.0, 38.0]),
        },
    )
    np.testing.assert_allclose(mixed["a"], [11.0])
    np.testing.assert_allclose(mixed["b"], [21.0, 32.0])

    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(
            handle["calc/Mixing/global/a/last"][()],
            [11.0],
        )
        np.testing.assert_allclose(
            handle["calc/Mixing/global/b/last"][()],
            [21.0, 32.0],
        )


def test_hdf5_pulay_uses_persisted_history(tmp_path):
    path = tmp_path / "mix.h5"

    _mixer(path, method="pulay", npulay=2)(
        iter=1,
        mix=0.5,
        key="global",
        quantities={"x": np.asarray([0.0, 0.0])},
    )
    _mixer(path, method="pulay", npulay=2)(
        iter=2,
        mix=0.5,
        key="global",
        quantities={"x": np.asarray([1.0, 0.0])},
    )
    mixed = _mixer(path, method="pulay", npulay=2)(
        iter=3,
        mix=0.5,
        key="global",
        quantities={"x": np.asarray([0.5, 1.0])},
    )

    np.testing.assert_allclose(mixed["x"], [0.5, 0.25])

    with h5py.File(path, "r") as handle:
        comp = handle["calc/Mixing/global/x"]
        assert comp.attrs["num_history"] == 2
        assert comp.attrs["next_slot"] == 0
        assert "0" in comp["input_history"]
        assert "1" in comp["residual_history"]


def test_hdf5_pulay_singular_overlap_falls_back_to_linear(tmp_path):
    path = tmp_path / "mix.h5"

    _mixer(path, method="pulay", npulay=3)(
        iter=1,
        mix=0.5,
        key="global",
        quantities={"x": np.asarray([0.0, 0.0])},
    )
    _mixer(path, method="pulay", npulay=3)(
        iter=2,
        mix=0.5,
        key="global",
        quantities={"x": np.asarray([1.0, 0.0])},
    )
    mixed = _mixer(path, method="pulay", npulay=3)(
        iter=3,
        mix=0.5,
        key="global",
        quantities={"x": np.asarray([1.5, 0.0])},
    )

    np.testing.assert_allclose(mixed["x"], [1.0, 0.0])


def test_iter_one_resets_existing_component_history(tmp_path):
    path = tmp_path / "mix.h5"
    mixer = _mixer(path, method="pulay", npulay=2)

    mixer(iter=1, mix=0.5, key="global", quantities={"x": np.asarray([0.0])})
    mixer(iter=2, mix=0.5, key="global", quantities={"x": np.asarray([1.0])})
    mixer(iter=1, mix=0.5, key="global", quantities={"x": np.asarray([7.0])})

    with h5py.File(path, "r") as handle:
        comp = handle["calc/Mixing/global/x"]
        assert comp.attrs["num_history"] == 0
        np.testing.assert_allclose(comp["last"][()], [7.0])
        assert list(comp["input_history"].keys()) == []
