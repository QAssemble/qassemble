import h5py
import numpy as np
import pytest

from QAssemble.utility.HDF5 import IO
from QAssemble.utility.Mixing import Mixing


def _mix_component(
    path,
    *,
    method="linear",
    npulay=5,
    iter=1,
    mix=0.5,
    key="global",
    component="x",
    value=None,
    group="calc",
):
    return IO.MixComponent(
        hdf5file=str(path),
        group=group,
        key=key,
        component=component,
        value=np.asarray([0.0]) if value is None else value,
        iter=iter,
        mix=mix,
        method=method,
        npulay=npulay,
        mixer=Mixing(),
    )


def test_linear_mixing_formula():
    mixed = Mixing().linear(
        mix=0.25,
        fnew=np.asarray([14.0, 38.0]),
        fold=np.asarray([10.0, 30.0]),
    )

    np.testing.assert_allclose(mixed, [11.0, 32.0])


def test_pulay_mixing_uses_passed_history():
    mixed = Mixing().pulay(
        mix=0.5,
        fnew=np.asarray([0.5, 1.0]),
        fold=np.asarray([0.5, 0.0]),
        inputs=[
            np.asarray([0.0, 0.0]),
            np.asarray([0.5, 0.0]),
        ],
        residuals=[
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
        ],
    )

    np.testing.assert_allclose(mixed, [0.5, 0.25])


def test_pulay_singular_overlap_falls_back_to_linear():
    mixed = Mixing().pulay(
        mix=0.5,
        fnew=np.asarray([1.5, 0.0]),
        fold=np.asarray([0.5, 0.0]),
        inputs=[
            np.asarray([0.0, 0.0]),
            np.asarray([0.5, 0.0]),
        ],
        residuals=[
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
        ],
    )

    np.testing.assert_allclose(mixed, [1.0, 0.0])


def test_mixing_rejects_invalid_inputs():
    mixer = Mixing()

    with pytest.raises(ValueError, match="Unknown mixing method"):
        mixer(method="bad", mix=0.5, fnew=[1.0], fold=[0.0])

    with pytest.raises(ValueError, match="mix must satisfy"):
        mixer(method="linear", mix=0.0, fnew=[1.0], fold=[0.0])

    with pytest.raises(ValueError, match="non-finite"):
        mixer(method="linear", mix=0.5, fnew=[np.nan], fold=[0.0])


def test_hdf5_linear_mixing_restarts_from_last(tmp_path):
    path = tmp_path / "mix.h5"

    mixed = _mix_component(
        path,
        method="linear",
        iter=1,
        mix=0.25,
        component="a",
        value=np.asarray([10.0]),
    )
    np.testing.assert_allclose(mixed, [10.0])

    mixed = _mix_component(
        path,
        method="linear",
        iter=2,
        mix=0.25,
        component="a",
        value=np.asarray([14.0]),
    )
    np.testing.assert_allclose(mixed, [11.0])

    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(
            handle["calc/Mixing/global/a/last"][()],
            [11.0],
        )
        assert list(handle["calc/Mixing/global/a/input_history"].keys()) == []


def test_hdf5_pulay_uses_persisted_history_and_layout(tmp_path):
    path = tmp_path / "mix.h5"

    _mix_component(
        path,
        method="pulay",
        npulay=2,
        iter=1,
        value=np.asarray([0.0, 0.0]),
    )
    _mix_component(
        path,
        method="pulay",
        npulay=2,
        iter=2,
        value=np.asarray([1.0, 0.0]),
    )
    mixed = _mix_component(
        path,
        method="pulay",
        npulay=2,
        iter=3,
        value=np.asarray([0.5, 1.0]),
    )

    np.testing.assert_allclose(mixed, [0.5, 0.25])

    with h5py.File(path, "r") as handle:
        comp = handle["calc/Mixing/global/x"]
        np.testing.assert_allclose(comp["last"][()], [0.5, 0.25])
        assert comp.attrs["method"] == "pulay"
        assert comp.attrs["npulay"] == 2
        assert comp.attrs["num_history"] == 2
        assert comp.attrs["next_slot"] == 0
        assert "0" in comp["input_history"]
        assert "1" in comp["residual_history"]


def test_iter_one_resets_existing_component_history(tmp_path):
    path = tmp_path / "mix.h5"

    _mix_component(path, method="pulay", npulay=2, iter=1, value=np.asarray([0.0]))
    _mix_component(path, method="pulay", npulay=2, iter=2, value=np.asarray([1.0]))
    _mix_component(path, method="pulay", npulay=2, iter=1, value=np.asarray([7.0]))

    with h5py.File(path, "r") as handle:
        comp = handle["calc/Mixing/global/x"]
        assert comp.attrs["num_history"] == 0
        np.testing.assert_allclose(comp["last"][()], [7.0])
        assert list(comp["input_history"].keys()) == []


def test_hdf5_mixing_rejects_shape_mismatch(tmp_path):
    path = tmp_path / "mix.h5"

    _mix_component(path, method="linear", iter=1, value=np.asarray([1.0, 2.0]))

    with pytest.raises(ValueError, match="shape mismatch"):
        _mix_component(
            path,
            method="linear",
            iter=2,
            value=np.asarray([[1.0], [2.0]]),
        )


def test_absolute_hdf5file_mixes_across_per_iteration_workdirs(monkeypatch, tmp_path):
    """Mixing must survive the per-iteration cwd change CTQMC performs.

    CTQMC.PostProcessing chdir's into ctqmc/impurity_<iter>_<key> before calling
    .Mixing(). With the bare relative 'glob.h5' this used to resolve to a fresh
    file inside that directory, so every call hit the reset branch and returned
    its input unmixed. An absolute path keeps one history across iterations.
    """
    path = str(tmp_path / "glob.h5")
    monkeypatch.chdir(tmp_path)
    _mix_component(
        path,
        method="linear",
        iter=1,
        mix=0.25,
        key="1",
        component="sighimp",
        value=np.asarray([10.0]),
        group="edmft",
    )

    workdir = tmp_path / "ctqmc" / "impurity_2_1"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)

    mixed = _mix_component(
        path,
        method="linear",
        iter=2,
        mix=0.25,
        key="1",
        component="sighimp",
        value=np.asarray([14.0]),
        group="edmft",
    )

    # 0.25*14 + 0.75*10 == 11.0; an unmixed pass-through would return 14.0.
    np.testing.assert_allclose(mixed, [11.0])
    # The scratch directory must stay free of a stray per-iteration glob.h5.
    assert not (workdir / "glob.h5").exists()


def test_relative_hdf5file_in_scratch_dir_is_rejected(monkeypatch, tmp_path):
    """A relative hdf5file under a fresh cwd must fail loudly, not mix silently."""
    monkeypatch.chdir(tmp_path)
    _mix_component(
        "glob.h5",
        method="linear",
        iter=1,
        value=np.asarray([10.0]),
        group="edmft",
    )

    workdir = tmp_path / "ctqmc" / "impurity_2_1"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)

    with pytest.raises(ValueError, match="does not exist"):
        _mix_component(
            "glob.h5",
            method="linear",
            iter=2,
            value=np.asarray([14.0]),
            group="edmft",
        )


def test_mix_component_rejects_missing_file_after_first_iteration(tmp_path):
    """iter > 1 with no existing file means the history was lost — fail loudly."""
    with pytest.raises(ValueError, match="does not exist"):
        _mix_component(tmp_path / "absent.h5", method="linear", iter=2)

    # iter == 1 legitimately creates the file.
    _mix_component(tmp_path / "absent.h5", method="linear", iter=1)
    assert (tmp_path / "absent.h5").exists()
