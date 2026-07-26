"""The driver's HDF5 path must be absolute.

Downstream impurity objects mix and save while CTQMC.PostProcessing has
os.chdir'd into ctqmc/impurity_<iter>_<key>.  A relative 'glob.h5' resolves to a
new file inside each of those scratch directories, so IO.MixComponent never
finds a stored history and silently returns its input unmixed on every
iteration.  Resolving the path once, up front, is what keeps mixing alive.
"""

import os

from QAssemble.CorrelationFunction import CorrelationFunction


def _hdf5path(control):
    """Run only the hdf5path line of __init__, skipping the heavy crystal setup."""
    obj = CorrelationFunction.__new__(CorrelationFunction)
    obj.control = control
    obj.hdf5path = os.path.abspath(control["run"]["fn"] + ".h5")
    return obj.hdf5path


def test_hdf5path_is_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = _hdf5path({"run": {"fn": "glob"}})

    assert os.path.isabs(path)
    assert path == str(tmp_path / "glob.h5")


def test_hdf5path_is_stable_across_chdir(monkeypatch, tmp_path):
    """The resolved path must not follow a later cwd change."""
    monkeypatch.chdir(tmp_path)
    path = _hdf5path({"run": {"fn": "glob"}})

    workdir = tmp_path / "ctqmc" / "impurity_2_1"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)

    assert path == str(tmp_path / "glob.h5")
    assert os.path.abspath("glob.h5") != path


def test_driver_source_resolves_hdf5path_absolutely():
    """Guard the actual __init__ line, not just the helper above."""
    import inspect

    src = inspect.getsource(CorrelationFunction.__init__)
    assignments = [
        line.strip()
        for line in src.splitlines()
        if "self.hdf5path" in line and "=" in line
    ]

    assert assignments, "CorrelationFunction.__init__ must define self.hdf5path"
    assert any("abspath" in line for line in assignments), (
        "self.hdf5path must be wrapped in os.path.abspath so mixing survives "
        f"the CTQMC chdir; found: {assignments}"
    )


def test_driver_has_no_relative_hdf5file_construction():
    """No site may rebuild the path from control['run']['fn'] without abspath."""
    import inspect

    src = inspect.getsource(inspect.getmodule(CorrelationFunction))
    offenders = [
        line.strip()
        for line in src.splitlines()
        if '["fn"]' in line and ".h5" in line and "abspath" not in line
    ]

    assert not offenders, (
        f"these lines rebuild a relative HDF5 path; use self.hdf5path: {offenders}"
    )
