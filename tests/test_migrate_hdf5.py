from pathlib import Path

import h5py
import numpy as np
import pytest

from QAssemble.migrate_hdf5 import MigrationError, migrate_file


def _legacy_file(path: Path) -> None:
    with h5py.File(path, "w") as h5file:
        input_group = h5file.create_group("input")
        input_hamiltonian = input_group.create_group("Hamiltonian")
        input_hamiltonian.create_dataset("setting", data=np.array([7]))

        gw = h5file.create_group("gw")
        green = gw.create_group("GreenInt")
        green.attrs["units"] = "eV"
        green.create_dataset(
            "gkf", data=np.arange(12).reshape(3, 4), compression="gzip"
        )
        gw.create_group("SigmaHartree").create_dataset("sigmah", data=np.eye(2))

        hf = h5file.create_group("hf")
        hf.create_group("Hamiltonian").create_dataset("hk", data=np.eye(2))


def test_dry_run_does_not_write(tmp_path):
    source = tmp_path / "result.h5"
    _legacy_file(source)

    targets = migrate_file(source, dry_run=True)

    assert ("/gw", "GreenInt", "G") in targets
    assert not Path(f"{source}.pre-class-rename.bak").exists()
    with h5py.File(source, "r") as h5file:
        assert "GreenInt" in h5file["gw"]
        assert "Hamiltonian" in h5file["input"]


def test_in_place_migration_preserves_data_and_input(tmp_path):
    source = tmp_path / "result.h5"
    _legacy_file(source)

    migrate_file(source)

    backup = Path(f"{source}.pre-class-rename.bak")
    assert backup.is_file()
    with h5py.File(backup, "r") as h5file:
        assert "GreenInt" in h5file["gw"]
        assert "Hamiltonian" in h5file["hf"]
    with h5py.File(source, "r") as h5file:
        assert "G" in h5file["gw"] and "GreenInt" not in h5file["gw"]
        assert "SigH" in h5file["gw"] and "SigmaHartree" not in h5file["gw"]
        assert "H" in h5file["hf"] and "Hamiltonian" not in h5file["hf"]
        assert "Hamiltonian" in h5file["input"]
        np.testing.assert_array_equal(h5file["gw/G/gkf"][:], np.arange(12).reshape(3, 4))
        assert h5file["gw/G/gkf"].compression == "gzip"
        assert h5file["gw/G"].attrs["units"] == "eV"


def test_collision_aborts_without_backup(tmp_path):
    source = tmp_path / "collision.h5"
    with h5py.File(source, "w") as h5file:
        gw = h5file.create_group("gw")
        gw.create_group("GreenInt")
        gw.create_group("G")

    with pytest.raises(MigrationError, match="collision"):
        migrate_file(source)

    assert not Path(f"{source}.pre-class-rename.bak").exists()
    with h5py.File(source, "r") as h5file:
        assert "GreenInt" in h5file["gw"] and "G" in h5file["gw"]
