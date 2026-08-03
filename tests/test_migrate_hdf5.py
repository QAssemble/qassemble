import h5py

from QAssemble.migrate_hdf5 import migrate_file


def test_migrate_file_dry_run_reports_legacy_groups(tmp_path):
    h5_path = tmp_path / "legacy.h5"
    with h5py.File(h5_path, "w") as h5file:
        group = h5file.create_group("gw")
        group.create_group("SigmaGWC")
        group.create_group("GreenInt")

    targets = migrate_file(h5_path, dry_run=True)

    assert ("/gw", "SigmaGWC", "SigGWC") in targets
    assert ("/gw", "GreenInt", "G") in targets


def test_migrate_file_renames_legacy_groups(tmp_path):
    h5_path = tmp_path / "legacy.h5"
    with h5py.File(h5_path, "w") as h5file:
        group = h5file.create_group("hf")
        group.create_group("SigmaHartree")
        group.create_group("SigmaFock")

    targets = migrate_file(h5_path)

    assert ("/hf", "SigmaHartree", "SigH") in targets
    assert ("/hf", "SigmaFock", "SigF") in targets
    with h5py.File(h5_path, "r") as h5file:
        assert "SigmaHartree" not in h5file["hf"]
        assert "SigmaFock" not in h5file["hf"]
        assert "SigH" in h5file["hf"]
        assert "SigF" in h5file["hf"]
    assert h5_path.with_name("legacy.h5.pre-class-rename.bak").exists()
