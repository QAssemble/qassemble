from pathlib import Path

import pytest

import QAssemble


PAPER_CLASSES = {
    "G0": QAssemble.FLatDyn,
    "G": QAssemble.FLatDyn,
    "SigGWC": QAssemble.FLatDyn,
    "H0": QAssemble.FLatStc,
    "H": QAssemble.FLatStc,
    "SigH": QAssemble.FLatStc,
    "SigF": QAssemble.FLatStc,
    "P": QAssemble.BLatDyn,
    "W": QAssemble.BLatDyn,
    "V": QAssemble.BLatStc,
}

LEGACY_NAMES = {
    "NIHamiltonian",
    "Hamiltonian",
    "SigmaHartree",
    "SigmaFock",
    "GreenBare",
    "GreenInt",
    "SigmaGWC",
    "PolLat",
    "WLat",
    "VBare",
}


@pytest.mark.parametrize(("name", "base"), PAPER_CLASSES.items())
def test_manuscript_class_contract(name, base):
    cls = getattr(QAssemble, name)
    assert cls.__name__ == name
    assert issubclass(cls, base)
    assert name in QAssemble.__all__


@pytest.mark.parametrize("name", LEGACY_NAMES)
def test_legacy_class_names_are_not_exported(name):
    assert not hasattr(QAssemble, name)
    with pytest.raises(ImportError):
        exec(f"from QAssemble import {name}", {})


def test_source_contains_no_legacy_class_names():
    package_dir = Path(QAssemble.__file__).parent
    forbidden = LEGACY_NAMES - {"Hamiltonian"}
    for path in package_dir.rglob("*.py"):
        if path.name == "migrate_hdf5.py" or path.suffixes[-2:] == [".py", ".bak"]:
            continue
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{name} remains in {path}"
        assert "class Hamiltonian(" not in text
