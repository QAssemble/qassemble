from contextlib import nullcontext

import pytest

from QAssemble.CLI import main
from QAssemble.Run import InputFormatError, Run, load_input_file, resolve_input_file


def _input_sections(prefix):
    return {
        "Crystal": {
            "RVec": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "Basis": [[[0.0, 0.0, 0.0], 1], [[0.5, 0.0, 0.0], 1]],
            "CorF": "F",
            "NSpin": 1,
            "SOC": False,
            "NElec": 2,
            "KGrid": [2, 2, 1],
        },
        "Hamiltonian": {
            "OneBody": {
                "Hopping": {
                    ((0, 0), (1, 0)): {
                        1.0: [[0, 0, 0], [-1, 0, 0]],
                    },
                },
                "Onsite": {
                    0: {(0, 0): 0.0, (1, 0): 0.0},
                },
            },
            "TwoBody": {
                "Local": {
                    "Parameter": "SlaterKanamori",
                    "option": {
                        (0, (0,)): {"l": 0, "U": 2.0, "Up": 0.0},
                        (1, (0,)): {"l": 0, "U": 2.0, "Up": 0.0},
                    },
                },
                "NonLocal": "None",
            },
        },
        "Control": {
            "Method": "gw",
            "Prefix": str(prefix),
            "NSCF": 1,
            "Mix": 0.1,
            "T": 2000,
            "MatsubaraCutOff": 5,
            "ConstantW": 1.0,
        },
    }


def _write_current_input(path, prefix):
    path.write_text(repr(_input_sections(prefix)), encoding="utf-8")


def _write_legacy_input(path, prefix):
    sections = _input_sections(prefix)
    path.write_text(
        "\n\n".join(f"{name} = {sections[name]!r}" for name in sections),
        encoding="utf-8",
    )


def test_load_qassemble_in_preserves_tuple_keys(tmp_path):
    input_path = tmp_path / "qassemble.in"
    _write_current_input(input_path, tmp_path / "tuple_key")

    data = load_input_file(input_path)

    hopping = data["Hamiltonian"]["OneBody"]["Hopping"]
    assert ((0, 0), (1, 0)) in hopping
    assert data["Crystal"]["SOC"] is False


def test_load_legacy_input_ini_with_deprecation_warning(tmp_path):
    input_path = tmp_path / "input.ini"
    _write_legacy_input(input_path, tmp_path / "legacy")

    with pytest.warns(FutureWarning, match="deprecated"):
        data = load_input_file(input_path)

    assert set(data) == {"Crystal", "Hamiltonian", "Control"}


@pytest.mark.parametrize(
    "content",
    [
        '{"Crystal": __import__("os"), "Hamiltonian": {}, "Control": {}}',
        '{"Crystal": 1 + 2, "Hamiltonian": {}, "Control": {}}',
        "import os\n",
        "Crystal = {}\nHamiltonian = {}\nControl = {}\nExtra = {}\n",
    ],
)
def test_rejects_executable_or_nonliteral_input(tmp_path, content):
    name = (
        "input.ini"
        if content.startswith("Crystal =") or content.startswith("import")
        else "bad.in"
    )
    input_path = tmp_path / name
    input_path.write_text(content, encoding="utf-8")
    warning_context = (
        pytest.warns(FutureWarning, match="deprecated")
        if name == "input.ini"
        else nullcontext()
    )

    with warning_context:
        with pytest.raises(InputFormatError):
            load_input_file(input_path)


def test_rejects_missing_required_section(tmp_path):
    input_path = tmp_path / "qassemble.in"
    input_path.write_text(repr({"Crystal": {}, "Hamiltonian": {}}), encoding="utf-8")

    with pytest.raises(InputFormatError, match="Control"):
        load_input_file(input_path)


def test_resolve_prefers_qassemble_in_over_legacy_input_ini(tmp_path, monkeypatch):
    current_path = tmp_path / "qassemble.in"
    legacy_path = tmp_path / "input.ini"
    current_path.write_text("{}", encoding="utf-8")
    legacy_path.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert resolve_input_file() == current_path


def test_run_test_mode_reads_explicit_input_file(tmp_path):
    input_path = tmp_path / "custom.in"
    prefix = tmp_path / "run_smoke"
    _write_current_input(input_path, prefix)

    run = Run(test=True, input_file=input_path)

    assert run.control["run"]["fn"] == str(prefix)
    assert run.control["run"]["method"] == "gw"
    assert run.func is not None


def test_cli_passes_custom_input_file(monkeypatch, capsys):
    calls = []

    class FakeRun:
        def __init__(self, input_file=None):
            calls.append(input_file)

    monkeypatch.setattr("QAssemble.CLI.Run", FakeRun)

    main(["custom.in"])

    assert calls == ["custom.in"]
    assert "Calculation Start" in capsys.readouterr().out
