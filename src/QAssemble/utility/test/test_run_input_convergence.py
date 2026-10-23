import textwrap

import pytest

from QAssemble.Run import Run


def _write_minimal_input(path, control_extra):
    path.write_text(
        textwrap.dedent(
            f"""
            Crystal = {{
                'RVec': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                'Basis': [[[0, 0, 0], 1]],
                'KGrid': [1, 1, 1],
                'NElec': 1.0,
            }}

            Hamiltonian = {{
                'OneBody': {{
                    'Hopping': {{}},
                    'Onsite': {{}},
                }},
                'TwoBody': {{
                    'Local': {{
                        'Parameter': 'SlaterKanamori',
                    }},
                }},
            }}

            Control = {{
                'Method': 'dmft',
                'Prefix': 'calc',
                'beta': 100,
                {control_extra}
            }}

            Impurity = {{}}
            """
        )
    )


@pytest.mark.parametrize("extra,expected", [("", True), ("'BosonSymmetry': False,", False),
                                           ("'BosonSymmetry': True,", True)])
def test_input_boson_symmetry(monkeypatch, tmp_path, extra, expected):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(tmp_path / "input.ini", extra)
    runner = object.__new__(Run)
    runner.ReadInput()
    assert runner.control["ft"]["symmetry"] is expected


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ("", (64, 64)),
        ("'CTQMCMPIRanks': 32, 'MeasureMPIRanks': 16,", (32, 16)),
    ],
)
def test_input_mpi_ranks(monkeypatch, tmp_path, extra, expected):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(tmp_path / "input.ini", extra)
    runner = object.__new__(Run)
    runner.ReadInput()
    assert (
        runner.control["run"]["CTQMCMPIRanks"],
        runner.control["run"]["MeasureMPIRanks"],
    ) == expected


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
@pytest.mark.parametrize("key", ["CTQMCMPIRanks", "MeasureMPIRanks"])
def test_input_mpi_ranks_reject_nonpositive_or_noninteger(
    monkeypatch, tmp_path, key, value
):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(tmp_path / "input.ini", f"'{key}': {value!r},")
    with pytest.raises(ValueError, match=key):
        object.__new__(Run).ReadInput()


@pytest.mark.parametrize("original", ["", "'CTQMCMPIRanks': 16, 'MeasureMPIRanks': 8,"])
def test_restart_allows_changed_mpi_ranks(monkeypatch, tmp_path, original):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(tmp_path / "input.ini", original)
    object.__new__(Run).ReadInput()

    _write_minimal_input(
        tmp_path / "input.ini", "'CTQMCMPIRanks': 32, 'MeasureMPIRanks': 16,"
    )
    runner = object.__new__(Run)
    runner.ReadInput()
    assert runner.control["run"]["CTQMCMPIRanks"] == 32
    assert runner.control["run"]["MeasureMPIRanks"] == 16


def test_input_tolerance_g_expands_to_gloc_tolerances(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(tmp_path / "input.ini", "'ToleranceG': 1.0e-6,")

    runner = object.__new__(Run)
    runner.ReadInput()

    assert runner.control["run"]["tol_dGLoc_abs"] == 1.0e-6
    assert runner.control["run"]["tol_dGLoc_rel"] == 1.0e-6
    assert runner.control["run"]["tol_GLoc_GImp_abs"] == 1.0e-6
    assert "tol_dWLoc_abs" not in runner.control["run"]
    assert "tol_dmu_abs" not in runner.control["run"]


def test_input_tolerence_g_spelling_is_supported(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(tmp_path / "input.ini", "'TolerenceG': 1.0e-6,")

    runner = object.__new__(Run)
    runner.ReadInput()

    assert runner.control["run"]["tol_dGLoc_abs"] == 1.0e-6
    assert runner.control["run"]["tol_dGLoc_rel"] == 1.0e-6
    assert runner.control["run"]["tol_GLoc_GImp_abs"] == 1.0e-6


def test_input_tolerance_w_expands_to_wloc_tolerances(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(tmp_path / "input.ini", "'ToleranceW': 2.0e-6,")

    runner = object.__new__(Run)
    runner.ReadInput()

    assert runner.control["run"]["tol_dWLoc_abs"] == 2.0e-6
    assert runner.control["run"]["tol_dWLoc_rel"] == 2.0e-6
    assert "tol_dGLoc_abs" not in runner.control["run"]
    assert "tol_GLoc_GImp_abs" not in runner.control["run"]


def test_input_tolerance_shortcuts_do_not_override_explicit_tol(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(
        tmp_path / "input.ini",
        "'ToleranceG': 1.0e-6,\n"
        "                'ToleranceW': 2.0e-6,\n"
        "                'tol_dGLoc_abs': 3.0e-6,\n"
        "                'tol_dWLoc_abs': 4.0e-6,",
    )

    runner = object.__new__(Run)
    runner.ReadInput()

    assert runner.control["run"]["tol_dGLoc_abs"] == 3.0e-6
    assert runner.control["run"]["tol_dGLoc_rel"] == 1.0e-6
    assert runner.control["run"]["tol_GLoc_GImp_abs"] == 1.0e-6
    assert runner.control["run"]["tol_dWLoc_abs"] == 4.0e-6
    assert runner.control["run"]["tol_dWLoc_rel"] == 2.0e-6


def test_input_mixing_controls(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(
        tmp_path / "input.ini",
        "'Mix': 0.1,\n"
        "                'MixingMethod': 'linear',\n"
        "                'NPulay': 7,",
    )

    runner = object.__new__(Run)
    runner.ReadInput()

    assert runner.control["run"]["mix"] == 0.1
    assert "mix_sig" not in runner.control["run"]
    assert "mix_p" not in runner.control["run"]
    assert runner.control["run"]["MixingMethod"] == "linear"
    assert runner.control["run"]["NPulay"] == 7
    assert runner.control["run"]["mixing_method"] == "linear"
    assert runner.control["run"]["npulay"] == 7

def test_input_min_scf_defaults_to_five_for_pulay(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(tmp_path / "input.ini", "")

    runner = object.__new__(Run)
    runner.ReadInput()

    assert runner.control["run"]["MinSCF"] == 5
    assert runner.control["run"]["min_iter"] == 5
    assert runner.control["run"]["NPulay"] == 5
    assert runner.control["run"]["npulay"] == 5


def test_input_min_scf_defaults_to_one_for_non_pulay(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(
        tmp_path / "input.ini",
        "'MixingMethod': 'linear',",
    )

    runner = object.__new__(Run)
    runner.ReadInput()

    assert runner.control["run"]["MinSCF"] == 1
    assert runner.control["run"]["min_iter"] == 1
    assert runner.control["run"]["NPulay"] == 1
    assert runner.control["run"]["npulay"] == 1


def test_input_min_scf_uses_explicit_value(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(tmp_path / "input.ini", "'MinSCF': 4,")

    runner = object.__new__(Run)
    runner.ReadInput()

    assert runner.control["run"]["MinSCF"] == 4
    assert runner.control["run"]["min_iter"] == 4
    assert runner.control["run"]["NPulay"] == 4
    assert runner.control["run"]["npulay"] == 4


def test_input_ignores_obsolete_sigimp_smoothing_options(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(
        tmp_path / "input.ini",
        "'SigImpSmoothing': True, 'SigImpSmoothingWidth': 0.3,",
    )
    runner = object.__new__(Run)
    runner.ReadInput()
    assert not {
        "SigImpSmoothing", "sigimp_smoothing",
        "SigImpSmoothingWidth", "sigimp_smoothing_width",
    }.intersection(runner.control["run"])


@pytest.mark.parametrize("shortcut,name", [("ToleranceG", "G"), ("TolerenceG", "G"), ("ToleranceW", "W")])
def test_gwedmft_tolerance_shortcut_uses_lattice_name(monkeypatch, tmp_path, shortcut, name):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(tmp_path / "input.ini", f"'Method': 'gw+edmft', '{shortcut}': 1e-6,")
    runner = object.__new__(Run)
    runner.ReadInput()
    assert runner.control["run"][f"tol_d{name}_abs"] == 1e-6
    assert runner.control["run"][f"tol_d{name}_rel"] == 1e-6
    for key in ("tol_dGLoc_abs", "tol_dWLoc_abs", "tol_GLoc_GImp_abs"):
        assert key not in runner.control["run"]


@pytest.mark.parametrize("name", ["G", "W"])
@pytest.mark.parametrize("kind", ["abs", "rel"])
def test_gwedmft_explicit_lattice_tolerance_overrides_shortcuts(monkeypatch, tmp_path, caplog, name, kind):
    monkeypatch.chdir(tmp_path)
    key = f"tol_d{name}_{kind}"
    _write_minimal_input(tmp_path / "input.ini",
                         f"'Method': 'gw+edmft', 'ToleranceG': 1e-6, 'ToleranceW': 2e-6, '{key}': 3e-6,")
    runner = object.__new__(Run)
    runner.ReadInput()
    assert runner.control["run"][key] == 3e-6
    other_kind = "rel" if kind == "abs" else "abs"
    assert runner.control["run"][f"tol_d{name}_{other_kind}"] == (1e-6 if name == "G" else 2e-6)
    assert f"'{key}' is deprecated" not in caplog.text


@pytest.mark.parametrize("method", ["dmft", "edmft"])
def test_local_methods_keep_legacy_g_tolerance_warning(monkeypatch, tmp_path, caplog, method):
    monkeypatch.chdir(tmp_path)
    _write_minimal_input(tmp_path / "input.ini",
                         f"'Method': '{method}', 'ToleranceG': 1e-6, 'tol_dG_abs': 3e-6,")
    runner = object.__new__(Run)
    runner.ReadInput()
    assert runner.control["run"]["tol_dGLoc_abs"] == 1e-6
    assert "'tol_dG_abs' is deprecated" in caplog.text
