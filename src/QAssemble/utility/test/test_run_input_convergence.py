import textwrap

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
