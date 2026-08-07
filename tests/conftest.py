import os

import numpy as np
import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/qassemble-mpl-cache")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def minimal_crystal_input():
    return {
        "RVec": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "Basis": [[[0.0, 0.0, 0.0], 1], [[0.5, 0.0, 0.0], 1]],
        "CorF": "F",
        "NSpin": 1,
        "SOC": False,
        "NElec": 2,
        "KGrid": [2, 2, 1],
    }


@pytest.fixture
def minimal_crystal(minimal_crystal_input):
    from QAssemble import Crystal

    return Crystal(cry=minimal_crystal_input)


def graphene_sections(prefix, method="gw", kgrid=(5, 5, 1), nscf=100, cutoff=100):
    """Manuscript graphene model (t=1.0, U=2.0, V=0.2) with adjustable size knobs."""
    return {
        "Crystal": {
            "RVec": [[1, 0, 0], [0.5, 0.866, 0], [0, 0, 1]],
            "SOC": False,
            "CorF": "F",
            "Basis": [[[0.33333, 0.33333, 0], 1], [[0.66667, 0.66667, 0], 1]],
            "NSpin": 1,
            "NElec": 2,
            "KGrid": list(kgrid),
        },
        "Hamiltonian": {
            "OneBody": {
                "Hopping": {
                    ((0, 0), (1, 0)): {1.0: [[0, 0, 0], [-1, 0, 0], [0, -1, 0]]},
                },
                "Onsite": {0: {(0, 0): 0.0, (1, 0): 0.0}},
            },
            "TwoBody": {
                "Local": {
                    "Parameter": "SlaterKanamori",
                    "option": {
                        (0, (0,)): {"l": 0, "U": 2.0, "Up": 0.0},
                        (1, (0,)): {"l": 0, "U": 2.0, "Up": 0.0},
                    },
                },
                "NonLocal": {
                    ((0, 0), (1, 0)): {0.20: [[0, 0, 0], [-1, 0, 0], [0, -1, 0]]},
                },
            },
        },
        "Control": {
            "Method": method,
            "Prefix": str(prefix),
            "NSCF": nscf,
            "Mix": 0.1,
            "T": 2000,
            "MatsubaraCutOff": cutoff,
            "ConstantW": 1.0,
        },
    }


def write_qassemble_input(path, sections):
    path.write_text(repr(sections), encoding="utf-8")
