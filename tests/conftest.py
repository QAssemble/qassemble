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
