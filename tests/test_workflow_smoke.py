import copy

import h5py
import pytest

from QAssemble import CorrelationFunction


CRYSTAL = {
    "RVec": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "SOC": False,
    "CorF": "F",
    "Basis": [[[0, 0, 0], 1], [[0.5, 0, 0], 1]],
    "NSpin": 1,
    "NElec": 1,
    "KGrid": [1, 1, 1],
}
FT = {"T": 300, "MatsubaraCutOff": 10}
HOPPING = {((0, 0), (1, 0)): {1.0: [[0, 0, 0]]}}
ONSITE = {0: {(0, 0): 0.0, (1, 0): 0.0}}
LOCAL_COULOMB = {
    "Parameter": "SlaterKanamori",
    "option": {
        1: {"l": 0, "value": [2.0, 0.0, 0.0], "orbitals": [0]},
        2: {"l": 0, "value": [2.0, 0.0, 0.0], "orbitals": [0]},
    },
}


def _workflow():
    return CorrelationFunction(cry=copy.deepcopy(CRYSTAL), ft=copy.deepcopy(FT))


def test_tight_binding_writes_h0_group(tmp_path):
    output = tmp_path / "tb.h5"
    workflow = _workflow()
    workflow.TightBinding(hopping=HOPPING, onsite=ONSITE, hdf5file=str(output))

    with h5py.File(output, "r") as h5file:
        assert set(h5file["tb"]) == {"H0"}
        assert "h0k" in h5file["tb/H0"]


@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
def test_one_iteration_hf_uses_manuscript_groups(tmp_path):
    output = tmp_path / "hf.h5"
    workflow = _workflow()
    workflow.HartreeFock(
        itermax=1,
        mix=0.1,
        hopping=HOPPING,
        onsite=ONSITE,
        loccoulomb=LOCAL_COULOMB,
        hdf5file=str(output),
    )

    with h5py.File(output, "r") as h5file:
        assert {"H0", "H", "SigH", "SigF", "V"} <= set(h5file["hf"])


def test_one_iteration_gw_uses_manuscript_groups(tmp_path):
    output = tmp_path / "gw.h5"
    workflow = _workflow()
    workflow.GWApproximation(
        itermax=1,
        mix=0.1,
        hoppinglist=HOPPING,
        onsitelist=ONSITE,
        loccoulomb=LOCAL_COULOMB,
        hdf5file=str(output),
    )

    with h5py.File(output, "r") as h5file:
        assert {"H0", "G0", "G", "SigH", "SigF", "SigGWC", "P", "W", "V"} <= set(
            h5file["gw"]
        )
