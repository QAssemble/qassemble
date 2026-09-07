import h5py
import numpy as np
import pytest

from QAssemble.Crystal import Crystal
from QAssemble.FLatDyn import G, G0
from QAssemble.utility.DLR import DLR


def test_interacting_green_mu_is_physical(tmp_path):
    crystal = Crystal(
        {
            "RVec": np.eye(3),
            "Basis": [[[0, 0, 0], 1]],
            "NSpin": 1,
            "NElec": 1,
            "KGrid": [1, 1, 1],
        }
    )
    dlr = DLR({"beta": 10.0, "cutoff": 5.0, "eps": 1e-10})
    g0 = G0(crystal, dlr, np.zeros((1, 1, 1, 1), dtype=complex))
    output = tmp_path / "green.h5"
    green = G(
        crystal,
        dlr,
        g0.kf,
        sigh=np.full((1, 1, 1, 1), 2.0),
        hdf5file=output,
        group="gw",
    )

    green.Save("gkf", chem=True)

    assert green.mu == pytest.approx(2.0, abs=1e-6)
    assert green.c == 0.0
    with h5py.File(output, "r") as h5:
        assert h5["/gw/G/mu"][()] == pytest.approx(green.mu)
