from types import SimpleNamespace

import h5py
import numpy as np
from QAssemble.BLocDyn import WImp


class _DLR:
    def __init__(self):
        self.nu = np.asarray([0.0, 1.0])

    def MatsubaraDLR2UniformGrid(self, value, sign=1):
        assert sign == 1
        return np.asfortranarray(np.asarray(value) + 10.0)

    def BatchBF2T(self, value):
        return 2.0 * np.asarray(value)


def _inputs():
    projector = SimpleNamespace(
        bprojector={"1": np.ones((1, 1, 1))},
        equiv={"1": np.eye(1, dtype=int)},
    )
    utilde = np.asarray([[[[[2.0, 3.0]]]]], dtype=np.complex128)
    pimp = np.asarray([[[[[0.1, -0.2]]]]], dtype=np.complex128)
    return projector, utilde, pimp


def test_wimp_builds_screened_interaction_from_mixed_pimp(tmp_path):
    projector, utilde, polarization = _inputs()
    obj = WImp(
        crystal=SimpleNamespace(ns=1),
        dlr=_DLR(),
        projector=projector,
        key="1",
        utilde=utilde,
        polarization=polarization,
        hdf5file=str(tmp_path / "calc.h5"),
        group="edmft",
        iteration=2,
    )

    expected = utilde / (1.0 - polarization * utilde)
    np.testing.assert_allclose(obj.f, expected)
    np.testing.assert_allclose(obj.f_uniform, expected + 10.0)
    np.testing.assert_allclose(obj.t, 2.0 * expected)

    obj.Save("wimp")
    with h5py.File(tmp_path / "calc.h5", "r") as handle:
        np.testing.assert_allclose(handle["edmft/WImp/wimp.2.1"][:], expected)
        np.testing.assert_allclose(
            handle["edmft/WImp/wimp.2.1_uniform"][:], expected + 10.0
        )
