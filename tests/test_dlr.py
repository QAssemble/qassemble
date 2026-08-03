import numpy as np

from QAssemble import DLR


def test_dlr_grids_are_nonempty_and_finite():
    dlr = DLR({"beta": 10.0, "cutoff": 5.0, "eps": 1e-10})

    for grid in (dlr.tauF, dlr.tauB, dlr.omega, dlr.nu):
        assert grid.size > 0
        assert np.isfinite(grid).all()


def test_dlr_fermion_transform_smoke():
    dlr = DLR({"beta": 10.0, "cutoff": 5.0, "eps": 1e-10})
    ftau = np.exp(-0.1 * dlr.tauF)

    ff = dlr.FT2F(ftau)
    back = dlr.FF2T(ff)

    assert ff.shape == dlr.omega.shape
    assert back.shape == dlr.tauF.shape
    assert np.isfinite(ff).all()
    assert np.isfinite(back).all()
