import numpy as np


def test_minimal_crystal_index_maps(minimal_crystal):
    crystal = minimal_crystal

    assert crystal.nk == 4
    assert len(crystal.kpoint) == 4
    assert len(crystal.find) == 2
    assert len(crystal.bind) == 2

    assert crystal.basisf.shape == (2, 3)
    assert crystal.forb2atom.shape == (2,)
    assert crystal.borb2atom.shape == (2,)
    assert np.isfinite(crystal.basisf).all()
    assert np.isfinite(crystal.forb2atom).all()
    assert np.isfinite(crystal.borb2atom).all()
