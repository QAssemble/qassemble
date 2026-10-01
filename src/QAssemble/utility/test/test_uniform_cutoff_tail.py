"""Input-cutoff grids and moment extension beyond the solver's measured grid."""
import numpy as np
import pytest
from QAssemble.utility.DLR import DLR


@pytest.mark.parametrize("beta,cutoff", [(100., 300.), (20., 8.)])
def test_uniform_defaults_match_full_counts_not_dlr_endpoints(beta, cutoff):
    dlr = DLR(dict(beta=beta, cutoff=cutoff))
    count = int(cutoff / (2*np.pi/beta))
    np.testing.assert_allclose(dlr.MatsubaraFermionUniform(), (2*np.arange(count)+1)*np.pi/beta)
    np.testing.assert_allclose(dlr.MatsubaraBosonUniform(), np.arange(count)*2*np.pi/beta)
    assert len(dlr.MatsubaraFermionUniform()) == len(dlr.MatsubaraBosonUniform()) == count
    assert dlr.MatsubaraFermionUniform()[-1] < cutoff
    assert dlr.MatsubaraBosonUniform()[-1] < cutoff
    # Explicit Emax retains the prior inclusive floor/ceil behavior.
    emax = 2*cutoff
    assert len(dlr.MatsubaraFermionUniform(Emax=emax)) == int(np.floor((beta*emax/np.pi-1)/2))+1
    assert len(dlr.MatsubaraBosonUniform(Emax=emax)) == int(np.ceil(beta*emax/(2*np.pi)))+1


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("highzero,oddzero", [(True, False), (True, True), (False, False), (False, True)])
def test_complex_matrix_tail_extension_and_hermitian_fold(sign, highzero, oddzero):
    dlr = DLR(dict(beta=20., cutoff=8.))
    freq = dlr.MatsubaraFermionUniform() if sign == -1 else dlr.MatsubaraBosonUniform()
    shape = (2,2,1) if sign == -1 else (2,2,1,1)
    matrix = np.array([[1., 0.2+0.3j], [0.2-0.3j, 0.7]])
    moments = np.zeros((*shape,4), complex)
    powers = [k for k in range(4) if not(highzero and k==0) and not(oddzero and k%2)]
    for k in powers:
        moments[...,k] = matrix.reshape(shape)*(k+1)*0.1
    positive = freq > 0
    values = np.zeros((*shape,len(freq)), complex)
    values[...,positive] = sum(moments[...,k,None]/(1j*freq[positive])**k for k in powers)
    # The zero mode is independent of the high-frequency expansion.
    values[...,~positive] = matrix.reshape(shape)[...,None]
    supplied, extended_freq = dlr._ExtendUniformTail(values, freq, sign, tail=moments, highzero=highzero, oddzero=oddzero)
    inferred, inferred_freq = dlr._ExtendUniformTail(values, freq, sign, highzero=highzero, oddzero=oddzero)
    np.testing.assert_allclose(inferred_freq, extended_freq)
    np.testing.assert_allclose(inferred, supplied, atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(supplied[...,:len(freq)], values)
    assert extended_freq[-1] >= np.max(np.abs(dlr.omega if sign==-1 else dlr.nu))
    if sign == -1:
        signed = dlr.MatsubaraAddNegativeFrequency(values)
        signed_freq = np.concatenate((-freq[::-1],freq))
    else:
        neg = np.conjugate(np.swapaxes(np.swapaxes(values[...,1:][...,::-1],0,1),2,3))
        signed = np.concatenate((neg,values),axis=-1)
        signed_freq = np.concatenate((-freq[1:][::-1],freq))
    extended, full_freq = dlr._ExtendUniformTail(signed, signed_freq, sign, tail=moments, highzero=highzero, oddzero=oddzero)
    negative = extended[...,full_freq<0]
    expected = np.conjugate(np.swapaxes(supplied[...,1 if sign==1 else 0:][...,::-1],0,1))
    if sign == 1:
        expected = np.swapaxes(expected,2,3)
    np.testing.assert_allclose(negative,expected)
    np.testing.assert_allclose(dlr.MatsubaraUniformGrid2DLR(values,omega=freq,sign=sign,tail=moments,highzero=highzero,oddzero=oddzero),
                               dlr.MatsubaraUniformGrid2DLR(signed,omega=signed_freq,sign=sign,tail=moments,highzero=highzero,oddzero=oddzero),atol=1e-10)


@pytest.mark.parametrize("sign", [-1,1])
def test_pole_model_truncation_matches_covering_grid(sign):
    dlr = DLR(dict(beta=100., cutoff=100.))
    freq = dlr.MatsubaraFermionUniform() if sign==-1 else dlr.MatsubaraBosonUniform()
    target = dlr.omega if sign==-1 else dlr.nu
    full = dlr.MatsubaraFermionUniform(Emax=np.max(abs(target))) if sign==-1 else dlr.MatsubaraBosonUniform(Emax=np.max(abs(target)))
    shape = (1,1,1) if sign==-1 else (1,1,1,1)
    evaluate = (lambda f: 1/(1j*f-0.05)) if sign==-1 else (lambda f: -1/(f*f+0.05**2))
    partial = evaluate(freq).reshape(*shape,-1)
    reference = evaluate(full).reshape(*shape,-1)
    result = dlr.MatsubaraUniformGrid2DLR(partial,omega=freq,sign=sign,oddzero=sign==1)
    expected = dlr.MatsubaraUniformGrid2DLR(reference,omega=full,sign=sign,oddzero=sign==1)
    # Compare the spectrum with relative L2 tolerance; a finite-order tail
    # is approximate and tiny outer-node values are sensitive to DLR LU noise.
    assert np.linalg.norm(result-expected) / np.linalg.norm(expected) < 1e-6


def test_tail_extension_rejects_invalid_signed_grid_and_nonfinite_data():
    dlr=DLR(dict(beta=20.,cutoff=8.))
    freq=dlr.MatsubaraFermionUniform()
    value=np.ones((1,1,1,len(freq)),complex)
    with pytest.raises(ValueError,match="finite"):
        dlr._ExtendUniformTail(value*np.nan,freq,-1)
    with pytest.raises(ValueError,match="symmetric"):
        dlr._ExtendUniformTail(np.ones((1,1,1,4)),np.array([-3,-1,1,2.]),-1)
    with pytest.raises(ValueError,match="uniform"):
        dlr._ExtendUniformTail(value,freq*1.1,-1)
