"""Full last-five tail convention and instantaneous bosonic terms."""
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
from QAssemble.BLocDyn import BLocDyn, BWeiss, WLoc
from QAssemble.utility.DLR import DLR


def full_static_reference(values, nu, tail_points=5):
    # FullGWEDMFT/bin/causal_boson.py:fit_boson_static_tail (0e9b72d).
    arr = np.asarray(values, dtype=np.complex128)
    nonzero = np.flatnonzero(np.abs(nu)>0)
    if nonzero.size < 2:
        return complex(arr[-1]), 0.0
    idx = nonzero[-min(int(tail_points), nonzero.size):]
    x = 1.0 / nu[idx]**2
    coeff, *_ = np.linalg.lstsq(np.column_stack((np.ones_like(x), x)), arr[idx], rcond=None)
    return complex(coeff[0]), float(np.real(coeff[1]))


def fixtures():
    dlr=DLR(dict(beta=20.,cutoff=8.))
    crystal=SimpleNamespace(ns=1)
    projector=SimpleNamespace(bprojector={"1":np.ones((1,1,1))},equiv={"1":np.ones((1,1),int)},
                              blocal2pair={"1":[{0:(0,0)}]},ProbFPair2Borb=lambda *args:0)
    return dlr,crystal,projector


@pytest.mark.parametrize("static", [0.,0.05])
def test_static_tail_matches_full_reference(static):
    dlr,crystal,projector=fixtures()
    local=BLocDyn(crystal,dlr,projector)
    nu=dlr.MatsubaraBosonUniform()
    values=static-0.7/(nu**2+0.6**2)
    moment,high=local.Moment(values[None,None,None,None,:],grid="uniform",oddzero=True,
                              highzero=False,tail_points=5,tail_log_spaced=False)
    c0,c2=full_static_reference(values,nu)
    np.testing.assert_allclose(high[0,0,0,0],c0,atol=1e-10)
    np.testing.assert_allclose(-moment[0,0,0,0,1],c2,atol=1e-10)
    np.testing.assert_array_equal(moment[..., [0,2]],0)


def test_wloc_static_does_not_contaminate_tau_and_saves_diagnostics(tmp_path):
    dlr,crystal,projector=fixtures()
    v=np.ones((1,1,1,1))*4.
    pole=(-0.7/(dlr.nu**2+0.6**2))[None,None,None,None,:]
    path=str(tmp_path/'static.h5')
    objects=[]
    for index,constant in enumerate((0.,0.05)):
        w=WLoc(crystal,dlr,projector,'1',wlat=(v[...,None]+pole+constant)[...,None,:],
               vloc=v,causal=True,hdf5file=path,group=f'case{index}',iteration=1)
        w.Save('wloc')
        objects.append(w)
    np.testing.assert_allclose(objects[0].ct,objects[1].ct,atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(objects[1].cstatic-objects[0].cstatic,0.05,atol=1e-10)
    with h5py.File(path,'r') as handle:
        for suffix in ('raw_uniform','cstatic','c2','projection_delta_abs','projection_delta_rel'):
            assert 'wloc.1.1_'+suffix in handle['case1/WLoc']
        np.testing.assert_allclose(handle['case1/WLoc/wloc_cstatic_brd_prev.1'][()],objects[1].cstatic)


def test_bweiss_pure_static_survives_solver_grid_without_tau_term(tmp_path):
    dlr,crystal,projector=fixtures()
    v=np.ones((1,1,1,1))*4.
    w=np.broadcast_to((v+0.05)[...,None],(1,1,1,1,len(dlr.nu))).copy()
    bath=BWeiss(crystal,dlr,projector,'1',vloc=SimpleNamespace(vproj={'1':v}),w=w,
                 p=np.zeros_like(w),static_fit=True,hdf5file=str(tmp_path/'bath.h5'),group='gwedmft',iteration=1)
    np.testing.assert_allclose(bath.cstatic,0.05,atol=1e-10)
    np.testing.assert_allclose(bath.ct,0.,atol=1e-8)
    np.testing.assert_allclose(bath.cf_to_solver_uniform,0.05,atol=1e-10)
    np.testing.assert_allclose(bath.f_to_solver_uniform,4.05,atol=1e-10)
    bath.Mixing(dict(mix=.5,mixing_method='linear',npulay=5))
    np.testing.assert_allclose(bath.ct,0.,atol=1e-8)
    np.testing.assert_allclose(bath.cf_to_solver_uniform,0.05,atol=1e-10)
    bath.Save('bweiss')
    with h5py.File(bath.hdf5file,'r') as handle:
        assert 'bweiss.1.1_cstatic' in handle['gwedmft/BWeiss']


def test_x_only_fallback_tracks_selected_static_and_leaves_no_tau_constant(monkeypatch):
    import importlib
    module=importlib.import_module('QAssemble.BLocDyn')
    dlr,crystal,projector=fixtures()
    local=BLocDyn(crystal,dlr,projector)
    current=np.array([[1.,.2],[.2,2.]])[:,:,None,None]
    previous=np.array([[3.,.4],[.4,4.]])[:,:,None,None]
    raw=np.broadcast_to(current[...,None],(2,2,1,1,len(dlr.MatsubaraBosonUniform()))).copy()
    fallback=np.broadcast_to(previous[...,None],(2,2,1,1,len(dlr.nu))).copy()
    calls=[]
    def select(projector,target,tail,**kwargs):
        calls.append(target)
        return kwargs['fallback_channel'] if len(calls)==3 else target
    monkeypatch.setattr(module,'ProjectBosonComponentWithFallback',select)
    out=local.CausalProjection(raw,grid='uniform',oddzero=True,highzero=False,
                              tail_points=5,tail_log_spaced=False,
                              fallback_matrix=fallback,fallback_static=previous)
    assert len(calls)==3
    np.testing.assert_allclose(out[0,1],2.4,atol=1e-12)
    np.testing.assert_allclose(local._projection_cstatic[0,1],2.4,atol=1e-12)
    np.testing.assert_allclose(local.F2T(out-local._projection_cstatic[...,None]),0.,atol=1e-8)
