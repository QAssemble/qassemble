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
    bath.Mixing(dict(mix=.5,mixing_method='linear',npulay=5))
    np.testing.assert_allclose(bath.cstatic,0.05,atol=1e-10)
    np.testing.assert_allclose(bath.ct,0.,atol=1e-8)
    np.testing.assert_allclose(bath.cf_to_solver_uniform,0.05,atol=1e-10)
    np.testing.assert_allclose(bath.f_to_solver_uniform,4.05,atol=1e-10)
    bath.Save('bweiss')
    with h5py.File(bath.hdf5file,'r') as handle:
        assert 'bweiss.1.1_cstatic' in handle['gwedmft/BWeiss']


@pytest.mark.parametrize('static_fit', [False, True])
@pytest.mark.parametrize('with_p', [False, True])
def test_bweiss_cal_does_not_project(monkeypatch, tmp_path, static_fit, with_p):
    dlr, crystal, projector = fixtures()
    v = np.ones((1, 1, 1, 1)) * 4.0
    pole = (-0.7 / (dlr.nu**2 + 0.6**2))[None, None, None, None, :]
    w = v[..., None] + pole + 0.05
    p = np.full_like(w, -0.01) if with_p else None
    bath = BWeiss(crystal, dlr, projector, '1',
                  vloc=SimpleNamespace(vproj={'1': v}), p=p,
                  static_fit=static_fit, hdf5file=str(tmp_path / 'bath.h5'),
                  group='gwedmft', iteration=2)
    bath.w = w
    previous_cf = np.full_like(w, -0.2)
    previous_static = np.full_like(v, 0.03)
    bath.WriteBrdPrev('bweiss', previous_cf)
    bath.WriteBrdPrev('bweiss_cstatic', previous_static)

    def unexpected_projection(*args, **kwargs):
        pytest.fail('Cal must leave projection and cache updates to Mixing')

    monkeypatch.setattr(bath, 'CausalProjection', unexpected_projection)
    monkeypatch.setattr(bath, '_StaticFitProjection', unexpected_projection)
    monkeypatch.setattr(bath, 'WriteBrdPrev', unexpected_projection)
    raw_f = w if p is None else bath.Dyson(w, -p)
    bath.Cal()
    np.testing.assert_allclose(bath.f, raw_f)
    np.testing.assert_allclose(bath.cf, raw_f - v[..., None])
    np.testing.assert_allclose(bath.cf_raw, bath.cf)
    assert not np.shares_memory(bath.cf_raw, bath.cf)
    np.testing.assert_allclose(bath.cf_uniform,
                               dlr.MatsubaraDLR2UniformGrid(bath.cf, sign=1))
    np.testing.assert_array_equal(bath.ReadBrdPrev('bweiss', w.shape), previous_cf)
    np.testing.assert_array_equal(bath.ReadBrdPrev('bweiss_cstatic', v.shape),
                                  previous_static)
    assert np.isnan(bath.projection_delta_abs)
    assert np.isnan(bath.projection_delta_rel)
    if static_fit:
        np.testing.assert_array_equal(bath.cstatic, np.zeros_like(v))
    bath.Save('bweiss')
    with h5py.File(bath.hdf5file, 'r') as handle:
        group = handle['gwedmft/BWeiss']
        np.testing.assert_allclose(group['bweiss.2.1_correlated'][()], bath.cf)
        np.testing.assert_allclose(group['bweiss.2.1_correlated_raw'][()], bath.cf_raw)
        assert group['bweiss.2.1_correlated_raw'].attrs['stage'] == 'before mixing'
        assert np.isnan(group['bweiss.2.1_projection_delta_rel'][()])
        np.testing.assert_allclose(group['bweiss.2.1_c2_raw'][()], bath.c2_raw)
        assert 'bweiss.2.1_c2_mixed' not in group


@pytest.mark.parametrize('static_fit', [False, True])
def test_bweiss_mix_project_save_roundtrip(monkeypatch, tmp_path, static_fit):
    dlr, crystal, projector = fixtures()
    v = np.ones((1, 1, 1, 1)) * 4.0
    path = str(tmp_path / 'bath.h5')
    calls = []
    project = BWeiss.CausalProjection

    def record_projection(self, value, **kwargs):
        calls.append((np.array(value, copy=True), kwargs))
        return project(self, value, **kwargs)

    monkeypatch.setattr(BWeiss, 'CausalProjection', record_projection)
    previous_cf = previous_static = None
    for iteration in (1, 2):
        pole = (-0.1 * iteration / (dlr.nu**2 + 0.6**2))[None, None, None, None, :]
        bath = BWeiss(crystal, dlr, projector, '1',
                      vloc=SimpleNamespace(vproj={'1': v}), w=v[..., None] + pole,
                      static_fit=static_fit, hdf5file=path, group='gwedmft',
                      iteration=iteration)
        assert len(calls) == iteration - 1
        raw_cf = bath.cf.copy()
        if static_fit:
            raw_src = dlr.MatsubaraDLR2UniformGrid(raw_cf, sign=1)
            moment_kw = dict(grid='uniform', highzero=False, tail_points=5,
                             tail_log_spaced=False)
        else:
            raw_src, moment_kw = raw_cf, dict(grid='dlr', highzero=True)
        expected_c2_raw = -bath.Moment(raw_src, oddzero=True, **moment_kw)[0][..., 1]
        np.testing.assert_allclose(bath.c2_raw, expected_c2_raw)
        bath.Save('bweiss')
        mixed_cf = raw_cf if iteration == 1 else 0.5 * (raw_cf + previous_cf)
        bath.Mixing(dict(mix=.5, mixing_method='linear', npulay=5))
        assert len(calls) == iteration
        projected_input, kwargs = calls[-1]
        assert 'enforce_moments' not in kwargs
        expected_input = (dlr.MatsubaraDLR2UniformGrid(mixed_cf, sign=1)
                          if static_fit else mixed_cf)
        np.testing.assert_allclose(projected_input, expected_input)
        expected_c2_mixed = -bath.Moment(expected_input, oddzero=True, **moment_kw)[0][..., 1]
        np.testing.assert_allclose(bath.c2_mixed, expected_c2_mixed)
        if iteration == 1:
            assert kwargs['fallback_matrix'] is None
        else:
            np.testing.assert_allclose(kwargs['fallback_matrix'], previous_cf)
            if static_fit:
                np.testing.assert_allclose(kwargs['fallback_static'], previous_static)
        np.testing.assert_allclose(bath.projection_delta_abs,
                                   np.max(np.abs(bath.cf - mixed_cf)))
        np.testing.assert_allclose(bath.projection_delta_rel,
                                   np.linalg.norm(bath.cf - mixed_cf)
                                   / max(np.linalg.norm(mixed_cf), np.finfo(float).eps))
        bath.Save('bweiss')
        with h5py.File(path, 'r') as handle:
            group = handle['gwedmft/BWeiss']
            name = f'bweiss.{iteration}.1'
            np.testing.assert_allclose(group[name][()], bath.f)
            np.testing.assert_allclose(group[name + '_correlated'][()], bath.cf)
            np.testing.assert_allclose(group[name + '_correlated_raw'][()], raw_cf)
            assert group[name + '_correlated_raw'].attrs['stage'] == 'before mixing'
            np.testing.assert_allclose(group[name + '_to_solver'][()], bath.f_to_solver)
            np.testing.assert_allclose(group[name + '_correlated_uniform'][()],
                                       bath.cf_uniform)
            np.testing.assert_allclose(group['bweiss_brd_prev.1'][()], bath.cf)
            np.testing.assert_allclose(group[name + '_c2_raw'][()], bath.c2_raw)
            np.testing.assert_allclose(group[name + '_c2_mixed'][()], bath.c2_mixed)
            for stage in ('raw', 'mixed'):
                assert group[name + '_c2_' + stage].attrs['grid'] == ('uniform' if static_fit else 'dlr')
            if static_fit:
                np.testing.assert_allclose(group[name + '_raw_uniform'][()], expected_input)
                assert group[name + '_raw_uniform'].attrs['stage'] == 'after mixing, before projection'
            np.testing.assert_allclose(handle['gwedmft/Mixing/1/bweiss/last'][()], bath.cf)
        previous_cf = bath.cf.copy()
        if static_fit:
            previous_static = bath.cstatic.copy()


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
