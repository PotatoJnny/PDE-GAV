"""Tiny CPU-only scientific checks for structured factorization and reuse."""
from dataclasses import replace
import importlib
import json

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp import basic_oracle as basic
from pde_agent_verifier_solver.plugins.vpfp.structured_oracle import StructuredOracle
from tests.test_vpfp_derivative_features import tiny
from tests.test_vpfp_batched_source import decoder, _queries, _numpy, _reference


@pytest.mark.parametrize("kind", ["cartesian","sparse","sparse_shuffled_duplicates"])
@pytest.mark.parametrize("chunk", [1,7,8192])
def test_source_pinned_equivalence_for_exact_requested_pairs(decoder,kind,chunk):
    query=_queries(kind)
    before=query.copy()
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    actual=oracle.source(query,chunk_size=chunk)
    np.testing.assert_allclose(_numpy(actual),_reference(decoder,query),rtol=decoder.rtol,atol=decoder.atol)
    np.testing.assert_array_equal(query,before)
    info=oracle.cache_info()
    assert info["requested_points"]==len(query)
    assert info["assembled_pairs"]<=len(query)
    assert info["maximum_projection_batch"]<=8
    assert info["time_entries"]<=8
    assert info["cached_final_source_arrays"] is False
    json.dumps(info,allow_nan=False)


@pytest.mark.parametrize("field",["rho","g"])
def test_initial_matches_basic_and_original_fields(decoder,field):
    query=_queries("sparse_shuffled_duplicates");query[:,0]=0.
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    actual=oracle.initial(query,field_name=field)
    expected=decoder.model.fields(query,decoder.latents[1],equation_condition=decoder.conditions[1])[field]
    prior=basic.evaluate_initial_basic(decoder.model,query,decoder.latents[1],
        equation_condition=decoder.conditions[1],field_name=field)
    np.testing.assert_allclose(_numpy(actual),_numpy(expected),rtol=decoder.rtol,atol=decoder.atol)
    np.testing.assert_allclose(_numpy(actual),_numpy(prior),rtol=decoder.rtol,atol=decoder.atol)


def test_cartesian_gemm_only_if_complete_actual_pair_set(decoder):
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    query=_queries("cartesian")
    actual=oracle.source(query)
    assert oracle.cache_info()["cartesian_blocks"]==1
    assert oracle.cache_info()["sparse_blocks"]==0
    np.testing.assert_allclose(_numpy(actual),_reference(decoder,query),rtol=decoder.rtol,atol=decoder.atol)
    # Keep all marginal coordinates but remove one requested pair.
    query=query[1:]
    actual=oracle.source(query)
    assert oracle.cache_info()["cartesian_blocks"]==1
    assert oracle.cache_info()["mixed_cartesian_blocks"]==1
    assert oracle.cache_info()["sparse_pairs"]==4
    np.testing.assert_allclose(_numpy(actual),_reference(decoder,query),rtol=decoder.rtol,atol=decoder.atol)


def test_shuffled_duplicated_full_cartesian_scatter_is_exact(decoder):
    query=_queries("cartesian")
    indices=np.random.default_rng(718).permutation(len(query)).tolist()+[3,3,12]
    query=query[indices]
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    actual=oracle.source(query)
    info=oracle.cache_info()
    assert info["cartesian_blocks"]==1
    assert info["assembled_pairs"]==len(indices)-3
    np.testing.assert_allclose(_numpy(actual),_reference(decoder,query),rtol=decoder.rtol,atol=decoder.atol)


def test_cache_hits_and_exact_float64_time_keys(decoder):
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    query=_queries("sparse")
    query[:,0]=.3
    first=oracle.source(query)
    info=oracle.cache_info()
    second=oracle.source(query)
    warm=oracle.cache_info()
    np.testing.assert_array_equal(_numpy(first),_numpy(second))
    assert warm["time_hits"]>info["time_hits"]
    assert warm["velocity_hits"]>info["velocity_hits"]
    nearby=query.copy();nearby[:,0]=np.nextafter(.3,1.)
    oracle.source(nearby)
    assert oracle.cache_info()["time_entries"]==2


def test_velocity_key_changes_and_temperature_objects_isolated(decoder):
    query=_queries("sparse")
    first=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    baseline=first.source(query)
    changed=query.copy();changed[:,3]+=0.013
    shifted=first.source(changed)
    assert first.cache_info()["velocity_misses"]==2
    condition=decoder.conditions[1].copy();condition[1]=.92
    other=StructuredOracle(decoder.model,decoder.latents[1],condition)
    different=other.source(query)
    assert not np.allclose(_numpy(different),_numpy(baseline),rtol=decoder.rtol,atol=decoder.atol)
    np.testing.assert_allclose(_numpy(different),_reference(decoder,query,condition=condition),
                               rtol=decoder.rtol,atol=decoder.atol)
    np.testing.assert_allclose(_numpy(shifted),_reference(decoder,changed),rtol=decoder.rtol,atol=decoder.atol)


def test_time_lru_is_bounded_and_eviction_does_not_change_answer(decoder):
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    query=_queries("sparse")[:2]
    first=None
    for time in np.linspace(0.,1.,19):
        points=query.copy();points[:,0]=time
        result=oracle.source(points)
        if first is None:first=result
        assert oracle.cache_info()["time_entries"]<=8
    assert oracle.cache_info()["time_evictions"]==11
    query[:,0]=0.
    np.testing.assert_allclose(_numpy(oracle.source(query)),_numpy(first),rtol=decoder.rtol,atol=decoder.atol)


def test_all_unique_times_fall_back_in_bounded_blocks(decoder):
    query=np.tile(_queries("sparse")[1],(65,1))
    query[:,0]=np.linspace(.001,.999,len(query))
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    result=oracle.source(query)
    info=oracle.cache_info()
    assert info["macro_fallback_blocks"]==1 and info["time_entries"]==0
    np.testing.assert_allclose(_numpy(result),_reference(decoder,query),rtol=decoder.rtol,atol=decoder.atol)


def test_constructor_owns_immutable_copy_of_latent_and_condition(decoder):
    latent=decoder.latents[1].copy();condition=decoder.conditions[1].copy()
    oracle=StructuredOracle(decoder.model,latent,condition)
    query=_queries("sparse")
    expected=oracle.source(query)
    latent.fill(0.);condition[1]=.8
    np.testing.assert_array_equal(_numpy(oracle.source(query)),_numpy(expected))


@pytest.mark.parametrize("zero", [False,True])
def test_floor_ground_state_and_nonunit_latent_scale(tiny,zero):
    cfg=replace(tiny.model.config,latent_input_scale=3.7,velocity_floor_mixture=.13)
    model=tiny.api.build_deterministic_direct_model(cfg,device="cpu")
    latent=tiny.latents[0 if zero else 1]
    oracle=StructuredOracle(model,latent,tiny.conditions[1])
    query=_queries("sparse")
    analytic=importlib.import_module("agent_verifier.vpfp.analytic_source")
    expected=analytic.manufacture_standard_vpfp_source_analytic(model,query,latent,equation_condition=tiny.conditions[1])["source"]
    np.testing.assert_allclose(_numpy(oracle.source(query)),_numpy(expected),rtol=2e-10,atol=2e-11)
    query[:,0]=0.
    for field in ("rho","g"):
        expected=model.fields(query,latent,equation_condition=tiny.conditions[1])[field]
        np.testing.assert_allclose(_numpy(oracle.initial(query,field)),_numpy(expected),rtol=2e-10,atol=2e-11)


def test_chebyshev_distinct_orders(tiny):
    cfg=replace(tiny.model.config,structured_time_basis="global_chebyshev_v1",
                structured_rho_time_order=2,structured_local_time_order=3)
    cfg=replace(cfg,latent_dim=cfg.structured_latent_layout()["total_latent_dim"])
    model=tiny.api.build_deterministic_direct_model(cfg,device="cpu")
    latent=np.random.default_rng(718).normal(scale=.01,size=cfg.latent_dim)
    oracle=StructuredOracle(model,latent,tiny.conditions[1])
    query=_queries("cartesian")
    expected=basic.evaluate_source_basic(model,query,latent,equation_condition=tiny.conditions[1])
    np.testing.assert_allclose(_numpy(oracle.source(query)),_numpy(expected),rtol=2e-10,atol=2e-11)
    query[:,0]=0.
    for field in ("rho","g"):
        expected=model.fields(query,latent,equation_condition=tiny.conditions[1])[field]
        np.testing.assert_allclose(_numpy(oracle.initial(query,field)),_numpy(expected),rtol=2e-10,atol=2e-11)


def test_independent_coordinate_autograd_residual(tiny):
    torch,model=tiny.api.torch,tiny.model
    points=np.array([[.133,.21,.34,.41,-.53],[.619,.43,.11,-.7,.2]])
    query=torch.tensor(points,dtype=torch.float64,requires_grad=True)
    fields=model.fields(query,tiny.latents[1],equation_condition=tiny.conditions[1])
    gradient=torch.autograd.grad(fields["f"].sum(),query,create_graph=True)[0]
    vx2=torch.autograd.grad(gradient[:,3].sum(),query,retain_graph=True)[0][:,3]
    vy2=torch.autograd.grad(gradient[:,4].sum(),query)[0][:,4]
    nu,temp=map(float,tiny.conditions[1])
    collision=2*fields["f"]+query[:,3]*gradient[:,3]+query[:,4]*gradient[:,4]+temp*(vx2+vy2)
    expected=gradient[:,0]+(query[:,3:5]*gradient[:,1:3]).sum(-1)+(fields["electric"]*gradient[:,3:5]).sum(-1)-nu*collision
    oracle=StructuredOracle(model,tiny.latents[1],tiny.conditions[1])
    np.testing.assert_allclose(_numpy(oracle.source(points)),_numpy(expected),rtol=2e-10,atol=2e-11)


def test_no_online_tensor_scalar_or_finite_checks(decoder,monkeypatch):
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    def forbidden(*args,**kwargs):raise AssertionError("online Tensor scalar extraction")
    monkeypatch.setattr(decoder.model.torch.Tensor,"__bool__",forbidden)
    monkeypatch.setattr(decoder.model.torch.Tensor,"item",forbidden)
    monkeypatch.setattr(decoder.model.torch,"isfinite",forbidden)
    query=_queries("cartesian")
    assert oracle.source(query).shape==(len(query),)
    query[:,0]=0.
    for field in ("rho","g"):
        assert oracle.initial(query,field).shape==(len(query),)


def test_nonunit_time_domain_retains_physical_time_derivatives(tiny):
    cfg=replace(tiny.model.config,time_domain=(0.,3.))
    model=tiny.api.build_deterministic_direct_model(cfg,device="cpu")
    latent=tiny.latents[1].copy()
    # Random synthetic carrier has nonzero stored slopes; include endpoints
    # and each interior side so a missing 1/duration factor cannot pass.
    query=_queries("sparse")
    query[:,0]*=3.
    analytic=importlib.import_module("agent_verifier.vpfp.analytic_source")
    expected=analytic.manufacture_standard_vpfp_source_analytic(
        model,query,latent,equation_condition=tiny.conditions[1])["source"]
    actual=StructuredOracle(model,latent,tiny.conditions[1]).source(query)
    np.testing.assert_allclose(_numpy(actual),_numpy(expected),rtol=2e-10,atol=2e-11)


def test_initial_source_velocity_cache_order_never_drops_derivatives(decoder):
    query=_queries("cartesian")
    initial=query.copy();initial[:,0]=0.
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    before=oracle.initial(initial,"g")
    assert len(oracle._velocity[2])==1
    source=oracle.source(query)
    assert len(oracle._velocity[2])==5
    after=oracle.initial(initial,"g")
    assert len(oracle._velocity[2])==5
    np.testing.assert_array_equal(_numpy(before),_numpy(after))
    np.testing.assert_allclose(_numpy(source),_reference(decoder,query),rtol=decoder.rtol,atol=decoder.atol)


def test_projection_batch_limit_and_transition_to_sparse_time_fallback(decoder):
    query=np.tile(_queries("sparse")[1],(32,1))
    query[:,0]=np.linspace(.001,.999,len(query))
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    actual=oracle.source(query)
    info=oracle.cache_info()
    assert info["maximum_projection_batch"]==8 and info["time_entries"]==8
    assert info["macro_fallback_blocks"]==0
    np.testing.assert_allclose(_numpy(actual),_reference(decoder,query),rtol=decoder.rtol,atol=decoder.atol)
    query=np.vstack((query,query[-1]));query[-1,0]=.9991
    oracle.source(query)
    assert oracle.cache_info()["macro_fallback_blocks"]==1


@pytest.mark.parametrize("degree,spatial",[(18,1),(1,6)])
def test_actual_mode_axis_sizes_with_small_synthetic_carrier(decoder,degree,spatial):
    cfg=replace(decoder.model.config,hermite_total_degree=degree,
        structured_local_channels=(degree+1)*(degree+2)//2,
        structured_local_spatial_mode=spatial,structured_time_control_count=33)
    cfg=replace(cfg,latent_dim=cfg.structured_latent_layout()["total_latent_dim"])
    model=decoder.api.build_deterministic_direct_model(cfg,device="cpu")
    latent=np.random.default_rng(917).normal(scale=.01,size=cfg.latent_dim)
    query=_queries("cartesian")
    analytic=importlib.import_module("agent_verifier.vpfp.analytic_source")
    expected=analytic.manufacture_standard_vpfp_source_analytic(
        model,query,latent,equation_condition=decoder.conditions[1])["source"]
    oracle=StructuredOracle(model,latent,decoder.conditions[1])
    np.testing.assert_allclose(_numpy(oracle.source(query)),_numpy(expected),
                               rtol=decoder.rtol,atol=decoder.atol)


def test_s0_81_velocity_8192_chunk_uses_complete_rows_without_completion(decoder):
    # Exact S0-relevant nondivisibility: 8192 = 101*81 + 11.
    space=np.column_stack((np.full(102,.319),np.linspace(.01,.99,102),np.full(102,.417)))
    v=np.array([(x,y) for x in np.linspace(-2.,2.,9) for y in np.linspace(-2.,2.,9)])
    query=np.concatenate((np.repeat(space,len(v),axis=0),np.tile(v,(len(space),1))),axis=1)[:8192]
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    actual=oracle.source(query)
    info=oracle.cache_info()
    assert info["mixed_cartesian_blocks"]==1
    assert info["cartesian_pairs"]==101*81
    assert info["sparse_pairs"]==11
    assert info["assembled_pairs"]==8192
    np.testing.assert_allclose(_numpy(actual),_reference(decoder,query),rtol=decoder.rtol,atol=decoder.atol)


def test_mixed_missing_patterns_and_duplicates_preserve_request_order(decoder):
    query=_queries("cartesian")
    # Complete rows 0/3/6, arbitrary missing pairs elsewhere, and duplicate
    # rows in both the complete and incomplete sets. Never fill missing pairs.
    keep=[i for i in range(len(query)) if i//5 in (0,3,6) or i%5 in (1,3)]
    indices=keep+[0,0,6,6,6,19]
    np.random.default_rng(919).shuffle(indices)
    query=query[indices]
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    actual=oracle.source(query)
    info=oracle.cache_info()
    assert info["mixed_cartesian_blocks"]==1
    assert info["cartesian_pairs"]==15
    assert info["sparse_pairs"]==11
    assert info["assembled_pairs"]==26<len(query)
    np.testing.assert_allclose(_numpy(actual),_reference(decoder,query),rtol=decoder.rtol,atol=decoder.atol)


def test_duplicate_pairs_do_not_make_incomplete_row_cartesian(decoder):
    query=_queries("cartesian")
    indices=[1,1,1,1,1,2,2,2,2,2,8,8,8,8,9,9,9,9,9]
    oracle=StructuredOracle(decoder.model,decoder.latents[1],decoder.conditions[1])
    actual=oracle.source(query[indices])
    assert oracle.cache_info()["mixed_cartesian_blocks"]==0
    assert oracle.cache_info()["sparse_blocks"]==1
    np.testing.assert_allclose(_numpy(actual),_reference(decoder,query[indices]),rtol=decoder.rtol,atol=decoder.atol)
