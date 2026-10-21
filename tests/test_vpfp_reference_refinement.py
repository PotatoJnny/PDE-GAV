"""Tiny same-equation refinement checks; all numerical execution on compute."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import reference_refinement as rr


@pytest.fixture(autouse=True)
def threads():
    n = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def generation():
    return {"nx":16,"ny":16,"velocity_grid_size":21,"nt":33,"t1":1.,
            "velocity_max":8.,"internal_dt_max":.001,"dtype":"float32"}


def domain_config(period=2*np.pi):
    return {"time_domain":[0.,1.],"equation":{"x_domain":[period,period]}}


def test_domain_exact_float32_coordinate_reconstruction_not_tolerance():
    axis=np.linspace(0,2*np.pi,16,endpoint=False,dtype=np.float32).astype(np.float64)
    period=float(axis[-1]-axis[0]+axis[1]-axis[0])
    assert period==6.2831853330135345
    receipt=rr.reference_domain_receipt(generation(),domain_config(period))
    assert receipt["decoder_x_domain"]==[period,period]
    assert receipt["numerical_x_domain"]==[2*np.pi,2*np.pi]
    assert not receipt["domain_equality_tolerance_used"]
    assert not receipt["numerical_solver_domain_modified"]
    assert not receipt["same_physical_point_claimed_for_native_comparison"]
    assert rr.reference_domain_receipt(generation(),domain_config())["reason"]=="exact_identical_domains"


@pytest.mark.parametrize("change",["one_ulp","physical_domain","time","wrong_precision","wrong_axis_size","generation_domain"])
def test_domain_rejects_unexplained_or_physical_changes(change):
    gen=generation(); cfg=domain_config(6.2831853330135345)
    if change=="one_ulp": cfg["equation"]["x_domain"][0]=float(np.nextafter(cfg["equation"]["x_domain"][0],np.inf))
    elif change=="physical_domain": cfg["equation"]["x_domain"]=[7.,7.]
    elif change=="time": cfg["time_domain"][1]=float(np.nextafter(1.,np.inf))
    elif change=="wrong_precision": gen["dtype"]="float64"
    elif change=="wrong_axis_size": gen["nx"]=17
    else: gen["x_domain"]=[7.,7.]
    with pytest.raises(ValueError): rr.reference_domain_receipt(gen,cfg)


def test_mapped_native_derivatives_keep_physical_units_and_points(monkeypatch):
    receipt=rr.reference_domain_receipt(generation(),domain_config(6.2831853330135345))
    points=np.array([[.1,.4,.7,1.,-1.]])
    seen=[]
    def native(*args):
        seen.append(args[3].copy())
        return {"values":{"rho":[2.],"jx":[3.],"K":[4.],"dt_f":[5.],"dx_f":[6.],"dy_K":[7.],"dvx_f":[8.]}}
    monkeypatch.setattr(rr,"_native_observations",native)
    result=rr._mapped_native_observations(None,None,None,points,None,receipt)
    ratio=receipt["native_query_dx_dnumerical_x"][0]
    np.testing.assert_array_equal(seen[0][:,[0,3,4]],points[:,[0,3,4]])
    np.testing.assert_array_equal(seen[0][:,1:3],points[:,1:3]*ratio)
    assert result["values"]["rho"]==[2.] and result["values"]["dt_f"]==[5.]
    assert result["values"]["dx_f"]==[6.*ratio] and result["values"]["dy_K"]==[7.*ratio]
    assert result["values"]["dvx_f"]==[8.]


def test_two_levels_preserve_domains_and_record_simultaneous_changes():
    levels = rr.default_levels(generation())
    assert [levels[1][k] for k in ("nx","ny","nvx","nvy","nt")]==[24,24,31,31,65]
    assert levels[1]["dtype"]=="float64" and levels[1]["internal_dt_max"]==.0005
    assert levels[0]["v_bounds"]==levels[1]["v_bounds"]==[-8.,8.,-8.,8.]
    assert rr.validate_levels(levels,generation())==levels


@pytest.mark.parametrize("field,value",[("v_bounds",[-12.,12.,-12.,12.]),("t1",2.),
    ("rho_floor",1e-8),("source_interpolation","nearest"),("dtype","float32"),("nx",16)])
def test_refinement_rejects_changed_problem_or_nonrefinement(field,value):
    levels = rr.default_levels(generation()); levels[1][field]=value
    with pytest.raises(ValueError): rr.validate_levels(levels,generation())


def test_selection_is_condition_stratum_first_row_not_feature_search():
    assert rr.select_stratified_rows(np.array([2,0,1,0,2]),max_cases=3)==[1,2,0]
    assert rr.select_stratified_rows(np.array([2,0,1,0,2]))==[1]
    assert rr.select_stratified_rows(np.array([2,0,1,0,2]),rows=[4],max_cases=1)==[4]
    with pytest.raises(ValueError): rr.select_stratified_rows(np.array([0,1]),rows=[0,0],max_cases=2)
    with pytest.raises(ValueError): rr.select_stratified_rows(np.array([0,1]),max_cases=9)


def tiny_level():
    return {**rr.default_levels(generation())[0],"nx":4,"ny":4,"nvx":5,"nvy":5,
        "nt":3,"t1":1.,"v_bounds":[-2.,2.,-2.,2.],"dtype":"float64"}


def test_raw_multilinear_interpolation_physical_units_no_extrapolation():
    level = tiny_level()
    t,x,y,vx,vy = torch.meshgrid(torch.linspace(0,1,3,dtype=torch.float64),
        torch.arange(4,dtype=torch.float64),torch.arange(4,dtype=torch.float64),
        torch.linspace(-2,2,5,dtype=torch.float64),torch.linspace(-2,2,5,dtype=torch.float64),indexing="ij")
    f = 2+3*t+.4*vx-.2*vy
    q = np.array([[.25,.2,.3,.2,-.4],[.5,.7,1.3,.1,.4],[1.,6.4,-.2,-.3,-.1]])
    actual = rr._interpolate(f,level,q)
    expected = 2+3*q[:,0]+.4*q[:,3]-.2*q[:,4]
    np.testing.assert_allclose(actual.numpy(),expected,rtol=2e-15,atol=2e-15)
    # Grid-step one-sided derivatives at actual endpoints and central at middle.
    q[:,0]=[0.,.5,1.]
    jets,_ = rr._point_jets(f,level,q)
    np.testing.assert_allclose(jets.numpy(),np.tile([3.,0.,0.,.4,-.2],(3,1)),rtol=1e-13,atol=2e-14)
    q[0,3]=2.01
    with pytest.raises(ValueError,match="extrapolate"): rr._interpolate(f,level,q)


def test_fd_stencil_never_clamps_material_domain_mismatch():
    level = tiny_level()
    f = torch.ones((3,4,4,5,5),dtype=torch.float64)
    with pytest.raises(ValueError,match="stencil"):
        rr._point_jets(f,level,np.array([[.25,.2,.3,0.,0.]]))


def test_comparison_zero_reference_and_signed_units_are_explicit():
    result = rr.compare_observations({"values":{"f":[0.,0.],"K":[1.,3.]}},
                                    {"values":{"f":[1.,1.],"K":[.5,2.5]}})
    assert result["f"]["relative_rms"] is None
    assert result["f"]["absolute_rms"]==1.
    assert result["K"]["signed_mean"]==-.5
    with pytest.raises(ValueError): rr.compare_observations({"values":{"f":[1.]}},{"values":{"f":[np.nan]}})


def test_real_tiny_solver_same_spec_raw_grid_and_tail_receipts():
    _,modules = rr._modules()
    cfg = replace(modules["high_fidelity"].VPFPHighFidelityConfig(),
        max_spatial_mode=1,rho_mode_count=1,source_mode_count=1,
        g0_component_range=(1,1),g0_weight_mode_count=1,g0_parameter_mode_count=1,
        collision_nu_range=(.03,.03),temperature_range=(1.,1.))
    spec = modules["high_fidelity"].sample_continuous_equation_spec(seed=123,config=cfg,defer_rho_log_partition=True)
    first = {**tiny_level(),"t1":.001,"internal_dt_max":.0005}
    second = {**first,"name":"refined_same_domain","nx":6,"ny":6,"nvx":7,"nvy":7,"nt":5,"internal_dt_max":.00025}
    observations,meta = [],[]
    q = np.array([[0.,.2,.3,0.,0.],[.0005,.5,.7,0.,0.],[.001,.1,1.3,0.,0.]])
    for level in (first,second):
        solution,metadata = rr._solve_level(spec,level,modules,"cpu")
        observations.append(rr._raw_observations(solution,level,q,tail_width=.5))
        meta.append(metadata)
        assert torch.isfinite(solution["f"]).all()
        assert not metadata["projected_back_to_N14"]
        assert "repair_mass" in metadata["solver_diagnostics"]
        assert not observations[-1]["tail"]["outside_domain_tail_error_bounded"]
    assert meta[0]["same_unrounded_equation_spec_sha256"]==meta[1]["same_unrounded_equation_spec_sha256"]
    result = rr.compare_observations(observations[0],observations[1])
    assert set(("f","rho","jx","jy","K","dt_f","dvx_f","dt_K")).issubset(result)
    json.dumps(result,allow_nan=False)


def test_native_projection_observations_use_complete_stored_state():
    from tests.test_vpfp_moving_frame import tiny,points
    from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import lift_value_controls
    base,value,_,z,condition = tiny(dtype="float64")
    _,modules = rr._modules()
    result = rr._native_observations(base,lift_value_controls(z,value)[0],condition,points().numpy(),modules)
    assert result["macro_velocity_scope"].startswith("full R2")
    np.testing.assert_allclose(result["values"]["jx"],0.,atol=1e-14)
    np.testing.assert_allclose(result["values"]["K"],float(condition[1]),rtol=2e-14)
    json.dumps(result,allow_nan=False)


def test_entry_requires_allocation_and_fresh_output(tmp_path,monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(RuntimeError,match="Slurm"):
        rr.refine_reference_samples(tmp_path,"calibration",expected_manifest_sha256="a"*64,output=tmp_path/"out")
    monkeypatch.setenv("SLURM_JOB_ID","synthetic"); monkeypatch.setenv("SLURM_TMPDIR",str(tmp_path))
    with pytest.raises(ValueError,match="fresh"):
        rr.refine_reference_samples(tmp_path,"calibration",expected_manifest_sha256="a"*64,output=tmp_path)


def test_refined_failure_keeps_coarse_result_plan_seed_and_unresolved(tmp_path,monkeypatch):
    from pde_agent_verifier_solver.plugins.vpfp import diffusion_data as data
    corpus=tmp_path/"corpus"; corpus.mkdir()
    (corpus/"manifest.json").write_text('{"synthetic":true}')
    (corpus/"seed-plan.json").write_text('{"synthetic":true}')
    arrays={"coefficients":np.zeros((1,2),dtype=np.float32),"conditions":np.array([[.03,1.]],dtype=np.float32),
            "equation_seeds":np.array([19],dtype=np.int64),"case_ids":np.array(["synthetic:19"]),
            "stratum_ids":np.array([0],dtype=np.int64)}
    files={}
    for name,array in arrays.items():
        path=corpus/(name+".npy"); np.save(path,array)
        files[name]={"path":path.name,"sha256":rr._file_sha(path)}
    fixture_config=domain_config()
    base=corpus/"base.pt"; torch.save({"decoder_config":fixture_config},base)
    cfg=SimpleNamespace(time_domain=(0.,1.),equation=SimpleNamespace(x_domain=(2*np.pi,2*np.pi)),
                        to_dict=lambda:fixture_config)
    manifest={"manifest_sha256":"a"*64,"seed_plan":{"path":"seed-plan.json","sha256":rr._file_sha(corpus/"seed-plan.json")},
        "splits":{"calibration":{"files":files}},"checkpoints":{"base_decoder":{"path":str(base),"sha256":rr._file_sha(base)}},
        "provenance":{"generation_config":generation(),"equation_config":{},
                      "decoder_config":fixture_config,"physical_module_sha256":{}}}
    monkeypatch.setattr(data,"read_manifest",lambda *a,**k:manifest)
    def checked(root,item,verify): return root/item["path"]
    monkeypatch.setattr(data,"_checked_file",checked)
    physical=SimpleNamespace(__file__=__file__,RICH_EQUATION_FAMILY="rich",
        VPFPHighFidelityConfig=SimpleNamespace(from_dict=lambda x:SimpleNamespace(equation_family="rich")),
        sample_continuous_equation_spec=lambda **k:{"seed":k["seed"],"collision_nu":.03,"temperature":1.})
    hermite=SimpleNamespace(__file__=__file__,VPFPContinuousHermiteConfig=SimpleNamespace(from_dict=lambda x:cfg))
    monkeypatch.setattr(rr,"_modules",lambda:(SimpleNamespace(build_deterministic_direct_model=lambda *a,**k:object()),
        {"high_fidelity":physical,"continuous_hermite":hermite}))
    def solve(spec,level,modules,device):
        if level["name"]=="refined_same_domain": raise FloatingPointError("synthetic refined failure")
        return {},{"same_unrounded_equation_spec_sha256":rr._sha_json(spec),"level":level,"solver_diagnostics":{}}
    monkeypatch.setattr(rr,"_solve_level",solve)
    monkeypatch.setattr(rr,"_raw_observations",lambda *a,**k:{"values":{"f":[1.]}})
    monkeypatch.setenv("SLURM_JOB_ID","synthetic"); monkeypatch.setenv("SLURM_TMPDIR",str(tmp_path))
    output=tmp_path/"out"
    result=rr.refine_reference_samples(corpus,"calibration",expected_manifest_sha256="a"*64,output=output,device="cpu")
    assert not result["execution_passed"] and result["numerical_convergence_unresolved"]
    assert result["new_pde_solves"]==1 and result["new_pde_solve_attempts"]==2
    assert not result["all_reference_rows_refined"]
    assert result["cases"][0]["equation_seed"]==19
    assert result["cases"][0]["levels"][0]["execution_passed"]
    assert not result["cases"][0]["levels"][1]["execution_passed"]
    assert "synthetic refined failure" in result["cases"][0]["failure"]["message"]
    assert (output/"frozen-plan.json").exists() and (output/"root-0-replay_coarse.json").exists()
    assert json.loads((output/"result.json").read_text())==result
    original_coarse=(output/"root-0-replay_coarse.json").read_bytes()
    calls=[]
    def recovered(spec,level,modules,device):
        calls.append(level["name"])
        return {},{"same_unrounded_equation_spec_sha256":rr._sha_json(spec),"level":level,"solver_diagnostics":{}}
    monkeypatch.setattr(rr,"_solve_level",recovered)
    monkeypatch.setattr(rr,"_native_observations",lambda *a,**k:{"values":{"f":[1.]}})
    again=rr.refine_reference_samples(corpus,"calibration",expected_manifest_sha256="a"*64,
        output=output,device="cpu",resume=True)
    assert again["execution_passed"] and calls==["refined_same_domain"]
    assert again["new_pde_solves"]==2 and again["new_pde_solve_attempts"]==3
    assert (output/"root-0-replay_coarse.json").read_bytes()==original_coarse
    (output/"root-0-replay_coarse.json").unlink()
    with pytest.raises(ValueError,match="completed refinement level report changed"):
        rr.refine_reference_samples(corpus,"calibration",expected_manifest_sha256="a"*64,
            output=output,device="cpu",resume=True)
