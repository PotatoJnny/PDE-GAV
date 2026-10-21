"""Tiny real-decoder tests; no corpus, training, or distribution references."""
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import moving_frame as mf
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import _fourier_basis


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny(degree=1, floor=.12, dtype="float64"):
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    api = _legacy_api()
    legacy = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
    channels = (degree+1)*(degree+2)//2
    cfg = replace(legacy.VPFPContinuousHermiteConfig(), latent_dim=1, latent_input_scale=3.7,
        structured_local_scale=2.4, micro_latent_dim=2, context_dim=4, hidden_width=4, hidden_layers=1,
        rho_max_mode_x=1, rho_max_mode_y=1, rho_positivity_gauge="phase_invariant_group_l1_v2",
        spatial_feature_modes=1, time_feature_modes=1, hermite_total_degree=degree, coefficient_width=4,
        coefficient_blocks=1, structured_spectral_latent=True, structured_kinetic_global_dim=2,
        structured_local_channels=channels, structured_local_spatial_mode=1,
        structured_time_basis="piecewise_cubic_hermite_v1", structured_time_control_count=3,
        structured_direct_coefficients=True, structured_residual_scale=0., time_domain=(2.,5.),
        velocity_floor_mixture=floor, dtype=dtype)
    cfg = replace(cfg, latent_dim=cfg.structured_latent_layout()["total_latent_dim"])
    base = api.build_deterministic_direct_model(cfg, device="cpu")
    value = mf.value_config_from_decoder(base)
    config = mf.MovingFrameMacroConfig(control_count=3, max_mode=1, time_domain=(2.,5.),
                                      spatial_domain_lengths=cfg.equation.x_domain)
    z = torch.zeros((1, value.latent_dimension), dtype=getattr(torch, dtype))
    z[:, value.rho_dimension:].reshape(1, channels, 3, 9)[:, 0, :, 0] = 1.
    return base, value, config, z, torch.tensor([.03, .9], dtype=getattr(torch, dtype))


def constant_macro(config, u=(.4, -.3), covariance=((1.3, .2), (.2, .7))):
    t, m = config.control_count, config.modes
    global_u = torch.tensor(u, dtype=torch.float64)[None, :, None].expand(1,2,t).clone()
    local = torch.zeros((1, 2, t, m-1), dtype=torch.float64)
    chol = torch.zeros((1, 3, t, m), dtype=torch.float64)
    chol[..., 0] = mf.log_cholesky(torch.tensor(covariance, dtype=torch.float64))[None, :, None]
    return mf.pack_macro_controls(global_u, local, chol, config)


def points():
    return torch.tensor([[2.,.1,.3,.2,-.4], [2.37,.8,.1,-.7,.3], [4.6,1.3,1.7,.4,.1],
                         [5.,2.4,.7,-.3,-.2]], dtype=torch.float64)


def test_layout_matches_shared_macro_and_fourier_order():
    from pde_agent_verifier_solver.plugins.vpfp.coherent_diffusion import VPFPCoherentMacroConfig, split_macro
    base, value, cfg, z, cond = tiny()
    root_cfg = VPFPCoherentMacroConfig(value.full_config, maximum_mode=1, coherent_mode=1)
    m = constant_macro(cfg)
    for actual, expected in zip(mf.split_macro_controls(m,cfg),split_macro(m,root_cfg)):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    xy = mf.spatial_grid(cfg, 5, device="cpu")
    torch.testing.assert_close(mf.macro_fourier_basis(xy,cfg).T.reshape(9,5,5),
                               _fourier_basis(1,5,constant=True,normalized=True),atol=2e-15,rtol=2e-15)
    assert cfg.macro_dimension == 5*3*9
    assert mf.MovingFrameMacroConfig.from_dict(cfg.to_dict()) == cfg


def test_clean_constant_pair_recovers_complete_native_density_with_floor():
    base, value, cfg, z, cond = tiny()
    a = z[:, value.rho_dimension:].reshape(1, 3, 3, 9)
    a[:, 1, :, 0], a[:, 2, :, 0] = .4, -.3
    # A nonuniform rho must also stay exactly unchanged.
    z[:, :value.rho_dimension].reshape(1,3,8)[:, :, 0] = .11
    m, report = mf.extract_macro_controls(base,z,cond,cfg,value_config=value,fit_grid=5,chunk_size=7)
    decoder = mf.MovingFrameDecoder(base,cfg,value_config=value)
    q = points()
    original = decoder.source_fields(q,z,cond)
    actual = decoder.fields(q,{"m":m,"z":z},equation_condition=cond)
    for name in ("f","rho","g","phi","electric"):
        torch.testing.assert_close(actual[name],original[name],rtol=2e-13,atol=2e-14)
    assert "hermite_coefficients" not in actual
    assert report["macro_reconstruction_exact"] is False
    assert report["source_covariance_eigenvalue_min"] > 0


@pytest.mark.parametrize("degree", [1,14])
def test_affine_density_quadrature_has_target_moments_and_entire_floor(degree):
    base, value, cfg, z, cond = tiny(degree=degree, floor=.27)
    a = z[:,value.rho_dimension:].reshape(1,value.hermite_channels,3,9)
    a[:, -1, :, 0], a[:, -2, :, 0] = .7, .3
    decoder = mf.MovingFrameDecoder(base,cfg,value_config=value)
    m = constant_macro(cfg)
    q0 = points()[:1]
    source = mf.source_local_moments(decoder.source_fields(q0,z,cond),base)
    target = decoder.macro_at(q0,m,decoder.source_fields(q0,z,cond))
    nodes, weights = np.polynomial.hermite.hermgauss(degree+3)
    mesh = np.stack(np.meshgrid(nodes,nodes,indexing="ij"),-1).reshape(-1,2)
    vsrc = torch.tensor(mesh*math.sqrt(2*float(cond[1])), dtype=torch.float64)
    standardized = torch.linalg.solve_triangular(source["factor"][0],(vsrc-source["mean"][0]).T,upper=False).T
    vtarget = target["mean"]+standardized@target["factor"][0].T
    q = q0.expand(len(vsrc),-1).clone(); q[:,3:5] = vtarget
    actual = decoder.fields(q,{"m":m,"z":z},equation_condition=cond)
    jac = torch.linalg.det(target["factor"])/torch.linalg.det(source["factor"])
    measure = torch.tensor(np.outer(weights,weights).ravel()*np.exp((mesh*mesh).sum(1))*2*float(cond[1]))*jac
    mass = (actual["g"]*measure).sum()
    mean = (actual["g"][:,None]*measure[:,None]*vtarget).sum(0)
    centered = vtarget-target["mean"]
    cov = torch.einsum("n,ni,nj->ij",actual["g"]*measure,centered,centered)
    torch.testing.assert_close(mass,torch.tensor(1.,dtype=torch.float64),rtol=2e-12,atol=2e-13)
    torch.testing.assert_close(mean,target["mean"][0],rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(cov,target["covariance"][0],rtol=2e-12,atol=2e-12)
    assert (actual["f"] >= 0).all()


def test_weighted_global_velocity_is_exact_and_not_plain_dc():
    base,value,cfg,z,cond = tiny()
    z[:,:value.rho_dimension].reshape(1,3,8)[:,:,0] = .2
    m = constant_macro(cfg)
    u, local, chol = mf.split_macro_controls(m,cfg)
    local[:,:, :,0] = torch.tensor([.5,-.8])[:,None]
    m = mf.pack_macro_controls(u,local,chol,cfg)
    decoder = mf.MovingFrameDecoder(base,cfg,value_config=value)
    for count in (9,17):
        xy = mf.spatial_grid(cfg,count,device="cpu",offset=.37)
        q = torch.cat((torch.full((len(xy),1),3.1),xy,torch.zeros_like(xy)),1)
        out = decoder.fields(q,{"m":m,"z":z},equation_condition=cond)
        torch.testing.assert_close(out["momentum"].mean(0)/out["rho"].mean(),u[0,:,0],rtol=2e-14,atol=2e-14)
        assert abs(float(out["velocity"][:,0].mean()-u[0,0,0])) > .01


def test_point_jets_and_batch_chunking_are_independent():
    base,value,cfg,z,cond = tiny()
    z[:,:value.rho_dimension].reshape(1,3,8)[0,:,0] = torch.tensor([.1,.2,.4])
    m = constant_macro(cfg)
    u,local,chol = mf.split_macro_controls(m,cfg)
    u[0,0] = torch.tensor([.2,.5,.4]); local[0,0,:,0] = .3
    m = mf.pack_macro_controls(u,local,chol,cfg).requires_grad_()
    decoder = mf.MovingFrameDecoder(base,cfg,value_config=value)
    q = points()[1:3].clone().requires_grad_()
    def evaluate(p):
        out = decoder.fields(p,{"m":m,"z":z},equation_condition=cond)
        jet = torch.autograd.grad(out["f"].sum(),p,create_graph=True)[0]
        return out,jet
    out,jet = evaluate(q)
    for i in range(2):
        one,onejet = evaluate(q[i:i+1].detach().requires_grad_())
        torch.testing.assert_close(one["f"],out["f"][i:i+1],rtol=2e-13,atol=2e-13)
        torch.testing.assert_close(onejet,jet[i:i+1],rtol=2e-12,atol=2e-12)
    delta = 1e-5
    for axis in (0,1,2,3,4):
        plus,minus = q.detach().clone(),q.detach().clone()
        plus[:,axis]+=delta; minus[:,axis]-=delta
        finite_difference = (evaluate(plus.requires_grad_())[0]["f"]-evaluate(minus.requires_grad_())[0]["f"])/(2*delta)
        torch.testing.assert_close(jet[:,axis],finite_difference,rtol=2e-6,atol=3e-10)
    torch.autograd.grad(jet.square().sum(),m)


def test_native_carrier_identity_copy_device_and_tamper_rejection():
    base,value,cfg,z,cond = tiny()
    decoder = mf.MovingFrameDecoder(base,cfg,value_config=value)
    carrier = {"m":constant_macro(cfg),"z":z}
    payload = decoder.carrier_to_payload(carrier)
    # Native legacy field contains a torch module reference and is not itself
    # pickle/deepcopy-able. Rebuild the deterministic decoder, copy the carrier.
    new_base, _, _, _, _ = tiny()
    loaded = mf.MovingFrameDecoder(new_base,cfg,value_config=value).to("cpu").carrier_from_payload(deepcopy(payload))
    torch.testing.assert_close(decoder.fields(points(),loaded,equation_condition=cond)["f"],
                               decoder.fields(points(),carrier,equation_condition=cond)["f"])
    payload["identity"]["floor"] = "only_amplitude"
    with pytest.raises(ValueError,match="identity"):
        decoder.carrier_from_payload(payload)
    carrier["m"][0,0] = torch.nan
    with pytest.raises(ValueError,match="finite"):
        decoder.carrier_to_payload(carrier)


@pytest.mark.parametrize("bad",[torch.tensor([[1.,2.],[2.,1.]]),torch.tensor([[1.,0.],[.1,1.]])])
def test_spd_failures_are_not_clipped(bad):
    with pytest.raises(ValueError):
        mf.log_cholesky(bad.double())


def test_failure_contracts_and_time_units():
    base,value,cfg,z,cond = tiny()
    with pytest.raises(ValueError,match="aliases"):
        mf.spatial_grid(cfg,2,device="cpu")
    for x in (-1000.,1000.):
        with pytest.raises(ValueError):
            mf.cholesky_from_log(torch.tensor([[x,0.,x]],dtype=torch.float64))
    controls = torch.tensor([[[[2.],[3.5],[5.]]]],dtype=torch.float64)
    t = torch.tensor([2.2,4.8],dtype=torch.float64,requires_grad=True)
    out = mf.interpolate_controls(controls,t,cfg)
    torch.testing.assert_close(out[:,0,0],t)
    torch.testing.assert_close(torch.autograd.grad(out.sum(),t)[0],torch.ones_like(t))
    with pytest.raises(ValueError,match="outside"):
        mf.interpolate_controls(controls,torch.tensor([1.]),cfg)


def preflight_module():
    path = Path(__file__).resolve().parents[1]/"scripts/diagnose_vpfp_moving_frame.py"
    spec = importlib.util.spec_from_file_location("moving_frame_preflight_test",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_macro_restriction_preserves_shared_fourier_coordinates():
    base,value,cfg,z,cond = tiny()
    m = constant_macro(cfg)
    low = replace(cfg,max_mode=0)
    out = mf.restrict_macro_controls(m,cfg,low)
    assert out.shape == (1,5*3)
    torch.testing.assert_close(mf.split_macro_controls(out,low)[0],mf.split_macro_controls(m,cfg)[0])
    with pytest.raises(ValueError,match="lower"):
        mf.restrict_macro_controls(out,low,cfg)


@pytest.mark.parametrize("dtype",["float64","float32"])
def test_preflight_actual_tiny_mechanics_in_both_native_precisions(dtype):
    driver = preflight_module()
    base,value,cfg,z,cond = tiny(dtype=dtype)
    result = driver.mechanical_check(base,z,cond,value,cfg)
    assert result["passed"] is True
    assert result["complete_floor_transformed"] is True
    assert len(result["quadrature"]) == 2


def test_preflight_actual_clean_case_separates_mechanics_from_compression():
    driver = preflight_module()
    base,value,cfg,z,cond = tiny()
    result = driver.diagnose_case(base,z,cond,modes=(0,1),fit_grids=(3,5),evaluation_grids=(5,),
        samples=[{"kind":"node","tau":0.},{"kind":"midpoint","tau":.25},{"kind":"quarter","tau":.625}],chunk_size=7)
    assert result["execution_passed"] and result["numerical_mechanics_passed"]
    assert len(result["compression_diagnostics"]) == 12
    assert len(result["derivative_probe_diagnostics"]) == 12
    assert {row["time_kind"] for row in result["compression_diagnostics"]} == {"node","midpoint","quarter"}
    for row in result["compression_diagnostics"]:
        for metric in row["metrics"].values():
            # Compression curves also include the real FP32 macro cache cast.
            assert metric["absolute_rms"] < 2e-7
        assert row["macro_coordinate_storage_dtype"] == "torch.float32"
        assert row["density_weighted_global_U_grid_check_maximum_absolute"] < 2e-13
    for row in result["derivative_probe_diagnostics"]:
        assert row["physical_coordinate_units"] and not row["scientific_distribution_audit"]
        assert set(row["metrics"]) == {f"{axis}_{field}" for axis in driver.JET_AXES for field in driver.JET_FIELDS}
        assert all(metric["absolute_rms"] < 2e-7 for metric in row["metrics"].values())
        assert np.array(row["A_derivative_values"]).shape == (2,9,3)


def test_preflight_no_allocation_or_fresh_output_guard(tmp_path,monkeypatch):
    driver = preflight_module()
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(RuntimeError,match="Slurm"):
        driver.run(tmp_path/"source",tmp_path/"config",tmp_path/"output","cpu")
    monkeypatch.setenv("SLURM_JOB_ID","tiny-test-no-real-data")
    with pytest.raises(ValueError,match="fresh"):
        driver.run(tmp_path/"source",tmp_path/"config",tmp_path,"cpu")
    with pytest.raises(ValueError,match="fresh"):
        driver.run(tmp_path/"source",tmp_path/"config",tmp_path/"source/output","cpu")


def test_preflight_source_failure_keeps_result(tmp_path,monkeypatch):
    driver = preflight_module()
    monkeypatch.setenv("SLURM_JOB_ID","tiny-test-no-real-data")
    def fail(*args):
        raise ValueError("synthetic invalid manifest")
    monkeypatch.setattr(driver.source,"load_development",fail)
    out = tmp_path/"out"
    result = driver.run(tmp_path/"source",tmp_path/"config",out,"cpu")
    assert result["execution_passed"] is False and result["numerical_mechanics_passed"] is False
    assert result["failed_stage"] == "source_binding" and (out/"result.json").exists()


def test_preflight_time_probes_and_null_scale_do_not_claim_admission():
    driver = preflight_module()
    rows = driver.time_samples(33)
    assert len(rows) == 20 and {row["kind"] for row in rows} == {"node","midpoint","quarter"}
    assert min(row["tau"] for row in rows)==0 and max(row["tau"] for row in rows)==1
    summary = driver._summary(torch.ones(3),torch.zeros(3))
    assert summary["relative_rms"] is None and summary["absolute_rms"]==1
    derivatives = driver.derivative_time_samples(rows)
    assert len(derivatives)==9
    assert [r["tau"] for r in derivatives if r["kind"]=="node"] == [0.,.5,1.]


def test_derivative_probes_do_not_replace_compressed_target_by_source_moments():
    driver = preflight_module()
    base,value,cfg,z,cond = tiny()
    decoder = mf.MovingFrameDecoder(base,cfg,value_config=value)
    macro = constant_macro(cfg)
    u,local,chol = mf.split_macro_controls(macro,cfg)
    # Time-varying explicit U has physical derivative 0.2; old source is static.
    u[0,0] = torch.tensor([0.,.3,.6])
    macro = mf.pack_macro_controls(u,local,chol,cfg)
    q = points()[1:3]
    old = driver._probe_jets(decoder,z,cond,q)
    new = driver._probe_jets(decoder,z,cond,q,macro)
    ux = driver.JET_FIELDS.index("ux")
    jx = driver.JET_FIELDS.index("jx")
    torch.testing.assert_close(old[:,ux,0],torch.zeros(2,dtype=torch.float64),atol=1e-15,rtol=0)
    torch.testing.assert_close(new[:,ux,0],torch.full((2,),.2,dtype=torch.float64),atol=2e-8,rtol=2e-8)
    torch.testing.assert_close(new[:,jx,0],new[:,ux,0]*base.config.equation.background_density)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="tiny device-copy test runs in compute allocation")
def test_cuda_query_accepts_cpu_carrier_and_preserves_query_derivatives():
    base,value,cfg,z,cond = tiny()
    decoder = mf.MovingFrameDecoder(base,cfg,value_config=value)
    carrier = {"m":constant_macro(cfg),"z":z}
    expected = decoder.fields(points(),carrier,equation_condition=cond)["f"]
    decoder.to("cuda")
    query = points().requires_grad_()
    output = decoder.fields(query,carrier,equation_condition=cond)["f"]
    torch.testing.assert_close(output.cpu(),expected,rtol=3e-12,atol=3e-12)
    assert torch.isfinite(torch.autograd.grad(output.sum(),query)[0]).all()
