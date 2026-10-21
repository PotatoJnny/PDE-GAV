"""Synthetic tiny fixtures; no real checkpoints or numerical tests on login."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_vpfp_moving_frame import tiny, constant_macro, points
from pde_agent_verifier_solver.plugins.vpfp import moving_frame_backend as backend
from pde_agent_verifier_solver.plugins.vpfp import moving_frame_binding as binding
from pde_agent_verifier_solver.plugins.vpfp.moving_frame import MovingFrameDecoder
from pde_agent_verifier_solver.sdk import GenerationRequest, GenerationKind, GenerationPurpose, GeneratorMode, CaseVisibility, AnswerQuery

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def setup():
    base, value, cfg, z, condition = tiny()
    return MovingFrameDecoder(base,cfg,value_config=value), {"m":constant_macro(cfg),"z":z}, condition


def test_A_MMS_matches_analytic_shifted_anisotropic_Gaussian():
    decoder, carrier, condition = setup()
    q = points().requires_grad_()
    result = backend.manufacture_moving_frame_source(decoder,q,carrier,condition,create_graph=True)
    covariance = torch.tensor([[1.3,.2],[.2,.7]],dtype=torch.float64)
    inverse = torch.linalg.inv(covariance)
    displacement = q[:,3:5]-torch.tensor([.4,-.3],dtype=torch.float64)
    factor = displacement@inverse
    rho = decoder.base_decoder.config.equation.background_density
    f = rho/(2*torch.pi*torch.linalg.det(covariance).sqrt())*torch.exp(-.5*(displacement*factor).sum(1))
    lap = (factor.square().sum(1)-inverse.trace())*f
    collision = 2*f-(q[:,3:5]*factor).sum(1)*f+condition[1]*lap
    torch.testing.assert_close(result["f"],f,rtol=2e-12,atol=2e-13)
    torch.testing.assert_close(result["source"],-condition[0]*collision,rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(result["f_velocity_laplacian"],lap,rtol=3e-11,atol=3e-12)
    assert result["points"] is q
    actual_gradient = torch.autograd.grad(result["source"].sum(),q)[0]
    assert torch.isfinite(actual_gradient).all()
    analytic_gradient = torch.autograd.grad((-condition[0]*collision).sum(),q)[0]
    torch.testing.assert_close(actual_gradient,analytic_gradient,rtol=5e-10,atol=3e-12)


@pytest.mark.parametrize("dtype",["float32","float64"])
@pytest.mark.parametrize("per_query",[False,True])
def test_A_sparse_zero_rho_groups_preserve_native_forward_bits(dtype,per_query):
    base,value,cfg,z,condition = tiny(dtype=dtype)
    decoder = MovingFrameDecoder(base,cfg,value_config=value)
    # One nonzero, time-varying pair; every other pair is identically zero.
    z[:,:value.rho_dimension].reshape(1,cfg.control_count,-1)[0,:,0] = torch.tensor(
        [.08,.11,.17],dtype=z.dtype)
    q = points().to(z)
    if per_query:
        z = z.expand(len(q),-1).clone()
    full = binding.lift_value_controls(z,value)
    old = base.fields(q,full,equation_condition=condition)
    new = decoder.source_fields(q,z,condition)
    for name in old:
        assert torch.equal(old[name],new[name]), name


def test_sparse_rho_source_gradient_matches_independent_coordinate_difference():
    decoder,carrier,condition = setup()
    cfg,value = decoder.macro_config,decoder.value_config
    carrier["z"][:,:value.rho_dimension].reshape(1,cfg.control_count,-1)[0,:,0] = torch.tensor([.08,.11,.17])
    # Avoid knots: this compares the actual within-interval derivative.
    q = points()[1:3].clone().requires_grad_()
    result = backend.manufacture_moving_frame_source(decoder,q,carrier,condition,create_graph=True)
    actual = torch.autograd.grad(result["source"].sum(),q)[0]
    assert torch.isfinite(actual).all()
    estimates = []
    for axis in range(5):
        plus,minus = q.detach().clone(),q.detach().clone()
        plus[:,axis] += 1e-5; minus[:,axis] -= 1e-5
        sp = backend.manufacture_moving_frame_source(decoder,plus,carrier,condition)["source"].detach()
        sm = backend.manufacture_moving_frame_source(decoder,minus,carrier,condition)["source"].detach()
        estimates.append((sp-sm)/(2e-5))
    torch.testing.assert_close(actual,torch.stack(estimates,1),rtol=3e-6,atol=2e-9)


def test_nonconstant_rho_zero_crossing_does_not_invent_source_gradient():
    decoder,carrier,condition = setup()
    cfg,value = decoder.macro_config,decoder.value_config
    carrier["z"][:,:value.rho_dimension].reshape(1,cfg.control_count,-1)[0,:,0] = torch.tensor([-.1,0.,.1])
    q = points()[1:2].clone()
    q[:,0] = 3.5
    # Gauge r/(1+||r||) remains C1; public source value is still defined.
    ordinary = backend.manufacture_moving_frame_source(decoder,q,carrier,condition)
    assert torch.isfinite(ordinary["source"]).all()
    with pytest.raises(ValueError,match="nonconstant rho-group zero crossing"):
        backend.manufacture_moving_frame_source(decoder,q,carrier,condition,create_graph=True)


def test_time_dependent_Gaussian_source_gradient_with_constant_zero_density_modes():
    from pde_agent_verifier_solver.plugins.vpfp.moving_frame import split_macro_controls,pack_macro_controls
    decoder,carrier,condition = setup()
    cfg = decoder.macro_config
    u,local,chol = split_macro_controls(carrier["m"],cfg)
    acceleration = torch.tensor([.07,-.04],dtype=torch.float64)
    offsets = torch.linspace(0,3,cfg.control_count,dtype=torch.float64)
    u = u+acceleration[None,:,None]*offsets[None,None,:]
    carrier["m"] = pack_macro_controls(u,local,chol,cfg)
    q = points()[1:3].clone().requires_grad_()
    actual = backend.manufacture_moving_frame_source(decoder,q,carrier,condition,create_graph=True)
    covariance = torch.tensor([[1.3,.2],[.2,.7]],dtype=torch.float64)
    inverse = torch.linalg.inv(covariance)
    velocity = q.new_tensor([.4,-.3])+acceleration*(q[:,0:1]-2.)
    displacement = q[:,3:5]-velocity
    factor = displacement@inverse
    rho = decoder.base_decoder.config.equation.background_density
    f = rho/(2*torch.pi*torch.linalg.det(covariance).sqrt())*torch.exp(-.5*(displacement*factor).sum(1))
    lap = (factor.square().sum(1)-inverse.trace())*f
    source = (factor*acceleration).sum(1)*f-condition[0]*(2*f-(q[:,3:5]*factor).sum(1)*f+condition[1]*lap)
    torch.testing.assert_close(actual["source"],source,rtol=3e-11,atol=3e-12)
    gradient = torch.autograd.grad(actual["source"].sum(),q)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient,torch.autograd.grad(source.sum(),q)[0],rtol=5e-10,atol=3e-12)


def test_macros_keep_graph_and_density_weighted_raw_second():
    decoder,carrier,condition = setup()
    q = points()[:,:3].clone().requires_grad_()
    result = backend.evaluate_moving_frame_macros(decoder,q,carrier,condition)
    rho,u,c = result["rho"],result["velocity"],result["central_covariance"]
    raw = rho[:,None,None]*(c+u[:,:,None]*u[:,None,:])
    torch.testing.assert_close(result["raw_second_moment"],torch.stack((raw[:,0,0],raw[:,0,1],raw[:,1,1]),1))
    torch.testing.assert_close(result["kinetic_energy"],.5*(result["raw_second_moment"][:,0]+result["raw_second_moment"][:,2]))
    assert torch.isfinite(torch.autograd.grad(result["momentum"].sum(),q)[0]).all()


def test_actual_source_chunk_order_independent():
    decoder,carrier,condition = setup()
    q = points()
    full = backend.manufacture_moving_frame_source(decoder,q,carrier,condition)["source"].detach()
    separate = torch.cat([backend.manufacture_moving_frame_source(decoder,p,carrier,condition)["source"].detach() for p in q.split(1)])
    torch.testing.assert_close(full,separate,rtol=3e-12,atol=3e-13)
    order = torch.tensor([3,1,0,2])
    shuffled = backend.manufacture_moving_frame_source(decoder,q[order],carrier,condition)["source"].detach()
    torch.testing.assert_close(shuffled,full[order],rtol=3e-12,atol=3e-13)


class FixedSampler:
    identity = {"schema":"synthetic_moving_frame_fixture","pde_solves_per_sample":0}
    def __init__(self,carrier,condition):
        self.carrier,self.condition = carrier,condition
    def generate(self,seeds):
        return {"schema":"vpfp_moving_frame_native_answer_batch_v1",
            **{name:value.expand(len(seeds),-1).clone() for name,value in self.carrier.items()},
            "equation_conditions":self.condition[None].expand(len(seeds),-1).clone(),"equation_seeds":seeds}


def request(**kwargs):
    values = dict(round_index=1,count=1,mode=GeneratorMode.OFFLINE,kind=GenerationKind.BASE,
        visibility=CaseVisibility.DEVELOPMENT,seed=11,purpose=GenerationPurpose.DEVELOPMENT,
        settings={"fixed_generator_id":"synthetic-A","fixed_generator_sha256":"a"*64})
    values.update(kwargs)
    return GenerationRequest(**values)


def test_runtime_initial_truth_private_replay_snapshot_and_release(tmp_path):
    decoder,carrier,condition = setup()
    runtime = backend.MovingFrameBackend(decoder,FixedSampler(carrier,condition),
        output_root=tmp_path/"private",public_grid=3,public_times=3,source_chunk=1,field_chunk=2)
    record, = runtime.generate(None,None,request())
    public = record.case.public_data
    assert set(public)=={"equation","question","arrays"}
    assert not any(name in repr(public) for name in ("macro_checkpoint","private-answer.pt","source_decoder_config_sha256"))
    assert public["arrays"]["momentum0"].shape==(3,3,2)
    session = runtime.public_session_factory(record.case)
    assert session.source(points().numpy()).shape==(4,)
    assert session.rho0(np.array([[.1,.2]])).shape==(1,)
    assert session.g0(np.array([[.1,.2,.3,.4]])).shape==(1,)
    assert session.usage_snapshot["evaluated_points"]==6
    query = AnswerQuery.at_points(fields=("rho","momentum_x","momentum_y","kinetic_energy"),
        points=({"t":3.,"x":.2,"y":.3},))
    assert record.truth.answer(query).shape==(1,)
    with pytest.raises(PermissionError):
        runtime.development_solution_session_factory(replace(record.case,visibility=CaseVisibility.HIDDEN))
    altered = deepcopy(public)
    altered["arrays"]["rho0"][0,0]+=1
    with pytest.raises(ValueError,match="snapshot"):
        runtime.public_session_factory(replace(record.case,public_data=altered))
    private = tmp_path/"private"/record.case.case_id/"private-answer.pt"
    saved = torch.load(private,weights_only=True)
    assert saved["sampler_identity_sha256"]==binding.json_sha(saved["sampler_identity"])
    assert saved["sampler_identity"]==runtime.sampler.identity
    reloaded = decoder.carrier_from_payload(saved["carrier"])
    torch.testing.assert_close(decoder.fields(points(),reloaded,equation_condition=saved["condition"])["f"],
        decoder.fields(points(),carrier,equation_condition=condition)["f"],rtol=2e-7,atol=2e-8)
    runtime.release_cases((record.case.case_id,))
    assert private.exists()
    with pytest.raises(ValueError,match="no longer active"):
        session.source(points().numpy())
    with pytest.raises(ValueError,match="reused"):
        runtime.generate(None,None,request())


def test_runtime_rejects_unimplemented_paths_and_bad_queries(tmp_path):
    decoder,carrier,condition = setup()
    runtime = backend.MovingFrameBackend(decoder,FixedSampler(carrier,condition),output_root=tmp_path/"private",public_grid=3)
    with pytest.raises(ValueError,match="BASE"):
        runtime.generate(None,object(),request())
    record, = runtime.generate(None,None,request())
    with pytest.raises(ValueError,match="BASE"):
        runtime.generate(None,None,request(kind=GenerationKind.PERTURBED,anchors=(record.case,)))
    with pytest.raises(ValueError,match="finite"):
        runtime.evaluate_source(record.case.case_id,np.full((1,5),np.nan))
    with pytest.raises(ValueError,match="outside"):
        runtime.evaluate_source(record.case.case_id,np.array([[1.,0.,0.,0.,0.]]))


def smoke_module():
    spec = importlib.util.spec_from_file_location("moving_frame_smoke_test",ROOT/"scripts/run_vpfp_moving_frame_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_Controller_path_is_A_macro_only_with_public_Oracles(tmp_path):
    smoke = smoke_module()
    decoder,carrier,condition = setup()
    bound = binding.binding_from_sampler(decoder,FixedSampler(carrier,condition),output_root=tmp_path/"private",
        regime_id="synthetic-fixture",public_grid=3,public_times=3,source_chunk=2,field_chunk=8)
    agent = smoke.MacroOnlyAgent()
    result = bound.run(smoke.smoke_recipe(decoder.macro_config.time_domain),
        {"moving-frame-macro-agent":agent},local_development=True)
    assert np.isfinite(result.selected.primary_error)
    assert len(result.rounds)==len(result.terminal_round_reports)==1
    runtime = bound.plugin.generator.backend
    assert not runtime._resident_cases and len(runtime.generation_receipts)==3
    assert all(runtime.oracle_calls[key]>=3 for key in ("source","rho0","g0"))
    assert not bound.plugin.generator.capabilities.perturbations
    assert not bound.plugin.generator.capabilities.guidance


def test_smoke_allocation_and_output_guards(tmp_path,monkeypatch):
    smoke = smoke_module()
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(RuntimeError,match="Slurm"):
        smoke.run("unused","0"*64,tmp_path/"out","cpu")
    monkeypatch.setenv("SLURM_JOB_ID","fixture")
    monkeypatch.setenv("SLURM_TMPDIR",str(tmp_path))
    with pytest.raises(ValueError,match="fresh"):
        smoke.run("unused","0"*64,tmp_path,"cpu")


@pytest.fixture
def native_assets(tmp_path):
    from pde_agent_verifier_solver.plugins.vpfp.coherent_diffusion import VPFPCoherentMacroConfig,VPFPCoherentMacroEDM,VPFPMacroConditionalEDM,VPFPCoherentMacroNormalizer
    from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig,lift_value_controls
    from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer
    base,original,_,_,_ = tiny()
    value = VPFPValueEDMConfig(replace(original.full_config,width=16,depth=1,grid_size=5))
    macro = VPFPCoherentMacroConfig(value.full_config,maximum_mode=1,coherent_mode=1)
    rng = torch.Generator().manual_seed(12)
    data = torch.randn(4,value.latent_dimension,generator=rng)
    normal = VPFPTemporalCovarianceNormalizer.fit(value,lift_value_controls(data,value),
        sample_ids=[1,2,3,4],covariance_strength=.5)
    macro_normal = VPFPCoherentMacroNormalizer.fit(macro,torch.randn(4,macro.latent_dimension,generator=rng),sample_ids=[1,2,3,4])
    base_payload = {"decoder_config":base.config.to_dict(),"robust_answer_first_adapter":{
        "unused_neural_parameters_zeroed":True,"continuous_hidden_answer_is_primary":True,"generator_parameterization_public":False}}
    paths = {name:tmp_path/(name+".pt") for name in ("base_decoder","macro_checkpoint","conditional_checkpoint")}
    torch.save(base_payload,paths["base_decoder"])
    graph = {"src/pde_agent_verifier_solver/plugins/vpfp/"+name:binding.file_sha(Path(binding.__file__).with_name(name))
        for name in ("coherent_diffusion.py","structured_diffusion.py","value_control_diffusion.py","temporal_covariance_diffusion.py","noise_geometry.py","noise_training.py")}
    def payload(model,norm):
        return {"schema":model.checkpoint_schema,"config":model.config.to_dict(),"model":model.state_dict(),"normalizer":norm.to_payload(),
            "base_decoder_sha256":binding.file_sha(paths["base_decoder"]),"corpus_manifest_sha256":"c"*64,
            "training":{"step":20000,"presentations":80000,"seed":7,"train_case_ids":["a","b","c","d"],
                "validation_case_ids":["v"],"source_file_sha256":graph,"ablation":{},
                "data_origin":"synthetic_checkpoint_header_fixture_NOT_training_results"}}
    macro_payload = payload(VPFPCoherentMacroEDM(macro),macro_normal)
    torch.save(macro_payload,paths["macro_checkpoint"])
    micro_payload = payload(VPFPMacroConditionalEDM(value,macro),normal)
    micro_payload["training"]["ablation"] = {"macro_config":macro.to_dict(),
        "macro_normalizer_sha256":binding.json_sha(macro_payload["normalizer"]),
        "macro_checkpoint_sha256":binding.file_sha(paths["macro_checkpoint"])}
    torch.save(micro_payload,paths["conditional_checkpoint"])
    manifest = {"schema":binding.MANIFEST_SCHEMA,"regime_id":"synthetic-fixture","sampler_steps":64,
        **{name:{"path":str(path),"sha256":binding.file_sha(path)} for name,path in paths.items()}}
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    return path,manifest,macro_payload,micro_payload


def test_native_loader_models_normalizers_and_shapes(native_assets):
    path,manifest,macro,micro = native_assets
    sampler,decoder = binding.load_moving_frame_sampler(path,binding.file_sha(path),device="cpu")
    assert sampler.macro.config.latent_dimension==decoder.macro_config.macro_dimension
    assert sampler.micro.config.latent_dimension==decoder.value_config.latent_dimension
    assert sampler.identity["pde_solves_per_sample"]==0
    assert not any(p.requires_grad for p in sampler.micro.parameters())
    assert sampler.micro_norm.fit_metadata["covariance_strength"]==.5


@pytest.mark.parametrize("kind",["macro_sha","normalizer_sha","legacy_schema","not_full_budget","wrong_base"])
def test_native_pair_header_rejects_mismatch(native_assets,kind):
    _,manifest,macro,micro = native_assets
    changed = deepcopy(micro)
    if kind=="macro_sha": changed["training"]["ablation"]["macro_checkpoint_sha256"]="a"*64
    if kind=="normalizer_sha": changed["training"]["ablation"]["macro_normalizer_sha256"]="a"*64
    if kind=="legacy_schema": changed["schema"]="vpfp_structured_diffusion_checkpoint_v1"
    if kind=="not_full_budget": changed["training"]["step"]=19999
    if kind=="wrong_base": changed["base_decoder_sha256"]="a"*64
    with pytest.raises(ValueError):
        binding.validate_hierarchical_payloads(macro,changed,
            base_decoder_sha256=manifest["base_decoder"]["sha256"],
            macro_checkpoint_sha256=manifest["macro_checkpoint"]["sha256"])


def test_native_loader_rejects_repinned_fixed_Fourier_tamper(native_assets):
    path,manifest,macro,micro = native_assets
    micro["model"]["macro_transform.basis"][0,0,0] += .1
    target = Path(manifest["conditional_checkpoint"]["path"])
    torch.save(micro,target)
    manifest["conditional_checkpoint"]["sha256"] = binding.file_sha(target)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="immutable"):
        binding.load_moving_frame_sampler(path,binding.file_sha(path),device="cpu")
