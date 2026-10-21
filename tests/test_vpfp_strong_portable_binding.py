"""Explicit codec/proof integration; all inputs here are synthetic fixtures."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import vpfp_strong_portable_binding as binding
import run_vpfp_strong_solver_guidance_round as round_driver
import vpfp_strong_guidance_rollouts as rollouts
from pde_agent_verifier_solver.plugins.vpfp import structured_diffusion as legacy
from pde_agent_verifier_solver.plugins.vpfp import portable_initial_conditioned as portable
from pde_agent_verifier_solver.plugins.vpfp import solver_error_guidance as guidance
from tests.test_vpfp_strong_guidance_rollouts import _fixture_assets
from tests.test_vpfp_portable_initial_conditioned import write_assets


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def proof_fixture(tmp_path, *, config=None, codec_path=None, compact=None):
    """Synthetic receipt with the actual exporter schema, never an admission."""
    config = deepcopy(config) if config is not None else {
        "base_checkpoint": str(tmp_path/"base.pt"), "base_checkpoint_sha256": "a"*64}
    if codec_path is None:
        codec_path = tmp_path/"codec.pt"
        codec_path.write_bytes(b"synthetic codec bytes; pure receipt tests do not load tensors")
    codec_sha = sha256(codec_path.read_bytes()).hexdigest()
    if compact is None:
        identity = {"schema": binding.SAMPLER_SCHEMA, "source_checkpoint_sha256": config["base_checkpoint_sha256"],
            "model_state_sha256": "b"*64, "original_normalizer_sha256": "c"*64,
            "candidate_normalizer_sha256": "d"*64, "buffer_state_sha256": "e"*64}
        compact = {"schema": binding.CODEC_SCHEMA, "sha256": "f"*64, "identity": identity,
            "buffer_manifest": {f"synthetic-{i}": {"dtype": "torch.float64", "shape": [1], "sha256": "0"*64} for i in range(15)}}
    identity = compact["identity"]
    sampler_identity = {"checkpoint_schema": binding.SAMPLER_SCHEMA, "checkpoint_sha256": config["base_checkpoint_sha256"],
        "portable_bundle_sha256": codec_sha, "portable_content_sha256": compact["sha256"], "portable_codec": identity}
    proof = {"schema": binding.CONVERSION_SCHEMA, "stage": "complete", "source_revision": "9"*40,
        "execution_passed": True, "inputs_final_verified": True, "outputs_final_verified": True, "source_final_verified": True,
        "old_checkpoint_restored": False, "guidance_ready": False, "guidance_enabled": False,
        "scientific_admission": False, "base_weights_modified": False, "new_samples": 0, "pde_solves": 0,
        "failures": [], "tests": {"returncode": 0},
        "replay_markers": {"completed": 128, "all128_bitwise_equal": False, "all128_every_value_within_one_ulp": False},
        "candidate": {"path": str(codec_path), "sha256": codec_sha, "content_sha256": compact["sha256"], "identity": identity},
        "input_sha256": {config["base_checkpoint"]: config["base_checkpoint_sha256"]},
        "output_sha256": {str(codec_path): codec_sha},
        "load_validation": {"execution_passed": True, "eigh_eigvalsh_disabled": True,
            "all_buffer_bytes_unchanged": True, "all_original_model_bytes_unchanged": True,
            "loaded_buffer_count": 15, "samples_generated": 0, "prior_runtime_loaded": False,
            "compact_normalizer_identity": compact, "compact_normalizer_sha256": binding.json_sha(compact),
            "sampler_identity": sampler_identity}}
    path = tmp_path/"conversion-result.json"
    path.write_text(json.dumps(proof))
    config["portable_codec"] = {"schema": binding.SCHEMA, "path": str(codec_path), "sha256": codec_sha,
        "conversion_result": str(path), "conversion_result_sha256": sha256(path.read_bytes()).hexdigest()}
    return config, proof


def update_proof(config, proof):
    path = Path(config["portable_codec"]["conversion_result"])
    path.write_text(json.dumps(proof))
    config["portable_codec"]["conversion_result_sha256"] = sha256(path.read_bytes()).hexdigest()


def test_default_absent_is_legacy_but_null_and_partial_are_errors(tmp_path):
    assert binding.portable_config({}) is None
    assert binding.portable_roles({}) == {}
    assert binding.validate_portable_proof({}) is None
    for item in (None, {}, {"schema": binding.SCHEMA}, {"schema": "wrong"}):
        with pytest.raises(ValueError): binding.portable_config({"portable_codec": item})
    config, _ = proof_fixture(tmp_path)
    assert binding.validate_portable_proof(config)["scientific_admission"] is False
    config["portable_codec"]["extra"] = True
    with pytest.raises(ValueError): binding.portable_config(config)


@pytest.mark.parametrize("change", ["schema", "incomplete", "failed_tests", "bool_tests", "not128", "no_final",
    "sampled", "weights_changed", "bad_codec", "wrong_base", "wrong_compact", "changed_sampler", "not15", "claim_ready", "missing_output"])
def test_proof_rejects_self_consistent_bad_receipts(tmp_path, change):
    config, proof = proof_fixture(tmp_path)
    if change == "schema": proof["schema"] = "wrong"
    elif change == "incomplete": proof["execution_passed"] = False
    elif change == "failed_tests": proof["tests"]["returncode"] = 1
    elif change == "bool_tests": proof["tests"]["returncode"] = False
    elif change == "not128": proof["replay_markers"]["completed"] = 127
    elif change == "no_final": proof["inputs_final_verified"] = False
    elif change == "sampled": proof["new_samples"] = 1
    elif change == "weights_changed": proof["load_validation"]["all_original_model_bytes_unchanged"] = False
    elif change == "bad_codec": proof["candidate"]["sha256"] = "8"*64
    elif change == "wrong_base": proof["candidate"]["identity"]["source_checkpoint_sha256"] = "8"*64
    elif change == "wrong_compact": proof["load_validation"]["compact_normalizer_sha256"] = "8"*64
    elif change == "changed_sampler": proof["load_validation"]["sampler_identity"]["portable_bundle_sha256"] = "8"*64
    elif change == "not15": proof["load_validation"]["loaded_buffer_count"] = 14
    elif change == "claim_ready": proof["guidance_ready"] = True
    elif change == "missing_output": proof["output_sha256"] = {}
    update_proof(config, proof)
    with pytest.raises(ValueError): binding.validate_portable_proof(config)


def test_markers_do_not_become_physics_threshold_and_sources_do_not_mutate(tmp_path):
    config, proof = proof_fixture(tmp_path)
    result = binding.validate_portable_proof(config)
    assert result["replay_markers"]["all128_every_value_within_one_ulp"] is False
    proof_path = Path(config["portable_codec"]["conversion_result"])
    proof_path.write_text("{}")
    with pytest.raises(ValueError, match="schema/stage"): binding.validate_portable_proof(config)


def test_freezing_adds_only_two_readonly_assets_without_content_rescan(tmp_path):
    config, _ = proof_fixture(tmp_path, config=_fixture_assets(tmp_path))
    manifest, _ = rollouts.freeze_round_assets(config, tmp_path/"round")
    assert len(manifest["assets"]) == 8
    expected = binding.validate_portable_proof(config, frozen_assets=manifest["assets"])
    assert expected["codec_file_sha256"] == config["portable_codec"]["sha256"]
    for name in ("portable_codec", "portable_codec_proof"):
        assert Path(manifest["assets"][name]["path"]).stat().st_mode & 0o222 == 0
    repeated, _ = rollouts.freeze_round_assets(config, tmp_path/"round")
    assert repeated == manifest
    target = Path(manifest["assets"]["portable_codec"]["path"])
    target.chmod(0o600)
    target.write_bytes(b"changed")
    rollouts.freeze_round_assets(config, tmp_path/"round")
    assert target.read_bytes() == b"changed"


def test_legacy_dispatch_unchanged_and_invalid_portable_never_falls_back(tmp_path, monkeypatch):
    cfg = {"base_checkpoint_sha256": "a"*64, "sampling": {"heun_steps": 64}}
    entries = {"assets": {"base_checkpoint": {"path": "unchanged-original.pt"}}}
    calls = []
    def old(path, **kwargs):
        calls.append((path, kwargs))
        return "legacy-result"
    monkeypatch.setattr(legacy, "load_vpfp_diffusion_sampler", old)
    monkeypatch.setattr(portable, "load_portable_initial_conditioned_sampler", lambda *a, **k: pytest.fail("not an explicit portable config"))
    assert round_driver.load_round_base(cfg, entries, device="cpu") == ("legacy-result", None)
    assert calls[0] == ("unchanged-original.pt", {"expected_sha256": "a"*64, "device": "cpu", "steps": 64, "churn": 0.})
    cfg["portable_codec"] = None
    with pytest.raises(ValueError): round_driver.load_round_base(cfg, entries, device="cpu")
    assert len(calls) == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_portable_round_and_guidance_zero_positive_gradients_preserve_base(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA allocation required")
    config, raw, _, checkpoint_path, kwargs = write_assets(tmp_path)
    normalizer_bundle = torch.load(kwargs["portable_bundle_path"], map_location="cpu", weights_only=True)
    norm = portable.load_frozen_normalizer(normalizer_bundle, expected_sha256=normalizer_bundle["sha256"])
    cfg = {"base_checkpoint": str(checkpoint_path), "base_checkpoint_sha256": kwargs["checkpoint_sha256"],
        "sampling": {"heun_steps": 3}}
    cfg, _ = proof_fixture(tmp_path, config=cfg, codec_path=kwargs["portable_bundle_path"], compact=norm.to_payload())
    roles = binding.portable_roles(cfg)
    entries = {"base_checkpoint": {"path": str(checkpoint_path)}}
    for role, (path, digest, _) in roles.items():
        entries[role] = {"path": path, "source_path": path, "sha256": digest}
    monkeypatch.setattr(legacy, "load_vpfp_diffusion_sampler", lambda *a, **k: pytest.fail("portable may never call strict legacy loader"))
    base, proof_binding = round_driver.load_round_base(cfg, {"assets": entries}, device=device)
    assert proof_binding["normalizer_sha256"] == binding.json_sha(base.normalizer.to_payload())
    assert base.identity["checkpoint_sha256"] == kwargs["checkpoint_sha256"]
    calls = []
    def prior(seeds, conditions=None):
        calls.append(list(seeds))
        return {"initial_controls": torch.ones(len(seeds), config.context_dimension, device=device),
            "equation_conditions": np.array([[20., 1.]]*len(seeds), dtype=np.float32) if conditions is None else conditions}
    monkeypatch.setattr(base.prior, "sample", prior)
    theta = torch.tensor([[20., 1.]], dtype=torch.float32, device=device)
    context, future = base.normalizer.encode(raw[:1].to(device))
    gc = guidance.SolverErrorGuidanceConfig(config, kwargs["checkpoint_sha256"],
        binding.json_sha(base.normalizer.to_payload()), width=4, depth=1)
    torch.manual_seed(621)
    guide = guidance.make_solver_error_regressor(gc, parameter_dtype=torch.float32, device=device).eval().requires_grad_(False)
    weights_before = portable.tensor_state_sha256(base.model.state_dict())
    codec_before = base.normalizer.to_payload()
    zero = guidance.GuidedIC40Sampler(base, guide, strength=0.)
    assert torch.equal(zero.draw([24], conditions=theta)["full"], guidance.draw_frozen_ic40(base, [24], conditions=theta)["full"])
    positive = guidance.GuidedIC40Sampler(base, guide, strength=.1)
    generated = positive.sample_future(context, theta, [31])
    assert torch.isfinite(generated).all() and generated.shape == future.shape
    assert positive.last_report["nonzero_gradient_calls"] > 0
    assert portable.tensor_state_sha256(base.model.state_dict()) == weights_before
    assert base.normalizer.to_payload() == codec_before
    assert all(p.grad is None for p in base.model.parameters())
    base.normalizer.square_root.data.add_(.1)
    positive.assert_frozen()
