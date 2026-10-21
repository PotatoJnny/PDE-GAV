"""Regression tests for bound-runtime provenance and preflight readiness."""

from hashlib import sha256
import json

import pytest

import pde_agent_verifier_solver.plugins.gkw.canonical_binding as binding_module
from pde_agent_verifier_solver.sdk import (
    CaseVisibility, GenerationKind, GenerationPurpose, GenerationRequest,
    GeneratorMode,
)
from tests.test_gkw_manufactured_backend import _PhysicalRuntime


@pytest.mark.parametrize("settings", [
    {"generator_sampling_mode": "direct_noise"},
    {"decoder_precision": "float32"},
    {"interaction_protocol": "a-different-distribution"},
    {"generation_batch_size": 8},
])
def test_bound_gkw_rejects_silent_physical_setting_overrides(settings):
    runtime = _PhysicalRuntime()
    binding = binding_module.bind_gkw_runtime(runtime)
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    request = GenerationRequest(1, 1, GeneratorMode.ONLINE, GenerationKind.BASE,
        CaseVisibility.HIDDEN, 311, GenerationPurpose.SELECTION, settings=settings)
    with pytest.raises(ValueError):
        generator.sample(state, request)
    assert not runtime.clean_seeds
    with pytest.raises(ValueError):
        generator.load(binding.checkpoint, settings)


def _minimal_asset_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(binding_module, "_checkout", lambda *_args: None)
    paths, digests = {}, {}
    for name in binding_module.ARTIFACT_NAMES:
        path = tmp_path / name
        content = f"test fixture {name}".encode()
        path.write_bytes(content)
        paths[name] = str(path)
        digests[name] = sha256(content).hexdigest()
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    report = b"{}"
    (qualification / "qualification_report.json").write_bytes(report)
    certificate = {
        "physical_git_commit": binding_module.PHYSICAL_COMMIT,
        "qualification_code_commit": binding_module.PHYSICAL_COMMIT,
        "qualification_sample_count": 2700,
        "qualification_report_sha256": sha256(report).hexdigest(),
    }
    (qualification / "production_certificate.json").write_text(json.dumps(certificate))
    return {
        "physical_root": str(tmp_path), "external_root": str(tmp_path),
        "gyaradax_root": str(tmp_path), "artifacts": paths,
        "artifact_sha256": digests, "qualification_root": str(qualification),
    }


@pytest.mark.parametrize("verify_hashes", [False, True])
def test_minimal_unqualified_certificate_cannot_report_ready(tmp_path, monkeypatch, verify_hashes):
    settings = _minimal_asset_settings(tmp_path, monkeypatch)
    # These thirteen files and the report all hash correctly, but there is no
    # eligible production report, complete certificate or bound manifests.
    with pytest.raises((ValueError, FileNotFoundError)):
        binding_module.inspect_gkw_assets(settings, verify_hashes=verify_hashes)


def test_preflight_respects_canonical_self_excluding_seed_digest(tmp_path, monkeypatch):
    settings = _minimal_asset_settings(tmp_path, monkeypatch)
    from pathlib import Path
    root = Path(settings["qualification_root"])
    certificate = json.loads((root / "production_certificate.json").read_text())
    certificate["schema"] = "fusion_gkw_offline_perturbation_qualification_v2"
    certificate["scale_rule_sha256"] = "a" * 64
    for name in ("generator", "operator", "physical_source"):
        content = json.dumps({"synthetic_manifest": name}).encode()
        (root / f"{name}_manifest.json").write_bytes(content)
        certificate[f"{name}_manifest_sha256"] = sha256(content).hexdigest()
    report = json.dumps({"production_certificate_eligible": True}).encode()
    (root / "qualification_report.json").write_bytes(report)
    certificate["qualification_report_sha256"] = sha256(report).hexdigest()
    keys = ("physical_git_commit", "qualification_code_commit", "scale_rule_sha256",
            "generator_manifest_sha256", "operator_manifest_sha256", "physical_source_manifest_sha256")
    plan = {"schema": "fusion_gkw_perturbation_qualification_seed_plan_v4",
            "campaign_mode": "production", "case_count": 2700,
            "bindings": {key: certificate[key] for key in keys}}
    digest = sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    plan["seed_plan_manifest_sha256"] = certificate["seed_plan_manifest_sha256"] = digest
    (root / "seed_plan.json").write_text(json.dumps(plan, indent=2))
    (root / "production_certificate.json").write_text(json.dumps(certificate))
    # Readiness means file-level preflight only, never a production runtime token.
    checked = binding_module.inspect_gkw_assets(settings)
    assert checked["assets_ready"] is True
    assert checked["qualification_environment_validated"] is False
    assert binding_module.inspect_gkw_assets(settings, verify_hashes=False)["assets_ready"] is False
    plan["case_count"] = 2699
    (root / "seed_plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="digest mismatch"):
        binding_module.inspect_gkw_assets(settings)
