"""CPU synthetic mechanics only; no PDE or scientific qualification."""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
from pde_agent_verifier_solver.plugins.vpfp.sobolev_matching import VPFPSobolevMetric

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_vpfp_sobolev_ablation.py"
specification = importlib.util.spec_from_file_location("vpfp_ablation_under_test", SCRIPT)
cli = importlib.util.module_from_spec(specification)
specification.loader.exec_module(cli)


@pytest.fixture(autouse=True)
def preserve_torch_settings():
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark)
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous[0])
    torch.use_deterministic_algorithms(previous[1])
    torch.backends.cuda.matmul.allow_tf32 = previous[2]
    torch.backends.cudnn.allow_tf32 = previous[3]
    torch.backends.cudnn.benchmark = previous[4]


def tiny_config():
    return VPFPStructuredEDMConfig(control_count=3, hermite_channels=2, rho_max_mode=1,
        kinetic_max_mode=1, inactive_dim=3, grid_size=3, width=8, depth=1)


def test_requires_compute_allocation_before_creating_artifacts(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--config", "missing", "--source-corpus", str(tmp_path / "old"),
                  "--output", str(tmp_path / "new"), "--seed", "3"])
    assert not (tmp_path / "new").exists()


@pytest.fixture
def mock_pipeline(monkeypatch, tmp_path):
    config = tiny_config()
    spec = VPFPOnlineModelSpec("fixture", "base.pt", "a" * 64, "expanded.pt", "b" * 64,
                               config.collision_nu_range, config.temperature_range, config.time_domain)
    source = {"model_spec": asdict(spec), "base_decoder_sha256": spec.base_decoder_sha256,
        "manifest_sha256": "c" * 64, "split_counts": {name: 32 for name in
             ("train", "validation", "calibration", "reference_a", "reference_b")},
        "provenance": {"decoder_config": {"equation": {"x_domain": [6., 7.]},
                     "latent_input_scale": 1., "structured_local_scale": 1.}},
        **{name: getattr(spec, name) for name in ("collision_nu_range", "temperature_range", "time_domain")}}
    fresh = {"source_corpus": {"manifest_sha256": source["manifest_sha256"]}}
    events, calls = [], []
    config_file, output, source_path = tmp_path / "config.json", tmp_path / "new", tmp_path / "old"
    config_file.write_text(json.dumps(asdict(spec)))
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")
    monkeypatch.setattr(VPFPOnlineModelSpec, "preflight", lambda self: {})
    monkeypatch.setattr(cli.shared, "model_config", lambda *a: config)
    monkeypatch.setattr(cli, "read_manifest", lambda path, **k: source if path == source_path else fresh)
    monkeypatch.setattr(cli, "metric_preflight", lambda *a: {"execution_passed": True, "synthetic": True})
    monkeypatch.setattr(cli.shared.subprocess, "check_output", lambda *a, **k: "fixture-revision")
    def collect(*a, **k):
        events.append("collect")
        return fresh
    monkeypatch.setattr(cli, "collect_fresh_evaluation_corpus", collect)
    def train(arguments, spec, corpus, arm_root, **kwargs):
        events.append("train-" + arm_root.name)
        calls.append((arguments, kwargs))
        checkpoint = arm_root / "best.pt"
        checkpoint.write_bytes(arm_root.name.encode())
        return checkpoint, {"initial_model_sha256": "same", "normalizer_sha256": "same",
                            "checkpoint_sha256": cli.shared.file_sha(checkpoint),
                            "common_validation_metric": {"fixed": True}}
    monkeypatch.setattr(cli.shared, "train", train)
    def audit(arguments, spec, corpus, checkpoint, arm_root, **kwargs):
        assert events[:3] == ["collect", "train-baseline", "train-sobolev"]
        assert (output / "frozen-checkpoints.json").is_file()
        assert kwargs["expected_checkpoint_sha256"] == cli.shared.file_sha(checkpoint)
        events.append("audit-" + arm_root.name)
        return {"status": "synthetic-mechanics-only"}
    monkeypatch.setattr(cli.shared, "frozen_audit", audit)
    def derivatives(*a):
        events.append("derivatives")
        return {"synthetic": True}
    monkeypatch.setattr(cli, "derivative_audits", derivatives)
    argv = ["--config", str(config_file), "--source-corpus", str(source_path), "--output", str(output),
            "--seed", "3", "--device", "cpu"]
    return SimpleNamespace(argv=argv, output=output, source=source, fresh=fresh, events=events, calls=calls, spec=spec)


def test_two_checkpoints_freeze_before_audits_with_common_validation(mock_pipeline):
    fixture = mock_pipeline
    assert cli.main(fixture.argv) == 0
    assert fixture.events == ["collect", "train-baseline", "train-sobolev", "audit-baseline", "audit-sobolev", "derivatives"]
    baseline, sobolev = fixture.calls
    assert baseline[1]["training_metric_factory"] is None
    assert sobolev[1]["training_metric_factory"] is sobolev[1]["validation_metric_factory"]
    assert baseline[1]["validation_metric_factory"] is sobolev[1]["validation_metric_factory"]
    assert baseline[0].seed == sobolev[0].seed
    assert baseline[0].train_seconds == sobolev[0].train_seconds == 1200
    assert baseline[0].spatial_lengths == (6., 7.)
    assert baseline[0].latent_input_scale == 1.
    plan = json.loads((fixture.output / "frozen-plan.json").read_text())
    assert plan["physical_derivative_distribution_loss"] is False
    assert plan["full_reverse_process_backpropagation"] is False
    assert plan["admitted"] is False


def test_changed_source_fails_before_training_and_keeps_evidence(mock_pipeline):
    fixture = mock_pipeline
    fixture.fresh["source_corpus"]["manifest_sha256"] = "d" * 64
    assert cli.main(fixture.argv) == 1
    assert fixture.events == ["collect"]
    result = json.loads((fixture.output / "result.json").read_text())
    assert "source corpus changed" in result["error"]
    assert result["partial_artifacts_preserved"] is True


def test_changed_checkpoint_between_audits_fails_closed(monkeypatch, mock_pipeline):
    fixture = mock_pipeline
    original = cli.shared.frozen_audit
    def mutate(*a, **k):
        result = original(*a, **k)
        (fixture.output / "sobolev/best.pt").write_bytes(b"changed-after-freeze")
        return result
    monkeypatch.setattr(cli.shared, "frozen_audit", mutate)
    assert cli.main(fixture.argv) == 1
    assert "audit-sobolev" not in fixture.events
    assert "changed after both-arm freeze" in json.loads((fixture.output / "result.json").read_text())["error"]


def test_source_rejects_undersized_reference_and_full_spec_mismatch(mock_pipeline):
    fixture = mock_pipeline
    fixture.source["split_counts"] = dict.fromkeys(fixture.source["split_counts"], 16)
    with pytest.raises(ValueError, match=">=32"):
        cli.check_source(fixture.spec, fixture.source)
    fixture.source["model_spec"]["expanded_decoder_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="complete frozen"):
        cli.check_source(fixture.spec, fixture.source)


def test_shared_audit_checks_frozen_digest_before_loading_model(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"invalid-replacement")
    with pytest.raises(ValueError, match="changed after selection freeze"):
        cli.shared.frozen_audit(None, None, None, checkpoint, None, expected_checkpoint_sha256="a" * 64)


@pytest.fixture
def tiny_training(monkeypatch, tmp_path):
    config = tiny_config()
    random = np.random.default_rng(21)
    class Split:
        def __init__(self, seed):
            self.latents = random.normal(size=(8, config.latent_dimension)).astype("float32")
            self.latents[:, config.rho_dimension:config.rho_dimension + config.inactive_dim] = 0
            self.conditions = np.tile([.05, 1.], (8, 1)).astype("float32")
            self.sample_ids = np.arange(seed, seed + 8)
            self.case_ids = list(map(str, self.sample_ids))
        def __len__(self):
            return 8
    splits = {"train": Split(1), "validation": Split(21)}
    monkeypatch.setattr(cli.shared, "load_split", lambda path, role, **k: splits[role])
    monkeypatch.setattr(cli.shared, "read_manifest", lambda *a, **k: {
        "latent_dimension": config.latent_dimension, "base_decoder_sha256": "a" * 64,
        "manifest_sha256": "b" * 64})
    monkeypatch.setattr(cli.shared, "model_config", lambda *a: config)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-not-real-job")
    args = SimpleNamespace(width=8, depth=1, seed=14, device="cpu", learning_rate=2e-4,
        batch_size=2, steps=2, train_seconds=30, validate_every=1, source_revision="fixture")
    return args, SimpleNamespace(base_decoder_sha256="a" * 64)


def test_real_tiny_paired_training_initialization_normalizer_validation_identical(tiny_training, tmp_path):
    args, spec = tiny_training
    factory = lambda config, normalizer: VPFPSobolevMetric(config, normalizer, spatial_weight=.25, temporal_weight=.25)
    results, payloads, initial = [], [], []
    for arm in ("baseline", "sobolev"):
        root = tmp_path / arm
        root.mkdir()
        checkpoint, result = cli.shared.train(args, spec, tmp_path / "corpus", root,
            training_metric_factory=factory if arm == "sobolev" else None, validation_metric_factory=factory)
        results.append(result)
        payloads.append(torch.load(checkpoint, weights_only=True))
        initial.append(json.loads((root / "untrained-validation.json").read_text()))
    for key in ("initial_model_sha256", "normalizer_sha256", "common_validation_metric"):
        assert results[0][key] == results[1][key]
    assert initial[0] == initial[1]
    assert payloads[0]["training"]["residual_metric"] is None
    assert payloads[1]["training"]["residual_metric"]["spatial_weight"] == .25
    assert any(not torch.equal(payloads[0]["model"][key], payloads[1]["model"][key]) for key in payloads[0]["model"])


def test_validation_crossing_deadline_does_not_allow_extra_update(monkeypatch, tiny_training, tmp_path):
    args, spec = tiny_training
    elapsed, calls = [0.], []
    original = cli.shared.validation_report
    def validate(*a, **k):
        calls.append(1)
        report = original(*a, **k)
        if len(calls) > 1:
            elapsed[0] = 31.
        return report
    monkeypatch.setattr(cli.shared.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(cli.shared, "validation_report", validate)
    _, result = cli.shared.train(args, spec, tmp_path / "corpus", tmp_path)
    assert result["final_step"] == 1
    assert result["stop_reason"] == "training_wall_time_limit"
    assert len(calls) == 2
