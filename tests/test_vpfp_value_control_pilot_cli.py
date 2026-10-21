"""Tiny synthetic pipeline checks, never plasma distribution validation."""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig, load_vpfp_diffusion_sampler
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig, lift_value_controls

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_vpfp_value_control_pilot.py"
specification = importlib.util.spec_from_file_location("vpfp_value_pilot_under_test", SCRIPT)
cli = importlib.util.module_from_spec(specification)
specification.loader.exec_module(cli)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
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


def test_refuses_login_node_before_creating_output(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--config", "missing", "--cache", str(tmp_path / "old"),
                  "--output", str(tmp_path / "new"), "--seed", "1", "--regime", "weak"])
    assert not (tmp_path / "new").exists()


def test_fixed_reference_rows_are_balanced_and_deterministic(monkeypatch):
    config = tiny_config()
    conditions = cli.shared.balanced_conditions(config, 128, 8)
    random = np.random.default_rng(9)
    random.shuffle(conditions)
    monkeypatch.setattr(cli, "load_split", lambda *a, **k: SimpleNamespace(conditions=conditions))
    rows = cli.fixed_reference_rows(None, config, 64)
    assert rows == cli.fixed_reference_rows(None, config, 64)
    for indices in rows.values():
        bins = cli._strata(conditions[indices], (np.linspace(*config.collision_nu_range, 5), np.linspace(*config.temperature_range, 3)))
        np.testing.assert_array_equal(np.bincount(bins), [8] * 8)
    with pytest.raises(ValueError, match="enough cases"):
        cli.fixed_reference_rows(None, config, 256)


@pytest.fixture
def tiny_training(monkeypatch):
    config = tiny_config()
    value_config = VPFPValueEDMConfig.from_full_config(config)
    generator = torch.Generator().manual_seed(19)
    class Split:
        def __init__(self, seed):
            self.latents = lift_value_controls(torch.randn(8, value_config.latent_dimension, generator=generator), value_config).numpy()
            self.conditions = np.tile([.05, 1.], (8, 1)).astype("float32")
            self.sample_ids = np.arange(seed, seed + 8)
            self.case_ids = list(map(str, self.sample_ids))
        def __len__(self):
            return len(self.sample_ids)
    splits = {"train": Split(1), "validation": Split(21)}
    monkeypatch.setattr(cli.shared, "read_manifest", lambda *a, **k: {
        "latent_dimension": config.latent_dimension, "base_decoder_sha256": "a" * 64, "manifest_sha256": "b" * 64})
    monkeypatch.setattr(cli.shared, "load_split", lambda path, role, **k: splits[role])
    monkeypatch.setattr(cli.shared, "model_config", lambda *a: config)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-fixture-not-job")
    args = SimpleNamespace(seed=14, device="cpu", learning_rate=2.e-4, batch_size=2, steps=2,
        train_seconds=30, validate_every=1, temporal_weight=.25, source_revision="fixture")
    spec = SimpleNamespace(base_decoder_sha256="a" * 64)
    decoder = {"equation": {"x_domain": [6., 7.]}, "latent_input_scale": 1., "structured_local_scale": 1.}
    return args, spec, decoder, config


def test_factories_share_compatible_trunk_but_not_false_full_identity(tiny_training):
    args, spec, decoder, config = tiny_training
    models = []
    for representation in ("full", "value"):
        cfg, factory, _, _ = cli.factories(args, spec, representation, decoder)
        models.append(factory(cfg))
    old, new = (dict(model.named_parameters()) for model in models)
    shared_names = [name for name in old if name in new and old[name].shape == new[name].shape]
    assert any(name.startswith("blocks.") for name in shared_names)
    assert all(torch.equal(old[name], new[name]) for name in shared_names)
    assert models[0].input.weight.shape != models[1].input.weight.shape
    assert cli.shared.model_state_sha(models[0]) != cli.shared.model_state_sha(models[1])


def test_actual_tiny_four_arm_training_loads_complete_native_output(tiny_training, tmp_path):
    args, spec, decoder, full_config = tiny_training
    results = {}
    for arm in cli.ARMS:
        representation, objective = arm.split("_")
        config, make_model, fit_normalizer, metric = cli.factories(args, spec, representation, decoder)
        root = tmp_path / arm
        root.mkdir()
        checkpoint, report = cli.shared.train(args, spec, tmp_path / "corpus", root,
            model_config_factory=lambda *a, cfg=config: cfg, model_factory=make_model,
            normalizer_factory=fit_normalizer, training_metric_factory=metric if objective == "time" else None,
            validation_metric_factory=metric)
        results[arm] = report
        sampler = load_vpfp_diffusion_sampler(checkpoint, report["checkpoint_sha256"], device="cpu", steps=2)
        assert sampler.identity["latent_dimension"] == full_config.latent_dimension
        assert sampler.generate([100])["latents"].shape == (1, full_config.latent_dimension)
        assert report["training_presentations"] == 4
    for representation in ("full", "value"):
        for key in ("initial_model_sha256", "normalizer_sha256", "common_validation_metric"):
            assert results[f"{representation}_base"][key] == results[f"{representation}_time"][key]


@pytest.fixture
def mock_orchestration(monkeypatch, tmp_path):
    config = tiny_config()
    spec = VPFPOnlineModelSpec("weak_fixture", "base.pt", "a" * 64, "expanded.pt", "b" * 64,
        config.collision_nu_range, config.temperature_range, config.time_domain)
    manifest = {"model_spec": asdict(spec), "manifest_sha256": "c" * 64,
        "provenance": {"decoder_config": {"equation": {"x_domain": [6., 7.]}, "latent_input_scale": 1., "structured_local_scale": 1.}}}
    output, cache, config_file = tmp_path / "output", tmp_path / "cache", tmp_path / "config.json"
    config_file.write_text(json.dumps(asdict(spec)))
    events, report_constraints = [], {"passed": True, "source_manifest_sha256": "c" * 64}
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-not-job")
    monkeypatch.setattr(VPFPOnlineModelSpec, "preflight", lambda self: {})
    monkeypatch.setattr(cli, "read_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(cli.shared, "model_config", lambda *a: config)
    monkeypatch.setattr(cli.shared.subprocess, "check_output", lambda *a, **k: "fixture-revision")
    monkeypatch.setattr(cli, "cache_inventory", lambda *a: {})
    monkeypatch.setattr(cli, "fixed_reference_rows", lambda *a: {})
    def constraint(*a, **k):
        events.append("constraint")
        return report_constraints
    monkeypatch.setattr(cli, "audit_value_control_corpus", constraint)
    def require(report):
        if not report["passed"]:
            raise ValueError("off-subspace corpus")
    monkeypatch.setattr(cli, "require_value_control_audit", require)
    monkeypatch.setattr(cli, "full_layout_preflight", lambda *a: {"synthetic_only": True})
    def train(args, spec, corpus, arm_root, **kw):
        events.append("train:" + str(arm_root.relative_to(output)))
        checkpoint = arm_root / "best.pt"
        checkpoint.write_bytes(str(arm_root).encode())
        return checkpoint, {"checkpoint_sha256": cli.shared.file_sha(checkpoint),
            "initial_model_sha256": "shared", "normalizer_sha256": "shared", "common_validation_metric": {"fixed": True}}
    monkeypatch.setattr(cli.shared, "train", train)
    def load_features(*a):
        assert len([event for event in events if event.startswith("train:")]) == 8
        assert (output / "frozen-checkpoints.json").is_file()
        events.append("open_development_features")
        matrix = np.ones((2, 1))
        return {kind: {role: SimpleNamespace(features=matrix) for role in ("reference_a", "reference_b")}
                for kind in ("physical", "derivative")}, {"physical": (), "derivative": ()}
    monkeypatch.setattr(cli, "load_cached_features", load_features)
    monkeypatch.setattr(cli, "load_vpfp_diffusion_sampler", lambda *a, **k: None)
    runtime = SimpleNamespace(base_model=None)
    monkeypatch.setattr(cli, "build_vpfp_online_binding", lambda *a, **k: SimpleNamespace(plugin=SimpleNamespace(generator=SimpleNamespace(backend=SimpleNamespace(runtime=runtime)))))
    def sample(*a):
        events.append("sample")
        return {kind: np.ones((2, 1)) for kind in ("physical", "derivative")}, {
            kind: {"status": "synthetic_only", "comparisons": {"generated": {"statistics": {"sliced_wasserstein_max": {"effect": 1.}}}}}
            for kind in ("physical", "derivative")}
    monkeypatch.setattr(cli, "sample_and_audit", sample)
    monkeypatch.setattr(cli, "build_vpfp_direction_scorecard", lambda **k: {"objective": k["objective"], "seed": k["seed"]})
    def aggregate(cards, **kwargs):
        assert len({card["seed"] for card in cards}) == 2
        assert "required_regimes" not in kwargs
        return {"recommend_scale_up": False, "fresh_confirmation_required": True}
    monkeypatch.setattr(cli, "aggregate_vpfp_direction_scorecards", aggregate)
    argv = ["--config", str(config_file), "--cache", str(cache), "--output", str(output),
            "--regime", "weak", "--seed", "8", "--device", "cpu"]
    return SimpleNamespace(argv=argv, output=output, events=events, constraints=report_constraints)


def test_all_eight_arms_freeze_before_opening_development_features(mock_orchestration):
    fixture = mock_orchestration
    assert cli.main(fixture.argv) == 0
    expected = [f"train:seed-0/{arm}" for arm in cli.ARMS] + [f"train:seed-1/{arm}" for arm in reversed(cli.ARMS)]
    assert fixture.events[:10] == ["constraint", *expected, "open_development_features"]
    assert fixture.events.count("sample") == 8
    report = json.loads((fixture.output / "result.json").read_text())
    assert report["large_experiment_launched"] is False
    assert report["admitted"] is False
    assert report["cross_regime_scorecard_aggregation_required"] is True


def test_off_subspace_full_corpus_gate_prevents_all_training(mock_orchestration):
    fixture = mock_orchestration
    fixture.constraints["passed"] = False
    assert cli.main(fixture.argv) == 1
    assert fixture.events == ["constraint"]
    assert (fixture.output / "value-slope-corpus-audit.json").is_file()
    report = json.loads((fixture.output / "result.json").read_text())
    assert report["failed_stage"] == "full_corpus_constraint_audit"
    assert report["partial_artifacts_preserved"] is True
