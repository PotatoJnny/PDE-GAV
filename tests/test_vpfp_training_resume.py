"""Tiny real EDM/AdamW continuation; no real corpus or GPU computation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("training_resume_under_test", ROOT / "scripts/train_vpfp_structured_diffusion.py")
shared = importlib.util.module_from_spec(loader)
loader.loader.exec_module(shared)
loader = importlib.util.spec_from_file_location("resume_prior_fixture", Path(__file__).with_name("test_vpfp_initial_conditioned_diffusion.py"))
fixture = importlib.util.module_from_spec(loader)
loader.loader.exec_module(fixture)
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    VPFPStructuredEDM, VPFPCoefficientNormalizer, load_vpfp_diffusion_sampler)
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import (
    VPFPValueControlEDM, VPFPValueControlNormalizer, lift_value_controls)
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import (
    VPFPTemporalCovarianceEDM, VPFPTemporalCovarianceNormalizer)


def same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


class Split:
    def __init__(self, raw, offset):
        self.latents = raw.numpy()
        self.conditions = np.tile([20., 1.], (len(raw), 1)).astype("float32")
        self.sample_ids = np.arange(offset, offset + len(raw))
        self.case_ids = self.sample_ids.astype(str).tolist()

    def __len__(self):
        return len(self.latents)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    conditional = fixture.fixture_config()
    full, value = conditional.full_config, conditional.value_config
    raw = lift_value_controls(torch.randn(11, value.latent_dimension,
        generator=torch.Generator().manual_seed(73)), value)
    splits = {"train": Split(raw[:7], 10), "validation": Split(raw[7:], 30)}
    monkeypatch.setattr(shared, "load_split", lambda path, role, **kwargs: splits[role])
    manifest = {"latent_dimension": full.latent_dimension, "base_decoder_sha256": "a"*64, "manifest_sha256": "c"*64}
    monkeypatch.setattr(shared, "read_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(shared, "source_fingerprints", lambda: {"tiny_fixture": "d"*64})
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-no-real-job")
    spec = SimpleNamespace(base_decoder_sha256="a"*64)

    def run(name, representation="temporal", source=None, *, mutate=None, **overrides):
        split_loader = overrides.pop("split_loader", None)
        config, model, normalizer = {"full": (full, VPFPStructuredEDM, VPFPCoefficientNormalizer),
            "value": (value, VPFPValueControlEDM, VPFPValueControlNormalizer),
            "temporal": (value, VPFPTemporalCovarianceEDM, VPFPTemporalCovarianceNormalizer),
            "conditional": (conditional, fixture.joint.VPFPInitialConditionedEDM, fixture.joint.VPFPInitialConditionedNormalizer)}[representation]
        args = SimpleNamespace(seed=28, device="cpu", learning_rate=2.e-4, batch_size=2,
            steps=7, train_seconds=1000., validate_every=1, source_revision="tiny-fixture")
        if source is not None:
            args.resume_checkpoint = source / "checkpoints/last.pt"
            args.resume_best_checkpoint = source / "checkpoints/best.pt"
            if mutate is not None:
                data = torch.load(args.resume_checkpoint, weights_only=True)
                mutate(data)
                torch.save(data, args.resume_checkpoint)
            args.expected_resume_sha256 = shared.file_sha(args.resume_checkpoint)
            args.expected_resume_best_sha256 = shared.file_sha(args.resume_best_checkpoint)
        for key, val in overrides.items():
            setattr(args, key, val)
        output = tmp_path / name
        output.mkdir()
        def fit(cfg, values, *, sample_ids):
            if source is not None:
                raise AssertionError("resume must never fit normalization")
            extra = {"initial_prior": fixture.prior_spec(conditional)} if representation == "conditional" else {}
            return normalizer.fit(cfg, values, sample_ids=sample_ids, **extra)
        _, result = shared.train(args, spec, tmp_path / "synthetic-corpus", output,
            model_config_factory=lambda *a: config, model_factory=model,
            normalizer_factory=fit, normalizer_loader=normalizer.from_payload, split_loader=split_loader)
        return output, result
    yield SimpleNamespace(run=run, tmp_path=tmp_path, splits=splits, full=full)
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize("representation", ["full", "value", "temporal", "conditional"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_continuous_equals_split_raw_ema_optimizer_rng_and_partial_epoch(rig, representation, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA save/restore equivalence requires a compute GPU allocation")
    whole, _ = rig.run("whole", representation, device=device)
    first, _ = rig.run("first", representation, steps=3, device=device)
    resumed, report = rig.run("resumed", representation, source=first, device=device)
    left = torch.load(whole / "checkpoints/last.pt", weights_only=True)
    right = torch.load(resumed / "checkpoints/last.pt", weights_only=True)
    same(left["model"], right["model"])
    same(left["normalizer"], right["normalizer"])
    same(left["continuation_state"], right["continuation_state"])
    assert report["additional_updates"] == 4 and report["final_step"] == 7
    assert right["training"]["presentations"] == left["training"]["presentations"] == 13
    assert report["continuation"]["normalizer_refitted"] is False
    same(torch.load(whole / "checkpoints/best.pt", weights_only=True)["model"],
         torch.load(resumed / "checkpoints/best.pt", weights_only=True)["model"])


def test_fixed_step_snapshots_do_not_change_update_state_and_load_natively(rig):
    plain, _ = rig.run("plain", validate_every=4)
    snap, report = rig.run("snap", validate_every=4, checkpoint_steps=(2, 7))
    one, two = (torch.load(path / "checkpoints/last.pt", weights_only=True) for path in (plain, snap))
    same(one["model"], two["model"])
    same(one["continuation_state"], two["continuation_state"])
    assert [row["step"] for row in report["step_snapshots"]] == [2, 7]
    for row in report["step_snapshots"]:
        sampler = load_vpfp_diffusion_sampler(row["checkpoint"], row["sha256"], device="cpu", steps=2)
        assert sampler.identity["latent_dimension"] == rig.full.latent_dimension
        assert len(torch.load(row["checkpoint"], weights_only=True)) == 7
    registry = json.loads((snap / "checkpoint-step-snapshots.json").read_text())
    assert registry["completed"] and not registry["unreached_steps"]


@pytest.mark.parametrize("change,match", [
    (lambda data: data["continuation_state"]["epoch_order"].__setitem__(0, -1), "permutation"),
    (lambda data: data["continuation_state"].__setitem__("best_score", 999.), "historical best"),
    (lambda data: data["training"].__setitem__("seed", 999), "lineage"),
    (lambda data: data["continuation_state"]["optimizer"]["param_groups"][0].__setitem__("betas", (.8, .99)), "hyperparameters"),
    (lambda data: data["continuation_state"].__setitem__("sigma_counts", [1, 2]), "diagnostics"),
    (lambda data: data["continuation_state"]["raw_model"][next(iter(data["continuation_state"]["raw_model"]))].fill_(float("nan")), "weight"),
])
def test_tampered_resume_rejected_before_updates(rig, change, match):
    source, _ = rig.run("source", steps=3)
    with pytest.raises(ValueError, match=match):
        rig.run("invalid", source=source, mutate=change)
    assert not (rig.tmp_path / "invalid/training.jsonl").exists()


@pytest.mark.parametrize("options", [{"batch_size": 3}, {"validate_every": 2}, {"steps": 3},
    {"checkpoint_steps": (3, 7)}, {"expected_resume_sha256": "0"*64},
    {"ablation_metadata": {"data_transform_sha256": "b"*64}}])
def test_resume_recipe_or_hash_changes_fail_closed(rig, options):
    source, _ = rig.run("source", steps=3)
    with pytest.raises(ValueError):
        rig.run("invalid", source=source, **options)


def test_legacy_explicit_rng_restore_matches_continuous_updates(rig):
    whole, _ = rig.run("whole")
    first, _ = rig.run("first", steps=3)
    last = torch.load(first / "checkpoints/last.pt", weights_only=True)
    legacy_fields = {"optimizer", "raw_model", "torch_rng", "noise_rng", "numpy_rng", "epoch_order", "epoch_position"}
    last["continuation_state"] = {key: value for key, value in last["continuation_state"].items() if key in legacy_fields}
    torch.save(last, first / "checkpoints/last.pt")
    with pytest.raises(ValueError, match="legacy"):
        rig.run("no-contract", source=first)
    contract = {key: last["training"][key] for key in ("source_revision", "source_script_sha256")}
    contract.update(batch_size=2, validate_every=1)
    resumed, report = rig.run("legacy", source=first, legacy_resume_contract=contract)
    left, right = (torch.load(path / "checkpoints/last.pt", weights_only=True) for path in (whole, resumed))
    same(left["model"], right["model"])
    for key in legacy_fields:
        same(left["continuation_state"][key], right["continuation_state"][key])
    assert report["continuation"]["sigma_accounting_start_step"] == 3
    assert report["continuation"]["legacy_explicit_rng_graph"]


def test_resumed_historical_best_is_preserved_when_all_later_scores_worse(rig, monkeypatch):
    monkeypatch.setattr(shared, "validation_report", lambda *a, **k: {"selection_score": 1.})
    first, _ = rig.run("first", steps=3)
    monkeypatch.setattr(shared, "validation_report", lambda *a, **k: {"selection_score": 2.})
    resumed, report = rig.run("resumed", source=first)
    assert report["best_step"] == 1 and report["validation_selection_score"] == 1.
    assert shared.file_sha(first / "checkpoints/best.pt") == shared.file_sha(resumed / "checkpoints/best.pt")


def test_transformed_split_loader_is_bound_and_opens_only_train_validation(rig):
    opened = []
    def load(path, role, verify_hashes):
        opened.append(role)
        return rig.splits[role]
    with pytest.raises(ValueError, match="data_transform_sha256"):
        rig.run("unbound-view", split_loader=load)
    assert opened == []
    options = {"split_loader": load, "ablation_metadata": {"data_transform_sha256": "a"*64}}
    source, _ = rig.run("source", steps=3, **options)
    resumed, _ = rig.run("resumed", source=source, **options)
    assert opened == ["train", "validation", "train", "validation"]
    state = torch.load(resumed / "checkpoints/last.pt", weights_only=True)
    assert state["continuation_state"]["contract"]["data_transform_sha256"] == "a"*64


@pytest.mark.parametrize("values", [(0,), (True,), (2, 2), (float("nan"),), (1.5,), "2"])
def test_bad_step_thresholds_rejected(values):
    with pytest.raises(ValueError):
        shared._checkpoint_steps(SimpleNamespace(checkpoint_steps=values))
