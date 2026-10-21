"""Tiny synthetic time-snapshot mechanics; no plasma/distribution validation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    VPFPStructuredEDM, VPFPStructuredEDMConfig, load_vpfp_diffusion_sampler,
)
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import (
    VPFPValueEDMConfig, VPFPValueControlEDM, VPFPValueControlNormalizer, lift_value_controls,
)
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import (
    VPFPTemporalCovarianceEDM, VPFPTemporalCovarianceNormalizer,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_vpfp_structured_diffusion.py"
SPEC = importlib.util.spec_from_file_location("vpfp_training_snapshots_under_test", SCRIPT)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)
NATIVE_FIELDS = {"schema", "config", "model", "normalizer", "base_decoder_sha256",
                 "corpus_manifest_sha256", "training"}


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Clock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now


class Split:
    def __init__(self, values, first_id):
        self.latents = values
        self.conditions = np.tile([.05, 1.], (len(values), 1)).astype("float32")
        self.sample_ids = np.arange(first_id, first_id + len(values))
        self.case_ids = list(map(str, self.sample_ids))

    def __len__(self):
        return len(self.latents)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    full = VPFPStructuredEDMConfig(control_count=4, hermite_channels=2, rho_max_mode=1,
        kinetic_max_mode=1, inactive_dim=3, grid_size=3, width=8, depth=1)
    reduced = VPFPValueEDMConfig.from_full_config(full)
    generator = torch.Generator().manual_seed(17)
    raw = lift_value_controls(torch.randn(16, reduced.latent_dimension, generator=generator), reduced).numpy()
    splits = {"train": Split(raw[:8], 10), "validation": Split(raw[8:], 30)}
    opened, clock = [], Clock()

    def load(path, role, verify_hashes):
        opened.append(role)
        return splits[role]

    monkeypatch.setattr(cli, "load_split", load)
    monkeypatch.setattr(cli, "read_manifest", lambda *a, **k: {
        "latent_dimension": full.latent_dimension, "base_decoder_sha256": "a"*64,
        "manifest_sha256": "b"*64})
    monkeypatch.setattr(cli, "source_fingerprints", lambda: {"tiny_fixture": "c"*64})
    monkeypatch.setattr(cli, "time", clock)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-fixture-not-real-job")
    original_loss = VPFPStructuredEDM.loss

    def timed_loss(self, *args, **kwargs):
        if self.training:
            clock.now += 1.
        return original_loss(self, *args, **kwargs)

    monkeypatch.setattr(VPFPStructuredEDM, "loss", timed_loss)

    def run(name, *, representation="full", **overrides):
        clock.now = 0.
        output = tmp_path / name
        output.mkdir()
        arguments = SimpleNamespace(width=8, depth=1, seed=14, device="cpu", learning_rate=2e-4,
            batch_size=2, steps=3, train_seconds=100., validate_every=100, source_revision="fixture")
        for key, value in overrides.items():
            setattr(arguments, key, value)
        spec = SimpleNamespace(collision_nu_range=full.collision_nu_range,
            temperature_range=full.temperature_range, time_domain=full.time_domain,
            base_decoder_sha256="a"*64)
        factories = {}
        config = full
        if representation in ("value", "temporal"):
            config = reduced
            factories = dict(model_factory=VPFPValueControlEDM, normalizer_factory=VPFPValueControlNormalizer.fit)
        if representation == "temporal":
            factories = dict(model_factory=VPFPTemporalCovarianceEDM,
                normalizer_factory=VPFPTemporalCovarianceNormalizer.fit)
        checkpoint, result = cli.train(arguments, spec, tmp_path / "synthetic-corpus", output,
            model_config_factory=lambda *a: config, **factories)
        return output, checkpoint, result

    return SimpleNamespace(run=run, full=full, clock=clock, opened=opened, tmp_path=tmp_path)


def _assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_tree_equal(left, right)
    else:
        assert first == second


def _read(path):
    return json.loads(path.read_text())


def test_missing_and_empty_preserve_all_training_state_and_outputs(rig):
    first, _, first_result = rig.run("missing")
    second, _, second_result = rig.run("empty", checkpoint_seconds=())
    for filename in ("best.pt", "last.pt"):
        _assert_tree_equal(torch.load(first / "checkpoints" / filename, weights_only=True),
                           torch.load(second / "checkpoints" / filename, weights_only=True))
    assert (first / "training.jsonl").read_text() == (second / "training.jsonl").read_text()
    assert {k: v for k, v in first_result.items() if k != "checkpoint"} == {
        k: v for k, v in second_result.items() if k != "checkpoint"}
    assert not (first / "checkpoint-snapshots.json").exists()
    assert not (second / "checkpoint-snapshots.json").exists()
    assert list((first / "checkpoints").glob("snapshot-*")) == []


def test_snapshot_only_validation_and_publication_leave_training_rng_and_updates_unchanged(rig):
    first, _, _ = rig.run("default")
    second, _, result = rig.run("snapshots", checkpoint_seconds=(1.5,))
    one = torch.load(first / "checkpoints/last.pt", weights_only=True)
    two = torch.load(second / "checkpoints/last.pt", weights_only=True)
    _assert_tree_equal(one, two)
    assert result["checkpoint_snapshots"][0]["step"] == 2
    assert (second / "snapshot-validation-step-000002.json").exists()


@pytest.mark.parametrize("representation, schema", (
    ("full", "vpfp_structured_diffusion_checkpoint_v1"),
    ("value", "vpfp_value_control_diffusion_checkpoint_v1"),
    ("temporal", "vpfp_temporal_covariance_diffusion_checkpoint_v1"),
))
def test_snapshots_are_seven_field_native_assets_with_sha_and_sampler_abi(rig, representation, schema):
    output, _, result = rig.run(representation, representation=representation, checkpoint_seconds=(.5, 20.))
    receipt, = result["checkpoint_snapshots"]
    assert receipt["requested_seconds"] == .5 and receipt["step"] == 1
    assert receipt["actual_elapsed_seconds"] >= .5
    assert receipt["sha256"] == cli.file_sha(receipt["checkpoint"])
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    assert set(payload) == NATIVE_FIELDS and payload["schema"] == schema
    assert payload["training"]["step"] == 1
    assert payload["training"]["checkpoint_role"] == "elapsed_time_snapshot"
    assert payload["training"]["snapshot_validation_used_for_checkpoint_selection"] is False
    sampler = load_vpfp_diffusion_sampler(receipt["checkpoint"], expected_sha256=receipt["sha256"],
        device="cpu", steps=2)
    generated = sampler.generate([700])["latents"]
    assert generated.shape == (1, rig.full.latent_dimension) and np.isfinite(generated).all()
    registry = _read(output / "checkpoint-snapshots.json")
    assert registry["receipts"] == result["checkpoint_snapshots"]
    assert registry["unreached_seconds"] == result["unreached_checkpoint_seconds"] == [20.]
    assert registry["status"] == "completed" and registry["physical_audit_performed"] is False
    assert _read(output / "training-summary.json")["checkpoint_snapshots"] == registry["receipts"]


def test_extra_validation_is_never_a_selection_candidate(rig, monkeypatch):
    scores = iter((100., 10., .001, 20.))  # initial, step 1, snapshot-only step 2, final step 3
    monkeypatch.setattr(cli, "validation_report", lambda *a, **k: {"selection_score": next(scores)})
    output, checkpoint, result = rig.run("selection", checkpoint_seconds=(1.5,))
    assert result["best_step"] == 1 and result["validation_selection_score"] == 10.
    assert result["checkpoint_snapshots"][0]["validation_score"] == .001
    assert torch.load(checkpoint, weights_only=True)["training"]["step"] == 1
    assert not (output / "validation-step-000002.json").exists()


def test_initial_validation_crossing_is_deferred_until_completed_update(rig, monkeypatch):
    calls = []

    def validate(*args, **kwargs):
        if not calls:
            rig.clock.now += 3.
        calls.append(None)
        return {"selection_score": 1.}

    monkeypatch.setattr(cli, "validation_report", validate)
    _, _, result = rig.run("initial-crossing", checkpoint_seconds=(1., 2.), steps=1)
    assert [r["step"] for r in result["checkpoint_snapshots"]] == [1, 1]
    assert all(r["actual_elapsed_seconds"] == 4. for r in result["checkpoint_snapshots"])


def test_initial_validation_budget_failure_never_labels_untrained_snapshot(rig, monkeypatch):
    def validate(*args, **kwargs):
        rig.clock.now += 3.
        return {"selection_score": 1.}

    monkeypatch.setattr(cli, "validation_report", validate)
    with pytest.raises(TimeoutError, match="before any update"):
        rig.run("initial-timeout", checkpoint_seconds=(1.,), train_seconds=2.)
    output = rig.tmp_path / "initial-timeout"
    registry = _read(output / "checkpoint-snapshots.json")
    assert registry["receipts"] == [] and registry["unreached_seconds"] == [1.]
    assert registry["failure_reason"] == "initial_validation_exhausted_budget_before_any_training_update"
    assert list((output / "checkpoints").iterdir()) == []


def test_validation_checkpoint_and_receipt_io_crossings_share_one_state_and_stop_budget(rig, monkeypatch):
    validations = []

    def validate(*args, **kwargs):
        if validations:
            rig.clock.now += 5.
        validations.append(None)
        return {"selection_score": 1.}

    save, write = cli.save_checkpoint, cli.write_json

    def timed_save(path, payload):
        save(path, payload)
        rig.clock.now += 2.

    def timed_write(path, payload):
        write(path, payload)
        if Path(path).name == "checkpoint-snapshots.json":
            rig.clock.now += 2.

    monkeypatch.setattr(cli, "validation_report", validate)
    monkeypatch.setattr(cli, "save_checkpoint", timed_save)
    monkeypatch.setattr(cli, "write_json", timed_write)
    _, _, result = rig.run("io-crossings", checkpoint_seconds=(2., 4., 5., 13., 15., 100.),
        train_seconds=12.)
    receipts = result["checkpoint_snapshots"]
    assert result["final_step"] == 1 and result["training_presentations"] == 2
    assert len(receipts) == 5 and {r["step"] for r in receipts} == {1}
    assert len({r["checkpoint"] for r in receipts}) == len({r["sha256"] for r in receipts}) == 1
    assert [r["actual_elapsed_seconds"] for r in receipts] == [14., 14., 14., 14., 16.]
    assert result["wall_seconds"] == 20. and result["wall_seconds"] >= result["wall_budget_seconds"]
    assert result["stop_reason"] == "training_wall_time_limit"
    assert len(validations) == 2


@pytest.mark.parametrize("thresholds", (None, "1", (0.,), (-1.,), (float("nan"),),
    (float("inf"),), (True,), (np.bool_(False),), (1., 1.), ({},)))
def test_invalid_thresholds_fail_before_reading_any_training_data(rig, thresholds):
    with pytest.raises(ValueError, match="checkpoint_seconds"):
        rig.run("invalid", checkpoint_seconds=thresholds)
    assert rig.opened == []


def test_thresholds_sorted_and_copied():
    values = [3., 1.]
    thresholds = cli._checkpoint_seconds(SimpleNamespace(checkpoint_seconds=values))
    values.append(2.)
    assert thresholds == (1., 3.)


def test_later_training_failure_preserves_published_native_snapshot_and_receipt(rig, monkeypatch):
    original = VPFPStructuredEDM.loss
    updates = []

    def fail_later(self, *args, **kwargs):
        if self.training:
            updates.append(None)
            if len(updates) == 2:
                raise FloatingPointError("intentional synthetic step 2 failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VPFPStructuredEDM, "loss", fail_later)
    with pytest.raises(FloatingPointError, match="synthetic step 2"):
        rig.run("failure", checkpoint_seconds=(.5, 2.))
    registry = _read(rig.tmp_path / "failure/checkpoint-snapshots.json")
    assert registry["status"] == "not_completed" and registry["unreached_seconds"] == [2.]
    receipt, = registry["receipts"]
    assert receipt["step"] == 1 and cli.file_sha(receipt["checkpoint"]) == receipt["sha256"]
    assert set(torch.load(receipt["checkpoint"], weights_only=True)) == NATIVE_FIELDS


def test_snapshot_recorder_rejects_nonfinite_score_and_untrained_state(tmp_path):
    recorder = cli._TrainingSnapshots(tmp_path, (1.,))
    for step, report in ((0, {"selection_score": 1.}), (1, {"selection_score": float("nan")})):
        with pytest.raises(ValueError):
            recorder.capture(step=step, report=report, make_payload=lambda _: pytest.fail("must fail before payload"),
                elapsed=lambda: 1.)
    assert _read(tmp_path / "checkpoint-snapshots.json")["receipts"] == []


def test_snapshot_recorder_refuses_continuation_payload(tmp_path):
    recorder = cli._TrainingSnapshots(tmp_path, (1.,))
    payload = {key: {} for key in NATIVE_FIELDS}
    payload["continuation_state"] = {}
    with pytest.raises(ValueError, match="seven-field"):
        recorder.capture(step=1, report={"selection_score": 1.}, make_payload=lambda _: payload,
            elapsed=lambda: 1.)
    assert recorder.receipts == [] and not (tmp_path / "checkpoints").exists()


def test_snapshot_io_failure_keeps_previous_receipt_and_incomplete_new_file(rig, monkeypatch):
    original = cli.save_checkpoint

    def fail_second_snapshot(path, payload):
        if path.name == "snapshot-step-000002.pt":
            torch.save(payload, path.with_name(path.name + ".partial"))
            raise OSError("intentional synthetic snapshot publication failure")
        original(path, payload)

    monkeypatch.setattr(cli, "save_checkpoint", fail_second_snapshot)
    with pytest.raises(OSError, match="snapshot publication failure"):
        rig.run("io-failure", checkpoint_seconds=(.5, 1.5))
    output = rig.tmp_path / "io-failure"
    registry = _read(output / "checkpoint-snapshots.json")
    assert registry["status"] == "not_completed" and registry["unreached_seconds"] == [1.5]
    receipt, = registry["receipts"]
    assert receipt["step"] == 1 and cli.file_sha(receipt["checkpoint"]) == receipt["sha256"]
    assert (output / "checkpoints/snapshot-step-000002.pt.partial").exists()
    assert not (output / "checkpoints/snapshot-step-000002.pt").exists()
