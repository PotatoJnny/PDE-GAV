"""Bounded, persistent-data VPFP diffusion pilot; never automatic admission.

The five numerical-answer splits are fixed before any solve. Only training fits
normalization/weights, only validation selects a checkpoint. Calibration and
reference answers remain closed until checkpoint selection is complete.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec, build_vpfp_online_binding
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import collect_corpus, load_split, read_manifest
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    VPFPCoefficientNormalizer, VPFPStructuredEDM, VPFPStructuredEDMConfig, VPFPDiffusionBatchSampler,
)
from pde_agent_verifier_solver.plugins.vpfp.diffusion_features import VPFPFeaturePlan, extract_vpfp_features
from pde_agent_verifier_solver.plugins.vpfp.diffusion_audit import (
    AuditFeature, AuditSample, DiffusionAuditConfig, audit_diffusion_distribution, audit_selfcheck,
)


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha(model):
    """Serialization-independent identity of initialization, without RNG use."""
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_json(path, payload):
    """Atomic publication within this run only; caller owns this artifact."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as target:
        json.dump(payload, target, sort_keys=True, indent=2, allow_nan=False)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def save_checkpoint(path, payload):
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def emit(stage, **details):
    print(json.dumps({"stage": stage, **details}, allow_nan=False), flush=True)


def source_fingerprints():
    paths = sorted((ROOT / "src/pde_agent_verifier_solver/plugins/vpfp").glob("*.py"))
    paths += [Path(__file__).resolve(), ROOT / "scripts/run_plasma_canonical_smoke.py"]
    return {str(path.relative_to(ROOT)): file_sha(path) for path in paths}


def model_config(arguments, spec):
    return VPFPStructuredEDMConfig(collision_nu_range=spec.collision_nu_range,
        temperature_range=spec.temperature_range, time_domain=spec.time_domain,
        width=arguments.width, depth=arguments.depth, sigma_max=80.,
        log_sigma_mean=-.6, log_sigma_std=1.8)


def allocation_preflight(config, arguments):
    """Full-layout synthetic compute/memory check, BEFORE costly PDE collection."""
    started = time.monotonic()
    device = torch.device(arguments.device)
    torch.manual_seed(arguments.seed + 99)
    model = VPFPStructuredEDM(config).to(device)
    values = torch.randn(arguments.batch_size, config.latent_dimension, device=device)
    values[:, config.rho_dimension:config.rho_dimension + config.inactive_dim] = 0.
    conditions = values.new_tensor([[np.mean(config.collision_nu_range), np.mean(config.temperature_range)]])
    conditions = conditions.expand(len(values), 2)
    loss, _ = model.loss(values, conditions)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("full-layout synthetic allocation preflight failed")
    model.eval()
    sampled = model.sample(conditions[:1], (arguments.seed + 98,), steps=2)
    if not torch.isfinite(sampled).all():
        raise FloatingPointError("synthetic sampler preflight failed")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"execution_passed": True, "data_origin": "synthetic_mechanics_fixture", "scientific_validation": False,
        "full_layout_dimension": config.latent_dimension, "batch_size": arguments.batch_size,
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


def assert_checkpoint_corpus(sampler, manifest, training_ids):
    if sampler.identity.get("corpus_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("frozen checkpoint is bound to a different corpus manifest")
    if sampler.identity["base_decoder_sha256"] != manifest["base_decoder_sha256"]:
        raise ValueError("frozen checkpoint and corpus decoder identities differ")
    if tuple(sampler.normalizer.fit_metadata["sample_ids"]) != tuple(map(int, training_ids)):
        raise ValueError("frozen normalizer was fitted on a different training split")


def balanced_conditions(config, count, seed):
    if count < 16 or count % 8:
        raise ValueError("independent generated audit needs equal >=2 cases in each of eight bins")
    random = np.random.default_rng(seed)
    nu_edges = np.linspace(*config.collision_nu_range, 5)
    temperature_edges = np.linspace(*config.temperature_range, 3)
    rows = []
    for nu in range(4):
        for temperature in range(2):
            rows.extend(zip(random.uniform(nu_edges[nu], nu_edges[nu + 1], count // 8),
                            random.uniform(temperature_edges[temperature], temperature_edges[temperature + 1], count // 8)))
    return np.asarray(rows, dtype=np.float32)


def validation_report(model, normalizer, split, *, batch_size, seed, residual_metric=None):
    """Same whole held-out answers at four fixed log-bin midpoints each time."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    edges = np.asarray(model.config.sigma_bin_edges)
    levels = np.sqrt(edges[:-1] * edges[1:])
    sums = {key: np.zeros(4) for key in ("loss", "rho_loss", "kinetic_loss", "isotropic_baseline", "base_loss")}
    counts = np.zeros(4, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for bin_index, sigma in enumerate(levels):
            for start in range(0, len(split), batch_size):
                raw = torch.from_numpy(np.array(split.latents[start:start + batch_size], copy=True)).to(device)
                conditions = torch.from_numpy(np.array(split.conditions[start:start + batch_size], copy=True)).to(device)
                _, metrics = model.loss(normalizer.encode(raw), conditions, generator=generator,
                                         sigma=torch.full((len(raw),), float(sigma), device=device),
                                         residual_metric=residual_metric)
                for key, source in (("loss", "loss_per_case"), ("rho_loss", "rho_loss_per_case"),
                                    ("kinetic_loss", "kinetic_loss_per_case"),
                                    ("isotropic_baseline", "zero_mean_isotropic_baseline_per_case")):
                    sums[key][bin_index] += float(metrics[source].sum())
                sums["base_loss"][bin_index] += float(metrics.get("base_loss_per_case", metrics["loss_per_case"]).sum())
                counts[bin_index] += len(raw)
    result = {key: (value / counts).tolist() for key, value in sums.items()}
    return {"selection_score": float(np.mean(result["loss"])), "sigma_levels": levels.tolist(),
            "counts_by_sigma": counts.tolist(), "loss_by_sigma": result,
            "criterion": "equal_weight_four_log_sigma_midpoint_validation_loss",
            "residual_metric": None if residual_metric is None else residual_metric.report(),
            "seed": seed, "distribution_qualification": False,
            "baseline": "unfitted_zero_mean_isotropic_gaussian_skip_not_conditional_gaussian"}


def _checkpoint_seconds(arguments):
    """Freeze optional positive time thresholds before any training work."""
    raw = getattr(arguments, "checkpoint_seconds", ())
    if isinstance(raw, (str, bytes)):
        raise ValueError("checkpoint_seconds must be a sequence of finite positive seconds")
    try:
        thresholds = tuple(raw)
    except TypeError as error:
        raise ValueError("checkpoint_seconds must be a sequence of finite positive seconds") from error
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
           or not math.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("checkpoint_seconds must contain finite positive seconds")
    thresholds = tuple(sorted(map(float, thresholds)))
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("checkpoint_seconds must not contain duplicate thresholds")
    return thresholds


def _checkpoint_steps(arguments):
    values = getattr(arguments, "checkpoint_steps", ())
    if isinstance(values, (str, bytes)):
        raise ValueError("checkpoint_steps must be distinct positive integer global updates")
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError("checkpoint_steps must be a sequence") from error
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           or value <= 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("checkpoint_steps must be distinct positive integer global updates")
    return tuple(sorted(map(int, values)))


def _read_training_asset(path, expected_sha256):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('resume requires a regular immutable checkpoint with its expected SHA256')
    value = torch.load(path, map_location="cpu", weights_only=True)
    return value


def _training_contract(arguments, model, training_metric, validation_metric):
    return {"batch_size": arguments.batch_size, "validate_every": arguments.validate_every,
        "data_transform_sha256": (getattr(arguments, "ablation_metadata", None) or {}).get("data_transform_sha256"),
        "learning_rate": arguments.learning_rate, "seed": arguments.seed,
        "precision": "float32_ieee_no_amp_no_tf32", "optimizer": "AdamW_weight_decay_zero",
        "gradient_clip_norm": 1., "ema": "warmup_min(exp(-1/500),1-1/(step+1))",
        "residual_metric": None if training_metric is None else training_metric.report(),
        "common_validation_metric": None if validation_metric is None else validation_metric.report(),
        "training_noise_distribution": (model.training_noise_report() if hasattr(model, "training_noise_report") else {
            "schema": "vpfp_original_clamped_lognormal_training_sigma_v1",
            "log_sigma_mean": model.config.log_sigma_mean, "log_sigma_std": model.config.log_sigma_std,
            "sigma_min": model.config.sigma_min, "sigma_max": model.config.sigma_max})}


def _restore_training(arguments, payload, best, *, model, ema, normalizer, optimizer,
                      noise_rng, np_rng, training, validation, manifest, initial_sha, contract):
    """Restricted exact-update continuation, not checkpoint/model migration.

    New states save global CUDA generators. The authorized old VPFP graph used
    only its saved CUDA noise generator and NumPy epoch generator; it contains
    no dropout or other global-CUDA random operation after initialization.
    Legacy sigma sums cannot be reconstructed exactly from rounded JSON means:
    start a labelled diagnostic segment, never invent cumulative loss sums.
    """
    fields = {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
    if set(payload) != fields | {"continuation_state"} or set(best) != fields:
        raise ValueError("resume needs last training state and a separate seven-field historical best")
    metadata, state = payload["training"], payload["continuation_state"]
    for value in (payload, best):
        if (value["schema"] != getattr(model, "checkpoint_schema", "vpfp_structured_diffusion_checkpoint_v1")
                or value["config"] != model.config.to_dict()
                or value["normalizer"] != normalizer.to_payload()
                or value["corpus_manifest_sha256"] != manifest["manifest_sha256"]
                or value["base_decoder_sha256"] != manifest["base_decoder_sha256"]):
            raise ValueError("resume model/config/normalizer/corpus binding differs")
        if (value["training"]["seed"] != arguments.seed
                or value["training"]["train_case_ids"] != list(map(str, training.case_ids))
                or value["training"]["validation_case_ids"] != list(map(str, validation.case_ids))
                or value["training"]["initial_model_sha256"] != initial_sha):
            raise ValueError("resume initialization or train/validation lineage differs")
    if tuple(normalizer.fit_metadata["sample_ids"]) != tuple(map(int, training.sample_ids)):
        raise ValueError("resume normalizer train sample IDs differ")
    normalizer_sha = sha256(json.dumps(normalizer.to_payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for key in contract:
        actual = (metadata.get("ablation") or {}).get(key) if key == "data_transform_sha256" else metadata.get(key)
        if key not in ("batch_size", "validate_every") and actual != contract[key]:
            raise ValueError(f"resume training contract differs: {key}")
    # Bind modules that implement the actual score/noise/coordinate graph, not
    # unrelated audit code or the shared trainer that now implements restore.
    current = source_fingerprints()
    graph = {"structured_diffusion.py", "value_control_diffusion.py", "temporal_covariance_diffusion.py",
             "initial_conditioned_diffusion.py", "known_initial_prior.py", "noise_geometry.py", "noise_training.py", "coherent_diffusion.py"}
    for name, digest in metadata["source_file_sha256"].items():
        if Path(name).name in graph and current.get(name) != digest:
            raise ValueError(f"resume recorded model source changed: {name}")
    step = metadata["step"]
    if (isinstance(step, bool) or not isinstance(step, int) or step < 1 or arguments.steps <= step
            or metadata["presentations"] < step):
        raise ValueError("resume target must exceed a valid completed global update")
    legacy = "schema" not in state
    if legacy:
        required = {"optimizer", "raw_model", "torch_rng", "noise_rng", "numpy_rng", "epoch_order", "epoch_position"}
        expected = {key: metadata[key] for key in ("source_revision", "source_script_sha256")}
        expected.update(batch_size=arguments.batch_size, validate_every=arguments.validate_every)
        if (set(state) != required or getattr(arguments, "legacy_resume_contract", None) != expected
                or payload["schema"] not in {"vpfp_temporal_covariance_diffusion_checkpoint_v1",
                                             "vpfp_initial_conditioned_future_diffusion_checkpoint_v1"}
                or contract["residual_metric"] is not None or contract["common_validation_metric"] is not None
                or contract["data_transform_sha256"] is not None):
            raise ValueError("legacy resume requires the explicit frozen original VPFP recipe contract")
    else:
        fields = {"schema", "contract", "device_type", "cuda_rng", "best_score", "best_step", "sigma_counts",
            "sigma_loss_sums", "sigma_accounting_start_step", "optimizer", "raw_model", "torch_rng", "noise_rng",
            "numpy_rng", "epoch_order", "epoch_position", "torch_version", "cuda_version"}
        if (set(state) != fields or state.get("schema") != "vpfp_training_continuation_v2" or state.get("contract") != contract
                or state.get("device_type") != next(model.parameters()).device.type
                or state.get("torch_version") != str(torch.__version__) or state.get("cuda_version") != torch.version.cuda):
            raise ValueError("resume state version, runtime, device type or update contract differs")
        counts = np.asarray(state["sigma_counts"])
        sums = np.asarray(state["sigma_loss_sums"])
        origin = state["sigma_accounting_start_step"]
        if (counts.shape != (4,) or counts.dtype.kind not in "iu" or np.any(counts < 0)
                or sums.shape != (4,) or not np.isfinite(sums).all() or np.any(sums < 0)
                or isinstance(origin, bool) or not isinstance(origin, int) or not 0 <= origin <= step):
            raise ValueError("resume cumulative noise diagnostics are invalid")
    order, position = np.asarray(state["epoch_order"], dtype=np.int64), state["epoch_position"]
    if (order.shape != (len(training),) or not np.array_equal(np.sort(order), np.arange(len(training)))
            or isinstance(position, bool) or not isinstance(position, int) or not 0 <= position <= len(order)):
        raise ValueError("resume epoch permutation or position is invalid")
    best_step, best_score = best["training"]["step"], best["training"]["validation"]["selection_score"]
    if not 1 <= best_step <= step or not math.isfinite(best_score):
        raise ValueError("resume historical best step/score is invalid")
    if not legacy and (state["best_step"] != best_step or state["best_score"] != best_score):
        raise ValueError("resume historical best does not match saved selection state")
    for target, values in ((model, state["raw_model"]), (ema, payload["model"])):
        expected = target.state_dict()
        if set(values) != set(expected):
            raise ValueError("resume state tensor names differ")
        buffers = dict(target.named_buffers())
        for name, value in values.items():
            if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                    or value.dtype != expected[name].dtype or not torch.isfinite(value).all()
                    or (name in buffers and not torch.equal(value, expected[name].cpu()))):
                raise ValueError(f"resume invalid weight or immutable buffer: {name}")
        target.load_state_dict(values, strict=True)
    saved_optimizer = state["optimizer"]
    fresh_optimizer = optimizer.state_dict()
    if saved_optimizer["param_groups"] != fresh_optimizer["param_groups"]:
        raise ValueError("resume optimizer hyperparameters differ")
    parameters = list(model.parameters())
    if set(saved_optimizer["state"]) != set(range(len(parameters))):
        raise ValueError("resume optimizer parameter inventory differs")
    for index, entry in saved_optimizer["state"].items():
        if (set(entry) != {"step", "exp_avg", "exp_avg_sq"} or entry["step"].item() != step
                or any(not torch.isfinite(v).all() for v in entry.values() if torch.is_tensor(v))
                or any(entry[key].shape != parameters[index].shape or entry[key].dtype != parameters[index].dtype
                       for key in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("resume optimizer step/nonfinite state is invalid")
    optimizer.load_state_dict(saved_optimizer)
    torch.set_rng_state(state["torch_rng"])
    noise_rng.set_state(state["noise_rng"])
    np_rng.bit_generator.state = state["numpy_rng"]
    if not legacy and next(model.parameters()).is_cuda:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA generator inventory differs")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return {"step": step, "presentations": metadata["presentations"], "order": order, "position": position,
        "best_score": best_score, "best_step": best_step, "legacy": legacy,
        "sigma_counts": np.zeros(4, dtype=np.int64) if legacy else np.asarray(state["sigma_counts"], dtype=np.int64),
        "sigma_loss_sums": np.zeros(4) if legacy else np.asarray(state["sigma_loss_sums"], dtype=np.float64),
        "sigma_accounting_start_step": step if legacy else state["sigma_accounting_start_step"]}


class _TrainingSnapshots:
    """Time-only receipts; no model selection, physical audit or extra RNG.

    First observe thresholds after a completed optimizer update. An initial
    validation crossing waits for that update; if it exhausts the budget, no
    trained snapshot is published. Validation/checkpoint/receipt I/O can cross
    further thresholds at the same step, sharing the same immutable file.
    """

    def __init__(self, output, thresholds):
        self.output, self.thresholds = Path(output), tuple(thresholds)
        self.receipts = []
        self._last_file = None
        self._write()

    def _write(self, *, completed=False, failure_reason=None):
        write_json(self.output / "checkpoint-snapshots.json", {
            "schema": "vpfp_elapsed_training_snapshots_v1", "requested_seconds": list(self.thresholds),
            "receipts": deepcopy(self.receipts),
            "unreached_seconds": list(self.thresholds[len(self.receipts):]),
            "status": "completed" if completed else "not_completed",
            "failure_reason": failure_reason,
            "selection_policy": "time_only_extra_validation_never_changes_best_selection",
            "initial_validation_policy": "defer_until_first_completed_training_update_or_record_no_trained_snapshot",
            "actual_elapsed_semantics": "after_native_checkpoint_publication_and_sha256_read",
            "physical_audit_performed": False})

    def due(self, elapsed):
        return len(self.receipts) < len(self.thresholds) and self.thresholds[len(self.receipts)] <= elapsed

    def capture(self, *, step, report, make_payload, elapsed):
        if step < 1:
            raise ValueError("a trained snapshot requires a completed optimizer update")
        score = float(report["selection_score"])
        if not math.isfinite(score):
            raise ValueError("snapshot elapsed time and validation score must be finite")
        if self._last_file is None or self._last_file["step"] != step:
            checkpoint = self.output / "checkpoints" / f"snapshot-step-{step:06d}.pt"
            payload = make_payload(report)
            fields = {"schema", "config", "model", "normalizer", "base_decoder_sha256", "corpus_manifest_sha256", "training"}
            if not isinstance(payload, dict) or set(payload) != fields:
                raise ValueError("time snapshots require the native-safe seven-field checkpoint")
            payload["training"].update(checkpoint_role="elapsed_time_snapshot",
                snapshot_validation_used_for_checkpoint_selection=False)
            save_checkpoint(checkpoint, payload)
            self._last_file = {"step": step, "checkpoint": str(checkpoint), "sha256": file_sha(checkpoint)}
        actual = float(elapsed())
        if not math.isfinite(actual) or actual < 0:
            raise ValueError("snapshot elapsed time and validation score must be finite")
        while self.due(actual):
            self.receipts.append({"requested_seconds": self.thresholds[len(self.receipts)],
                "actual_elapsed_seconds": actual, **self._last_file, "validation_score": score})
        self._write()


def train(arguments, spec, corpus, output, *, training_metric_factory=None, validation_metric_factory=None,
          model_config_factory=None, model_factory=None, normalizer_factory=None, normalizer_loader=None,
          split_loader=None):
    snapshot_thresholds = _checkpoint_seconds(arguments)
    step_thresholds = _checkpoint_steps(arguments)
    if any(value > arguments.steps for value in step_thresholds):
        raise ValueError("fixed checkpoint updates exceed the final global step")
    resume_path = getattr(arguments, "resume_checkpoint", None)
    resume = best_asset = None
    if resume_path is not None:
        resume = _read_training_asset(resume_path, getattr(arguments, "expected_resume_sha256", None))
        best_asset = _read_training_asset(arguments.resume_best_checkpoint,
            getattr(arguments, "expected_resume_best_sha256", None))
    manifest = read_manifest(corpus, verify_hashes=True)
    get_split = load_split if split_loader is None else split_loader
    if split_loader is not None:
        binding = (getattr(arguments, "ablation_metadata", None) or {}).get("data_transform_sha256")
        if not isinstance(binding, str) or len(binding) != 64 or any(c not in "0123456789abcdef" for c in binding):
            raise ValueError("custom split loader requires a frozen data_transform_sha256 in ablation metadata")
    training = get_split(corpus, "train", verify_hashes=False)
    validation = get_split(corpus, "validation", verify_hashes=False)
    config = (model_config if model_config_factory is None else model_config_factory)(arguments, spec)
    if getattr(config, "full_dimension", config.latent_dimension) != manifest["latent_dimension"] or manifest["base_decoder_sha256"] != spec.base_decoder_sha256:
        raise ValueError("corpus/decoder/current full-coefficient EDM layout mismatch")
    torch.manual_seed(arguments.seed + 100)
    np_rng = np.random.default_rng(arguments.seed + 101)
    device = torch.device(arguments.device)
    # This fitting operation reads ONLY train. Complete trajectories are not
    # treated as 33 or 66 independent training examples.
    fit_normalizer = VPFPCoefficientNormalizer.fit if normalizer_factory is None else normalizer_factory
    if resume is None:
        normalizer = fit_normalizer(config,
            torch.from_numpy(np.array(training.latents, copy=True)), sample_ids=training.sample_ids.tolist()).to(device)
    else:
        if normalizer_factory is not None and normalizer_loader is None:
            raise ValueError("custom normalizer resume requires its strict payload loader, never a refit")
        load_normalizer = VPFPCoefficientNormalizer.from_payload if normalizer_loader is None else normalizer_loader
        normalizer = load_normalizer(resume["normalizer"]).to(device)
    training_metric = None if training_metric_factory is None else training_metric_factory(config, normalizer)
    validation_metric = None if validation_metric_factory is None else validation_metric_factory(config, normalizer)
    model = (VPFPStructuredEDM if model_factory is None else model_factory)(config).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    initial_model_sha256 = model_state_sha(model)
    normalizer_sha256 = sha256(json.dumps(normalizer.to_payload(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=0.)
    noise_rng = torch.Generator(device=device).manual_seed(arguments.seed + 102)
    best_score, best_step, presentations = float("inf"), 0, 0
    step = 0
    started = time.monotonic()
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    history = output / "training.jsonl"
    # Every case is seen once per epoch; partial batches are retained.
    order, position = np_rng.permutation(len(training)), 0
    sigma_counts, sigma_loss_sums = np.zeros(4, dtype=np.int64), np.zeros(4)
    start_step, sigma_accounting_start_step, resume_receipt = 0, 0, None
    contract = _training_contract(arguments, model, training_metric, validation_metric)
    if resume is not None:
        restored = _restore_training(arguments, resume, best_asset, model=model, ema=ema,
            normalizer=normalizer, optimizer=optimizer, noise_rng=noise_rng, np_rng=np_rng,
            training=training, validation=validation, manifest=manifest, initial_sha=initial_model_sha256, contract=contract)
        step = start_step = restored["step"]
        presentations, order, position = (restored[key] for key in ("presentations", "order", "position"))
        best_score, best_step = restored["best_score"], restored["best_step"]
        sigma_counts, sigma_loss_sums = restored["sigma_counts"], restored["sigma_loss_sums"]
        sigma_accounting_start_step = restored["sigma_accounting_start_step"]
        if any(value <= start_step for value in step_thresholds):
            raise ValueError("resume checkpoint_steps must be strictly after the restored global step")
        shutil.copyfile(arguments.resume_best_checkpoint, checkpoint_dir / "best.pt")
        resume_receipt = {"source_checkpoint": str(resume_path), "source_sha256": arguments.expected_resume_sha256,
            "historical_best_checkpoint": str(arguments.resume_best_checkpoint),
            "historical_best_sha256": arguments.expected_resume_best_sha256, "start_step": start_step,
            "target_global_step": arguments.steps, "normalizer_refitted": False,
            "legacy_explicit_rng_graph": restored["legacy"], "sigma_accounting_start_step": sigma_accounting_start_step,
            "update_state_restored": ["raw_model", "EMA", "AdamW", "noise_rng", "torch_rng", "numpy_rng", "epoch_order", "epoch_position", "global_step", "presentations", "historical_best"],
            "global_cuda_rng_restored": not restored["legacy"] and device.type == "cuda",
            "scope": "same_recorded_score_graph_and_deterministic_arithmetic_not_cross_platform_bitwise_guarantee"}
        write_json(output / "resume-receipt.json", resume_receipt)
        del resume, best_asset, restored
    snapshots = _TrainingSnapshots(output, snapshot_thresholds) if snapshot_thresholds else None
    step_receipts = []

    def write_step_receipts(completed=False):
        write_json(output / "checkpoint-step-snapshots.json", {"schema": "vpfp_global_update_snapshots_v1",
            "requested_steps": list(step_thresholds), "receipts": step_receipts,
            "unreached_steps": [value for value in step_thresholds if value not in {row['step'] for row in step_receipts}],
            "completed": completed, "used_for_checkpoint_selection": False, "physical_audit_performed": False})

    if step_thresholds:
        write_step_receipts()

    def payload(report, include_optimizer=False):
        result = {"schema": getattr(model, "checkpoint_schema", "vpfp_structured_diffusion_checkpoint_v1"), "config": config.to_dict(),
            "model": {key: value.detach().cpu().clone() for key, value in ema.state_dict().items()},
            "normalizer": normalizer.to_payload(), "base_decoder_sha256": spec.base_decoder_sha256,
            "corpus_manifest_sha256": manifest["manifest_sha256"],
            "training": {"step": step, "presentations": presentations, "unique_training_answers": len(training),
                "mean_presentations_per_training_answer": presentations / len(training),
                "train_case_ids": list(map(str, training.case_ids)), "validation_case_ids": list(map(str, validation.case_ids)),
                "seed": arguments.seed, "validation": report, "parameters": sum(p.numel() for p in model.parameters()),
                "ema": "warmup_min(exp(-1/500),1-1/(step+1))", "source_revision": arguments.source_revision,
                "source_script_sha256": file_sha(__file__), "source_file_sha256": source_fingerprints(),
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "precision": "float32_ieee_no_amp_no_tf32", "optimizer": "AdamW_weight_decay_zero",
                "learning_rate": arguments.learning_rate, "gradient_clip_norm": 1.,
                "residual_metric": None if training_metric is None else training_metric.report(),
                "training_noise_distribution": (model.training_noise_report()
                    if hasattr(model, "training_noise_report") else {
                        "schema": "vpfp_original_clamped_lognormal_training_sigma_v1",
                        "log_sigma_mean": config.log_sigma_mean, "log_sigma_std": config.log_sigma_std,
                        "sigma_min": config.sigma_min, "sigma_max": config.sigma_max}),
                "common_validation_metric": None if validation_metric is None else validation_metric.report(),
                "ablation": getattr(arguments, "ablation_metadata", None),
                "initial_model_sha256": initial_model_sha256, "normalizer_sha256": normalizer_sha256,
                "scientific_qualification": "unqualified_research_checkpoint", "experimental_data_validation": False}}
        if include_optimizer:
            result["continuation_state"] = {"schema": "vpfp_training_continuation_v2", "contract": contract,
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "device_type": device.type, "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "best_score": best_score, "best_step": best_step,
                "sigma_counts": sigma_counts.tolist(), "sigma_loss_sums": sigma_loss_sums.tolist(),
                "sigma_accounting_start_step": sigma_accounting_start_step,
                "optimizer": optimizer.state_dict(), "raw_model": model.state_dict(),
                "torch_rng": torch.get_rng_state(), "noise_rng": noise_rng.get_state(),
                "numpy_rng": np_rng.bit_generator.state, "epoch_order": order.tolist(), "epoch_position": position}
        if resume_receipt is not None:
            result["training"]["continuation"] = resume_receipt
        return result

    emit("train_start", parameters=sum(p.numel() for p in model.parameters()), unique_answers=len(training), config=config.to_dict())
    report = validation_report(ema, normalizer, validation, batch_size=arguments.batch_size,
                               seed=arguments.seed + 103, residual_metric=validation_metric)
    write_json(output / ("untrained-validation.json" if start_step == 0 else "restored-validation.json"), report)
    if time.monotonic() - started >= arguments.train_seconds:
        if snapshots is not None:
            snapshots._write(failure_reason="initial_validation_exhausted_budget_before_any_training_update")
        raise TimeoutError("initial validation exhausted the training wall budget before any update")
    for step in range(start_step + 1, arguments.steps + 1):
        if position == len(order):
            order, position = np_rng.permutation(len(training)), 0
        indices = order[position:position + arguments.batch_size]
        position += len(indices)
        raw = torch.from_numpy(np.array(training.latents[indices], copy=True)).to(device)
        conditions = torch.from_numpy(np.array(training.conditions[indices], copy=True)).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(normalizer.encode(raw), conditions, generator=noise_rng, residual_metric=training_metric)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite diffusion training loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        decay = min(math.exp(-1. / 500), 1. - 1. / (step + 1))
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, 1. - decay)
        presentations += len(indices)
        sigma_counts += metrics["sigma_bin_counts"].cpu().numpy()
        sigma_loss_sums += metrics["sigma_bin_loss_sums"].cpu().numpy()
        elapsed = time.monotonic() - started
        stop = elapsed >= arguments.train_seconds or step == arguments.steps
        if step == 1 or step % 50 == 0 or stop:
            record = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(gradient), "seconds": elapsed,
                "presentations": presentations, "sigma_bin_counts": sigma_counts.tolist(),
                "sigma_bin_loss_mean": (sigma_loss_sums / np.maximum(sigma_counts, 1)).tolist()}
            if sigma_accounting_start_step:
                record["sigma_accounting_start_step"] = sigma_accounting_start_step
            with history.open("a") as target:
                target.write(json.dumps(record, allow_nan=False) + "\n")
            emit("train", **record)
        report_for_step = None
        if step == 1 or step % arguments.validate_every == 0 or stop:
            report = validation_report(ema, normalizer, validation, batch_size=arguments.batch_size,
                                       seed=arguments.seed + 103, residual_metric=validation_metric)
            write_json(output / f"validation-step-{step:06d}.json", report)
            if report["selection_score"] < best_score:
                best_score, best_step = report["selection_score"], step
                save_checkpoint(checkpoint_dir / "best.pt", payload(report))
            save_checkpoint(checkpoint_dir / "last.pt", payload(report, include_optimizer=True))
            emit("validation", step=step, score=report["selection_score"], best_step=best_step)
            report_for_step = report
        if step in step_thresholds:
            if report_for_step is None:
                report_for_step = validation_report(ema, normalizer, validation,
                    batch_size=arguments.batch_size, seed=arguments.seed + 103, residual_metric=validation_metric)
                write_json(output / f"snapshot-validation-step-{step:06d}.json", report_for_step)
            fixed = checkpoint_dir / f"fixed-step-{step:06d}.pt"
            fixed_payload = payload(report_for_step)
            fixed_payload["training"].update(checkpoint_role="fixed_global_update_snapshot",
                snapshot_validation_used_for_checkpoint_selection=False)
            save_checkpoint(fixed, fixed_payload)
            step_receipts.append({"step": step, "checkpoint": str(fixed), "sha256": file_sha(fixed),
                "validation_score": report_for_step["selection_score"], "actual_elapsed_seconds": time.monotonic()-started})
            write_step_receipts()
        # Validation/checkpoint I/O may itself cross the deadline. Do not take
        # another optimizer step after that completed boundary.
        if snapshots is None:
            if stop or time.monotonic() - started >= arguments.train_seconds:
                break
        else:
            # Recheck after every publication: multiple thresholds crossed by
            # validation or receipt I/O share this completed-update state.
            # The final observed boundary also controls the wall budget.
            while True:
                boundary_elapsed = time.monotonic() - started
                if not snapshots.due(boundary_elapsed):
                    break
                if report_for_step is None:
                    report_for_step = validation_report(ema, normalizer, validation,
                        batch_size=arguments.batch_size, seed=arguments.seed + 103,
                        residual_metric=validation_metric)
                    write_json(output / f"snapshot-validation-step-{step:06d}.json", report_for_step)
                snapshots.capture(step=step, report=report_for_step, make_payload=payload,
                    elapsed=lambda: time.monotonic() - started)
            if stop or boundary_elapsed >= arguments.train_seconds:
                break
    checkpoint = checkpoint_dir / "best.pt"
    result = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha(checkpoint), "best_step": best_step,
        "final_step": step, "unique_training_answers": len(training), "training_presentations": presentations,
        "validation_selection_score": best_score, "wall_seconds": time.monotonic() - started,
        "stop_reason": "step_limit" if step == arguments.steps else "training_wall_time_limit",
        "audit_used_for_selection": False, "scientific_qualification": "not_established",
        "initial_model_sha256": initial_model_sha256, "normalizer_sha256": normalizer_sha256,
        "wall_budget_seconds": arguments.train_seconds,
        "budget_semantics": "initial_validation_included_stop_at_completed_update_or_validation_boundary",
        "training_noise_distribution": (model.training_noise_report()
            if hasattr(model, "training_noise_report") else {
                "schema": "vpfp_original_clamped_lognormal_training_sigma_v1",
                "log_sigma_mean": config.log_sigma_mean, "log_sigma_std": config.log_sigma_std,
                "sigma_min": config.sigma_min, "sigma_max": config.sigma_max}),
        "residual_metric": None if training_metric is None else training_metric.report(),
        "common_validation_metric": None if validation_metric is None else validation_metric.report()}
    if snapshots is not None:
        snapshots._write(completed=True)
        result["wall_seconds"] = time.monotonic() - started
        result["checkpoint_snapshots"] = deepcopy(snapshots.receipts)
        result["unreached_checkpoint_seconds"] = list(snapshot_thresholds[len(snapshots.receipts):])
        result["snapshot_receipts_file"] = str(output / "checkpoint-snapshots.json")
    if step_thresholds:
        write_step_receipts(completed=True)
        result["step_snapshots"] = step_receipts
        result["unreached_checkpoint_steps"] = [value for value in step_thresholds if value not in {row['step'] for row in step_receipts}]
    if resume_receipt is not None:
        result["continuation"] = resume_receipt
        result["additional_updates"] = step - start_step
    write_json(output / "training-summary.json", result)
    return checkpoint, result


def frozen_audit(arguments, spec, corpus, checkpoint, output, *, expected_checkpoint_sha256=None):
    """Open calibration/reference answers only after freezing selected weights."""
    from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
    digest = file_sha(checkpoint)
    if expected_checkpoint_sha256 is not None and digest != expected_checkpoint_sha256:
        raise ValueError("checkpoint changed after selection freeze")
    sampler = load_vpfp_diffusion_sampler(checkpoint, expected_sha256=digest,
                                         device=arguments.device, steps=arguments.sample_steps)
    model, normalizer = sampler.model, sampler.normalizer
    manifest = read_manifest(corpus, verify_hashes=True)
    training_split = load_split(corpus, "train", verify_hashes=False)
    assert_checkpoint_corpus(sampler, manifest, training_split.sample_ids)
    if any(manifest["split_counts"][role] != arguments.heldout_answers for role in ("validation", "calibration", "reference_a", "reference_b")):
        raise ValueError("frozen audit sample count differs from the declared corpus split counts")
    binding = build_vpfp_online_binding(spec, output_root=output / "audit-runtime", device=arguments.device, latent_sampler=sampler)
    decoder = binding.plugin.generator.backend.runtime.base_model
    feature_root = output / "physical-features"
    feature_root.mkdir()
    plan = VPFPFeaturePlan()
    feature_schema, audit_samples = None, {}
    total_count = arguments.heldout_answers
    conditions = balanced_conditions(model.config, total_count, arguments.seed + 200)
    seeds = list(range(arguments.seed + 300000000, arguments.seed + 300000000 + total_count))
    generated = np.lib.format.open_memmap(output / "generated-coefficients.npy", mode="w+", dtype="float32",
                                         shape=(total_count, model.config.latent_dimension))
    for index, (condition, seed) in enumerate(zip(conditions, seeds)):
        generated[index] = normalizer.decode(model.sample(condition[None], (seed,), steps=arguments.sample_steps))[0].cpu().numpy()
        if (index + 1) % 8 == 0:
            generated.flush()
            emit("sample", completed=index + 1, total=total_count, pde_solves=0)
    generated.flush()
    np.save(output / "generated-conditions.npy", conditions, allow_pickle=False)
    write_json(output / "generated-provenance.json", {"checkpoint_sha256": digest, "sampler": sampler.identity,
        "noise_seeds": seeds, "condition_seed": arguments.seed + 200,
        "condition_design": "independent_uniform_in_same_joint_4x2_bins_not_reference_conditions",
        "coefficient_sha256": file_sha(output / "generated-coefficients.npy"), "pde_solves": 0})
    for role in ("calibration", "reference_a", "reference_b", "generated", "train", "validation"):
        if role == "generated":
            latents, cond = generated, conditions
            ids = tuple(f"generated:{digest}:{seed}" for seed in seeds)
        else:
            split = load_split(corpus, role, verify_hashes=False)
            latents, cond = split.latents, split.conditions
            ids = tuple(f"equation:{int(seed)}" for seed in split.sample_ids)
        matrix, descriptors, physical = extract_vpfp_features(decoder, latents, cond, plan=plan,
            progress=lambda done, total, role=role: emit("physical_features", split=role, completed=done, total=total) if done % 16 == 0 else None)
        if feature_schema is not None and feature_schema != descriptors:
            raise RuntimeError("physical feature schema differs between audit splits")
        feature_schema = descriptors
        # Exact answer-content identities include conditions; equal low-dimensional
        # summaries are NOT evidence that two complete answers are duplicates.
        answer_hashes = tuple(row["input_answer_sha256"] for row in physical["per_case"])
        evaluated_conditions = np.asarray([row["input_equation_condition"] for row in physical["per_case"]], dtype=np.float32)
        np.save(feature_root / f"{role}.npy", matrix, allow_pickle=False)
        write_json(feature_root / f"{role}.json", {"physical": physical, "root_ids": ids, "answer_sha256": answer_hashes,
            "conditions": evaluated_conditions.tolist(), "features_sha256": file_sha(feature_root / f"{role}.npy")})
        audit_samples[role] = AuditSample(matrix, evaluated_conditions, ids, answer_sha256=answer_hashes)
    # Detect external modifications to published numerical/generated artifacts;
    # each feature row also carries the hash of its actual local answer snapshot.
    read_manifest(corpus, verify_hashes=True)
    generated_provenance = json.loads((output / "generated-provenance.json").read_text())
    features = tuple(AuditFeature(**entry) for entry in feature_schema)
    write_json(feature_root / "schema.json", {"features": feature_schema, "plan": asdict(plan), "plan_sha256": plan.digest})
    audit_config = DiffusionAuditConfig(permutations=arguments.permutations, bootstrap_repeats=arguments.bootstrap_repeats,
                                       seed=arguments.seed + 400)
    nu_edges, temperature_edges = np.linspace(*spec.collision_nu_range, 5), np.linspace(*spec.temperature_range, 3)
    report = audit_diffusion_distribution(audit_samples, features=features, nu_edges=nu_edges,
                                         temperature_edges=temperature_edges, config=audit_config)
    report.update(checkpoint_sha256=digest, corpus_manifest_sha256=manifest["manifest_sha256"],
        feature_plan_sha256=plan.digest, data_origin=manifest["data_origin"],
        high_fidelity_validation=False, spatial_quadrature_convergence_qualified=False,
        audit_was_not_used_for_checkpoint_selection=True)
    write_json(output / "distribution-audit.json", report)
    empirical = audit_selfcheck(config=audit_config, features=features, calibration_bank=audit_samples["calibration"],
        nu_edges=nu_edges, temperature_edges=temperature_edges, trials=arguments.selfcheck_trials,
        cases_per_stratum=arguments.heldout_answers // 8)
    write_json(output / "audit-empirical-selfcheck.json", empirical)
    return {"status": report["status"], "admitted": False, "checkpoint_sha256": digest,
            "reference_null_difference_detected": report["comparisons"]["reference_null"]["difference_detected"],
            "generated_joint_pvalue": report["comparisons"]["generated"]["joint_pvalue"]}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--train-answers", type=int, default=512)
    result.add_argument("--heldout-answers", type=int, default=128)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--collect-batch-size", type=int, default=8)
    result.add_argument("--steps", type=int, default=2000)
    result.add_argument("--train-seconds", type=int, default=5400)
    result.add_argument("--validate-every", type=int, default=250)
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--width", type=int, default=256)
    result.add_argument("--depth", type=int, default=4)
    result.add_argument("--sample-steps", type=int, default=64)
    result.add_argument("--permutations", type=int, default=99)
    result.add_argument("--bootstrap-repeats", type=int, default=99)
    result.add_argument("--selfcheck-trials", type=int, default=20)
    result.add_argument("--device", default="cuda")
    return result


def main(argv=None):
    cli = parser()
    arguments = cli.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        cli.error("real VPFP collection/training/audit requires a Slurm compute allocation")
    if arguments.output.exists() or arguments.output.is_symlink():
        cli.error("choose a fresh output: partial and failed experiments must be preserved")
    for key in ("train_answers", "heldout_answers"):
        if getattr(arguments, key) < 32 or getattr(arguments, key) % 8:
            cli.error(f"{key} must be a multiple of eight and >=32")
    for key in ("batch_size", "collect_batch_size", "steps", "train_seconds", "validate_every", "selfcheck_trials"):
        if getattr(arguments, key) < 1:
            cli.error(f"{key} must be positive")
    if arguments.seed < 0 or not np.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0:
        cli.error("invalid seed/learning rate")
    DiffusionAuditConfig(permutations=arguments.permutations, bootstrap_repeats=arguments.bootstrap_repeats).validate()
    payload = json.loads(arguments.config.read_text())
    spec = VPFPOnlineModelSpec.from_mapping(payload.get("vpfp", payload))
    spec.preflight()
    config = model_config(arguments, spec)
    if arguments.sample_steps < 2:
        cli.error("Heun sample_steps must be >=2, checked before any PDE collection")
    arguments.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
    plan.update(schema="vpfp_persistent_full_coefficient_edm_pilot_v1", model_spec=asdict(spec),
        model_design={"new_AE_or_PCA_bottleneck": False, "noise_space": "active_full_coefficients",
            "train_log_sigma_mean": -.6, "train_log_sigma_std": 1.8, "sigma_max": 80.,
            "purpose": "broaden_high_sigma_exposure_relative_to_old_pilot_not_a_proven_optimum"},
        physical_feature_plan=asdict(VPFPFeaturePlan()), physical_equivalence_margins=None,
        experimental_data_validation=False, automatic_production_admission=False,
        script_sha256=file_sha(__file__), source_file_sha256=source_fingerprints(),
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        training_selection="validation_sigma_balanced_loss_only", audit_selection=False)
    write_json(arguments.output / "frozen-plan.json", plan)
    stage = "allocation_preflight"
    started = time.monotonic()
    try:
        write_json(arguments.output / "allocation-preflight.json", allocation_preflight(config, arguments))
        stage = "audit_synthetic_selfcheck"
        write_json(arguments.output / "audit-synthetic-selfcheck.json", audit_selfcheck(
            config=DiffusionAuditConfig(permutations=arguments.permutations, bootstrap_repeats=arguments.bootstrap_repeats,
                                        seed=arguments.seed + 400), trials=arguments.selfcheck_trials,
            cases_per_stratum=arguments.heldout_answers // 8))
        stage = "collect"
        write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
        counts = {"train": arguments.train_answers, **{name: arguments.heldout_answers for name in
                    ("validation", "calibration", "reference_a", "reference_b")}}
        corpus = arguments.output / "corpus"
        collect_corpus(spec, corpus, counts, arguments.seed, batch_size=arguments.collect_batch_size, device=arguments.device)
        stage = "train"
        write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
        checkpoint, training = train(arguments, spec, corpus, arguments.output)
        stage = "frozen_physical_distribution_audit"
        write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
        audit = frozen_audit(arguments, spec, corpus, checkpoint, arguments.output)
        report = {"execution_passed": True, "training": training, "audit": audit, "admitted": False,
                  "wall_seconds": time.monotonic() - started, "canonical_campaign_smoke": "separate_required_step"}
    except Exception as error:
        report = {"execution_passed": False, "failed_stage": stage, "error": repr(error),
                  "traceback": traceback.format_exc(), "partial_artifacts_preserved": True, "admitted": False}
    write_json(arguments.output / "result.json", report)
    write_json(arguments.output / "stage.json", {"stage": stage, "status": "complete" if report["execution_passed"] else "failed"})
    emit("finished", **report)
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
