"""Fresh hierarchical Guidance from canonical selected-Solver feedback.

No Solver invocation, resampling, IC40 codec, historical pilot labels or pilot
heads are used. Invalid feedback keeps its fixed slot and has target zero.
The snapshot is an execution result, never a physical qualification.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
from collections.abc import Mapping

import numpy as np  # Frozen initialization order: NumPy before Torch.
import torch

from .round_solver_guidance import (RoundGuidanceConfig, _file_sha, _json_sha,
    _round_identity, _seed, _tensor_sha, _write)
from .weak_solver_guidance import WeakSolverGuidanceHead, guidance_payload, load_guidance_payload

SCHEMA = "vpfp_weak_campaign_round_guidance_v1"
TENSORS = ("m", "z", "white_m", "white_z", "condition_m", "conditions", "full_coefficients")
STREAMS = ("roots", "macro_noise_seeds", "micro_noise_seeds")
LEVELS = (.002, .05, .2, 1., 2.)
COORDINATE_REPLAY_ATOL, COORDINATE_REPLAY_RTOL = 2e-6, 5e-5


def _snapshot(heads, report, path, digest):
    from .weak_iteration_backend import WeakGuidanceSnapshot
    return WeakGuidanceSnapshot(heads=heads, report=report,
        checkpoint_path=str(path), checkpoint_sha256=digest)


def _coordinate_replay(base, draw):
    """Check decoded FP32 values without replacing the original captures.

    Replaying one row after batched capture can select a different GEMM
    reduction order. Validate white-to-physical replay with the fixed tolerance,
    then validate the saved conditioning exactly against the saved physical m.
    Re-encoding replayed m can amplify its roundoff and is diagnostic only.
    Original captures, training inputs, stream identities and hashes do not change.
    """
    from .coherent_velocity_diffusion import decode_macro_coordinates
    from .innovation_diffusion import decode_complete_innovation_answer
    from .value_control_diffusion import extract_value_controls, lift_value_controls
    device = next(base.micro.parameters()).device
    if not isinstance(draw, Mapping):
        raise ValueError("actual native captured draw must be a mapping")
    item = {}
    for name in TENSORS:
        value = draw.get(name)
        if (not isinstance(value, torch.Tensor) or value.ndim != 2 or not len(value)
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError("finite native FP32 captured tensor required: " + name)
        item[name] = value.to(device)
    count = len(item["conditions"])
    if item["conditions"].shape != (count, 2) or any(len(value) != count for value in item.values()):
        raise ValueError("captured native tensor batch/condition shape differs")
    report = dict(atol=COORDINATE_REPLAY_ATOL, rtol=COORDINATE_REPLAY_RTOL, fields={},
        recomputed_values_only=True, original_captures_replaced=False,
        condition_m_check="encode_saved_m_exact")
    with torch.no_grad():
        m = decode_macro_coordinates(base.macro_prior_norm.decode(item["white_m"]), item["conditions"], base.macro.config)
        context = base.macro_norm.encode(m)
        saved_context = base.macro_norm.encode(item["m"])
        full = decode_complete_innovation_answer(item["white_z"], base.micro_norm, base.transform, physical_macro=m)
        z = extract_value_controls(full, base.micro.config, validate_slopes=True)
        expected = {"m": m, "condition_m": saved_context, "full_coefficients": full, "z": z}
        for name, value in expected.items():
            saved = item[name]
            if (saved.shape != value.shape or saved.dtype != value.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("captured native replay shape/dtype/finiteness differs: " + name)
            metrics = dict(exact=torch.equal(saved, value),
                equivalent=torch.allclose(value, saved, atol=COORDINATE_REPLAY_ATOL, rtol=COORDINATE_REPLAY_RTOL),
                max_abs=float((value.double() - saved.double()).abs().max()))
            if name == "condition_m":
                metrics["equivalent"] = metrics["exact"]
            report["fields"][name] = metrics
            if not metrics["equivalent"]:
                if name == "condition_m":
                    raise ValueError("feedback captured native condition_m differs from exact encode(saved_m): "
                        + json.dumps(metrics, sort_keys=True))
                raise ValueError("feedback actual answer differs from its captured native white coordinates: "
                    + name + " " + json.dumps(metrics, sort_keys=True))
        saved = item["condition_m"]
        if (context.shape != saved.shape or context.dtype != saved.dtype
                or not bool(torch.isfinite(context).all())):
            raise ValueError("captured native composite condition replay shape/dtype/finiteness differs")
        report["composite_condition_replay"] = dict(exact=torch.equal(context, saved),
            equivalent=torch.allclose(context, saved, atol=COORDINATE_REPLAY_ATOL, rtol=COORDINATE_REPLAY_RTOL),
            max_abs=float((context.double() - saved.double()).abs().max()), hard_gate=False)
        if not torch.equal(lift_value_controls(z, base.micro.config), full):
            raise ValueError("feedback full native lift changed")
        if not torch.equal(lift_value_controls(item["z"], base.micro.config), item["full_coefficients"]):
            raise ValueError("feedback saved full native lift changed")
    report.update(equivalent=True, exact=all(row["exact"] for row in report["fields"].values()),
        recomputed_full_lift_exact=True, saved_full_lift_exact=True)
    return report


def prepare_feedback(base, rows, identity, *, check=lambda: None):
    """Preserve the supplied first80 order and all validity slots before fitting."""
    if not isinstance(rows, (tuple, list)) or len(rows) != 80:
        raise ValueError("exactly80 actual clean BASE feedback roots required")
    base.assert_frozen()
    for model in (base.macro, base.micro):
        if model.training or next(model.parameters()).dtype != torch.float32 or any(p.requires_grad for p in model.parameters()):
            raise ValueError("native FP32 macro/micro Base must be frozen and eval")
    private, records, ids, roots = [], [], set(), set()
    for index, row in enumerate(rows):
        check()
        if not isinstance(row, Mapping):
            raise TypeError("feedback row must be a mapping")
        case_id, root = row.get("case_id"), row.get("root_seed")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError("unique actual Case IDs required")
        if type(root) is not int or not 0 <= root < 2**63 or root in roots:
            raise ValueError("unique nonnegative signed64 physical roots required")
        if row.get("source") != "base":
            raise ValueError("only clean BASE feedback is eligible")
        if any(row.get(key) != identity[key] for key in ("source_round_index", "target_solver_sha256")):
            raise ValueError("feedback selected Solver/round binding changed")
        valid, error = row.get("valid"), row.get("error")
        if type(valid) is not bool or (valid and (type(error) not in (int, float) or not math.isfinite(error) or error < 0)):
            raise ValueError("valid feedback requires a finite nonnegative actual error")
        if not valid and error is not None:
            raise ValueError("invalid errors must be None, never an imputed high reward")
        raw = row.get("draw")
        if not isinstance(raw, Mapping):
            raise ValueError("actual native draw required; no root replay")
        draw = {}
        for name in TENSORS:
            value = raw.get(name)
            if (not isinstance(value, torch.Tensor) or value.ndim != 2 or len(value) != 1
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError("finite one-row native FP32 capture required: " + name)
            draw[name] = value.detach().cpu().clone()
        if draw["conditions"].shape != (1, 2):
            raise ValueError("one actual nu/T condition required")
        from hashlib import sha256
        expected = {"roots": root, **{name + "_noise_seeds": int.from_bytes(sha256(
            f"vpfp-innovation-{name}:{root}".encode()).digest()[:8], "little") % 2**63 for name in ("macro", "micro")}}
        for name in STREAMS:
            sequence = raw.get(name)
            if (not isinstance(sequence, (list, tuple)) or len(sequence) != 1
                    or type(sequence[0]) is not int or sequence[0] != expected[name]):
                raise ValueError("actual draw noise/root stream changed: " + name)
            draw[name] = list(sequence)
        replay = _coordinate_replay(base, draw)
        split = "fit" if index < 64 else "validation"
        records.append({"case_id": case_id, "root_seed": root, "source": "base", "slot": index,
            "split": split, "valid": valid, "error": float(error) if valid else None,
            "source_round_index": identity["source_round_index"], "target_solver_sha256": identity["target_solver_sha256"],
            "conditions": draw["conditions"][0].tolist(), "streams": {name: draw[name] for name in STREAMS},
            "tensor_sha256": {name: _tensor_sha(draw[name]) for name in TENSORS},
            "coordinate_replay": deepcopy(replay)})
        private.append(draw); ids.add(case_id); roots.add(root)
    base.assert_frozen()
    device = next(base.micro.parameters()).device
    prepared = {}
    for split, begin, end in (("fit", 0, 64), ("validation", 64, 80)):
        selected = records[begin:end]
        prepared[split] = {key: torch.cat([row[key] for row in private[begin:end]]).to(device)
                          for key in ("white_m", "white_z", "condition_m", "conditions")}
        prepared[split].update(errors=torch.tensor([r["error"] if r["valid"] else float("nan") for r in selected],
            device=device, dtype=torch.float64), valid=torch.tensor([r["valid"] for r in selected], device=device, dtype=torch.bool))
    return prepared, records, private


def _context(stage, data, index):
    return {"macro_context": data["condition_m"][index]} if stage == "micro" else {}


def _validation(head, data, *, seed, check):
    """Same16 roots/noise at all checkpoints, not a generator/Solver audit."""
    clean = data["white_m" if head.stage == "macro" else "white_z"]
    rng = torch.Generator(device=clean.device).manual_seed(seed)
    records = []
    head.eval()
    for level in LEVELS:
        for begin in range(0, 16, 4):
            check(); index = slice(begin, begin + 4)
            sigma = clean.new_full((4,), level)
            noisy = clean[index] + sigma[:, None] * torch.randn(clean[index].shape,
                device=clean.device, dtype=torch.float32, generator=rng)
            with torch.no_grad():
                output = head(noisy, sigma, data["conditions"][index], **_context(head.stage, data, index))
                target, valid = head.targets(data["errors"][index], data["valid"][index], output)
                predicted = output.exp()
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("nonfinite fixed development prediction")
            for offset in range(4):
                records.append({"validation_slot": begin + offset, "sigma": level,
                    "valid": bool(valid[offset]), "prediction": float(predicted[offset]), "target": float(target[offset])})
    # One fixed differentiation probe verifies a usable saved score gradient.
    check()
    with torch.enable_grad():
        query = clean[:1].detach().clone().requires_grad_(True)
        output = head(query, clean.new_tensor([.2]), data["conditions"][:1], **_context(head.stage, data, slice(0, 1)))
        gradient, = torch.autograd.grad(output.sum(), query)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("nonfinite fixed development score gradient")
    return {"records": records, "independent_roots": 16, "noisy_records": len(records),
        "mse": math.fsum((r["prediction"] - r["target"])**2 for r in records) / len(records),
        "constant_one_mse": math.fsum((1 - r["target"])**2 for r in records) / len(records),
        "gradient_sha256": _tensor_sha(gradient), "gradient_finite": True,
        "gradient_nonzero": bool(torch.any(gradient != 0)), "solver_hardness_measured": False}


def load_weak_round_guidance(path, expected_sha256, *, base_sampler, expected_solver_sha256=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or _file_sha(path) != expected_sha256:
        raise ValueError("weak round Guidance checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or set(payload) != {"schema", "identity", "heads", "feedback_manifest", "report"}
            or payload["schema"] != SCHEMA or set(payload["heads"]) != {"macro", "micro"}):
        raise ValueError("invalid weak round Guidance schema")
    identity = payload["identity"]
    _round_identity(**identity)
    if expected_solver_sha256 is not None and identity["target_solver_sha256"] != expected_solver_sha256:
        raise ValueError("Guidance belongs to a different selected Solver")
    manifest, report = payload["feedback_manifest"], payload["report"]
    if (any(manifest.get(key) != value or report.get(key) != value for key, value in identity.items())
            or report.get("feedback_manifest_content_sha256") != _json_sha(manifest)
            or manifest.get("base_sampling_identity") != json.loads(json.dumps(base_sampler.identity))):
        raise ValueError("Guidance feedback/Base/round binding differs")
    config = RoundGuidanceConfig(**report["config"])
    if (report.get("execution_passed") is not True or report.get("completed_steps") != {"macro": config.steps, "micro": config.steps}
            or report.get("diagnostic_only") != config.diagnostic_only):
        raise ValueError("incomplete weak round Guidance cannot be loaded")
    rows = manifest.get("rows")
    if (not isinstance(rows, list) or len(rows) != 80
            or len({r.get("case_id") for r in rows}) != 80
            or len({r.get("root_seed") for r in rows}) != 80
            or any(r.get("slot") != i or r.get("split") != ("fit" if i < 64 else "validation")
                or r.get("source") != "base"
                or any(r.get(k) != identity[k] for k in ("source_round_index", "target_solver_sha256"))
                for i, r in enumerate(rows))):
        raise ValueError("Guidance fixed80 actual-feedback manifest changed")
    invalid_fit = sum(r.get("valid") is False for r in rows[:64])
    heads = {}
    for stage in ("macro", "micro"):
        head = load_guidance_payload(payload["heads"][stage], base_sampler)
        if (head.stage != stage or head.calibration["fit_count"] != 64
                or head.calibration["invalid_fit_count"] != invalid_fit):
            raise ValueError("Guidance stage or fixed64 calibration changed")
        heads[stage] = head.to(next(getattr(base_sampler, stage).parameters()).device).eval().requires_grad_(False)
    base_sampler.assert_frozen()
    if _file_sha(path) != expected_sha256:
        raise ValueError("Guidance changed while loading")
    return _snapshot(heads, deepcopy(report), path, expected_sha256)


def train_weak_round_guidance(*, base_sampler, rows, source_round_index, target_round_index,
        target_solver_sha256, reward_matcher_sha256, controller_seed, output_root, config=None,
        progress=None, label_source="canonical_selected_solver_primary_error"):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("weak round Guidance requires a Slurm compute allocation")
    config = RoundGuidanceConfig(**dict(config or {})) if isinstance(config, (dict, type(None))) else config
    if not isinstance(config, RoundGuidanceConfig):
        raise TypeError("RoundGuidanceConfig or its dictionary required")
    config.__post_init__()
    if label_source != "canonical_selected_solver_primary_error" and not (label_source == "synthetic_fixture" and config.diagnostic_only):
        raise ValueError("synthetic labels are diagnostic-only")
    identity = _round_identity(source_round_index, target_round_index, target_solver_sha256, reward_matcher_sha256, controller_seed)
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("weak round Guidance needs fresh private output")
    output.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    report = {"schema": SCHEMA, **identity, "config": asdict(config), "stage": "feedback",
        "execution_passed": False, "completed_steps": {"macro": 0, "micro": 0}, "label_source": label_source,
        "diagnostic_only": config.diagnostic_only, "scientific_admission": False, "solver_calls": 0,
        "base_training_updates": 0, "previous_guidance_loaded": False, "pilot_labels_used": False,
        "training": {}, "failures": []}
    def save():
        report["wall_seconds"] = time.monotonic() - started
        _write(output / "training.json", report)
    def check():
        if time.monotonic() - started >= config.wall_seconds:
            raise TimeoutError("weak round Guidance budget exhausted; no steps or roots reduced")
    save()
    try:
        prepared, manifest_rows, private = prepare_feedback(base_sampler, rows, identity, check=check)
        private_path = output / "feedback-draws.controller-private.pt"
        with private_path.open("xb") as stream:
            torch.save({"schema": SCHEMA, **identity, "draws": private}, stream)
        private_path.chmod(0o600)
        manifest = {"schema": SCHEMA + "_feedback", **identity, "label_source": label_source,
            "base_sampling_identity": json.loads(json.dumps(base_sampler.identity)), "rows": manifest_rows,
            "split_rule": "Controller_order_first64_fit_last16_validation_before_label_inspection",
            "invalid_labels_replace_roots": False, "private_draws": {"path": str(private_path), "sha256": _file_sha(private_path)}}
        _write(output / "feedback-manifest.json", manifest, exclusive=True)
        report.update(feedback_manifest_content_sha256=_json_sha(manifest),
            feedback_manifest={"path": str(output / "feedback-manifest.json"), "sha256": _file_sha(output / "feedback-manifest.json")},
            independent_fit_roots=64, independent_validation_roots=16,
            invalid_fit_roots=int((~prepared["fit"]["valid"]).sum()),
            invalid_validation_roots=int((~prepared["validation"]["valid"]).sum()))
        del private
        heads, validations = {}, {}
        device = next(base_sampler.micro.parameters()).device
        # torch.manual_seed seeds all visible CUDA devices; restore all of them.
        devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            for stage, key in (("macro", "white_m"), ("micro", "white_z")):
                check(); torch.manual_seed(_seed(controller_seed, "weak-init-" + stage))
                head = WeakSolverGuidanceHead(base_sampler, stage, width=32, depth=2).to(device)
                fit, development = prepared["fit"], prepared["validation"]
                head.fit_calibration(fit[key], fit["conditions"], fit["errors"], fit["valid"], **_context(stage, fit, slice(None)))
                seed = _seed(controller_seed, "weak-validation-" + stage)
                initial = _validation(head, development, seed=seed, check=check)
                _write(output / (stage + "-initial-validation.json"), initial, exclusive=True)
                _write(output / (stage + "-calibration.json"), head.calibration, exclusive=True)
                optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=config.learning_rate)
                rng = torch.Generator(device=device).manual_seed(_seed(controller_seed, "weak-training-" + stage))
                trace = []; head.train(); report["stage"] = "training_" + stage; save()
                for step in range(1, config.steps + 1):
                    check()
                    index = torch.randint(64, (4,), device=device, generator=rng)
                    clean = fit[key][index]; cfg = head.config
                    sigma = (torch.randn(4, device=device, dtype=torch.float32, generator=rng) * cfg.log_sigma_std + cfg.log_sigma_mean).exp().clamp(cfg.sigma_min, cfg.sigma_max)
                    noisy = clean + sigma[:, None] * torch.randn(clean.shape, device=device, dtype=torch.float32, generator=rng)
                    loss, metrics = head.loss(noisy, sigma, fit["conditions"][index], fit["errors"][index], fit["valid"][index], **_context(stage, fit, index))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite weak Guidance loss")
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); report["completed_steps"][stage] = step
                    if step <= 4 or step % 50 == 0 or step == config.steps:
                        trace.append({"step": step, "loss": float(loss.detach()), "gradient_norm_before_clip": float(grad), **metrics})
                        _write(output / (stage + "-training-trace.json"), trace); save()
                        if progress is not None:
                            progress(f"Weak {stage} Guidance {step}/{config.steps}", ((0 if stage == "macro" else 1) + step / config.steps) / 2)
                head.eval(); validations[stage] = _validation(head, development, seed=seed, check=check)
                _write(output / (stage + "-validation.json"), validations[stage], exclusive=True)
                report["training"][stage] = {"updates": config.steps, "sample_presentations": 4 * config.steps,
                    "trainable_parameters": sum(p.numel() for p in head.parameters() if p.requires_grad),
                    "final_loss": trace[-1]["loss"], "validation_mse": validations[stage]["mse"],
                    "validation_constant_one_mse": validations[stage]["constant_one_mse"]}
                heads[stage] = head
            base_sampler.assert_frozen(); check()
            if _file_sha(private_path) != manifest["private_draws"]["sha256"]:
                raise ValueError("actual feedback draws changed during training")
            report.update(stage="complete", execution_passed=True)
            payload = {"schema": SCHEMA, "identity": identity, "heads": {name: guidance_payload(head) for name, head in heads.items()},
                "feedback_manifest": manifest, "report": deepcopy(report)}
            path = output / f"fixed-step-{config.steps:06d}.pt"
            with path.open("xb") as stream:
                torch.save(payload, stream)
            path.chmod(0o400); digest = _file_sha(path)
            restored = load_weak_round_guidance(path, digest, base_sampler=base_sampler, expected_solver_sha256=target_solver_sha256)
            for stage in heads:
                before, after = heads[stage].state_dict(), restored.heads[stage].state_dict()
                if before.keys() != after.keys() or any(not torch.equal(before[k], after[k]) for k in before):
                    raise ValueError("weak Guidance checkpoint state differs")
                repeated = _validation(restored.heads[stage], prepared["validation"], seed=_seed(controller_seed, "weak-validation-" + stage), check=check)
                if repeated != validations[stage]:
                    raise ValueError("weak Guidance reload changed fixed predictions or score-gradient bytes")
        base_sampler.assert_frozen()
        report.update(checkpoint={"path": str(path), "sha256": digest}, reload_parameters_predictions_gradient_exact=True)
        save()
        return _snapshot(restored.heads, deepcopy(report), path, digest)
    except BaseException as error:
        report.update(stage="failed", execution_passed=False)
        report["failures"].append({"type": type(error).__name__, "message": str(error)})
        save()
        raise
