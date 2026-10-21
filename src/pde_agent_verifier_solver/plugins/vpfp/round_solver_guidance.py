"""Train the validated VPFP guide from this Campaign round's actual feedback.

The Controller supplies 80 independently drawn, clean BASE answers and the
previous selected Solver's errors. This module never runs a Solver, resamples
an answer, or reads historical labels/checkpoints. The 64 ``fit-xxxx`` names
are local calibration slots ONLY; a private manifest retains real case/root
identities, selected-Solver identity and the actual tensors for every round.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping


SCHEMA = "vpfp_campaign_round_solver_guidance_v1"
_DRAW_TENSORS = ("full", "initial_context", "future_white", "equation_conditions")
_DRAW_STREAMS = ("root_seeds", "known_prior_seeds", "future_noise_seeds")


@dataclass(frozen=True)
class RoundGuidanceConfig:
    steps: int = 2000
    batch_size: int = 4
    learning_rate: float = 2.e-4
    beta_std: float = .5
    fit_count: int = 64
    validation_count: int = 16
    target_ratio: float = .1
    max_flow_ratio: float = .25
    wall_seconds: float = 1800.

    def __post_init__(self):
        fixed = {"batch_size": 4, "learning_rate": 2.e-4, "beta_std": .5,
                 "fit_count": 64, "validation_count": 16,
                 "target_ratio": .1, "max_flow_ratio": .25}
        if type(self.steps) is not int or self.steps not in (2, 3, 4, 2000):
            raise ValueError("2000 updates required; 2-4 updates are diagnostic-only")
        for name, expected in fixed.items():
            value = getattr(self, name)
            if (type(value) not in (int, float) or value != expected
                    or (name in ("batch_size", "fit_count", "validation_count") and type(value) is not int)):
                raise ValueError("round Guidance preserves the validated setting: " + name)
        if type(self.wall_seconds) not in (int, float) or not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("Guidance wall_seconds must be a finite positive allocation")

    @property
    def diagnostic_only(self):
        return self.steps != 2000


@dataclass(frozen=True)
class TrainedRoundGuidance:
    guide: object
    sampler: object
    checkpoint_path: str
    checkpoint_sha256: str
    manifest_sha256: str
    report: Mapping[str, object]


def _json_sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = sha256(json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write(path, payload, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(label + " requires a SHA256")


def _seed(parent, label):
    return int.from_bytes(sha256(f"{SCHEMA}:{parent}:{label}".encode()).digest()[:8], "little") % 2**63


def _round_identity(source_round_index, target_round_index, target_solver_sha256,
                    reward_matcher_sha256, controller_seed):
    if (type(source_round_index) is not int or source_round_index < 0
            or type(target_round_index) is not int or target_round_index != source_round_index + 1):
        raise ValueError("Guidance target round must immediately follow source round")
    if type(controller_seed) is not int or not 0 <= controller_seed < 2**63:
        raise ValueError("nonnegative signed-64-bit Controller seed required")
    _sha(target_solver_sha256, "selected Solver")
    _sha(reward_matcher_sha256, "reward matcher")
    return {"source_round_index": source_round_index, "target_round_index": target_round_index,
            "target_solver_sha256": target_solver_sha256, "reward_matcher_sha256": reward_matcher_sha256,
            "controller_seed": controller_seed}


def _prepare_rows(base, rows, identity):
    """Snapshot exact supplied draws; reject non-BASE/duplicates/misbound labels."""
    import torch
    from . import solver_error_guidance as legacy

    if not isinstance(rows, (tuple, list)) or len(rows) != 80:
        raise ValueError("exactly 80 actual clean BASE feedback roots required; no silent filtering")
    cfg, parameter = base.model.config, next(base.model.parameters())
    if (parameter.dtype != torch.float32 or base.model.training
            or any(p.requires_grad for p in base.model.parameters())
            or base.normalizer.training or any(p.requires_grad for p in base.normalizer.parameters())):
        raise ValueError("the native FP32 Base and normalizer must already be eval/frozen")
    private, manifest_rows = [], []
    seen_ids, seen_roots = set(), set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("Guidance feedback rows must be mappings")
        case_id, root = row.get("case_id"), row.get("root_seed")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
            raise ValueError("80 unique actual feedback case IDs required")
        if type(root) is not int or not 0 <= root < 2**63 or root in seen_roots:
            raise ValueError("80 unique actual physical roots required before noising/splitting")
        if row.get("source") != "base":
            raise ValueError("round Guidance fits only clean BASE feedback, never guided/perturbed answers")
        for name in ("source_round_index", "target_solver_sha256"):
            if row.get(name) != identity[name]:
                raise ValueError("feedback changed " + name)
        valid, error = row.get("valid"), row.get("error")
        if type(valid) is not bool:
            raise ValueError("feedback numerical validity must be explicit boolean")
        if valid and (type(error) not in (int, float) or not math.isfinite(error) or error < 0):
            raise ValueError("valid feedback needs the finite nonnegative actual Solver error")
        if not valid and error is not None:
            raise ValueError("invalid feedback error must be unavailable, not a high reward or imputed penalty")
        draw = row.get("draw")
        if not isinstance(draw, Mapping):
            raise TypeError("each feedback row requires its actual completed draw")
        for name, expected in (("root_seeds", root),
                ("known_prior_seeds", int.from_bytes(sha256(f"vpfp-initial-conditioned-prior:{root}".encode()).digest()[:8], "little") % 2**63),
                ("future_noise_seeds", int.from_bytes(sha256(f"vpfp-initial-conditioned-future:{root}".encode()).digest()[:8], "little") % 2**63)):
            values = draw.get(name)
            if not isinstance(values, (list, tuple)) or len(values) != 1 or type(values[0]) is not int or values[0] != expected:
                raise ValueError("actual draw changed its " + name)
        captured = {name: list(draw[name]) for name in _DRAW_STREAMS}
        for name in _DRAW_TENSORS:
            value = draw.get(name)
            if (not isinstance(value, torch.Tensor) or value.ndim != 2 or len(value) != 1
                    or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                raise ValueError("actual draw needs a finite one-row native FP32 " + name)
            captured[name] = value.detach().to(device=parameter.device).clone()
        future, context, conditions = captured["future_white"], captured["initial_context"], captured["equation_conditions"]
        legacy._coefficients(future, cfg, device=parameter.device, dtype=parameter.dtype)
        legacy._context(context, cfg, prototype=future)
        base.model._conditions(conditions, 1, future)
        with torch.no_grad():
            decoded = base.normalizer.decode((context, future))
        if not torch.equal(captured["full"], decoded):
            raise ValueError("actual full answer differs from its supplied clean IC/future coordinates")
        split = "fit" if index < 64 else "validation"
        slot = f"{split}-{index if index < 64 else index - 64:04d}"
        metadata = {"case_id": case_id, "root_seed": root, "source": "base",
            "source_round_index": identity["source_round_index"], "target_solver_sha256": identity["target_solver_sha256"],
            "valid": valid, "error": float(error) if valid else None, "split": split, "local_slot": slot,
            "draw_streams": {name: captured[name] for name in _DRAW_STREAMS},
            "draw_tensor_sha256": {name: _tensor_sha(captured[name]) for name in _DRAW_TENSORS}}
        private.append(captured)
        manifest_rows.append(metadata)
        seen_ids.add(case_id)
        seen_roots.add(root)
    prepared = {}
    for split, start, stop in (("fit", 0, 64), ("validation", 64, 80)):
        selected = manifest_rows[start:stop]
        prepared[split] = {"split": split, "root_ids": [row["local_slot"] for row in selected], "excluded_rows": [],
            "future_white": torch.cat([row["future_white"] for row in private[start:stop]]),
            "initial_context": torch.cat([row["initial_context"] for row in private[start:stop]]),
            "conditions": torch.cat([row["equation_conditions"] for row in private[start:stop]]),
            "errors": torch.tensor([row["error"] if row["valid"] else float("nan") for row in selected], dtype=torch.float64, device=parameter.device),
            "valid": torch.tensor([row["valid"] for row in selected], dtype=torch.bool, device=parameter.device)}
    return prepared, manifest_rows, private


def _validation_cache(base, rows, *, seed, check):
    import torch
    rng = torch.Generator(device=rows["future_white"].device).manual_seed(seed)
    result = []
    for index, root in enumerate(rows["root_ids"]):
        for sigma_value in (.002, .05, .2, 1., 2.):
            for replicate in range(2):
                check()
                clean = rows["future_white"][index:index+1]
                sigma = clean.new_tensor([sigma_value])
                noisy = clean + sigma[:, None] * torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=rng)
                initial, condition = rows["initial_context"][index:index+1], rows["conditions"][index:index+1]
                with torch.no_grad():
                    denoised = base.model.denoise(noisy, sigma, condition, initial_context=initial)
                result.append({"index": index, "root": root, "replicate": replicate,
                    "noisy": noisy, "sigma": sigma, "initial": initial, "condition": condition, "denoised": denoised})
    return result


def _probe(model, cache, rows, *, check):
    import torch
    from .standardized_solver_error_guidance import standardized_targets
    targets, valid = standardized_targets(rows["errors"], rows["valid"], prototype=rows["future_white"],
        fit_reward_mean=model.fit_reward_mean, fit_reward_std=model.fit_reward_std,
        log_fit_mean_exp=model.log_fit_mean_exp, beta_std=model.beta_std, error_scale=model.config.error_scale)
    records = []
    for item in cache:
        check()
        with torch.enable_grad():
            state = item["noisy"].detach().requires_grad_(True)
            output = model(state, item["sigma"], item["condition"], initial_context=item["initial"])
            gradient = torch.autograd.grad(model.score_potential(output).sum(), state)[0]
        norm = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
        scale = torch.where(norm > 1., 1. / norm.clamp_min(torch.finfo(norm.dtype).tiny), torch.ones_like(norm))
        displacement = (item["denoised"] + item["sigma"].square()[:, None] * gradient * scale) - item["denoised"]
        predicted = float(output["desirability"].detach()[0])
        record = {"local_slot": item["root"], "sigma": float(item["sigma"][0]), "replicate": item["replicate"],
            "valid": bool(valid[item["index"]]), "target": float(targets[item["index"]]), "desirability": predicted,
            "log_desirability": float(output["log_desirability"].detach()[0]),
            "gradient_l2": float(torch.linalg.vector_norm(gradient.detach().double())),
            "full_gradient_sha256": _tensor_sha(gradient), "full_displacement_sha256": _tensor_sha(displacement),
            "actual_unit_lambda_displacement_nonzero": bool(torch.any(displacement != 0.))}
        if any(not math.isfinite(record[key]) for key in ("target", "desirability", "log_desirability", "gradient_l2")):
            raise FloatingPointError("nonfinite fixed development probe; no silent clipping")
        records.append(record)
    return {"records": records, "summary": {"record_count": len(records), "independent_roots": 16,
        "desirability_mse": math.fsum((r["desirability"]-r["target"])**2 for r in records) / len(records),
        "fit_constant_mse": math.fsum((1.-r["target"])**2 for r in records) / len(records),
        "nonzero_gradient_records": sum(r["gradient_l2"] > 0. for r in records),
        "regression_gate": False, "diagnostic_external_lambda": 1., "solver_hardness_measured": False}}


def train_round_solver_guidance(*, base_sampler, rows, source_round_index,
        target_round_index, target_solver_sha256, reward_matcher_sha256,
        controller_seed, output_root, config=None, progress=None,
        label_source="actual_public_solver_macro_error"):
    """Fit once from supplied real feedback; return a frozen relative-RMS sampler.

    Rows require case_id/root_seed/source='base'/error/valid/draw plus
    source_round_index and target_solver_sha256. Invalid errors must be None:
    they receive zero desirability, never an imputed large reward. Non-BASE,
    duplicate or mismatched rows fail the whole boundary without silent drops.
    """
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("round Guidance numerical work requires a Slurm compute allocation")
    config = config or RoundGuidanceConfig()
    if not isinstance(config, RoundGuidanceConfig):
        raise TypeError("RoundGuidanceConfig required")
    config.__post_init__()
    if label_source not in ("actual_public_solver_macro_error", "synthetic_fixture") or (label_source == "synthetic_fixture" and not config.diagnostic_only):
        raise ValueError("synthetic fixtures are diagnostic-only, never real training evidence")
    identity = _round_identity(source_round_index, target_round_index, target_solver_sha256,
                               reward_matcher_sha256, controller_seed)
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("round Guidance requires a fresh private output directory")
    output.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    report = {"schema": SCHEMA, **identity, "config": asdict(config), "stage": "validate_feedback",
        "execution_passed": False, "completed_step": 0, "target_step": config.steps,
        "diagnostic_only": config.diagnostic_only, "label_source": label_source,
        "scientific_admission": False, "online_assets_replaced": False,
        "base_training_updates": 0, "solver_calls": 0, "solver_training_updates": 0,
        "regression_gate": False, "failures": []}

    def save():
        report["wall_seconds"] = time.monotonic() - started
        _write(output / "training.json", report)

    def check():
        if time.monotonic() - started >= config.wall_seconds:
            raise TimeoutError("round Guidance wall allocation exhausted; required updates/probes are not reduced")

    def notify(message, fraction):
        if progress is not None:
            progress(message, fraction)

    save()
    try:
        import torch
        from . import solver_error_guidance as legacy
        from .standardized_solver_error_guidance import (make_standardized_regressor,
            fit_standardized_calibration, standardized_payload)
        from .rescaled_solver_error_guidance import RescaledIC40Sampler, RescalingPolicy

        check()
        parameter = next(base_sampler.model.parameters())
        base_sha = legacy._module_sha(base_sampler.model)
        normalizer_sha = legacy._json_sha(base_sampler.normalizer.to_payload())
        base_identity_sha = legacy._json_sha(base_sampler.identity)
        prepared, manifest_rows, private = _prepare_rows(base_sampler, rows, identity)
        private_path = output / "feedback-draws.controller-private.pt"
        with private_path.open("xb") as stream:
            torch.save({"schema": SCHEMA, **identity,
                "draws": [{name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                           for name, value in draw.items()} for draw in private]}, stream)
        private_path.chmod(0o600)
        manifest = {"schema": SCHEMA + "_feedback", **identity, "label_source": label_source,
            "base_sampling_identity": deepcopy(base_sampler.identity), "base_state_sha256": base_sha,
            "normalizer_sha256": normalizer_sha, "rows": manifest_rows,
            "split_rule": "Controller_feedback_order_first_64_fit_last_16_development_before_noising",
            "local_slots_are_physical_root_identity": False,
            "independence_unit": "actual unique root_seed; all noisy replicas keep their root split",
            "private_draws": {"path": str(private_path), "sha256": _file_sha(private_path)}}
        manifest_path = output / "feedback-manifest.json"
        _write(manifest_path, manifest, exclusive=True)
        manifest_path.chmod(0o444)
        manifest_sha = _file_sha(manifest_path)
        report.update(feedback_manifest={"path": str(manifest_path), "sha256": manifest_sha},
            independent_fit_roots=64, independent_validation_roots=16,
            invalid_fit_roots=int((~prepared["fit"]["valid"]).sum()),
            invalid_validation_roots=int((~prepared["validation"]["valid"]).sum()),
            base_checkpoint_sha256=base_sampler.identity["checkpoint_sha256"],
            base_state_sha256=base_sha, normalizer_sha256=normalizer_sha,
            fixed_base_identity_sha256=base_identity_sha)
        save()
        del private
        seeds = {name: _seed(controller_seed, name) for name in ("initialization", "calibration", "training", "validation")}
        report["derived_seeds"] = seeds
        devices = list(range(torch.cuda.device_count())) if parameter.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seeds["initialization"])
            gc = legacy.SolverErrorGuidanceConfig(base_sampler.model.config,
                base_sampler.identity["checkpoint_sha256"], normalizer_sha,
                width=32, depth=2, error_scale=1., validity_weight=.1)
            model = make_standardized_regressor(gc, parameter_dtype=torch.float32,
                device=parameter.device, beta_std=config.beta_std)
            report["stage"] = "fit_calibration"
            save()
            calibration = fit_standardized_calibration(model, prepared["fit"],
                seed=seeds["calibration"], batch_size=config.batch_size)
            _write(output / "calibration.json", calibration, exclusive=True)
            cache = _validation_cache(base_sampler, prepared["validation"], seed=seeds["validation"], check=check)
            model.eval()
            initial_validation = _probe(model, cache, prepared["validation"], check=check)
            _write(output / "initial-validation.json", initial_validation, exclusive=True)
            report.update(stage="training", calibration=calibration,
                initial_validation=initial_validation["summary"], initialization_state_sha256=legacy._module_sha(model))
            save()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            rng = torch.Generator(device=parameter.device).manual_seed(seeds["training"])
            trace = []
            model.train()
            notify(f"Training VPFP Guidance against the selected round-{source_round_index} Solver", 0.)
            fit = prepared["fit"]
            for step in range(1, config.steps + 1):
                check()
                indices = torch.randint(64, (config.batch_size,), device=parameter.device, generator=rng)
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = model.loss(fit["future_white"][indices], fit["conditions"][indices],
                    fit["initial_context"][indices], fit["errors"][indices], fit["valid"][indices], generator=rng)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite round Guidance loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                report["completed_step"] = step
                if step <= 5 or step % 100 == 0 or step == config.steps:
                    trace.append({"step": step, "loss": float(loss.detach()),
                        "regression_loss": float(metrics["regression_loss"]),
                        "validity_loss": float(metrics["validity_loss"]),
                        "gradient_norm_before_clip": float(gradient_norm)})
                    _write(output / "training-trace.json", trace)
                    save()
                    notify(f"VPFP Guidance {step}/{config.steps} updates", .9 * step / config.steps)
            model.eval()
            report["stage"] = "freeze_and_verify"
            save()
            validation = _probe(model, cache, prepared["validation"], check=check)
            _write(output / "validation.json", validation, exclusive=True)
            training = {"label_source": label_source, "step": config.steps, **identity,
                "seed": seeds["initialization"], "objective": model.objective,
                "checkpoint_choice": "fixed_step_not_best", "diagnostic_only": config.diagnostic_only,
                "base_trainable": False, "solver_trainable": False,
                "real_case_ids": [row["case_id"] for row in manifest_rows],
                "real_root_seeds": [row["root_seed"] for row in manifest_rows],
                "fit_real_root_seeds": [row["root_seed"] for row in manifest_rows[:64]],
                "validation_real_root_seeds": [row["root_seed"] for row in manifest_rows[64:]],
                "feedback_distribution": "fresh_clean_BASE", "previous_guide_loaded": False}
            payload = standardized_payload(model, label_manifest_sha256=manifest_sha,
                train_root_ids=fit["root_ids"], validation_root_ids=prepared["validation"]["root_ids"], training=training)
            checkpoint = output / f"fixed-step-{config.steps:06d}.pt"
            with checkpoint.open("xb") as stream:
                torch.save(payload, stream)
            checkpoint.chmod(0o444)
            checkpoint_sha = _file_sha(checkpoint)
            restored = legacy.load_solver_error_regressor(checkpoint, checkpoint_sha, device=parameter.device)
            replay = _probe(restored, cache, prepared["validation"], check=check)
            if replay != validation:
                raise ValueError("round Guidance reload changed development scores, full gradients or actual FP32 displacements")
            policy = RescalingPolicy(mode="relative", target_ratio=config.target_ratio, max_flow_ratio=config.max_flow_ratio)
            sampler = RescaledIC40Sampler(base_sampler, restored, policy, max_gradient_norm=1., sigma_min=.002, sigma_max=2.)
            sampler.assert_frozen()
            with (output / "continuation.controller-private.pt").open("xb") as stream:
                torch.save({"schema": SCHEMA + "_continuation", **identity, "step": config.steps,
                    "checkpoint_sha256": checkpoint_sha, "optimizer": optimizer.state_dict(),
                    "training_rng": rng.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if parameter.is_cuda else []}, stream)
            (output / "continuation.controller-private.pt").chmod(0o600)
        check()
        report.update(stage="complete", execution_passed=True, final_step=config.steps,
            checkpoint={"path": str(checkpoint), "sha256": checkpoint_sha},
            validation=validation["summary"], parameter_count=sum(p.numel() for p in restored.parameters()),
            sample_presentations=config.steps * config.batch_size, objective=restored.objective,
            beta_std=restored.beta_std, effective_beta=restored.beta,
            reload_full_gradient_and_displacement_hashes_exact=True, reload_probe_count=len(cache),
            base_final_verified=True, feedback_final_verified=True, sampler_identity=deepcopy(sampler.identity),
            policy=policy.to_dict(), actual_solver_hardness_measured=False)
        save()
        notify("VPFP Guidance trained on this round's selected Solver and frozen", 1.)
        return TrainedRoundGuidance(restored, sampler, str(checkpoint), checkpoint_sha, manifest_sha, deepcopy(report))
    except BaseException as error:
        report["failures"].append({"stage": report["stage"], "type": type(error).__name__, "message": str(error)})
        report["stage"] = "failed"
        save()
        raise


__all__ = ["RoundGuidanceConfig", "TrainedRoundGuidance", "train_round_solver_guidance"]
