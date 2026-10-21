#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.fusion_simulation_first.coverage import (
    effective_rank,
    prdc_metrics,
    rbf_mmd,
    sliced_wasserstein,
    standardize_features,
)
from agent_verifier.neural_generator.structured_ns_distribution import (
    STRUCTURED_NS_FLOW_SCHEMA,
    StructuredCoefficientFlowConfig,
    StructuredCoefficientPCA,
    StructuredCoefficientRectifiedFlow,
    compatible_projection_signatures,
    load_structured_coefficient_corpus,
)
from agent_verifier.neural_generator.structured_ns_guidance import (
    StructuredRewardGradient,
    load_structured_reward_guidance_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and audit a structured NS coefficient rectified flow."
    )
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--pca-checkpoint", default="")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--sample-batch", type=int, default=256)
    parser.add_argument("--integration-steps", type=int, default=64)
    parser.add_argument("--train-audit-cases", type=int, default=4096)
    parser.add_argument("--validation-audit-cases", type=int, default=1024)
    parser.add_argument("--metric-rank", type=int, default=64)
    parser.add_argument("--baseline-repeats", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--state", choices=("ema", "raw"), default="ema")
    parser.add_argument("--guidance-checkpoint", default="")
    parser.add_argument("--guidance-strength", type=float, default=0.10)
    parser.add_argument("--guidance-max-flow-ratio", type=float, default=0.25)
    parser.add_argument("--paired-base-audit-cases", type=int, default=128)
    parser.add_argument(
        "--sampling-policy",
        choices=("rectified_flow", "pca_standard_gaussian"),
        default="rectified_flow",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if min(
        int(args.samples),
        int(args.sample_batch),
        int(args.integration_steps),
        int(args.train_audit_cases),
        int(args.validation_audit_cases),
        int(args.metric_rank),
    ) < 1:
        raise SystemExit("structured flow sampling dimensions must be positive")
    if int(args.paired_base_audit_cases) < 0:
        raise SystemExit("paired base audit count cannot be negative")
    if int(args.baseline_repeats) < 2:
        raise SystemExit("empirical D0 baseline needs at least two repeats")
    if float(args.guidance_strength) < 0.0:
        raise SystemExit("guidance strength cannot be negative")
    if float(args.guidance_max_flow_ratio) <= 0.0:
        raise SystemExit("guidance max flow ratio must be positive")
    output = Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse structured flow sample output: {output}")
    output.mkdir(parents=True)
    torch = _torch()
    device = _device(str(args.device))
    started = time.perf_counter()

    flow_path = Path(args.flow_checkpoint).resolve()
    payload = torch.load(flow_path, map_location=device, weights_only=False)
    if payload.get("schema") != STRUCTURED_NS_FLOW_SCHEMA:
        raise SystemExit("unexpected structured flow checkpoint schema")
    pca_path = Path(
        args.pca_checkpoint or payload.get("pca_checkpoint", "")
    ).resolve()
    if not pca_path.is_file():
        raise SystemExit("structured flow PCA checkpoint is missing")
    pca = StructuredCoefficientPCA.load(pca_path, map_location=device).with_rank(
        int(payload["pca_rank"])
    )
    model = None
    reward_model = None
    guidance = None
    guidance_payload = None
    if str(args.sampling_policy) == "rectified_flow":
        model_config = StructuredCoefficientFlowConfig.from_dict(
            dict(payload["model_config"])
        )
        model = StructuredCoefficientRectifiedFlow.build(model_config).to(device)
        state_key = (
            "ema_state_dict" if str(args.state) == "ema" else "model_state_dict"
        )
        model.load_state_dict(payload[state_key])
        model.eval()
        if str(args.guidance_checkpoint):
            guidance_path = Path(args.guidance_checkpoint).resolve()
            reward_model, guidance_payload = (
                load_structured_reward_guidance_checkpoint(
                    guidance_path,
                    map_location=device,
                )
            )
            if int(reward_model.config.feature_dim) != int(payload["pca_rank"]):
                raise SystemExit("reward guidance feature dimension is incompatible")
            if guidance_payload.get("reward_metric") != "relative_l2":
                raise SystemExit("reward guidance does not target relative_l2")
            if guidance_payload.get("reward_objective") != "solver_error":
                raise SystemExit("reward guidance does not target pure solver error")
            guidance = StructuredRewardGradient(
                reward_model,
                strength=float(args.guidance_strength),
                max_flow_ratio=float(args.guidance_max_flow_ratio),
            )
    elif str(args.guidance_checkpoint):
        raise SystemExit("reward guidance requires rectified_flow sampling")

    train = load_structured_coefficient_corpus(
        args.train_manifest,
        max_cases=int(args.train_audit_cases),
        device=device,
    )
    validation = load_structured_coefficient_corpus(
        args.validation_manifest,
        max_cases=int(args.validation_audit_cases),
        device=device,
    )
    for corpus, label in ((train, "train"), (validation, "validation")):
        if not compatible_projection_signatures(
            pca.source_signature,
            dict(corpus.manifest["encoding_signature"]),
        ):
            raise SystemExit(f"{label} audit coefficients are incompatible with PCA")
    with torch.no_grad():
        train_scores = pca.encode(train.coefficients).float()
        validation_scores = pca.encode(validation.coefficients).float()
    del train, validation

    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    generated_parts = []
    paired_base_parts = []
    paired_remaining = (
        min(int(args.paired_base_audit_cases), int(args.samples))
        if guidance is not None
        else 0
    )
    with torch.no_grad():
        for start in range(0, int(args.samples), int(args.sample_batch)):
            count = min(int(args.sample_batch), int(args.samples) - start)
            if str(args.sampling_policy) == "rectified_flow":
                initial_values = torch.randn(
                    (count, int(payload["pca_rank"])),
                    device=device,
                    generator=generator,
                )
                generated = model.sample(
                    count,
                    steps=int(args.integration_steps),
                    device=device,
                    initial_values=initial_values,
                    guidance=guidance,
                    guidance_scale=1.0 if guidance is not None else 0.0,
                    method="heun",
                ).float()
                generated_parts.append(generated)
                paired_count = min(count, paired_remaining)
                if paired_count:
                    paired_base_parts.append(
                        model.sample(
                            paired_count,
                            steps=int(args.integration_steps),
                            device=device,
                            initial_values=initial_values[:paired_count],
                            method="heun",
                        ).float()
                    )
                    paired_remaining -= paired_count
            else:
                generated_parts.append(
                    torch.randn(
                        (count, int(payload["pca_rank"])),
                        device=device,
                        generator=generator,
                    ).float()
                )
    generated_scores = torch.cat(generated_parts, dim=0)
    paired_base_scores = (
        torch.cat(paired_base_parts, dim=0) if paired_base_parts else None
    )
    with torch.no_grad():
        generated_coefficients = pca.decode(generated_scores).float()

    train_numpy = train_scores.detach().cpu().numpy()
    validation_numpy = validation_scores.detach().cpu().numpy()
    generated_numpy = generated_scores.detach().cpu().numpy()
    metric_rank = min(
        int(args.metric_rank),
        train_numpy.shape[1],
        validation_numpy.shape[1],
        generated_numpy.shape[1],
    )
    train_metric, validation_metric, generated_metric = standardize_features(
        train_numpy[:, :metric_rank],
        validation_numpy[:, :metric_rank],
        generated_numpy[:, :metric_rank],
    )
    comparison_count = min(
        generated_metric.shape[0],
        validation_metric.shape[0] // 2,
    )
    if comparison_count < 2:
        raise SystemExit(
            "validation audit needs two D0 groups with at least two cases each"
        )
    comparison_rng = np.random.default_rng(int(args.seed))
    comparison_indices = comparison_rng.permutation(
        validation_metric.shape[0]
    )[:comparison_count]
    validation_comparison = validation_metric[comparison_indices]
    generated_comparison = generated_metric[:comparison_count]
    empirical_trials = _empirical_two_sample_trials(
        validation_metric,
        sample_count=comparison_count,
        repeats=int(args.baseline_repeats),
        seed=int(args.seed),
    )
    first_baseline = empirical_trials[0]
    distribution = {
        "metric_rank": int(metric_rank),
        "comparison_case_count": int(comparison_count),
        "empirical_baseline_repeats": int(args.baseline_repeats),
        "empirical_d0_trials": empirical_trials,
        "generated_vs_validation_prdc": prdc_metrics(
            validation_comparison,
            generated_comparison,
        ),
        "validation_half_baseline_prdc": {
            name: first_baseline[name]
            for name in ("precision", "recall", "density", "coverage")
        },
        "generated_vs_validation_rbf_mmd": rbf_mmd(
            validation_comparison,
            generated_comparison,
        ),
        "validation_half_baseline_rbf_mmd": first_baseline["mmd"],
        "generated_vs_validation_sliced_wasserstein": sliced_wasserstein(
            validation_comparison,
            generated_comparison,
            seed=int(args.seed),
        ),
        "validation_half_baseline_sliced_wasserstein": (
            first_baseline["sliced_wasserstein"]
        ),
        "nearest_train_distance": {
            "validation": _nearest_distance_summary(validation_metric, train_metric),
            "generated": _nearest_distance_summary(generated_metric, train_metric),
        },
        "score_statistics": {
            "train": _score_statistics(train_numpy),
            "validation": _score_statistics(validation_numpy),
            "generated": _score_statistics(generated_numpy),
        },
        "effective_rank": {
            "train": effective_rank(train_metric),
            "validation": effective_rank(validation_metric),
            "generated": effective_rank(generated_metric),
        },
    }
    sample_path = output / "structured_flow_samples.npz"
    sample_arrays = {
        "scores": generated_numpy.astype(np.float32),
        "coefficients": generated_coefficients.detach().cpu().numpy().astype(np.float32),
    }
    if paired_base_scores is not None:
        sample_arrays["paired_base_scores"] = (
            paired_base_scores.detach().cpu().numpy().astype(np.float32)
        )
    np.savez_compressed(sample_path, **sample_arrays)
    guidance_report = None
    if guidance is not None:
        assert reward_model is not None and guidance_payload is not None
        pair_count = int(paired_base_scores.shape[0])
        with torch.no_grad():
            endpoint_time = torch.ones(pair_count, device=device)
            guided_prediction = reward_model(
                generated_scores[:pair_count], endpoint_time
            )
            base_prediction = reward_model(paired_base_scores, endpoint_time)
        normalization = dict(guidance_payload["reward_normalization"])
        reward_mean = float(normalization["mean"])
        reward_std = float(normalization["std"])
        guided_raw = guided_prediction * reward_std + reward_mean
        base_raw = base_prediction * reward_std + reward_mean
        guidance_path = Path(args.guidance_checkpoint).resolve()
        guidance_report = {
            "checkpoint": str(guidance_path),
            "checkpoint_sha256": _file_sha256(guidance_path),
            "target_solver": str(guidance_payload["target_solver"]),
            "target_solver_sha256": str(
                guidance_payload["target_solver_sha256"]
            ),
            "strength": float(args.guidance_strength),
            "max_flow_ratio": float(args.guidance_max_flow_ratio),
            "gradient_refresh": "every_heun_velocity_evaluation",
            "base_flow_frozen": True,
            "diagnostics": guidance.diagnostics(),
            "paired_base_audit": {
                "cases": pair_count,
                "same_initial_noise": True,
                "base_predicted_reward_mean": float(base_raw.mean().cpu()),
                "guided_predicted_reward_mean": float(guided_raw.mean().cpu()),
                "mean_predicted_reward_increase": float(
                    (guided_raw - base_raw).mean().cpu()
                ),
                "fraction_predicted_harder": float(
                    (guided_raw > base_raw).float().mean().cpu()
                ),
                "mean_score_shift_rms": float(
                    (generated_scores[:pair_count] - paired_base_scores)
                    .square()
                    .mean(dim=1)
                    .sqrt()
                    .mean()
                    .cpu()
                ),
            },
        }
    report = {
        "schema": "ns2d_structured_coefficient_flow_sample_audit_v1",
        "flow_checkpoint": str(flow_path),
        "flow_checkpoint_sha256": _file_sha256(flow_path),
        "flow_step": int(payload["step"]),
        "flow_state": str(args.state),
        "sampling_policy": (
            "solver_reward_guided_rectified_flow"
            if guidance is not None
            else str(args.sampling_policy)
        ),
        "pca_checkpoint": str(pca_path),
        "pca_rank": int(payload["pca_rank"]),
        "samples": int(args.samples),
        "integration_steps": int(args.integration_steps),
        "seed": int(args.seed),
        "sample_file": sample_path.name,
        "sample_file_sha256": _file_sha256(sample_path),
        "distribution": distribution,
        "guidance": guidance_report,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output / "sample_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "sample_audit.md").write_text(_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)



def _empirical_two_sample_trials(
    reference: np.ndarray,
    *,
    sample_count: int,
    repeats: int,
    seed: int,
) -> list[dict[str, float]]:
    """Measure the finite-sample variation between independent D0 subsets."""

    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 2 or 2 * int(sample_count) > values.shape[0]:
        raise ValueError("empirical D0 trials need two disjoint equal-size groups")
    random = np.random.default_rng(int(seed))
    rows = []
    for repeat_index in range(int(repeats)):
        indices = random.permutation(values.shape[0])
        left = values[indices[:sample_count]]
        right = values[indices[sample_count : 2 * sample_count]]
        prdc = prdc_metrics(left, right)
        rows.append(
            {
                "repeat": repeat_index,
                "mmd": float(rbf_mmd(left, right)),
                "sliced_wasserstein": float(
                    sliced_wasserstein(
                        left,
                        right,
                        seed=int(seed) + repeat_index,
                    )
                ),
                "precision": float(prdc["precision"]),
                "recall": float(prdc["recall"]),
                "density": float(prdc["density"]),
                "coverage": float(prdc["coverage"]),
            }
        )
    return rows

def _nearest_distance_summary(query: np.ndarray, reference: np.ndarray):
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    minimum_parts = []
    for start in range(0, query.shape[0], 256):
        values = query[start : start + 256]
        squared = (
            np.sum(values * values, axis=1)[:, None]
            + np.sum(reference * reference, axis=1)[None, :]
            - 2.0 * values @ reference.T
        )
        minimum_parts.append(np.sqrt(np.maximum(np.min(squared, axis=1), 0.0)))
    minimum = np.concatenate(minimum_parts)
    return {
        "mean": float(np.mean(minimum)),
        "median": float(np.median(minimum)),
        "q10": float(np.quantile(minimum, 0.1)),
        "q90": float(np.quantile(minimum, 0.9)),
        "exact_or_near_copy_fraction": float(np.mean(minimum <= 1.0e-6)),
    }


def _score_statistics(values: np.ndarray):
    array = np.asarray(values, dtype=np.float64)
    coordinate_mean = np.mean(array, axis=0)
    coordinate_std = np.std(array, axis=0)
    case_rms = np.sqrt(np.mean(array * array, axis=1))
    return {
        "coordinate_abs_mean_max": float(np.max(np.abs(coordinate_mean))),
        "coordinate_std_mean": float(np.mean(coordinate_std)),
        "coordinate_std_median": float(np.median(coordinate_std)),
        "coordinate_std_min": float(np.min(coordinate_std)),
        "coordinate_std_max": float(np.max(coordinate_std)),
        "case_rms_mean": float(np.mean(case_rms)),
        "case_rms_median": float(np.median(case_rms)),
        "case_rms_iqr": float(np.quantile(case_rms, 0.75) - np.quantile(case_rms, 0.25)),
        "case_rms_min": float(np.min(case_rms)),
        "case_rms_q10": float(np.quantile(case_rms, 0.10)),
        "case_rms_q90": float(np.quantile(case_rms, 0.90)),
        "case_rms_q99": float(np.quantile(case_rms, 0.99)),
        "case_rms_max": float(np.max(case_rms)),
    }


def _markdown(report):
    distribution = report["distribution"]
    prdc = distribution["generated_vs_validation_prdc"]
    nearest = distribution["nearest_train_distance"]
    return "\n".join(
        (
            "# Structured NS coefficient flow sample audit",
            "",
            f"- checkpoint step: `{report['flow_step']}`",
            f"- sampling policy: `{report['sampling_policy']}`",
            f"- samples: `{report['samples']}`",
            f"- PRDC precision: `{prdc['precision']:.6g}`",
            f"- PRDC recall: `{prdc['recall']:.6g}`",
            f"- generated nearest-train median: `{nearest['generated']['median']:.6g}`",
            f"- validation nearest-train median: `{nearest['validation']['median']:.6g}`",
            "",
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch():
    import torch

    return torch


def _device(requested: str):
    torch = _torch()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    return device


if __name__ == "__main__":
    main()
