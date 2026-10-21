#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.neural_generator.structured_ns_distribution import (
    STRUCTURED_NS_FLOW_SCHEMA,
    StructuredCoefficientPCA,
)
from agent_verifier.neural_generator.structured_ns_guidance import (
    STRUCTURED_NS_REWARD_GUIDANCE_SCHEMA,
    StructuredRewardModel,
    StructuredRewardModelConfig,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a noisy-state reward model to the current solver's relative-L2 "
            "errors on cumulative continuous structured NS cases."
        )
    )
    parser.add_argument("--reward-table", required=True)
    parser.add_argument("--loop-root", required=True)
    parser.add_argument("--pca-checkpoint", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--solver-submission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--time-width", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-cases", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--minimum-lr-fraction", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--split-seed", type=int, default=20260810)
    parser.add_argument("--amp", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    torch = _torch()
    device = _device(args.device)
    output = Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse structured reward output: {output}")
    output.mkdir(parents=True)

    reward_path = Path(args.reward_table).resolve()
    loop_root = Path(args.loop_root).resolve()
    pca_path = Path(args.pca_checkpoint).resolve()
    flow_path = Path(args.flow_checkpoint).resolve()
    solver_path = Path(args.solver_submission).resolve()
    for label, path in (
        ("reward table", reward_path),
        ("PCA checkpoint", pca_path),
        ("flow checkpoint", flow_path),
        ("solver submission", solver_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    flow_payload = torch.load(flow_path, map_location="cpu", weights_only=False)
    if flow_payload.get("schema") != STRUCTURED_NS_FLOW_SCHEMA:
        raise SystemExit("unexpected structured flow checkpoint schema")
    pca = StructuredCoefficientPCA.load(pca_path, map_location=device).with_rank(
        int(flow_payload["pca_rank"])
    )

    reward_rows = _load_reward_rows(
        reward_path, minimum_cases=int(args.minimum_cases)
    )
    private_cases = _private_case_index(loop_root)
    case_ids, rewards, coefficients = _load_case_coefficients(
        reward_rows,
        private_cases=private_cases,
        expected_shape=pca.coefficient_shape,
    )
    coefficients = coefficients.to(device=device)
    with torch.no_grad():
        scores = pca.encode(coefficients).float()
    del coefficients

    train_indices, validation_indices = _deterministic_split(
        case_ids,
        validation_fraction=float(args.validation_fraction),
        seed=int(args.split_seed),
        device=device,
    )
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    train_reward = reward_tensor[train_indices]
    reward_mean = train_reward.mean()
    reward_std = train_reward.std(unbiased=False).clamp_min(1.0e-8)
    normalized_rewards = (reward_tensor - reward_mean) / reward_std

    model_config = StructuredRewardModelConfig(
        feature_dim=int(pca.rank),
        width=int(args.width),
        depth=int(args.depth),
        time_width=int(args.time_width),
    )
    model = StructuredRewardModel.build(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics_path = output / "metrics.jsonl"
    started = time.perf_counter()
    running_loss = 0.0
    running_steps = 0

    for step in range(1, int(args.steps) + 1):
        model.train()
        selected = train_indices[
            torch.randint(
                int(train_indices.numel()),
                (int(args.batch_size),),
                device=device,
                generator=generator,
            )
        ]
        clean = scores[selected]
        target = normalized_rewards[selected]
        noise = torch.randn(
            clean.shape,
            device=device,
            dtype=clean.dtype,
            generator=generator,
        )
        sample_time = torch.rand(
            clean.shape[0],
            device=device,
            dtype=clean.dtype,
            generator=generator,
        )
        noisy = (1.0 - sample_time[:, None]) * noise + sample_time[:, None] * clean
        lr = _learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=str(args.amp) == "bfloat16" and device.type == "cuda",
        ):
            prediction = model(noisy, sample_time)
            loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(args.max_grad_norm),
        )
        optimizer.step()
        running_loss += float(loss.detach().cpu())
        running_steps += 1

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            row = {
                "event": "train",
                "step": int(step),
                "loss": running_loss / max(running_steps, 1),
                "lr": float(lr),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            _append_jsonl(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)
            running_loss = 0.0
            running_steps = 0

    validation = _validation_audit(
        model,
        scores=scores[validation_indices],
        rewards=reward_tensor[validation_indices],
        reward_mean=reward_mean,
        reward_std=reward_std,
        seed=int(args.seed) + 1,
    )
    validation_index_set = set(validation_indices.detach().cpu().tolist())
    dataset_manifest = {
        "schema": "ns2d_structured_reward_dataset_v1",
        "case_count": len(case_ids),
        "train_count": int(train_indices.numel()),
        "validation_count": int(validation_indices.numel()),
        "reward_metric": "relative_l2",
        "reward_objective": "solver_error",
        "split_seed": int(args.split_seed),
        "reward_mean_train": float(reward_mean.detach().cpu()),
        "reward_std_train": float(reward_std.detach().cpu()),
        "cases": [
            {
                "case_id": case_id,
                "reward": float(reward),
                "split": (
                    "validation"
                    if index in validation_index_set
                    else "train"
                ),
            }
            for index, (case_id, reward) in enumerate(zip(case_ids, rewards, strict=True))
        ],
    }
    dataset_manifest_path = output / "reward_dataset.json"
    dataset_manifest_path.write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n"
    )
    checkpoint_path = output / "reward_guidance.pt"
    checkpoint = {
        "schema": STRUCTURED_NS_REWARD_GUIDANCE_SCHEMA,
        "model_config": model_config.to_dict(),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "reward_normalization": {
            "mean": float(reward_mean.detach().cpu()),
            "std": float(reward_std.detach().cpu()),
        },
        "reward_metric": "relative_l2",
        "reward_objective": "solver_error",
        "reward_table": str(reward_path),
        "reward_table_sha256": _file_sha256(reward_path),
        "reward_dataset": str(dataset_manifest_path),
        "reward_dataset_sha256": _file_sha256(dataset_manifest_path),
        "target_solver": str(solver_path),
        "target_solver_sha256": _file_sha256(solver_path),
        "pca_checkpoint": str(pca_path),
        "pca_sha256": _file_sha256(pca_path),
        "pca_rank": int(pca.rank),
        "flow_checkpoint": str(flow_path),
        "flow_checkpoint_sha256": _file_sha256(flow_path),
        "flow_step": int(flow_payload["step"]),
        "training_steps": int(args.steps),
        "parameter_count": int(parameter_count),
        "validation": validation,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": "ns2d_structured_reward_guidance_train_report_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "target_solver_sha256": checkpoint["target_solver_sha256"],
        "reward_table_sha256": checkpoint["reward_table_sha256"],
        "case_count": len(case_ids),
        "train_count": int(train_indices.numel()),
        "validation_count": int(validation_indices.numel()),
        "parameter_count": int(parameter_count),
        "training_steps": int(args.steps),
        "validation": validation,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def _load_reward_rows(
    path: Path, *, minimum_cases: int = 32
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = []
    seen: set[str] = set()
    for row in rows:
        if str(row.get("status")) != "ok":
            continue
        if str(row.get("reward_metric", "relative_l2")) != "relative_l2":
            raise ValueError("structured guidance requires relative_l2 rewards")
        objective = str(row.get("reward_objective", "solver_error"))
        if objective != "solver_error":
            raise ValueError("structured guidance requires pure solver_error rewards")
        case_id = str(row.get("case_id") or "")
        reward = row.get("reward")
        if not case_id or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            continue
        if case_id in seen:
            raise ValueError(f"duplicate reward row for {case_id}")
        seen.add(case_id)
        selected.append({**row, "case_id": case_id, "reward": float(reward)})
    if len(selected) < int(minimum_cases):
        raise ValueError(
            "structured reward guidance needs at least "
            f"{int(minimum_cases)} valid cases, found {len(selected)}"
        )
    return selected


def _private_case_index(loop_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    manifests = sorted(
        loop_root.glob("rounds/round_*/private_eval/generated_private_manifest.json")
    )
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text())
        for row in payload.get("cases", []):
            case_dir = Path(str(row.get("private_case_dir") or ""))
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            prior = index.get(case_id)
            if prior is not None and prior.resolve() != case_dir.resolve():
                raise ValueError(f"ambiguous private case id: {case_id}")
            index[case_id] = case_dir
    return index


def _load_case_coefficients(
    rows: list[dict[str, Any]],
    *,
    private_cases: dict[str, Path],
    expected_shape: tuple[int, ...],
):
    torch = _torch()
    case_ids: list[str] = []
    rewards: list[float] = []
    coefficient_parts = []
    missing = []
    for row in rows:
        case_id = str(row["case_id"])
        case_dir = private_cases.get(case_id)
        if case_dir is None:
            missing.append(case_id)
            continue
        state_path = case_dir / "hidden_continuous_case.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        psi = torch.as_tensor(state["coefficients"]["psi"], dtype=torch.float32)
        pressure = torch.as_tensor(
            state["coefficients"]["pressure"], dtype=torch.float32
        )
        coefficients = torch.stack((psi, pressure), dim=0).reshape(expected_shape)
        coefficient_parts.append(coefficients)
        case_ids.append(case_id)
        rewards.append(float(row["reward"]))
    if missing:
        raise ValueError(f"could not map {len(missing)} reward rows to private cases")
    return case_ids, np.asarray(rewards, dtype=np.float32), torch.stack(coefficient_parts)


def _deterministic_split(
    case_ids: list[str],
    *,
    validation_fraction: float,
    seed: int,
    device,
):
    torch = _torch()
    validation = []
    train = []
    threshold = int(round(float(validation_fraction) * 1_000_000))
    for index, case_id in enumerate(case_ids):
        digest = hashlib.sha256(f"{seed}:{case_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 1_000_000
        (validation if bucket < threshold else train).append(index)
    minimum_validation = min(16, max(1, len(case_ids) // 5))
    if len(validation) < minimum_validation:
        ordered = sorted(
            range(len(case_ids)),
            key=lambda index: hashlib.sha256(
                f"fallback:{seed}:{case_ids[index]}".encode()
            ).digest(),
        )
        validation = ordered[:minimum_validation]
        train = ordered[minimum_validation:]
    if not train or not validation:
        raise ValueError("structured reward split produced an empty partition")
    return (
        torch.as_tensor(train, dtype=torch.long, device=device),
        torch.as_tensor(validation, dtype=torch.long, device=device),
    )


def _validation_audit(
    model,
    *,
    scores,
    rewards,
    reward_mean,
    reward_std,
    seed: int,
) -> dict[str, Any]:
    torch = _torch()
    model.eval()
    generator = torch.Generator(device=scores.device).manual_seed(int(seed))
    fixed_noise = torch.randn(
        scores.shape,
        device=scores.device,
        dtype=scores.dtype,
        generator=generator,
    )
    by_time = {}
    with torch.no_grad():
        for value in (0.25, 0.5, 0.75, 0.95, 1.0):
            time_tensor = torch.full(
                (scores.shape[0],),
                float(value),
                device=scores.device,
                dtype=scores.dtype,
            )
            noisy = (1.0 - time_tensor[:, None]) * fixed_noise + time_tensor[:, None] * scores
            normalized = model(noisy, time_tensor)
            prediction = normalized * reward_std + reward_mean
            by_time[f"{value:.2f}"] = _regression_metrics(
                rewards.detach().cpu().numpy(),
                prediction.detach().cpu().numpy(),
            )
    clean = by_time["1.00"]
    return {
        "fixed_noise_seed": int(seed),
        "by_flow_time": by_time,
        "clean_state_pearson": clean["pearson"],
        "clean_state_spearman": clean["spearman"],
        "clean_state_top_quartile_reward_lift": clean[
            "top_predicted_quartile_reward_lift"
        ],
    }


def _regression_metrics(target, prediction) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = prediction - target
    pearson = _correlation(target, prediction)
    spearman = _correlation(_ranks(target), _ranks(prediction))
    threshold = np.quantile(prediction, 0.75)
    top = target[prediction >= threshold]
    target_mean = float(np.mean(target))
    return {
        "mse": float(np.mean(residual * residual)),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
        "pearson": pearson,
        "spearman": spearman,
        "target_mean": target_mean,
        "prediction_mean": float(np.mean(prediction)),
        "top_predicted_quartile_target_mean": float(np.mean(top)),
        "top_predicted_quartile_reward_lift": float(
            np.mean(top) / max(abs(target_mean), 1.0e-12)
        ),
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if float(np.std(first)) <= 1.0e-12 or float(np.std(second)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _learning_rate(step: int, args) -> float:
    progress = (int(step) - 1) / max(int(args.steps) - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    factor = float(args.minimum_lr_fraction) + (
        1.0 - float(args.minimum_lr_fraction)
    ) * cosine
    return float(args.lr) * factor


def _validate_args(args) -> None:
    positive = (
        args.steps,
        args.batch_size,
        args.width,
        args.depth,
        args.time_width,
        args.lr,
        args.max_grad_norm,
        args.log_every,
        args.minimum_cases,
    )
    if min(float(value) for value in positive) <= 0.0:
        raise SystemExit("structured reward training values must be positive")
    if not 0.0 < float(args.validation_fraction) < 0.5:
        raise SystemExit("validation fraction must lie in (0, 0.5)")
    if not 0.0 < float(args.minimum_lr_fraction) <= 1.0:
        raise SystemExit("minimum LR fraction must lie in (0, 1]")
    if int(args.time_width) % 2:
        raise SystemExit("time width must be even")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch():
    import torch

    return torch


def _device(name: str):
    torch = _torch()
    if str(name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(str(name))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    return device


if __name__ == "__main__":
    main(sys.argv[1:])
