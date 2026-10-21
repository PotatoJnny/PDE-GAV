#!/usr/bin/env python3
"""Measure how many NS feedback Cases the production reward model needs.

This benchmark uses real ``(PCA latent, solver error)`` pairs from a completed
NS campaign.  Every training size is evaluated on the same untouched Cases.
The useful question is not whether training loss falls, but whether the model
can rank unseen hard Cases while the Rectified Flow moves from noise to data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY_NS_SRC = ROOT / "legacy" / "ns" / "src"
if str(LEGACY_NS_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_NS_SRC))

from agent_verifier.neural_generator.structured_ns_distribution import (  # noqa: E402
    StructuredCoefficientPCA,
)
from agent_verifier.neural_generator.structured_ns_guidance import (  # noqa: E402
    StructuredRewardModel,
    StructuredRewardModelConfig,
)


FLOW_TIMES = (0.25, 0.50, 0.75, 0.95, 1.00)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-table", type=Path, required=True)
    parser.add_argument("--loop-root", type=Path, required=True)
    parser.add_argument("--pca-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=(64, 128, 256, 512, 1024),
    )
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--heldout-cases", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--time-width", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--plateau-spearman-tolerance",
        type=float,
        default=0.02,
        help="Allowed absolute gap from the largest-size median rank correlation.",
    )
    parser.add_argument(
        "--plateau-recall-tolerance",
        type=float,
        default=0.03,
        help="Allowed absolute gap from the largest-size median hard-quartile recall.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    _validate_arguments(args)
    torch = _torch()
    device = _device(args.device)
    output = args.out.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse benchmark output: {output}")
    output.mkdir(parents=True)

    pca_path = args.pca_checkpoint.expanduser().resolve()
    reward_path = args.reward_table.expanduser().resolve()
    loop_root = args.loop_root.expanduser().resolve()
    for label, path in (
        ("PCA checkpoint", pca_path),
        ("reward table", reward_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if not loop_root.is_dir():
        raise SystemExit(f"missing loop root: {loop_root}")

    pca = StructuredCoefficientPCA.load(pca_path, map_location=device)
    dataset = _load_dataset(
        reward_path=reward_path,
        loop_root=loop_root,
        pca=pca,
        device=device,
        cache_path=output / "encoded_reward_dataset.pt",
    )
    case_ids = list(dataset["case_ids"])
    scores = torch.as_tensor(dataset["scores"], dtype=torch.float32, device=device)
    rewards = torch.as_tensor(dataset["rewards"], dtype=torch.float32, device=device)
    heldout_indices, training_pool_indices = _fixed_split(
        case_ids,
        heldout_count=args.heldout_cases,
        seed=args.seed,
        device=device,
    )
    if max(args.train_sizes) > int(training_pool_indices.numel()):
        raise SystemExit("largest train size exceeds the post-holdout training pool")

    model_config = StructuredRewardModelConfig(
        feature_dim=int(scores.shape[1]),
        width=args.width,
        depth=args.depth,
        time_width=args.time_width,
    )
    result_path = output / "runs.jsonl"
    run_rows: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    for repeat_index in range(args.repeat_count):
        repeat_seed = args.seed + 100_000 * (repeat_index + 1)
        nested_pool = _nested_training_order(
            case_ids,
            training_pool_indices,
            seed=repeat_seed,
            device=device,
        )
        evaluation_noise = _fixed_noise(
            scores[heldout_indices],
            seed=args.seed + 900_000 + repeat_index,
        )
        for train_size in args.train_sizes:
            train_indices = nested_pool[:train_size]
            row = _train_and_evaluate(
                scores=scores,
                rewards=rewards,
                train_indices=train_indices,
                heldout_indices=heldout_indices,
                evaluation_noise=evaluation_noise,
                model_config=model_config,
                args=args,
                model_seed=repeat_seed + train_size,
                repeat_index=repeat_index,
                train_size=train_size,
            )
            run_rows.append(row)
            _append_jsonl(result_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)

    summary = _summarize(
        run_rows,
        train_sizes=tuple(args.train_sizes),
        spearman_tolerance=args.plateau_spearman_tolerance,
        recall_tolerance=args.plateau_recall_tolerance,
    )
    report = {
        "schema": "ns_reward_feedback_scale_benchmark_v1",
        "question": "How many solver-error feedback Cases train a useful NS reward model?",
        "source": {
            "reward_table": str(reward_path),
            "reward_table_sha256": _file_sha256(reward_path),
            "loop_root": str(loop_root),
            "pca_checkpoint": str(pca_path),
            "pca_sha256": _file_sha256(pca_path),
            "total_case_count": len(case_ids),
            "heldout_case_count": int(heldout_indices.numel()),
            "training_pool_case_count": int(training_pool_indices.numel()),
        },
        "model": {
            "config": model_config.to_dict(),
            "training_steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "repeat_count": args.repeat_count,
        },
        "selection_rule": {
            "reference_size": max(args.train_sizes),
            "spearman_tolerance": args.plateau_spearman_tolerance,
            "hard_quartile_recall_tolerance": args.plateau_recall_tolerance,
            "meaning": (
                "Choose the smallest size whose median guidance-path Spearman "
                "and hard-quartile recall are within the stated absolute gaps "
                "of the largest tested size."
            ),
        },
        "summary": summary,
        "elapsed_seconds": time.perf_counter() - benchmark_started,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def _load_dataset(*, reward_path, loop_root, pca, device, cache_path):
    torch = _torch()
    reward_sha256 = _file_sha256(reward_path)
    pca_sha256 = _file_sha256(Path(pca.path)) if hasattr(pca, "path") else None
    rows = _reward_rows(reward_path)
    private_cases = _private_case_index(loop_root)
    case_ids: list[str] = []
    rewards: list[float] = []
    encoded_parts = []
    missing = []
    batch_coefficients = []
    batch_ids = []
    batch_rewards = []

    def flush() -> None:
        if not batch_coefficients:
            return
        coefficients = torch.stack(batch_coefficients).to(device=device)
        with torch.no_grad():
            encoded = pca.encode(coefficients).float().cpu()
        encoded_parts.append(encoded)
        case_ids.extend(batch_ids)
        rewards.extend(batch_rewards)
        batch_coefficients.clear()
        batch_ids.clear()
        batch_rewards.clear()

    for row in rows:
        case_id = row["case_id"]
        case_dir = private_cases.get(case_id)
        if case_dir is None:
            missing.append(case_id)
            continue
        state_path = case_dir / "hidden_continuous_case.pt"
        if not state_path.is_file():
            missing.append(case_id)
            continue
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        psi = torch.as_tensor(state["coefficients"]["psi"], dtype=torch.float32)
        pressure = torch.as_tensor(
            state["coefficients"]["pressure"], dtype=torch.float32
        )
        batch_coefficients.append(
            torch.stack((psi, pressure), dim=0).reshape(pca.coefficient_shape)
        )
        batch_ids.append(case_id)
        batch_rewards.append(row["reward"])
        if len(batch_coefficients) == 128:
            flush()
    flush()
    if missing:
        raise ValueError(f"could not map {len(missing)} reward rows to private Cases")
    payload = {
        "schema": "ns_reward_feedback_encoded_dataset_v1",
        "reward_table_sha256": reward_sha256,
        "pca_sha256": pca_sha256,
        "case_ids": case_ids,
        "scores": torch.cat(encoded_parts, dim=0),
        "rewards": torch.as_tensor(rewards, dtype=torch.float32),
    }
    torch.save(payload, cache_path)
    return payload


def _reward_rows(path: Path) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        case_id = row.get("case_id")
        reward = row.get("reward")
        if (
            not isinstance(case_id, str)
            or not case_id
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            continue
        if case_id in seen:
            raise ValueError(f"duplicate reward row: {case_id}")
        seen.add(case_id)
        selected.append({"case_id": case_id, "reward": float(reward)})
    if not selected:
        raise ValueError("reward table contains no finite successful rows")
    return selected


def _private_case_index(loop_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    manifests = sorted(
        loop_root.glob("rounds/round_*/private_eval/generated_private_manifest.json")
    )
    if not manifests:
        raise ValueError("loop root contains no generated private manifests")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("cases", []):
            case_dir = Path(str(row.get("private_case_dir", ""))).resolve()
            case_id = case_dir.name
            if not case_id or not case_dir.is_dir():
                continue
            prior = index.get(case_id)
            if prior is not None and prior != case_dir:
                raise ValueError(f"ambiguous private Case id: {case_id}")
            index[case_id] = case_dir
    return index


def _fixed_split(case_ids, *, heldout_count, seed, device):
    torch = _torch()
    ordered = sorted(
        range(len(case_ids)),
        key=lambda index: hashlib.sha256(
            f"heldout:{seed}:{case_ids[index]}".encode("utf-8")
        ).digest(),
    )
    heldout = ordered[:heldout_count]
    pool = ordered[heldout_count:]
    return (
        torch.as_tensor(heldout, dtype=torch.long, device=device),
        torch.as_tensor(pool, dtype=torch.long, device=device),
    )


def _nested_training_order(case_ids, pool_indices, *, seed, device):
    torch = _torch()
    pool = pool_indices.detach().cpu().tolist()
    pool.sort(
        key=lambda index: hashlib.sha256(
            f"train:{seed}:{case_ids[index]}".encode("utf-8")
        ).digest()
    )
    return torch.as_tensor(pool, dtype=torch.long, device=device)


def _fixed_noise(scores, *, seed):
    torch = _torch()
    generator = torch.Generator(device=scores.device).manual_seed(seed)
    return torch.randn(
        scores.shape,
        dtype=scores.dtype,
        device=scores.device,
        generator=generator,
    )


def _train_and_evaluate(
    *,
    scores,
    rewards,
    train_indices,
    heldout_indices,
    evaluation_noise,
    model_config,
    args,
    model_seed,
    repeat_index,
    train_size,
) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(model_seed)
    if scores.device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
    model = StructuredRewardModel.build(model_config).to(scores.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_rewards = rewards[train_indices]
    reward_mean = train_rewards.mean()
    reward_std = train_rewards.std(unbiased=False).clamp_min(1.0e-8)
    normalized_rewards = (rewards - reward_mean) / reward_std
    generator = torch.Generator(device=scores.device).manual_seed(model_seed + 1)
    started = time.perf_counter()
    final_loss = None
    for _step in range(args.steps):
        selected = train_indices[
            torch.randint(
                int(train_indices.numel()),
                (args.batch_size,),
                device=scores.device,
                generator=generator,
            )
        ]
        clean = scores[selected]
        noise = torch.randn(
            clean.shape,
            dtype=clean.dtype,
            device=clean.device,
            generator=generator,
        )
        flow_time = torch.rand(
            clean.shape[0],
            dtype=clean.dtype,
            device=clean.device,
            generator=generator,
        )
        noisy = (1.0 - flow_time[:, None]) * noise + flow_time[:, None] * clean
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=scores.device.type,
            dtype=torch.bfloat16,
            enabled=args.amp == "bfloat16" and scores.device.type == "cuda",
        ):
            prediction = model(noisy, flow_time)
            loss = torch.nn.functional.mse_loss(
                prediction,
                normalized_rewards[selected],
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.maximum_gradient_norm
        )
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    heldout_scores = scores[heldout_indices]
    heldout_rewards = rewards[heldout_indices]
    by_time = {}
    model.eval()
    with torch.no_grad():
        for value in FLOW_TIMES:
            time_tensor = torch.full(
                (heldout_scores.shape[0],),
                value,
                dtype=heldout_scores.dtype,
                device=heldout_scores.device,
            )
            noisy = (
                (1.0 - time_tensor[:, None]) * evaluation_noise
                + time_tensor[:, None] * heldout_scores
            )
            normalized = model(noisy, time_tensor)
            prediction = normalized * reward_std + reward_mean
            by_time[f"{value:.2f}"] = _regression_metrics(
                heldout_rewards.detach().cpu().numpy(),
                prediction.detach().cpu().numpy(),
            )
    path_times = tuple(key for key in by_time if key != "1.00")
    return {
        "train_size": train_size,
        "repeat_index": repeat_index,
        "model_seed": model_seed,
        "final_training_loss": final_loss,
        "training_seconds": time.perf_counter() - started,
        "reward_mean_train": float(reward_mean.detach().cpu()),
        "reward_std_train": float(reward_std.detach().cpu()),
        "by_flow_time": by_time,
        "clean_spearman": by_time["1.00"]["spearman"],
        "guidance_path_spearman": float(
            np.mean([by_time[key]["spearman"] for key in path_times])
        ),
        "guidance_path_hard_quartile_recall": float(
            np.mean([by_time[key]["hard_quartile_recall"] for key in path_times])
        ),
        "guidance_path_top_quartile_lift": float(
            np.mean([by_time[key]["top_quartile_reward_lift"] for key in path_times])
        ),
    }


def _regression_metrics(target, prediction) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = prediction - target
    predicted_top = prediction >= np.quantile(prediction, 0.75)
    true_top = target >= np.quantile(target, 0.75)
    selected_target = target[predicted_top]
    return {
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
        "pearson": _correlation(target, prediction),
        "spearman": _correlation(_ranks(target), _ranks(prediction)),
        "hard_quartile_recall": float(
            np.count_nonzero(predicted_top & true_top)
            / max(np.count_nonzero(true_top), 1)
        ),
        "top_quartile_reward_lift": float(
            np.mean(selected_target) / max(abs(float(np.mean(target))), 1.0e-12)
        ),
    }


def _summarize(rows, *, train_sizes, spearman_tolerance, recall_tolerance):
    summaries = {}
    for size in train_sizes:
        selected = [row for row in rows if row["train_size"] == size]
        summaries[str(size)] = {
            name: _distribution([row[name] for row in selected])
            for name in (
                "clean_spearman",
                "guidance_path_spearman",
                "guidance_path_hard_quartile_recall",
                "guidance_path_top_quartile_lift",
                "training_seconds",
            )
        }
    reference = summaries[str(max(train_sizes))]
    reference_spearman = reference["guidance_path_spearman"]["median"]
    reference_recall = reference["guidance_path_hard_quartile_recall"]["median"]
    qualifying = []
    for size in train_sizes:
        row = summaries[str(size)]
        stable = (
            row["guidance_path_spearman"]["median"]
            >= reference_spearman - spearman_tolerance
            and row["guidance_path_hard_quartile_recall"]["median"]
            >= reference_recall - recall_tolerance
        )
        row["within_largest_size_plateau"] = stable
        if stable:
            qualifying.append(size)
    return {
        "by_train_size": summaries,
        "recommended_feedback_case_count": min(qualifying) if qualifying else None,
    }


def _distribution(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_arguments(args) -> None:
    sizes = tuple(args.train_sizes)
    if not sizes or any(size < 2 for size in sizes):
        raise SystemExit("train sizes must be integers of at least two")
    if len(sizes) != len(set(sizes)) or tuple(sorted(sizes)) != sizes:
        raise SystemExit("train sizes must be unique and increasing")
    positive = (
        args.repeat_count,
        args.heldout_cases,
        args.steps,
        args.batch_size,
        args.width,
        args.depth,
        args.time_width,
        args.learning_rate,
        args.maximum_gradient_norm,
    )
    if min(float(value) for value in positive) <= 0.0:
        raise SystemExit("benchmark counts and training values must be positive")
    if args.time_width % 2:
        raise SystemExit("time width must be even")
    if args.weight_decay < 0.0:
        raise SystemExit("weight decay must be nonnegative")
    if min(args.plateau_spearman_tolerance, args.plateau_recall_tolerance) < 0.0:
        raise SystemExit("plateau tolerances must be nonnegative")


def _torch():
    import torch

    return torch


def _device(name: str):
    torch = _torch()
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    return device


if __name__ == "__main__":
    main()
