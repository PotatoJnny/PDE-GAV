#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.neural_generator.structured_ns_distribution import (
    STRUCTURED_NS_FLOW_SCHEMA,
    StructuredCoefficientFlowConfig,
    StructuredCoefficientPCA,
    StructuredCoefficientRectifiedFlow,
    compatible_projection_signatures,
    load_structured_coefficient_corpus,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train rectified flow on train-fitted structured NS PCA scores."
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--pca-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument(
        "--rank-selection",
        default="",
        help=(
            "Optional PCA selection JSON. When provided, selected_rank from "
            "that validation result replaces --rank."
        ),
    )
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-cases", type=int, default=1024)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--time-width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--minimum-lr-fraction", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--amp", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if str(args.rank_selection):
        selection_path = Path(args.rank_selection).resolve()
        selection = json.loads(selection_path.read_text())
        if selection.get("schema") != "ns_pca_selection_v1":
            raise SystemExit(f"unexpected PCA selection schema: {selection_path}")
        args.rank = int(selection["selected_rank"])
    _validate_args(args)
    torch = _torch()
    device = _device(str(args.device))
    output = Path(args.out).resolve()
    resume_path = Path(args.resume).resolve() if str(args.resume) else None
    if output.exists() and resume_path is None:
        raise SystemExit(f"refusing to reuse structured flow output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    metrics_path = output / "metrics.jsonl"

    pca_path = Path(args.pca_checkpoint).resolve()
    pca = StructuredCoefficientPCA.load(pca_path, map_location=device).with_rank(
        int(args.rank)
    )
    train = load_structured_coefficient_corpus(
        args.train_manifest,
        device=device,
    )
    validation = load_structured_coefficient_corpus(
        args.validation_manifest,
        max_cases=int(args.validation_cases),
        device=device,
    )
    for corpus, label in ((train, "train"), (validation, "validation")):
        signature = dict(corpus.manifest["encoding_signature"])
        if not compatible_projection_signatures(pca.source_signature, signature):
            raise SystemExit(f"{label} coefficients are incompatible with PCA")
    with torch.no_grad():
        train_scores = pca.encode(train.coefficients).detach()
        validation_scores = pca.encode(validation.coefficients).detach()
    del train, validation

    model_config = StructuredCoefficientFlowConfig(
        feature_dim=int(args.rank),
        width=int(args.width),
        depth=int(args.depth),
        time_width=int(args.time_width),
    )
    # The CLI seed must also control parameter initialization, not only the
    # minibatch/noise stream. Frozen checkpoints and resume states are unchanged.
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    model = StructuredCoefficientRectifiedFlow.build(model_config).to(device)
    ema_model = StructuredCoefficientRectifiedFlow.build(model_config).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    training_generator = torch.Generator(device=device).manual_seed(int(args.seed))
    start_step = 0
    if resume_path is not None:
        start_step = _restore_checkpoint(
            resume_path,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            training_generator=training_generator,
            model_config=model_config,
            pca_path=pca_path,
            rank=int(args.rank),
        )
    fixed_noise, fixed_time = _fixed_validation_context(
        validation_scores,
        seed=int(args.seed) + 1,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_config = {
        "schema": "ns2d_structured_coefficient_flow_train_config_v1",
        "train_manifest": str(Path(args.train_manifest).resolve()),
        "validation_manifest": str(Path(args.validation_manifest).resolve()),
        "pca_checkpoint": str(pca_path),
        "pca_sha256": _file_sha256(pca_path),
        "rank": int(args.rank),
        "rank_selection": (
            str(Path(args.rank_selection).resolve())
            if str(args.rank_selection)
            else None
        ),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "train_cases": int(train_scores.shape[0]),
        "validation_cases": int(validation_scores.shape[0]),
        "model_config": model_config.to_dict(),
        "parameter_count": int(parameter_count),
        "lr": float(args.lr),
        "minimum_lr_fraction": float(args.minimum_lr_fraction),
        "warmup_steps": int(args.warmup_steps),
        "weight_decay": float(args.weight_decay),
        "ema_decay": float(args.ema_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "seed": int(args.seed),
        "amp": str(args.amp),
    }
    _require_matching_config(output / "train_config.json", train_config)
    started = time.perf_counter()
    running_loss = 0.0
    running_steps = 0
    last_validation = None

    for step in range(start_step + 1, int(args.steps) + 1):
        model.train()
        indices = torch.randint(
            int(train_scores.shape[0]),
            (int(args.batch_size),),
            device=device,
            generator=training_generator,
        )
        clean = train_scores[indices]
        lr = _learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=str(args.amp) == "bfloat16" and device.type == "cuda",
        ):
            loss = model.flow_matching_loss(
                clean,
                generator=training_generator,
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(args.max_grad_norm),
        )
        optimizer.step()
        _update_ema(ema_model, model, decay=float(args.ema_decay))
        running_loss += float(loss.detach().cpu())
        running_steps += 1

        if step % int(args.log_every) == 0 or step == 1:
            row = {
                "event": "train",
                "step": int(step),
                "loss": float(running_loss / max(running_steps, 1)),
                "lr": float(lr),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            _append_jsonl(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)
            running_loss = 0.0
            running_steps = 0

        if step % int(args.validate_every) == 0 or step == int(args.steps):
            last_validation = _validation_loss(
                ema_model,
                validation_scores,
                noise=fixed_noise,
                time=fixed_time,
                batch_size=int(args.batch_size),
                amp=str(args.amp),
            )
            row = {
                "event": "validation",
                "step": int(step),
                "loss": float(last_validation),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            _append_jsonl(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)

        if step % int(args.checkpoint_every) == 0 or step == int(args.steps):
            checkpoint_path = checkpoint_root / f"flow_step_{step:07d}.pt"
            _save_checkpoint(
                checkpoint_path,
                step=step,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                training_generator=training_generator,
                model_config=model_config,
                pca_path=pca_path,
                rank=int(args.rank),
                train_config=train_config,
            )

    final_checkpoint = checkpoint_root / f"flow_step_{int(args.steps):07d}.pt"
    report = {
        "schema": "ns2d_structured_coefficient_flow_train_report_v1",
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": _file_sha256(final_checkpoint),
        "final_step": int(args.steps),
        "final_validation_loss": last_validation,
        "parameter_count": int(parameter_count),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.rank,
        args.steps,
        args.batch_size,
        args.validation_cases,
        args.width,
        args.depth,
        args.time_width,
        args.lr,
        args.log_every,
        args.validate_every,
        args.checkpoint_every,
    )
    if min(float(value) for value in positive) <= 0.0:
        raise SystemExit("structured flow dimensions and optimization values must be positive")
    if not 0.0 < float(args.ema_decay) < 1.0:
        raise SystemExit("EMA decay must lie in (0,1)")
    if not 0.0 < float(args.minimum_lr_fraction) <= 1.0:
        raise SystemExit("minimum LR fraction must lie in (0,1]")
    if int(args.warmup_steps) < 0 or int(args.warmup_steps) >= int(args.steps):
        raise SystemExit("warmup steps must lie in [0, steps)")


def _learning_rate(step: int, args: argparse.Namespace) -> float:
    warmup = int(args.warmup_steps)
    if warmup > 0 and step <= warmup:
        return float(args.lr) * step / warmup
    progress = (step - warmup) / max(int(args.steps) - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    fraction = float(args.minimum_lr_fraction) + (
        1.0 - float(args.minimum_lr_fraction)
    ) * cosine
    return float(args.lr) * fraction


def _fixed_validation_context(scores, *, seed: int):
    torch = _torch()
    generator = torch.Generator(device=scores.device).manual_seed(int(seed))
    noise = torch.randn(
        scores.shape,
        device=scores.device,
        dtype=scores.dtype,
        generator=generator,
    )
    time = torch.rand(
        scores.shape[0],
        device=scores.device,
        dtype=scores.dtype,
        generator=generator,
    )
    return noise, time


def _validation_loss(model, scores, *, noise, time, batch_size: int, amp: str):
    torch = _torch()
    losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(scores.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(scores.shape[0]))
            with torch.autocast(
                device_type=scores.device.type,
                dtype=torch.bfloat16,
                enabled=amp == "bfloat16" and scores.device.type == "cuda",
            ):
                loss = model.flow_matching_loss(
                    scores[start:end],
                    noise=noise[start:end],
                    time=time[start:end],
                )
            losses.append((float(loss.cpu()), end - start))
    return sum(loss * count for loss, count in losses) / sum(
        count for _loss, count in losses
    )


def _update_ema(ema_model, model, *, decay: float) -> None:
    torch = _torch()
    with torch.no_grad():
        for ema_parameter, parameter in zip(
            ema_model.parameters(), model.parameters(), strict=True
        ):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(
            ema_model.buffers(), model.buffers(), strict=True
        ):
            ema_buffer.copy_(buffer)


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    model,
    ema_model,
    optimizer,
    training_generator,
    model_config,
    pca_path: Path,
    rank: int,
    train_config,
) -> None:
    payload = {
        "schema": STRUCTURED_NS_FLOW_SCHEMA,
        "step": int(step),
        "model_config": model_config.to_dict(),
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_generator_state": training_generator.get_state(),
        "pca_checkpoint": str(pca_path),
        "pca_sha256": _file_sha256(pca_path),
        "pca_rank": int(rank),
        "train_config": train_config,
    }
    _atomic_torch_save(path, payload)


def _restore_checkpoint(
    path: Path,
    *,
    model,
    ema_model,
    optimizer,
    training_generator,
    model_config,
    pca_path: Path,
    rank: int,
) -> int:
    torch = _torch()
    payload = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    if payload.get("schema") != STRUCTURED_NS_FLOW_SCHEMA:
        raise SystemExit("unexpected structured flow resume schema")
    if payload.get("model_config") != model_config.to_dict():
        raise SystemExit("resume model configuration differs")
    if int(payload.get("pca_rank", -1)) != int(rank):
        raise SystemExit("resume PCA rank differs")
    model.load_state_dict(payload["model_state_dict"])
    ema_model.load_state_dict(payload["ema_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    training_generator.set_state(payload["training_generator_state"])
    return int(payload["step"])


def _require_matching_config(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise SystemExit("existing structured flow train configuration differs")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _device(requested: str):
    torch = _torch()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    return device


if __name__ == "__main__":
    main()
