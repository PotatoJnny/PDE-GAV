#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.vpfp.continuous_cases import materialize_continuous_case  # noqa: E402
from agent_verifier.vpfp.continuous_diffusion import (  # noqa: E402
    denormalize_vectors,
    load_continuous_diffusion_checkpoint,
    normalize_vectors,
    sample_continuous_latents,
)
from agent_verifier.vpfp.continuous_guidance import (  # noqa: E402
    load_continuous_reward_checkpoint,
    make_reward_guidance,
)
from agent_verifier.vpfp.continuous_oracle import (  # noqa: E402
    VPFPContinuousPrivateCase,
    VPFPOracleQueryLimits,
)
from agent_verifier.vpfp.continuous_training import (  # noqa: E402
    load_continuous_autoencoder_checkpoint,
)
from agent_verifier.vpfp.online_generator_protocol import (  # noqa: E402
    ONLINE_GENERATOR_REQUEST_SCHEMA,
    ONLINE_GENERATOR_RESPONSE_SCHEMA,
    audit_online_output,
    assert_reward_checkpoint_target,
    export_numerical_online_case,
    validate_controller_config,
)
from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve bounded numerical VPFP generation to one instruction-only Agent."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class OnlineGeneratorState:
    def __init__(self, config_path: Path, *, device: str) -> None:
        self.config_path = config_path.resolve()
        self.config = validate_controller_config(
            json.loads(self.config_path.read_text(encoding="utf-8"))
        )
        self.workspace = Path(self.config["workspace"]).resolve()
        self.private_root = Path(self.config["private_root"]).resolve()
        self.private_root.mkdir(parents=True, exist_ok=False)
        self.output_root = self.workspace / "online_generator_outputs"
        self.output_root.mkdir(parents=True, exist_ok=False)
        (self.output_root / "cases").mkdir()
        (self.output_root / "queries").mkdir()
        self.registered_output_paths: set[Path] = set()
        readme_path = self.output_root / "README.md"
        readme_path.write_text(
            "# Online VPFP Development Samples\n\n"
            "Every case here is numerical training/development data generated during "
            "this Agent session. These files are excluded from private promotion and "
            "final evaluation.\n",
            encoding="utf-8",
        )
        self.registered_output_paths.add(readme_path)
        self.registered_output_paths.add(self.output_root / "manifest.json")
        self.device = str(device)
        self.torch = _torch()
        if self.device.startswith("cuda") and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.diffusion, self.diffusion_payload = load_continuous_diffusion_checkpoint(
            self.config["diffusion_checkpoint"], device=self.device
        )
        self.reward_model, self.reward_payload = load_continuous_reward_checkpoint(
            self.config["reward_checkpoint"], device=self.device
        )
        _encoder, self.decoder, self.autoencoder_payload = (
            load_continuous_autoencoder_checkpoint(
                self.config["autoencoder_checkpoint"], device=self.device
            )
        )
        assert_reward_checkpoint_target(
            self.reward_payload, self.config["target_solver_sha256"]
        )
        for model in (self.diffusion, self.reward_model, self.decoder):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        with np.load(self.config["encoded_latents"], allow_pickle=False) as payload:
            conditions = np.asarray(payload["conditions"], dtype=np.float32)
            group_ids = np.asarray(payload["group_ids"])
            train_indices = np.asarray(payload["train_indices"], dtype=np.int64)
            validation_indices = np.asarray(
                payload["validation_indices"], dtype=np.int64
            )
        self.condition_pools = {
            "train": _unique_conditions(conditions, group_ids, train_indices),
            "validation": _unique_conditions(
                conditions, group_ids, validation_indices
            ),
        }
        self.grid = VPFPGrid(**dict(self.config["grid"]))
        self.time = VPFPTime(**dict(self.config["time"]))
        self.limits = dict(self.config["limits"])
        self.rng = np.random.default_rng(int(self.config["seed"]))
        self.sample_calls = 0
        self.sample_count = 0
        self.answer_count = 0
        self.query_calls = 0
        self.query_points = 0
        self.records: dict[str, dict[str, Any]] = {}
        self.private_log_path = self.private_root / "request_log.jsonl"
        self.private_log_path.write_text("", encoding="utf-8")
        self.lock = threading.Lock()
        self.started = time.time()
        self._write_public_manifest()

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("schema") != ONLINE_GENERATOR_REQUEST_SCHEMA:
            raise ValueError("unexpected request schema")
        operation = str(request.get("operation") or "")
        if operation == "status":
            return self.status()
        if operation == "sample":
            return self.sample(request)
        if operation == "query":
            return self.query(request)
        raise ValueError("unsupported operation")

    def status(self) -> dict[str, Any]:
        return {
            "target_solver_sha256": self.config["target_solver_sha256"],
            "budgets": self._budget_report(),
            "available_condition_pools": {
                name: int(values[0].shape[0])
                for name, values in self.condition_pools.items()
            },
            "dynamic_output_root": "online_generator_outputs",
            "primary_truth": "continuous hidden function; files are numerical access views",
            "model_parameterization_exposed": False,
        }

    def sample(self, request: dict[str, Any]) -> dict[str, Any]:
        count = int(request.get("count", 0))
        mode = str(request.get("mode") or "")
        answers = str(request.get("answers") or "")
        pool_name = str(request.get("condition_pool") or "")
        if count < 1 or count > int(self.limits["max_batch"]):
            raise ValueError("sample count is outside the documented per-call limit")
        if self.sample_count + count > int(self.limits["max_samples"]):
            raise ValueError("total sample budget would be exceeded")
        if mode not in {"base", "guided", "mixed"}:
            raise ValueError("sample mode must be base, guided, or mixed")
        if answers not in {"none", "grid"}:
            raise ValueError("answers must be none or grid")
        if answers == "grid" and self.answer_count + count > int(
            self.limits["max_answer_grids"]
        ):
            raise ValueError("numerical answer-grid budget would be exceeded")
        if pool_name not in self.condition_pools:
            raise ValueError("unknown condition pool")

        condition_table, group_table = self.condition_pools[pool_name]
        rows = self.rng.integers(0, condition_table.shape[0], size=count)
        selected_condition = np.asarray(condition_table[rows], dtype=np.float32)
        selected_groups = np.asarray(group_table[rows])
        dtype = next(self.diffusion.parameters()).dtype
        normalized_condition = normalize_vectors(
            self.torch.as_tensor(selected_condition, dtype=dtype),
            self.diffusion_payload["condition_normalization"],
        ).to(device=self.device)
        seed = int(self.rng.integers(1, 2**31 - 1))
        generator = self.torch.Generator(device=self.device)
        generator.manual_seed(seed)
        initial_noise = self.torch.randn(
            (count, int(self.diffusion.config.latent_dim)),
            dtype=dtype,
            device=self.device,
            generator=generator,
        )
        variants = _variants(mode, count)
        base = None
        guided = None
        if "base" in variants:
            with self.torch.no_grad():
                base = sample_continuous_latents(
                    self.diffusion,
                    normalized_condition,
                    initial_noise=initial_noise,
                    sample_steps=int(self.config["sample_steps"]),
                )
        if "guided" in variants:
            guidance = make_reward_guidance(
                self.reward_model,
                absolute_scale=float(self.config["guidance"]["absolute_scale"]),
                centered_scale=float(self.config["guidance"]["centered_scale"]),
                max_gradient_norm=float(
                    self.config["guidance"]["max_gradient_norm"]
                ),
            )
            guided = sample_continuous_latents(
                self.diffusion,
                normalized_condition,
                initial_noise=initial_noise,
                sample_steps=int(self.config["sample_steps"]),
                guidance=guidance,
            )
        rows_out = []
        for index, variant in enumerate(variants):
            normalized = base[index] if variant == "base" else guided[index]
            with self.torch.no_grad():
                latent = denormalize_vectors(
                    normalized[None], self.diffusion_payload["latent_normalization"]
                )[0]
            sequence = self.sample_count + 1
            public_id = f"online_{sequence:06d}"
            private_id = f"controller_online_{sequence:06d}"
            private_case = self.private_root / "cases" / private_id
            private_case.parent.mkdir(exist_ok=True)
            materialize_continuous_case(
                private_case,
                case_id=private_id,
                model=self.decoder,
                decoder_checkpoint=self.config["autoencoder_checkpoint"],
                latent=latent,
                equation_condition=selected_condition[index],
                grid=self.grid,
                time=self.time,
                source_origin="generated_continuous_autodiff_residual",
                generator_metadata={
                    "policy_variant": variant,
                    "condition_group_id": str(selected_groups[index]),
                    "sample_stream_id": "agent_online_training",
                    "target_solver_sha256": self.config["target_solver_sha256"],
                },
                case_spec={
                    "training_only": True,
                    "eligible_for_private_selection": False,
                    "eligible_for_final_evaluation": False,
                },
                evaluation_chunk_size=int(self.config["materialization_chunk_size"]),
            )
            exported = export_numerical_online_case(
                private_case,
                self.output_root / "cases" / public_id,
                public_case_id=public_id,
                include_answer=answers == "grid",
            )
            self.registered_output_paths.update(
                {
                    Path(exported["public_question"]),
                    Path(exported["public_arrays"]),
                }
            )
            if exported["answer"] is not None:
                self.registered_output_paths.add(Path(exported["answer"]))
            oracle_limits = VPFPOracleQueryLimits(
                max_points_per_call=int(self.limits["max_points_per_call"]),
                max_total_points=int(self.limits["max_total_points"]),
                evaluation_chunk_size=int(self.config["oracle_chunk_size"]),
            )
            self.records[public_id] = {
                "private_case": private_case,
                "private": VPFPContinuousPrivateCase(
                    case_id=public_id,
                    model=self.decoder,
                    latent=latent,
                    equation_condition=selected_condition[index],
                    limits=oracle_limits,
                ),
                "variant": variant,
                "answers": answers,
                "condition_pool": pool_name,
            }
            self.sample_count += 1
            if answers == "grid":
                self.answer_count += 1
            rows_out.append(
                {
                    "case_id": public_id,
                    "case_dir": _relative(exported["case_dir"], self.workspace),
                    "public_question": _relative(
                        exported["public_question"], self.workspace
                    ),
                    "public_arrays": _relative(
                        exported["public_arrays"], self.workspace
                    ),
                    "answer": (
                        None
                        if exported["answer"] is None
                        else _relative(exported["answer"], self.workspace)
                    ),
                    "policy_variant": variant,
                    "condition_pool": pool_name,
                }
            )
        self.sample_calls += 1
        self._private_log(
            {
                "operation": "sample",
                "count": count,
                "mode": mode,
                "answers": answers,
                "condition_pool": pool_name,
                "sampling_seed": seed,
                "public_case_ids": [row["case_id"] for row in rows_out],
            }
        )
        self._write_public_manifest()
        return {"cases": rows_out, "budgets": self._budget_report()}

    def query(self, request: dict[str, Any]) -> dict[str, Any]:
        case_id = str(request.get("case_id") or "")
        field = str(request.get("field") or "")
        if case_id not in self.records:
            raise ValueError("query case_id is not an online sample from this session")
        if field not in {"source", "rho0", "g0", "truth"}:
            raise ValueError("unsupported query field")
        points_path = _workspace_file(
            self.workspace, request.get("points_file"), suffixes={".npy", ".npz"}
        )
        points = _load_points(points_path)
        count = int(points.shape[0])
        if count > int(self.limits["max_points_per_call"]):
            raise ValueError("point query exceeds the documented per-call limit")
        if self.query_points + count > int(self.limits["max_total_points"]):
            raise ValueError("total point-query budget would be exceeded")
        private = self.records[case_id]["private"]
        if field == "source":
            values = {"values": private.public_oracle().source(points)}
        elif field == "rho0":
            values = {"values": private.public_oracle().rho0(points)}
        elif field == "g0":
            values = {"values": private.public_oracle().g0(points)}
        else:
            values = private.released_truth(points)
        output = _query_output_path(
            self.output_root,
            request.get("out"),
            sequence=self.query_calls + 1,
        )
        np.savez_compressed(output, points=points, **values)
        self.registered_output_paths.add(output)
        self.query_calls += 1
        self.query_points += count
        self._private_log(
            {
                "operation": "query",
                "case_id": case_id,
                "field": field,
                "point_count": count,
                "output": str(output),
            }
        )
        self._write_public_manifest()
        return {
            "case_id": case_id,
            "field": field,
            "point_count": count,
            "output": _relative(output, self.workspace),
            "keys": ["points", *values.keys()],
            "budgets": self._budget_report(),
        }

    def finalize(self) -> dict[str, Any]:
        self._write_public_manifest()
        audit = audit_online_output(
            self.output_root,
            registered_files=sorted(self.registered_output_paths),
            allow_missing_registered_files=True,
        )
        report = {
            "schema": "vpfp_online_generator_service_report_v1",
            "target_solver_sha256": self.config["target_solver_sha256"],
            "uptime_seconds": float(time.time() - self.started),
            "budget_usage": self._budget_report(),
            "registered_output_count": len(self.registered_output_paths),
            "output_audit": audit,
        }
        (self.private_root / "service_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report

    def _budget_report(self) -> dict[str, Any]:
        return {
            "sample_calls": int(self.sample_calls),
            "samples_used": int(self.sample_count),
            "samples_limit": int(self.limits["max_samples"]),
            "answer_grids_used": int(self.answer_count),
            "answer_grids_limit": int(self.limits["max_answer_grids"]),
            "query_calls": int(self.query_calls),
            "query_points_used": int(self.query_points),
            "query_points_limit": int(self.limits["max_total_points"]),
        }

    def _write_public_manifest(self) -> None:
        rows = [
            {
                "case_id": case_id,
                "case_dir": f"cases/{case_id}",
                "policy_variant": record["variant"],
                "condition_pool": record["condition_pool"],
                "numerical_answer_released": record["answers"] == "grid",
                "training_only": True,
                "eligible_for_private_selection": False,
                "eligible_for_final_evaluation": False,
            }
            for case_id, record in sorted(self.records.items())
        ]
        manifest = {
            "schema": "vpfp_online_generator_output_manifest_v1",
            "target_solver_sha256": self.config["target_solver_sha256"],
            "primary_object": "continuous hidden function",
            "visible_files_are": "numerical access views only",
            "model_parameterization_exposed": False,
            "cases": rows,
            "budget_usage": self._budget_report(),
        }
        (self.output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _private_log(self, row: dict[str, Any]) -> None:
        with self.private_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(1024 * 1024 + 1)
        if not raw or len(raw) > 1024 * 1024:
            return
        state: OnlineGeneratorState = self.server.state  # type: ignore[attr-defined]
        request: Any = None
        try:
            request = json.loads(raw)
            with state.lock:
                payload = state.dispatch(request)
            response = {
                "schema": ONLINE_GENERATOR_RESPONSE_SCHEMA,
                "ok": True,
                **payload,
            }
        except Exception as exc:  # noqa: BLE001 - private log retains exact failure.
            state._private_log(
                {
                    "operation": "failed_request",
                    "exception": repr(exc),
                    "request_operation": (
                        request.get("operation") if isinstance(request, dict) else None
                    ),
                }
            )
            response = {
                "schema": ONLINE_GENERATOR_RESPONSE_SCHEMA,
                "ok": False,
                "error": f"{type(exc).__name__}: request rejected; check the documented interface and budgets",
            }
        try:
            self.wfile.write(
                json.dumps(response, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
        except (BrokenPipeError, ConnectionResetError):
            # The Agent may abandon a request after a local timeout or quota error.
            pass


class UnixServer(socketserver.UnixStreamServer):
    allow_reuse_address = False


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise SystemExit("online VPFP generator service requires a Slurm compute node")
    socket_path = Path(args.socket).resolve()
    state = OnlineGeneratorState(Path(args.config), device=str(args.device))
    allowed_socket_roots = (
        state.workspace,
        Path(os.environ["SLURM_TMPDIR"]).resolve(),
    )
    if not any(_is_within(socket_path, root) for root in allowed_socket_roots):
        raise SystemExit(
            "online generator socket must be inside the Agent workspace "
            "or the job-private SLURM_TMPDIR"
        )
    if socket_path.exists():
        raise SystemExit(f"refusing to replace existing socket path: {socket_path}")
    server = UnixServer(str(socket_path), RequestHandler)
    server.state = state  # type: ignore[attr-defined]

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if socket_path.exists():
            socket_path.unlink()
        print(json.dumps(state.finalize(), indent=2, sort_keys=True), flush=True)


def _variants(mode: str, count: int) -> list[str]:
    if mode == "base":
        return ["base"] * int(count)
    if mode == "guided":
        return ["guided"] * int(count)
    return ["guided" if index % 2 == 0 else "base" for index in range(int(count))]


def _unique_conditions(conditions, group_ids, indices):
    selected_groups = np.unique(group_ids[indices])
    rows = []
    for group in selected_groups:
        group_indices = indices[group_ids[indices] == group]
        rows.append(conditions[int(group_indices[0])])
    if not rows:
        raise ValueError("condition pool is empty")
    return np.asarray(rows, dtype=np.float32), selected_groups


def _workspace_file(root: Path, raw: Any, *, suffixes: set[str]) -> Path:
    path = Path(str(raw or "")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("points file must stay inside the Agent workspace") from exc
    if not path.is_file() or path.is_symlink() or path.suffix not in suffixes:
        raise ValueError("points file must be a regular .npy or .npz file")
    return path


def _load_points(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        points = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"points"}:
                raise ValueError("points .npz must contain exactly the key 'points'")
            points = np.asarray(payload["points"])
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite rank-2 array")
    return points


def _query_output_path(root: Path, raw: Any, *, sequence: int) -> Path:
    queries = root / "queries"
    path = (
        queries / f"query_{int(sequence):06d}.npz"
        if raw in {None, ""}
        else Path(str(raw)).resolve()
    )
    try:
        relative = path.relative_to(queries)
    except ValueError as exc:
        raise ValueError("query output must stay under online_generator_outputs/queries") from exc
    if len(relative.parts) != 1 or path.suffix != ".npz":
        raise ValueError("query output must be one .npz file directly under the query root")
    if path.exists():
        raise FileExistsError("query output already exists")
    return path


def _relative(path: str | Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(root.resolve()))


def _torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch is required for the online VPFP generator") from exc
    return torch


if __name__ == "__main__":
    main()
