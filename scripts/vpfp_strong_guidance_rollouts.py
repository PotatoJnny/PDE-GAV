"""Frozen assets and one-shot public-solver rollouts for strong IC40 guidance.

No training, scheduling, resume selection, or new sampling is owned here.
Controller-private draws/truth remain outside the public solver process.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:
    sys.path.insert(0, str(ROOT/"src"))
if str(ROOT/"scripts") not in sys.path:
    sys.path.insert(0, str(ROOT/"scripts"))

from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import (
    VPFPOnlineModelSpec, build_vpfp_online_binding,
)
from pde_agent_verifier_solver.plugins.vpfp.frozen_solver_reward import (
    load_frozen_solver, materialize_public_case, native_macro_query,
    run_frozen_solver, score_solver_prediction,
)
from pde_agent_verifier_solver.plugins.vpfp.demand_solver_reward import (
    build_demand_public_case, run_demand_solver,
)
from pde_agent_verifier_solver.sdk import CaseVisibility

SCHEMA = "vpfp_strong_solver_guidance_rollout_v1"
SPLITS = ("fit", "validation", "selection", "confirmation")
COUNTS = {"fit": 64, "validation": 16, "selection": 16, "confirmation": 32}
PUBLIC_DESIGN = {"nx": 33, "ny": 33, "nt": 13, "nvx": 9, "nvy": 9,
                 "velocity_bounds": [-5, 5, -5, 5]}
PUBLIC_ACCESS_MODES = ("demand_points_v1", "demand_legacy_arrays_v1", "materialized_legacy_v1")


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_new_json(path, value):
    # Exclusive writes retain every earlier success/failure report.
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n"
    with Path(path).open("x") as handle:
        handle.write(encoded)


def _check_asset(path, digest):
    path = Path(path)
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("asset requires explicit SHA256")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'asset is absent or symlinked: {path}')
    return path.resolve()


def _config_scope(config):
    if config.get("public_access_mode", "materialized_legacy_v1") not in PUBLIC_ACCESS_MODES:
        raise ValueError("unknown public access mode; never silently pre-materialize")
    if "public_materialization" in config and "legacy_solver_array_grid" in config:
        raise ValueError("ambiguous historical and demand grid declarations")
    grid = config.get("legacy_solver_array_grid", config.get("public_materialization"))
    if config.get("counts") != COUNTS or grid != PUBLIC_DESIGN:
        raise ValueError("rollout counts/public numerical grid differ from the frozen round")
    expected = {"collision_nu_edges": [10, 15, 20, 25, 30], "temperature_edges": [.8, 1., 1.2]}
    if config.get("condition_strata") != expected:
        raise ValueError("strong round requires its registered eight condition strata")
    for name in ("base_weights_trainable", "solver_weights_or_code_trainable",
                 "failed_solver_outputs_are_high_rewards", "old_scores_used_as_training_labels"):
        if config.get(name) is not False:
            raise ValueError(f"frozen round requires {name}=false")


def freeze_round_assets(config, output):
    """Return (manifest, copied-path native_spec); originals are never changed.

    Files in output/frozen-assets are copied, hash-verified, and chmod 0444.
    An existing matching manifest may be verified but is never rewritten.
    Native model loading/structural validation is the caller's compute preflight.
    """
    _config_scope(config)
    from vpfp_strong_portable_binding import portable_roles, validate_portable_proof
    optional_roles = portable_roles(config)
    if optional_roles:
        validate_portable_proof(config)
    output = Path(output)
    if output.is_symlink():
        raise ValueError("freeze output must not be a symlink")
    directory = output/"frozen-assets"
    if directory.is_symlink():
        raise ValueError("frozen asset directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    config_path = Path(config["native_config"])
    if not config_path.is_absolute(): config_path = ROOT/config_path
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("native config must be a regular source file")
    config_hash = file_sha(config_path)
    native_config = json.loads(config_path.read_text())
    source_spec = VPFPOnlineModelSpec.from_mapping(native_config["vpfp"])
    roles = {
        "base_checkpoint": (config["base_checkpoint"], config["base_checkpoint_sha256"], "base-ic40.pt"),
        "retained_control_checkpoint": (config["retained_control_checkpoint"], config["retained_control_checkpoint_sha256"], "control-ic40.pt"),
        "base_decoder": (source_spec.base_decoder_checkpoint, source_spec.base_decoder_sha256, "base-decoder.pt"),
        "expanded_decoder": (source_spec.expanded_decoder_checkpoint, source_spec.expanded_decoder_sha256, "expanded-decoder.pt"),
        "solver": (config["solver"], config["solver_sha256"], "solver.py"),
        "native_config": (config_path, config_hash, "native-config.json"),
    }
    roles.update(optional_roles)
    assets = {}
    for role, (source, digest, name) in roles.items():
        source = _check_asset(source, digest)
        target = directory/name
        if target.exists() or target.is_symlink():
            _check_asset(target, digest)
        else:
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024*1024)
            _check_asset(target, digest)
        _check_asset(source, digest)  # source mutation during a copy fails closed.
        target.chmod(0o444)
        assets[role] = {"source_path": str(source), "path": str(target.resolve()), "sha256": digest}
    if optional_roles:
        validate_portable_proof(config, frozen_assets=assets)
    load_frozen_solver(assets["solver"]["path"], assets["solver"]["sha256"])
    native = asdict(source_spec)
    for role in ("base_decoder", "expanded_decoder"):
        native[role+"_checkpoint"] = assets[role]["path"]
    # Normalize tuple/list representation for strict saved-manifest comparison.
    native = json.loads(json.dumps(native))
    manifest = {"schema": SCHEMA, "stage": "assets_frozen", "config_sha256": _json_sha(config),
        "assets": assets, "native_spec": native, "registered_roots": registered_roots(config),
        "native_asset_structural_preflight_completed": False, "scientific_admission": False,
        "original_assets_modified": False, "base_or_solver_training_permitted": False}
    path = directory/"manifest.json"
    if path.exists():
        if path.is_symlink() or json.loads(path.read_text()) != manifest:
            raise ValueError("existing frozen manifest belongs to a different round")
    else:
        _write_new_json(path, manifest)
        path.chmod(0o444)
    manifest = {**manifest, "manifest_path": str(path.resolve()), "manifest_sha256": file_sha(path)}
    return manifest, VPFPOnlineModelSpec.from_mapping(native)


def _seed(tag):
    return int.from_bytes(sha256(tag.encode()).digest()[:8], "little") % (2**63)


def _fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _interior_condition(low, high, tag):
    bits = int.from_bytes(sha256(tag.encode()).digest()[:8], "little") >> 11
    value = _fp32(low+(high-low)*(bits+.5)/(2**53))
    # Uniform SHA-derived draw, then represent it in FP32. If rounding lands
    # exactly on a bin edge, move one representable FP32 inward, not another draw.
    if value <= low:
        raw = struct.unpack("<I", struct.pack("<f", low))[0]
        value = struct.unpack("<f", struct.pack("<I", raw+1))[0]
    if value >= high:
        raw = struct.unpack("<I", struct.pack("<f", high))[0]
        value = struct.unpack("<f", struct.pack("<I", raw-1))[0]
    if not low < value < high:
        raise ValueError("FP32 condition draw did not remain inside its declared stratum")
    return value


def registered_roots(config):
    """Return four independent split lists; same slot may pair guide strengths."""
    _config_scope(config)
    seeds = config["seeds"]
    if any(type(seeds[name]) is not int or not 0 <= seeds[name] < 2**63 for name in SPLITS):
        raise ValueError("split seeds must be nonnegative signed-int64 integers")
    if len({seeds[name] for name in SPLITS}) != len(SPLITS):
        raise ValueError("split seeds must be distinct")
    nu, temp = config["condition_strata"]["collision_nu_edges"], config["condition_strata"]["temperature_edges"]
    result, used = {}, set()
    for split in SPLITS:
        rows = []
        for index in range(COUNTS[split]):
            stratum = index % 8
            tag = f"{SCHEMA}:{split}:{seeds[split]}:{index}"
            seed = _seed(tag+":root")
            if seed in used: raise ValueError("registered root collision; do not silently redraw")
            used.add(seed)
            rows.append({"slot_id": f"{split}-{index:04d}", "split": split, "index": index,
                "stratum": stratum, "root_seed": seed, "conditions": [
                    _interior_condition(nu[stratum//2], nu[stratum//2+1], tag+":nu"),
                    _interior_condition(temp[stratum%2], temp[stratum%2+1], tag+":T")]})
        result[split] = rows
    return result


class CompletedDrawReplay:
    """Single-use completed native batch; generate never invokes a sampler."""
    def __init__(self, identity):
        self.identity = deepcopy(dict(identity))
        self.identity["completed_draw_replay"] = True
        self.identity["shared_backend_guidance_enabled"] = False
        self.config = SimpleNamespace(batch_size=1)
        self._pending = None
        self.consumed_count = 0

    def set_draw(self, draw, *, guidance_report=None):
        from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import native_batch_from_draw
        if self._pending is not None:
            raise ValueError("cannot overwrite an unconsumed completed draw")
        batch = native_batch_from_draw(draw, guidance_report=guidance_report)
        if len(batch["equation_seeds"]) != 1:
            raise ValueError("rollout replay owns exactly one registered draw")
        self._pending = batch

    def generate(self, seeds):
        if self._pending is None:
            raise ValueError("no completed draw registered; replay cannot resample")
        if tuple(int(value) for value in seeds) != tuple(int(value) for value in self._pending["equation_seeds"]):
            raise ValueError("replay request changed registered root seeds")
        batch, self._pending = self._pending, None
        self.consumed_count += 1
        return batch

    def clear(self):
        self._pending = None


def build_rollout_runtime(native_spec, base_sampler, output, device):
    replay = CompletedDrawReplay(base_sampler.identity)
    binding = build_vpfp_online_binding(native_spec, output_root=Path(output)/"resident-cases",
        device=device, guidance=False, latent_sampler=replay)
    return binding, replay


def _validate_draw(draw, row):
    import torch
    import numpy as np
    if draw["root_seeds"] != [row["root_seed"]]:
        raise ValueError("draw root differs from preregistered slot")
    for name, tag in (("known_prior_seeds", "prior"), ("future_noise_seeds", "future")):
        expected = _seed(f"vpfp-initial-conditioned-{tag}:{row['root_seed']}")
        if draw[name] != [expected]:
            raise ValueError(f"draw {name} differs from native derived stream")
    result = {}
    for name in ("full", "initial_context", "future_white", "equation_conditions"):
        value = draw[name]
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or len(value) != 1 or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"draw {name} requires a finite one-row native FP32 tensor")
        result[name] = value.detach().cpu().clone()
    condition = result["equation_conditions"].numpy().astype(np.float32)
    expected = np.asarray(row["conditions"], dtype=np.float32)[None]
    if condition.shape != (1, 2) or not np.array_equal(condition.view(np.uint32), expected.view(np.uint32)):
        raise ValueError("draw conditions differ from preregistered FP32 values")
    for name in ("root_seeds", "known_prior_seeds", "future_noise_seeds"):
        result[name] = list(draw[name])
    return result


def _save_pt(path, value):
    import torch
    with Path(path).open("xb") as handle:
        torch.save(value, handle)
    Path(path).chmod(0o600)
    return {"path": str(Path(path).resolve()), "sha256": file_sha(path)}


def evaluate_draw(draw, row, *, runtime, replay, solver, output, budget_seconds, check,
                  public_access_mode="demand_points_v1"):
    """Save one original slot, solve from public data, then score private truth.

    ``output`` must be a new per-draw directory. The parent owns resume and
    pairs different guidance strengths at the same registered row. Exceptions
    become finite failed receipts, never positive-infinite rewards. Allocation
    checks retain their receipt and cleanup, then re-raise to the parent;
    they are not numerical-failure labels or completed root attempts.
    """
    if public_access_mode not in PUBLIC_ACCESS_MODES:
        raise ValueError("unknown public access mode")
    demand = public_access_mode != "materialized_legacy_v1"
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("rollout directory already exists; parent must verify/resume its receipt")
    output.mkdir(parents=True)
    started = perf_counter()
    report = {"schema": SCHEMA, "stage": "registered", "row": deepcopy(row),
        "execution_passed": False, "valid": False, "reward_eligible": False, "reward": None,
        "validity_label_available": False, "label_kind": "unavailable",
        "label_unavailable_reason": "solver_not_called",
        "error": None, "scientific_admission": False, "solver_sha256": solver.sha256,
        "public_access_mode": public_access_mode,
        "guidance": deepcopy(draw.get("guidance_report", row.get("guidance"))),
        "resource_scope": {"solver_cpu_deadline_seconds": float(budget_seconds),
            "public_materialization_is_gpu_work_outside_solver_cpu_deadline": not demand,
            "demand_oracle_backend_wait_separately_timed": demand,
            "reference_PDE_solves": 0, "resampling_inside_rollout": False,
            "shared_backend_guidance_enabled": False}}
    _write_new_json(output/"registered.json", report)
    session = None
    case_id = "guidance-"+sha256((str(output.resolve())+":"+row["slot_id"]).encode()).hexdigest()[:24]
    stage_times = {}
    interruption = None
    def allocation_check():
        nonlocal interruption
        try:
            check()
        except TimeoutError as exc:
            interruption = exc
            raise
    try:
        allocation_check()
        # Retain actual failed tensor draws too; JSON validity never erases the
        # original private numerical artifact. Validation follows this capture.
        import torch
        private = {name: draw[name].detach().cpu().clone() for name in
                   ("full", "initial_context", "future_white", "equation_conditions")}
        private.update({name: list(draw[name]) for name in
                        ("root_seeds", "known_prior_seeds", "future_noise_seeds")})
        private.update(schema=SCHEMA, row=deepcopy(row), guidance=report["guidance"])
        report["private_draw"] = _save_pt(output/"private-draw.pt", private)
        _validate_draw(draw, row)
        report["stage"] = "resident_registration"
        replay.set_draw(draw, guidance_report=report["guidance"])
        before = replay.consumed_count
        rows = runtime.generate_base(seeds=(row["root_seed"],), case_ids=(case_id,),
            visibility=CaseVisibility.HIDDEN, round_index=0, cohort=row["split"])
        if len(rows) != 1 or replay.consumed_count != before+1:
            raise ValueError("runtime did not consume exactly one original completed draw")
        wrapped = rows[0]
        if wrapped.transaction is not None: wrapped.transaction.commit()
        record = wrapped.record
        if record.case.case_id != case_id:
            raise ValueError("runtime changed the registered case ID")
        session = runtime.public_session_factory(record.case)
        report["stage"] = "public_descriptor" if demand else "public_materialization"
        allocation_check()
        stage_start = perf_counter()
        if demand:
            public_case = build_demand_public_case(record.case,
                legacy_array_adapter=public_access_mode == "demand_legacy_arrays_v1")
            stage_times["public_descriptor_seconds"] = perf_counter()-stage_start
            report["prematerialized_oracle_points"] = 0
            report["public_descriptor"] = _save_pt(output/"public-descriptor.pt", {
                "question": dict(public_case.public_data["question"]),
                "arrays": {name: torch.from_numpy(value.copy())
                           for name, value in public_case.public_data["arrays"].items()},
                "public_access_mode": public_access_mode})
        else:
            public_case, materialization = materialize_public_case(record.case, session)
            stage_times["public_materialization_seconds"] = perf_counter()-stage_start
            report["public_materialization"] = materialization
            report["public_case"] = _save_pt(output/"public-case.pt", {
                "question": dict(public_case.public_data["question"]),
                "arrays": {name: torch.from_numpy(value.copy())
                           for name, value in public_case.public_data["arrays"].items()}})
            _write_new_json(output/"public-materialization.json", materialization)
        allocation_check()
        query = native_macro_query(public_case)
        report["stage"] = "solver_call"
        if demand:
            prediction, receipt = run_demand_solver(solver, public_case, query,
                public_session=session, budget_seconds=budget_seconds,
                legacy_array_adapter=public_access_mode == "demand_legacy_arrays_v1",
                check=allocation_check)
            report["public_oracle_usage"] = receipt.get("oracle_usage", {})
            _write_new_json(output/"public-oracle-requests.json", {
                "public_access_mode": public_access_mode,
                "prematerialized_points": 0,
                "requests": receipt.get("oracle_requests", []),
                "usage": receipt.get("oracle_usage", {})})
        else:
            prediction, receipt = run_frozen_solver(solver, public_case, query, budget_seconds=budget_seconds)
        report["solver_receipt"] = receipt
        _write_new_json(output/"solver-receipt.json", receipt)
        if prediction is None or not receipt["execution_passed"]:
            report["stage"] = receipt["stage"]
            report["error"] = receipt.get("error") or {"type": "SolverFailed", "message": "solver did not return valid predictions"}
            eligible = receipt.get("validity_label_available") is True
            report.update(validity_label_available=eligible,
                label_kind="numerical_solver_failure" if eligible else "unavailable",
                label_unavailable_reason=None if eligible else receipt.get("failure_origin", "solver_receipt_not_numerically_eligible"))
        else:
            report["prediction"] = _save_pt(output/"solver-prediction.pt", {
                "fields": {k: list(v) for k, v in prediction.fields.items()},
                "axes": {k: list(v) for k, v in prediction.axes.items()}, "shape": list(prediction.shape)})
            allocation_check()
            report["stage"] = "verifier_truth_after_solver"
            stage_start = perf_counter()
            truth = record.truth.answer(query)  # ONLY here, after solver returns.
            stage_times["private_truth_seconds"] = perf_counter()-stage_start
            report["private_truth"] = _save_pt(output/"private-truth.pt", {
                "fields": {k: list(v) for k, v in truth.fields.items()},
                "axes": {k: list(v) for k, v in truth.axes.items()}, "shape": list(truth.shape)})
            scoring = score_solver_prediction(prediction, truth, query,
                background_density=public_case.public_data["question"]["background_density"],
                temperature=public_case.public_data["question"]["temperature"])
            report["scoring"] = scoring
            _write_new_json(output/"fresh-solver-error.json", scoring)
            report.update(execution_passed=scoring["execution_passed"], valid=scoring["reward_eligible"],
                reward_eligible=scoring["reward_eligible"], reward=scoring["reward"],
                error=(scoring["physical_scaled_rmse"] if scoring["reward_eligible"]
                       else scoring.get("error")), stage="post_scoring_deadline_check")
            report.update(validity_label_available=scoring["reward_eligible"] is True,
                label_kind="valid_solver_error" if scoring["reward_eligible"] else "unavailable",
                label_unavailable_reason=None if scoring["reward_eligible"] else "verifier_scoring_failed")
            allocation_check()
            report["stage"] = "completed" if scoring["reward_eligible"] else "scoring_failed"
    except Exception as exc:
        demand_receipt = getattr(exc, "demand_solver_receipt", None)
        if demand and demand_receipt is not None:
            report["solver_receipt"] = demand_receipt
            report["public_oracle_usage"] = demand_receipt.get("oracle_usage", {})
            _write_new_json(output/"solver-receipt.json", demand_receipt)
            _write_new_json(output/"public-oracle-requests.json", {
                "public_access_mode": public_access_mode, "prematerialized_points": 0,
                "requests": demand_receipt.get("oracle_requests", []),
                "usage": demand_receipt.get("oracle_usage", {})})
        if exc is interruption:
            # A completed genuine score stays genuine even if the allocation
            # expires immediately afterwards. Parent still sees interruption.
            report.update(execution_passed=False, allocation_interruption={
                "type": type(exc).__name__, "message": str(exc), "stage": report["stage"]},
                stage="allocation_interrupted")
            if not report["validity_label_available"]:
                report["label_unavailable_reason"] = "allocation_interrupted_before_valid_label"
        else:
            report.update(execution_passed=False, valid=False, reward_eligible=False, reward=None,
                          error={"type": type(exc).__name__, "message": str(exc)},
                          validity_label_available=False, label_kind="unavailable",
                          label_unavailable_reason="non_solver_error_at_"+report["stage"])
    finally:
        cleanup_errors = []
        for action in ((lambda: session.close()) if session is not None else (lambda: None),
                       lambda: runtime.release_cases((case_id,)), replay.clear):
            try: action()
            except Exception as exc:
                cleanup_errors.append({"type": type(exc).__name__, "message": str(exc)})
        if cleanup_errors:
            report.update(execution_passed=False, valid=False, reward_eligible=False, reward=None,
                          cleanup_errors=cleanup_errors, validity_label_available=False,
                          label_kind="unavailable", label_unavailable_reason="cleanup_failed")
        report["stage_wall_seconds"] = stage_times
        report["wall_seconds"] = perf_counter()-started
        report["resident_released"] = not cleanup_errors
        result_path = output/"result.json"
        _write_new_json(result_path, report)
    result = {**report, "result_path": str(result_path.resolve()), "result_sha256": file_sha(result_path)}
    if interruption is not None:
        # Thin parent handoff, not a new retry mechanism. Preserve the exact
        # attempt path/SHA so a caller can retain progress without re-scoring.
        interruption.rollout_result = result
        raise interruption
    return result


__all__ = ["freeze_round_assets", "registered_roots", "CompletedDrawReplay",
           "build_rollout_runtime", "evaluate_draw"]
