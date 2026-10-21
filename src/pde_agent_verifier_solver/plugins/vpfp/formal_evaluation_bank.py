"""Keep the actual common VPFP terminal questions for later comparisons.

The runner calls ``export_terminal_bank`` synchronously after both prescribed
terminal cohorts seal, before the first Agent starts. This is private storage,
not an Agent dataset, not Guidance feedback, and not a second sampler. Expanded
continuous latents include the *accepted* perturbations; replay never reruns
diffusion, an online PDE solve, or perturbation calibration.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Case, CaseRecord, CaseSource, CaseVisibility, EvaluationPhase,
    QueryKind, SampleSource,
)

from .problem import VPFPContinuousMacroTruth


SCHEMA = "vpfp_common_terminal_continuous_bank_v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_file(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return sha256(_json(value).encode()).hexdigest()


def _pack(value):
    """Weights-only-safe state; preserve public array dtype and tuple/list ABI."""
    import torch
    if isinstance(value, np.ndarray):
        return {"__ndarray__": torch.from_numpy(value.copy())}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [_pack(item) for item in value]}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _pack(item) for key, item in value.items()}
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"unsupported bank state: {type(value).__name__}")


def _unpack(value):
    if isinstance(value, dict):
        if set(value) == {"__ndarray__"}:
            return value["__ndarray__"].numpy().copy()
        if set(value) == {"__tuple__"}:
            return tuple(_unpack(item) for item in value["__tuple__"])
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    return value


def _runtime_identity(runtime):
    identity = {
        "expanded_decoder_config": runtime.expanded_model.config.to_dict(),
        "expanded_decoder_checkpoint_sha256": _sha_file(runtime.checkpoint),
        "implementation_sha256": {
            name: _sha_file(Path(__file__).with_name(name))
            for name in ("perturbation_backend.py", "problem.py", "batched_source.py")
        },
    }
    kernel = getattr(runtime, "oracle_kernel", "reference_v1")
    if kernel != "reference_v1":
        identity["oracle_kernel"] = kernel
        modules = ("basic_oracle.py",) + (("structured_oracle.py",) if kernel in
            ("structured_batch_v1", "structured_reuse_v1") else ())
        identity["implementation_sha256"].update({
            name: _sha_file(Path(__file__).with_name(name)) for name in modules})
    return json.loads(_json(identity))


def _terminal_records(controller, expected_cohort_ids):
    expected = tuple(expected_cohort_ids)
    cohorts = {item.cohort_id: item for item in controller.campaign.terminal_evaluation.cohorts}
    if tuple(cohorts) != expected:
        raise ValueError("common terminal cohort recipe differs from the approved order")
    events = controller.events.snapshot()
    sealed = [event.payload["cohort_id"] for event in events
              if event.event_type == "terminal_cohort_sealed"]
    if tuple(sealed) != expected or any(event.event_type == "agent_started" for event in events):
        raise RuntimeError("export requires both terminal cohorts sealed before the first Agent")
    samplers = tuple(controller._terminal_case_samplers)
    if len(samplers) != len(expected):
        raise RuntimeError("terminal sampler inventory is incomplete")
    hidden = {}
    for sampler, cohort_id in zip(samplers, expected, strict=True):
        if sampler.closed or sampler._seed_namespace != f"terminal:{cohort_id}":
            raise RuntimeError("terminal sampler identity/order changed")
        records = tuple(sampler.record_for(case.case_id) for case in sampler.issued_cases)
        if len(records) != cohorts[cohort_id].suite.case_count:
            raise RuntimeError("terminal sampler does not contain the full approved cohort")
        for source in (SampleSource.BASE, SampleSource.GUIDED):
            roots = tuple(record for record in records if record.case.parent_case_id is None
                          and sampler._clean_source_by_case_id.get(record.case.case_id) is source)
            root_ids = {record.case.case_id for record in roots}
            children = tuple(record for record in records if record.case.parent_case_id in root_ids)
            for suffix, group in (("", roots), ("_perturbed", children)):
                if group:
                    key = controller._terminal_distribution_key(cohort_id, source.value + suffix)
                    hidden[key] = group
    identifiers = [record.case.case_id for records in hidden.values() for record in records]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != controller._terminal_hidden_case_ids:
        raise RuntimeError("terminal bank does not exactly match Controller-owned terminal cases")
    for name in ("_development_case_ids", "_initial_guidance_hidden_case_ids", "_promotion_hidden_case_ids"):
        if set(identifiers).intersection(getattr(controller, name)):
            raise RuntimeError("terminal bank overlaps training/development/selection cases")
    return hidden


def export_terminal_bank(controller, binding, output_directory, *, initial_solver_sha256,
                         expected_cohort_ids=("common_base", "common_guided")):
    """Persist actual hidden answers and their exact 256-point Terminal plans.

    Failure propagates through the synchronous event callback, so Agent 1 cannot
    begin with an incomplete common bank. Partial files are retained, never
    silently overwritten. The active Controller keeps using its original cases.
    """
    import torch
    hidden = _terminal_records(controller, expected_cohort_ids)
    backend = binding.plugin.generator.backend
    runtime = backend.runtime
    if runtime.materialize_cases:
        raise ValueError("common bank requires the continuous demand-only runtime")
    audits = backend.training_audits
    if len(audits) != 1 or audits[0]["target_round_index"] != 1 or audits[0]["source_round_index"] != 0:
        raise ValueError("common guided cohort must use only the first S0 Guidance")
    first_guide = audits[0]
    if first_guide["target_solver_sha256"] != initial_solver_sha256:
        raise ValueError("common terminal Guidance does not target the prescribed S0")
    checkpoint = Path(first_guide["checkpoint_path"])
    settings = {**controller.campaign.task_settings,
                "default_query": controller.campaign.evaluator.default_query}
    if settings.get("terminal_point_count") != 256:
        raise ValueError("approved VPFP common bank requires exactly 256 scoring points")
    seed_for = controller._hidden_plan_seed_provider("terminal_evaluation", EvaluationPhase.TERMINAL, hidden)
    output = Path(output_directory)
    if output.is_symlink() or output.exists():
        raise FileExistsError("common terminal bank is immutable; choose a fresh private path")
    output.mkdir(mode=0o700, parents=True)
    entries = []
    queries = {}
    for distribution, records in hidden.items():
        for index, record in enumerate(records):
            case = record.case
            if case.visibility is not CaseVisibility.HIDDEN:
                raise ValueError("common bank cannot contain public development answers")
            seed = seed_for(index, case.case_id)
            plan = controller.evaluation.make_plan(case, EvaluationPhase.TERMINAL, settings, seed=seed)
            query = plan.query
            if query.kind is not QueryKind.POINTS or len(query.points) != 256:
                raise ValueError("terminal plan changed from approved private point query")
            query_data = {"kind": query.kind.value, "fields": list(query.fields),
                          "resolution": dict(query.resolution), "points": [dict(point) for point in query.points]}
            query_sha = _digest(query_data)
            if case.parent_case_id is not None and queries.get(case.parent_case_id) != query_sha:
                raise ValueError("parent and perturbation do not share exact scoring coordinates")
            queries[case.case_id] = query_sha
            reference = record.truth.answer(query)
            reference_data = {"kind": reference.kind.value, "fields": dict(reference.fields),
                              "axes": dict(reference.axes), "points": [dict(point) for point in reference.points],
                              "shape": list(reference.shape)}
            resident = runtime._active_resident(case)
            case_data = {"case_id": case.case_id, "source": case.source.value,
                         "visibility": case.visibility.value, "public_data": case.public_data,
                         "round_index": case.round_index, "parent_case_id": case.parent_case_id,
                         "tags": case.tags}
            cached = backend._draws.get(case.case_id)
            root_id = case.parent_case_id or case.case_id
            root_draw = backend._draws.get(root_id)
            if root_draw is None:
                raise RuntimeError("terminal root lacks its actual captured draw")
            payload = {
                "case": _pack(case_data),
                "expanded_latent": _pack(resident.latent),
                "equation_condition": _pack(resident.equation_condition),
                "actual_native_draw": _pack(cached),
                "root_draw_provenance": {"source": root_draw["source"], "root_seed": root_draw["root_seed"]},
                "query": query_data, "reference": reference_data,
            }
            filename = f"case-{len(entries):06d}.pt"
            path = output / filename
            with path.open("xb") as handle:
                torch.save(payload, handle)
            path.chmod(0o400)
            entries.append({"case_id": case.case_id, "parent_case_id": case.parent_case_id,
                            "distribution": distribution, "position": index, "file": filename,
                            "sha256": _sha_file(path), "query_seed": seed,
                            "query_sha256": query_sha, "reference_sha256": _digest(reference_data),
                            "root_case_id": root_id, "root_seed": root_draw["root_seed"],
                            "root_source": root_draw["source"]})
    manifest = {
        "schema": SCHEMA, "case_count": len(entries),
        "root_count": sum(item["parent_case_id"] is None for item in entries),
        "initial_solver_sha256": initial_solver_sha256,
        "first_guidance_sha256": first_guide["snapshot_sha256"],
        "first_guidance_target_solver_sha256": first_guide["target_solver_sha256"],
        "binding_identity": dict(binding.report), "runtime_identity": _runtime_identity(runtime),
        "cohort_ids": list(expected_cohort_ids), "terminal_point_count": 256,
        "uses": "private_common_terminal_comparison_only",
        "excluded_from_training_development_selection": True,
        "sealed_before_first_agent": True, "resampled_answers": False,
        "precomputed_source_grid": False, "state": "exact_accepted_expanded_continuous_latent",
        "cases": entries,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    with (output / "manifest.json").open("x") as handle:
        handle.write(_json(manifest) + "\n")
    (output / "manifest.json").chmod(0o400)
    output.chmod(0o500)
    return manifest


def _require_runtime_replay_identity(actual, saved, migration):
    if actual == saved:
        if migration is not None:
            raise ValueError("execution migration must describe a real identity change")
        return
    if (not isinstance(migration, Mapping)
            or set(migration) != {"source_runtime_identity", "target_runtime_identity"}
            or migration["source_runtime_identity"] != saved
            or migration["target_runtime_identity"] != actual):
        raise ValueError("continuous decoder differs from the frozen common bank")
    # An explicit handoff may change only Oracle execution modules. Physics,
    # decoder weights/configuration and the saved bank remain fully pinned.
    allowed = {"perturbation_backend.py", "basic_oracle.py", "structured_oracle.py"}
    if (set(actual) - {"oracle_kernel", "implementation_sha256"}
            != set(saved) - {"oracle_kernel", "implementation_sha256"}
            or any(actual[key] != saved[key] for key in saved
                if key not in ("oracle_kernel", "implementation_sha256"))
            or actual.get("oracle_kernel", "reference_v1") not in
                ("reference_v1", "basic_vectorized_v1", "structured_batch_v1", "structured_reuse_v1")):
        raise ValueError("execution-only migration changed decoder physics or weights")
    old, new = saved["implementation_sha256"], actual["implementation_sha256"]
    if any(old.get(name) != new.get(name) for name in (set(old) | set(new)) - allowed):
        raise ValueError("execution-only migration changed non-Oracle physical implementation")


def rehydrate_terminal_bank(runtime, directory, *, expected_manifest_sha256,
                            expected_execution_migration=None):
    """Verify and restore exact native residents; return records and saved queries.

    The independently recorded manifest digest is mandatory. Callers choose the
    same native decoder before loading; checkpoint/config/source hashes must
    match. No sampler, perturbation recalibration or full-grid materialization
    is called. This does not insert the bank into training or a Campaign.
    """
    import torch
    root = Path(directory)
    if root.is_symlink() or (root / "manifest.json").is_symlink():
        raise ValueError("bank and manifest cannot be symlinks")
    manifest = json.loads((root / "manifest.json").read_text())
    digest = manifest.pop("manifest_sha256", None)
    if manifest["schema"] != SCHEMA:
        raise ValueError("common terminal manifest integrity mismatch")
    manifest["manifest_sha256"] = digest
    if runtime.materialize_cases:
        raise ValueError("continuous decoder differs from the frozen common bank")
    _require_runtime_replay_identity(_runtime_identity(runtime), manifest["runtime_identity"],
                                    expected_execution_migration)
    pending = []
    hidden, queries = {}, {}
    for index, entry in enumerate(manifest["cases"]):
        if entry["file"] != f"case-{index:06d}.pt":
            raise ValueError("invalid terminal bank case filename")
        path = root / entry["file"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("common terminal case integrity mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        data = _unpack(payload["case"])
        data.update(source=CaseSource(data["source"]), visibility=CaseVisibility(data["visibility"]))
        case = Case(**data)
        if case.case_id != entry["case_id"] or case.visibility is not CaseVisibility.HIDDEN:
            raise ValueError("terminal case identity changed")
        arrays = case.public_data["arrays"]
        record = CaseRecord(case, VPFPContinuousMacroTruth(
            case_id=case.case_id, t_axis=tuple(arrays["t"]), x_axis=tuple(arrays["x"]),
            y_axis=tuple(arrays["y"]), evaluate_points=runtime.evaluate_macros,
            evaluate_transport_points=getattr(runtime, "evaluate_transport_macros", None)))
        query = payload["query"]
        queries[case.case_id] = AnswerQuery.at_points(fields=query["fields"], points=query["points"])
        hidden.setdefault(entry["distribution"], []).append(record)
        pending.append((record, _unpack(payload["expanded_latent"]), _unpack(payload["equation_condition"])))
    if len(pending) != manifest["case_count"]:
        raise ValueError("terminal bank inventory changed")
    runtime._preflight_resident_ids(tuple(record.case.case_id for record, _, _ in pending))
    registered = []
    try:
        for record, latent, condition in pending:
            runtime._register_resident(record, latent, condition)
            registered.append(record.case.case_id)
    except BaseException:
        runtime.release_cases(tuple(registered))
        raise
    return {key: tuple(records) for key, records in hidden.items()}, queries, manifest
