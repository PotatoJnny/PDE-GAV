"""Bind declared PDE-Agent comparison pairs and export their terminal manifest.

No scheduler submissions, model calls, Generator training or sampling occur here.
"""
from __future__ import annotations

import json
from pathlib import Path

from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json
from pde_agent_verifier_solver.sdk.candidate_bundle import inspect_candidate_bundle
from ..case_bank import load_ns_continuous_case_bank
from .environment import load_development_bank, continuous_field_identity, _digest
from .search import NSBaselineConfig


METHOD_KEYS = {
    "codepde": "codepde_refinement_ns_adaptation",
    "autonumerics": "autonumerics_ns_adaptation",
    "autopde": "autopde_ns_adaptation",
    "pde_sharp": "pde_sharp_ns_adaptation",
}
ENVIRONMENT_ARMS = ("fixed_sampled", "continuous_online_base")


def _method_keys(value):
    if (not isinstance(value, list) or not value
            or any(not isinstance(k, str) or k not in METHOD_KEYS for k in value)
            or len(set(value)) != len(value)):
        raise ValueError("methods must list distinct supported PDE-Agent names")
    return value


def _path(value, base):
    result = Path(value).expanduser()
    return (base / result).resolve() if not result.is_absolute() else result.resolve()


def _replace(value, values):
    if isinstance(value, dict):
        return {k: _replace(v, values) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace(v, values) for v in value]
    if isinstance(value, str) and value.startswith("@") and value.endswith("@"):
        if value not in values:
            raise ValueError("unbound baseline template field: " + value)
        return values[value]
    return value


def prepare_comparisons(binding_path, output):
    binding_path = Path(binding_path).expanduser().resolve()
    binding = json.loads(binding_path.read_text())
    if binding.get("schema") != "ns_literature_binding_v1":
        raise ValueError("expected ns_literature_binding_v1 binding")
    # Old four-arm bindings retain their original scope; eight arms are explicit.
    methods = _method_keys(binding.get("methods", ["codepde", "autonumerics"]))
    protocol = binding.get("protocol", "bounded_v1")
    if protocol not in {"bounded_v1", "completion_v2"}:
        raise ValueError("unknown literature protocol")
    base = binding_path.parent
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("comparison output must be new or empty")
    paths = {name: _path(binding[name], base) for name in
             ("starter_manifest", "selection_manifest", "terminal_manifest", "initial_bundle", "codex_binary", "auth_file")}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    expected_abi = "continuous_structured_ns_mms_v6"
    starter = load_development_bank(paths["starter_manifest"], arm="continuous_online_base",
                                    expected_cases=64, expected_abi=expected_abi,
                                    require_reference=protocol == "completion_v2")
    if starter.manifest["finite_sampling"] != {"grid_size": 64, "forcing_times": 32,
                                               "dtype": "float64", "spatial_nodes": "cell_centres"}:
        raise ValueError("paired finite inputs must use the agreed 64x64 / 32-time float64 sampling")
    banks = {purpose: load_ns_continuous_case_bank(paths[name], expected_purpose=purpose, device="cpu")
             for purpose, name in (("promotion", "selection_manifest"), ("terminal", "terminal_manifest"))}
    identities, ids = [starter.field_identities], [{r.case.case_id for r in starter.records}]
    for purpose, count, point_count in (("promotion", 24, 96), ("terminal", 96, 128)):
        bank = banks[purpose]
        if len(bank.records) != count or dict(bank.manifest["strata_counts"]) != {
                "base": count // 3, "guided": count // 3, "base_perturbed": count // 3}:
            raise ValueError(f"{purpose} must use the frozen {count}-Case 1:1:1 bank, with only Base-perturbed")
        if any(len(p.query.points) != point_count for p in bank.plans.values()):
            raise ValueError(f"{purpose} hidden point count differs from the agreed protocol")
        if any(a.raw_answer.continuous_mms_abi != expected_abi for a in bank._answers.values()):
            raise ValueError(f"{purpose} contains a different continuous MMS version")
        identities.append({continuous_field_identity(a) for a in bank._answers.values()})
        ids.append({r.case.case_id for r in bank.records})
    if any(identities[i] & identities[j] or ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError("development, selection and terminal banks must be disjoint")
    generator = dict(binding["generator"])
    if generator.get("artifact_sha256") != starter.manifest["provenance"].get("generator_artifact_sha256"):
        raise ValueError("online Generator weights do not match the frozen starter provenance")
    if any(list(generator.get("time_interval", ())) != list(a.time_interval)
           or generator.get("viscosity") != a.viscosity for a in starter._answers.values()):
        raise ValueError("online Generator time interval/viscosity differ from the fixed starter problems")
    saved_sampling = starter.manifest["provenance"].get("generator_sampling_settings")
    if saved_sampling is not None:
        for key in ("flow_state", "integration_method", "integration_steps", "cuda_graph_sampling", "max_points_per_request"):
            if generator.get(key) != saved_sampling.get(key):
                raise ValueError("online Generator sampling differs from the original starter: " + key)
    if generator.get("geometry_spec") is not None:
        from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
        from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
        declared_geometry = ContinuousGeometrySpec.from_dict(generator["geometry_spec"]).to_dict()
        if any(content_sha256(declared_geometry) != content_sha256(a.public_geometry)
               for a in starter._answers.values()):
            raise ValueError("online Generator geometry differs from the fixed starter problems")
    for name in ("flow_checkpoint", "pca_checkpoint"):
        path = _path(generator[name], base)
        generator[name] = str(path)
    generator["expected_continuous_mms_abi"] = expected_abi
    s0 = inspect_candidate_bundle(paths["initial_bundle"])
    workers = binding.get("candidate_case_workers", 4)
    if type(workers) is not int or workers < 1:
        raise ValueError("candidate_case_workers must be a positive integer")
    values = {"@STARTER@": str(paths["starter_manifest"]), "@SELECTION@": str(paths["selection_manifest"]),
              "@INITIAL@": str(paths["initial_bundle"]), "@INITIAL_SHA256@": s0.tree_sha256,
              "@CODEX@": str(paths["codex_binary"]), "@AUTH@": str(paths["auth_file"]),
              "@GPU@": str(binding.get("gpu_token", "0")), "@WORKERS@": workers, "@GENERATOR@": generator}
    repository = Path(__file__).resolve().parents[5]
    template_root = repository / ("configs/paper/ns_literature_v2" if protocol == "completion_v2" else "configs/paper/ns_literature_v1")
    templates = [template_root / f"{method}_{arm}.template.json"
                 for method in methods for arm in ENVIRONMENT_ARMS]
    if any(not path.is_file() for path in templates):
        raise ValueError("declared PDE-Agent comparison templates are missing")
    prepared = []
    for path in templates:
        label = path.name.removesuffix(".template.json")
        config = _replace(json.loads(path.read_text()), values)
        NSBaselineConfig(config["method"], output / "runs" / label, seed=config["seed"],
                         agent_budget_seconds=config["agent"]["hard_wall_seconds"], **config["search"])
        prepared.append((label, config))
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for label, config in prepared:
        config_path = output / (label + ".json")
        atomic_json(config_path, config)
        runs.append({"label": label, "config": str(config_path), "config_sha256": _digest(config_path),
                     "output": str(output / "runs" / label), "method": config["method"], "arm": config["dataset"]["arm"]})
    plan = {"schema": "ns_literature_comparisons_v1", "status": "prepared_not_submitted", "runs": runs,
            "methods": methods, "protocol": protocol,
            "terminal_manifest": str(paths["terminal_manifest"]), "terminal_manifest_sha256": _digest(paths["terminal_manifest"]),
            "initial_bundle": str(paths["initial_bundle"]), "initial_tree_sha256": s0.tree_sha256,
            "seed": 101, "candidate_case_workers": workers, "agent_wall_seconds_cap_per_method_arm": None if protocol == "completion_v2" else 43200,
            "method_status": "declared_NS_adaptations_not_exact_author_reproductions"}
    atomic_json(output / "comparisons.json", plan)
    return output / "comparisons.json"


def export_terminal_manifest(plan_path, output):
    plan_path = Path(plan_path).expanduser().resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "ns_literature_comparisons_v1":
        raise ValueError("expected a prepared literature comparison plan")
    runs = plan.get("runs", ())
    if "methods" in plan:
        methods = _method_keys(plan["methods"])
        expected = {(f"{method}_{arm}", METHOD_KEYS[method], arm)
                    for method in methods for arm in ENVIRONMENT_ARMS}
        actual = {(r.get("label"), r.get("method"), r.get("arm")) for r in runs}
        if len(runs) != len(expected) or actual != expected:
            raise ValueError("terminal comparison requires both distinct predeclared arms of every method")
    elif len(runs) != 4 or len({r["label"] for r in runs}) != 4:
        raise ValueError("legacy terminal comparison requires the four distinct predeclared arms")
    programs = [{"method": "S0", "run_id": "shared-s0", "artifact_id": "shared-s0", "seed": plan["seed"],
                 "bundle": plan["initial_bundle"], "role": "initial", "expected_tree_sha256": plan["initial_tree_sha256"]}]
    for run in plan["runs"]:
        result = json.loads((Path(run["output"]) / "baseline_result.json").read_text())
        actual_input = json.loads((Path(run["output"]) / "baseline_input.json").read_text())
        if result.get("status") != "completed" or result.get("method") != run["method"]:
            raise ValueError("every arm must finish using its predeclared method/config")
        frozen = inspect_candidate_bundle(result["selected_bundle"],
                                          recorded_identity=result.get("selected_tree_sha256"))
        programs.append({"method": run["label"], "run_id": run["label"], "artifact_id": result["selected_artifact_id"],
                         "seed": plan["seed"], "bundle": result["selected_bundle"], "role": "final",
                         "expected_tree_sha256": frozen.tree_sha256})
    result = {"schema": "ns_common_evaluation_v1", "task_id": "ns-full-field-solver", "require_initial": True,
              "datasets": [{"cohort": "independent_bgp_1_1_1", "dataset_id": "shared-v6-terminal",
                            "manifest": plan["terminal_manifest"]}], "programs": programs,
              "query": {"point_count": 128, "include_pressure": False},
              "budget": {"case_hard_wall_seconds": 300, "max_workers": plan["candidate_case_workers"]}}
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, result)
    return output


def export_starter_references(source, output):
    """Re-export the same continuous development fields with finite reference arrays."""
    from .environment import write_development_bank
    bank = load_development_bank(source, arm="continuous_online_base")
    sampling = bank.manifest["finite_sampling"]
    return write_development_bank(output, bank.records, provenance={
        **bank.manifest["provenance"], "source_starter_sha256": bank.manifest_sha256},
        grid_size=sampling["grid_size"], forcing_times=sampling["forcing_times"], include_reference=True)


def load_checkpoint_starter(recovery_path):
    """Read frozen development fields on CPU without resuming a live Generator.

    Runtime device/RNG/session checks remain untouched: this is a data export,
    not CheckpointStore.load or permission to continue an old campaign on CPU.
    """
    from types import SimpleNamespace
    from agent_verifier.neural_generator.structured_ns_answer_first import (
        ContinuousStructuredNSMMS, StructuredNSCaseCoefficients, StructuredNSProjectionConfig,
    )
    from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
    from pde_agent_verifier_solver.sdk import CaseRecord, CaseSource, CaseVisibility
    from ..case_bank import _case
    from ..continuous_backend import StructuredNSMMSAnswer, _NSVelocityTruth
    from ..resume import _get_tensors

    recovery_path = Path(recovery_path).expanduser().resolve()
    recovery = json.loads(recovery_path.read_text())
    if recovery.get("schema") != "ns_v6_asset_recovery_v1":
        raise ValueError("expected the received v6 asset recovery configuration")
    root = _path(recovery["asset_root"], recovery_path.parent)
    common, definition = recovery["common_development"], recovery["generator"]
    if common.get("case_count") != 64 or common.get("continuous_mms_abi") != "continuous_structured_ns_mms_v6":
        raise ValueError("the original common development pool must contain 64 v6 Cases")
    def asset(value):
        path = _path(value, root)
        if not path.is_relative_to(root):
            raise ValueError("received asset reference escapes its data directory")
        return path
    index_path = asset(common["manifest"])
    index = json.loads(index_path.read_text())
    directory = asset(common["checkpoint_directory"])
    checkpoint_path = directory / "checkpoint.json"
    checksum = common.get("checkpoint_sha256", index.get("checkpoint_sha256"))
    weights = {key: definition["artifact_sha256"][key] for key in ("flow_checkpoint", "pca_checkpoint")}
    checkpoint = json.loads(checkpoint_path.read_text())
    if (checkpoint.get("schema") != "campaign_checkpoint_v1"
            or checkpoint.get("identity") != common["checkpoint_identity"]):
        raise ValueError("common development checkpoint identity differs")
    runtime = checkpoint["generator_runtime"]
    if runtime.get("schema") != "ns_private_round_runtime_v1" or runtime.get("base") != {
            "generator_id": definition["generator_id"], "artifact_sha256": definition["generator_sha256"]}:
        raise ValueError("common development belongs to another Generator")
    identifiers = index.get("case_ids", ())
    rows = runtime.get("residents", ())
    by_id = {row["case"]["case_id"]: row for row in rows}
    if (len(identifiers) != 64 or len(set(identifiers)) != 64 or len(rows) != 64
            or set(by_id) != set(identifiers)):
        raise ValueError("common development index and checkpoint Cases differ")
    records = []
    for identifier in identifiers:
        row = by_id[identifier]
        case = _case(row["case"])
        if (row["cohort"] != "development" or case.visibility is not CaseVisibility.DEVELOPMENT
                or case.source is not CaseSource.GENERATED or case.parent_case_id is not None):
            raise ValueError("starter export accepts only original clean development Cases")
        payload = _get_tensors(directory / "objects", row["field"])
        if payload["abi"] != common["continuous_mms_abi"]:
            raise ValueError("received development field uses another continuous MMS version")
        raw = ContinuousStructuredNSMMS(
            StructuredNSCaseCoefficients(payload["psi"], payload["pressure"]),
            config=StructuredNSProjectionConfig.from_dict(payload["config"]),
            geometry_spec=payload["geometry"], continuous_mms_abi=payload["abi"],
            domain_bounds=tuple(payload["bounds"]), viscosity=payload["viscosity"],
            t0=payload["t0"], t1=payload["t1"], device="cpu",
        )
        answer = StructuredNSMMSAnswer(raw)
        domain = case.public_data["domain"]
        if (list(answer.domain_bounds) != [*domain["bounds"]["x"], *domain["bounds"]["y"]]
                or list(answer.time_interval) != list(domain["time_interval"])
                or answer.viscosity != case.public_data["parameters"]["viscosity"]
                or content_sha256(answer.public_geometry) != content_sha256(domain["geometry"])):
            raise ValueError("received Case and its frozen continuous field disagree")
        records.append(CaseRecord(case, _NSVelocityTruth(answer)))
    return SimpleNamespace(records=tuple(records), provenance={
        "source_common_index_sha256": _digest(index_path), "source_checkpoint_sha256": checksum,
        "generator_artifact_sha256": weights, "generator_id": definition["generator_id"],
        "original_case_order_preserved": True, "original_seed": index["seed"],
        "generator_sampling_settings": dict(definition["runtime"]),
        "restore_scope": "CPU_export_of_saved_Cases_no_sampling_no_runtime_resume",
    })


def export_checkpoint_starter(recovery_path, output):
    """Export original saved 64 Cases in the agreed paired-array representation."""
    from .environment import write_development_bank
    starter = load_checkpoint_starter(recovery_path)
    return write_development_bank(output, starter.records, provenance=starter.provenance,
                                  grid_size=64, forcing_times=32, include_reference=True)
