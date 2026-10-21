"""JSON entrypoint using original NS data and the existing command Agent ABI."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityFacade
from pde_agent_verifier_solver.engine.agent_execution import (
    CommandAgentProgram, SubprocessAgentExecutionService,
)
from pde_agent_verifier_solver.engine.campaign import AgentSpec, EvaluatorPolicy
from pde_agent_verifier_solver.engine.candidate_execution import (
    PythonTaskWorkerSpec, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, CaseVisibility, DevelopmentCoordinateMode, GenerationPurpose, TaskKind,
)

from ..common_evaluation import load_numerical_split
from ..tasks import NSSolverTask
from .public_problem import numerical_ns_public_problem_bundle
from .search import NSBaselineConfig, _json_value, _write_cost_report, _write_json, run_ns_baseline

SCHEMA = "ns_fixed_data_baseline_v1"


def _path(value, base):
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def allocated_evaluation_devices(gpu_tokens, *, environment=None):
    """Map allocated Agent GPUs into existing Controller visibility, without expanding it."""
    if isinstance(gpu_tokens, (str, bytes)):
        raise TypeError("evaluation GPU tokens must be a sequence")
    tokens = tuple(gpu_tokens)
    env = os.environ if environment is None else environment
    visible = tuple(env.get("CUDA_VISIBLE_DEVICES", "").split(","))
    for label, values in (("allocated", tokens), ("visible", visible)):
        if (not values or len(values) != len(set(values))
                or any(not isinstance(v, str) or not v or v == "-1"
                       or "," in v or any(c.isspace() for c in v) for v in values)):
            raise ValueError(f"{label} evaluation GPUs must be distinct nonempty GPU tokens")
    if not set(tokens).issubset(visible):
        raise ValueError("evaluation GPUs must already be CUDA-visible")
    return {token: f"cuda:{visible.index(token)}" for token in tokens}


def _public_problem_bundle(continuous_selection=False, continuous_development=False):
    if continuous_development:
        from ..public_problem import ns_public_problem_bundle
        return ns_public_problem_bundle(guidance_supported=False, boundary_velocity_trace=True)
    bundle = numerical_ns_public_problem_bundle()
    if not continuous_selection:
        return bundle
    note = (
        "\n\nPrivate evaluation also uses frozen generated NS Cases. Your delivered "
        "solver must support their continuous public ABI: case.public_data['domain'] "
        "and open_public_case_session(case).query_initial_velocity(points_xy) / "
        "query_forcing(points_txy), plus query_boundary_velocity(points_txy) when "
        "the Case declares a boundary trace. These expose only public equation inputs "
        "during Candidate execution. The Agent still receives only original finite-array "
        "development Cases; no hidden Cases, completed answers, or online sampling "
        "are provided during development. The no-oracle statement above applies to "
        "those development Cases.\n"
    )
    return replace(bundle, resources=tuple(
        replace(resource, data=resource.data + note.encode())
        if resource.relative_path == "ns_reference/README.md" else resource
        for resource in bundle.resources
    ))


def _run_from_config(config_path, output_directory, *, resume=False, segment_budget=None, gpu_token=None):
    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"baseline config requires schema {SCHEMA}")
    unknown = set(payload) - {"schema", "method", "seed", "dataset", "search",
                              "agent", "evaluation", "initial_candidate", "initial_candidate_sha256", "prompt"}
    if unknown:
        raise ValueError(f"unknown baseline config keys: {sorted(unknown)}")
    if gpu_token is not None:
        from .workflows import LITERATURE_METHODS
        if (payload["method"] not in LITERATURE_METHODS or not isinstance(gpu_token, str)
                or not gpu_token or "," in gpu_token or any(c.isspace() for c in gpu_token)):
            raise ValueError("--gpu-token requires one allocated device and a literature workflow")
        payload["agent"]["gpu_devices"] = [gpu_token]
    agent = payload["agent"]
    worker_settings = dict(agent.get("worker_settings", {}))
    nomination_policy = agent.get("nomination_policy", worker_settings.get("nomination_policy", "all"))
    if nomination_policy not in ("all", "latest"):
        raise ValueError("agent.nomination_policy must be 'all' or 'latest'")
    if worker_settings.get("nomination_policy", nomination_policy) != nomination_policy:
        raise ValueError("agent.nomination_policy conflicts with agent.worker_settings.nomination_policy")
    worker_settings["nomination_policy"] = nomination_policy
    completion_recovery = payload.get("search", {}).get("literature_options", {}).get("protocol") == "completion_v2"
    attempts = agent.get("transient_resume_max_attempts", worker_settings.get("transient_resume_max_attempts", 1))
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 3:
        raise ValueError("agent.transient_resume_max_attempts must be an integer from 1 to 3")
    if worker_settings.get("transient_resume_max_attempts", attempts) != attempts:
        raise ValueError("agent.transient_resume_max_attempts conflicts with agent.worker_settings.transient_resume_max_attempts")
    worker_settings["transient_resume_max_attempts"] = attempts
    until_deadline = agent.get("transient_resume_until_deadline",
        worker_settings.get("transient_resume_until_deadline", not completion_recovery))
    if type(until_deadline) is not bool:
        raise ValueError("agent.transient_resume_until_deadline must be a boolean")
    if worker_settings.get("transient_resume_until_deadline", until_deadline) != until_deadline:
        raise ValueError("agent.transient_resume_until_deadline conflicts with agent.worker_settings.transient_resume_until_deadline")
    worker_settings["transient_resume_until_deadline"] = until_deadline
    root = Path(output_directory).expanduser().resolve()
    checkpoint_path = root / "baseline_checkpoints" / "checkpoint.json"
    if resume and not checkpoint_path.is_file():
        raise FileNotFoundError("baseline resume requires a committed checkpoint")
    owned_empty = (segment_budget is not None and root.is_dir()
                   and {item.name for item in root.iterdir()} == {".checkpoint-writer.lock"})
    if root.exists() and any(root.iterdir()) and not resume and not owned_empty:
        raise FileExistsError("baseline output directory must be new or empty")
    dataset = payload["dataset"]
    manifest = _path(dataset["manifest"], path.parent)
    paired = dataset.get("kind") == "continuous_mms_paired"
    development_environment = None
    completion_mode = payload.get("search", {}).get("literature_options", {}).get("protocol") == "completion_v2"
    if paired and completion_mode and dataset.get("arm") == "continuous_online_base" and dataset.get("new_cases_per_branch", 64) is not None:
        raise ValueError("completion_v2 online acquisition requires an explicit null cumulative Case limit")
    continuous_development = dataset.get("kind") == "experimental_mms"
    if paired:
        from .environment import load_development_bank
        from .workflows import LITERATURE_METHODS
        if payload["method"] not in LITERATURE_METHODS:
            raise ValueError("paired input environments require a literature workflow")
        if not dataset.get("selection_manifest"):
            raise ValueError("paired environments require a separate frozen selection bank")
        starter = load_development_bank(manifest, arm=dataset["arm"],
            device=dataset.get("development_device", "cpu"), expected_cases=dataset.get("expected_cases", 64),
            expected_abi=dataset.get("expected_continuous_mms_abi"),
            require_reference=payload.get("search", {}).get("literature_options", {}).get("protocol") == "completion_v2")
        development = starter.records
        continuous_development = dataset["arm"] == "continuous_online_base"
    elif continuous_development:
        from ..experimental_mms_project import ExperimentalMMSProblem
        from pde_agent_verifier_solver.sdk import DatasetInput, DatasetKind
        if dataset["development_split"] != "development":
            raise ValueError("continuous Fixed development must use the declared development split")
        prepared = ExperimentalMMSProblem().prepare_dataset(DatasetInput(
            "ns-fixed-measured-development", DatasetKind.EXPERIMENTAL, location=str(manifest)))
        development = prepared.development_cases
        selected = dataset.get("development_case_ids")
        if selected is not None:
            indexed = {row.case.case_id: row for row in development}
            if not selected or len(set(selected)) != len(selected) or any(cid not in indexed for cid in selected):
                raise ValueError("development_case_ids must be a unique subset of development")
            development = tuple(indexed[cid] for cid in selected)
        if not development or not dataset.get("selection_manifest"):
            raise ValueError("continuous Fixed requires development Cases and a separate selection bank")
    else:
        development = load_numerical_split(
            manifest, split=dataset["development_split"],
            case_ids=dataset.get("development_case_ids"),
            visibility=CaseVisibility.DEVELOPMENT,
        )
    selection_bank = None
    if dataset.get("selection_manifest"):
        from ..case_bank import load_ns_continuous_case_bank

        if dataset.get("selection_case_ids") is not None:
            raise ValueError("a frozen continuous selection bank must be used in full")
        selection_bank = load_ns_continuous_case_bank(
            _path(dataset["selection_manifest"], path.parent),
            expected_purpose=GenerationPurpose.PROMOTION,
            device=dataset.get("selection_device", "cpu"),
        )
        selection = selection_bank.records
    else:
        selection = load_numerical_split(
            manifest, split=dataset["selection_split"],
            case_ids=dataset.get("selection_case_ids"),
            visibility=CaseVisibility.HIDDEN,
        )
    session_provider = selection_bank.generator if selection_bank else None
    if paired:
        from .environment import (PairedSessionProvider, OnlineBaseEnvironment, build_online_generator,
                                  paired_public_problem_bundle, PairedNSSolverTask, continuous_field_identity)
        hidden_states = {row["state_sha256"] for row in selection_bank.manifest["cases"]}
        if (starter.state_sha256 & hidden_states or starter.field_identities &
                {continuous_field_identity(a) for a in selection_bank._answers.values()}):
            raise ValueError("development and selection reuse a continuous field asset")
        dynamic = dynamic_state = None
        if continuous_development:
            generator_config = dict(dataset["online_generator"])
            starter_abis = {row["continuous_mms_abi"] for row in starter.manifest["cases"]}
            if len(starter_abis) != 1:
                raise ValueError("paired starter bank mixes continuous MMS versions")
            expected_abi = next(iter(starter_abis))
            if generator_config.get("expected_continuous_mms_abi", expected_abi) != expected_abi:
                raise ValueError("online Generator and starter declare different continuous MMS versions")
            generator_config["expected_continuous_mms_abi"] = expected_abi
            expected = starter.manifest["provenance"].get("generator_artifact_sha256")
            if not expected or expected != generator_config.get("artifact_sha256"):
                raise ValueError("starter bank must pin the same online Generator artifacts")
            dynamic, dynamic_state = build_online_generator(generator_config, path.parent)
            development_environment = OnlineBaseEnvironment(dynamic, dynamic_state,
                new_cases_per_branch=dataset.get("new_cases_per_branch", 64),
                reserved_case_ids=[r.case.case_id for r in (*development, *selection)], expected_abi=expected_abi)
        elif dataset.get("online_generator") or dataset.get("new_cases_per_branch", 0):
            raise ValueError("fixed_sampled cannot silently enable fresh Case acquisition")
        session_provider = PairedSessionProvider(starter, selection_bank, dynamic=dynamic,
            dynamic_state=dynamic_state, arm=dataset["arm"])
    elif continuous_development:
        from types import SimpleNamespace
        from ..case_bank import NSBankSessionProvider
        development_sessions = SimpleNamespace(
            records=development, manifest_sha256=sha256(manifest.read_bytes()).hexdigest(),
            _answers={row.case.case_id: row.truth._answer for row in development})
        session_provider = NSBankSessionProvider((development_sessions, selection_bank))
    # Case ids alone do not catch the same numerical files aliased across splits.
    def truth_paths(rows):
        return {str(getattr(row.truth, "case_dir", getattr(row.truth, "case_file", ""))) for row in rows} - {""}
    if truth_paths(development) & truth_paths(selection):
        raise ValueError("development and selection reuse the same numerical truth files")
    agent = payload["agent"]
    role_options = agent.get("roles")
    role_config = {}
    if role_options is not None:
        if payload["method"] != "competitive_refinement":
            raise ValueError("agent.roles is supported only for competitive_refinement")
        if not isinstance(role_options, list) or not role_options:
            raise ValueError("agent.roles must be a nonempty list")
        for role in role_options:
            if not isinstance(role, dict) or set(role) - {"agent_id", "name", "prompt", "gpu_devices"}:
                raise ValueError("each Agent role accepts agent_id, name, prompt and gpu_devices")
        role_config["competition_roles"] = tuple(
            AgentSpec(role["agent_id"], role["name"], role["prompt"]) for role in role_options
        )
    config = NSBaselineConfig(
        method=payload["method"], output_directory=root, seed=payload.get("seed", 0),
        prompt=payload.get("prompt", NSBaselineConfig.__dataclass_fields__["prompt"].default),
        agent_budget_seconds=agent["hard_wall_seconds"],
        **payload.get("search", {}), **role_config,
    )
    if config.method == "codepde_repeated_sample_ns_adaptation" and payload.get("initial_candidate"):
        raise ValueError("CodePDE repeated_sample requires cold start; S0 is not permitted")
    evaluation = dict(payload.get("evaluation", {}))
    allowed = {"allow_grid_truth", "allow_point_truth", "development_coordinate_mode",
               "development_point_count", "selection_point_count",
               "candidate_case_workers", "candidate_case_hard_wall_seconds",
               "development_batch_hard_wall_seconds", "borrow_agent_gpus"}
    if set(evaluation) - allowed:
        raise ValueError(f"unknown evaluation settings: {sorted(set(evaluation) - allowed)}")
    borrow_agent_gpus = evaluation.get("borrow_agent_gpus", False)
    if not isinstance(borrow_agent_gpus, bool):
        raise TypeError("evaluation.borrow_agent_gpus must be boolean")
    query = AnswerQuery.at_points(
        fields=("velocity_x", "velocity_y"), points=({"t": 0.0, "x": 0.0, "y": 0.0},),
    )
    policy = EvaluatorPolicy(
        default_query=query,
        allow_grid_truth=evaluation.get("allow_grid_truth", False),
        allow_point_truth=evaluation.get("allow_point_truth", False),
        development_coordinate_mode=DevelopmentCoordinateMode(
            evaluation.get("development_coordinate_mode", "private_random_points")
        ),
        development_point_count=evaluation.get("development_point_count", 32),
    )
    if paired and (policy.allow_grid_truth or policy.allow_point_truth):
        raise ValueError("paired literature protocol does not disclose solution truth")
    if policy.development_coordinate_mode.value == "user_fixed_query":
        raise ValueError(
            "baseline CLI does not support user_fixed_query without explicit coordinates; "
            "use the Python API with a validated AnswerQuery"
        )
    task_settings = {"official_point_count": evaluation.get("selection_point_count", 96),
                     "include_pressure": False}
    if selection_bank is not None:
        if any(plan.query.fields != query.fields
               or len(plan.query.points) != task_settings["official_point_count"]
               for plan in selection_bank.plans.values()):
            raise ValueError("selection settings differ from the continuous bank's frozen query")
    codec = PythonCandidateCodec(task_kind=TaskKind.SOLVER)
    source_root = Path(__file__).resolve().parents[4]
    repository = source_root.parent
    substitutions = {"{python}": sys.executable, "{repo}": str(repository),
                     "{output}": str(root)}
    command = []
    for value in agent["command"]:
        if not isinstance(value, str) or not value:
            raise ValueError("agent.command must contain nonempty strings")
        for name, replacement in substitutions.items():
            value = value.replace(name, replacement)
        command.append(value)
    # Canonical workers require a private runtime root outside their workspaces.
    (root / "agent_runtime").mkdir(parents=True, mode=0o700, exist_ok=True)
    # Shared project directories may inherit setgid even with mkdir(mode=0700).
    (root / "agent_runtime").chmod(0o700)
    worker = CommandAgentProgram(
        tuple(command), codec.decode_nomination, incumbent_encoder=codec.encode_incumbent,
        worker_settings=worker_settings,
        environment={"PYTHONPATH": str(source_root), **dict(agent.get("environment", {}))},
        gpu_devices=tuple(agent.get("gpu_devices", ())),
    )
    if config.method == "competitive_refinement":
        if role_options is None:
            role_options = [{"agent_id": role.agent_id} for role in config.competition_roles]
        worker = {
            role["agent_id"]: replace(worker, gpu_devices=tuple(
                role.get("gpu_devices", agent.get("gpu_devices", ()))
            )) for role in role_options
        }
    programs = tuple(worker.values()) if isinstance(worker, dict) else (worker,)
    device_map = (allocated_evaluation_devices(tuple(dict.fromkeys(
        token for program in programs for token in program.gpu_devices
    ))) if borrow_agent_gpus else None)
    execution = SubprocessAgentExecutionService(
        root / "agent_turns", hard_wall_seconds=agent["hard_wall_seconds"],
        termination_grace_seconds=15.0 if config.agent_budget_seconds is None else 2.0,
        evaluation_device_map=device_map,
        capability_facade=ControllerAgentCapabilityFacade(
            inline_array_bytes=agent.get("inline_array_bytes", 256),
        ),
    )
    candidate_execution = SubprocessCandidateExecutionService(
        root / "candidate_execution",
        task=PythonTaskWorkerSpec.from_task_type(PairedNSSolverTask if paired else NSSolverTask),
        evaluation_settings={**task_settings, "default_query": query},
        max_workers=evaluation.get("candidate_case_workers", 1),
        oracle_max_workers=min(128, evaluation.get("candidate_case_workers", 1)),
        case_hard_wall_seconds=evaluation.get("candidate_case_hard_wall_seconds", 60),
        development_batch_hard_wall_seconds=evaluation.get(
            "development_batch_hard_wall_seconds", 300
        ),
        environment={"PYTHONPATH": str(source_root)},
    )
    initial = None
    if payload.get("initial_candidate"):
        initial = codec.bind_existing(
            _path(payload["initial_candidate"], path.parent),
            storage_root=root / "initial_bundle",
            artifact_id="baseline-s0", name="NS starting candidate",
            summary="Shared predeclared starting solver.", usage="NS two-phase solver.",
        )
    _write_json(root / "baseline_input.json", {
        "config": payload, "config_path": str(path),
        "config_sha256": sha256(path.read_bytes()).hexdigest(),
        "dataset_manifest": str(manifest),
        "dataset_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "selection_manifest": (str(selection_bank.manifest_path) if selection_bank else str(manifest)),
        "selection_manifest_sha256": (selection_bank.manifest_sha256 if selection_bank
                                      else sha256(manifest.read_bytes()).hexdigest()),
        "selection_reference_kind": ("frozen_continuous_ns" if selection_bank else "original_numerical"),
        "development_environment": dataset.get("arm", "fixed_bank"),
        "expected_continuous_mms_abi": dataset.get("expected_continuous_mms_abi"),
        "effective_agent_command": command,
        "evaluation_device_map": device_map or {},
    })
    continuation = None
    pause_control = None
    audit_journals = {}
    if segment_budget is not None:
        from copy import deepcopy
        from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore, source_identity
        from pde_agent_verifier_solver.engine.audit_journal import AuditJournal
        from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPauseControl
        stable_payload = deepcopy(payload)
        stable_agent = stable_payload["agent"]
        if "gpu_devices" in stable_agent:
            stable_agent["gpu_devices"] = ["allocated"] * len(stable_agent["gpu_devices"])
        for role in stable_agent.get("roles", ()):
            role["gpu_devices"] = ["allocated"] * len(role.get("gpu_devices", ()))
        identity = {
            "config_sha256": sha256(json.dumps(stable_payload, sort_keys=True).encode()).hexdigest(),
            "source_sha256": source_identity(repository),
            "dataset_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
            "selection_manifest_sha256": (selection_bank.manifest_sha256 if selection_bank else
                                          sha256(manifest.read_bytes()).hexdigest()),
        }
        store = CheckpointStore(root / "baseline_checkpoints", identity=identity,
                                generator=session_provider if paired else None)
        continuation = store.load() if resume else None
        for filename in ("execution_audits.json", "candidate_execution_audits.json"):
            prior = (json.loads((root / filename).read_text())
                     if resume and (root / filename).is_file() else [])
            audit_journals[filename] = AuditJournal(
                (root / filename).with_suffix(".jsonl"), prior_rows=prior, encode=_json_value,
            )
        def save_progress(value):
            # Commit observed costs before the corresponding scientific state.
            # Only newly completed executions are appended at each boundary.
            for filename, service in (("execution_audits.json", execution),
                                      ("candidate_execution_audits.json", candidate_execution)):
                audit_journals[filename].append(service.audits)
            store.save(value)
        pause_control = CampaignPauseControl(save_progress, segment_budget.should_pause,
                                              hidden_batch_size=candidate_execution.max_workers)
        candidate_execution.allocation_cancellation_event = segment_budget.cancel
        if config.agent_budget_seconds is None:
            execution.allocation_should_pause = lambda: segment_budget.should_pause("literature_agent", 0)
    try:
        return run_ns_baseline(
            config, development_records=development, selection_records=selection,
            evaluator_policy=policy, agent_program=worker, agent_execution=execution,
            initial_candidate=initial, task_settings=task_settings,
            candidate_execution=candidate_execution,
            public_problem_bundle=(paired_public_problem_bundle(dataset["arm"]) if paired else
                _public_problem_bundle(selection_bank is not None, continuous_development)),
            generator=session_provider,
            generator_state=session_provider.state if session_provider else None,
            continuation=continuation, pause_control=pause_control,
            development_environment=development_environment,
            task_program=PairedNSSolverTask() if paired else None,
            selection_evaluation=(selection_bank.fixed_evaluation((PairedNSSolverTask() if paired else NSSolverTask()).create_evaluation({**task_settings, "default_query": query}))
                                  if selection_bank else None),
        )
    finally:
        try:
            for filename, service in (("execution_audits.json", execution),
                                      ("candidate_execution_audits.json", candidate_execution)):
                if segment_budget is not None:
                    journal = audit_journals[filename]
                    journal.append(service.audits)
                    _write_json(root / filename, journal.rows)
            if segment_budget is not None:
                # Search sees this segment's service objects. Final costs must
                # be calculated after restoring all preceding execution audits.
                _write_cost_report(root)
        finally:
            try:
                close = getattr(execution, "close", None)
                if callable(close):
                    close()
            finally:
                close = getattr(candidate_execution, "close", None)
                if callable(close):
                    close()


def run_from_config(config_path, output_directory, *, resume=False, segment_budget=None, gpu_token=None):
    if segment_budget is None:
        if resume:
            raise ValueError("baseline resume requires an allocation budget")
        return _run_from_config(config_path, output_directory, gpu_token=gpu_token)
    from pde_agent_verifier_solver.engine.checkpoint_store import run_writer_lock
    with run_writer_lock(output_directory), segment_budget.signals():
        return _run_from_config(config_path, output_directory, resume=resume,
                                segment_budget=segment_budget, gpu_token=gpu_token)



def main(argv=None):
    parser = argparse.ArgumentParser(description="NS external Agent baselines and paired environments")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-token", help="Rebind the one allocated Agent GPU without changing the scientific config")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--segment-seconds", type=int, default=None)
    parser.add_argument("--flush-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    from pde_agent_verifier_solver.engine.checkpoint_store import RunSegmentBudget
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused
    budget = (RunSegmentBudget(args.segment_seconds, flush_seconds=args.flush_seconds)
              if args.segment_seconds is not None else None)
    try:
        result = run_from_config(args.config, args.output, resume=args.resume, segment_budget=budget, gpu_token=args.gpu_token)
        print(json.dumps(dict(result.report), indent=2))
        return 0
    except CampaignPaused as exc:
        print(json.dumps({"status": "paused", "phase": exc.stage, "output": str(args.output)}))
        return 75
    finally:
        if budget is not None:
            budget.close()
