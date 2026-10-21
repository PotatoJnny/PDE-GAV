#!/usr/bin/env python3
"""Run the six approved plasma experiments through the shared PDE-GAV Controller.

This composition preserves the frozen science. It does not import a previous
checkout's launcher, replace the Controller, submit jobs, or retry failed Agents.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from threading import Lock
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
SCHEMA = "pde_gav_plasma_gpu_v1"
TRANSPORT_SCHEMA = "pde_gav_plasma_gpu_v2"
MODELS = {("gpt-5.6-sol", "max"), ("gpt-6-astra", "ultra")}
OFFICIAL = {"base": 86, "guided": 85, "base_perturbed": 85}
TERMINAL = {
    "cohorts": [
        {"cohort_id": "common_base", "guidance_source": "base",
         "suite": {"base": 256, "base_perturbed": 128}},
        {"cohort_id": "common_guided", "guidance_source": "round_guided", "round_index": 1,
         "suite": {"guided": 256, "guided_perturbed": 128}},
    ],
    "ranking": {"cohort_id": "overall", "distribution": "overall", "statistic": "mean"},
}


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(value):
    if not isinstance(value, Mapping) or any(type(v) is not int or v < 0 for v in value.values()):
        raise ValueError("Case counts must be nonnegative integers")
    return {key: val for key, val in value.items() if val}


def validate_config(config):
    """Validate this named experiment, without changing other engine defaults."""
    from pde_agent_verifier_solver.engine.gpu_execution_config import validate_gpu_run_budget
    if config.get("schema") not in (SCHEMA, TRANSPORT_SCHEMA) or config.get("equation") not in ("strong", "weak", "artemis"):
        raise ValueError("expected the explicit PDE-GAV plasma GPU experiment schema")
    if config.get("round_count") != 5 or config.get("master_seed") != 101:
        raise ValueError("preserve five rounds and seed 101")
    execution, task = config["execution"], config["task"]
    if config["schema"] == TRANSPORT_SCHEMA:
        from pde_agent_verifier_solver.plugins.plasma_transport import TRANSPORT_PROTOCOL_ID
        from pde_agent_verifier_solver.plugins.plasma_transport_settings import transport_calibration
        calibration = transport_calibration(task.get("metric_profile"), task.get("metric_calibration"))
        if (calibration is None or config.get("protocol", {}).get("metric_profile") != TRANSPORT_PROTOCOL_ID
                or config["protocol"].get("scoring_objective_hash") != calibration.objective_hash):
            raise ValueError("v2 requires an explicit, identical frozen transport objective")
        guide = config.get("generator", {}).get("settings", {})
        guide_calibration = transport_calibration(guide.get("metric_profile"), guide.get("metric_calibration"))
        if guide_calibration is None or guide_calibration.objective_hash != calibration.objective_hash:
            raise ValueError("Guidance must use the same frozen transport objective as evaluation")
    elif "metric_profile" in task or "metric_calibration" in task:
        raise ValueError("transport scoring requires a new v2 experiment, not a legacy resume")
    validate_gpu_run_budget(config)
    if (execution.get("agent_hard_wall_seconds") != 14400
            or execution.get("candidate_case_hard_wall_seconds") != 900):
        raise ValueError("this plasma experiment retains 14400-second Agent and 900-second Case budgets")
    if config.get("protocol", {}).get("parallel_evaluation") != "service_global_oracle_v1":
        raise ValueError("migrate legacy per-host Oracle limits before using the shared fair queue")
    parallel_execution_settings(config)
    if (execution.get("model"), execution.get("reasoning_effort")) not in MODELS:
        raise ValueError("only the approved Sol/max and Astra/ultra arms are supported")
    gpu = execution.get("gpu_execution", {})
    if (gpu.get("candidate_devices") != ["3"] or gpu.get("agent_sandbox_backend") != "bwrap-nvidia-v1"):
        raise ValueError("preserve the dedicated fourth Candidate GPU and qualified Agent sandbox")
    if (task.get("candidate_budget_mode") != "ns_remaining_case_v1"
            or task.get("require_prepared_solution") is not True):
        raise ValueError("query-blind preparation must use the remaining shared Case budget")
    if _counts(config["official"]) != OFFICIAL or config["terminal"] != TERMINAL:
        raise ValueError("preserve the 256-Case Official and common S0/G1 Terminal cohorts")
    if config["promotion"] != {"preset": "minimize_metric", "settings": {
            "primary": {"distribution": "overall", "statistic": "mean"}}}:
        raise ValueError("preserve the overall mean promotion objective")
    dev = config["development"]
    starter = _counts(dev["starter"]) if "starter" in dev else {"base": dev.get("starter_base")}
    if dev.get("max_cases_per_agent_per_round") != 32 or starter != {"base": 4}:
        raise ValueError("preserve the 32-Case allowance including four Base starters")
    if dev.get("case_replay", False) or dev.get("allowed_sources", ["base", "guided"]) != ["base", "guided"]:
        raise ValueError("preserve fresh adaptive Base/Guided development")
    policy = config.get("protocol", {}).get("official_nomination_policy")
    if policy != {"mode": "single_primary_v1", "start_round": 1}:
        raise ValueError("explicit single primary must apply from round one")
    agents = config["agents"]
    if (len(agents) != 2 or len({item["agent_id"] for item in agents}) != 2
            or any(not str(item.get("prompt", "")).strip() for item in agents)):
        raise ValueError("two distinct Agents with explicit scientific prompts required")
    for key, expected in (("development_point_count", 128), ("official_point_count", 256),
                          ("terminal_point_count", 256)):
        if task.get(key) != expected:
            raise ValueError("scoring point count differs: " + key)
    if any(config.get("evaluator", {}).get(key, False) for key in (
            "allow_grid_truth", "allow_point_truth", "allow_development_solution_session")):
        raise ValueError("development truth remains private")
    expected_workers = 16 if config["equation"] == "strong" else 8
    if execution.get("candidate_case_workers") != expected_workers:
        raise ValueError("preserve the frozen Candidate CPU-worker bound")
    if (execution.get("controller_rpc_timeout_seconds") != 7200
            or execution.get("development_candidate_batch_hard_wall_seconds") != 6900):
        raise ValueError("preserve development batch/RPC limits")
    legacy_attempts = 3 if config["equation"] == "weak" else 1
    attempts = execution.get("transient_resume_max_attempts")
    if (execution.get("native_session_persistence") is not True
            or type(attempts) is not int or attempts != legacy_attempts):
        raise ValueError("preserve the recorded legacy retry bound and private native sessions")
    if type(execution.get("transient_resume_until_deadline", True)) is not bool:
        raise ValueError("execution.transient_resume_until_deadline must be a boolean")
    if config["equation"] == "artemis" and task.get("scoring_coordinate_mode") != "continuous_x_v1":
        raise ValueError("Artemis scoring must preserve continuous manufactured macros")
    return config


def parallel_execution_settings(config):
    """Use upstream service-global limits; do not advertise unsupported Task ABIs."""
    execution = config["execution"]
    fields = ("solver_batch_size", "gpu_slots_per_device", "oracle_max_workers",
              "oracle_concurrency_per_device")
    result = {key: execution[key] for key in fields}
    if any(type(value) is not int or value < 1 for value in result.values()):
        raise ValueError("parallel execution counts must be explicit positive integers")
    if result["solver_batch_size"] != 1:
        raise ValueError("Plasma Tasks have not qualified prepare_solutions; keep solver_batch_size=1")
    if result["oracle_concurrency_per_device"] != 1:
        raise ValueError("Plasma uses independent numerical processes, not the NS concurrent-stream backend")
    workers = execution["candidate_case_workers"]
    if result["oracle_max_workers"] > min(workers, 128):
        raise ValueError("service-global Oracle concurrency exceeds the Candidate worker bound")
    if result["gpu_slots_per_device"] > workers:
        raise ValueError("GPU slots cannot exceed the Candidate worker bound")
    if (config["equation"] == "weak"
            and result["oracle_max_workers"] < execution["development_workers_per_gpu"]):
        raise ValueError("global Oracle limit must retain the admitted weak development lanes")
    return result


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def _checked(path, expected=None):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"expected an absolute regular asset file: {path}")
    actual = file_sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"asset SHA256 differs: {path}")
    return {"path": str(path.resolve()), "sha256": actual}


def static_check(config):
    """Check external bytes and local entrypoints; never load Torch or draw Cases."""
    from pde_agent_verifier_solver.sdk import inspect_candidate_bundle
    config = load_config(config) if isinstance(config, (str, Path)) else validate_config(config)
    if config["schema"] == TRANSPORT_SCHEMA:
        from scripts.prepare_plasma_transport_v2 import check_transport_qualification
        check_transport_qualification(config)
    worker = config["worker"]
    for key, name in (("script", "run_canonical_codex_agent.py"), ("codex_config", "canonical_codex.config.toml")):
        if Path(worker[key]).resolve() != ROOT / "scripts" / name:
            raise ValueError("use the shared current PDE-GAV worker/config, not an old runtime")
    checked = {key: _checked(value) for key, value in worker.items()}
    initial = inspect_candidate_bundle(config["candidate"]["initial_bundle"])
    if initial.tree_sha256 != config["candidate"]["initial_bundle_sha256"]:
        raise ValueError("initial Candidate bytes differ")
    assets, backend = {}, config["backend"]
    if config["equation"] == "strong":
        for key in ("base_checkpoint", "portable_codec"):
            assets[key] = _checked(backend[key], backend[key + "_sha256"])
        for key in ("base", "expanded"):
            assets[key + "_decoder"] = _checked(backend["native_spec"][key + "_decoder_checkpoint"],
                                                backend["native_spec"][key + "_decoder_sha256"])
    elif config["equation"] == "weak":
        item = backend["portable_base"]
        assets["portable_base"] = _checked(item["path"], item["sha256"])
        from pde_agent_verifier_solver.plugins.vpfp.weak_pde_gav_binding import validate_live_scientific_sources
        assets["scientific_sources"] = validate_live_scientific_sources()
    else:
        settings = backend["factory_settings"]
        for key, expected in (("admission_record_path", "expected_admission_record_sha256"),
                              ("science_config_path", "expected_science_config_sha256")):
            assets[key] = _checked(settings[key], settings[expected])
        if not Path(settings["physical_checkout"]).is_dir():
            raise ValueError("Artemis physical package is missing")
    return {"status": "static_checks_passed", "worker": checked, "assets": assets,
            "initial_bundle_sha256": initial.tree_sha256, "cuda_qualified": False}


def recipe(config, initial):
    from pde_agent_verifier_solver.engine.campaign import (AgentSpec, CampaignRecipe, CaseCounts,
        DevelopmentSamplingPolicy, EvaluatorPolicy, GeneratorRecipe, RoundSettings)
    from pde_agent_verifier_solver.sdk import (AnswerQuery, DevelopmentCoordinateMode,
        HiddenSuiteSpec, OfficialEvaluationSpec, SampleSource,
        SamplingStrategy, TerminalCohortSpec, TerminalEvaluationSpec, TerminalGuidanceSource, TerminalMetricSpec)
    artemis = config["equation"] == "artemis"
    from pde_agent_verifier_solver.plugins.vp_1d1v.problem import VP1D_MACRO_FIELDS
    from pde_agent_verifier_solver.plugins.vpfp.problem import _VPFP_MACRO_FIELDS, _VPFP_TRANSPORT_FIELDS
    vpfp_fields = _VPFP_TRANSPORT_FIELDS if config["schema"] == TRANSPORT_SCHEMA else _VPFP_MACRO_FIELDS
    query = AnswerQuery.at_points(fields=VP1D_MACRO_FIELDS if artemis else vpfp_fields,
                                  points=({"t": .5, "x": 0.} if artemis else {"t": .5, "x": 0., "y": 0.},))
    cohorts = tuple(TerminalCohortSpec(cohort_id=row["cohort_id"], suite=HiddenSuiteSpec(**row["suite"]),
        guidance_source=TerminalGuidanceSource(row["guidance_source"]), round_index=row.get("round_index"))
        for row in config["terminal"]["cohorts"])
    return CampaignRecipe(name=config["campaign_id"], round_count=5,
        agents=tuple(AgentSpec(**row) for row in config["agents"]), master_seed=101,
        default_round=RoundSettings(agent_budget_seconds=14400,
            development_sampling=DevelopmentSamplingPolicy(strategy=SamplingStrategy.AGENT_ADAPTIVE,
                max_cases_per_agent_per_round=32, starter=CaseCounts(base=4),
                allowed_sources=(SampleSource.BASE, SampleSource.GUIDED), hide_case_types=True,
                cumulative_fixed_cases=False),
            official_evaluation=OfficialEvaluationSpec(round=HiddenSuiteSpec(**OFFICIAL))),
        generator=GeneratorRecipe(guidance_enabled=True, perturbations_enabled=True,
                                  settings=dict(config.get("generator", {}).get("settings", {}))),
        evaluator=EvaluatorPolicy(default_query=query,
            development_coordinate_mode=DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS,
            development_point_count=128, allow_grid_truth=False, allow_point_truth=False,
            allow_development_solution_session=False),
        task_settings=dict(config["task"]), initial_candidate=initial, promotion=config["promotion"],
        terminal_evaluation=TerminalEvaluationSpec(cohorts=cohorts,
            ranking=TerminalMetricSpec(**config["terminal"]["ranking"])), terminal_score_distribution="overall")


@dataclass
class Composition:
    controller: object
    binding: object
    agent_execution: object
    candidate_execution: object
    development_oracle: object | None
    provenance: dict

    def close(self):
        errors = []
        for service, method in ((self.agent_execution, "close"), (self.candidate_execution, "close"),
                                (self.development_oracle, "shutdown")):
            cleanup = getattr(service, method, None)
            if callable(cleanup):
                try:
                    cleanup()
                except BaseException as error:
                    errors.append(error)
        if errors:
            failure = RuntimeError("plasma service cleanup could not be confirmed")
            for error in errors:
                failure.add_note(f"{type(error).__name__}: {error}")
            raise failure from errors[0]


def _binding(config, run_root):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("physical plasma composition requires a compute allocation")
    equation = config["equation"]
    metric = ({key: config["task"][key] for key in ("metric_profile", "metric_calibration")}
              if config.get("schema") == TRANSPORT_SCHEMA else {})
    if equation == "strong":
        from pde_agent_verifier_solver.plugins.vpfp.iteration_backend import build_vpfp_strong_iteration_binding
        from pde_agent_verifier_solver.plugins.vpfp.resume import install_resume_hooks
        os.environ["VPFP_STRONG_ASSET_ROOT"] = config["asset_bindings"]["strong_asset_root"]
        binding = build_vpfp_strong_iteration_binding(**config["backend"], **metric, output_root=run_root / "private_backend")
        install_resume_hooks(binding.plugin.generator)
        return binding, {}
    if equation == "weak":
        from pde_agent_verifier_solver.plugins.vpfp.weak_pde_gav_binding import build_pde_gav_weak_binding
        return build_pde_gav_weak_binding(config["backend"], run_root / "private_backend", **metric)
    from pde_agent_verifier_solver.plugins.vp_1d1v.product import load_artemis_backend
    binding = load_artemis_backend(**config["backend"]["factory_settings"], **metric)
    binding.plugin.generator._backend().continuous_macro_truth = True
    return binding, {}


def compose(config, config_path, run_root, *, codex_binary, auth_file, worker_gpus=("0", "1"),
            controller_gpu="2", events=None, evaluation_device_map=None, binding_check=None):
    from pde_agent_verifier_solver.engine.agent_execution import CommandAgentProgram, SubprocessAgentExecutionService
    from pde_agent_verifier_solver.engine.candidate_execution import SubprocessCandidateExecutionService, PythonTaskWorkerSpec
    from pde_agent_verifier_solver.engine.gpu_execution_config import (candidate_gpu_kwargs,
        candidate_execution_capabilities, agent_gpu_worker_settings)
    from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
    from pde_agent_verifier_solver.sdk import TaskKind
    validate_config(config)
    if tuple(worker_gpus) != ("0", "1") or controller_gpu != "2":
        raise ValueError("this four-card deployment reserves Agents 0/1, Controller 2, Candidate 3")
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    gpu_options = candidate_gpu_kwargs(config["execution"], reserved_devices=(controller_gpu, *worker_gpus))
    binding, receipt = _binding(config, run_root)
    if binding_check is not None:
        receipt = {**receipt, "interface_check": binding_check(binding, config, run_root)}
    codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
    item = config["candidate"]
    initial = codec.bind_existing(item["initial_bundle"], storage_root=run_root / "controller_candidates",
        artifact_id=f"{config['equation']}-s0", name=item["initial_name"], summary=item["initial_summary"],
        usage="prepare_solution then evaluate points")
    if initial.program.tree_sha256 != item["initial_bundle_sha256"]:
        raise ValueError("initial Candidate changed before composition")
    plan = recipe(config, initial)
    runtime = run_root / "agent_runtime"
    runtime.mkdir(exist_ok=True, mode=0o700)
    execution = config["execution"]
    programs = {row["agent_id"]: CommandAgentProgram(
        (sys.executable, config["worker"]["script"], "--codex-binary", str(codex_binary),
         "--auth-file", str(auth_file), "--runtime-root", str(runtime),
         "--config", config["worker"]["codex_config"]),
        codec.decode_nomination, incumbent_encoder=codec.encode_incumbent,
        environment={"PYTHONPATH": str(ROOT / "src"), "PATH": str(Path(sys.executable).parent)
                     + os.pathsep + os.environ.get("PATH", os.defpath),
                     "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        worker_settings={"profile": config["schema"], "model": execution["model"],
            "reasoning_effort": execution["reasoning_effort"],
            "native_session_persistence": True,
            "transient_resume_max_attempts": execution["transient_resume_max_attempts"],
            "transient_resume_until_deadline": execution.get("transient_resume_until_deadline", True),
            # Keep frozen scientific-config provenance, but use the one shared
            # upstream selection protocol (validated as active from round one).
            "nomination_policy": config["protocol"]["official_nomination_policy"]["mode"],
            "candidate_abi": "prepare_solution_then_evaluate_points_v1",
            "candidate_execution": candidate_execution_capabilities(execution),
            "controller_rpc_timeout_seconds": execution["controller_rpc_timeout_seconds"],
            **agent_gpu_worker_settings(execution)}, gpu_devices=(gpu,))
        for row, gpu in zip(config["agents"], worker_gpus, strict=True)}
    # All plasma families below use isolated Backend Oracle workers for hidden
    # evaluation. Do not also configure the in-process evaluation device pool;
    # the Backend scope owns the same Controller/idle-Agent GPUs at this boundary.
    agents = SubprocessAgentExecutionService(run_root / "agent_turns", hard_wall_seconds=14400,
        enforce_active_budget=False, controller_gpu_devices=(controller_gpu,),
        evaluation_device_map=None, poll_interval_seconds=.25)
    task = binding.plugin.task("observable_model")
    candidates = development = None
    try:
        options = {"oracle_inline_array_bytes": 8 * 1024 * 1024, "oracle_trusted_results": True} if config["equation"] == "strong" else {}
        candidates = SubprocessCandidateExecutionService(run_root / "candidate_execution",
            task=PythonTaskWorkerSpec.from_task_type(type(task)),
            evaluation_settings={**plan.task_settings, "default_query": plan.evaluator.default_query},
            max_workers=execution["candidate_case_workers"], case_hard_wall_seconds=900,
            development_batch_hard_wall_seconds=6900, python_executable=sys.executable,
            **parallel_execution_settings(config),
            oracle_drain_timeout_seconds=execution.get("oracle_drain_timeout_seconds", 30),
            **options, **gpu_options)
        devices = (controller_gpu, *worker_gpus)
        if config["equation"] == "strong":
            from pde_agent_verifier_solver.plugins.vpfp.evaluation_gpu_reuse import VPFPHiddenEvaluationScope
            options = config.get("migration", {}).get("oracle_execution_options", {})
            candidates.hidden_evaluation_scope = VPFPHiddenEvaluationScope(binding, candidates,
                devices=devices, events=events, **options)
        elif config["equation"] == "weak":
            from pde_agent_verifier_solver.plugins.vpfp.weak_parallel_oracle import (
                WeakVPFPHiddenEvaluationScope, install_development_oracle)
            development = install_development_oracle(binding, candidates, controller_gpu=controller_gpu,
                workers_per_gpu=execution["development_workers_per_gpu"], events=events)
            candidates.hidden_evaluation_scope = WeakVPFPHiddenEvaluationScope(binding, candidates,
                devices=devices, workers_per_gpu=execution["workers_per_gpu"], events=events)
        else:
            from pde_agent_verifier_solver.plugins.vp_1d1v.evaluation_gpu_reuse import ArtemisHiddenEvaluationScope
            candidates.hidden_evaluation_scope = ArtemisHiddenEvaluationScope(binding, candidates,
                backend_settings=config["backend"]["factory_settings"], devices=devices, events=events)
        controller = binding.controller(plan, programs, agent_execution=agents, candidate_execution=candidates,
                                        events=events, task_id="observable_model")
    except BaseException as error:
        try:
            Composition(None, binding, agents, candidates, development, {}).close()
        except BaseException as cleanup:
            error.add_note(f"composition cleanup failed: {cleanup}")
        raise
    return Composition(controller, binding, agents, candidates, development,
        {"schema": SCHEMA, "equation": config["equation"], "backend": dict(binding.report),
         "binding_receipt": receipt, "candidate_gpu_devices": ["3"],
         "parallel_execution": parallel_execution_settings(config),
         "hidden_oracle_devices": list(devices), "config_path": str(config_path)})


def jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (Path, Enum)):
        return str(value) if isinstance(value, Path) else value.value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [jsonable(item) for item in value]
    if is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in fields(value) if field.name != "program"}
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _publish(artifact, destination):
    from pde_agent_verifier_solver.sdk import inspect_candidate_bundle, freeze_candidate_bundle
    bundle = inspect_candidate_bundle(artifact.program.bundle_locator)
    if bundle.tree_sha256 != artifact.program.tree_sha256:
        raise ValueError("selected Candidate no longer matches its immutable identity")
    frozen = freeze_candidate_bundle(bundle, destination)
    return {"bundle": str(frozen.root), "sha256": frozen.tree_sha256}


def run(config_path, run_root, *, codex_binary, auth_file, worker_gpus=("0", "1"),
        controller_gpu="2", allocation_seconds, shutdown_margin_seconds=2400, resume=False,
        evaluation_device_map=None, binding_check=None):
    """One allocation segment. Only a clean Controller pause is retry-safe."""
    from pde_agent_verifier_solver.engine.checkpoint_store import (
        CheckpointStore, RunSegmentBudget, atomic_json, atomic_bytes, run_writer_lock, source_identity)
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPauseControl, CampaignPaused
    from pde_agent_verifier_solver.engine.audit_journal import AuditJournal
    from pde_agent_verifier_solver.engine.runtime import EventStream
    config_path, run_root = Path(config_path).resolve(), Path(run_root).resolve()
    config = load_config(config_path)
    checkpoint_path = run_root / "checkpoints/checkpoint.json"
    qualification_path = run_root / "qualification/accepted.json"
    if not resume and run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError("fresh campaign root is not empty")
    if resume and not checkpoint_path.is_file():
        raise ValueError("resume requires a durable campaign checkpoint")
    if resume and (run_root / "status.json").is_file():
        prior = json.loads((run_root / "status.json").read_text())
        if prior.get("status") != "paused" or prior.get("cleanup_complete") is not True:
            raise ValueError("resume requires a previously clean paused segment")
    budget = RunSegmentBudget(allocation_seconds, flush_seconds=shutdown_margin_seconds)
    composition = None
    journals = {}
    result_status = None
    primary_error = None
    lock_acquired = False
    def persist(name, value):
        atomic_json(run_root / name, jsonable(value))
    def flush():
        if composition is None:
            return
        for name, service in (("execution_audits", composition.agent_execution),
                              ("candidate_execution_audits", composition.candidate_execution)):
            if name not in journals:
                journals[name] = AuditJournal(run_root / f"{name}.jsonl", encode=jsonable)
            journals[name].append(service.audits)
            persist(name + ".json", journals[name].rows)
        generator = composition.binding.plugin.generator
        backend = getattr(generator, "backend", None)
        resolve = getattr(generator, "_backend", None)
        if backend is None and callable(resolve):
            backend = resolve()
        guide = getattr(backend, "guidance_adapter", None) or backend
        persist("guidance_audits.json", {
            "training": getattr(guide, "training_audits", ()),
            "sampling": getattr(guide, "sampling_audits", ()),
        })
        if composition.development_oracle is not None:
            persist("development_oracle_audits.json", composition.development_oracle.reports)
    try:
        with run_writer_lock(run_root), budget.signals():
            lock_acquired = True
            inputs = static_check(config)
            identity = {"config_sha256": file_sha(config_path), "source_sha256": source_identity(ROOT),
                        "runtime_sha256": file_sha(codex_binary)}
            if identity["runtime_sha256"] != config["execution"]["gpu_execution"]["agent_runtime_sha256"]:
                raise ValueError("runtime differs from the explicitly qualified binary")
            if resume:
                if qualification_path.is_symlink() or not qualification_path.is_file():
                    raise ValueError("resume requires the original interface qualification receipt")
                accepted = json.loads(qualification_path.read_text())
                if (accepted.get("schema") != "plasma_interface_qualification_v1"
                        or accepted.get("identity") != identity
                        or accepted.get("report", {}).get("status") != "passed"
                        or accepted.get("report", {}).get("private_probe_released") is not True):
                    raise ValueError("original interface qualification identity differs or did not pass")
            elif binding_check is None:
                from scripts.check_plasma_gpu_integration import check_binding
                def binding_check(binding, configuration, root):
                    return check_binding(configuration["equation"], binding,
                        seed=configuration["master_seed"], output_path=root / "qualification/binding.json")
            persist("resolved_config.json", config)
            persist("static_checks.json", inputs)
            persist("status.json", {"status": "running", "retry_safe": False, "updated_at_unix": time.time()})
            events = EventStream()
            event_lock = Lock()
            def record(event):
                with event_lock, (run_root / "events.jsonl").open("a") as stream:
                    os.chmod(stream.name, 0o600)
                    stream.write(json.dumps(jsonable(event), sort_keys=True) + "\n")
            events.subscribe(record)
            composition = compose(config, config_path, run_root, codex_binary=codex_binary, auth_file=auth_file,
                worker_gpus=worker_gpus, controller_gpu=controller_gpu, events=events,
                evaluation_device_map=evaluation_device_map, binding_check=None if resume else binding_check)
            if not resume:
                report = composition.provenance.get("binding_receipt", {}).get("interface_check", {})
                if report.get("status") != "passed" or report.get("private_probe_released") is not True:
                    raise ValueError("first segment requires a passed and cleaned interface qualification")
                persist("qualification/accepted.json", {"schema": "plasma_interface_qualification_v1",
                    "identity": identity, "report": report})
            identity["interface_qualification_sha256"] = file_sha(qualification_path)
            provenance = {**composition.provenance, "interface_qualification": {
                "path": str(qualification_path), "sha256": identity["interface_qualification_sha256"],
                "reused_without_resampling": resume}}
            persist("resume_provenance.json" if resume else "input_provenance.json", provenance)
            generator = composition.binding.plugin.generator
            if not callable(getattr(generator, "export_runtime_state", None)):
                raise ValueError("segmented campaign requires the plugin's exact-state checkpoint hooks")
            store = CheckpointStore(run_root / "checkpoints", identity=identity, generator=generator)
            continuation = store.load() if resume else None
            def save(value):
                flush()
                store.save(value)
            candidates = composition.candidate_execution
            candidates.allocation_cancellation_event = budget.cancel
            pause = CampaignPauseControl(save, budget.should_pause,
                hidden_batch_size=config["execution"].get("hidden_batch_size", candidates.max_workers),
                agent_start_margin_seconds=600, hidden_start_margin_seconds=120)
            try:
                result = composition.controller.run(continuation=continuation, pause_control=pause)
                persist("result.json", result)
                atomic_bytes(run_root / "CAMPAIGN_MEMORY.md", (result.memory.markdown + "\n").encode())
                persist("candidate_history.json", result.candidate_history)
                selected = _publish(result.selected.artifact, run_root / "selected_candidate")
                result_status = {"status": "completed", "retry_safe": False,
                                 "selected": selected, "rounds_completed": len(result.rounds)}
            except CampaignPaused as paused:
                if not store.path.is_file() or store.last_payload is None:
                    raise RuntimeError("Controller pause did not persist a checkpoint") from paused
                result_status = {"status": "paused", "retry_safe": True, "phase": paused.stage,
                                 "stage": paused.stage}
            finally:
                primary_error = sys.exc_info()[1]
                cleanup_errors = []
                try:
                    composition.close()
                except BaseException as error:
                    cleanup_errors.append(error)
                try:
                    flush()
                except BaseException as error:
                    cleanup_errors.append(error)
                if cleanup_errors:
                    failure = RuntimeError("campaign cleanup or audit persistence failed; do not auto-resume")
                    for error in cleanup_errors:
                        failure.add_note(str(error))
                    if primary_error is not None:
                        primary_error.add_note(str(failure))
                    else:
                        raise failure from cleanup_errors[0]
            result_status.update(cleanup_complete=True, checkpoint_path=str(checkpoint_path),
                                 updated_at_unix=time.time(), equation=config["equation"])
            persist("status.json", result_status)
            return result_status
    except BaseException as error:
        if not lock_acquired:
            raise
        # Errors before Controller.run can still leave a newly constructed manager.
        if composition is not None and result_status is None:
            try:
                composition.close()
            except BaseException as cleanup:
                error.add_note(f"cleanup also failed: {cleanup}")
        try:
            atomic_bytes(run_root / "failure_traceback.txt", traceback.format_exc().encode())
            persist("status.json", {"status": "failed", "retry_safe": False,
                "cleanup_complete": False, "error_type": type(error).__name__, "error": str(error),
                "updated_at_unix": time.time()})
        except BaseException as persistence:
            error.add_note(f"failure record also failed: {persistence}")
        raise
    finally:
        budget.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--codex-binary", required=True, type=Path)
    parser.add_argument("--auth-file", required=True, type=Path)
    parser.add_argument("--worker-gpus", default="0,1")
    parser.add_argument("--controller-gpu", default="2")
    parser.add_argument("--segment-seconds", required=True, type=float)
    parser.add_argument("--shutdown-margin-seconds", default=2400, type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.config, args.run_root, codex_binary=args.codex_binary, auth_file=args.auth_file,
        worker_gpus=tuple(args.worker_gpus.split(",")), controller_gpu=args.controller_gpu,
        allocation_seconds=args.segment_seconds, shutdown_margin_seconds=args.shutdown_margin_seconds,
        resume=args.resume)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 75 if result["status"] == "paused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
