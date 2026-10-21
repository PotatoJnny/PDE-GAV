"""One-round canonical smoke with real Plasma assets and a zero-macro baseline.

The Agent is a trusted local diagnostic program, not an isolated LLM run.
No training is performed; a finite score verifies the interface path only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from hashlib import sha256
from enum import Enum
import importlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
from time import perf_counter
import traceback
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CampaignRecipe, CaseCounts, DevelopmentSamplingPolicy,
    EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.evaluator import EvaluationRequest
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Artifact, HiddenSuiteSpec, OfficialEvaluationSpec,
    QueryKind, SampleSource, SamplingStrategy, TaskKind,
)


class ZeroMacroCandidate:
    """Return only declared macro scalars (or GKW complex pairs); never f."""

    def __init__(self, physics):
        self.physics = physics

    def predict_observables(self, case, query, budget=None):
        del case, budget
        if query.kind is not QueryKind.POINTS:
            raise ValueError("zero-macro smoke Candidate supports POINTS only")
        components = 2 if self.physics == "gkw" else 1
        return AnswerView(
            kind=QueryKind.POINTS, points=query.points,
            fields={name: (0.,) * (len(query.points) * components) for name in query.fields},
            shape=(len(query.points), 2) if components == 2 else (len(query.points),),
        )


class ZeroMacroAgent:
    def __init__(self, physics):
        self.physics = physics
        self.development_scores = []

    def run(self, context):
        cases = context.evaluator.cases
        if len(cases) != 1:
            raise ValueError("canonical smoke requires exactly one development Case")
        artifact = Artifact(
            artifact_id=f"{self.physics}-zero-macros-r{context.round_plan.round_index}",
            kind=TaskKind.OBSERVABLE_MODEL, name="Zero macro baseline",
            program=ZeroMacroCandidate(self.physics), summary="Diagnostic zero predictions only",
            usage="predict_observables(case, query)", round_index=context.round_plan.round_index,
            agent_id=context.agent.agent_id,
            metadata={"diagnostic_baseline": True, "full_solution_prediction_required": False},
        )
        response = context.evaluator.evaluate(
            artifact, EvaluationRequest(tuple(case.case_id for case in cases)),
        )
        self.development_scores.append(float(response.primary_error))
        return AgentOutcome(artifact, observations=(
            "Trusted local macro-only interface smoke; no learning or performance claim.",
            f"Development macro score: {response.primary_error}",
        ))


def smoke_query(physics):
    if physics == "artemis":
        # Artemis Task replaces this template with a non-initial native time
        # and native x coordinate for development, Official and Terminal plans.
        return AnswerQuery.at_points(fields=("M0", "M1", "M2", "M3", "E"),
                                     points=({"t": .5, "x": 0.},))
    if physics == "gkw":
        return AnswerQuery.at_points(
            fields=("particle_density", "parallel_momentum", "kinetic_energy", "phi"),
            points=({"t": .75, "s_index": 0, "kx_index": 42, "ky_index": 0},),
        )
    return AnswerQuery.at_points(
        fields=("rho", "momentum_x", "momentum_y", "kinetic_energy"),
        points=({"t": .5, "x": .25, "y": .25},),
    )


def smoke_recipe(physics):
    return CampaignRecipe(
        name=f"{physics} real-assets canonical macro-only smoke", round_count=1,
        agents=(AgentSpec("zero-macro-agent", "Trusted zero-macro diagnostic Agent"),),
        default_round=RoundSettings(
            agent_budget_seconds=120,
            development_sampling=DevelopmentSamplingPolicy(
                strategy=SamplingStrategy.AGENT_ADAPTIVE,
                allowed_sources=(SampleSource.BASE,),
                max_cases_per_agent_per_round=1, starter=CaseCounts(base=1),
            ),
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(base=1, base_perturbed=1),
                terminal=HiddenSuiteSpec(base=1, base_perturbed=1),
            ),
        ),
        generator=GeneratorRecipe(guidance_enabled=False, perturbations_enabled=True),
        evaluator=EvaluatorPolicy(smoke_query(physics), development_point_count=1),
        task_settings={"official_point_count": 1}, master_seed=20260904,
    )


def _validate_learned_options(physics, checkpoint, checkpoint_sha256):
    if (checkpoint is None) != (checkpoint_sha256 is None):
        raise ValueError("--learned-checkpoint and --learned-checkpoint-sha256 must be supplied together")
    if checkpoint is not None:
        if physics != "vpfp":
            raise ValueError("learned-checkpoint smoke options are supported only for VPFP")
        if (len(checkpoint_sha256) != 64
                or any(character not in "0123456789abcdef" for character in checkpoint_sha256)):
            raise ValueError("--learned-checkpoint-sha256 must be a lowercase 64-character SHA256")


def _validate_sampler_options(physics, learned_checkpoint, learned_checkpoint_sha256,
                              diffusion_checkpoint, diffusion_checkpoint_sha256):
    if learned_checkpoint is not None and diffusion_checkpoint is not None:
        raise ValueError("--learned-checkpoint and --diffusion-checkpoint are mutually exclusive")
    _validate_learned_options(physics, learned_checkpoint, learned_checkpoint_sha256)
    if (diffusion_checkpoint is None) != (diffusion_checkpoint_sha256 is None):
        raise ValueError("--diffusion-checkpoint and --diffusion-checkpoint-sha256 must be supplied together")
    if diffusion_checkpoint is not None:
        if physics != "vpfp":
            raise ValueError("diffusion-checkpoint smoke options are supported only for VPFP")
        if (len(diffusion_checkpoint_sha256) != 64
                or any(character not in "0123456789abcdef" for character in diffusion_checkpoint_sha256)):
            raise ValueError("--diffusion-checkpoint-sha256 must be a lowercase 64-character SHA256")


def load_binding(physics, payload, output, device, *,
                 learned_checkpoint=None, learned_checkpoint_sha256=None,
                 diffusion_checkpoint=None, diffusion_checkpoint_sha256=None):
    _validate_sampler_options(physics, learned_checkpoint, learned_checkpoint_sha256,
                              diffusion_checkpoint, diffusion_checkpoint_sha256)
    if payload.get("physics") not in (None, physics):
        raise ValueError("config physics does not match --physics")
    if physics == "artemis":
        from pde_agent_verifier_solver.plugins.vp_1d1v.product import load_artemis_backend
        settings = dict(payload["factory_settings"])
        if settings.get("guidance_adapter_factory") is not None or settings.get("reward_guidance_settings") is not None:
            raise ValueError("unguided canonical smoke rejects configured guidance")
        return load_artemis_backend(**{**settings, "device": device})
    if physics == "gkw":
        from pde_agent_verifier_solver.plugins.gkw.canonical_binding import load_gkw_binding
        return load_gkw_binding(payload["factory_settings"],
            run_root=output / "physical_runtime", storage_root=output, device=device, guidance=False)
    from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import build_vpfp_online_binding
    sampling = {}
    if learned_checkpoint is not None:
        from pde_agent_verifier_solver.plugins.vpfp.learned_latent import (
            VPFPConditionalLatentDistribution, VPFPLearnedBatchGenerator,
        )
        model = VPFPConditionalLatentDistribution.load(
            learned_checkpoint, expected_sha256=learned_checkpoint_sha256)
        sampling["latent_sampler"] = VPFPLearnedBatchGenerator(
            model, checkpoint_sha256=learned_checkpoint_sha256)
    if diffusion_checkpoint is not None:
        from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
        sampling["latent_sampler"] = load_vpfp_diffusion_sampler(
            diffusion_checkpoint, expected_sha256=diffusion_checkpoint_sha256, device=device)
    return build_vpfp_online_binding(payload.get("vpfp", payload.get("factory_settings", payload)),
        output_root=output / "physical_runtime", device=device, guidance=False, **sampling)


@contextmanager
def _forbid_online_vpfp_generation(report):
    """Fail on any old coarse-PDE generation attempt throughout the Campaign.

    Binding has already installed the pinned physical imports. This guard is
    for the trusted, single-process diagnostic only; it is not an isolation
    mechanism or a distribution-fidelity qualification.
    """
    online = importlib.import_module("agent_verifier.vpfp.online_training")
    original_generate = online.VPFPOnlineBatchGenerator.generate
    report["online_pde_calls_during_learned_sampling"] = 0
    report["online_pde_call_check"] = "guarded_entire_learned_campaign"
    report["online_pde_call_count_semantics"] = "attempted_calls_to_legacy_VPFPOnlineBatchGenerator.generate"
    report["online_pde_guard_restored"] = False

    def forbidden_generate(*args, **kwargs):
        report["online_pde_calls_during_learned_sampling"] += 1
        raise RuntimeError("learned canonical Campaign attempted an online VPFP coarse PDE solve")

    online.VPFPOnlineBatchGenerator.generate = forbidden_generate
    try:
        yield
        # A backend must not turn a caught forbidden generation into a pass.
        if report["online_pde_calls_during_learned_sampling"]:
            raise RuntimeError("learned canonical Campaign suppressed a forbidden online VPFP generation attempt")
    finally:
        online.VPFPOnlineBatchGenerator.generate = original_generate
        report["online_pde_guard_restored"] = True


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unexpected smoke report value: {type(value).__name__}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", choices=("artemis", "gkw", "vpfp"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learned-checkpoint", type=Path)
    parser.add_argument("--learned-checkpoint-sha256")
    parser.add_argument("--diffusion-checkpoint", type=Path)
    parser.add_argument("--diffusion-checkpoint-sha256")
    arguments = parser.parse_args(argv)
    try:
        _validate_sampler_options(arguments.physics, arguments.learned_checkpoint,
                                  arguments.learned_checkpoint_sha256, arguments.diffusion_checkpoint,
                                  arguments.diffusion_checkpoint_sha256)
    except ValueError as error:
        parser.error(str(error))
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real Plasma smoke requires a Slurm compute allocation")
    output = arguments.output.expanduser().resolve()
    if output.exists():
        parser.error("output already exists; choose a fresh run directory")
    output.mkdir(parents=True)
    started = perf_counter()
    learned_sampling = arguments.learned_checkpoint is not None or arguments.diffusion_checkpoint is not None
    report = {
        "schema": "plasma_canonical_real_assets_smoke_v1", "physics": arguments.physics,
        "config": str(arguments.config.resolve()), "device": arguments.device,
        "process_environment": {name: os.environ.get(name) for name in
            ("JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE")},
        "execution_assurance": "trusted_local_in_process_diagnostic",
        "isolated_formal_llm_validation": False, "model_training_performed": False,
        "scientific_performance_validation": False, "guidance_qualification": "not_tested",
        "scientific_qualification": "not_established_by_interface_smoke",
        "generation_mode": ("diffusion_vpfp_checkpoint" if arguments.diffusion_checkpoint is not None else
                            "learned_vpfp_checkpoint" if arguments.learned_checkpoint is not None
                            else "configured_backend"),
        "learned_checkpoint": (str(arguments.learned_checkpoint.resolve())
                               if arguments.learned_checkpoint is not None else None),
        "learned_checkpoint_sha256": arguments.learned_checkpoint_sha256,
        "diffusion_checkpoint": (str(arguments.diffusion_checkpoint.resolve())
                                 if arguments.diffusion_checkpoint is not None else None),
        "diffusion_checkpoint_sha256": arguments.diffusion_checkpoint_sha256,
        "online_pde_calls_during_learned_sampling": None,
        "online_pde_call_check": "not_instrumented",
        "online_pde_call_count_semantics": None,
        "online_pde_guard_restored": None,
        "full_solution_prediction_required": False,
        "macro_fields": list(smoke_query(arguments.physics).fields),
        "requested_cohorts": {"development_base": 1, "official_base": 1,
            "official_base_perturbed": 1, "terminal_base": 1, "terminal_base_perturbed": 1},
    }
    try:
        report["integration_git_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        report["config_sha256"] = sha256(arguments.config.read_bytes()).hexdigest()
        payload = json.loads(arguments.config.read_text())
        learned_settings = ({"learned_checkpoint": arguments.learned_checkpoint,
                             "learned_checkpoint_sha256": arguments.learned_checkpoint_sha256}
                            if arguments.learned_checkpoint is not None else {})
        if arguments.diffusion_checkpoint is not None:
            learned_settings = {"diffusion_checkpoint": arguments.diffusion_checkpoint,
                                "diffusion_checkpoint_sha256": arguments.diffusion_checkpoint_sha256}
        binding = load_binding(arguments.physics, payload, output, arguments.device, **learned_settings)
        report["binding_report"] = _json_value(binding.report)
        agent = ZeroMacroAgent(arguments.physics)
        guard = (_forbid_online_vpfp_generation(report)
                 if learned_sampling else nullcontext())
        with guard:
            result = binding.run(smoke_recipe(arguments.physics),
                                 {"zero-macro-agent": agent}, local_development=True)
        score = float(result.selected.primary_error)
        if not math.isfinite(score) or any(not math.isfinite(value) for value in agent.development_scores):
            raise ValueError("canonical macro scores must be finite")
        report.update(status="passed", final_macro_score=score,
            development_macro_scores=agent.development_scores,
            completed_rounds=len(result.rounds), terminal_round_reports=len(result.terminal_round_reports),
            official_cohort_counts=[{"total": row.hidden_suite_size,
                                    "agent_case_counts": _json_value(row.agent_case_counts)} for row in result.rounds],
            official_distribution_metrics=[_json_value(row.hidden_distribution_metrics) for row in result.rounds],
            terminal_distribution_metrics=_json_value(result.terminal_distribution_metrics),
            promotion_performed=all(row.promotion_decision is not None for row in result.rounds),
            selected_artifact_id=result.selected.artifact.artifact_id)
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    report["wall_seconds"] = perf_counter() - started
    with (output / "report.json").open("x") as target:
        json.dump(report, target, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
