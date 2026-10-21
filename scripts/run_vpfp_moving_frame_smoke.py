#!/usr/bin/env python3
"""Native A loading/MMS/BASE-only canonical macro Campaign; no training.

Uses a trusted zero-macro diagnostic Agent. Success establishes interface and
lifecycle execution, not solver quality, source-scale validity or admission.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pde_agent_verifier_solver.engine.campaign import (AgentSpec,CampaignRecipe,CaseCounts,
    DevelopmentSamplingPolicy,EvaluatorPolicy,GeneratorRecipe,RoundSettings)
from pde_agent_verifier_solver.engine.evaluator import EvaluationRequest
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.sdk import (AnswerQuery,AnswerView,Artifact,HiddenSuiteSpec,OfficialEvaluationSpec,
    QueryKind,SampleSource,SamplingStrategy,TaskKind,open_public_case_session)
from pde_agent_verifier_solver.plugins.vpfp.moving_frame_binding import build_vpfp_moving_frame_binding,file_sha


class ZeroMacroCandidate:
    def __init__(self):
        self.oracle_receipts = []

    def predict_observables(self,case,query,budget=None):
        # The SDK capability exists DURING candidate evaluation, not in the
        # Agent's outer run method. Its lifetime is managed by the Evaluator.
        session = open_public_case_session(case)
        t0,t1 = case.public_data["question"]["time_domain"]
        points = np.array([[t0+.37*(t1-t0),.173,.389,.41,-.29]],dtype=np.float64)
        values = {"source":session.source(points),"rho0":session.rho0(points[:,1:3]),
                  "g0":session.g0(points[:,1:5])}
        if any(np.asarray(value).shape!=(1,) or not np.isfinite(value).all() for value in values.values()):
            raise ValueError("actual A public Oracle returned malformed/nonfinite values")
        self.oracle_receipts.append({name:np.asarray(value).tolist() for name,value in values.items()})
        return AnswerView(kind=QueryKind.POINTS,points=query.points,shape=(len(query.points),),
            fields={name:(0.,)*len(query.points) for name in query.fields})


class MacroOnlyAgent:
    def __init__(self):
        self.scores,self.oracle_receipts = [],[]

    def run(self,context):
        cases = context.evaluator.cases
        if len(cases)!=1:
            raise ValueError("A smoke requires exactly one development case")
        case = cases[0]
        candidate = ZeroMacroCandidate()
        artifact = Artifact(artifact_id="vpfp-moving-frame-zero-macro",kind=TaskKind.OBSERVABLE_MODEL,
            name="Zero macro interface diagnostic",program=candidate,
            summary="Macro-only interface probe; no solver-quality claim",usage="predict_observables(case,query)",
            round_index=context.round_plan.round_index,agent_id=context.agent.agent_id,
            metadata={"full_solution_prediction_required":False,"synthetic_candidate":True})
        response = context.evaluator.evaluate(artifact,EvaluationRequest((case.case_id,)))
        self.oracle_receipts.extend(candidate.oracle_receipts)
        self.scores.append(float(response.primary_error))
        return AgentOutcome(artifact,observations=("Queried A source/rho0/g0; predicts macros only.",))


def smoke_recipe(time_domain):
    t0,t1 = time_domain
    query = AnswerQuery.at_points(fields=("rho","momentum_x","momentum_y","kinetic_energy"),
        points=({"t":t0+.5*(t1-t0),"x":.25,"y":.25},))
    return CampaignRecipe(name="VPFP moving-frame BASE-only macro smoke",round_count=1,
        agents=(AgentSpec("moving-frame-macro-agent","Trusted macro-only diagnostic"),),
        default_round=RoundSettings(agent_budget_seconds=180,
            development_sampling=DevelopmentSamplingPolicy(strategy=SamplingStrategy.AGENT_ADAPTIVE,
                allowed_sources=(SampleSource.BASE,),max_cases_per_agent_per_round=1,starter=CaseCounts(base=1)),
            official_evaluation=OfficialEvaluationSpec(round=HiddenSuiteSpec(base=1),terminal=HiddenSuiteSpec(base=1))),
        generator=GeneratorRecipe(guidance_enabled=False,perturbations_enabled=False),
        evaluator=EvaluatorPolicy(query,development_point_count=1),task_settings={"official_point_count":1},master_seed=20260907)


@contextmanager
def forbid_online_pde(report):
    module = importlib.import_module("agent_verifier.vpfp.online_training")
    original = module.VPFPOnlineBatchGenerator.generate
    report["online_pde_generation_attempts"] = 0
    def forbidden(*args,**kwargs):
        report["online_pde_generation_attempts"] += 1
        raise RuntimeError("A-only smoke attempted an online coarse PDE solve")
    module.VPFPOnlineBatchGenerator.generate = forbidden
    try:
        yield
        if report["online_pde_generation_attempts"]:
            raise RuntimeError("online PDE attempt was suppressed")
    finally:
        module.VPFPOnlineBatchGenerator.generate = original


def run(manifest,manifest_sha256,output,device="cuda"):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise RuntimeError("native A smoke requires a Slurm compute allocation")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("choose a fresh smoke output")
    output = output.resolve(); output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema":"vpfp_moving_frame_native_macro_smoke_v1","execution_passed":False,
        "manifest_sha256":manifest_sha256,"full_solution_prediction_required":False,
        "guidance_enabled":False,"perturbations_enabled":False,"model_training_performed":False,
        "scientific_admission":False,"execution_assurance":"trusted_local_in_process",
        "requested_cases":{"development_base":1,"official_base":1,"terminal_base":1}}
    try:
        binding = build_vpfp_moving_frame_binding(manifest,manifest_sha256,output_root=output/"private-runtime",device=device)
        backend = binding.plugin.generator.backend
        report["binding_identity"] = binding.report
        agent = MacroOnlyAgent()
        with forbid_online_pde(report):
            campaign = binding.run(smoke_recipe(backend.decoder.macro_config.time_domain),
                {"moving-frame-macro-agent":agent},local_development=True)
        score = float(campaign.selected.primary_error)
        if not math.isfinite(score) or not agent.scores or any(not math.isfinite(s) for s in agent.scores):
            raise ValueError("A canonical macro scores are not finite")
        if backend._resident_cases:
            raise ValueError("A Campaign did not release resident answers")
        if (len(backend.generation_receipts)!=3
                or any(backend.oracle_calls[name]<3 for name in ("source","rho0","g0"))):
            raise ValueError("A smoke did not exercise all three BASE public Oracle cases")
        report.update(execution_passed=True,macro_score=score,development_scores=agent.scores,
            actual_A_public_oracle_probes=agent.oracle_receipts,completed_rounds=len(campaign.rounds),
            terminal_round_reports=len(campaign.terminal_round_reports),resident_answers_released=True,
            actual_A_runtime_oracle_calls=dict(backend.oracle_calls),
            generation_receipts=backend.generation_receipts,
            scientific_qualification="not_established_by_loading_MMS_or_macro_interface_smoke")
    except Exception as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
    report["wall_seconds"] = time.monotonic()-started
    (output/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args(argv)
    result = run(args.manifest,args.manifest_sha256,args.output,args.device)
    print(json.dumps({"execution_passed":result["execution_passed"],"output":str(args.output)}),flush=True)
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
