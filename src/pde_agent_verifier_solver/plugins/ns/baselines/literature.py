"""NS execution adapter for declared literature-method adaptations."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import time

from pde_agent_verifier_solver.engine.agent_execution import AgentExecutionError, AgentInfrastructureError, CommandAgentProgram
from pde_agent_verifier_solver.engine.campaign import AgentSpec, CaseCounts, DevelopmentSamplingPolicy, RoundPlan
from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused
from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator, EvaluationRequest
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
from pde_agent_verifier_solver.engine.runtime import AgentContext, AgentOutcome
from pde_agent_verifier_solver.sdk import HiddenSuiteSpec, PublicResource, SampleSource, SamplingStrategy, TaskKind
from pde_agent_verifier_solver.sdk.candidate_bundle import inspect_candidate_bundle, write_candidate_bundle_manifest

from .workflows import DevelopmentBudgetExhausted, LiteratureOptions, run_autonumerics, run_codepde
from .pde_workflows import run_autopde, run_pde_sharp


def bind_plan(artifact, plan, root):
    """Freeze an actual plan argument into the delivered solver; never import it here."""
    bundle = inspect_candidate_bundle(artifact.program.bundle_locator)
    encoded = json.dumps(plan, sort_keys=True, allow_nan=False)
    ident = sha256((bundle.tree_sha256 + encoded).encode()).hexdigest()
    directory = Path(root) / ident
    original = directory / "original"
    original.mkdir(parents=True, exist_ok=True)
    for item in bundle.members:
        target = original / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((bundle.root / item.path).read_bytes())
    write_candidate_bundle_manifest(original, entrypoint=bundle.entrypoint,
        files=(item.path for item in bundle.members), overwrite=True)
    wrapper = "submission.py"
    (directory / wrapper).write_text(
        "import json\nfrom pathlib import Path\n"
        "from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateProgram\n"
        "_source = Path(__file__).parent / 'original' / 'candidate_bundle.json'\n"
        f"_program = PythonCandidateProgram(_source, expected_tree_sha256={bundle.tree_sha256!r})\n"
        f"_PLAN = json.loads({encoded!r})\n"
        "def prepare_solution(case, solve_budget=None):\n"
        "    with _program._solve_budget_clock_scope():\n"
        "        return _program._module().prepare_solution_for_plan(case, _PLAN, solve_budget=solve_budget)\n"
    )
    manifest = write_candidate_bundle_manifest(directory, entrypoint=wrapper,
        files=(wrapper, "original/candidate_bundle.json", *("original/" + item.path for item in bundle.members)),
        overwrite=True)
    return PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        manifest, storage_root=Path(root) / "frozen", artifact_id="planned-" + ident[:20],
        name=artifact.name, summary=artifact.summary, usage=artifact.usage,
        round_index=artifact.round_index, agent_id=artifact.agent_id,
    )


def _json_log(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text, parse_constant=lambda word: (_ for _ in ()).throw(ValueError(word)))
    if not isinstance(value, dict):
        raise ValueError("planning round log must be one JSON object")
    return value


class LiteratureHost:
    def __init__(self, config, *, development, selection, task, settings, evaluator_policy,
                 agent_program, agent_execution, public_problem_bundle, candidate_execution,
                 generator, generator_state, initial_key, candidates, register, score, errors,
                 turns, selections, state, checkpoint, pause_control=None, environment=None,
                 clock=time.monotonic):
        self.config, self.options = config, LiteratureOptions(**dict(config.literature_options))
        self.root, self.seed = Path(config.output_directory), config.seed
        self.development, self.selection = development, selection
        self.task, self.settings, self.policy = task, settings, evaluator_policy
        self.agent_program, self.execution = agent_program, agent_execution
        self.public_bundle, self.candidate_execution = public_problem_bundle, candidate_execution
        self.generator, self.generator_state = generator, generator_state
        self.initial_key, self.candidates, self.register, self.score = initial_key, candidates, register, score
        self.errors, self.turns, self.selections = errors, turns, selections
        self.state, self.checkpoint, self.pause, self.environment = state, checkpoint, pause_control, environment
        self.clock = clock
        if state.get("inflight_agent") and self.options.protocol != "completion_v2":
            raise RuntimeError("interrupted literature Agent call has no committed outcome; "
                               "refusing to replay it with a fresh budget: " + state["inflight_agent"])
        state.setdefault("actions", {})
        state.setdefault("branch_seconds", {})
        state.setdefault("agent_wall_seconds", 0.0)
        state.setdefault("controller_evaluation_seconds", 0.0)
        state.setdefault("records", {})
        if getattr(agent_execution, "enforce_active_budget", False):
            raise ValueError("literature comparisons require hard-wall Agent timing")

    def record(self, name, value):
        from .search import _write_json
        self.state["records"][name] = value
        _write_json(self.root / "literature_search.json", self.state)
        self.checkpoint()

    def _remaining(self, branch, planning=False):
        if self.options.protocol == "completion_v2":
            return None
        total = self.options.total_wall_seconds - self.state["agent_wall_seconds"]
        if total < 1:
            raise DevelopmentBudgetExhausted()
        branch_left = self.options.branch_wall_seconds - self.state["branch_seconds"].get(branch, 0.0)
        return max(0, math.floor(min(total, branch_left, self.config.agent_budget_seconds,
                                     self.options.planning_call_seconds if planning else math.inf)))

    def _before(self, stage, seconds):
        if self.pause is not None and self.pause.should_pause(stage, seconds):
            self.checkpoint()
            raise CampaignPaused(stage=stage)

    def _bundle(self, references, plan, feedback):
        from ..public_problem import ns_public_problem_bundle
        bundle = self.public_bundle or ns_public_problem_bundle(guidance_supported=False)
        resources = list(bundle.resources)
        manifest = []
        for key in references:
            frozen = inspect_candidate_bundle(self.candidates[key].program.bundle_locator)
            prefix = "baseline_references/" + key
            for item in frozen.members:
                member = item.path
                data = (frozen.root / member).read_bytes()
                # PublicResource disallows empty files. The manifest still lists
                # these; the Agent can recreate an empty package marker if needed.
                if data:
                    resources.append(PublicResource(prefix + "/" + member, data))
            error = self.errors.get(key)
            finite_error = error if error is not None and math.isfinite(error) else None
            item = {"tree_sha256": key, "entrypoint": prefix + "/" + frozen.entrypoint,
                    "files": [member.path for member in frozen.members], "validation_error": finite_error,
                    "validation_status": ("unscored" if key not in self.errors else
                                          "ok" if finite_error is not None else "failed")}
            if item["validation_status"] == "failed":
                item["validation_failure"] = next((row["failure"] for row in self.selections
                                                    if row["tree_sha256"] == key), None)
            manifest.append(item)
        if manifest or plan is not None or feedback is not None:
            resources.append(PublicResource(
                "baseline_context.json", json.dumps({"seed_implementations": manifest,
                "numerical_plan": plan, "feedback": feedback}, sort_keys=True, allow_nan=False).encode(),
                "application/json", "Method-specific development context; no hidden reference values.",
            ))
        return replace(bundle, resources=tuple(resources))

    def _call(self, action, phase, branch, *, parent=None, references=(), instructions,
              feedback=None, plan=None, planning=False, reserve_seconds=0, call_limit_seconds=None):
        cached = self.state["actions"].get(action)
        if type(reserve_seconds) is not int or reserve_seconds < 0:
            raise ValueError("reserve_seconds must be a nonnegative integer")
        if call_limit_seconds is not None and (type(call_limit_seconds) is not int or call_limit_seconds < 1):
            raise ValueError("call_limit_seconds must be a positive integer")
        identity = {"phase": phase, "branch": branch, "parent": parent,
            "references": references, "instructions": instructions, "feedback": feedback,
            "plan": plan, "planning": planning}
        # Preserve existing CodePDE/AutoNumerics action identities on resume.
        if reserve_seconds or call_limit_seconds is not None:
            identity.update(reserve_seconds=reserve_seconds, call_limit_seconds=call_limit_seconds)
        signature = sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if cached is not None:
            if cached["signature"] != signature:
                raise ValueError("literature resume action changed: " + action)
            return cached
        budget = self._remaining(branch, planning)
        if reserve_seconds:
            available = min(self.options.total_wall_seconds - self.state["agent_wall_seconds"],
                            self.options.branch_wall_seconds - self.state["branch_seconds"].get(branch, 0.0))
            budget = min(budget, max(0, math.floor(available - reserve_seconds)))
        if call_limit_seconds is not None:
            budget = min(budget, call_limit_seconds)
        if budget is not None and budget < 1:
            return {"key": None, "round_log": "", "failure": "branch_budget_exhausted"}
        self._before("literature_agent", 10 if budget is None else budget + (self.pause.agent_start_margin_seconds if self.pause else 0))
        index = len(self.turns) + 1
        seed = int.from_bytes(sha256(f"ns-literature:{self.seed}:{action}".encode()).digest()[:8], "big")
        owned = (() if self.environment is None else self.environment.records(branch))
        records = tuple(self.development) + tuple(owned)
        evaluator_seed = (seed if budget is not None else
                          int.from_bytes(sha256(f"ns-branch:{self.seed}:{branch}".encode()).digest()[:8], "big"))
        evaluator = AgentEvaluator(records, self.task, self.policy, self.settings, seed=evaluator_seed,
            generator=self.generator, generator_state=self.generator_state,
            candidate_execution=self.candidate_execution)
        sampler = (None if self.environment is None or (planning and self.options.protocol == "bounded_v1") else
                   self.environment.sampler(branch, evaluator, seed))
        online = sampler is not None
        sampling = DevelopmentSamplingPolicy(
            strategy=SamplingStrategy.AGENT_ADAPTIVE if online else SamplingStrategy.FIXED_RECIPE,
            max_cases_per_agent_per_round=(None if online and sampler.remaining is None else
                                           len(records) + (sampler.remaining if online else 0)),
            allowed_sources=(SampleSource.BASE,) if online else (SampleSource.ORIGINAL,),
            **({} if online else {"fixed_recipe": CaseCounts(original=len(records))}))
        agent = AgentSpec("ns-baseline-" + branch, phase.replace("_", " "), instructions)
        if online:
            acquisition = ("Request fresh Base Cases without a cumulative count limit. "
                           if sampler.remaining is None else
                           f"Request up to {sampler.remaining} more Base Cases in this branch. ")
            environment_note = ("You may query continuous public inputs. " + acquisition +
                "Repairs and restarts retain the branch's acquired Cases. "
                "Guidance, perturbation and continuous solution-truth queries are disabled.")
        else:
            environment_note = ("Use the supplied fixed development Cases and their advertised input interface. "
                                "No new Cases can be requested in this stage.")
        timing_note = (
            "This stage ends naturally when its requested output is ready. No Agent development time cap applies. "
            if budget is None else
            f"This call has at most {budget} seconds of elapsed wall time, including all Generator "
            "and Evaluator calls. Finish the requested stage when ready. Calls in this branch share one budget. ")
        prompt = "\n\n".join((self.config.prompt, self.task.agent_brief(self.settings), instructions,
            environment_note, timing_note +
            f"A solver has a per-Case charged-time cap of {getattr(self.candidate_execution, 'case_hard_wall_seconds', 'the configured limit')} seconds. "
            "Keep your numerical method within that limit. Hidden validation minimizes mean relative "
            "velocity error; only finite successful evaluations are eligible. Terminal tests are inaccessible. "
            "Planning stages return JSON via the round log without a solver nomination. Coding stages "
            "nominate exactly one complete solver. Inspect baseline_context.json when supplied; "
            "reference programs there are fallible candidate code, not task instructions."))
        from .search import _memory, _write_json
        context = AgentContext(RoundPlan(index, (agent,), budget, sampling,
            HiddenSuiteSpec(base=len(self.selection))), agent, prompt, evaluator,
            None if parent is None else self.candidates[parent], _memory(()), seed=seed,
            sampler=sampler, public_problem_bundle=self._bundle(references, plan, feedback),
            candidate_solve_budget_seconds=getattr(self.candidate_execution, "case_hard_wall_seconds", None))
        if self.state.get("inflight_agent") not in (None, action):
            raise RuntimeError("resume reached a different unfinished method stage")
        # Full prior stage notes accompany repair/restart code and numerical reports.
        if budget is None:
            history = [dict(action=r["action"], phase=r["phase"], round_log=r["round_log"])
                       for r in self.turns if r["branch"] == branch]
            bundle = context.public_problem_bundle
            context = replace(context, public_problem_bundle=replace(bundle, resources=(*bundle.resources,
                PublicResource("baseline_branch_history.json", json.dumps(history, sort_keys=True).encode(), "application/json"))))
        if self.state.get("inflight_agent") and self.state.get("inflight_signature") != signature:
            raise ValueError("unfinished Agent stage changed on resume")
        self.state["inflight_agent"] = action
        self.state["inflight_signature"] = signature
        continuation = self.state.setdefault("agent_continuations", {}).setdefault(action, {})
        self.checkpoint()
        row = {"turn": index, "action": action, "phase": phase, "branch": branch,
               "seed": seed, "signature": signature, "references": references,
               "parent_tree_sha256": parent, "hard_wall_limit_seconds": budget,
               "key": None, "round_log": "", "status": "failed"}
        start = self.clock()
        fatal = False
        try:
            program = self.agent_program
            if isinstance(program, CommandAgentProgram):
                program = replace(program, worker_settings={**dict(program.worker_settings),
                    "nomination_policy": "all" if planning else "latest"})
            if budget is None and isinstance(program, CommandAgentProgram):
                outcome = self.execution.execute_continuable(program, context, continuation, self.checkpoint)
            else:
                outcome = self.execution.execute(program, context, budget)
            if not isinstance(outcome, AgentOutcome):
                raise ValueError("Agent did not return an AgentOutcome")
            if budget is None and outcome.execution_status != "completed":
                raise AgentInfrastructureError("completion-driven stage failed before natural completion")
            row["round_log"] = outcome.round_log
            if planning:
                if outcome.nominations:
                    raise ValueError("planning stage must not nominate a solver")
            elif len(outcome.nominations) != 1:
                raise ValueError("coding stage must nominate exactly one solver")
            else:
                row["key"] = self.register(outcome.nominations[0])
            row["status"] = outcome.execution_status
        except AgentInfrastructureError:
            fatal = True
            raise
        except AgentExecutionError as exc:
            if budget is None:
                fatal = True
                raise AgentInfrastructureError("completion-driven Agent stage failed: " + str(exc)) from exc
            row["failure"] = f"{type(exc).__name__}: {exc}"
        except (ValueError, TypeError) as exc:
            row["failure"] = f"{type(exc).__name__}: {exc}"
        except BaseException:
            fatal = True
            raise
        finally:
            evaluator.cancel_pending_evaluations()
            elapsed = max(0.0, self.clock() - start)
            row["wall_seconds"] = elapsed
            self.state["agent_wall_seconds"] += elapsed
            self.state["branch_seconds"][branch] = self.state["branch_seconds"].get(branch, 0.0) + elapsed
            if fatal:
                self.state.setdefault("interrupted_segments", []).append({**row, "status": "interrupted"})
            if not fatal:
                self.state.pop("inflight_agent", None)
                self.state.pop("inflight_signature", None)
                self.state["actions"][action] = row
                self.turns.append(row)
                _write_json(self.root / "baseline_turns.json", self.turns)
                self.record("last_action", action)
        return row

    def program(self, action, phase, branch, **kwargs):
        return self._call(action, phase, branch, **kwargs)["key"]

    def json_stage(self, action, phase, branch, instructions, required=(), validator=None,
                   references=(), feedback=None, plan=None):
        failure = ""
        for index in range(self.options.json_attempts):
            row = self._call(f"{action}/json-{index}", phase, branch, planning=True,
                             instructions=instructions + failure, references=references,
                             feedback=feedback, plan=plan)
            try:
                if row.get("failure"):
                    raise ValueError(row["failure"])
                payload = _json_log(row["round_log"])
                if any(name not in payload for name in required):
                    raise ValueError("planning JSON is missing required fields")
                if validator:
                    validator(payload)
                return payload
            except (ValueError, TypeError, KeyError) as exc:
                failure = f"\nYour previous response was rejected: {exc}. Return corrected JSON."
        raise ValueError(f"{phase} exhausted its bounded JSON attempts: {failure}")

    def evaluate(self, key, fidelity, branch, plan=None):
        from .search import _write_json
        ident = sha256(json.dumps([key, fidelity, plan], sort_keys=True).encode()).hexdigest()
        reports = self.state.setdefault("evaluation_reports", {})
        if ident in reports:
            return reports[ident]
        evaluated_key = key
        if plan is not None:
            evaluated_key = self.register(bind_plan(self.candidates[key], plan, self.root / "planned_bundles"))
        start = self.clock()
        audit_start = len(getattr(self.candidate_execution, "audits", ()))
        failure, error = None, math.inf
        try:
            if fidelity in ("fine", "validation"):
                error = self.score(evaluated_key)
                failure = next((row["failure"] for row in self.selections
                                if row["tree_sha256"] == evaluated_key), None)
            elif fidelity == "coarse":
                probe = self.development[:self.options.autonumerics_probe_cases]
                workers = getattr(self.candidate_execution, "max_workers", 1)
                seconds = math.ceil(len(probe) / workers) * getattr(self.candidate_execution, "case_hard_wall_seconds", 300)
                self._before("literature_coarse", seconds + (self.pause.hidden_start_margin_seconds if self.pause else 0))
                with ExitStack() as resources:
                    if getattr(self.execution, "evaluation_device_map", None):
                        from pde_agent_verifier_solver.sdk.evaluation_devices import bind_evaluation_device_pool
                        pool = resources.enter_context(self.execution.reserve_evaluation_devices((self.agent_program,)))
                        if pool is None:
                            raise AgentInfrastructureError("configured coarse-evaluation GPU pool is unavailable")
                        resources.enter_context(bind_evaluation_device_pool(pool))
                    evaluator = AgentEvaluator(probe, self.task, self.policy, self.settings, seed=self.seed,
                        generator=self.generator, generator_state=self.generator_state,
                        candidate_execution=self.candidate_execution)
                    try:
                        response = evaluator.evaluate(self.candidates[evaluated_key], EvaluationRequest(
                            tuple(case.case_id for case in evaluator.cases)))
                        # Shared AgentEvaluator verifies the batch internally and
                        # deliberately redacts per-Case rows when truth is disabled.
                        error = float(response.primary_error)
                    finally:
                        evaluator.cancel_pending_evaluations()
            else:
                raise ValueError("unknown literature evaluation fidelity")
        except (CampaignPaused, AgentInfrastructureError):
            raise
        except Exception as exc:
            if fidelity == "coarse" and self.pause is not None and self.pause.should_pause("literature_coarse", 0):
                raise CampaignPaused(stage="literature_coarse") from exc
            from pde_agent_verifier_solver.engine.common_evaluation import _is_infrastructure_failure
            if _is_infrastructure_failure(exc):
                raise
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            elapsed = max(0.0, self.clock() - start)
            self.state["controller_evaluation_seconds"] += elapsed
            timing = self.state.setdefault("evaluation_timing", {}).setdefault(ident, {"wall_seconds": 0., "cases": []})
            timing["wall_seconds"] += elapsed
            timing["cases"].extend({"case_id": a.case_id, "status": a.status,
                                    "charged_seconds": a.charged_wall_seconds}
                                   for a in getattr(self.candidate_execution, "audits", ())[audit_start:])
        valid = failure is None and math.isfinite(error) and error >= 0
        report = {"status": "ok" if valid else "failed", "error": error if valid else None,
                  "failure": failure, "tree_sha256": evaluated_key, "fidelity": fidelity,
                  "branch": branch, "plan": plan,
                  "controller_evaluation_wall_seconds": timing["wall_seconds"],
                  "case_execution_timing": timing["cases"]}
        reports[ident] = report
        _write_json(self.root / "literature_evaluations.json", reports)
        self.checkpoint()
        return report

    def best_validated(self):
        finite = [key for key, value in self.errors.items() if math.isfinite(value)]
        return min(finite, key=self.errors.get) if finite else None


def run_literature_search(config, **kwargs):
    host = LiteratureHost(config, **kwargs)
    try:
        workflows = {"codepde_refinement_ns_adaptation": run_codepde,
                     "autonumerics_ns_adaptation": run_autonumerics,
                     "autopde_ns_adaptation": run_autopde,
                     "pde_sharp_ns_adaptation": run_pde_sharp}
        selected = workflows[config.method](host, host.options)
    except DevelopmentBudgetExhausted:
        host.record("stop_reason", "total_agent_wall_budget_exhausted")
        selected = host.best_validated()
    if host.environment is not None:
        host.record("case_acquisition", {branch: {"starter_count": len(host.development),
            "fresh_count": len(value["records"]), "requests": value["requests"]}
            for branch, value in host.environment.progress.get("branches", {}).items()})
    host.record("selected_tree_sha256", selected)
    return selected, len(host.turns)
