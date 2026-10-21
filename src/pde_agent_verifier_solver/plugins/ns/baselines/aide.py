"""Pinned AIDE search-policy adaptation using the NS baseline callbacks.

This retains upstream draft/improve/debug selection and Journal/Node behavior.
It is not a reproduction of AIDE's complete paper, Kaggle prompts, interpreter,
LLM metric parser, or model settings. See _vendor/AIDE_PROVENANCE.json.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time
from types import FunctionType, MethodType, SimpleNamespace

from ._vendor import aide_core

AIDE_REVISION = "60b3978ddf65b71f86eb7c64506965048a1398cf"
_CORE_SHA256 = "256c7e3a1db10c2b07a278235fc550f24f1ff15f9bdf0a7324e2c2b8995f20ab"
_MISSING = "__aide_missing_nomination__:"


@dataclass(frozen=True)
class AIDESearchOptions:
    """Defaults are the pinned author's agent.search configuration."""

    num_drafts: int = 5
    debug_prob: float = 0.5
    max_debug_depth: int = 3

    def __post_init__(self):
        for name in ("num_drafts", "max_debug_depth"):
            value = getattr(self, name)
            minimum = 1 if name == "num_drafts" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.debug_prob, bool)
                or not isinstance(self.debug_prob, (float, int))
                or not math.isfinite(self.debug_prob) or not 0 <= self.debug_prob <= 1):
            raise ValueError("debug_prob must be a finite probability in [0, 1]")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _provenance():
    root = Path(aide_core.__file__).parent
    metadata = json.loads((root / "AIDE_PROVENANCE.json").read_text())
    if metadata["revision"] != AIDE_REVISION:
        raise RuntimeError("AIDE provenance revision differs from the pinned revision")
    return metadata


class _NSAgent(aide_core.Agent):
    def __init__(self, config, options, *, propose, score, history_for, diagnostics_for):
        journal = aide_core.Journal(metric_maximize=False)
        super().__init__(config.prompt, SimpleNamespace(agent=SimpleNamespace(search=options)), journal)
        self._propose = propose
        self._score = score
        self._history_for = history_for
        self._diagnostics_for = diagnostics_for
        # Run the exact upstream method with an isolated RNG. This changes no
        # selection expression and cannot reseed another simultaneous run.
        policy = aide_core.Agent.search_policy
        isolated_policy = FunctionType(
            policy.__code__, {**policy.__globals__, "random": random.Random(config.seed)},
            policy.__name__, policy.__defaults__, policy.__closure__,
        )
        self.search_policy = MethodType(isolated_policy, self)

    def update_data_preview(self):
        # Real Cases and allowed reference resources are supplied by the NS
        # AgentContext. There is no Kaggle directory scan or hidden-data preview.
        self.data_preview = "Fixed NS development Cases are available through the AgentContext."

    def _new_node(self, key, *, parent, plan):
        index = len(self.journal)
        return aide_core.Node(
            code=key if key is not None else f"{_MISSING}{index}",
            plan=plan, parent=parent, id=f"ns-aide-node-{index:06d}",
        )

    def _generate(self, parent, stage):
        if stage == "draft":
            instruction = (
                "aide_draft. Create a new independent, simple NS solver design. "
                "Use the good-node journal below to avoid repeating an existing design. "
                "Do not start by tuning or ensembling previous solvers."
            )
        elif stage == "improve":
            instruction = (
                "aide_improve. Make one atomic, actionable improvement to the supplied "
                "parent solver. Use the good-node journal below and its measured "
                "validation results to choose the improvement."
            )
        else:
            instruction = (
                "aide_debug. Diagnose and repair the supplied buggy parent solver, "
                "using its execution feedback. Focus on fixing the failure."
            )
        if stage == "debug":
            # Like the author's _debug, expose the buggy parent and its output,
            # not the complete good-node journal used for draft/improve.
            instruction += f"\nParent execution feedback:\n{parent.term_out}"
            history = ((parent.plan + "\n" + parent.analysis, None),)
        else:
            instruction += "\nGood-node journal:\n" + self.journal.generate_summary()
            history = ()
        parent_key = None if parent is None or parent.code.startswith(_MISSING) else parent.code
        if parent is not None and parent_key is None:
            instruction += (
                "\nThe previous turn produced no valid frozen bundle. There is no "
                "recoverable parent source; correct the nomination failure using the "
                "feedback and submit a valid bundle. This still consumes a debug step."
            )
        key = self._propose(parent_key, history, instruction)
        if key is not None and (not isinstance(key, str) or not key or key.startswith(_MISSING)):
            raise ValueError("AIDE propose callback must return a nonempty bundle key or None")
        notes = self._history_for(key) if key is not None else ()
        plan = str(notes[-1][0]) if notes else (
            "No valid bundle was nominated." if key is None else "No generation notes recorded."
        )
        return self._new_node(key, parent=parent, plan=plan)

    def _draft(self):
        return self._generate(None, "draft")

    def _improve(self, parent_node):
        return self._generate(parent_node, "improve")

    def _debug(self, parent_node):
        return self._generate(parent_node, "debug")

    def execute(self, key, reset_session):
        del reset_session  # Every NS generation callback already has its own Agent turn.
        start = time.perf_counter()
        if key.startswith(_MISSING):
            error = None
            output = "The Agent turn produced no valid frozen Python solver bundle."
        else:
            value = float(self._score(key))
            error = value if math.isfinite(value) else None
            output = (f"Fixed hidden validation mean primary error: {error}." if error is not None
                      else "The solver failed fixed validation or returned a nonfinite error.")
            if self._diagnostics_for is not None:
                diagnostic = self._diagnostics_for(key)
                if diagnostic:
                    output += "\n" + str(diagnostic)
        result = aide_core.ExecutionResult(
            term_out=[output], exec_time=time.perf_counter() - start,
            exc_type=None if error is not None else "NSValidationFailure",
            exc_info=None if error is not None else {"message": output},
        )
        # The trusted NS numerical metric replaces AIDE's LLM metric extraction.
        result.ns_primary_error = error
        return result

    def parse_exec_result(self, node, exec_result):
        node.absorb_exec_result(exec_result)
        node.analysis = node.term_out
        node.is_buggy = exec_result.exc_type is not None or exec_result.ns_primary_error is None
        node.metric = (aide_core.WorstMetricValue() if node.is_buggy
                       else aide_core.MetricValue(exec_result.ns_primary_error, maximize=False))


def _journal_rows(journal):
    return [{
        "node_id": node.id, "step": node.step,
        "bundle_key": None if node.code.startswith(_MISSING) else node.code,
        "parent_node_id": None if node.parent is None else node.parent.id,
        "stage": node.stage_name, "debug_depth": node.debug_depth,
        "is_leaf": node.is_leaf, "is_buggy": node.is_buggy,
        "selection_error": node.metric.value, "plan": node.plan,
        "analysis": node.analysis, "exception_type": node.exc_type,
    } for node in journal.nodes]


def run_official_aide_search(
    config, *, initial_key, propose, score, history_for,
    options: AIDESearchOptions | None = None, diagnostics_for=None,
):
    """Return the best finite fixed-validation bundle after ``iterations`` turns.

    Callback contract matches the NS baseline composition: ``propose(parent_key,
    history, phase)`` runs its usual Agent and returns one frozen bundle key or
    None; ``score(key)`` is cached fixed-validation mean error; ``history_for``
    supplies generation notes. The phase includes the stage instructions and
    upstream-style journal summary. It must reach the generation prompt intact.

    An optional diagnostics callback may return permitted validation failure
    diagnostics, never hidden truth or terminal scores. S0, when supplied, is a
    pre-existing draft node and counts toward num_drafts, but consumes no new
    Agent turn. No terminal evaluator or terminal data is accepted here.

    Failed numerical execution retains the exact parent bundle for debug. A turn
    without a valid nomination remains a buggy node and consumes its budget, but
    has no recoverable source bundle; this limitation is recorded in the report.
    """
    metadata = _provenance()
    options = options or AIDESearchOptions()
    if not isinstance(options, AIDESearchOptions):
        raise TypeError("options must be AIDESearchOptions")
    if (isinstance(config.iterations, bool) or not isinstance(config.iterations, int)
            or config.iterations < 1):
        raise ValueError("iterations must be an integer >= 1")
    root = Path(config.output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    agent = _NSAgent(config, options, propose=propose, score=score,
                     history_for=history_for, diagnostics_for=diagnostics_for)
    _write_json(root / "aide_adaptation.json", {
        "schema": "aide_ns_search_policy_adaptation_v1",
        "upstream_repository": metadata["repository"], "upstream_revision": AIDE_REVISION,
        "upstream_sources": metadata["upstream_sources"],
        "local_core_sha256": _CORE_SHA256,
        "local_adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "retained_upstream_methods": ["Agent.search_policy", "Agent.step", "Journal.*", "Node.*"],
        "search_policy_method_modified": False,
        "search_rng": "isolated_random_Random_seeded_by_baseline_seed",
        "options": asdict(options), "agent_turn_budget": config.iterations,
        "s0_counts_as_draft": initial_key is not None,
        "solution_representation": "frozen_python_bundle_tree_sha256",
        "generation": "same_project_NS_agent_with_stage_specific_instructions_and_journal",
        "metric_parser": "trusted_fixed_hidden_validation_mean_primary_error_minimize",
        "no_nomination_behavior": "buggy_node_without_recoverable_parent_source",
        "terminal_selection": False, "paper_end_to_end_reproduction": False,
    })
    try:
        if initial_key is not None:
            if not isinstance(initial_key, str) or not initial_key or initial_key.startswith(_MISSING):
                raise ValueError("initial_key must be a valid frozen bundle key")
            initial = agent._new_node(initial_key, parent=None, plan="Supplied initial solver S0.")
            agent.parse_exec_result(initial, agent.execute(initial_key, True))
            agent.journal.append(initial)
        for _ in range(config.iterations):
            # This is the retained author's step, including policy choice,
            # stage dispatch, execution parsing, and journal insertion order.
            agent.step(agent.execute)
            _write_json(root / "aide_journal.json", _journal_rows(agent.journal))
        best = agent.journal.get_best_node()
        return None if best is None else best.code
    finally:
        _write_json(root / "aide_journal.json", _journal_rows(agent.journal))
