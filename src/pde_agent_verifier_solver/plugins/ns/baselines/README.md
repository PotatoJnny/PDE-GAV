# NS Agent baselines

**Completion protocol (2026-09-21):** use the [combined four-method execution guide](../../../../../docs/operations/NS_CODEPDE_AUTONUMERICS_20260920/INSTRUCTIONS.md)
and `configs/paper/ns_literature_v2`. Each Agent stage completes naturally; cumulative
online Base acquisition is unlimited (`null`, not a large numeric allowance).
Both arms receive finite development inputs and reference velocities from the same
continuous starter bank. Hidden reference queries remain disabled. CodePDE's three
refinements each read all valid initial programs, excluding S0. AutoNumerics adds
plan deduplication, final measured-result selection and a read-only delivery explanation.
AutoPDE uses one development session plus one fresh local review; the review output is delivered, even if its validation error is larger. PDE-SHARP uses one analysis, four Genesis tasks, three judges and two hybrid rounds; only nominated or hybrid programs compete. All four methods have paired completion_v2 templates.
Managed allocation pauses preserve the same native session, workspace, nominations
and acquired Cases. Per-Case solver limits still apply. The older bounded profiles
below remain available and are not the new experimental protocol.

These are external NS benchmark compositions, including paired finite-input and continuous-online environments. They reuse the NS Task,
CommandAgentProgram, development RPC, candidate subprocesses and
PythonCandidateCodec. They do not add replay to the canonical Controller.

| Method | Implemented generation and selection |
| --- | --- |
| competitive_refinement | Concurrent Agent roles receive the same round incumbent and full previous-round notes, reuse fixed development/validation banks, and compete with all valid nominations. Selection runs after all roles settle. |
| refinement | Repeated Agent turns on a fixed development bank. Each receives the current validation-best bundle and previous notes. Worse proposals do not replace the incumbent. |
| bon_repair | Independent turns receive the same S0 (or no S0) and empty history. Fixed validation chooses a winner; repair receives that winner and its branch history. |
| era_futs_ns_adaptation | The unmodified official ERA FUTS algorithm chooses expansion parents. NS callbacks provide Agent-generated bundles and negative validation error. |
| aide_ns_adaptation | Retained official AIDE draft/improve/debug policy and Journal/Node state choose the generation stage and parent; NS Agent prompts and trusted validation replace the original empirical task integration. |
| codepde_refinement_ns_adaptation | Independent cold samples and failure debugging, followed by frozen multi-program seed refinement. |
| autonumerics_ns_adaptation | Formulator, numerical planning/ranking, Coder and actual plan-parameterized coarse/fine Critic stages with bounded fresh restarts. |
| autopde_ns_adaptation | Explicit diagnosis/method cards, independent NS guides and autonomous pilot tuning, followed by fresh local-only review; repeated two-pass attempts are a declared budget adaptation. |
| pde_sharp_ns_adaptation | Five mathematical analyses, unranked Genesis candidates, full-pool judge selection, common nominee feedback and incremental hybrid proposals. |
| codepde_repeated_sample_ns_adaptation | Independently implemented source-audited CodePDE repeated sampling: cold-start samples, debug only execution/nonfinite failures, stop a sample at its first finite score. |

Solver-writing turns nominate one final bundle, except configurable competitive_refinement portfolios. Literature planning and judge-selection stages return JSON in their round log and nominate no solver.
Competitive refinement accepts a portfolio, matching the main framework.
Empty/malformed nominations spend the turn
and remain in the failure trace. Timed-out turns may retain a successfully
committed nomination, matching the existing execution service. BoN's N counts
independent Agent turns, not internal LLM calls or development experiments.
Match refinement/ERA iterations to independent_candidates + repair_iterations
when matching planned Agent hard-wall time.

Development and validation banks must be disjoint. Agents see development
Cases and the configured development truth permissions. Reused validation has
frozen Task-owned private points; its scores can affect search and are not an
unbiased test estimate. CodePDE selection minimizes mean primary error, with ties retaining the earlier candidate.
AutoNumerics completion_v2 uses its final Selector over successful fine-evaluated plans. The API/CLI accepts no terminal bank.
Exported bundles and S0 enter the separate common NS evaluator together.

## Run a command Agent

Use an original numerical NS ns_d0_v1 manifest with disjoint development and
validation splits. The shared load_numerical_split helper reads only those
splits and does not start a Generator.

Save this configuration, replacing the absolute data/credential paths and model.
Use the same backbone, reasoning setting, feedback permissions and execution
budgets as the comparison arm. Example counts below are illustrative, not the
paper's prescribed experimental sizes.

~~~json
{
  "schema": "ns_fixed_data_baseline_v1",
  "method": "refinement",
  "seed": 17,
  "dataset": {
    "manifest": "/absolute/path/ns_split_manifest.json",
    "development_split": "train",
    "selection_split": "validation",
    "development_case_ids": ["development-case-1", "development-case-2"],
    "selection_case_ids": ["validation-case-1", "validation-case-2"]
  },
  "initial_candidate": "/absolute/path/shared_s0_bundle",
  "search": {"iterations": 4},
  "agent": {
    "command": [
      "{python}", "{repo}/scripts/run_canonical_codex_agent.py",
      "--codex-binary", "/absolute/path/codex",
      "--auth-file", "/absolute/path/auth.json",
      "--runtime-root", "{output}/agent_runtime",
      "--config", "{repo}/scripts/canonical_codex.config.toml"
    ],
    "hard_wall_seconds": 300,
    "worker_settings": {
      "profile": "structured_numerical",
      "model": "REPLACE_WITH_SAME_MAIN_ARM_MODEL",
      "reasoning_effort": "high",
      "candidate_abi": "prepare_solution_then_evaluate_points_v1"
    }
  },
  "evaluation": {
    "allow_point_truth": false,
    "allow_grid_truth": false,
    "development_coordinate_mode": "private_random_points",
    "development_point_count": 32,
    "selection_point_count": 96,
    "candidate_case_workers": 1,
    "candidate_case_hard_wall_seconds": 60,
    "development_batch_hard_wall_seconds": 300
  }
}
~~~

Omit case_ids to use the entire chosen split. Omit initial_candidate for cold
start. For four BoN+repair turns set method=bon_repair and
search={"independent_candidates": 3, "repair_iterations": 1}. For official FUTS
set method=era_futs_ns_adaptation and search={"iterations": 4, "era_c_puct": 1.0}.
For AIDE set method=aide_ns_adaptation and search={"iterations": 10}; the
author defaults are aide_num_drafts=5, aide_debug_prob=0.5 and
aide_max_debug_depth=3. These can be explicitly changed in search. A supplied
S0 counts as an existing draft; iterations counts only new Agent turns. A small
turn budget may finish during the initial draft stage, which must be reported.

For CodePDE set method=codepde_repeated_sample_ns_adaptation, omit
initial_candidate, and use search={"codepde_num_repeated_samples": 2,
"codepde_num_debugging_trials_per_sample": 5}. The trial bound includes initial
generation. Ten turns is an upper bound: two immediately valid samples spend
only two turns. S0 is rejected before Agent execution; use this adapter in a
shared cold-start comparison, or explicitly report different initial information.
Planned bounds and observed turn/token/execution costs must both be reported.

~~~bash
.venv/bin/python scripts/run_ns_baseline.py --config /absolute/path/ns_baseline.json --output /absolute/path/new_baseline_run
~~~

Arguments execute directly without a shell. The CLI expands {python}, {repo}
and {output}. The output directory must be new/empty. Execution uses existing
bounded same-user subprocesses. The original-numerical public bundle reuses the
same NS operator code, describes actual question/case_parameters/arrays fields,
and declares no unsupported continuous oracle or completed-solution sessions.
The generated-source full environment has additional sampling/oracle tools;
this fixed-data control does not claim an identical tool set. Match applicable
development feedback permissions and report the environment difference.

The CLI supports private_random_points and agent_selected_query.
user_fixed_query is rejected because the CLI has no explicit coordinate field;
use the Python API with a validated AnswerQuery for that protocol.

The default CLI handles original numerical Cases. A structured resident-oracle
or PIV composition should call run_ns_baseline with its existing plugin
objects, public problem bundle, Generator session binding and task settings,
rather than mislabelling the numerical-data adapter.

Local verification needs neither a paid model nor GPU:

~~~bash
.venv/bin/python -m pytest tests/test_ns_baselines.py -q
~~~

The integration test creates an original-NS manifest and deterministic command
worker, exercises development RPC and candidate subprocesses, and exports a
standard bundle. It validates interfaces, not paper performance.

## Fixed-data two-Agent competition

For the paper's fixed-data iteration control, use method=competitive_refinement
and search={"competition_rounds": 5}. The current NS BGP v3 paper configuration
reuses 512 numerical development Cases and a separate frozen 1,024-Case hidden
selection bank (342 Base, 341 Guided, 341 Base-perturbed) across all five rounds.
Each Agent sees the entire development bank. It cannot request new Cases.

The CLI sets agent.inline_array_bytes=256 by default. Larger public arrays use
the existing read-only NPY sidecars in each Agent workspace, so a 512-Case bank
does not exceed the unchanged 8 MiB worker-request limit. This changes transport
only; Agents still receive every public array through the same capability API.
The value can be explicitly configured as a positive integer in the agent block.

The agent block accepts roles in addition to the shared command/settings:

~~~json
{
  "hard_wall_seconds": 14400,
  "roles": [
    {"agent_id": "ns-broad-explorer", "name": "Broad NS solver explorer",
     "prompt": "Explore distinct numerical methods using the supplied fixed development bank.",
     "gpu_devices": ["0"]},
    {"agent_id": "ns-critical-refiner", "name": "Critical NS solver refiner",
     "prompt": "Diagnose the incumbent and improve accuracy using the supplied fixed development bank.",
     "gpu_devices": ["1"]}
  ]
}
~~~

Use scheduler-visible GPU identifiers and reserve the corresponding GPUs in the
job. The existing Agent batch service rejects overlapping assignments. Omit
GPU devices for a CPU-only interface check. Both roles use the common command,
model, reasoning setting and environment; role prompts should reflect fixed
numerical-array Cases rather than advertise unavailable continuous oracles.
The supplied S0 must support those Cases as well as the comparison arm's Cases.

Each role has a 14,400-second (four-hour) hard wall per round, including tool
calls and all waiting. Five rounds with two roles allow 40 Agent-hours; there
is no separate active-time stopping rule in the paper configuration. Match the
main arm's evaluation settings: 28 shared candidate_case_workers,
900 candidate_case_hard_wall_seconds, and 6,900 development_batch_hard_wall_seconds.
Set agent.worker_settings.controller_rpc_timeout_seconds=7200 so Agent RPC does
not cut off these batches. The candidate's 900-second solve budget excludes
recorded trusted Oracle queue waiting; Oracle computation remains charged.

### Borrow idle Agent GPUs for hidden selection

Set `evaluation.borrow_agent_gpus=true` in future Fixed configurations. The
launcher must keep the allocated GPUs in `CUDA_VISIBLE_DEVICES`; each role's
`gpu_devices` lists its physical allocation tokens. The CLI maps these tokens
into the existing visible order, without adding or reordering GPUs. A missing
or inconsistent allocation raises an error instead of silently falling back
to CPU. Omit this option or use `false` for legacy CPU-only evaluation.

After Agent processes and capability calls are cleaned up, the existing
execution-service lease makes both GPUs available to the public initial-value
and forcing Oracles during selection. Candidate subprocesses remain CPU-only.
The hidden reference, frozen query plans, scoring, candidate order and 900-second
budget are unchanged. Cached truth and incremental evaluation progress survive
lease changes and allocation resumes. Cleanup finishes before the next Agent
turn; a cleanup failure quarantines the devices and stops the run.

`baseline_input.json` records the physical-to-logical map, and
`evaluation_device_events.jsonl` records selection borrowing, return or
quarantine. The NS draft dispatcher also passes both allocated GPUs to Fixed's
common terminal evaluation. Existing frozen run sources are not updated.
See the [verification and later common reevaluation plan](../../../../../docs/research/ns_fixed_evaluation_gpu_20260911.md).


Every round freezes both contexts before launching through execute_agent_batch.
It then scores the incumbent and all valid committed nominations on the same
fixed validation Cases and private points. Duplicate program bytes reuse their
frozen validation result. Any exception, missing Case result or nonfinite error
makes that nomination ineligible; scores are not clipped. Strict improvement is
required; incumbent ties are retained and challenger ties follow configured
Agent/nomination order. A timed-out Agent's committed nominations remain eligible.
An isolated Agent failure does not skip the other Agent. The two unchanged Agent
logs and trusted validation results enter the next round's shared memory.

Additional outputs are baseline_status.json, baseline_rounds.json (including
selected bundle paths), candidate_history.json, campaign_memory.json and
CAMPAIGN_MEMORY.md. All outputs remain outside the terminal/test selection path.
This external fixed-data control uses the existing Task, batch execution, memory,
candidate isolation and bundle interfaces; it does not add replay to the canonical
Controller or change the existing single-Agent refinement/BoN/AIDE methods.

## Outputs

- baseline_plan.json: fixed banks, planned search and feedback policy.
- baseline_input.json: CLI configuration and dataset-manifest identities.
- baseline_turns.json: every turn, parent, seed, status and validation error.
- selection_results.json: validation results and failures for each unique
  frozen program. Duplicate proposals spend turns but reuse frozen validation.
- baseline_result.json: selected bundle path/hash and validation error;
  terminal_score is null because external testing has not run.
- result_bundle/: standard frozen PythonCandidateCodec bundle.
- execution_audits.json and candidate_execution_audits.json: existing service
  audits. costs.json uses the shared collector, without inventing token prices
  or substituting zero for unknown usage.
- era_adaptation.json: source pin and declared adaptation boundaries.

Use selected_bundle as a programs[].bundle in an ns_common_evaluation_v1
manifest alongside S0 and competing outputs, after freezing all programs.
Do not feed those external test scores back into baseline search.

## Literature audit, 2026-09-06

**ERA — implemented search adaptation.**
The [official repository](https://github.com/google-research/era) exposes
generate_fn and execute_fn callbacks. The exact
[upstream FUTS source at eb56676ce0a9b22b0c357c218de09ac61e94fc4b](https://github.com/google-research/era/blob/eb56676ce0a9b22b0c357c218de09ac61e94fc4b/implementation/futs.py)
is retained unmodified in _vendor/era_futs.py, with the Apache-2.0 license and
SHA-256 provenance. Node ranking, PUCT, visit backpropagation and search remain
the author's code. Solution objects hold frozen bundle identities rather than
inline Python. NS callbacks replace the paper's model/prompt and empirical
task. This is an ERA FUTS NS adaptation, not an end-to-end paper reproduction.
Failed proposals receive negative infinity; cold-start initialization consumes
one of the planned Agent turns.

**CodePDE — implemented repeated-sampling policy adaptation.**
The [official repository](https://github.com/LithiumDA/CodePDE/tree/26938c5832ddd0a957426dff6f702b54649c14af)
was inspected at commit 26938c5832ddd0a957426dff6f702b54649c14af. The repository
contained no license file at that revision; this adapter independently implements
the observed control policy without vendoring or executing the upstream source.
File identities and the audit are recorded in _vendor/CODEPDE_PROVENANCE.json.

Each sample starts from scratch. Only execution errors or NaN/Inf trigger
another debugging trial; the first finite score ends the sample regardless of
accuracy. Generation/parsing failure aborts that sample and remains in the trace.
Debug gets its own branch's failed bundle/feedback, never another sample's history.
The original per-call model prompts and solver ABI are replaced with one NS
Agent turn and a frozen bundle; internal model calls are not fixed by this policy.
Selecting the best finite sample on fixed validation is an explicit NS addition:
the inspected upstream runner exports a CSV without defining our delivery rule.
CodePDE refine/FunSearch modes and a complete paper reproduction remain outside
this adapter. Outputs codepde_adaptation.json and codepde_trials.json retain
actual early stopping, failures and the declared adaptation boundaries.
If the Agent callback returns no frozen bundle after syntax/ABI/nomination
rejection, that sample ends as generation failure. An unavailable raw draft
cannot be treated as an executable frozen parent for debugging; its failure
remains in baseline_turns.json.

**AIDE — implemented NS search-policy adaptation.**
The pinned [official agent source](https://github.com/WecoAI/aideml/blob/60b3978ddf65b71f86eb7c64506965048a1398cf/aide/agent.py)
retains its draft/improve/debug selection, buggy-leaf and debug-depth rules,
greedy improvement and Journal/Node behavior in _vendor/aide_core.py. Its MIT
license, original sources and extraction hashes are included. Tests compare
retained method bodies against the pinned source AST; serialization-only
mixins and unused provider dependencies are excluded.

NS callbacks replace Kaggle-specific prompts/interpreter and LLM metric parsing
with the existing command Agent, frozen bundles and trusted validation error.
Draft/improve receive the good-node journal; debug receives the buggy parent and
failure feedback. This is not a complete AIDE paper reproduction. A malformed
turn with no frozen nomination consumes a buggy step but has no source bundle
to repair; that limitation is recorded. Search RNG is isolated and seeded.
Outputs aide_adaptation.json and aide_journal.json retain all search decisions;
actual resource reporting uses the shared execution audits, including failed turns.

**PDE-SHARP source availability.**
The author-linked https://github.com/udellgroup/PDE-SHARP returned HTTP 404
at the source audit and again on 2026-09-18. The new `pde_sharp_ns_adaptation`
is explicitly an independent paper/appendix-based implementation; it does not
claim an upstream code revision or exact native benchmark reproduction.

## CodePDE and AutoNumerics paired environments (2026-09-17)

The implemented NS adaptations preserve their distinct method policies in `workflows.py`
and use the existing executor/Evaluator through `literature.py`. `environment.py`
exports/restores the same continuous starter Cases for finite-input and continuous
query arms. It also owns bounded Base acquisition within each solver branch.
Private development feedback coordinates are continuous in both arms; scalar
metrics and terminal evaluation stay on the shared NS implementation.

See the [complete execution instruction](../../../../../docs/operations/NS_CODEPDE_AUTONUMERICS_20260917/INSTRUCTIONS.md),
the four templates under `configs/paper/ns_literature_v1`, and the source pins in
`_vendor/CODEPDE_REFINEMENT_PROVENANCE.json` and `_vendor/AUTONUMERICS_PROVENANCE.json`.
These are explicitly source-audited NS adaptations, not exact reproductions of the
authors' original benchmarks. CodePDE repeated sampling remains a separate method.

Run `scripts/prepare_ns_literature_baselines.py --binding ... --output ...` on the
receiver's v6 checkout, then the existing `scripts/run_ns_baseline.py` with the
generated config. `--segment-seconds 10800 --flush-seconds 300` enables safe phase
boundaries; return code 75 means use the same output and `--resume` next allocation.
Completed method actions, private validation batches, time and acquisition quota
are restored. An uncommitted interrupted Agent call fails closed. No jobs are
submitted automatically.


## Four PDE-Agent environment pairs (2026-09-18)

The current experiment scope is CodePDE, AutoNumerics, AutoPDE and PDE-SHARP,
each with `fixed_sampled` and `continuous_online_base` arms. The new policies
live in `pde_workflows.py` and use the same `LiteratureHost`, Case permissions,
Agent wall-time counters and safe phase-boundary continuation. Judge selection
receives complete candidate sources; Genesis candidates are not pre-ranked by
outer validation. AutoPDE's local reviewer shares its attempt's time/Case quota.

See the [eight-arm execution plan](../../../../../docs/operations/NS_PDE_AGENT_ENVIRONMENT_20260918/INSTRUCTIONS.md)
and `_vendor/AUTOPDE_PROVENANCE.json` / `_vendor/PDE_SHARP_PROVENANCE.json`.
These two are paper-based independent NS adaptations. The provenance explicitly
records unavailable original source/skills, smaller search sizes, repeated
AutoPDE attempts and bundle-format changes.

New bindings declare `methods=["codepde","autonumerics","autopde","pde_sharp"]`.
Omitting `methods` preserves the original four-arm scope. Final export verifies
every declared pair and emits eight selected bundles plus shared S0. The current
eight templates retain the earlier 64/24/96 compatibility protocol; they still
require alignment with HL's latest environment/cache/config and asset binding
before formal submission. No additional Panel control is implemented.
