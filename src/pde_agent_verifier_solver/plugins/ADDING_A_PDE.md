# Adding a geometry, Task, or PDE

The goal is a complete user flow: create a project, configure a run, inspect each round, and receive a callable scientific program. Choose a route below, complete a small run, and consult the detailed interface reference as needed.

| Requirement | Route | Minimum deliverable |
| --- | --- | --- |
| A new NS geometry with the same velocity-field output | [A: New NS geometry](#ns-geometry) | Consistent geometry, boundaries, Generator/cases, and run configuration |
| A new program objective for an existing PDE | [B: New Task](#new-task) | Task, evaluator, public instructions, and Task registration |
| A new PDE | [C: New PDE](#new-pde) | Problem, Generator, public Oracle, private reference, Task, evaluator, and plugin |

First run `python examples/toy_campaign.py` from the repository root.
[This complete example](../../../examples/toy_campaign.py) demonstrates `ProgramBundle → Project → CampaignRecipe → Run → Result` with local deterministic agents; it does not call a research model.

<a id="ns-geometry"></a>
## A: New NS geometry

If inputs remain NS initial/boundary conditions and forcing, and the output remains a velocity field, reuse [NSSolverTask](ns/tasks/solver.py). Create a project and experiment configuration for the new geometry. The shared Controller remains reusable.

| Step | What to write and where | Completion criterion |
| --- | --- | --- |
| 1. Describe the geometry | Domain bounds, obstacle vertices, physical coordinates, and boundary conditions; see [ContinuousGeometrySpec](../../../legacy/ns/src/agent_verifier/neural_generator/continuous_geometry.py) | The public Case fully describes the solver's domain |
| 2. Bind the continuous problem | Pass the geometry through the existing `geometry_spec` argument in [structured_flow_runtime.py](ns/structured_flow_runtime.py); bind the matching v6 Generator assets through the run configuration (see [site configuration](../../../docs/SITE_CONFIGURATION.md)) | Initial conditions, forcing, boundary values, and hidden references use the same geometry |
| 3. Connect the launcher | [canonical_backend.py](ns/canonical_backend.py) forwards `geometry_spec`; the current [NS runner](../../../scripts/run_canonical_ns_campaign.py) does not yet read it from configuration in `_compose_numerical`. Add that argument when connecting a new geometry | Changing the configured geometry changes generated cases, beyond their dataset labels |
| 4. Prepare data and instructions | Use a manifest if a data adapter already exists; extend [datasets.py](ns/datasets.py) for new formats. Document boundaries and coordinates in task/resource Markdown | Agent inputs, public Oracle queries, and scoring locations follow the new setting |
| 5. Verify and run | Check queries, boundaries, reference errors, and one or two rounds on a few cases | The panel previews the setting and returns a callable program with per-case scores |

For example, a square obstacle in a rectangular domain can be passed to the backend as this `geometry_spec`:

```json
{
  "abi_version": "continuous_geometry_convex_polygons_v2",
  "domain_bounds": [0.0, 2.0, 0.0, 1.0],
  "obstacle_polygons": [
    [[0.8, 0.4], [1.0, 0.4], [1.0, 0.6], [0.8, 0.6]]
  ],
  "outer_boundary": "no_slip"
}
```

Coordinates are physical coordinates. The order of `domain_bounds` is `x_min, x_max, y_min, y_max`. This is a geometry object, not a complete experiment configuration; a run also needs the data, v6 Generator, and launcher binding described above. Geometry format `v2` and continuous MMS construction `v6` are separate version fields.

The current implementation supports rectangular 2D domains with convex polygonal obstacles. Outer boundaries support no-slip conditions or a manufactured velocity trace. Curved boundaries, 3D domains, or other boundary types require corresponding NS geometry, continuous-field, and query implementations, together with suitable scoring-point sampling.

Mapping an existing coefficient distribution to a new geometry can define a new synthetic setting. To match data from a new target setting, examine the relationship between generated fields and those data before deciding whether to refit or retrain. Use new data and case-bank configurations when geometry changes.

<a id="new-task"></a>
## B: New Task for an existing PDE

A Task defines what the final program must do. An NS solver and a drag predictor may share the equation while differing in outputs, reference quantities, and metrics.

1. Add a task file under `plugins/<pde>/tasks/`. Follow [NS solver](ns/tasks/solver.py) or [VPFP Tasks](vpfp/tasks/) and implement the interfaces below.
2. If the objective needs an additional reference quantity, such as pressure-derived force, make it available through the PDE's private reference source. Existing reference fields may not provide it yet.
3. Register the Task instance in `PDEPlugin.tasks` in the PDE's `product.py`. Preserve existing Task IDs and assign a new ID to the new objective. [VPFP product.py](vpfp/product.py) demonstrates three Tasks.
4. Select it with `ProgramBundle.from_plugin(plugin, agents, task_id="my_task")`. Update any Task-specific factory branches as well.
5. Update task Markdown, public input/output instructions, and result cards, then complete [scoring and flow verification](#verify-integration).

| What the Task author supplies | Interface or location | What it defines |
| --- | --- | --- |
| Task identity and configuration | `info`, `kind`, `setup_fields()` | The delivered program type and parameters users configure |
| Agent instructions | `agent_brief(settings)`, task Markdown | Inputs, output fields, coordinates, units, and calling convention; supporting material belongs in resource Markdown |
| Evaluation plans | `create_evaluation()` returns an `EvaluationProgram`; `make_plan()` | Evaluation locations, times, and quantities; private development points use `make_private_development_plan()` or the corresponding helper |
| Candidate execution | Formal isolated execution uses `execute_candidate()`, optionally `execute_candidate_batch()` | Solve using public inputs in the candidate process and return predictions |
| Scientific scoring | `score_candidate()` and the evaluator's `aggregate()` | Compute per-case errors against private references in the Controller, then aggregate; reuse the [evaluation helper](_evaluation.py) where appropriate |
| Presentation | `result_card()` | Metrics, usage instructions, and results shown in the panel |

The [EvaluationProgram SDK](../sdk/evaluation.py) defines `make_plan/evaluate/aggregate`. Formal subprocess execution additionally needs separate execution and scoring implementations; see `NSSolverTask.create_evaluation()`. Supply an `AnswerQuery` matching the requested fields in the evaluation settings. Toy's in-process evaluator illustrates the flow; use the NS split implementation as the reference for formal candidate execution.

Promotion chooses the retention metric and selected program through the run configuration. A new Task does not need another selection loop. If the Task also changes the public problem or data representation, update the Problem, Generator, and public resources together and assemble the corresponding plugin.

<a id="new-pde"></a>
## C: New PDE

Start with a few small cases and connect the following components. A Generator may use an analytic construction or a simple sampler; a trained neural network is optional.

```text
src/pde_agent_verifier_solver/plugins/my_pde/
├── problem.py           # Data, public Cases, configuration forms, and previews
├── generator.py         # Train/load, sampling, optional G/P, and resource release
├── public_problem.py    # Public-input Oracle and scientific resources for agents
├── tasks/solver.py      # Program interface, plans, execution, metrics, and results
├── product.py           # Assemble PDEPlugin
└── _public_resources/   # Equation, coordinates, input/output, and tool instructions
```

These files describe responsibilities; small plugins need not create empty modules to match this layout. [toy.py](toy.py) combines several responsibilities in one file. See [ns/](ns/) for continuous scientific problems.

| Component | What to implement | Starting point |
| --- | --- | --- |
| Problem | `prepare_dataset()` returns a `PreparedDataset`; each `CaseRecord` separates the public Case from its private reference. Supply `setup_fields/case_card` | [ProblemProgram / TruthProgram](../sdk/programs.py), [NSProblem](ns/problem.py) |
| References and public inputs | Private `TruthProgram.answer(query)` supplies evaluation references; Case session methods expose initial conditions, forcing, or other public inputs | [Public-resource SDK](../sdk/public_problem.py), [NS public_problem](ns/public_problem.py), [continuous_backend](ns/continuous_backend.py) |
| Generator | `train/load/sample`, capability declarations, and the required `update_guidance`, perturbation, and `open_case_session/case_session_methods/release/preview` behavior | [GeneratorProgram](../sdk/programs.py), [Toy Generator](toy.py) |
| Task and evaluation | The interfaces in route B; begin with one explicit output objective | [NS Task](ns/tasks/solver.py), [evaluation SDK](../sdk/evaluation.py) |
| Plugin assembly | Combine the Problem, Generator, Tasks, default Task, and public resources in `PDEPlugin` | [NS product.py](ns/product.py), [registry.py](registry.py) |

Add the plugin factory to `built_in_plugins()` in `registry.py` so `get_plugin()` can discover it. The application still selects the plugin, data, and agents explicitly, as shown below.

Declare capabilities that are actually implemented. Base sampling can work independently. Guided sampling needs feedback updates and directed generation; Perturbed cases need admissible transformations with matching inputs and references. Each PDE supplies its own mathematics and GPU kernels. Shared scheduling, cancellation, timing, and recovery are reusable.

Agents receive public Markdown through `PublicProblemBundle`. When loading resources from a package directory, include those files in `pyproject.toml` package-data so they remain available after installation.

<a id="connect-product"></a>
## Connect Projects, Runs, and the panel

The following is an **assembly fragment**. The experiment supplies `plugin`, `agents`, `dataset`, `recipe`, and both execution services. [examples/toy_campaign.py](../../../examples/toy_campaign.py) is a complete local example.

```python
from pde_agent_verifier_solver.app import ProductApplication, ProgramBundle

programs = ProgramBundle.from_plugin(plugin, agents, task_id="solver")
app = ProductApplication(
    agent_execution=agent_execution,
    candidate_execution=candidate_execution,
)
project = app.create_project("My PDE project", dataset, programs)
run = app.launch_run(project.project_id, recipe)
result = app.wait(run.run_id).result
```

Formal runs inject agent and candidate execution services; see [app/service.py](../app/service.py). Local deterministic tests may use `ProductApplication.local_development()`. Both use `CampaignController`.

Configure rounds, agents, full wall-clock budgets, the Generator, development acquisition, evaluation rules, Promotion, and terminal cases in `CampaignRecipe`. Put Task-specific evaluation settings in `EvaluatorPolicy.settings`. Compose task and resource instructions with the [Markdown prompt modules](../../../docs/MODULE_INTEGRATION.md).

To create and run your projects through the panel, inject the same `programs` and `app` into its launcher:

```python
from pde_agent_verifier_solver.panel.controller import PanelController
from pde_agent_verifier_solver.panel.server import run_panel

controller = PanelController(app, programs)
run_panel(controller=controller)
```

Bind an existing `PreparedDataset` and Generator checkpoint through `prepared_dataset=` and `generator_checkpoint=`. Pass `default_query=` if needed. The plugin prepares these objects. `setup_fields()` supplies forms, while `case_card()` and `result_card()` supply scientific presentation. Standard forms and cards reuse the shared panel.

**The current CLI command `python -m pde_agent_verifier_solver panel` defaults to Toy.** Registering a plugin does not switch that entry point or install data, weights, or model credentials. Supply the assembly above for your experiment. A new type of visualization may also require a general renderer extension.

<a id="verify-integration"></a>
## Minimum verification and completion criteria

From the repository root, run the existing product and evaluation examples:

```bash
python examples/toy_campaign.py
python -m pytest tests/test_toy_product_flow.py tests/test_evaluator_contract.py
```

Then verify the new science and interfaces:

| Check | Expected result | Reference test |
| --- | --- | --- |
| One small Case | Public inputs and references agree on fields, shapes, coordinates, and units | [test_reference_pde_programs.py](../../../tests/test_reference_pde_programs.py) |
| Correct and incorrect programs | Expected errors and ordering; actionable failure diagnostics | [test_evaluator_contract.py](../../../tests/test_evaluator_contract.py) |
| Formal execution interface | Candidates receive public inputs; the Controller scores returned predictions. If batch solving is supported, compare batch and single-case results | [test_canonical_candidate_boundary.py](../../../tests/test_canonical_candidate_boundary.py) |
| New geometry | Boundaries, initial conditions, forcing, and references agree; scoring points lie in the declared domain | [test_ns_geometry_v6_main_interfaces.py](../../../tests/test_ns_geometry_v6_main_interfaces.py) |
| A two-round run and panel | Agents acquire, query, evaluate, and submit; the next round receives the program and logs; the final program is displayed and callable | [test_toy_product_flow.py](../../../tests/test_toy_product_flow.py) |

Use the intended formal execution services and GPU layout for the first short run with real agents. Passing the Toy example does not establish the numerical correctness of a new PDE, and plugin registration alone does not complete solving or scoring integration.

The remaining sections provide detailed references for scientific interfaces, reconstruction, perturbations, and historical adapters.

## Preferred integration: one CampaignController pipeline

New PDEs use the existing Controller, selection, and interaction protocol. Implement
`ProblemProgram`, a `GeneratorProgram` supporting fresh online sampling, and one or
more `TaskProgram` implementations. Each Task returns its own `EvaluationProgram`
from `create_evaluation()`. Assemble these through `PDEPlugin` and `ProgramBundle`,
bind data and execution services, and use the same product flow:

```text
Project creation
  -> DatasetInput + ProblemProgram
  -> optional ReconstructionProgram for experimental data
  -> frozen PreparedDataset + ReconstructionReport

Panel / CampaignRecipe
  -> CampaignCompiler
  -> CampaignController
  -> RoundCaseSampler + AgentEvaluator
  -> one fresh shared Official cohort
  -> Promotion + Campaign Memory Log
  -> user-declared Terminal cohorts over S_1..S_R
  -> deliver S_R
```

- `agent_adaptive` is the default path. The Generator must declare and implement online sampling.
- `fixed_recipe` is an explicit ablation. Offline sampling and cross-round fixed-case replay are outside the default path.
- Clean cases are original, base, or guided. A perturbed child uses a clean base/guided root acquired by that agent in the current round. The Generator owns the physical constraints on perturbations.
- `EvaluationProgram` defines finite queries, per-case scientific errors, and aggregation. Development feedback includes aggregate and per-case scalar errors/metrics. Finite prediction/truth views additionally require an enabled truth policy and an explicit agent request.
- Development coordinates default to `private_random_points`: the Task uses the evaluation seed to sample the configured number of points from the public physical domain. The agent cannot supply a query in this mode. Users may select a public but non-overridable `user_fixed_query`, or `agent_selected_query`. Coordinate selection and truth disclosure are independent permissions.
- Agents may submit multiple candidates. The configured nomination policy determines which enter selection. New NS templates use `single_primary_v1`: each agent explicitly selects one primary candidate, and ordinary submissions do not replace it. Eligible candidates and the incumbent share one fresh Official cohort for the round.
- The selected program and complete Campaign Memory Log carry across rounds; development cases do not carry over in this path.

### Cross-round logs are shared engine behavior

PDE plugins do not implement another memory system. Formal subprocess agents use
`canonical-agent-worker-request-v2`, with three separate records:

- `campaign_memory` points to a Controller-owned `CAMPAIGN_MEMORY.md` snapshot, also available through `PDE_AGENT_CAMPAIGN_MEMORY`.
- `candidate_history` points to `CANDIDATE_HISTORY.json`, also available through `PDE_AGENT_CANDIDATE_HISTORY`. It stores parent/artifact links, first Official evaluation, and selection lineage, rather than agent observations or scientific notes.
- `round_log` describes the writable `ROUND_LOG.md` and its size limit. The corresponding environment variables are `PDE_AGENT_ROUND_LOG` and `PDE_AGENT_ROUND_LOG_MAX_BYTES`.

The Controller sets `PDE_AGENT_PROTOCOL_MAX_BYTES` to the execution service's
request limit. `load_agent_worker_request()` uses it rather than assuming a fixed
8 MiB limit.

Both input sidecars are marked `0400` and must remain unchanged by the worker.
The subprocess and Controller currently use the same OS user, so these permissions
are not a read-only mount protecting against hostile code. The Controller does
not read back these workspace copies; editing them cannot alter the authoritative
snapshot used by the next round.

External agents **must use** `write_agent_round_log()` for atomic checkpoints,
rather than overwriting or appending to `ROUND_LOG.md` directly. Each call supplies
the complete Markdown log so far. Save after meaningful experiments or candidates
and before normal exit. The helper uses a temporary file, fsync, and atomic
replacement; after a timeout or failure, the Controller retains the last complete
checkpoint.

A minimal worker example:

```python
import os
from pathlib import Path

from pde_agent_verifier_solver.engine.agent_execution import (
    load_agent_worker_request,
    write_agent_round_log,
)

request = load_agent_worker_request()
previous_memory = Path(
    os.environ["PDE_AGENT_CAMPAIGN_MEMORY"]
).read_text(encoding="utf-8")

round_log = "# Round log\n\n## Experiments\n\nStarted baseline."
write_agent_round_log(round_log)

# ... run one meaningful experiment ...
round_log += "\n\n- Baseline failed near the boundary."
write_agent_round_log(round_log)
```

To migrate a v1 worker:

1. Use schema `canonical-agent-worker-request-v2`.
2. Read `CAMPAIGN_MEMORY.md` instead of the embedded `request["memory"]`.
3. Read candidate lineage from `CANDIDATE_HISTORY.json` at `request["candidate_history"]["path"]`.
4. Checkpoint `ROUND_LOG.md` throughout the turn as shown above.

`load_agent_worker_request()` rejects old v1 requests. Nomination and worker
response schemas remain v1. Trusted in-process agents return the complete
Markdown through `AgentOutcome.round_log` instead of using these workspace files.

The Controller preserves each agent's log after all agents finish. After shared
Official evaluation and Promotion, it appends measured scores, the selected
program, and the reason for selection. The next round receives the complete
Campaign Memory Log from all previous rounds. This path has no Memory Tree,
separate Curator, or additional summarization agent.

PDE authors can describe useful domain-specific observations in `agent_brief`.
They do not need another memory class or a copy of Candidate History inside the
scientific log.

`MacroClosedLoopRunner`, historical uses of `MacroMatcher`, the old
selection/evidence cascade, and staged campaign adapters remain for reproducing
historical formal experiments and control arms. They are not integration templates
for new PDEs. `AgentExecutionService` already provides subprocess execution, GPU
allocation, timeouts, and recovery. Formal `ProductApplication` instances require
both agent and candidate execution outside the Controller process. Only trusted
Toy/tests use cooperative execution through `local_development()`. These process
boundaries do not provide hostile filesystem isolation; subprocesses use the same
Unix user as the Controller by default.

PDE authors do not implement `AgentExecutionService`; it is the engine's shared
agent-turn executor. PDE-specific sampling, numerical Oracles, resident references
and state, and resource release belong in the Generator or Case session. Agent
process isolation, GPU worker allocation, wall-clock enforcement, and retry/recovery
belong in the shared execution adapter.

## Reference files to read first

A PDE plugin usually contains these components:

```text
plugins/my_pde/
├── problem.py          # Data loading, public Cases, and reference queries
├── reconstruction.py   # Optional one-time reconstruction of experimental data
├── generator.py        # Distribution training, guidance, and fresh sampling
├── tasks/
│   ├── solver.py       # Optional full-field solver Task
│   ├── observable.py   # Optional observable-model Task
│   └── ...             # One Task per objective
├── product.py          # Assemble Problem, Generator, and Tasks into a plugin
├── campaign.py         # Optional historical staged/control-arm reproduction
└── publish.py          # Optional preparation of downloadable results
```

Suggested reading order:

1. `ns/problem.py`: a numerical field dataset example.
2. `ns/tasks/solver.py`: a full-field solver Task.
3. `vpfp/problem.py`: a phase-space problem exposing finite moments as reference quantities to the Task.
4. `vpfp/tasks/`: three objectives for the same PDE.
5. Each plugin's `product.py`: final plugin assembly.

Long `campaign.py` files preserve cluster stages from historical paper experiments.
Start a new PDE with the interfaces above. Connect formal subprocess/GPU execution,
timeouts, and recovery to the canonical `AgentExecutionService`.

## 1. Problem: turn data into Cases

`ProblemProgram` answers four questions:

- Which PDE and data settings does the user provide?
- How is the initial numerical or experimental dataset read?
- Which parts of a Case can the agent see?
- How does the evaluator obtain a reference on a finite grid or set of points?

The core methods are:

```python
class MyProblem:
    info = ProgramInfo("my-pde", "My PDE", "A description for researchers")

    def setup_fields(self):
        # Return the PDE-specific fields shown in the panel.
        return (...)

    def prepare_dataset(self, dataset: DatasetInput) -> PreparedDataset:
        # Load data and build development_cases and hidden_cases separately.
        return PreparedDataset(...)

    def case_card(self, case: Case) -> PanelCard:
        # Describe how to preview a Case in the panel.
        return PanelCard(...)
```

Each internal record is a `CaseRecord(case, truth)`:

- `case` is the complete agent-visible problem statement. It may contain geometry, equation parameters, initial/boundary conditions, coarse solutions, or noisy observations.
- `truth.answer(query)` returns numerical reference values at the requested grid or points to the evaluator only.

Keep paths to private answer data out of `case.public_data`. NS `NSFieldTruth` and
VPFP `VPFPMacroTruth` demonstrate how Controller-owned arrays become an `AnswerView`.

### Numerical datasets

Load an existing coarse grid solution directly when appropriate. If scoring needs
more accurate references, compute a higher-accuracy solution and use it as the
`TruthProgram` data source. Coarse solutions may remain public observations or
Generator training data.

### Experimental datasets

`DatasetKind.EXPERIMENTAL` uses the same entry point. Put geometry, experimental
conditions, observation coordinates, noisy measurements, and noise descriptions
in the Case. Held-out observations may supply the reference. The Task's evaluator
defines observation error, physics residuals, or a combination.

When reconstruction is selected, the Problem additionally implements:

```python
def prepare_reconstruction_source(dataset, settings, seed):
    return PreparedReconstructionSource(
        program_input=ReconstructionInput(...),
        reconstruction_validation_reference=private_reference,
    )

def prepare_reconstructed_dataset(dataset, output):
    return PreparedDataset(...)

def evaluate_reconstruction(output, reconstruction_validation_reference, settings):
    return ReconstructionEvaluation(...)
```

Partition data by source-frame/data identity into Generator fit, Generator
validation, reconstruction validation, Development, Official, and Terminal roles.
Calibration frames, reconstruction target frames, and downstream frames must be
disjoint. Within a target frame, method-visible and reconstruction-validation
coordinates must also be disjoint. The framework passes only `program_input` and a
nonnegative seed to `ReconstructionProgram`. It calls
`prepare_reconstructed_dataset(dataset, output)` before giving the private
reference to trusted Problem evaluation. The materializer receives dataset/output,
not evaluation/report. This is a trusted-plugin contract, not hostile isolation.
Reconstruction-validation references cannot become downstream original/hidden truth.

`ReconstructionInput.public_split_id` must be a short publishable alias. Keep
private split manifests and paths in Problem-owned `data_settings`. Private paths
and arrays must stay out of metadata, program diagnostics/warnings, and public
reports. Reports also contain per-target public aliases and warnings supplied by
the Problem. JSON compatibility establishes serializability only; the trusted
plugin must remove private information. The completed field is `generator_target`;
the Case's trusted reference remains `CaseRecord.truth`. Built-in POD supports
real-valued data with a fixed basis. Plugins supply methods for complex/Hermitian
data.

### Independent macro matching

To compare candidate predictions with macroscopic projections of a manufactured
or reference answer, implement a PDE-specific `MacroMatcher`. Dataset observations
are not automatically the target, and matching does not require a separate Task:

```python
class MyMacroMatcher:
    contract_id = "my_macro_contract_v1"
    fields = ("density", "momentum", "energy")

    def prepare_target(self, case, truth, query):
        return MacroTarget(...)

    def match(self, case, prediction, target):
        return MacroMatchResult(...)
```

The Task's `EvaluationProgram` calls the matcher in the canonical path, using the
shared Controller. `PreparedMacroCohort`, `ControllerMacroEvaluator`, and
`evaluate_macro_prediction(...)` remain for legacy macro/history experiments.
If the Generator also uses matching to admit generated cases, its backend defines
thresholds, resampling, and private diagnostics. Keep `MacroTarget` and private
failure details out of `case.public_data`.

`PDEPlugin.macro_matcher` is an optional discovery field. Toy/NS plugins without
macro matching leave it as `None`. Matchers must be stateless: cached plugin
instances cannot retain cases, truth, or targets. The VPFP reference is
`vpfp/macro_matching.py`.

An Observable Model is a candidate output interface; a data observation is a
training or public input. Observable and Reduced Dynamics Tasks may share a
matcher.

## 2. Task: define the delivered program

Tasks are separate from equations. One `MyProblem` may support solver, observable,
reduced-dynamics, and statistics Tasks.

Each Task defines the candidate interface, agent brief, and result display, then
creates its own evaluator:

```python
class MyTask:
    info = ProgramInfo(...)
    kind = TaskKind.SOLVER  # Or another supported Task kind.

    def setup_fields(self): ...
    def agent_brief(self, settings): ...

    def create_evaluation(self, settings):
        return MyEvaluation(settings)

    def result_card(self, evaluation): ...


class MyEvaluation:
    program_id = "my-task-evaluation-v1"

    def make_private_development_plan(self, case, settings, seed):
        # Derive the valid domain only from the public Case.
        # Use seed to sample settings["development_point_count"] POINTS.
        # Do not use private truth to define the sampling domain.
        query = my_private_point_query(case, settings, seed)
        return EvaluationPlan(
            phase=EvaluationPhase.DEVELOPMENT,
            query=query,
            seed=seed,
            settings=settings,
        )

    def make_plan(self, case, phase, settings, seed, request=None):
        # Finite plans for fixed/agent-selected Development and Official/Terminal.
        return EvaluationPlan(...)

    def evaluate(self, artifact, case, truth, plan):
        # Execute the candidate; return primary_error and additional metrics.
        return CaseEvaluation(...)

    def aggregate(self, artifact, case_results, plans):
        # Define a mean, tail, or another Task-specific cross-case summary.
        return ArtifactEvaluation(...)
```

The shared Development Evaluator and Hidden Verifier execute `MyEvaluation`; they
do not infer the physical domain, scoring points, or scientific error. The default
private mode calls `make_private_development_plan(case, settings, seed)`. If the
Task lacks a valid random-point rule, it must report that limitation and require
an explicit `user_fixed_query`, rather than silently using a fixed query.
`PluginTaskEvaluationProgram.from_settings(...)` accepts the same Task-owned query
factory through `private_development_query=`. Hidden evaluation calls
`make_plan(...)` and freezes the plan for each hidden Case, so the incumbent and
all candidates use the same evaluation conditions.

The three coordinate modes determine who selects the query, independently of
returned feedback. `return_truth=False` returns aggregate and per-case scalar
errors/metrics, with empty prediction/truth views and no scoring coordinates.
`return_truth=True` additionally requires `allow_grid_truth` or `allow_point_truth`.

Formal `ProductApplication` uses `SubprocessAgentExecutionService` and an explicitly
injected `CandidateExecutionService`. The AgentProgram calls a small IPC facade
without holding evaluator records or a TruthProgram. Candidate workers receive
only the public Case, public plan, and program bundle. Once predictions return to
the Controller, the Task's `score_candidate(...)` uses the private reference.
Only the explicit `local_development()` cooperative adapter permits trusted Python
agents/candidates in the Controller process. Subprocesses currently use the same
Unix user, so this scoring and failure boundary is not a hostile filesystem
sandbox.

Common program forms are listed below. Each Task defines and documents its exact
calling convention:

| Task | Required program interface | Objective |
| --- | --- | --- |
| Solver | Continuous NS uses `prepare_solution(case, solve_budget=None)` returning an object with `evaluate(query)`; optional `prepare_solutions` handles multiple cases. Other Tasks may define a `solve` interface | Full state fields |
| Observable Model | `predict_observables(...)` | Moments or observables, with no prescribed internal method |
| Reduced Dynamics | `initialize`, `advance`, `observe` | An explicit low-dimensional state and dynamics |
| Statistics | `statistics(case, requested, settings)` | Scalar or low-dimensional statistics |

Implement a Custom Task for other objectives. The engine does not impose one error
metric on every PDE: `EvaluationProgram.evaluate` may return a primary error and
additional scientific metrics, and `aggregate` defines the cross-case summary.

Use plain language in `agent_brief` to describe inputs, required outputs, permitted
data, and budgets. Agents should not need framework internals to discover their
program interface.

## 3. Generator: one interface

The shared engine uses one `GeneratorProgram` interface, with separate training
and loading:

```python
class MyGenerator:
    def train(self, initial_dataset, settings, progress=None):
        # Train the basis/latent model and return a GeneratorCheckpoint.
        ...

    def load(self, checkpoint, settings, progress=None):
        # Load an existing or newly trained checkpoint into a GeneratorState.
        ...

    def update_guidance(self, state, feedback, settings, progress=None):
        # Update guidance from the previous selected program's per-case errors.
        ...

    def sample(self, state, request, progress=None):
        # request.kind is BASE, GUIDED, or PERTURBED.
        ...

    def open_case_session(self, state, case, access):
        # Optional public-input or permitted development-solution queries.
        ...

    def case_session_methods(self, access):
        ...

    def release(self, state, case_ids):
        ...
```

Choosing a new Generator runs `train → checkpoint → load`. Selecting an existing
checkpoint runs only `load`. Both produce the same `GeneratorState` interface; the
Controller does not otherwise distinguish their origins.

`GenerationRequest` includes the base/guided/perturbed kind; development, guidance
feedback, selection, promotion, or terminal purpose; development/hidden visibility;
and the count, round index, seed, and perturbation anchors. Its `mode` is a legacy
backend compatibility field. Case selection is controlled by
`DevelopmentSamplingPolicy.strategy` (`agent_adaptive | fixed_recipe`), rather
than a separate online/offline lifecycle. BASE and GUIDED both produce clean
roots eligible as PERTURBED anchors. Perturbed children are not perturbed recursively.
D0 is not sampled from the Generator: adaptive development uses `allowed_sources`
and optional `starter.original`; a fixed ablation uses `fixed_recipe.original`;
Official/Terminal use the `original` counts in their suites/cohorts.

### Add perturbations to a continuous-solution Generator

Define perturbations in the `PERTURBED` branch of `sample(..., request)`. A basis
representation is optional, and no separate `PerturbationProgram` is required.
If the Generator produces a continuous solution `U`, the SDK offers a default
method: draw a dimensionless smooth random field `delta_U` and construct
pointwise

```text
U_child = (U + epsilon * field_reference_scale)
          * (1 + alpha * delta_U)
          - epsilon * field_reference_scale
```

`epsilon` is a dimensionless fraction. The PDE plugin supplies
`field_reference_scale` for each field, so their product has the same units as `U`.
Subtracting the same offset ensures `U_child == U` when `alpha=0`. The offset lets
the multiplicative field change regions where the original solution is zero.

Expose the desired relative-change range for the final continuous solution through
Generator settings, for example:

```python
def setup_fields(self):
    return (
        FormField(
            "perturbation_relative_change_min",
            "Minimum perturbation relative change",
            "number",
            0.05,
        ),
        FormField(
            "perturbation_relative_change_max",
            "Maximum perturbation relative change",
            "number",
            0.10,
        ),
        FormField(
            "perturbation_epsilon_fraction",
            "Perturbation epsilon fraction",
            "number",
            0.001,
        ),
        FormField(
            "perturbation_maximum_alpha",
            "Maximum multiplicative alpha",
            "number",
            10.0,
        ),
    )
```

The following is an SDK integration skeleton for plugin authors; it does not claim
that NS, VPFP, or GKW use this perturbation method. The plugin implements methods
starting with `_`; the other helpers come from the SDK. `ToyPoissonGenerator` is
a runnable reference.

```python
from pde_agent_verifier_solver.sdk import (
    GenerationKind,
    MultiplicativePerturbationSettings,
    PerturbationValidation,
    RelativeChangeRange,
    SmoothRandomFieldSettings,
    draw_smooth_random_field,
    epsilon_shifted_multiplicative_value,
    sample_multiplicative_mms_perturbation,
)


def sample(self, state, request, progress=None):
    if request.kind is not GenerationKind.PERTURBED:
        return self._sample_clean_or_guided(state, request, progress)

    if len(request.anchors) != request.count:
        raise ValueError("perturbation needs one clean anchor per child")

    settings = MultiplicativePerturbationSettings(
        target_relative_change=RelativeChangeRange(
            minimum=float(
                request.settings["perturbation_relative_change_min"]
            ),
            maximum=float(
                request.settings["perturbation_relative_change_max"]
            ),
        ),
        epsilon_fraction=float(
            request.settings.get("perturbation_epsilon_fraction", 0.001)
        ),
        max_alpha_attempts_per_random_field=int(
            request.settings.get("perturbation_alpha_attempts", 8)
        ),
        max_random_field_attempts=int(
            request.settings.get("perturbation_random_field_attempts", 8)
        ),
        maximum_alpha=float(
            request.settings.get("perturbation_maximum_alpha", 10.0)
        ),
    )
    field_settings = SmoothRandomFieldSettings(
        # Derive coordinate order and units from this Case geometry.
        length_scales=self._random_field_length_scales(state, request),
        term_count=int(request.settings.get("perturbation_field_terms", 32)),
        maximum_frequency_factor=float(
            request.settings.get("perturbation_max_frequency_factor", 4.0)
        ),
    )

    records = []
    for index, anchor in enumerate(request.anchors):
        # This dictionary is private to the Generator. Do not place the
        # continuous answer in Case.public_data.
        parent = self._active_continuous_answers[anchor.case_id]

        result = sample_multiplicative_mms_perturbation(
            parent,
            settings=settings,
            seed=self._child_seed(request.seed, index),
            draw_random_field=lambda field_seed: draw_smooth_random_field(
                field_settings,
                seed=field_seed,
            ),
            compose_candidate=self._compose_multiplicative_solution,
            measure_relative_change=self._solution_relative_change,
            make_admissible=self._enforce_solution_constraints,
            manufacture_problem=self._manufacture_problem_from_solution,
            validate=self._validate_manufactured_child,
        )
        record = self._materialize_perturbed_case(
            anchor=anchor,
            solution=result.solution,
            manufactured_problem=result.manufactured_problem,
            perturbation_report={
                "method": "epsilon_shifted_multiplicative_mms",
                "requested_solution_change": {
                    "minimum": settings.target_relative_change.minimum,
                    "maximum": settings.target_relative_change.maximum,
                },
                "achieved_solution_change": result.relative_change,
                "validation_metrics": dict(result.validation.metrics),
                "alpha": result.alpha,
                "random_field_attempt": result.random_field_attempt,
                "alpha_attempt": result.alpha_attempt,
            },
            request=request,
        )
        self._active_continuous_answers[record.case.case_id] = result.solution
        records.append(record)

    return tuple(records)
```

Apply the same formula in the plugin's continuous representation. Here `map_fields`
illustrates constructing a continuous function for each field and coordinate; it
is not a required SDK data structure:

```python
def _compose_multiplicative_solution(self, parent, delta_u, alpha, epsilon):
    return parent.map_fields(
        lambda field_name, coordinates, value:
        epsilon_shifted_multiplicative_value(
            value,
            delta_u(coordinates),
            alpha=alpha,
            epsilon_fraction=epsilon,
            reference_scale=self._field_reference_scale(parent, field_name),
        )
    )
```

The plugin also supplies five operations:

1. `_solution_relative_change(parent, child)`: measure the requested relative change on the PDE's physical domain, measure, and field combination. For NS, an example is quadrature-weighted relative velocity L2.
2. `_enforce_solution_constraints(child)`: enforce incompressibility, positivity, conservation, fixed initial conditions, or boundaries as required. Return the child unchanged if no adjustment is needed.
3. `_manufacture_problem_from_solution(child)`: apply the PDE operator to an accepted continuous solution to obtain forcing/source, initial/boundary data, and hidden truth.
4. `_validate_manufactured_child(parent, child, problem)`: this required callback returns `PerturbationValidation`. It checks the manufactured forcing/source, initial/boundary data, derivative scales, and PDE structure, and reports the resulting problem change in its metrics.
5. `_random_field_length_scales(state, request)`: derive coordinate order and correlation lengths from the Dataset/Case geometry. The default SDK field uses Euclidean coordinates; periodic domains, manifolds, or other topologies need a suitable `draw_random_field` implementation or geometry-aware envelope.

`_materialize_perturbed_case(...)` then creates a record with `CaseSource.PERTURBED`
and preserves `parent_case_id`. Continuous truth stays out of `Case.public_data`.
The configured range controls solution change; it does not replace validation of
the actual PDE inputs exposed to the agent in step 4. Store `perturbation_report`
in a plugin-private registry and expose only public fields through previews or
reports. `CaseRecord` has no general metadata field; keep private truth, seeds,
and internal random fields out of `Case.public_data`.

The SDK tries different `alpha` values on the same random field, measuring the
final relative change after enforcing constraints each time. MMS construction or
validation may raise `RecoverableSamplingRejection` with JSON-compatible diagnostics
when a proposal cannot be constructed. The helper retries alpha on the same field,
then draws another field only after exhausting those attempts. If all fields fail,
it raises `RecoverableSamplingRejection`. The shared CaseSampler already handles
root-local retries, case budgets, parent/child transactions, and resource release;
reuse that behavior instead of adding another Controller retry loop.

The mathematics belongs to the plugin. Examples include:

- NS: continuous basis coefficients → PCA latent → rectified flow.
- VPFP: a positivity-preserving Fourier–Hermite continuous representation and online sampler.

For an NS-style basis/PCA/flow approach, inherit `CoefficientFlowGenerator` and
provide the PDE backend, forms, and preview. Other mathematical approaches can
implement `GeneratorProgram` directly. Existing training scripts may remain in
place and be called by the backend. Use the same Generator for the panel and
Controller.

## 4. Product: assemble the components

Keep `product.py` small:

```python
def build_my_plugin() -> PDEPlugin:
    return PDEPlugin(
        plugin_id="my_pde",
        title="My PDE",
        description="The scientific program users can obtain",
        aliases=("my-pde",),
        problem=MyProblem(),
        generator=MyGenerator(),
        tasks={
            "solver": MySolverTask(),
            "statistics": MyStatisticsTask(),
        },
        default_task="solver",
    )
```

Add `build_my_plugin()` to `built_in_plugins()` in `plugins/registry.py` for
`get_plugin()` discovery. The application must still build a `ProgramBundle`, bind
data and execution services, and pass them to `PanelController`; see
[connecting the panel](#connect-product). The generic `panel` command remains a
Toy entry point. The shared engine and standard forms need no equation-specific
branches.

Reconstruction is a choice when creating an experimental Project, rather than a
separate PDE or Controller:

```python
project = app.create_project(
    "PIV project",
    dataset,
    programs,
    reconstruction=ReconstructionRecipe(
        program=get_reconstruction_program("gappy_pod_ls"),
        data_settings={"split_manifest": "piv_split_v1"},
        method_settings={"rank": 12},
        seed=7,
    ),
)
```

The selection entry point is currently Python/API. Keep individual PIV file
formats out of `PDEPlugin` and the shared panel.

## 5. Legacy staged runs: historical reproduction

New product runs use `AgentExecutionService` for long-running subprocess/GPU work.
The same `CampaignController` owns scientific rounds, Official evaluation,
Promotion, memory, and Terminal evaluation.

Register a `LegacyStagedCampaignAdapter` in `PDEPlugin.legacy_staged_campaign` only
when reproducing a historical paper experiment or an explicit control arm. It
provides three operations:

1. `load_config`: read JSON and resolve relative paths against the configuration file.
2. `preflight`: check that data, programs, and required settings are available before submission.
3. `compile`: return a `StagedRunPlan` with stage dependencies.

This compatibility path requires explicit commands:

```bash
./pde legacy-plan historical-config.json
./pde legacy-run historical-config.json
./pde legacy-resume historical-config.json
```

Historical `MacroClosedLoopRunner` reference scripts additionally require
`PDE_ENABLE_LEGACY_MACRO_REFERENCE=1` and `--legacy-macro-reference`; otherwise they
refuse to run. These commands and switches are not templates for new PDEs.

Use recognizable scientific actions as stage names, and declare commands,
CPU/GPU/time requirements, and expected historical result files. Existing NS/VPFP
staged adapters serve reproduction rather than defining the canonical product flow.

## 6. Where settings belong

### In PDE and Task code

- Data semantics and loading.
- Public Case contents.
- Queryable fields and coordinates.
- Candidate calling conventions.
- Scientific errors and aggregation. `PromotionProgram` independently decides promotion; the plugin may provide a recommended preset.
- Generator representation and capabilities.
- Panel forms, Case previews, and result cards.

### At Project creation

- Dataset location, kind, and Problem/geometry settings.
- Whether experimental data use reconstruction.
- If enabled: `ReconstructionProgram`, the six data roles, rank, noise/prior settings, and a nonnegative seed.
- Changing any reconstruction choice creates a new Project/prepared-data variant; these choices do not switch within a round.

### In each Campaign configuration

- Task selection.
- Generator basis, latent, flow, and other parameters.
- Rounds, per-agent case limits, allowed sources, and optional starter cases.
- Guidance/perturbation switches and the shared Official suite for each round. When using the public helper, include solution-change bounds, epsilon, and any random-field controls exposed by the plugin.
- Agents, prompts, full wall-clock budgets including acquisition, generation, evaluation, and waiting, master seed, and optional truth disclosure.
- Terminal cohorts, ranking diagnostics, and the result directory.

## 7. Integration verification checklist

Before integrating a new PDE, verify at least the following:

1. `prepare_dataset` reads a small real sample.
2. Agent-visible Cases contain no private answer data.
3. Grid and point queries return the expected shapes.
4. Each Task can evaluate a correct and a clearly incorrect candidate.
5. `create_evaluation()` ranks those candidates correctly.
6. Declared Generator capabilities match actual online/guided/perturbed support.
7. If using the perturbation helper, exercise `Generator → helper → CaseRecord → RoundCaseSampler → CampaignController`. Check Development, Official, and Terminal settings, lineage, target solution change, MMS construction, rejection, and release.
8. If supporting experimental reconstruction, check disjoint calibration/target/downstream frames and visible/validation coordinates. The reconstruction program receives no private reference; the materializer receives no evaluation/report. Public aliases, diagnostics, and warnings contain no private data, and reports are deeply immutable. Multiple runs in one Project reuse reconstruction, and explicit reconstruction of numerical datasets is rejected.
9. If using built-in POD, test shapes under two geometry/flattening layouts, exact least-squares recovery, a noisy/prior MAP case, rank-deficient sensors, and invalid covariance handling against the method contract.
10. In a two-round Campaign, preserve both first-round agent logs, append Controller results afterward, and make the complete first-round Campaign Memory Log available to the second round.
11. External workers use request schema v2, treat `CAMPAIGN_MEMORY.md` and `CANDIDATE_HISTORY.json` as Controller-owned inputs, and checkpoint atomically through `write_agent_round_log()`. Timeout/failure recovery uses the last complete log.
12. A small panel run proceeds from configuration through all rounds, the Terminal matrix, Campaign Memory Log, Candidate History, and a callable final `S_R`.

Use `tests/test_reference_pde_programs.py`, `tests/test_gkw_reference_plugin.py`,
`tests/test_evaluator_contract.py`, `tests/test_runtime_multi_candidate_terminal.py`,
`tests/test_project_reconstruction.py`, `tests/test_reconstruction_gappy_pod.py`, and
`tests/test_toy_product_flow.py` as verification examples. `tests/test_macro_*`,
historical PDE campaigns, and staged tests cover compatibility with older runs.
