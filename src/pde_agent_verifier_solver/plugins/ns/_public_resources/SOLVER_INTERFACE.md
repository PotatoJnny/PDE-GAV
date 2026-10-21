# Writing and submitting an NS solver

The run prompt supplies your objective, research direction, time budget and
submission rule. [README.md](README.md) defines the equation and scientific
inputs; [AGENT_TOOLS.txt](AGENT_TOOLS.txt) lists the commands. This file describes
the program interface. It does not prescribe a numerical method.

## Read the actual Case

`prepare_solution` receives a `Case`. For a generated continuous NS Case,
`case.public_data` contains:

| Field | Meaning |
|---|---|
| `parameters.viscosity` | The viscosity for this Case |
| `domain.time_interval` | Physical `[t0, t1]` |
| `domain.bounds.x`, `domain.bounds.y` | Physical coordinate bounds |
| `domain.geometry` | Geometry, including any obstacle polygons; use the supplied geometry |
| `boundary_condition` | The velocity boundary condition and any declared query method |
| `initial_condition`, `forcing` | The public input access declarations |
| `oracle_protocol.limits.max_points_per_request` | Maximum coordinate rows in one input query |

Use the Case values rather than assuming all Cases have the same geometry or
normalizing physical coordinates implicitly. `status` summarizes the public
Case fields; `export-cases` preserves complete fields when a summary is too short.

A `no_slip_velocity` boundary gives its prescribed velocity directly. A
`manufactured_velocity_trace` boundary declares `query_boundary_velocity`;
query it at physical `(t,x,y)` points on the boundary. Do not assume a zero
boundary when the Case declares a different one.

Public input access inside `prepare_solution`:

```python
from pde_agent_verifier_solver.sdk import open_public_case_session

session = open_public_case_session(case)
initial_velocity = session.query_initial_velocity(xy)  # float64 [N, 2]: x, y
forcing = session.query_forcing(txy)                    # float64 [N, 3]: t, x, y
```

Both return `[N, 2]` values in x/y component order. For larger grids, split
coordinates into calls that fit the declared per-request limit. These calls
supply equation inputs; they do not reveal the hidden solution. The solver
chooses its own mesh, time steps, representation and query coordinates.

## Two calls to your program

The entry file must define this undecorated synchronous module-level function:

```python
def prepare_solution(case, solve_budget=None):
    # Build your solution for this Case, then return its evaluation handle.
    ...
```

Implement its body; `...` is a placeholder. Complete the numerical solve and
construct the representation here, before receiving scoring coordinates.
The returned object must implement `evaluate(points_query)`. That method
samples or projects the prepared solution at the requested points; it does
not start another solve or query equation inputs at the scoring points.

`points_query.points` is an ordered sequence of dictionaries with physical
`t`, `x`, `y`. Return an `AnswerView` with the same point order, `shape=(N,)`,
and one finite value per point in each of `velocity_x` and `velocity_y`.
Pressure may be needed internally but is not an output required for scoring.

This reusable output adapter wraps your already-prepared interpolant or field:

```python
import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind

class PreparedVelocity:
    def __init__(self, velocity_at):
        # velocity_at only samples your prepared solution; it returns [N, 2].
        self.velocity_at = velocity_at

    def evaluate(self, points_query):
        txy = np.asarray(
            [[p["t"], p["x"], p["y"]] for p in points_query.points],
            dtype=np.float64,
        )
        velocity = np.asarray(self.velocity_at(txy), dtype=np.float64)
        if velocity.shape != (len(txy), 2) or not np.isfinite(velocity).all():
            raise ValueError("Expected finite velocity values with shape [N, 2]")
        return AnswerView(
            kind=QueryKind.POINTS,
            fields={
                "velocity_x": tuple(velocity[:, 0].tolist()),
                "velocity_y": tuple(velocity[:, 1].tolist()),
            },
            points=points_query.points,
            shape=(len(txy),),
        )
```

For GPU implementations, perform any needed transfer to host memory before
constructing this answer. `status.candidate_execution` gives the allocated
devices and per-Case budget. A GPU-enabled worker exposes its assigned card as
`cuda:0`; CPU, GPU and combined computation are supported. Check remaining
charged time with the SDK functions documented in `AGENT_TOOLS.txt` and leave
time to return the answer.

## Portable program directory

Create a directory in your workspace, for example:

```text
solver_work/
    candidate_bundle.json
    submission.py
    numerics.py          # optional implementation helper
```

Its manifest is ordinary JSON. For the directory above:

```json
{
  "schema": "pde_agent_verifier.candidate_bundle.v1",
  "entrypoint": "submission.py",
  "files": ["submission.py", "numerics.py"]
}
```

List every algorithm file, including any needed data or weights, relative to
the bundle directory. Remove `numerics.py` from the list if it is not used.
Keep research notes, exported Cases and evaluation outputs outside this bundle.
Use libraries available in the candidate environment; development-shell imports
alone do not establish that a submitted program will run. `evaluate` exercises
the actual candidate worker and reports import or interface failures.

If an incumbent is available, `clone-incumbent solver_work` creates a writable
bundle with the manifest already filled in. You may instead start your own
program. The original incumbent directory is read-only.

## Including trained weights

A submitted program may contain weights learned using the Development information
available in this run. Include them in the same directory and manifest, for
example `files=["submission.py", "model.pt"]`, and load them relative to
`Path(__file__).resolve().parent`. The frozen submission and next round's cloned
program carry these declared files together. Use the allocated device when loading
weights and return CPU values through `AnswerView`. Training, loading and GPU
synchronization follow the published budgets; the bundle size limits still apply.

## Command example

An online run may start with no acquired Cases. This example obtains one Base
Case before evaluating an incumbent copy; an already acquired Case can also be
used. Directory names must be fresh; adjust them if already used.

```bash
python -m pde_agent_verifier_solver.engine.agent_tool sample base 1 > acquired_case.json
python -m pde_agent_verifier_solver.engine.agent_tool status > run_status.json
case_id=$(python -c 'import json; print(json.load(open("run_status.json"))["cases"][0]["case_id"])')
python -m pde_agent_verifier_solver.engine.agent_tool clone-incumbent solver_work
python -m pde_agent_verifier_solver.engine.agent_tool evaluate solver_work "$case_id" > feedback.json
```

Edit the program and evaluate whichever acquired Cases you choose. The example
above only checks that the current bundle runs. `feedback.json` includes
`primary_error`, `metrics`, and per-Case results. For example, an error of
`0.04` means 4%. A failed command returns nonzero and writes an error object;
inspect its diagnostic and any completed partial results.

When this is the program you want to nominate:

```bash
python -m pde_agent_verifier_solver.engine.agent_tool submit solver_work --sequence 1 --name "NS solver" --summary "Describe your implementation" --usage "prepare_solution then evaluate"
```

For `single_primary_v1` (the new NS template), add `--primary` to the submission
above to select it for Official evaluation. A plain submit saves a version but
does not change the current primary. To select an existing frozen version:

```bash
python -m pde_agent_verifier_solver.engine.agent_tool select-primary 1
```

The selection refers to the saved code and weights, not later edits to solver_work.
`status` reports the chosen sequence. There is no automatic latest-version fallback.
Use the next contiguous sequence number for later submissions. Write your
research notes to `notes.md`, then save them with:

```bash
python -m pde_agent_verifier_solver.engine.agent_tool log notes.md
```

Each submission freezes the current program files. Subsequent edits require
another submission to change the nominated program. The run prompt states
whether your explicit primary, only your latest submission, or all nominations enter round-end
selection. The Controller handles hidden selection; the Agent submits the
program and research notes, not its own selection score.

## Optional batch solving

When `status.candidate_execution.solver_batch_size > 1`, keep the single-Case
entry point and optionally add an undecorated module-level function:

```python
def prepare_solutions(cases, solve_budget=None):
    # Solve these Cases together, for example with a batched neural operator
    # or batched linear algebra. Return one prepared handle for each Case id.
    return {case.case_id: handles[index] for index, case in enumerate(cases)}
```

The body above illustrates the return format; construct `handles` with your
chosen method. The Controller calls this function once per batch. It does not
pass scoring coordinates until all handles have been returned. It then queries
and scores each handle independently, preserving the original Case order.
Different Cases can have different geometry and grids; group compatible inputs
inside your implementation when needed. Smaller final batches are possible.

To query a batch of public inputs concurrently:

```python
from pde_agent_verifier_solver.sdk.public_sessions import query_public_cases
forcing_by_case = query_public_cases(cases, "query_forcing", txy_by_case)
initial_by_case = query_public_cases(cases, "query_initial_velocity", xy_by_case)
```

The coordinate dictionaries must contain exactly these Case ids. Values are the
usual per-Case coordinate arrays; results use the same keys. Each Case retains
its own authorized session and point limit. This helper does not expose answers.
The helper combines transport for Cases served by the same Oracle host; their
numerical work is still scheduled independently. The worker batch size is not
a limit on how many Cases you may pass to the evaluate command.

For a grid and known future forcing times, you can query several times at once
and retain the returned arrays. For example, construct a `(times * points, 3)`
array in `(t, x, y)` order, query it once, and reshape the response. Stay within
the published per-request point limit. This changes neither the input values
nor your integration method. Adaptive methods may instead query as needed;
choose batch sizes based on memory and the experiments you want to perform.


A solver batch shares a charged-wall limit equal to the published single-Case
limit multiplied by the **actual** number of Cases in that batch. `solve_budget`
is the remaining shared time, including startup. Only intervals when **all**
batch members are waiting for Controller service are refunded, once. This is
shared batch timing, not a measurement of individual Case solve times. The
Agent's wall clock continues throughout. Single-Case programs retain the
original per-Case limit and are scheduled concurrently when resources allow.
