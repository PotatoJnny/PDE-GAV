# Public 2D incompressible Navier--Stokes contract

Each case solves, on its declared physical time/space domain,

```text
du/dt + u du/dx + v du/dy = -dp/dx + nu Laplacian(u) + forcing_x
dv/dt + u dv/dx + v dv/dy = -dp/dy + nu Laplacian(v) + forcing_y
du/dx + dv/dy = 0.
```

Velocity uses the boundary condition declared by the case. Existing structured
cases prescribe zero no-slip velocity. Experimental-derived manufactured cases
may instead declare `manufactured_velocity_trace`: query `query_boundary_velocity`
with physical `(t,x,y)` points on the rectangle boundary or declared obstacle
polygon edges. Interior and out-of-domain points are rejected. Its values are the
trace of the same hidden continuous answer used to derive initial data and forcing.
Pressure is defined only up to an additive gauge and is not part of the default score.  The public numerical
oracle evaluates forcing and initial velocity at caller-selected coordinates.
The hidden manufactured answer is continuous: it is chosen first, and forcing,
initial conditions, and boundary data are then derived from that answer.

The controller scores `velocity_x` and `velocity_y` at a sealed physical
`(t,x,y)` point set.  A candidate owns its internal grid, basis, time stepper,
and interpolation to those requested points.

Official velocity-match points are sampled in the public fluid domain.  The
controller rejects points inside the public obstacle polygons so solid-region
zeros cannot dilute the velocity error.  Boundary values may be inspected as
diagnostics but are not mixed into the default volume-error panel.  A fixed
controller seed and public geometry deterministically define the common point
panel used for paired candidate comparisons.

## Cases and available research tools

A **Case** is one specified NS initial-boundary-value problem: its geometry,
parameters, initial velocity and forcing define the inputs to your solver.
A Case is not a grid; for generated Cases you choose where to query the public
inputs. The solution used for scoring stays with the Controller unless a
separate development-truth capability is explicitly enabled.

The run's `status` lists which tools are enabled:

- **Base** draws a new problem from the configured Generator's base distribution.
- **Guided** draws a new problem using a learned difficulty signal from measured
  reference-solver errors. It aims at challenging problems; it does not guarantee
  that every sample is harder, and it does not retrain on each tool call.
- **Perturbed** creates a controlled variation of an eligible acquired clean Case.
  The configured perturbation changes the manufactured field and recomputes its
  consistent public inputs. You receive a new Case to solve and compare with
  its parent. It is not noise added to a reported error.

Inside `prepare_solution`, public input queries use the SDK session:

```python
from pde_agent_verifier_solver.sdk import open_public_case_session

session = open_public_case_session(case)
initial_velocity = session.query_initial_velocity(xy)  # float64 [N, 2]: x, y
forcing = session.query_forcing(txy)                    # float64 [N, 3]: t, x, y
```

Here `xy` and `txy` are your chosen physical coordinates within the Case domain;
queries return `[N, 2]` arrays in velocity/force component order `(x, y)`.
Read the domain and query limits from `case.public_data`; respect the declared
boundary condition and per-request point limit. Boundary queries are available
only for Cases whose public contract declares that method.

The hidden scoring points do not determine your internal grid or time steps.
They are supplied to `handle.evaluate` only after you have prepared the solution.

## Error and round-end selection

For each Case, the default velocity error is the arithmetic mean of the two
component relative L2 errors. Each component uses
`sqrt(sum((prediction - reference)**2) / max(sum(reference**2), 1e-30))`
on that Case's scoring points. The default mean objective averages these
Case errors without clipping. An error of `0.04` is 4%, not 0.04%.
The run prompt specifies its aggregation and selection rule if different.

During development, testing your program returns feedback for your own work.
At the end of the round, the Controller evaluates the nominated programs and
the entering incumbent on a common hidden suite to choose the program that
continues into the next round. Development results do not choose it directly.
The final test is separate from development and round-end selection.

## Using the environment

[AGENT_TOOLS.txt](AGENT_TOOLS.txt) explains sampling, evaluation, submission and
feedback. [SOLVER_INTERFACE.md](SOLVER_INTERFACE.md) gives the exact input/output
format, program-directory manifest and executable command examples. `CAMPAIGN_MEMORY.md` in your workspace
contains previous research notes and round outcomes; `CANDIDATE_HISTORY.json`
lists submitted programs and their recorded evaluation/selection results.
These records are generated for the current run; they are not a static method guide.
