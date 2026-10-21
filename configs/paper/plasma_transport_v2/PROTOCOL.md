# Recorded VP / VPFP protocol

Researchers give two Agents access to public PDE inputs and a limited
development budget. Each Agent develops a macroscopic solver and explicitly
nominates one program. A common evaluation selects the next incumbent. After
six rounds, the retained program is the result. This document records the
completed seed-101 experiments, including changes made during those runs.

## Equations and requested outputs

| Family | Equation and domain | Candidate output |
| --- | --- | --- |
| ARTEMIS VP | Forced 1D1V Vlasov–Poisson, periodic space, continuous velocity | `M0`, `M1`, `M2`, `M3`, `E` at `(t,x)` |
| Strong VPFP | Forced 2D2V Vlasov–Poisson–Fokker–Planck, periodic spatial torus, full velocity space; collision frequency 10–30 | Ten transport fields at `(t,x,y)` |
| Weak VPFP | Same 2D2V equation; collision frequency 0.02–0.08 | Same ten transport fields |

VP uses `f_t + v f_x + E f_v = source` and
`E_x = M0 - background`, with zero-mean periodic electric field. VPFP uses
`f_t + v·grad_x f + E·grad_v f - nu div_v(v f + theta grad_v f) = source`,
`-Delta phi = rho - rho_bg`, and `E = -grad phi`. VPFP bath temperatures range
from 0.8 to 1.2. Each public Case supplies its domain and physical parameters.
The native families use manufactured solutions and their matching sources;
ARTEMIS-derived inputs do not make these observed spacetime trajectories.

VPFP's ten fields are `rho`, `electric_x`, `electric_y`, `momentum_x`,
`momentum_y`, `pressure_xx`, `pressure_xy`, `pressure_yy`, `heat_flux_x`, and
`heat_flux_y`. Pressure and heat flux are central moments at unit particle
mass. A full phase-space distribution is not a required output. The Agent may
choose a kinetic method, a closure, a reduced model, or another representation.

The scientific interfaces are the [VP plugin](../../../src/pde_agent_verifier_solver/plugins/vp_1d1v/README.md)
and [VPFP plugin](../../../src/pde_agent_verifier_solver/plugins/vpfp/README.md).
VP exposes initial-distribution and source queries; VPFP exposes `rho0`, `g0`,
and source queries. Solver-selected grids and velocity cutoffs are numerical
choices, not hidden-answer coordinates. Source access respects the Case's
causal horizon. Hidden answers, Generator latents and weights are unavailable
to the candidate. These runs disable development truth and completed-solution
sessions.

## Agent development and timing

| Setting | Recorded value |
| --- | --- |
| Model arms | `gpt-5.6-sol` / `max`; `gpt-6-astra` / `ultra` |
| Seed and rounds | Seed 101; six rounds |
| Agents | Two per round: broad explorer and critical refiner |
| Agent budget | 10,800 seconds per Agent per round; all elapsed time, including tools and waits |
| Development cases | 32 per Agent per round, including four initial Base cases and perturbation children |
| Case acquisition | Fresh adaptive Base / Guided acquisition; no forced mixture or case replay |
| Development scoring | 128 private points |
| Official / terminal scoring | 256 private points per case |
| Solver budget | One shared 900 charged-second budget; no separate preparation allowance |
| Nomination | One explicitly selected primary per Agent; ordinary submissions archive alternatives |
| Initial state | Simple S0, empty Campaign Memory, no imported previous solver or Guidance |

Candidates export `prepare_solution(case, solve_budget=None)` and return a
handle exposing `evaluate(points_query) -> AnswerView`. Scoring coordinates
arrive after preparation. Startup/imports, preparation, evaluation, and
serialization share the Case budget. Only Controller-metered Oracle queue and
shared-lock waits are excluded; Oracle numerical work and transfer are charged.
An Agent's own total wall clock continues while waiting.

CPU and GPU methods are allowed. The run configuration records the dedicated
Agent, Oracle and candidate resources. Solver batch size is one and GPU slots
per device are one; candidate worker bounds are 16 for strong VPFP and eight
for weak VPFP / ARTEMIS. Historical allocation fields are configuration
records, not proof of the GPU used by every migrated execution. Consult the
archived execution audits for per-attempt hardware and timing.

The exact six role prompts are under [prompts/](prompts/). Their hashes match
the original configurations; the same family prompts serve both model arms.

## Selection, Guidance and terminal reporting

The final protocol is `balanced_48_with_base32_feedback_v1`. Every contender
and incumbent is evaluated on the same round subset: 16 Base, 16 Guided, and
16 Base-perturbed cases. Promotion minimizes the case-weighted mean transport
score. A failed case disqualifies a contender; a successful-subset average
cannot promote it.

The selected program receives another 16 clean Base evaluations, giving 32
Base feedback rows for Guidance. These additional cases do not change the
promotion comparison. Guidance uses the first 32 clean Base rows in recorded
order and the same transport objective. The Base model stays fixed; the fitted
Guidance snapshot is shared and frozen for the round.

Two completed first rounds predate that subset rule:

| Arm and round | Actual selection cases | Later rounds |
| --- | --- | --- |
| Strong-Sol R1 | 256: 86 Base + 85 Guided + 85 Base-perturbed | R2–R6 use 48 |
| Weak-Sol R1 | 192: 64 each of Base, Guided, Base-perturbed | R2–R6 use 48 |
| All other arm/round pairs | 48: 16 in each group | Same rule |

These earlier decisions remain unchanged. The native promotion reports are
the evidence for their actual counts, even where the later configuration or
policy-transition record describes a different bank/subset. See
[selection.csv](results/selection.csv), [native decisions](results/selection_decisions.json),
and [policy transitions](results/selection_policies.json). The 32-row Guidance
rule describes the final protocol; retained early-round state is not rewritten.

The common terminal comparison has 64 cases: 16 Base, 16 Base-perturbed,
16 Guided from the common round-one G1, and 16 G1-Guided-perturbed. Terminal
is for reporting. It does not select a program or update Guidance. Deliver
S6 even if an earlier solver has a lower terminal or independent error.

The exact resolved configurations retain the larger sealed-bank declarations
(`official`: 86/85/85; terminal: 256/128 per cohort) and some older feedback
fields. **These declarations are not the final evaluated subset sizes.**
The subset policy, native decisions and table above must be read together.
The repository's older five-round templates are separate historical protocols.

## Transport score

For each of density `n`, electric field `E`, particle flux `j`, central
pressure `P`, and central heat flux `qh`, take the RMS error over all query
points and divide by `sqrt(reference_RMS**2 + floor**2)`. The total is

```text
score = (e_n + e_E)/8 + (e_j + e_P + e_qh)/4
```

Lower is better. Scalars use absolute magnitude, vectors use Euclidean norm,
and pressure uses the full Frobenius norm (`Pxy` contributes twice in 2D).
There is no division by a small pointwise reference value. Nonfinite values
and nonpositive density are invalid rather than clipped.

VP returns raw velocity moments. Center prediction and reference separately
using their own `u = M1/M0`:

```text
n = M0; j = M1
P = M2 - M1**2/M0
qh = (M3 - 3*u*M2 + 2*M0*u**3)/2
```

`M3` itself is not heat flux. VPFP already supplies the central fields.

| Quantity | Weight | Frozen scale floor |
| --- | ---: | ---: |
| Density | 0.125 | 1.0041253663782175 |
| Electric field | 0.125 | 0.0698185715388132 |
| Particle flux | 0.25 | 0.4788345668393978 |
| Central pressure | 0.25 | 2.930763214320111 |
| Central heat flux | 0.25 | 0.9900774471118267 |

Calibration uses independent reference roots, with equal weight across the
three families and equal weight across condition strata within a family.
It uses 64 strong, 64 weak and 32 ARTEMIS roots, 128 points per root, with
explicit unit scales of one. It does not use solver predictions or tune the
floors to solver performance.

The shared objective identity is
`855cf8984dc60dee1d654fa3f0e2e2afbd184a23b8608f3181456077f51e14ae`.
[metric_calibration.json](metric_calibration.json) is the exact serialized
asset used by the independent follow-up (SHA256
`973c82defb0f0e5d3d2ac7612a62cdb1c1c8b8bf733977a9ce8200c8b94e1da4`).
It equals the calibration mapping embedded in all six native configurations.
The older file serialization has SHA256
`04742192d469784180a2c893012f6d762b20f5309ec7ba4b6fb2ba9c332070d0`;
file-byte identity and semantic objective identity are recorded separately.

## Independent validation and reproduction

The [result guide](results/README.md) separates native terminal results,
independent original-protocol results, and declared timeout supplements.
The independent VPFP comparison uses two non-neural eight-case interpolation
banks. The independent VP comparison uses six new ARTEMIS profile-derived MMS
cases excluded from historical Generator training. These comparisons use
the same frozen programs and score, but distinct input families and recorded
execution policies. They are not additional Agent-development rounds.

The [provenance record](provenance.json) identifies the original source
documents and the archived framework by content hash. The published copies of
the recorded configurations and result tables replace site-specific storage
locations with documented placeholders; `provenance.json` records both the
archived and the published hashes. Full historical replay still needs the
recorded scientific assets and execution records; current `main` is not claimed
to be the frozen runtime. Publishing these files does not change the shared engine or launch
new runs.
