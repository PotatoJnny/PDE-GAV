# Public forced 1D1V Vlasov--Poisson contract

Each case solves, on its declared time interval and periodic spatial domain,

```text
d_t f + v d_x f + E d_v f = source,
d_x E = M0 - n_background,
mean_x(E) = 0,
Mk(t,x) = integral_R v^k f(t,x,v) dv.
```

The sign convention absorbs particle charge into the acceleration field `E`.
The public velocity array is a finite numerical query window, not a physical
velocity boundary. Each case declares one of two oracle schemas in
`case.public_data.equation.oracle_protocol`:

- `vp_1d1v_cached_array_oracle_v1` is the backwards-compatible default. The
  public arrays contain authoritative `f0(x,v)` and source samples on one
  materialized `(t,x,v)` grid. The optional numerical session performs bounded,
  Agent-selected interpolation of those same immutable arrays.
- `vp_1d1v_exact_coordinate_oracle_v1` evaluates `f0` and source directly at
  every legal Agent-selected coordinate from the same frozen continuous hidden
  answer that manufactured the equation. In this mode, `f0` is evaluated at
  the initial time and the source is evaluated as the exact coordinate-jet
  expression `d_t f + v d_x f + E d_v f`. No public-array interpolation
  determines either response. Materialized arrays remain present for ABI
  compatibility but are non-authoritative convenience samples only.

Both modes keep the continuous answer's parameters and latent state private.
Any candidate work using direct array reads or session calls during
`prepare_solution` is included in the measured preparation wall time. A runner
must not advertise the exact-coordinate schema unless its session implements
the direct frozen-answer evaluations above.

## Causal-horizon evaluation

Development numerical oracles may expose the complete trajectory. In an
official causal evaluation, however, the Controller selects one non-initial
native horizon and exposes only f0 and answer-first source access on the closed
prefix from the initial time through that horizon. In cached-array mode, the
public source array is truncated to this prefix. In exact-coordinate mode, the
declared time interval bounds legal source-query coordinates; convenience
samples do not determine oracle values. Causal evaluation is the default for
the canonical Artemis Task; an explicit Task setting
`causal_horizon: false` is the compatibility opt-out. The public
time array and the declared equation time interval end at the same horizon.
Source requests after it are rejected.

Every sealed scoring point for that case has time exactly equal to the
information horizon. Candidates must therefore support arbitrary truncated
horizons and must not assume the full development trajectory is available at
evaluation time. The manufactured answer and its exactly derived equation are
unchanged; only the Agent-visible evaluation view is truncated. A clean case
and its paired perturbation use the same horizon and native-x point draw.

The Task requires only `M0`, `M1`, `M2`, `M3`, and `E` at requested physical
`(t,x)` points. No kinetic distribution output is required. Canonical
file-submitted Candidates must export
`prepare_solution(case, solve_budget=None)` returning a handle with
`evaluate(query) -> AnswerView`; the Python Candidate publisher requires this
two-phase ABI. Preparation runs before scoring coordinates are revealed, and
the formal runner sets `require_prepared_solution: true`. The older
`predict_observables(case, query, budget=None)` entrypoint remains available
only for explicitly selected in-process compatibility use; it is not a
valid canonical Python bundle entrypoint. Official points are sampled
without replacement from the public native panel with controller-private
entropy. Paired-query mode couples clean/perturbed horizon and spatial draws.

During Candidate preparation, `open_public_case_session(case)` from the public
SDK grants initial-distribution and causal source access. A prepared handle
must return exactly the requested POINTS coordinates, shape `(N,)`, and the
five declared fields. Completed-solution access is governed by the Run's
Development disclosure policy. Explicit authorization can enable it for both
Agent Development queries and Candidate Development evaluation; it is never
available in Official/Terminal. The formal Artemis experiment disables
`allow_grid_truth`, `allow_point_truth`, and
`allow_development_solution_session`, so neither Agent nor Candidate receives
completed answers in that experiment. The submitted Candidate must work
without such access. Sessions and retained bound methods are
revoked when their evaluation scope ends. Optional handle `close()` runs
after every evaluation; do not keep a session for another case or round.

## Versioned macro-transport scoring

For a run selecting `plasma_macro_transport_v2`, the raw output ABI above is
unchanged. The verifier independently centers the prediction and reference:
`n=M0`, `j=M1`, `u=M1/M0`, `P=M2-M1^2/M0`, and
`q_h=(M3-3*u*M2+2*M0*u^3)/2`. In particular, `M3` is not the heat flux.
Each error is an all-point RMS difference with the Task's frozen physical
scale floor in its reference denominator. The score is
`(e_n+e_E)/8+(e_j+e_P+e_q_h)/4`. The Task brief records the exact version,
common units, calibrated floors, and objective hash. Legacy runs retain their
original score and are not numerically comparable to this version.
