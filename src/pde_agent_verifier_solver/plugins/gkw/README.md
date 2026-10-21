# GKW unified infrastructure

This plugin pins its compatibility boundary to stable GKW commit
`6947a64c81b34dfbaa386fa58d1700bcc18e112d`. The concrete Torch adapter lazily
imports the separately installed selected-GKW physical package; the unified
package does not import its historical controller or ranking implementation.

The trained-model/public-ABI compatibility pin above remains unchanged. The
continuous-condition equation-side implementation is pinned independently to
physical-interface commit
`80632f11a39b8577124ffa652556af1521d9b992`. It keeps the trained weights
unchanged. An external same-discrete-branch prior draws a fresh rank-four
continuous `(rlt_i, rln_i, shat, q)` condition and supplies it to both selected
conditional EDMs. The EDMs sample the complete kinetic and potential latent
code tensors; no solution-code interpolation, low-rank proxy, or finite latent
dictionary is used. The complete public operator and answer-first source are
then rebuilt from the decoded answer and sampled condition.

The qualified sampler is nevertheless an empirical local conditional kernel,
not an unconditional full-prior model. Each 20-channel condition contains 16
anchor-derived per-frame mean/log-standard-deviation values and four fresh
continuous public values. Sampling starts from an empirical code anchor plus
full-dimensional Gaussian noise and runs the validated 32-step, sigma-0.1 Heun
trajectory. Offline production qualification verifies the actual full code
shapes, causally reachable parameter counts, full-rank causal output
projections, all 63 network calls per EDM, and a finite positive latent
residual at every call. It therefore proves that the trained high-dimensional
models participate in generation without claiming lookup-free physical-state
or pure-noise unconditional generation.

No qualification certificate from the prior finite-condition commit, including
the 2026-08-28 v5 certificate, is valid for this ABI. Active configs require
new certificate paths and digests explicitly and therefore fail closed until a
new GPU qualification completes. GeneratedCodes v1 and its lookup-table
champion cannot be resumed or used as an incumbent: the old recovery config and
wrapper are explicitly retired, and the first continuous-condition campaign
must cold-start.

The canonical Candidate ABI is the shared two-phase contract:

```python
prepared = prepare_solution(public_case, solve_budget=None)
answer = prepared.evaluate(points_query)
```

During `prepare_solution`, a Candidate may open only the plugin-declared public
session methods: `query_source`, `query_initial_h`, and `query_initial_phi`.
Their complex results are `float64` `[N, 2]` real/imaginary pairs; a scalar
source time is broadcast to the submitted phase-space rows. The separate
`query_completed_macros` session is available only for DEVELOPMENT Cases and
is never part of Official or Terminal truth access. The prepared program's
`evaluate` result contains exactly `particle_density`, `parallel_momentum`,
`kinetic_energy`, and `phi` at the requested points.

`GKWLegacyObservableAdapter` exposes a trusted implementation through both
`predict_observables(case, query, budget=None)` and
`predict_macros(case, query, budget=None)`. Both are SDK/non-adversarial
compatibility paths, not an untrusted-code sandbox.
The historical evaluator remains authoritative for multi-budget ranking, but
its same-process Python session is reflectable and remains a trusted SDK path.
Formal candidate code accesses numerical data only through the shared
controller/IPC boundary.

`GKWMacroMatcher` independently reproduces the stable metric: each complex
field uses weighted relative L2 with public
`abs(B[s] * ints[s] * parseval_ky[ky])` weights, and the four field errors are
averaged equally. Dataset observations and Generator guidance are never macro
targets. Bridge evidence v2 binds coordinates, structured budget, solver and
legacy-evaluation digests, the public packet, quadrature weights, prediction,
truth, every result metric/outcome and accepted-call identity without
persisting private truth or changing winner selection. Solver/evaluation
digests must be supplied by the controller after it verifies the underlying
artifacts; the evidence is a commitment, not a signature or external trust
anchor.

`GKWProblem` reads completed macro releases with schema
`fusion_gkw_released_completed_macro_answer_v1`. It checks the canonical
array/scalar whitelist, exact pinned shapes/dtypes and recursive equation,
macro and oracle schemas. It also checks bounded archive sizes, duplicate JSON
and ZIP entries, all manifest payload sizes and SHA-256 values, case identity
across manifest/public/oracle/answer, and cohort disjointness. Manifest hashes
provide package integrity only; a formal campaign must establish provenance
from controller-owned storage or another external trust anchor.
Public Cases contain immutable GKW equation/oracle metadata plus query times and
`(s,kx,ky)` rows, but never released values, hidden spline controls, dense
phase-space states, Generator latents or private seeds.

Release-backed GRID/POINTS queries use only released coordinates. They are a
valid development/SDK panel, not a replacement for the historical private
formal panel (different time selection and usually different field coverage).

For historical formal reproduction, `bind_gkw_macro_loop` binds the frozen
Generator, native public query and `GKWMacroMatcher` to the legacy
`MacroClosedLoopRunner` also used by old VPFP experiments. It is not the product
entrypoint. New GKW product runs register `GKWObservableTask.create_evaluation()`
and enter the common `CampaignController`; the remaining formal execution
adapter is still migration work. A real release-package regression covers
recursive immutable mappings and read-only arrays through the legacy public-
snapshot boundary.

Generator capability now reflects the injected backend. The built-in
`GKWManufacturedGeneratorBackend` dispatches the shared `BASE` and
`PERTURBED + anchors` requests while keeping clean hidden handles private. It
binds every root and child to the same controller-verified Generator artifact,
requires unique issued anchors, consumes roots only after a successful child,
and binds evaluator snapshots to the trusted issued case by `case_id`, checking
its envelope, small metadata and array headers without rescanning large values.

`GKWSelectedTorchRuntime` is the concrete physical implementation. It accepts
an initialized, frozen `GKWSelectedOnlineGeneratorService` and runs one of two
explicit, mutually exclusive solution-sampling modes. `local_edit` is the
backward-compatible default and requires one
`GKWValidatedProductionQualification` token. Experimental `direct_noise`
requires a `GKWValidatedDirectNoiseCompatibility` token, rejects the production
qualification token, and cannot claim production-certificate eligibility.
Neither mode trains or alters the sampler or decoder. Clean generation uses
the selected sampler and decoder,
constructs the continuous spline state, rebuilds `S_balance = d_t h - R_GKW[h]`,
applies the existing source/hidden-support/off-grid/decoder gates, and contracts
macro truth onto the public panel. The unified runtime holds only the terminal
decoded frame until the decoder report passes, since that frame triggers the full
nonlinear physical audit. Decoder-rejected attempts therefore retry before
unnecessary source/operator evaluation. Decoder-passing attempts still run the
complete source and physical gates, preserving acceptance and retry ordering for
well-formed numerical outputs. Anchored generation uses the clean CPU-resident
controls directly and makes exactly one draw under the selected mode's
authority. Each child receives the exact public PDE packet of its clean parent
(only the child Case id changes), so perturbation never silently rebuilds or
resamples the parent equation. It performs no per-child numerical qualification,
parameter hash, retry, or failed-seed replacement, and never replays the decoder
or writes a hidden artifact. For `local_edit`, startup first validates the
certificate, final report, generator/operator/physical-source manifests, and
seed plan against the actual artifact bytes, runtime policy, checkouts, and
physical source closure. Only a complete 2700-case production campaign can
produce that production token. For `direct_noise`, compatibility validation is
owned by the physical plugin; the runtime requires its validated token and
retains the explicit non-production eligibility flags on every child.
The controller keeps each fresh raw seed in case identity and private audit, maps it
once modulo `2147483647` for the qualified Torch draw, and records both values;
this range conversion is not a retry or failed-seed replacement.

After an audited case is accepted, the runtime captures only its compact CPU
resident answer and opaque controller lineage. It performs no per-case
hidden/public/state hash, private sample save/load, serialized oracle artifact,
or compact/decoder replay. Cohorts are manufactured lazily and release their
residents after the last candidate comparison. Oracle open promotes just the
current resident to a capacity-one GPU backend; source frames use a bounded
shared cache of at least eight entries. Every candidate open still creates an
independent metered session with its own row and unique-time limits. Hidden
controls, operators, and Generator parameters never enter `Case.public_data`.

`private_root` is only a short-lived manufacturing workspace; no
per-case audit or hidden-state artifact is retained there. Each transaction root
is removed on success or failure, and no oracle replay/cache directory is part
of the online contract.

Controller-private telemetry records clean/perturbed generation wall time,
resident/backend releases, and clearly labeled derived clean-frame counts. It
reports zero perturbed online numerical audits, parameter hashes, and failed-seed
replacements, and confirms that the removed unified save/state-hash/replay stack
remains unused. Qualified perturbation telemetry also separates sample,
operator-precompute, macro-truth host dispatch, and macro-to-public-pair wall time.
The internal stage clocks do not synchronize CUDA and are not additive.
CUDA-synchronized smoke sections separately report generation,
first/repeated oracle queries and case-major comparison with RSS, `ru_maxrss`,
`/proc/self/io`, and section-local CUDA peaks. The smoke requires and records the
completed offline qualification certificate; independent reference parity remains
a separate development-stage check.

Manufacturing publication is request-transactional in memory. The runtime keeps
new resident handles pending until the backend validates and atomically commits
the complete batch. Abort drops only those exact pending handles; cohort release
drops committed residents and closes a matching capacity-one backend exactly
once.

The mode and parent-inheritance record are retained as plugin-private immutable
provenance on each resident handle. They are scientific audit data, not a user
hash field and not a new Controller protocol. Both public and DEVELOPMENT
numerical sessions are exposed through `BackendGeneratorProgram`; the shared
Campaign Controller needs no GKW-specific branch.

The controller must compute one digest over the frozen model, calibration,
decoder and physics implementation manifest and pass that same value as both
`implementation_sha256` and `generator_sha256`:

```python
runtime = GKWSelectedTorchRuntime(
    selected_service,
    implementation_sha256=generator_sha256,
    production_qualification=validated_production_qualification,
    private_root=private_case_root,
    private_storage_root=durable_storage_root,
)
generator = GKWAnswerFirstGenerator(
    GKWManufacturedGeneratorBackend(runtime)
)
plugin = bind_gkw_macro_loop(
    generator=generator,
    base_state=None,
    generator_sha256=generator_sha256,
)
legacy_program = GKWLegacyObservableAdapter(
    submission=solver_submission,
    compute_budget=compute_budget,
    runner=runtime_runner,
    public_session_factory=generator.public_session_factory,
)
```

The optional selected-GKW physical package and trained checkpoints remain
experiment inputs rather than shared controller code. This plugin deliberately
does not copy the historical GKW ranking/controller protocol. Formal Agents use
the shared equation-neutral IPC numerical Oracle, sealed multi-file candidate
bundles, phase-blind Cases, and fresh candidate runtimes. Activation remains
fail-closed until the pinned physical commit has a completed production
qualification certificate and the unified GPU runtime smoke passes.
