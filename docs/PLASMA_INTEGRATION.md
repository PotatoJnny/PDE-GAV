# Plasma GPU migration

This checkout is the development home for the strong VPFP, weak VPFP and
Artemis GPU experiments, which were first developed in a separate collaborator
repository. The October 2 frozen deployments use an earlier integration commit;
development has since merged the shared primary-selection and parallel-evaluation
protocol (October 3). The frozen queued jobs have not been edited, cancelled or
resubmitted by that merge.

## Integration boundary

Keep PDE-GAV's single Controller, modular prompts, per-case development
feedback, candidate diagnostics, cancellation and original-deadline recovery.
Retain the accepted physical backends without replacing newer shared services.
Explicit primary selection now uses upstream's single implementation, including
`select-primary`; the earlier private policy helper is removed. The Plasma
composition translates its recorded policy to upstream's worker setting.
Weights, scientific evidence, historical
results retain their original identities and locations.
Importing source does not validate a new CUDA deployment or resume an old run.

## Experimental settings

The existing six arms are strong VPFP, weak VPFP and Artemis, each with
`gpt-5.6-sol` / `max` and `gpt-6-astra` / `ultra`. Preserve five rounds, seed 101,
two Agents, four isolated GPUs, 14400-second Agent wall time and one 900-second
charged Case budget with no additional preparation cap. Official evaluation
uses 86 Base, 85 Guided and 85 Base-perturbed cases, with an explicit primary
from each Agent and the incumbent. Historical v1 keeps each PDE's own scoring
and initial program. The separately versioned transport-v2 experiments below
change the scoring target and VPFP S0, without importing old scores or Guidance.
Perturbations, model assets and the reporting-only Terminal suite remain fixed.

The historical six-arm development allowance is 32 cases including four
starters. New main supports unlimited sampling, but that capability is not
permission to silently change this experimental parameter. These migrated
configurations preserve 32 until a new experimental choice is made.

## Shared launch and continuation

`scripts/prepare_plasma_gpu_migration.py` imports the six approved configurations
from a frozen, clean historical source plan. It checks the recorded source
revisions, input hashes and initial program identities before writing any new
configuration. Weak external bindings can add only the portable Base and its
historical evidence locators; they cannot override scientific settings.

`scripts/plasma_gpu_campaign.py` composes all three plugins with the current
shared Controller, Candidate execution service, native Agent worker and
CheckpointStore. The shared `scripts/canonical_codex.config.toml` includes the
conditional restricted-namespace editing notice; this is not a private
per-equation configuration. The current native GPU qualification gate
still applies to the exact runtime and shared configuration.

`scripts/submit_plasma_gpu_batch.py prepare` produces six pinned site records,
Slurm wrappers and a source manifest from a clean committed checkout. Its
`submit` subcommand prevalidates all six deployments and durably records each
acknowledged job ID. An uncertain submission stops without retrying. It never
cancels old jobs. Replace old pending jobs only after independently verifying
the new submissions, and use a pending-state cancellation guard so a job that
has started is not accidentally stopped.

Each allocation uses four GPUs: Agents on devices 0/1, the development Oracle
on 2 and Candidate programs on 3. Hidden evaluation can reuse 2/0/1 after the
Agents stop; Candidate device 3 remains separate. Each run requests 24 CPUs;
VPFP uses 128 GiB and 12-hour segments, Artemis 64 GiB and 6-hour segments.
These are the existing resource profiles, not extra evaluation allocations.
New deployments select `gpubase_l40s_b2,gpubase_l40s_b3` for both the initial
allocation and automatic successors. The versioned `plasma_gpu_12h_b2_b3` and
`plasma_gpu_6h_b2_b3` profiles retain the same resources and original time limits.
The runner selects the continuation profile from the pinned site partition;
legacy b2-only sites and profiles keep their previous identity. Expanding an
already-pending frozen job through Slurm does not modify its source or make its
legacy successors automatically use both partitions.

`scripts/run_plasma_gpu_segment.py` verifies the source/config/runtime hashes
and four distinct physical GPU identities before constructing the Campaign.
On the first allocation, `scripts/check_plasma_gpu_integration.py` generates
one non-official Base case and checks the source, initial values and continuous
macros. Weak VPFP additionally compares accelerated source and g0 paths with
independent native calculations using numerical tolerances, not bitwise
identity. This is a small interface qualification, not a new model audit.

Clean saved Campaign boundaries automatically continue through the shared
`ns_slurm_continuation.py` helper with an `afterok` dependency, unchanged resource
profile and a 12-segment safety cap. A failed Agent, failed cleanup, changed
deployment, no-progress checkpoint or ambiguous submission does not receive a
fresh retry or Agent budget. New campaigns start from the original S0 and empty
memory; historical campaign state is not imported.

CPU tests and asset checks do not constitute CUDA qualification. Each deployment
records these separately in its migration, allocation, binding and native Agent
receipts. A successful Slurm submission likewise means queued or running, not
that the scientific experiment has passed.

## October 3 parallel execution compatibility

The shared fair Candidate and Oracle queues are integrated, along with explicit
`gpu_slots_per_device`. Setting it to 2 can run two ordinary solvers on the
dedicated Candidate GPU; the other three GPU assignments are unchanged. New
migration output explicitly defaults to 1 slot, preserving the previous setting
until a new run chooses otherwise. The constructor, Agent capability description,
run provenance and shared panel now receive the same configured value.

Upstream `oracle_max_workers` limits the whole service, whereas the old Plasma
value applied per host. New migration output records that conversion and uses
global limits of 12 for strong VPFP, 8 for weak VPFP and 3 for Artemis, bounded by
their existing worker counts. Weak development no longer mutates a visible
attribute after the shared admission queue has been constructed. This changes
only service wiring; its source hash is updated explicitly. Models, manufactured
source formulas, perturbations and scientific assets are unchanged.

Batch preparation is not yet integrated into the Plasma Tasks. Their causal
information views and per-case public sessions need dedicated adapters; simply
copying the NS implementation would be unsafe. Plasma therefore rejects
`solver_batch_size > 1` instead of claiming a disabled feature is active. It also
retains `oracle_concurrency_per_device=1`: independent Plasma numerical processes
are not NS's audited concurrent CUDA streams. The strong 8 MiB transport and
three Plasma Oracle pools remain in place.

The new generic GPU protocol permits configurable time budgets. This named
Plasma experiment still explicitly validates 900-second Cases and 14400-second
Agent turns, with no preparation sublimit. Upstream primary selection decodes
only the selected version and treats saved submissions without a primary as a
protocol error; it does not fall back to the last archived submission.

These changes have CPU/subprocess checks, not a new Plasma multi-slot GPU
throughput qualification. The NS benchmark's speedup is not a measured Plasma
speedup. The October 2 six jobs still use their original one-slot frozen source.

## Shared Plasma macro transport v2

New transport experiments explicitly use `pde_gav_plasma_gpu_v2` and
`metric_profile: plasma_macro_transport_v2`. Historical configurations with no
profile retain v1; they must not be silently resumed with v2 scores. The same
frozen calibration and semantic objective hash bind the Task, development and
official scores, promotion labels, Guidance and resume state in all three
families. Configurations with conflicting identities fail before model loading.

The case score is `(e_n+e_E)/8 + (e_j+e_P+e_qh)/4`, lower is better. Each error
is the all-query RMS norm error divided by `sqrt(reference_RMS^2+s_A^2)`.
Vectors use Euclidean norms and pressure uses its full Frobenius norm:
`Pxy` contributes twice in 2V. There is no pointwise division by a small true
value and no score-dependent adjustment of group weights. Failed cases retain
the shared formal ineligibility rule, not a successful-subset promotion mean.

VPFP v2 requests ten fields: `rho`, `electric_x`, `electric_y`, `momentum_x`,
`momentum_y`, `pressure_xx`, `pressure_xy`, `pressure_yy`, `heat_flux_x`,
`heat_flux_y`. These are number density, acceleration-convention electric
field, particle flux, **central** pressure and **central** heat flux at unit
particle mass. Artemis keeps its `M0,M1,M2,M3,E` candidate ABI and converts each
prediction/reference using its own `u=M1/M0`:
`P=M2-M1^2/M0`, `qh=(M3-3*u*M2+2*M0*u^3)/2`.
Nonpositive density and nonfinite values are invalid, never clipped in scoring.

The 2V oracle computes all four raw third moments, including `xxy` and `xyy`,
by sparse Hermite recurrence with sufficient padding beyond the retained basis.
The weak moving frame pushes the complete central third-order tensor through
its affine map before contracting it to heat flux. It includes the entire
Gaussian floor. There is no fixed velocity-grid truncation, reprojection or
third-moment source JVP. Old macro/source APIs remain available; v2 obtains all
its truth fields together through a values-only transport query.

Calibration is independent of solver predictions. The frozen rule is the RMS
of reference group norms after explicit nondimensionalization (`n0,v0,E0`):
equal cases within condition strata, equal strata within a family, then equal
weights across strong VPFP, weak VPFP and Artemis. Flux, pressure and heat-flux
units are `n0*v0`, `n0*v0^2`, `n0*v0^3`. Current physical coordinates use explicit
unit scales of one. The rule is a transparent benchmark convention, not a claim
that this is the uniquely physical floor. A zero or unresolved channel is a
calibration failure; there is no epsilon or solver-score-tuned fallback.

`scripts/calibrate_plasma_transport.py` collects independent Base roots on a
compute allocation, using 64 strong, 64 weak and 32 Artemis cases, each with
128 independently drawn uniform spacetime points. Separate processes isolate
the legacy physical packages. Reference values, conditions, root provenance,
failures and timings are retained privately. A complete set is required before
the `fit` command emits a checked, frozen shared asset. Public prompts expose
definitions, units, floors and objective identity, not reference root records.

`scripts/check_plasma_transport.py` uses real model assets, paired Base/perturbed
cases, the shared public-session bridge and a subprocess S0 to qualify the
continuous field ABI and score/feedback identity. It reuses the existing
tolerance-based source readiness check; it is not another distribution audit,
LLM run or Guidance-fit experiment. CPU algebra, configuration and mocked Task
tests are reported separately from this CUDA integration receipt.

VPFP's new simple S0 samples public `rho0` on a fixed 32x32 periodic mesh, solves
Poisson once, holds density/field in time, and explicitly assumes `j=qh=0`,
`P=rho*T*I`. It is a stable local-Maxwellian closure, not an accurate kinetic
solver. Artemis retains its public-initial-macro baseline. Both model arms use
identical S0s per equation. The single shared 900-second Case limit is unchanged.

After real calibration and integration pass,
`scripts/prepare_plasma_transport_v2.py --source-directory OLD_CONFIGS
--calibration CALIBRATION --validation-root VALIDATION_ROOT
--output NEW_CONFIGS --run-tag NEW_TAG` creates six
new configurations and rewritten, consistent Agent instructions. The existing
frozen-source submission workflow then applies. This preparation script cannot
submit or cancel jobs. Both preparation and static deployment checks require
passed receipts for the same executable source, calibration and baseline;
calibration success alone cannot authorize a formal run. Accept and verify all replacements before canceling only
the exact old jobs that are still pending; preserve their frozen artifacts.

## October 5 shared-service integration

The shared candidate service now combines Plasma's inline/trusted-result
transport with NS bulk queries, timing traces, and optional Agent-owned
GPU routing. Bulk result decoding uses the same numerical-result policy as
single-case decoding. These capabilities do not change the six Plasma
configurations: their candidate placement and single-case Tasks remain explicit.

The live weak-source check verifies available modules and inference interfaces
without requiring every maintained source file to match historical bytes.
Historical asset records remain available; numerical acceptance must come from
the family-specific integration tests, not a claim of unchanged source text.
Existing frozen deployments are not rewritten by this code merge.
