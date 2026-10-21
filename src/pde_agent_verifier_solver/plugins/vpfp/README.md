# VPFP reference plugin

The completed strong/weak six-round transport-v2 experiments and all archived
Agent solver bundles are published in the [VP / VPFP experiment package](../../../../configs/paper/plasma_transport_v2/README.md).
It records the effective protocol, final S6 programs, native terminal results
and independent eight-case comparisons, with original failures and declared
supplements kept distinct.

This folder shows how one PDE can support several independent scientific tasks.

- `problem.py` reads continuous answer-first VPFP cases and exposes only finite density, momentum and kinetic-energy truth queries.
- `generator.py` describes the positive Fourier--Hermite Generator interface. Perturbation capability is advertised only when the injected backend explicitly supports it.
- `perturbation_backend.py` binds that interface directly to the existing Fourier--Hermite perturbation, source reconstruction, and case verification kernel.
- `legacy_adapter.py` exposes a trusted historical `solve_with_budget` implementation through the unified Observable Model ABI.
- `macro_matching.py` independently matches Candidate density, vector momentum and kinetic energy against controller-private full-velocity truth; dataset observations and Generator guidance are not targets.
- `legacy_macro_bridge.py` evaluates legacy prediction/truth arrays with that matcher while binding the exact formal coordinates.
- `macro_contract_audit.py` verifies the additive unified evidence produced by formal evaluation before publication.
- `tasks/observable.py` asks only for macroscopic fields and does not constrain the internal method.
- `tasks/reduced_dynamics.py` requires an explicit reduced state, reduced evolution and observation map.
- `tasks/statistics.py` asks for named scalar statistics.
- `product.py` registers all three choices against the same Problem and Generator.
- `campaign.py` preserves the legacy paper route for explicit control-arm reproduction; canonical product runs use the registered Tasks and `CampaignController`.

`VPFPAnchoredPerturbationBackend` now implements the shared `BASE` and anchored
`PERTURBED` interface. It owns only deterministic seed derivation, exact clean
anchor identity, Generator provenance, and controller-private latent handles.
This backend explicitly disables guidance because its runtime is frozen; older
VPFP backends that do not declare `supports_guidance` keep the historical
guidance-enabled default. Clean anchors are consumed only after every child in
the request passes the exact v2 schema and all 12 physical gates.
`VPFPLegacyStructuredRuntime` owns the equation-specific work: it reuses the
unchanged base Generator, embeds its answer into the expanded representation,
calls the vendored robust calibration kernel, rebuilds the nonlinear public
source and hidden macro truth, and runs `verify_continuous_case_dir`. Every
child must pass all existing positivity, smoothness, displacement, initial
condition, exact full-velocity normalization, and source-increment gates. The
audit is stored beside the controller-private case and is not added to
Agent-visible `Case.public_data`.

Legacy disk publication is also one batch transaction: all deterministic case
targets are preflighted before generation, every newly published target remains
rollback-owned until the backend has validated the full request, and issued IDs
are recorded only at commit. A failed later case therefore leaves anchors and
pre-existing directories untouched and permits the same request to retry.

The vendored physical bundle is fail-closed to three regular, non-symlink
modules from one resolved package directory: `robust_perturbation.py`
(SHA-256
`38e27d17246e582616f0d73c324ff35d8d68901f8c0dad6815333aace2ad37b6`),
`analytic_source.py`
(`c44daa8d87c6a3ff019521dc650798afd88e8c43d6e255b5e487b3873d8734b0`),
and `continuous_cases.py`
(`3e637e4239574c2b82d1b3487cce81c5958a7b830b070fbb06628dfd8ae7ef9d`).
The perturbation report schema remains `vpfp_structured_answer_perturbation_v2`.
New density-mode groups follow seeded random cubic amplitude/phase paths that
vanish with zero derivative only at the initial anchor and remain nonzero in
the open time interval. Continuous-case verification also rematerializes
`rho0` and `g0` from the primary hidden answer and hard-gates their agreement
with the public initial arrays. An already-qualified legacy runtime can be
bound without changing its Generator:

```python
runtime = VPFPLegacyStructuredRuntime.from_development_runtime(
    legacy_runtime, output_root=private_case_root
)
generator = VPFPContinuousGenerator(VPFPAnchoredPerturbationBackend(runtime))
plugin = bind_vpfp_macro_loop(
    generator=generator, base_state=None, generator_sha256=generator_sha256
)
```

Materialization and verification occur inside controller-private staging
before transaction commit. The manifest hash inventory detects corruption in
that trusted staging flow; it is not an external signature or trust root for a
hostile mutable artifact directory.

The checked-in release-bank source and fresh-U0 reference remain deliberately
base-only because replayed cases do not carry live latent handles. A fresh
paired run additionally requires the qualified expanded decoder checkpoint and
a Torch runtime; these are experiment inputs, not shared controller policy.

The checked-in VPFP staged paper campaign accepts only `observable_model`. The reduced-dynamics and statistics Task SDK contracts remain available for separate experiments, but no checked-in reduced/statistics staged campaign or promotion route is exposed.

`VPFPLegacyObservableAdapter(submission, compute_budget, runner)` maps the historical `solve_with_budget(public_case, query_times, budget)` call to `predict_observables(case, query, budget=None) -> AnswerView`. It preserves native `x/y` coordinates and the complete structured VPFP compute budget, accepts only the full supported macro `GRID` query, and fails closed on `POINTS`, spatial resampling, or a float budget override. This is a trusted in-process compatibility surface; formal evaluation continues to execute untrusted Candidates in the isolated legacy subprocess.

The adapter requires the exact historical macro-output contract. If a release-bank v2 question predates the compute-budget marker, the adapter adds that marker only to its deep-copied runner question after validating the complete structured budget; an explicit wrong marker is rejected. A guarded submission proxy also rejects object, complex, or otherwise non-real raw arrays before the legacy runner can coerce their dtype.

`vpfp_full_velocity_macro_match_v1` weights the three physical groups rho, two-component vector momentum, and kinetic energy equally. The Task setup and result cards advertise `vpfp_macro_match_v1_relative_l2`; the unversioned `macro_relative_l2` remains a compatibility alias. The pre-matcher four-scalar average is retained as `legacy_four_scalar_macro_relative_l2` for migration diagnostics, but it is not the v1 primary score.

For every accepted formal solver call, the legacy runtime feeds the same Candidate prediction arrays and the same controller-private continuous truth to both the official legacy scorer and the additive unified matcher. The resulting `unified_macro_match` is stored as coordinate-bound dual-score evidence with `used_for_winner_selection: false`. The legacy density/kinetic-energy positivity penalties, early/middle/late time-band scores, and robust winner ranking remain authoritative. Before `publish_result`, the `validate_macro_contract` stage checks complete accepted-call coverage, exact per-call budget identities and stage schedules, case/shape/coordinate bindings, and matcher-metric coherence; it does not re-rank the finalists.

Candidate coordinate metadata is part of this contract, rather than an implicit controller detail. For regular SDK `GRID` queries, candidates return fields flattened in `(t, x, y)` order, the matching shape, and exact axes derived from the public `case.public_data["arrays"]` coordinates: keep a native axis when its requested count is omitted or unchanged, use its endpoint midpoint for count 1, and otherwise use inclusive linear spacing between its public endpoints. For matcher `POINTS` queries, candidates copy `query.points` exactly. The matcher rejects values attached to different coordinates even when their shapes agree.

Formal evaluation is deliberately still different: its deterministic jittered `query_times` remain explicit inputs to the legacy runner, and `legacy_macro_bridge.py` binds those exact times plus native `x/y` to the dual-score artifact. The generic `AnswerQuery` does not yet represent an arbitrary explicit time axis, so the adapter and bridge must not be interpreted as a complete migration of formal jittered axes.

Both checked-in campaign templates resolve these five user-provided `VPFP_*` paths before planning or running:

```bash
export VPFP_RELEASE_BANK_REPORT=/absolute/path/to/release_bank_report.json
export VPFP_RUN_STORAGE_ROOT=/absolute/path/to/run-storage
export VPFP_VIRTUAL_ENV=/absolute/path/to/venv
export VPFP_BASE_DECODER_CHECKPOINT=/absolute/path/to/base_decoder.pt
export VPFP_EXPANDED_DECODER_CHECKPOINT=/absolute/path/to/expanded_decoder.pt
```
