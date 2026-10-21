# Artemis canonical integration

The completed six-round transport-v2 experiment and all archived Agent solver
bundles are published in the [VP / VPFP experiment package](../../../../configs/paper/plasma_transport_v2/README.md).
That package records the effective protocol, final S6 programs, native terminal
results and independent profile-derived validation through 2026-10-21.

A researcher can now bind the admitted compact-v5 Artemis Generator, submit a
macro-only Candidate, and run the shared CampaignController through development,
common Official evaluation, promotion, memory and cross-round Terminal reports.
The Task requires only `M0`, `M1`, `M2`, `M3`, and `E`; no `f(t,x,v)` output or
kinetic internal representation is required.

`build_artemis_plugin()` supports registry discovery without loading Torch or
physical assets. `load_artemis_backend(...)` returns the shared
`BackendRunBinding`; its `controller()` and `run()` methods accept a canonical
CampaignRecipe and Agent programs. Formal runs supply the shared subprocess
Agent execution service. The explicit `local_development=True` setting enables
small in-process interface checks.

## Existing assets

The physical implementation is loaded lazily from an existing checkout pinned
to `7c22b7f1dbf46388b114ca274c0e1fec2a5b0a36`. Its module bytes are verified
against that Git revision and imported under a private namespace, without
executing its broad package initializer or old Controller. The canonical
package contains the small public equation/operator/matcher contract only.

```python
from pde_agent_verifier_solver.plugins.vp_1d1v import load_artemis_backend

binding = load_artemis_backend(
    physical_checkout=artemis_checkout,
    admission_record_path=admission_path,
    expected_admission_record_sha256=admission_sha256,
    science_config_path=science_config_path,
    expected_science_config_sha256=science_config_sha256,
    device="cuda",
)
result = binding.run(recipe, agents, agent_execution=execution_service)
```

`preflight_artemis_backend` authenticates the same source/admission/science
bindings without loading the physical model or generating cases. Loading uses
the existing admission and checkpoint; it does not train a replacement model.
PreparedDataset is empty with `online_only=True`, because fresh manufactured
cases come from the frozen Generator, not from a release-bank manifest.

The cluster binding in `configs/plasma_canonical_artemis.json` was preflighted
on 2026-09-04: admission `7b036e6c...`, science config `8f16ab9a...`, admitted
checkpoint `37922337...`, and the bound V6 report `5cc9760e...` all verified.
This is existing-asset authentication, not a new real-model Campaign run.

## Candidate and oracle semantics

The native Observable Task accepts
`predict_observables(case, query, budget=None) -> AnswerView`. Existing
`predict_macros` programs can be wrapped explicitly in
`LegacyMacroCandidateAdapter`. A program with
`prepare_solution(case, solve_budget=None)` is called before scoring coordinates
are revealed; its handle then answers `evaluate(query)`. Setting
`task_settings["require_prepared_solution"]=True` requires this stronger timing
contract. One-shot predictions do not claim query-blind preparation.

Causal evaluation is enabled by default. Every scored point lies at one
non-initial native time; the Candidate receives a detached source prefix and
resident oracle calls reject times beyond that horizon. Exact-coordinate
oracles remain exact evaluations, while cached-array oracles retain their
declared interpolation semantics. Development completed-solution sessions are
available only when explicitly enabled by the common evaluator policy.
Sessions are revoked after each call and resident cases are released by the
common Generator lifecycle. Native scoring uses at most `nx` points without
replacement; set development/official point counts accordingly.

Artemis plans use the fresh seed supplied by the shared verifier. Coupled
clean/perturbed coordinate draws require its paired-plan option; the plugin
does not cache a first seed or infer a cohort from call order.

## Optional guidance and perturbation

The base factory exposes the admitted BASE + paired PERTURBED backend.
Passing `reward_guidance_settings` explicitly binds `ArtemisRewardGuidance`,
which reuses the pinned reward-model fitting and bounded primitive transport.
Canonical selected-Solver feedback can come from Official/PROMOTION roots;
the adapter retains only three scalar primitive coordinates for each active
root. It never copies kinetic arrays into the reward model. A guidance fit
requires at least two valid feedback roots in every admitted condition, and
fails explicitly when that data is missing.

The condition stream is unchanged by guidance. A zero/no-op guidance update
reuses the exact original quantized capsule, avoiding inverse-CDF round-trip
drift. Guided cases and perturbations are still manufactured and admitted by
the existing physical source. The current reward-guidance science result is
`insufficient_evidence` for perturbation hardness; this integration exposes the
arm as experimental and makes no new qualification claim. The default factory
does not silently reuse a reward checkpoint trained against another Solver.

The existing sensor-bin perturbation mathematics is preserved. Canonical
integration itself does not establish that it is the same scientific
perturbation family as NS; the shared perturbation-policy work must record and
qualify any changed geometry/radius/constraints separately.

## Validation

`tests/test_artemis_canonical.py` covers macro-only matching, causal source and
oracle bounds, session revocation, checkpoint binding, BASE/PERTURBED lineage,
prepared Candidates, and a complete two-round canonical Campaign with memory,
promotion and Terminal evaluation. A small optional test uses the pinned
reward-model mathematics with synthetic scalar feedback to verify the adapter
and exact zero-guidance behavior. These are interface/numerical checks, not a
new physical model qualification or a real-data validation claim.
