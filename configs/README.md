# Configuration entry points and timing conventions

## Published VP / VPFP experiments

[paper/plasma_transport_v2](paper/plasma_transport_v2/README.md) records the
completed seed-101, six-round ARTEMIS / strong VPFP / weak VPFP experiments.
It includes original configurations, the effective protocol and historical
exceptions, Agent prompts, all 77 solver bundles, and results through
2026-10-21. The recorded configurations keep their original resource layout
with site placeholders; they are not portable launch templates. Use the
package's protocol and source provenance when reproducing them.

## NS examples

`ns_canonical_structured_5round.json` and `ns_canonical_piv_5round.json` are
retained canonical run configurations with historical parameters, not confirmed
new main-experiment protocols. The first solves the structured synthetic NS
problem; the second is a sparse PIV reconstruction. The measured-data NS solver
used in the paper (`experimental_mms`) is a different NS solving task.

Formal run configurations bind their resources explicitly in an experiment
directory outside the repository. The case counts, selection rules, seeds and
resource placeholders in the examples here cannot be taken as a new protocol;
see [site configuration](../docs/SITE_CONFIGURATION.md) for the placeholders.

The historical paper configurations give each Agent a full four-hour wall clock
per round (14,400 seconds), including Generator work, evaluation and waiting.
That convention remains available. Queue state is not recorded here.

| File type | Scope |
|---|---|
| `ns_canonical_*_5round.json` | Configuration examples: two Agents, five rounds, four-hour full wall clock by default. Task, case counts, selection rules and resource bindings must be fixed per experiment |
| `paper/ns_draft_v1/{fixed,base,full}_s101.json` | Historical paper configurations; the archived run configurations are authoritative for completed experiments |
| `*_smoke.json`, `*_rehearsal.json`, `*_pilot.json` | Short checks or historical trial runs, not formal paper budgets |
| `unified_plasma_agent_iteration_v1` / `controller_policy` configurations (existing VPFP/GKW and old NS unified configs) | Historical Macro-protocol or collaborator adapter sources, retained for traceability; not canonical paper configurations |

New formal VPFP/GKW configurations are frozen separately once the matching task
adapters are complete. Old `active`/`soft`/`hard` fields do not prove that a
configuration uses the current timing rules. Check the actual reader, the SDK
protocol and the four-hour timing tests before launching.

The structured NS template and the paper Base/Full configurations select the
accelerated sampler in the NS plugin with
`generator.runtime.cuda_graph_sampling=true`. The SDK default is off. The
implementation runs only inside an independent Controller process, keeps FP32,
256 Heun steps and the original Guidance mathematics; CPU, non-builtin models
and unsupported execution modes fall back to ordinary inference. Graph
preparation per sampling call counts toward the original four-hour budget.
