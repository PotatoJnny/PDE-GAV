artemis / plasma_macro_transport_v2: Independently diagnose the incumbent and test improvements in stability, quadrature, conservation and generalization.

This is a fresh seed-101 six-round experiment with a deliberately simple S0 and empty Campaign Memory.
Do not import previous workspaces, candidates, scores, hidden cases or Guidance. Only use staged public
resources, the incumbent, Campaign Memory and approved tools. Predict macros, not the full distribution.
Return M0, M1, M2, M3 and E. These are raw velocity moments; the evaluator converts each answer's own moments into central pressure and heat flux. M3 itself is NOT the scored heat flux.
 Scoring uses plasma_macro_transport_v2: S=(e_n+e_E)/8+(e_j+e_P+e_qh)/4, lower is better. Each e_A is the RMS error over all scoring points divided by sqrt(reference RMS squared + frozen scale floor squared), never a pointwise relative error. Vector norms are Euclidean; pressure uses the full Frobenius norm, so Pxy counts twice in 2D. P is central pressure and q_h=0.5*integral |v-u|^2*(v-u)*f dv; raw third moments are not heat flux. Prediction and reference are centered using their own velocity. Density/field, particle transport, stress and heat transport each receive 25%. Failures retain the formal ineligibility rule; successful-subset averages cannot promote a failed candidate. Frozen score calibration: {"floors":{"density":1.0041253663782175,"electric_field":0.0698185715388132,"heat_flux":0.9900774471118267,"particle_flux":0.4788345668393978,"pressure":2.930763214320111},"protocol_id":"plasma_macro_transport_v2","units":{"density":1.0,"electric_field":1.0,"velocity":1.0}}. Objective SHA256=855cf8984dc60dee1d654fa3f0e2e2afbd184a23b8608f3181456077f51e14ae.

Your Agent has 10800 seconds TOTAL elapsed time per round, including tool calls and Controller waits.
The adaptive allowance is 32 fresh Development Cases INCLUDING four initial BASE cases and every
perturbation child. Base/Guided use is optional; no Development mix is forced. Hidden truth is unavailable.
Development scores use 128 private points; Official/Terminal use 256. Respect each case's public time
domain and causal source access. Scoring coordinates are not available during preparation.

Export prepare_solution(case, solve_budget=None), returning a handle with evaluate(points_query)
-> AnswerView. A Case has ONE shared 900 charged-second budget for startup/imports, preparation,
evaluation and serialization; NO separate preparation cap. solve_budget is remaining time, not a fresh
allowance. Only Controller-metered Oracle queue/shared-lock waiting is excluded. Oracle numerical work,
transfer, and your solver work are charged. Use the SDK solve-budget clock and leave time to return outputs.

You have a dedicated GPU and CPU resources. GPU-native computation is permitted, not mandatory.
Candidate evaluation uses its assigned GPU. Follow the staged SDK public Oracle interface; do not assume
Oracle tensors or CUDA contexts share your process. You may train models and include checkpoints/data
needed at inference in your candidate bundle manifest; keep imports and asset loading within the Case
budget. Never include credentials, hidden answers or dependencies on files outside the submitted bundle.

Every Official contender is scored on the SAME 48-case subset of the sealed round bank: 16 BASE,
16 GUIDED and 16 BASE-perturbed children of those BASE parents. Existing completed scores are reused.
After selection, only the selected program receives the additional 16 BASE evaluations needed for 32 Guidance feedback rows in total; those extra rows do not change promotion. Promotion minimizes the case-weighted overall mean of the transport score.
Failures cannot be removed to improve that mean; successful-subset diagnostics do not determine promotion.
The incumbent is evaluated separately. Only ONE explicitly selected primary from each Agent enters Official.
Archive alternatives if useful, but do not submit the incumbent just to retain it.

Use python -m pde_agent_verifier_solver.engine.agent_tool submit candidate_bundle --sequence 1
--name NAME --summary EVIDENCE --usage 'prepare_solution then evaluate' --primary (one shell command).
All submissions share contiguous sequence numbers starting at 1. A successful --primary updates your
immutable primary; plain submit only archives. Re-select an older alternative by a new primary submission.
A rejected pre-commit update leaves the old primary intact: fix and retry the same next sequence.
Verify receipts; edits or chat statements do not change a frozen submission. No primary means no contender.
Record evidence, selected primary and known limitations using agent_tool log NOTES.md before finishing.

Guidance uses this SAME transport objective and the incumbent's first 32 clean BASE feedback rows in order;
the fitted snapshot is shared/frozen for the round and the Base generator stays fixed. Terminal uses common
S0/G1 cohorts: 16 BASE+16 BASE-perturbed, 16 G1-GUIDED+16 G1-GUIDED-perturbed. Deliver S6;
Terminal is diagnostic, not a mechanism to select a solver retrospectively.
