# Result tables and their interpretation

Use `selection.csv` to identify each round's retained solver and its Official
selection score. `selection_decisions.json` preserves the native promotion
decision, contender eligibility and per-distribution means. `selection_policies.json`
preserves the recorded policy transitions. These are distinct from terminal
and independent evaluation results.

## Native terminal comparison

`native_round_scores.csv` contains 36 round records. `native_case_scores.csv`
contains 2,240 unique program/case records: 35 selected arm/program pairs,
64 common cases each. Strong-Astra R4 uses its R3 solver, so its terminal
result is shared. Original failures and uncertain evidence remain explicit.

`observed_mean` includes the archive's declared supplemental observations.
It is not automatically a valid original-budget result. Read
`official_eligibility`, `official_mean`, retained failure/uncertainty counts,
and `source_view_counts` with it. In this terminal CSV, `official_mean` is the
archive's original-eligibility result; it is not the round's promotion mean.
The promotion mean is `official_selection_mean` in `selection.csv`.

The per-case table retains original and observed errors, the selected attempt
ID, source view, solve-budget hint, guard status and available timing fields.
Missing timing values remain missing. No attempt was chosen by minimum error
when preparing this release.

## Independent validation completed on October 20–21

| Files in `independent/` | Cases | Interpretation |
| --- | ---: | --- |
| `artemis_round-summary.csv`, `artemis_per-case.csv` | Six | S0 and R1–R6 for both models; 78 distinct executions, 84 displayed cells because S0 is shared |
| `vpfp_original_round-summary.csv`, `vpfp_original_per-case.csv` | Eight per regime | Original execution policy, including 11 timeouts; incomplete rounds have no mean |
| `vpfp_completed_round-summary.csv`, `vpfp_completed_per-case.csv` | Same eight per regime | Original successes plus declared supplements for exactly those 11 timeouts; 28 complete round means |
| `vpfp_merge_status.json`, `verification.json` | Same comparisons | Completion, failure-preservation and saved-array verification evidence |

The ARTEMIS reference uses previously unused observation windows
2011-10-23 02:00–03:00 UTC and 2016-07-29 06:00–07:00 UTC. Metadata-only
selection yields three questions per event, each from two distinct profiles.
Gaussian-smoothed velocity histograms define positive continuous profiles;
periodic density and mixture weights supply manufactured spacetime variation.
The source is derived from that field. This is independent profile-derived
MMS validation, not a reconstruction of the observed spacetime trajectory.
Every program uses the same 256 private spatial points at endpoint 0.875.
Bank SHA256: `c663245943b7e736dea634851472719a26b56b1be3790cd4c5f54888f6c4fb7b`.

The VPFP references interpolate existing coarse numerical trajectories using
squared Gaussian velocity RBFs, periodic Fourier space representations and
cubic time splines, with mass normalization. The interpolants define the
hidden answer and manufactured source. They are not claimed to be converged
solutions of the original coarse equations. Full-line third moments extend
these references to the frozen transport score without changing the underlying
distribution or source.

These follow-ups retain a nominal 900-second solver input and a real clock.
ARTEMIS uses a separately recorded 5,400-second operational case wall and an
independently approved preparation policy; it is not the native campaign's
strict shared 900-second acceptance test. VPFP supplements remove only audited
pure stopping guards in memory for Strong-Sol R6 (eight cases) and
Strong-Astra R2 (three cases). The archived source bytes, adaptive numerical
choices, nominal input and real clocks remain unchanged. Original failures
are retained, and supplements are selected by the declared supersession rule,
not their error. `accuracy_source` marks these 11 rows. The VPFP verification
retains 212 attempts: 200 original and 12 supplementary, including one failed
supplementary startup.

Do not combine the independent means with native terminal means, relabel
supplementary accuracy as original-budget success, or use these evaluations
to replace the protocol-selected S6. Field images produced during follow-up
are not included here: a completed dense export does not by itself establish
reproduction of time-adaptive predictions from a different GPU execution.

The CSV files are copied from the verified archives; only the site storage
locations in their path columns were replaced by the placeholders described in
`docs/SITE_CONFIGURATION.md`, and `provenance.json` records both the archived and
the published hashes. This publication checks their identities and scalar
aggregation; the original
postprocessors verified scores against saved prediction arrays on compute
nodes. Original arrays, reference audits and all attempt logs remain at the
locations identified in [provenance.json](../provenance.json).
