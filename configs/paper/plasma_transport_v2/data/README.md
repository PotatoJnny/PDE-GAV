# External scientific assets

The release includes solver source, original configurations, prompts,
calibration and compact result tables. Generator weights, private cases,
execution logs and saved predictions remain in the authors' experiment archive,
written as `${PDE_GAV_ARCHIVE_ROOT}/20261019` in [provenance.json](../provenance.json).
These are locators inside that archive, not paths expected on another machine;
see [site configuration](../../../../docs/SITE_CONFIGURATION.md).

| Asset | Location relative to the archive |
| --- | --- |
| All original program manifests and members | `programs/INDEX.json`, `programs/<arm>/` |
| Final numerical predictions and exact attempt choice | `indexes/terminal-solutions/terminal_solution_index.json` |
| All attempts, including absent predictions | `indexes/terminal-solutions/attempt_solution_index.json` |
| Prediction reader | `indexes/terminal-solutions/READER.md`, `read_terminal_solution.py` |
| Scientific inputs, Generator weights and latent provenance | `indexes/scientific-inventory-*.json`, one per compute site |
| Frozen framework | `sources/frozen-framework-source/framework-source/` |
| Original object bytes | `objects/<sha256>` |

Saved terminal solutions contain 256 query coordinates and macroscopic values
per program/case, not full spacetime or phase-space trajectories. The release's
`solutions/` directory contains reusable solver programs. The indices above
locate their numerical predictions; no unsafe pickle loading is needed to
inspect their metadata.

Independent validation uses the source and sealed plans under
`${PDE_GAV_SOURCE_ROOT}/experiments/kinetic-followup-20261019/`. Its completed
outputs are under `${PDE_GAV_RUN_ROOT}/`:

- `artemis-independent-pilot-6487632/`: the sealed six-case bank and S0/R6.
- `artemis-independent-rounds-6496845/`: R1–R5 and the complete VP comparison.
- `kinetic-followup-postprocess-6464177/`: original VPFP comparison.
- `kinetic-followup-relaxed-postprocess-6485371/`: verified VPFP supplements.

[provenance.json](../provenance.json) records exact source-file hashes for
the published tables and configurations, together with the hashes of the
published copies. Archive and run-root paths identify existing evidence; they
are not a promise of long-term public hosting. The transfer tools under
`tools/` build sealed manifests for moving such assets between sites. The
GitHub package alone does not include the weights and private references
needed for a fresh full run.
