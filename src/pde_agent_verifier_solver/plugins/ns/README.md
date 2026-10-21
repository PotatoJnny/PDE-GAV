# NS reference plugin

This folder is the reference for a full-field solver task built from a numerical PDE dataset.

- `problem.py` reads the `ns_d0_v1` manifest, builds public flow Cases, and keeps velocity/pressure answers inside `NSFieldTruth`.
- `generator.py` describes the coefficient-basis → PCA → Rectified Flow Generator used by the paper experiment.
- `tasks/solver.py` defines the candidate call `solve(case, query, budget=None)` and the velocity-field error.
- `product.py` is the short file that joins the Problem, Generator, Task and optional long-run adapter.
- `campaign.py` preserves the mature NS staged scripts for explicit legacy/control-arm reproduction; it is not a second product pipeline.
- `publish.py` turns the selected solver and scientific summaries into the common Result view.

The `agent` section of a Run may also set `prompt_file`. Its researcher-written scientific objective is placed before the stable NS solver interface, so prompts can change without copying or editing the Agent loop. Leaving it out reproduces the historical NS prompt.

For a new PDE with a full-state solver goal, read the first four files in that order. Copy scientific ideas from `campaign.py` only when the new experiment really has the same long-running stages.
