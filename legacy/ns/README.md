# Bundled NS implementation

This directory contains the historical structured Navier--Stokes experiment code retained for explicit history/control-arm reproduction. It is intentionally kept behind the legacy staged adapter: reproduce one of these campaigns from the repository root with `./pde legacy-run <config>`, rather than calling these scripts one by one. Canonical product runs use `CampaignController`.

The legacy staged integration is defined in:

- `src/pde_agent_verifier_solver/plugins/ns/campaign.py` — translates one NS config into the full run;
- `configs/ns_structured_5round.json` — the current formal five-round experiment;
- `configs/ns_structured_smoke.json` — a very small end-to-end integration run.

The bundled scientific path is:

1. map training, validation, and held-out numerical fields to structured continuous coefficients;
2. fit the PCA basis on the training/validation data;
3. train the base rectified-flow Generator;
4. create an independent hidden set;
5. run generated, guided, perturbed, replayed Agent rounds;
6. publish the selected `submission.py` solver and the NS reports.

When adapting or improving the NS methods, keep the customer command and shared staged-run experience stable. NS-specific numerical choices stay inside this directory and the NS plugin.
