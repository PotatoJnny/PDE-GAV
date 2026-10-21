# VPFP scientific implementation

This directory contains the numerical implementation used by the VPFP plugin.
Historical/control-arm reproduction is launched explicitly from the repository root:

```bash
./pde legacy-plan configs/vpfp_long_time_robust_smoke.json
./pde legacy-run configs/vpfp_long_time_robust_smoke.json
```

The shared product code lives under
`src/pde_agent_verifier_solver/plugins/vpfp/`.  Read that smaller directory to
learn how a Problem, Generator, and several Tasks are registered.  The files
here are the established Fourier--Hermite Generator, online Evaluator, and
two-Agent experiment implementation; they are kept behind the legacy staged
adapter so future PDEs do not need to understand VPFP internals. Canonical
product runs use `CampaignController`.

Only scripts needed by the one-click long-time experiment are included.  The
full historical repository remains available in Git history.
