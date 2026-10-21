# VPFP canonical binding and learned-model transition

`build_vpfp_online_binding` accepts the `vpfp` section in either existing
`configs/vpfp_unified_live_weak_collision.json` or
`configs/vpfp_unified_live_strong_collision.json`. It returns the shared
`BackendRunBinding`, which compiles and runs the ordinary `CampaignController`.
The old script name is not used as the execution path. The two checkpoint
SHA256s and pinned physical module identities are checked before construction.
The runtime keeps continuous answers resident and exposes public source,
initial density and initial conditional velocity-density oracles.

```python
binding = build_vpfp_online_binding(
    json.loads(Path(config).read_text())["vpfp"],
    output_root=private_run_root / "vpfp", device="cuda", guidance=True,
)
result = binding.run(recipe, agents, agent_execution=execution_service)
```

The recipe must select a frozen Generator, online base sampling, and no initial
release-bank cases. This online-only PreparedDataset contains no fabricated
training or hidden examples. Real evaluations sample fresh cases from the
bound runtime. Guidance is an optional frozen public-feature reward surrogate
with fresh finite-pool resampling, not a claim of diffusion gradient guidance.
At least four informative prior-round feedback cases are required.

Only `rho`, `momentum_x`, `momentum_y`, and `kinetic_energy` are scored by the
Observable Task. The Candidate need not return the full kinetic distribution
`f(t,x,y,vx,vy)`. `VPFPHoldInitialMacroProgram` already implements the canonical
`predict_observables`; `LegacyMacroObservableAdapter` explicitly wraps an older
`predict_macros` program while preserving its original supported query modes.

## Replacing online coarse solves with a learned answer distribution

The existing model is recorded honestly as
`online_coarse_solve_then_continuous_projection`. Sampling costs an online
coarse PDE solve. `VPFPConditionalLatentDistribution` is a separate genuine
conditional Gaussian generative baseline over complete answer coefficients:

1. Collect actual numerical trajectories and project them to the existing
   continuous positive Fourier–Hermite decoder offline.
2. Fit a condition-dependent mean and low-rank-plus-diagonal residual covariance
   on a seed-separated training split; retain the held-out split in the artifact.
3. Sample fresh coefficient noise and equation conditions directly at runtime.
4. Evaluate the continuous answer and derive its initial data and exact VPFP
   residual source using the unchanged pinned physical implementation.

On a Slurm allocation, for each weak/strong configuration:

```sh
python scripts/train_vpfp_answer_distribution.py collect \
  --config configs/vpfp_unified_live_weak_collision.json \
  --output /scratch/USER/VPFP_RUN/weak-corpus.npz --samples 256 --batch-size 4
python scripts/train_vpfp_answer_distribution.py train \
  --corpus /scratch/USER/VPFP_RUN/weak-corpus.npz \
  --output /scratch/USER/VPFP_RUN/weak-learned.npz --rank 16
```

Both commands fail outside a Slurm allocation and refuse to overwrite outputs.
Read the reported checkpoint digest, load with
`VPFPConditionalLatentDistribution.load(..., expected_sha256=...)`, construct
`VPFPLearnedBatchGenerator(model, checkpoint_sha256=...)`, and pass it as
`latent_sampler=` to the same binding factory. Learned sampling performs zero
PDE solves and can use the same reward-guidance wrapper and resident lifecycle.

These new weights require a training job. No new physical corpus has been
collected or qualified by the unit tests. The baseline is explicitly an
`unqualified_research_checkpoint`: successful statistical fitting is not
high-fidelity physics validation or proof of better hard-case generation.
Before production admission, compare both regimes on held-out continuous
answer/source derivatives, physical residual/source norms, moment coverage,
and fresh canonical macro-only Agent campaigns. The current coefficient-space
perturbation remains the legacy calibrated method; changing to a principled
function-space kernel also requires exact derivative/source and moment support.

`scripts/slurm_vpfp_learned_answer_qualification.sh` is the bounded first
qualification job: one L40S, eight CPU threads, 48 GiB, two-hour ceiling; 32
actual numerical answers per regime, a held-out split, and eight fresh learned
draws. It has not been submitted by this change. It retains collection/training
logs even when a regime fails and checks both regimes independently.
`qualify_vpfp_answer_distribution.py` verifies the corpus/checkpoint binding,
disjoint held-out IDs, source finiteness, sampled positivity and zero online
PDE-solver calls. Its separate reports compare held-out coefficient statistics
and macro marginal distributions on common continuous points. These small
marginal diagnostics cannot certify the full joint answer law. Successful
execution deliberately remains `scientific_qualification=insufficient_evidence`;
manufactured-source validity is not a fidelity or hard-case-lift certificate.
