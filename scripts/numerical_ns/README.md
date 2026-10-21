# Classical NS numerical comparisons

These candidates use the existing NS SDK and Controller evaluator. They receive
only public initial velocity, body force and boundary data. Each integrates the
full physical interval before receiving hidden scoring coordinates. No Agent,
Generator training, private solution or test-based parameter tuning is used.

- `fem.py`: FEniCSx 0.9, P2/P1 Taylor–Hood, BDF2 incremental pressure correction,
  backward-Euler startup, Temam-stabilized extrapolated convection. Body force is
  queried at quadrature points. Boundary roundoff within 64 machine epsilons is
  clipped to the declared domain; genuinely exterior coordinates fail.
- `nektar.py`, `native/`: Nektar++ 5.7.0, polynomial spectral elements, IMEX2,
  full numerical coefficient snapshots. Hidden points are interpolated from
  those snapshots only after the solve. Force projection uses extra quadrature.
  Every native process is single-rank; the adapter explicitly uses Open MPI
  `ob1` / `self,tcp` and disables RDMA-related paths to prevent restricted mlx5
  initialization from hanging before numerical work or during field evaluation.
- `mac.py`: staggered MAC central spatial discretization, SSPRK2 with a pressure
  projection at both stages, CFL/diffusion time-step limits, dynamic Dirichlet
  data. Pressure fixes its constant mode. Boundary flux quadrature mismatch is
  measured and reported; it is not concealed by changing boundary velocities.

The FEM splitting follows Jørgen S. Dokken's `dolfinx_ipcs` (MIT; attribution and
license retained). The Nektar public forcing/mesh bridge derives from the
received Nektar bridge implementation; its reference-probe output and
independent wall timeout were replaced. All children inherit the Controller
process group. Only the Controller's charged solver clock is authoritative.

## Execution

Persistent campaign root:
`${PDE_GAV_SOURCE_ROOT}/.paper_runs/ns_numerical_baselines_20260924`

The environment and original Cases are already prepared there. `control/` holds
runtime versions, frozen cohorts, candidate profiles and actual Slurm receipts.
`qualification/` contains the known-solution convergence checks. `results/`
holds per-case feedback, native solver diagnostics, Controller audits and
checkpoints. `engineering_failures/` retains preflight integration failures;
these are not numerical benchmark scores.

`manage.py launch FAMILY` submits a family exactly once. Its bounded chain is:
4 calibration Cases -> eligible profiles on 24 validation Cases -> one selected
profile on 96 terminal Cases. Families are `fem_obstacle`, `fem_piv`,
`nektar_obstacle`, `mac_piv`. Use the existing `manage.sh` wrapper, which sets the
project interpreter and SDK paths. Repeating launch with its existing receipt
returns the receipt; it does not duplicate jobs.

Each evaluation job requests 8 CPUs, 32 GB, zero GPUs and three hours, with four
candidate workers and one BLAS/OpenMP thread per process. Each family has at
most one running evaluation job; all four families may run concurrently. Later
stages are split into 12-Case jobs. A clean paused shard can continue once from
its existing checkpoint. Startup failures, incomplete checkpoints and a second
pause require inspection; there is no unbounded resubmission loop.

The common frozen-program evaluator executes terminal-phase plans. The runner
adapts the *in-memory invocation tag* of calibration/validation plans to that
API, while retaining their exact queries, references and scientific split.
Original plan phases are recorded in `input_identity.json`; frozen banks remain
unchanged. Selection never reads terminal outcomes. All requested cases and
failures remain in the tables; means on successful Cases have explicit counts.

## Checks

```bash
PYTHONPATH=src:legacy/ns/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/numerical_ns/test_contract.py
.venv/bin/python -m py_compile scripts/numerical_ns/*.py
```

Run the supplied `qualify.sh METHOD` under a CPU compute allocation for the
actual backend convergence checks. The smooth manufactured test has full
nonlinear convection and nonzero body force; FEM/MAC additionally test dynamic
boundary traces. These checks do not replace calibration on the actual Cases.

Versions are explicit: FEM Gmsh comes from its isolated Conda environment
(4.15.2); Nektar mesh conversion uses the cluster Gmsh 4.13.1 module. Their
unstructured meshes are independently generated from the same public geometry.

The 2026-09-24 Nektar MPI startup repair is recorded separately as family
`nektar_obstacle_serialmpi`, with the same two numerical profiles and Cases.
`qualify_nektar_parallel.py` verifies four simultaneous single-rank solves at
two resolutions against the prior known-solution errors. Existing candidate
bundles and results remain frozen. The campaign runlog records the exact job
IDs and the remaining fine-profile timing limitation.
