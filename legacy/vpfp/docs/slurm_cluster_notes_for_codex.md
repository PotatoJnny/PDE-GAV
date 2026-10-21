# Slurm Cluster Notes For Codex

This file is copied into the workspace of independent solver agents launched by
the legacy VPFP campaign. It states how to behave on a shared Slurm cluster.
It applies even when a task prompt does not repeat every rule.

## Login Nodes

Login nodes are for editing source and documentation, `git` operations, short
file inspections, `squeue`, `sacct` and `sinfo`, submitting or cancelling jobs,
and tiny argument, syntax or import checks.

Do not run full `pytest`, numerical data generation, audits, solver benchmarks,
training, sampling, rollout generation, policy optimization, or CPU/GPU-heavy
agent tools on a login node. Those tasks must run through `sbatch`/`srun` or
inside an existing allocation.

## Storage

- Keep source, tests, configurations, Markdown and compact summaries in the
  project checkout (`${PDE_GAV_SOURCE_ROOT}`).
- Keep datasets, public solver workspaces, checkpoints, trajectories, model
  caches, images, logs and temporary outputs under the run root
  (`${PDE_GAV_RUN_ROOT}`) or `$SLURM_TMPDIR`.
- A solver agent must not install packages, write caches, or create experiment
  outputs inside the project checkout.

## Environment

Every repository Slurm wrapper initializes the site environment before doing
work (for example by loading the site's Python and scientific modules and
activating the project virtual environment), then verifies that it runs inside
an allocation:

```bash
test -n "${SLURM_JOB_ID:-}"
test -n "${SLURM_TMPDIR:-}"
test -n "${VIRTUAL_ENV:-}"
case "$(command -v python)" in
  "${VIRTUAL_ENV}/"*) ;;
  *) exit 2 ;;
esac
```

Do not install into system Python or mutate the shared virtual environment
during an experiment. If a missing dependency is essential, create a task-local
virtual environment under the run's scratch workspace with
`--system-site-packages`, activate it, and install only the missing dependency.
Do not reinstall PyTorch when the environment already provides it.

## Resource Requests

Use the smallest allocation that supports the planned experiment. The default
solver-agent allocation is one GPU, 8 CPUs, 96 GB of memory and 6 hours.
Account, QoS and partition are site-specific and are passed to `sbatch` by the
launcher. Request several GPUs only when a measured bottleneck justifies it.
Keep Slurm stdout and stderr under the run root.

The independent solver agent is launched after the wrapper has acquired a node
and must use that existing allocation. It must not start a nested interactive
job or move a workload to a login node. If the allocation is insufficient, the
agent writes `requested_slurm_resources.json` and `requested_job.sh` in its
public workspace for the outer controller, which reviews and submits the
follow-up job.

During solver development, keep each exploratory numerical or training command
below 10 minutes. Start with smaller grids or fewer optimizer steps, then scale
only after a measured result justifies the additional allocation time.

The outer runner verifies the allocated GPU before Codex starts. Codex's normal
`workspace-write` sandbox may hide NVIDIA device files even though Slurm
allocated a GPU. If an agent-side `nvidia-smi` or `torch.cuda.is_available()`
check fails, the agent requests a narrowly scoped, reviewed escalation for that
GPU command. It must not request persistent host access or a blanket sandbox
bypass.

## Network

Independent solver agents may use the internet when it improves the numerical
method: live search for scientific papers, primary literature and official
software documentation, small reference implementations, and minimal missing
package downloads.

Network use must remain bounded: no bulk crawling, site mirroring, network
scanning or remote services; no upload of public case arrays, cluster paths,
credentials, logs or code; stop a stalled network command after 10 minutes;
avoid downloads larger than 250 MB each or 1 GB in total unless the public task
clearly requires them; record useful URLs and their role in
`solver_research_notes.md`.

## Approval Policy

The instruction-only mode uses:

```toml
approval_policy = "on-request"
approvals_reviewer = "auto_review"
sandbox_mode = "workspace-write"
web_search = "live"

[sandbox_workspace_write]
network_access = true
```

This lets the reviewer approve a narrowly scoped action when a normal sandboxed
command cannot run. It does not require full access or bypassing the sandbox.
Solver agents must never request blanket access. An approved command must still
respect the public-context, storage, network and Slurm rules in this file.
Auto-review is approval routing, not host filesystem isolation. Results from
this mode are labeled `blind_instruction_only`.

The returned solver is executed separately in the public-only evaluator before
trusted code loads its prediction arrays.
