# GPU Agent development and evaluation protocol

Updated 2026-10-01, with integration notes from 2026-10-02 and 2026-10-03.

> The merged code additionally supports sharing the Controller GPU, per-case
> exception feedback, Markdown prompts, explicit primary selection, batch
> solving, configurable per-device concurrency and independent time budgets;
> see [module integration](MODULE_INTEGRATION.md) and
> [parallel evaluation](PARALLEL_EVALUATION.md). The fixed values below are the
> historical protocol of the six Plasma campaigns, not new defaults of the
> shared engine.

Agents can develop, train and deliver solver programs with checkpoints on a
GPU. The system verifies the same submission in an independent GPU worker and
hands the complete program to the next round's Agent. This reuses the single
`CampaignController` and the existing bundle, Task, scoring and Memory
interfaces; no separate experiment system is built. GPU execution is an
explicit choice: an allocated device does not by itself mean the Agent can use
CUDA.

## Experimental protocol and resource split

The Plasma experiments use seed 101, five rounds and two Agents, each with a
14,400-second full wall clock per round. Official evaluation uses 256 Cases:
86 Base, 85 Guided and 85 Base-perturbed. Guidance is updated from the selected
program's feedback along the original flow. Each Agent enters exactly one
primary program into Official; the Controller retains the incumbent separately.
The two model arms are `gpt-5.6-sol` / `max` and `gpt-6-astra` / `ultra`, with
identical scientific configuration, initial S0, empty Memory and separate run
directories within one equation; models are not substituted on failure.

The deployment uses four GPUs with simple isolation rather than a three-GPU
shared scheduler:

| Phase | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
| --- | --- | --- | --- | --- |
| Agent development | Agent A | Agent B | Controller and numerical Oracle | Development-evaluation Candidates |
| Official and Terminal | idle Agent GPU lent to the Oracle | idle Agent GPU lent to the Oracle | Controller and Oracle | Official-evaluation Candidates |

Actual tokens come from the allocation; a subprocess usually sees one device as
`cuda:0`. Physical GPUs are never inferred from logical indices: the native
checks compare UUIDs and NVIDIA device minors, and the deployment uses
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. GPU 3 is never borrowed by the hidden Oracle
pool. GPUs 0/1 are reused only after development ends and the Agent leases are
released; leaving hidden evaluation cleans the Oracle pool and restores the
original binding.

The Candidate service allows one active Case per device. `candidate_case_workers`
of 8 or 16 is a CPU worker limit, not a claim that one Candidate GPU runs 8 or
16 GPU Cases at once. Four-GPU isolation is a deployment choice, not a
mathematical requirement; there is no dynamic device sharing, memory
preemption or CUDA MPS.

## Budgets and preparation

The `shared_gpu_v1` profile validates the 900-second whole-Case budget and the
14,400-second Agent wall clock as the constraints of this comparable protocol,
not as the only values the GPU service accepts.

- `prepare_solution(case, solve_budget=None)` has no separate 180/600-second
  cap. The NS-aligned VPFP and Artemis Tasks select
  `candidate_budget_mode="ns_remaining_case_v1"` and no longer set
  `candidate_compute_budget`; the remaining whole-Case time is passed through.
- Preparation, queries, GPU computation and result return share the whole-Case
  budget; CUDA is synchronized before completion, so kernel submission time
  alone does not count as finished work.
- The charged-wall convention is unchanged: only Controller-metered Oracle
  queue and lock waits are excluded. Oracle numerical work, data transfer and
  the program's own waiting are charged. Raw wall clock and wait components are
  recorded separately.
- Waiting for a free Candidate GPU happens before the worker starts and does
  not consume the Case budget; it is recorded as `gpu_queue_wait_seconds` and
  still counts toward the outer development batch deadline and the Agent wall
  clock.
- The Agent's four hours include model, tool, training and platform waits.
  Temporary native sessions do not reset the deadline; whether a failed session
  may be resumed is decided by the run's existing retry policy, and GPU use adds
  no attempts.

Old CPU configurations, historical preparation caps and existing scientific
recipes do not change automatically. CPU and GPU experiments are reported as
different execution profiles; devices and budgets are never switched silently
during a run.

## Submitting code together with checkpoints

Attachments reuse the existing `pde_agent_verifier.candidate_bundle.v1`; there
is no GPU-only protocol. A minimal directory is:

```text
candidate/
  submission.py
  model.pt
  candidate_bundle.json
```

Train inside the Agent's permitted workspace, then declare the complete file
set explicitly:

```python
from pde_agent_verifier_solver.sdk import write_candidate_bundle_manifest

write_candidate_bundle_manifest(
    "candidate",
    entrypoint="submission.py",
    files=["submission.py", "model.pt"],
)
```

Submit the directory with the current run's `agent_tool submit`; whether
`--primary` is required depends on the published nomination protocol. Do not
submit only the entry `.py` while leaving weights in an external directory. The
Controller computes the tree hash from the actual file contents, freezes the
declared files, moves them to an independent evaluation workspace, and the next
round's clone carries the same attachment. Bundles reject symbolic links,
directory traversal and undeclared files; the default limits are 512 MiB and
4,096 files in total, and a deployment may restrict them further.

Candidate programs should load weights relative to their own location, not
from development-time absolute paths or in-memory objects:

```python
from pathlib import Path
import torch

weights = torch.load(
    Path(__file__).resolve().with_name("model.pt"),
    map_location="cuda:0",
    weights_only=True,
)
```

`prepare_solution` never receives private scoring coordinates; the returned
handle answers `evaluate(query)` under the public ABI. A submission must
rebuild its model in a fresh process and cannot rely on the development
process's CUDA context. A CPU fallback cannot be reported as GPU acceptance.

## Connecting the shared execution service

The new configuration lives under `execution.gpu_execution`. This is an
execution fragment, not a complete scientific configuration:

```python
from hashlib import sha256
from pathlib import Path

runtime = Path("/approved/runtime/bin/codex")  # independent runtime configured and accepted by the deployer
config["execution"]["gpu_execution"] = {
    "protocol": "shared_gpu_v1",
    "candidate_devices": ["3"],
    "agent_sandbox_backend": "bwrap-nvidia-v1",
    "agent_runtime_sha256": sha256(runtime.read_bytes()).hexdigest(),
}
```

The hash binds an already accepted runtime; hashing an arbitrary binary is not
acceptance. Allocated devices must exist and must not overlap the Agent and
Controller tokens. Missing CUDA, UUID mismatches, and runtime or configuration
changes fail explicitly instead of silently falling back to CPU or lifting the
sandbox.

Shared code entry points:

- `engine/gpu_execution_config.py`: explicit configuration, device separation,
  Agent capability description and the budget checks of the new profile.
- `engine/gpu_execution.py`: per-device leases, real CUDA readiness
  computation, synchronization and device audits.
- `engine/candidate_execution.py`: GPU tokens passed to isolated processes,
  verified execution receipts; a device whose cleanup cannot be confirmed is
  not reused.
- `scripts/shared_agent_gpu.py` and `scripts/qualify_agent_gpu_sandbox.py`:
  real CUDA forward/backward and workspace-restriction checks inside the native
  sandbox, bound to the runtime, configuration and allocated devices, without
  reading model credentials.
- `scripts/run_canonical_codex_agent.py`: explains CUDA and attachment
  capabilities to the Agent and launches the checked runtime.

The NS, Artemis and strong VPFP canonical runners are connected. When creating
a new GPU run from a complete upstream configuration, align the budgets: NS
already has the 900/14,400-second configuration; Artemis removes
`execution.agent_active_budget_seconds` and, like strong VPFP, removes
`task.candidate_compute_budget` and sets
`task.candidate_budget_mode="ns_remaining_case_v1"`. These changes apply to new
GPU runs only. The NS runner takes model and effort through its original
interface; the frozen Artemis and strong VPFP runners stay locked to Sol/max,
and the GPU switch does not open Astra. The weak VPFP deployment uses the same
shared service, but its complete migration launcher, path map and model assets
are not part of this repository; pulling this code alone does not rebuild the
six formal experiments.

The CPU path remains the default. GPU wiring is enabled through Python or the
CLI and reported through the existing Run events, Agent status, Candidate
audits and result pages; there is no dedicated GPU configuration form or live
utilization panel. Audits include `gpu_device`, `gpu_queue_wait_seconds` and
CUDA execution metadata.

## Oracle data transport

Agent, Candidate and numerical Oracle are independent processes and may sit on
different GPUs. The public Oracle ABI uses CPU arrays over the existing
loopback RPC: a request travels to the service, the Oracle computes on its own
GPU and converts the result to a public array, and the caller moves it to its
own GPU if needed. Attachment support and GPU workers do not turn arrays into
cross-process CUDA tensors. No CUDA IPC, GPU pointers, zero-copy tensors or new
server-side mathematics are added; transport micro-benchmarks do not replace
whole-workload throughput evidence.

## Native runtime and host limits

The runtime documentation distinguishes the sandbox boundary from the approval
policy; `never` does not mean the sandbox is removed. The integration keeps
workspace-write permissions and the deployment's network policy and does not
fall back to full access.

The frozen acceptance used an isolated build of Codex 0.155.1 with a narrow
compatibility patch for explicitly authorized NVIDIA character devices, rather
than making all of `/dev` writable. The source patch and the lockfile version
patch are under `patches/`; the build entry point is
`scripts/build_compatible_gpu_codex.py --help`. This native Agent gate targets
a single non-MIG NVIDIA device on a Linux Slurm host; nothing is claimed for
other operating systems, MIG, other network policies or arbitrary Codex
versions. The builder requires the original official package, the pinned source
commit, both patches, an independent output directory and a build cache; it
runs only inside a compute allocation, installs or replaces no system client,
and reads no credentials.

The build is pinned to an upstream commit and uses `--locked`; the lockfile
patch only sets workspace version markers. The current build profile is
`dev-small` and is not an official release equivalent. The output starts as
`built_unqualified`; real sandbox and CUDA evidence must be collected before an
Agent may use it. The retained `legacy-landlock` path is a historical
compatibility entry, not a current deployment.

Some hosts restrict user namespaces, which makes native file tools fail; the
NVIDIA patch does not fix that host policy. The frozen deployments add an
authorized in-sandbox shell editing notice for both model arms; it is a host
adaptation, not a general default, and not permission to skip file tools or
approvals elsewhere. Any bubblewrap or namespace configuration is an
administrator task, not something the experiment changes.

Native child Agents may request `native_session_persistence`: the session and
its SQLite state are kept only in the private temporary directory of the
current call, run within the original deadline and are cleaned up on exit. It
imports no chat history from old runs, shares no sessions between the two
experiment arms, and adds no retries. Model and effort remain explicit.

## Verification evidence and open items

This repository carries CPU/mock regressions for the GPU service, the two Task
budget combinations, the runner configurations, the native worker, the GPU gate
and the runtime build. Real GPU and native-Agent qualification of a deployment
is a separate step performed on the target host with its runtime, and the
historical acceptance records of the frozen deployments are not acceptance of
newly merged code. Tests that need no scientific assets or CUDA:

```bash
PYTHONPATH="$PWD:$PWD/src:$PWD/scripts" python -m pytest -q \
  tests/test_candidate_gpu_execution.py \
  tests/test_gpu_execution_config.py \
  tests/test_shared_gpu_runner_integration.py \
  tests/test_artemis_ns900_budget.py \
  tests/test_vpfp_ns900_budget.py \
  tests/test_canonical_native_session_persistence.py \
  tests/test_shared_agent_gpu_bwrap.py \
  tests/test_compatible_gpu_codex_build.py
```

These suites overlap; their counts are not added as independent totals. No
model weights, runtime binaries, credentials, private sessions, raw run logs or
personal launch directories are published.

The 2026-10-04 NS template selects `agent_sandbox_backend="bwrap-nvidia-v1"`
and takes the independent GPU-compatible runtime from `--codex-binary`
(`CODEX_BINARY` for the Slurm entry); the old legacy-landlock backend is no
longer the default GPU route. The native preflight and the real Agent use the
same runtime and the same derived permission configuration, exposing only the
allocated NVIDIA device nodes.
