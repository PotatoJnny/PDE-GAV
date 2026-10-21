# Shared modules and experiment configuration

A researcher selects a PDE and Task, writes task/resource/role Markdown, and supplies a run configuration. Every new run uses the same `CampaignController`, candidate format, evaluation service, and results interface. PDE plugins provide the scientific definitions and numerical implementations through the SDK.

## Integrated capabilities

The current integration combines the shared GPU execution protocol, the Plasma campaigns and transport updates, and the NS performance and feedback improvements in one Controller path.

| Module | Shared behavior |
| --- | --- |
| Controller | One canonical lifecycle for rounds, acquisition, evaluation, selection, and recovery |
| Development feedback | Aggregate and per-case errors, partial progress, solver exception messages and tracebacks, code locations, and cancellation |
| Case acquisition | A numeric limit or `null`; actual acquisition remains recorded. Historical configurations retain their original limits |
| Candidate GPU | Explicit device pools, CUDA checks and synchronization, recorded admission wait and memory use; NS can route development evaluation to the requesting agent's GPU |
| Agent GPU | Optional native sandbox GPU qualification; enabling another module does not install or replace an Agent runtime |
| Program delivery | Existing bundles freeze code and declared weights for evaluation and subsequent rounds |
| Recovery | Workflow progress and optional native sessions persist; configured retries use the original deadline |
| Oracle transport | Inline public arrays, transient sidecars, optional trusted-result codecs, batch queries, fair admission, and optional timing traces |
| Prompts | SDK task/resource/role sections plus Controller task interfaces and records; the execution adapter adds actual budgets and tool instructions |

Integration does not import another experiment's cases, model assets, checkpoint, or scientific settings. NS keeps its velocity-error metric. Plasma transport v2 is an explicitly selected scientific protocol; see [Plasma integration](PLASMA_INTEGRATION.md#shared-plasma-macro-transport-v2).

## Temporary model-service failures

Managed Codex rounds default to `transient_resume_until_deadline: true`. If the
selected model is temporarily at capacity, rate limited, disconnected, timed
out, or unavailable, the worker resumes the same saved conversation and
workspace. Completed submissions and primary selection remain available. The
previous native process and in-flight evaluations are settled before reconnecting.

Capacity errors wait 30 seconds before the first resume and 60 seconds thereafter.
Other temporary errors wait 5, 15, 30, then at most 60 seconds. The final wait is
limited to the remaining budget. All attempts, service waits,
cleanup, and backoff share the original Controller deadline. A four-hour round
has four hours of total wall time, not four extra hours after each reconnect.
Recovery stops on success, cancellation, deadline, an unrecoverable error, or an
unavailable saved session. Account exhaustion, authentication errors, invalid
requests, and policy rejections are not temporary capacity failures; recovery
does not change the account or model.

NS, VPFP, and Artemis launchers accept `execution.transient_resume_until_deadline`;
custom Command Agents set it in `worker_settings`. To reproduce the old bounded
policy, set it to `false`; `transient_resume_max_attempts` then limits total
attempts to 1-3. Completion-driven literature stages retain their bounded policy
because they have no elapsed deadline.

During reconnection the Agent remains in its running round. Its stderr log
reports the reason, next delay, and remaining time; `codex-recovery.json` records
each attempt and the unchanged deadline. The panel shows when an Agent is waiting
for the service and when it resumes the same session. Display failures cannot
interrupt scientific work. Frozen or already running deployments
must be updated explicitly before they receive this behavior.

The native response watchdog defaults to 900 seconds and accepts a positive
`worker_settings.native_response_timeout_seconds`. It detects missing service
activity while no native tool is running. Tool execution remains subject to the
original Controller deadline; large tool-output events retain their lifecycle
metadata so they cannot disable the watchdog or cause a false interruption.

### Recovery reconciliation

Earlier development snapshots already contained capacity classification,
30/60-second backoff, and same-session tests, while some deployed runtimes had
retained the retry loop but omitted that capacity handling. The integrated code
keeps its single `transient_resume_until_deadline` setting and existing loop,
adding the reviewed watchdog, progress display, and stronger error
classification. No second Controller recovery state machine is introduced.

The earlier Plasma startup corrections are also integrated: formal scoring
settings reach the Generator, and hidden evaluation has one GPU-scope owner.
New deployment acceptance recomputes all 160 reference draws, refits the
calibration, and verifies the established calibration bytes before testing all
three PDE families and both native model routes.

A run audit on 2026-10-05 invalidated weak Sol, weak Astra, strong Sol, and
Artemis Sol because capacity failures prematurely ended seven Agent turns.
Their old results are excluded from comparisons and replaced by fresh runs.
Strong Astra and Artemis Astra keep their existing progress: native reconnects
recovered, and their turns completed normally or reached the original deadline.
Frozen runs receive no implicit source or configuration changes.

## Primary candidate selection

New NS templates use `single_primary_v1`. An ordinary `submit` saves a code/weight snapshot; `submit --primary` also selects it for this round. `select-primary N` selects an earlier saved submission. Later ordinary submissions do not replace that choice.

The selection belongs to the current agent and round, is saved atomically, and survives a failed Agent process. A missing selection does not silently become the latest or best development-scoring submission. Only the explicit primary enters Official evaluation alongside the retained incumbent. Status and `final_nomination_sequence` show the actual choice.

The panel exposes the selection policy for Command Agents and displays it in the Run. Local Toy Agents return their programs directly. Historical `all` and `latest` policies remain available. Selection policy does not change GPU allocation or scientific budgets.

## GPU resource choices

NS exposes the choices below. Artemis and strong VPFP retain their explicitly configured `shared_gpu_v1` entry points and their own protocol validation. The common execution service does not force NS settings on them.

| Choice | Configuration | Behavior |
| --- | --- | --- |
| CPU | No GPU configuration, or `candidate_device: "cpu"` | Candidate workers run with CUDA hidden |
| Share the Controller GPU | `candidate_device: "cpu_or_cuda"` | Candidates use the assigned Controller devices with configurable `gpu_slots_per_device`; programs may use CPU, GPU, or both |
| Separate candidate GPU | `execution.gpu_execution` | Explicit candidate devices remain separate from the Agent/Controller devices; this takes precedence over `candidate_device` |
| Agent-owned development GPU | Add `development_solver_placement: "agent_gpu"` | Development evaluation uses the requester's Agent GPU. Official/terminal evaluation uses the configured candidate pool. The Oracle serves both agents separately |

For example, a separate candidate device can be selected with:

```json
{
  "protocol": "shared_gpu_v1",
  "candidate_devices": ["3"]
}
```

Time limits are read from the run configuration independently of GPU placement. The NS template defaults are 900 seconds per case and 14400 seconds of full wall time per Agent. There is no additional preparation deadline. Batch support and concurrency controls are described in [parallel evaluation](PARALLEL_EVALUATION.md). NS defaults to batches of eight; Plasma remains single-case until its Task supplies a batch interface.

GPU admission happens before the worker starts and is recorded separately. Solver and Oracle computation count toward the case budget. Trusted Controller queue waiting is tracked separately; all waiting still consumes Agent wall time and the outer development evaluation deadline.

Native Agent GPU use requires a compatible, qualified runtime, as described in the [GPU protocol](SHARED_GPU_AGENT_PROTOCOL.md). Source-level support and CPU tests do not establish that a particular machine's sandbox or CUDA works. This integration does not request a Slurm allocation or restart existing jobs.

## Configure task instructions with Markdown

[ns_modular_prompt_example.json](../configs/ns_modular_prompt_example.json) demonstrates the file interface using historical NS template scientific parameters. Before a new run, bind accessible resources through the run configuration (see [site configuration](SITE_CONFIGURATION.md)) and confirm the experiment protocol. Existing inline `agents[].prompt` configurations remain supported.

```json
{
  "prompt_files": {
    "task": "prompts/ns/TASK.md",
    "resources": "prompts/ns/RESOURCES.md"
  },
  "agents": [
    {
      "agent_id": "ns-broad-explorer",
      "name": "Explorer",
      "role_file": "prompts/ns/EXPLORER.md"
    },
    {
      "agent_id": "ns-critical-refiner",
      "name": "Refiner",
      "role_file": "prompts/ns/REFINER.md"
    }
  ]
}
```

Paths are relative to the configuration file. `TASK.md` describes the scientific task and deliverable. `RESOURCES.md` describes available materials. Optional role files give broad research directions. Do not specify both file sections and an inline prompt for the same agent.

The NS reader loads file contents into `agents[].prompt_sections` and the existing `prompt` field. `resolved_config.json` stores the contents so reloading it does not depend on subsequently edited Markdown files. The final native prompt, including execution budgets and interfaces, is saved as `CANONICAL_AGENT_PROMPT.md` in each round's workspace.

The panel supports editing/importing task, resource, and per-agent Markdown and previews the Controller-provided instructions. The preview is labeled with its scope: the execution adapter adds runtime-specific instructions at launch. The old `agent_prompt` API remains available; browser-supplied paths do not cause arbitrary server-file reads.

A resource-description file is prose, not a data-copy instruction. Public data and scientific capabilities still enter through the plugin's resources and interfaces.

## Restricted-namespace host compatibility

The shared `scripts/canonical_codex.config.toml` developer instructions include `restricted_namespace_workspace_edit_v1`. NS, strong VPFP, Artemis, and other canonical workers using this configuration receive the same guidance.

After confirming that the host restricts user namespaces (`max_net_namespaces=0` or the documented namespace-related `ENOSPC`), the Agent may use the sandbox shell/Python to edit within the already authorized workspace. It should not repeatedly invoke an affected native file tool. Other hosts keep their normal tools. If the sandbox shell also fails, preserve progress and report the failure; do not enlarge permissions, change sysctl settings, or replace the sandbox.

This is tool-use guidance, not a host or binary repair. It does not change approvals, network settings, GPU access, models, budgets, or hidden-data permissions. The native worker copies the full configuration into its private run directory, and GPU configuration derivation preserves the instructions. Panel task previews do not include these adapter-level instructions. Frozen deployments and queued jobs are not rewritten automatically.

## Verification and deployment

Use the smallest tests covering the modules selected for a run. Shared integration checks include:

```bash
PYTHONPATH="$PWD:$PWD/src:$PWD/scripts" python -m pytest -q \
  tests/test_restricted_namespace_editing_protocol.py \
  tests/test_candidate_oracle_inline_transport.py \
  tests/test_oracle_bulk_transport.py \
  tests/test_agent_owned_candidate_gpu.py \
  tests/test_gpu_execution_config.py \
  tests/test_canonical_native_session_persistence.py
```

Product checks and PDE-specific acceptance are described in the [extension guide](../src/pde_agent_verifier_solver/plugins/ADDING_A_PDE.md#verify-integration). CPU/mock checks cover configuration, subprocess communication, recovery, and numerical contracts. A target deployment must separately qualify its actual GPU runtime and model assets. Plasma transport v2 also requires its independent calibration and family-specific numerical acceptance.

Historical verification records describe the versions tested at the time; they are not qualification receipts for a new merged deployment. Keep frozen experiments and new research runs separate.
