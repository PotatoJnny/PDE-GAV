# Batch solving and Oracle concurrency

A run selects its evaluation resources and concurrency. An Agent may deliver an
ordinary single-case solver, or add `prepare_solutions(cases, solve_budget=None)`
to solve several Cases in one call. Both paths use the existing `evaluate`,
Official, Terminal, per-case error and Promotion entry points; there is no
second experiment loop.

## Configuration

The fields live in the `execution` section of the NS JSON configuration. Older
configurations without them default to batch 1, one GPU slot per device and one
Oracle request per device. The new NS templates set:

| Field | New template | Meaning |
|---|---:|---|
| `solver_batch_size` | 8 | Maximum Cases a batch-capable solver receives per call; the last batch may be smaller |
| `gpu_slots_per_device` | 2 | Solver workers running concurrently on one evaluation GPU |
| `oracle_max_workers` | 8 | Total Oracle admission limit across all phases; Official evaluation may use several Oracle GPUs |
| `development_oracle_max_workers` | 2 | Combined Oracle limit for the two Agents during Development, sized for one Oracle GPU |
| `oracle_concurrency_per_device` | 2 | Concurrent numerical queries of the read-only NS continuous field on one GPU |
| `candidate_case_workers` | 28 | Total solver worker limit; GPU mode is further bounded by device slots |
| `agent_hard_wall_seconds` | 14400 | Full Agent wall-clock budget, adjustable per run |
| `candidate_case_hard_wall_seconds` | 900 | Per-case execution budget, adjustable per run |

GPU placement still offers CPU, sharing the Controller GPU, or a separate
evaluation GPU through `execution.gpu_execution.candidate_devices`; the separate
device is no longer tied to the 4-hour / 900-second budgets. These fields do not
request GPUs; the allocation must provide the devices.

The panel's Setup and Run pages show the batch, GPU and Oracle settings of the
running service. New run templates enable parallelism explicitly; historical
paper configurations are not rewritten.

## Four-GPU layout and requester routing

New NS templates select `execution.development_solver_placement: "agent_gpu"`
and declare `gpu_execution.candidate_devices` for official evaluation (`["3"]`
in the four-GPU example). Agent and Oracle GPUs are given by the launcher's
`--worker-gpus` and `--controller-gpu`; devices are never inferred from program
authorship. The formal NS Slurm entry requests four GPUs and binds the official
evaluation role to the fourth allocated token with `--candidate-gpus`. That
argument only binds resources; the actual allocation is written to the run
record.

| Phase | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|---|
| Development | Agent A and the solvers A requests | Agent B and the solvers B requests | Public Oracle queries of both Agents | idle |
| Official / Terminal | lent to the Oracle by configuration | lent to the Oracle by configuration | Oracle | common solver evaluation |

Each Agent has an independent execution queue, so one Agent's long batch cannot
block the other's device. Solvers on one Agent GPU share it with the Agent's
own computation. The Oracle occupies a service slot only during a query, not
for the whole solver run.

`execution.trace_calls: true` records `candidate_execution/execution_trace.jsonl`
for short tests: solver submission, actual start, completion or cancellation
before start, and for every Oracle query its method, point count, service time
and queue time, including cancelled queued queries. It writes no coordinates,
answers or session tokens. Per-case audits add `requester_agent_id` and
`solver_queue_wait_seconds`; batch members share queue time, so statistics must
de-duplicate per batch. Oracle waits of different Cases overlap and cannot be
summed as Agent wall-clock loss. Historical configurations without tracing
produce no per-call log.

## Execution and fairness

A solver's evaluation request is split into batch or single-case tasks. The
shared execution queue alternates free workers between unfinished requests
instead of finishing one large request first; it does not preempt running
programs. Each Agent's operation lock allows at most one scientific evaluation
request at a time, so request rotation also rotates between the two Agents. CPU
worker counts do not bypass the GPU concurrency limits.

All Oracle hosts share one bounded admitter and rotate between requests. Every
Case keeps its own input permissions, Case id and output. The audited read-only
NS continuous field evaluates concurrent numerical queries on independent CUDA
streams; unknown mutable backends keep their original locks. Independent CPU
Cases can also run concurrently.

Concurrent NS queries keep the original analytic formulas and CUDA-graph
acceleration. In evaluations with an existing graph cache, each concurrent
channel uses its own buffers and stream: at most eight cached channels with at
most eight graph entries each. Development queries without a graph cache keep
the eager path. Shutdown waits for sessions to end before releasing channels,
and a numerical fault prevents the resource from being reported healthy.
Concurrency raises memory and CPU pressure; measure real batch throughput
rather than assuming proportional speedup.

## Solver interface, results and timing

The entry points and usage examples are in
`plugins/ns/_public_resources/SOLVER_INTERFACE.md`. The Controller only checks
the function declarations of frozen code; candidate code loads only in workers.
A batch-capable solver receives one `prepare_solutions` call and returns
`{case_id: handle}`. Scoring coordinates are supplied only after all handles
are ready; reference values stay in the Controller. Ordinary solvers keep
`prepare_solution` and are never silently converted into serial batches.

A batch of N Cases shares N times the single-case limit; the actual N may be
smaller than the configured maximum. This limit is not a statement of how many
seconds each Case used. Only the time during which all members were waiting is
deducted from the shared wall clock; overlapping waits are not deducted twice,
and Oracle computation is charged. If a whole batch fails, members without a
reliable result fail together; successful results of other independent batches
are kept under the existing partial-feedback rule. Cancellation applies to the
request and cleans up all member sessions and processes.

Per-case audits add `solver_batch_size` and `solver_batch_id`; batch timing is
labelled `shared_batch_wall_excluding_all_members_waiting`. The shared time
appears in every member's audit, so cost statistics must de-duplicate by batch
id. `candidate_batch_prepare_wall_seconds` in the SDK is likewise shared
preparation time.

The configuration is stored with `resolved_config` and the run provenance.
Recovery continues with the original configuration, programs, case banks and
completed results; batch size and time budgets of an existing run cannot be
changed to alter a historical experiment.

## Verification

`tests/test_batch_solver_execution.py` uses real isolated subprocesses to check
batch calls, genuinely overlapping Oracle computations, per-case equality, tail
batches, permissions, independent cancellation, continued service after
timeouts, rotation and wait timing. `tests/test_batch_solver_cuda.py` checks on
an allocated GPU that four NS queries use distinct streams with identical
numerics, and that two workers share one GPU for real CUDA batches. CPU/mock
checks do not replace that CUDA acceptance.

On an allocated GPU (use a fresh work directory):

```bash
PDE_REAL_GPU_TEST=1 python -m pytest -p no:cacheprovider tests/test_batch_solver_cuda.py tests/test_candidate_gpu_execution.py tests/test_ns_analytic_cuda_cache.py
python scripts/benchmark_parallel_evaluation.py --device cuda:0 --work-root /tmp/pde-batch-check --output /tmp/pde-batch-check.json
python scripts/benchmark_parallel_evaluation.py --device cuda:0 --oracle-cache --steps 96 --modes batch_serial_oracle,batch_oracle_2,batch_parallel_oracle --work-root /tmp/pde-batch-cache-check --output /tmp/pde-batch-cache-check.json
python scripts/benchmark_ns_oracle_cache.py --output /tmp/pde-oracle-cache-check.json
```

## Measured behavior (summary)

The template values were chosen from small performance checks on single L40S
GPUs with a frozen coarse-grid projection program, eight frozen NS Cases and two
simulated Agents. Batching four Cases per worker with two workers was about five
to six times faster end to end than single-case serial execution, with
identical predictions. Two concurrent Oracle requests per device was the best
setting in those checks; four and eight were slower. Combining the public-input
RPCs of one evaluation into a single HTTP request, and querying several known
time points at once, reduced per-call overhead further. These are not paper
solvers and the ratios do not transfer to complete formal experiments; the
measured gain mostly comes from reusing workers, library initialization and
cross-case numerics within a batch.

A four-GPU check confirmed that routing each Agent's development solvers to its
own GPU removes the cross-Agent solver queue wait, without claiming the same
proportional speedup for the whole workflow. The current template therefore
uses batch 8, two process slots per solver GPU, a Development Oracle limit of 2,
an Official/Terminal Oracle limit of 8, and 2 concurrent requests per Oracle
GPU. All of these can be changed per run.

## Loading-path simplification

Programs receive an identifier when submitted; later worker loading, copying
and recovery reuse it instead of rescanning the bundle. Array sidecars, Case
banks, model weights and recovery files are read directly, without recomputing
file checksums as an admission condition. Path boundaries, type and shape
checks, task settings, request ownership and numerical error handling are
preserved; source snapshots of completed runs are not modified. In a local CPU
measurement of a 16 MB bundle, the full identity scan took about 48.5 ms and the
structural check with a reused identifier about 0.25 ms; this covers only the
bundle check, not Python, Torch or CUDA initialization.
