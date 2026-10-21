# Site configuration and placeholders

The experiments in this repository were run on a Slurm cluster. The published
copies of launchers, historical configurations, provenance records and result
tables replace that site's storage locations, account and credential paths with
the placeholders below. They mark where a value must be supplied; they are not
resolved automatically unless stated.

| Placeholder | Meaning |
| --- | --- |
| `${PDE_GAV_SOURCE_ROOT}` | A checkout of this repository, or of the frozen source used by a recorded run |
| `${PDE_GAV_WORKTREES_ROOT}` | Parent directory of the sibling Git worktrees that the acceptance and benchmark scripts compare (`<root>/<worktree-name>`) |
| `${PDE_GAV_RUN_ROOT}` | Directory for run outputs, checkpoints, logs and job receipts (fast scratch storage) |
| `${PDE_GAV_DEPS_ROOT}` | Checkouts of external physical backends used by the GKW and Artemis plugins |
| `${PDE_GAV_SHARED_DATA_ROOT}` | Shared dataset root, for example the NS D0 cases and their split manifest |
| `${PDE_GAV_ARCHIVE_ROOT}` | The authors' experiment archive: Generator weights, private cases, execution logs and saved predictions |
| `${PDE_GAV_AGGREGATE_ROOT}` | Local workspace used when result tables were aggregated |
| `${PDE_GAV_VENV}`, `${PDE_GAV_PYTHON}` | The Python environment and interpreter used by shell launchers |
| `${CODEX_BINARY}`, `${CODEX_AUTH_FILE}` | The Codex CLI executable and its authentication file. Never commit an auth file |
| `${PDE_GAV_SLURM_ACCOUNT}` | The Slurm account to charge |
| `${PDE_GAV_PHYSICAL_REPOSITORY}` | The Git remote of an external physical-backend repository |

## Shell launchers

Scripts under `scripts/` and `legacy/*/scripts/` that submit Slurm jobs no
longer carry `#SBATCH --account`, `--exclude` or `--chdir` directives, and their
log paths are relative to the submission directory. Pass site options on the
command line, for example:

```bash
sbatch --account=my-account --partition=my-partition scripts/slurm_plasma_canonical_smoke.sh
```

The remaining `--partition` directives name the L40S partitions of the original
cluster and are examples only. Each script lists the placeholder variables it
uses in a comment near the top; with a variable unset, the script fails at the
first path it cannot resolve.

## Python launchers

`scripts/ns_slurm_continuation.py`, `scripts/numerical_ns/manage.py` and the NS
and VPFP campaign plugins read `PDE_GAV_SLURM_ACCOUNT` from the environment and
omit `--account` when it is unset. Scripts that start a Codex worker read
`CODEX_BINARY` and `CODEX_AUTH_FILE`, falling back to `codex` on `PATH` and
`~/.codex/auth.json`. The web panel publisher reads `PDE_PANEL_SITE_ORIGIN`.

## Historical configurations and records

The NS canonical launcher (`scripts/run_canonical_ns_campaign.py`) expands
environment variables when it loads a JSON configuration, so exporting the
variables above makes the historical NS configurations resolvable. Other
readers treat the placeholders as literal text.

JSON configurations under `configs/`, the data manifests under
`configs/paper/ns_draft_v1/data/`, and the published experiment package under
`configs/paper/plasma_transport_v2/` are records of completed runs. Their
placeholder strings are literal text. To reproduce a run, copy the configuration,
replace every placeholder with a real location, and verify the referenced data,
weights and reference cases before launching. One-off experiment scripts under
`scripts/` that embed `${PDE_GAV_RUN_ROOT}/...` paths refer to run directories of
the original campaigns and are kept for reference.
