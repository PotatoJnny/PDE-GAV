#!/usr/bin/env bash
# Site-specific Slurm options (account, node excludes, working directory) are not set here; pass them on the job submission command line.
# Build the small, public-data part of the VPFP long-time example.
#
# Submit from the repository root:
#   sbatch legacy/vpfp/scripts/slurm_prepare_vpfp_example_data.sh
#
# The script creates:
#   1. the two direct Fourier--Hermite generator checkpoints;
#   2. 16 development cases and 32 released cases with numerical answers.
#
# Dense arrays are created in SLURM_TMPDIR.  Only the compact example dataset
# is kept under project_runs/, so this works even when shared scratch is full.

#SBATCH --qos=normal
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=vpfp_example_data
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

source /etc/profile
module load StdEnv/2023 python/3.11 scipy-stack/2024b

set -euo pipefail

project_root="${VPFP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
storage_root="${VPFP_RUN_STORAGE_ROOT:-${project_root}/project_runs}"
python_bin="${VPFP_PYTHON:-${project_root}/.venv/bin/python}"
checkpoint_root="${storage_root}/_inputs/vpfp/strong/generator/time_controls_033"
release_bank_root="${storage_root}/_inputs/vpfp/strong/release_bank"

test -n "${SLURM_JOB_ID:-}"
test -n "${SLURM_TMPDIR:-}"
test -x "${python_bin}"
mkdir -p "${storage_root}/_inputs/vpfp/strong"

export VPFP_RUN_STORAGE_ROOT="${storage_root}"
export PYTHONPATH="${project_root}/legacy/vpfp/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export PYTHONUNBUFFERED=1

cd "${project_root}"

if [[ ! -f "${checkpoint_root}/checkpoint_report.json" ]]; then
  if [[ -e "${checkpoint_root}" ]]; then
    echo "Checkpoint folder is incomplete: ${checkpoint_root}" >&2
    exit 2
  fi
  "${python_bin}" legacy/vpfp/scripts/create_vpfp_direct_generator_checkpoints.py \
    --out "${checkpoint_root}" \
    --collision-nu-min 10 \
    --collision-nu-max 30 \
    --temperature-min 0.8 \
    --temperature-max 1.2 \
    --t1 1.0 \
    --time-controls 33 \
    --hermite-degree 14
fi

if [[ -f "${release_bank_root}/release_bank_report.json" ]]; then
  "${python_bin}" -c \
    'import json,sys; report=json.load(open(sys.argv[1])); assert report["passed"] is True' \
    "${release_bank_root}/release_bank_report.json"
  echo "VPFP example dataset is already complete: ${release_bank_root}"
  exit 0
fi

if [[ -e "${release_bank_root}" ]]; then
  echo "Release-bank folder is incomplete: ${release_bank_root}" >&2
  exit 2
fi

srun "${python_bin}" legacy/vpfp/scripts/build_vpfp_long_time_answer_first_release_bank_v2.py \
  --decoder-checkpoint "${checkpoint_root}/base_online_decoder.pt" \
  --out "${release_bank_root}" \
  --regime-id long_time_strong_collision_v2 \
  --collision-nu-min 10 \
  --collision-nu-max 30 \
  --temperature-min 0.8 \
  --temperature-max 1.2 \
  --seed-start 7500000 \
  --development-count 16 \
  --released-count 32 \
  --generation-batch-size 16 \
  --device cuda
