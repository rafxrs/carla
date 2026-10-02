#!/bin/bash
# Usage: sbatch --array=0-2 scripts/slurm/train.sbatch configs/experiments/gnn.yaml [key=value ...]
# One task per seed (array index); rollout workers use all but two of the allocated cores.
#SBATCH --job-name=carla
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x-%A_%a.out
##SBATCH --gres=gpu:1            # uncomment if the partition has GPUs
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
config="$1"
shift
carla train "${config}" \
  seed="${SLURM_ARRAY_TASK_ID:-0}" \
  runtime.num_workers=$((SLURM_CPUS_PER_TASK - 2)) \
  "$@"
