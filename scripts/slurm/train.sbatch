#!/bin/bash
# Usage: sbatch --array=0-2 scripts/slurm/train.sbatch configs/experiments/ppo_mlp_baseline.yaml [key=value ...]
# One independent synchronous MLP PPO run per seed (array index).
#SBATCH --job-name=carla
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
  "$@"
