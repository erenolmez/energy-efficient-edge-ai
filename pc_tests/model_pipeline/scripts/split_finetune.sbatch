#!/usr/bin/env bash
#SBATCH --job-name=adam-prune
#SBATCH --partition=gpu_a40_ext
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
set -euo pipefail
root=${1:?Provide the absolute extracted bundle path}
py=${2:?Provide the absolute Python executable path}
data=${3:?Provide the absolute CIFAR-100 dataset directory}
cd "$root"
exec "$py" -B -u split_finetune.py worker --root "$root" --handoff "$root/handoff.json" \
  --data "$data" --out "$root/results" --owner "hpc${SLURM_ARRAY_TASK_ID:?Submit as an array}"
