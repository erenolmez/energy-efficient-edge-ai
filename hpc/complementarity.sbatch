#!/bin/bash
#SBATCH --job-name=complementarity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=complementarity-%j.log
# Development analysis: predictions on classifier-validation, router-train and
# router-calibration only (never the official test set), then CPU-side oracle analysis.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
: "${HPC_PYTHON:?Set the absolute path to the project Python executable}"
: "${STUDY_BASE:?Set the clean_classifier_study directory}"
: "${CIFAR_DATA:?Set the CIFAR-100 root used by the classifier training}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f hpc/complementarity_analysis.py
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
out="${COMPLEMENTARITY_OUT:-${STUDY_BASE}/complementarity_v1}"
hostname; date -u; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
git rev-parse HEAD
srun --ntasks=1 "$HPC_PYTHON" -m hpc.complementarity_analysis collect \
    --base "$STUDY_BASE" --data "$CIFAR_DATA" --output "$out"
"$HPC_PYTHON" -m hpc.complementarity_analysis analyze \
    --predictions "$out/predictions.npz" --costs "$out/costs.json" --output "$out"
