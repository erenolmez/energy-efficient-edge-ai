#!/bin/bash
#SBATCH --job-name=classifier-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=classifier-pilot-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
: "${SCRATCH_FLASH:?Set the allocated scratch directory}"
: "${HPC_PYTHON:?Set the absolute path to the project Python executable}"
test -x "$HPC_PYTHON"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f hpc/train_classifier_pilot.py
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCH_HOME="${SCRATCH_FLASH}/cache/torch"
data_dir="${SCRATCH_FLASH}/datasets/cifar100"
split_manifest="${SCRATCH_FLASH}/clean_classifier_study/split_seed42.json"
run_dir="${PILOT_RUN_DIR:-${SCRATCH_FLASH}/clean_classifier_study/pilot-${SLURM_JOB_ID}}"
resume_args=()
if [[ "${PILOT_RESUME:-0}" == "1" ]]; then
    : "${PILOT_RUN_DIR:?Set the existing run directory for a resume}"
    resume_args=(--resume)
fi
hostname
date -u
nvidia-smi
# Runtime-only records: no private cluster names or account paths are committed.
mkdir -p "$run_dir"
nvidia-smi -q > "$run_dir/gpu-${SLURM_JOB_ID}.txt"
"$HPC_PYTHON" -m pip freeze > "$run_dir/packages-${SLURM_JOB_ID}.txt"
srun --ntasks=1 "$HPC_PYTHON" hpc/train_classifier_pilot.py \
    --data-dir "$data_dir" --split-manifest "$split_manifest" \
    --output-dir "$run_dir" --epochs 3 --batch-size 128 --workers 4 \
    --seed 42 --split-seed 42 --download "${resume_args[@]}"
