#!/bin/bash
#SBATCH --job-name=scratch-search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@180
#SBATCH --array=0-11%2
#SBATCH --output=scratch-search-%A_%a.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array, including a single-task array for preflight}"
: "${SCRATCH_FLASH:?Set scratch storage}"
: "${HPC_PYTHON:?Set the project Python executable}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
phase="${STUDY_PHASE:-search}"
case "$phase" in
    search|confirm) ;;
    *) echo "GPU phase must be search or confirm" >&2; exit 2 ;;
esac
resume_args=()
if [[ "${STUDY_RESUME:-0}" == "1" ]]; then
    resume_args=(--resume)
fi
hostname
date -u
nvidia-smi
srun --ntasks=1 "$HPC_PYTHON" hpc/train_scratch_sweep.py train \
    --study-dir "${SCRATCH_FLASH}/clean_classifier_study/scratch32_resnet18_v1" \
    --data-dir "${SCRATCH_FLASH}/datasets/cifar100" \
    --split-manifest "${SCRATCH_FLASH}/clean_classifier_study/split_seed42.json" \
    --phase "$phase" --task-id "$SLURM_ARRAY_TASK_ID" \
    --epochs-this-job "${EPOCHS_THIS_JOB:-200}" "${resume_args[@]}"
