#!/bin/bash
#SBATCH --job-name=grid-gpu-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=grid-gpu-v2-%j.log
set -euo pipefail
: "${HPC_PYTHON:?Set project Python}"
: "${SCRATCH_FLASH:?Set scratch storage}"
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec srun --ntasks=1 --gres=gpu:1 --gpu-bind=single:1 \
    "$HPC_PYTHON" hpc/full_factorial_worker_gpu_step_v2.py \
    --study-dir "${SCRATCH_FLASH}/clean_classifier_study/scratch32_full_factorial_v1" \
    --data-dir "${SCRATCH_FLASH}/datasets/cifar100" \
    --split-manifest "${SCRATCH_FLASH}/clean_classifier_study/split_seed42.json"
