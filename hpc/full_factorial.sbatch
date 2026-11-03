#!/bin/bash
#SBATCH --job-name=full-grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=full-grid-%j.log
set -euo pipefail
: "${HPC_PYTHON:?Set project Python}"
: "${SCRATCH_FLASH:?Set scratch storage}"
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
# exec makes the signalled batch process the worker itself. It forwards USR1
# to the current trainer, which saves its completed epoch before stopping.
exec "$HPC_PYTHON" hpc/full_factorial_worker.py \
    --study-dir "${SCRATCH_FLASH}/clean_classifier_study/scratch32_full_factorial_v1" \
    --data-dir "${SCRATCH_FLASH}/datasets/cifar100" \
    --split-manifest "${SCRATCH_FLASH}/clean_classifier_study/split_seed42.json"
