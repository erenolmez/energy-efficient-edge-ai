#!/bin/bash
#SBATCH --job-name=scratch-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=scratch-prepare-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
: "${SCRATCH_FLASH:?Set scratch storage}"
: "${HPC_PYTHON:?Set the project Python executable}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
if [[ "${INSTALL_TRACKING:-0}" == "1" ]]; then
    "$HPC_PYTHON" -m pip install --requirement hpc/requirements-tracking.txt
fi
study_dir="${SCRATCH_FLASH}/clean_classifier_study/scratch32_resnet18_v1"
action="${STUDY_ACTION:-prepare}"
case "$action" in
    prepare|rank|finalize) ;;
    *) echo "CPU job action must be prepare, rank or finalize" >&2; exit 2 ;;
esac
srun --ntasks=1 "$HPC_PYTHON" hpc/train_scratch_sweep.py "$action" \
    --study-dir "$study_dir" --data-dir "${SCRATCH_FLASH}/datasets/cifar100" \
    --split-manifest "${SCRATCH_FLASH}/clean_classifier_study/split_seed42.json"
