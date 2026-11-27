#!/bin/bash
#SBATCH --job-name=a40-engines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_a40
#SBATCH --account=eolmez
#SBATCH --qos=normal
#SBATCH --exclude=compute-5-11,compute-7-13
#SBATCH --output=a40-engines-%j.log
# Job 3, stage A: TensorRT FP16 engines for the eight 128px checkpoints, engine accuracy vs
# PyTorch on the non-test roles, VRAM residency, and per-model latency + GPU energy on the A40.
# Independent of jobs 1 and 2; never opens the test set. Refuses non-A40 GPUs.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/eolmez}"
STUDY_BASE="${STUDY_BASE:-${SCRATCH_FLASH}/clean_classifier_study}"
CIFAR_DATA="${CIFAR_DATA:-${SCRATCH_FLASH}/datasets/cifar100}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f hpc/a40_engines.py
# Overlays (timm 1.0.28, TensorRT) on PYTHONPATH + import preflight; edge-ai env untouched.
source hpc/a40_engines_env.sh
test -d "$CIFAR_DATA/cifar-100-python"
test -f "$STUDY_BASE/split_seed42.json"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
out="${A40_OUT:-${STUDY_BASE}/a40_engines_v1}"
hostname; date -u; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version --format=csv
git rev-parse HEAD
srun --ntasks=1 "$HPC_PYTHON" -m hpc.a40_engines run \
    --base "$STUDY_BASE" --data "$CIFAR_DATA" --output "$out" ${A40_ARGS:-}
date -u
