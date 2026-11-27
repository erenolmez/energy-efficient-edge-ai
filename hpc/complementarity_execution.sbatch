#!/bin/bash
#SBATCH --job-name=debatch-exec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a40
#SBATCH --account=eolmez
#SBATCH --qos=normal
#SBATCH --exclude=compute-5-11,compute-7-13
#SBATCH --output=debatch-exec-%j.log
# Job 2: debatch execution strategies (sequential, concurrent CUDA streams, per-image at
# batch 1, whole-batch). Development analysis on non-test roles only; independent of job 1.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
# Defaults follow the cluster setup used for the earlier HPC jobs; override by exporting.
# Partition: gpu_a40 by default (cost probe on the target GPU); for a faster start use
# `sbatch --partition=gpu_a100 ...` (the probe then reports A100 costs, labelled by GPU name).
SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/eolmez}"
HPC_PYTHON="${HPC_PYTHON:-${SCRATCH_FLASH}/envs/edge-ai/bin/python}"
STUDY_BASE="${STUDY_BASE:-${SCRATCH_FLASH}/clean_classifier_study}"
CIFAR_DATA="${CIFAR_DATA:-${SCRATCH_FLASH}/datasets/cifar100}"
test -x "$HPC_PYTHON"
test -d "$CIFAR_DATA/cifar-100-python"
test -f "$STUDY_BASE/split_seed42.json"
# The 128px checkpoints were trained with timm 1.0.28 from a --no-deps overlay
# ($SCRATCH_FLASH/cache/imgcmp-deps.*), not from the edge-ai env itself. Reuse that overlay
# read-only so the env other jobs use is never modified. Override with IMGCMP_DEPS=<dir>.
if [[ -z "${IMGCMP_DEPS:-}" ]]; then
    for d in "$SCRATCH_FLASH"/cache/imgcmp-deps.*; do
        if [[ -d "$d/timm" && -n "$(ls -d "$d"/timm-1.0.28.dist-info 2>/dev/null)" ]]; then IMGCMP_DEPS="$d"; break; fi
    done
fi
: "${IMGCMP_DEPS:?No timm 1.0.28 overlay found under $SCRATCH_FLASH/cache; set IMGCMP_DEPS}"
export PYTHONPATH="$IMGCMP_DEPS${PYTHONPATH:+:$PYTHONPATH}"
"$HPC_PYTHON" -B -c 'import timm, sys; print("timm", timm.__version__, timm.__file__); sys.exit(timm.__version__ != "1.0.28")'
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f hpc/complementarity_execution.py
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
out="${EXECUTION_OUT:-${STUDY_BASE}/debatch_execution_v1}"
hostname; date -u; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
git rev-parse HEAD
srun --ntasks=1 "$HPC_PYTHON" -m hpc.complementarity_execution run \
    --base "$STUDY_BASE" --data "$CIFAR_DATA" --output "$out"
"$HPC_PYTHON" -m hpc.complementarity_execution analyze \
    --predictions "$out/predictions.npz" --costs "$out/costs.json" --output "$out"
