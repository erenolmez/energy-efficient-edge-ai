#!/bin/bash
#SBATCH --job-name=feature-routers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_a40
#SBATCH --account=eolmez
#SBATCH --qos=normal
#SBATCH --output=feature-routers-%j.log
# Development screening, CPU only (no GPU requested): hand-made image features vs the TinyCNN
# correctness router on the eight-model HPC bank. Fits on val+3k (and 3k as a control), scores on
# router-calibration 2k. Never opens the test set. Override the partition with
# `sbatch --partition=<cpu partition> hpc/feature_router_ablation.sbatch` if one exists.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/eolmez}"
HPC_PYTHON="${HPC_PYTHON:-${SCRATCH_FLASH}/envs/edge-ai/bin/python}"
STUDY_BASE="${STUDY_BASE:-${SCRATCH_FLASH}/clean_classifier_study}"
SCREEN="${SCREEN:-${STUDY_BASE}/router_arch_screen_v1}"
ROUTER_SCORES="${ROUTER_SCORES:-${STUDY_BASE}/a40_routers_v1/scores}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f hpc/feature_router_ablation.py && test -x "$HPC_PYTHON"
test -f "$STUDY_BASE/split_seed42.json"
for f in images.npy labels.npy split.json bank.json; do
    test -e "$SCREEN/evidence/$f" || { echo "Missing $SCREEN/evidence/$f" >&2; exit 1; }
done
# Extra Python packages may live in a --no-deps overlay; set FEATURE_DEPS=<dir> to use one.
[[ -n "${FEATURE_DEPS:-}" ]] && export PYTHONPATH="${FEATURE_DEPS}${PYTHONPATH:+:${PYTHONPATH}}"
"$HPC_PYTHON" -c "import numpy, sklearn; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__)"
"$HPC_PYTHON" -c "import xgboost; print('xgboost', xgboost.__version__)" \
    || echo "xgboost unavailable: running the sklearn HistGradientBoosting fallback only"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
out="${FEATURES_OUT:-${STUDY_BASE}/feature_router_ablation_v1}"
hostname; date -u; lscpu | grep -m1 "Model name" || true
git rev-parse HEAD
srun --ntasks=1 "$HPC_PYTHON" -m hpc.feature_router_ablation --base "$STUDY_BASE" \
    --evidence "$SCREEN/evidence" --router-scores "$ROUTER_SCORES" --output "$out" \
    --threads "${SLURM_CPUS_PER_TASK:-8}"
date -u
