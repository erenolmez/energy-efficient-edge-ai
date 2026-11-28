#!/bin/bash
#SBATCH --job-name=fast-feature-routers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --partition=gpu_a40
#SBATCH --account=eolmez
#SBATCH --qos=normal
#SBATCH --exclude=compute-5-11,compute-7-13
#SBATCH --output=fast-feature-routers-%j.log
# Fast (CUDA-graph) hand-made-feature and Taylor routers, 5-fold cross-fit on the 5k router pool
# (router-train + router-calibration; no classifier-validation images), A40 router cost per batch.
# Writes a40_policy score files for the final comparison. The test set is never read.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/eolmez}"
HPC_PYTHON="${HPC_PYTHON:-${SCRATCH_FLASH}/envs/edge-ai/bin/python}"
STUDY_BASE="${STUDY_BASE:-${SCRATCH_FLASH}/clean_classifier_study}"
CIFAR_DATA="${CIFAR_DATA:-${SCRATCH_FLASH}/datasets/cifar100}"
SCREEN="${SCREEN:-${STUDY_BASE}/router_arch_screen_v1}"
STAGE_A="${STAGE_A:-${STUDY_BASE}/a40_engines_v1}"
FEATURE_DEPS="${FEATURE_DEPS:-${SCRATCH_FLASH}/envs/feature-overlay}"
TRT_OVERLAY="${TRT_OVERLAY:-${SCRATCH_FLASH}/envs/trt-overlay}"   # pynvml only
OUT="${FAST_ROUTERS_OUT:-${STUDY_BASE}/fast_feature_routers_v1}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f hpc/fast_feature_routers.py && test -x "$HPC_PYTHON"
test ! -e "$OUT" || { echo "$OUT exists; set FAST_ROUTERS_OUT to a new directory" >&2; exit 1; }
for f in "$STUDY_BASE/feature_router_ablation_v1/features_compact.npz" "$STUDY_BASE/taylor_premodel_v1/features29.npz" \
         "$STAGE_A/predictions.npz" "$CIFAR_DATA/cifar-100-python/train"; do
    test -f "$f" || { echo "Missing $f" >&2; exit 1; }
done
export PYTHONPATH="${FEATURE_DEPS}:${TRT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$HPC_PYTHON" -c "import torch, numpy, scipy, sklearn, xgboost, pynvml, hpc.fast_feature_routers; print('torch', torch.__version__, 'cuda', torch.ones(16, device='cuda').sum().item(), 'xgboost', xgboost.__version__)"
hostname; date -u; nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
git rev-parse HEAD
srun --ntasks=1 "$HPC_PYTHON" -m hpc.fast_feature_routers --base "$STUDY_BASE" --evidence "$SCREEN/evidence" \
    --data "$CIFAR_DATA" --stage-a "$STAGE_A" --feature-ablation "$STUDY_BASE/feature_router_ablation_v1" \
    --taylor "$STUDY_BASE/taylor_premodel_v1" --output "$OUT"
ls -l "$OUT/scores"
date -u
