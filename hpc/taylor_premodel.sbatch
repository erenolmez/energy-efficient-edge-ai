#!/bin/bash
#SBATCH --job-name=taylor-premodel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_sapphire
#SBATCH --account=eolmez
#SBATCH --qos=normal
#SBATCH --output=taylor-premodel-%j.log
# Development screening, CPU only: Taylor et al. (LCTES 2018) KNN-chain premodel reproduced on the
# eight-model HPC bank, fit on val+3k, scored on router-calibration 2k, then locked per batch with
# hpc.a40_policy next to the a40_routers scores (if present). Never opens the test set.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/eolmez}"
HPC_PYTHON="${HPC_PYTHON:-${SCRATCH_FLASH}/envs/edge-ai/bin/python}"
STUDY_BASE="${STUDY_BASE:-${SCRATCH_FLASH}/clean_classifier_study}"
SCREEN="${SCREEN:-${STUDY_BASE}/router_arch_screen_v1}"
STAGE_A="${STAGE_A:-${STUDY_BASE}/a40_engines_v1}"
ROUTER_SCORES="${ROUTER_SCORES:-${STUDY_BASE}/a40_routers_v1/scores}"
FEATURE_DEPS="${FEATURE_DEPS:-${SCRATCH_FLASH}/envs/feature-overlay}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f hpc/taylor_premodel.py && test -x "$HPC_PYTHON"
test -f "$STUDY_BASE/split_seed42.json"
test -f "$STAGE_A/predictions.npz" && test -f "$STAGE_A/costs_pipeline.json"
for f in images.npy labels.npy split.json bank.json; do
    test -e "$SCREEN/evidence/$f" || { echo "Missing $SCREEN/evidence/$f" >&2; exit 1; }
done
test -d "$FEATURE_DEPS/sklearn" && test -d "$FEATURE_DEPS/scipy" || { echo "Missing overlay: $FEATURE_DEPS" >&2; exit 1; }
export PYTHONPATH="${FEATURE_DEPS}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
"$HPC_PYTHON" -c "import numpy, scipy, sklearn, hpc.a40_policy, hpc.taylor_premodel; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'sklearn', sklearn.__version__)"
out="${TAYLOR_OUT:-${STUDY_BASE}/taylor_premodel_v1}"
hostname; date -u; lscpu | grep -m1 "Model name" || true
git rev-parse HEAD
srun --ntasks=1 "$HPC_PYTHON" -m hpc.taylor_premodel --base "$STUDY_BASE" --evidence "$SCREEN/evidence" \
    --stage-a "$STAGE_A" --output "$out"
scores=("$out"/scores/*.npz)
if compgen -G "$ROUTER_SCORES/*.npz" > /dev/null; then scores+=("$ROUTER_SCORES"/*.npz); else echo "No a40_routers scores yet: locking Taylor only"; fi
"$HPC_PYTHON" -m hpc.a40_policy --stage-a "$STAGE_A" --scores "${scores[@]}" --output "$out/policy"
date -u
