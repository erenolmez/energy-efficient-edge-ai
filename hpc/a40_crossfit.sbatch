#!/bin/bash
#SBATCH --job-name=rcrossfit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_sapphire
#SBATCH --account=eolmez
#SBATCH --qos=normal
#SBATCH --exclude=compute-5-11,compute-7-13
#SBATCH --output=rcrossfit-%j.log
# Job 3: 5-fold cross-fit of the routers on the 5k router pool (router-train + router-calibration),
# CPU only, so it does not wait behind the A40 jobs. Never opens the test set. Steps:
#   1. train fold routers, out-of-fold scores, deployment copies (hpc.a40_crossfit --phase train)
#   2. lock on the 5k out-of-fold scores, and on the same scores restricted to calibration 2k
#   3. re-lock the existing 3k/2k scores with the same code (like-for-like reference)
#   4. random mix + budget sweep on the 5k pool (hpc.a40_baselines)
#   5. report: bias check, Brier scores, winner rule on every lock, routers vs random mix
# Accuracy-only by default. To include A40 router costs once a40_routers_v1 exists:
#   sbatch --export=ALL,ROUTER_COSTS=$STUDY_BASE/a40_routers_v1/router_costs.json hpc/a40_crossfit.sbatch
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/eolmez}"
STUDY_BASE="${STUDY_BASE:-${SCRATCH_FLASH}/clean_classifier_study}"
CIFAR_DATA="${CIFAR_DATA:-${SCRATCH_FLASH}/datasets/cifar100}"
SCREEN="${SCREEN:-${STUDY_BASE}/router_arch_screen_v1}"
STAGE_A="${STAGE_A:-${STUDY_BASE}/a40_engines_v1}"
ROUTER_COSTS="${ROUTER_COSTS:-}"
if [[ -n "$ROUTER_COSTS" ]]; then
    OLD_SCORES="${OLD_SCORES:-${STUDY_BASE}/a40_routers_v1/scores}"
    OUT="${CROSSFIT_OUT:-${STUDY_BASE}/a40_crossfit_v1}"
    COST_ARGS=(--router-costs "$ROUTER_COSTS")
else
    OLD_SCORES="${OLD_SCORES:-${STUDY_BASE}/a40_router_scores_v1/scores}"
    OUT="${CROSSFIT_OUT:-${STUDY_BASE}/a40_crossfit_accuracy_only_v1}"
    COST_ARGS=()
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f hpc/a40_crossfit.py
source hpc/a40_engines_env.sh   # overlays + import preflight; edge-ai env untouched
test -d "$CIFAR_DATA/cifar-100-python"
test -f "$STUDY_BASE/split_seed42.json"
test -f "$STAGE_A/predictions.npz" && test -f "$STAGE_A/costs_pipeline.json" && test -f "$STAGE_A/engines.json"
for f in evidence/labels.npy evidence/images.npy evidence/bank.json code/hpc_router_arch_sweep.py; do
    test -e "$SCREEN/$f" || { echo "Missing $SCREEN/$f" >&2; exit 1; }
done
ls "$OLD_SCORES"/*.npz > /dev/null
test ! -e "$OUT" || { echo "$OUT exists; set CROSSFIT_OUT to a new directory" >&2; exit 1; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
POOL=(--roles router_train router_calibration)
GROUP=(--group-by-fold)   # batches drawn within one fold (needed by the batch-set router)
# Controls and the classifier-validation ablations are reported but never selected as the winner.
CONTROL=(--exclude-from-selection direct-tiny16-val4k blended-tiny16-tval4k lookup-res32-tval4k)
OLD_CONTROL=(--exclude-from-selection direct-tiny16-val3k blended-tiny16-tval3k lookup-res32-tval3k)
# The 48k-table control is dropped: leave its old score files out of the re-lock.
OLD_FILES=(); for f in "$OLD_SCORES"/*.npz; do [[ "$f" == *t48k* ]] || OLD_FILES+=("$f"); done
hostname; date -u; git rev-parse HEAD
srun --ntasks=1 "$HPC_PYTHON" -m hpc.a40_crossfit --phase train --base "$STUDY_BASE" --data "$CIFAR_DATA" \
    --screen "$SCREEN" --stage-a "$STAGE_A" --output "$OUT" "${COST_ARGS[@]}"
"$HPC_PYTHON" -m hpc.a40_policy --stage-a "$STAGE_A" --scores "$OUT"/scores/*.npz "${POOL[@]}" "${GROUP[@]}" "${CONTROL[@]}" \
    --output "$OUT/policy_pool5k" > /dev/null
"$HPC_PYTHON" -m hpc.a40_policy --stage-a "$STAGE_A" --scores "$OUT"/scores/*.npz "${GROUP[@]}" "${CONTROL[@]}" \
    --output "$OUT/policy_cal2k" > /dev/null
"$HPC_PYTHON" -m hpc.a40_policy --stage-a "$STAGE_A" --scores "${OLD_FILES[@]}" "${OLD_CONTROL[@]}" \
    --output "$OUT/policy_old3k2k" > /dev/null
"$HPC_PYTHON" -m hpc.a40_baselines --stage-a "$STAGE_A" --scores "$OUT"/scores/*.npz "${POOL[@]}" "${GROUP[@]}" \
    --boot 100 --boot-batchings 3 --output "$OUT/baselines_pool5k" > /dev/null
"$HPC_PYTHON" -m hpc.a40_crossfit --phase report --stage-a "$STAGE_A" --crossfit "$OUT" \
    --old-policy "$OUT/policy_old3k2k/policy.json" --old-scores "$OLD_SCORES"
echo "Report: $OUT/crossfit_report.md"
date -u
