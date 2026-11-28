#!/bin/bash
# Re-run the cross-fit locks, the random-mix baseline and the report on an existing cross-fit
# output, without retraining (CPU, a few minutes). Test set never read.
#   bash hpc/a40_crossfit_relock.sh <crossfit output dir> [classes per batch K] [fold]
# With no second argument the shuffled-batch results in <dir> are rewritten in place. With K, results go
# to <dir>/coherent<K>/ (each batch holds K classes: a non-i.i.d. stream, where the router can matter more).
# Shuffled locks draw every batch within one cross-fit fold (needed by the batch-set router). Coherent
# locks draw across folds so B=64/128 can be formed (50 pool images per class, 10 per fold); the
# batch-set router is left out there. With "fold" as third argument, coherent batches are also drawn
# within folds (all routers incl. batch-set, but only B <= 32) into <dir>/coherent<K>_fold/.
# Run inside an allocation, e.g. srun -p cpu_sapphire -c 8 --mem=16G --time=01:00:00 bash hpc/a40_crossfit_relock.sh ...
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm allocation (srun)}"
cd "$(dirname "$(readlink -f "$0")")/.."   # repo root, whatever the working directory
CF="${1:?crossfit output dir}"; K="${2:-}"; WITHIN="${3:-}"
SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/eolmez}"
STUDY_BASE="${STUDY_BASE:-${SCRATCH_FLASH}/clean_classifier_study}"
STAGE_A="${STAGE_A:-${STUDY_BASE}/a40_engines_v1}"
source hpc/a40_engines_env.sh > /dev/null
test -f "$CF/crossfit.json" && ls "$CF"/scores/*.npz > /dev/null
if grep -q '"router_costs": null' "$CF/crossfit.json"; then OLD_SCORES="${OLD_SCORES:-${STUDY_BASE}/a40_router_scores_v1/scores}"
else OLD_SCORES="${OLD_SCORES:-${STUDY_BASE}/a40_routers_v1/scores}"; fi
OUT="$CF"; COH=(); GROUP=(--group-by-fold); SKIP_SET=""
if [[ -n "$K" ]]; then
    OUT="$CF/coherent$K"; COH=(--coherent "$K")
    if [[ "$WITHIN" == fold ]]; then OUT="${OUT}_fold"; else GROUP=(); SKIP_SET=1; fi
    mkdir -p "$OUT"
    ln -sfn "$(readlink -f "$CF/scores")" "$OUT/scores"; ln -sfn "$(readlink -f "$CF/crossfit.json")" "$OUT/crossfit.json"
fi
# EXTRA_SCORES: optional directory of more out-of-fold score files on the same 5k pool and folds
# (e.g. fast_feature_routers_v1/scores); they are locked and reported alongside.
EXTRA=(); if [[ -n "${EXTRA_SCORES:-}" ]]; then EXTRA=("$EXTRA_SCORES"/*.npz); fi
NEW=(); for f in "$CF"/scores/*.npz "${EXTRA[@]}"; do   # 48k/49k table control dropped
    [[ "$f" == *t49k* ]] && continue; [[ -n "$SKIP_SET" && "$(basename "$f")" == set-* ]] && continue; NEW+=("$f"); done
OLD=(); for f in "$OLD_SCORES"/*.npz; do [[ "$f" == *t48k* ]] || OLD+=("$f"); done
POOL=(--roles router_train router_calibration)
CONTROL=(--exclude-from-selection direct-tiny16-val4k blended-tiny16-tval4k lookup-res32-tval4k)
OLD_CONTROL=(--exclude-from-selection direct-tiny16-val3k blended-tiny16-tval3k lookup-res32-tval3k)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$HPC_PYTHON" -m hpc.a40_policy --stage-a "$STAGE_A" --scores "${NEW[@]}" "${POOL[@]}" "${GROUP[@]}" "${CONTROL[@]}" "${COH[@]}" --output "$OUT/policy_pool5k" > /dev/null
"$HPC_PYTHON" -m hpc.a40_policy --stage-a "$STAGE_A" --scores "${NEW[@]}" "${GROUP[@]}" "${CONTROL[@]}" "${COH[@]}" --output "$OUT/policy_cal2k" > /dev/null
"$HPC_PYTHON" -m hpc.a40_policy --stage-a "$STAGE_A" --scores "${OLD[@]}" "${OLD_CONTROL[@]}" "${COH[@]}" --output "$OUT/policy_old3k2k" > /dev/null
"$HPC_PYTHON" -m hpc.a40_baselines --stage-a "$STAGE_A" --scores "${NEW[@]}" "${POOL[@]}" "${GROUP[@]}" "${COH[@]}" \
    --boot 100 --boot-batchings 3 --output "$OUT/baselines_pool5k" > /dev/null
"$HPC_PYTHON" -m hpc.a40_crossfit --phase report --stage-a "$STAGE_A" --crossfit "$OUT" \
    --old-policy "$OUT/policy_old3k2k/policy.json" --old-scores "$OLD_SCORES" \
    ${EXTRA_SCORES:+--extra-scores "$EXTRA_SCORES"} > /dev/null
echo "Report: $OUT/crossfit_report.md"
