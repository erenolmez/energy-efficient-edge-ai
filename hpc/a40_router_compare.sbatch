#!/bin/bash
#SBATCH --job-name=a40-router-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_sapphire
#SBATCH --exclude=compute-3-9,compute-6-9
# Final router comparison: numpy re-analysis of numbers already measured on the A40 (no new
# measurement, so CPU is enough). Locks every router from a40_crossfit_v2 together with the fast
# feature and Taylor routers (fast_feature_routers_v1, same 5k pool and folds), the random mix and
# the oracle, on shuffled and 4-classes-per-batch streams. Test set never read.
# Submit after both A40 jobs: --dependency=afterok:<newrouters job>:<fast feature job>.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/eolmez}"
STUDY_BASE="${STUDY_BASE:-${SCRATCH_FLASH}/clean_classifier_study}"
CF="${CROSSFIT_OUT:-${STUDY_BASE}/a40_crossfit_v2}"
export EXTRA_SCORES="${EXTRA_SCORES:-${STUDY_BASE}/fast_feature_routers_v1/scores}"
REPO="$(cd "${SLURM_SUBMIT_DIR:?}" && pwd)"
test -f "$CF/router_costs_new.json" || { echo "New router A40 costs missing in $CF" >&2; exit 1; }
ls "$EXTRA_SCORES"/*.npz > /dev/null
bash "$REPO/hpc/a40_crossfit_relock.sh" "$CF"
bash "$REPO/hpc/a40_crossfit_relock.sh" "$CF" 4
bash "$REPO/hpc/a40_crossfit_relock.sh" "$CF" 4 fold
echo "Reports: $CF/crossfit_report.md, $CF/coherent4/crossfit_report.md, $CF/coherent4_fold/crossfit_report.md"
