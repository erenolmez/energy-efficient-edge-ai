# Explicit paused-case continuation

Prepared locally; no remote file transfer or job submission performed by the agent.
Only the new `hpc/continue_paused_grid_20261104.py` needs copying to the existing HPC repository.
Do not pull/overwrite frozen training/recovery sources, delete claims, or remove stop markers.

User-pasted remote evidence established an empty queue, four graceful checkpoint stops,
matching checkpoint/source/recipe/environment checks, passing `verify_rearm`, and absence
of the **shared** `recoveries/gpu_step_v1/stop.json`. The precise scheduler stop origin
remains unconfirmed because the accounting database was inaccessible.

Four explicitly scoped cases: 154/178 epochs, 155/94, 158/32, 159/9. No failed cases retried.
Each resumes with the unchanged trainer's `--resume` and original online W&B identity,
then audits its completed checkpoint before handing off to the unchanged revision-2 worker.
Frozen trainer/protocol/phase/data splits remain unchanged. No test/router evaluation.

Submission requires an empty user queue, current recovery authority, unchanged inputs,
no trainer locks, the original public tracking destination and pinned SDK. Four independent
24-hour one-GPU/4-CPU/16GiB jobs use the chosen verified partition (default A40), explicitly
exclude compute-5-11, and retain the original 180-second checkpoint signal. This is not a
claim about current capacity, guaranteed starts, or a personal GPU maximum.

Jobs are submitted held, all exact job IDs recorded, then released. The exclusive campaign
receipt and per-case attempt directories prevent duplication/retry. Partial submissions
or release failures require inspection; held jobs are NOT automatically cancelled/released.
Original owners remain immutable. Earlier checkpoint, summary, original run/log, tracking
identity and claim records are copied and hash-verified before resume; the original claim
result is updated with a continuation link only after the trainer returns. Existing source
receipts and legacy stops are retained. Expect approximately 1GiB extra preserved checkpoint
storage, subject to actual file sizes and user quota; the wrapper also checks free space.

Worker-local CUDA preflight failure does not claim/resume or retry; unexpected source,
storage, training or audit failures latch the shared stop without overwriting prior evidence.
A fresh checkpointed allocation timeout is recorded paused, never automatically resubmitted.
No automated repeated allocation loop is installed.

After copying only the new helper, run on HPC from the existing repository:

```bash
export HPC_PYTHON="/mnt/beegfs/eolmez/envs/edge-ai/bin/python"
export SCRATCH_TRACKING_MODE=online
export WANDB_ENTITY=erenolmez-politecnico-di-torino
export WANDB_PROJECT=energy-efficient-edge-ai
"$HPC_PYTHON" -B hpc/continue_paused_grid_20261104.py submit
```

This is read-only. If it passes and the user is ready to submit the four jobs:

```bash
"$HPC_PYTHON" -B hpc/continue_paused_grid_20261104.py submit --submit
squeue -u "$USER"
```

Keep submission output and logs `grid-cont-<job>.log`. Campaign receipts are under
`$SCRATCH_FLASH/clean_classifier_study/scratch32_full_factorial_v1/recoveries/paused_continuation_20261104_v1`.
Do not rerun submission after any ambiguous response. The five older failed claims remain
separate unresolved dispositions; this continuation does not make the full study complete.

Local CPU-only unit tests do not prove live Slurm/CUDA/W&B behavior; these require inspection
of actual new scheduler IDs, preflight logs, resumed epochs, and automatic advancement.
