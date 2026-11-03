# Versioned GPU-step recovery

The observed case2 ResNet50 attempt failed at its first model.cuda() with
CUDA-capable device(s) busy or unavailable. No epoch, run.json or checkpoint
was saved. One subsequent allocated srun CUDA context/allocation/sync test
passed. This supports testing explicit job-step execution; it does not prove
the failed node/device root cause or general GPU availability.

Do not edit or remove the original stop marker. Legacy workers must finish
their current case and exit. New GPU-step workers require a separate verified
receipt, use explicit srun GPU binding, and check real allocation before claims.
The preflight is a child process that fully exits before the trainer starts,
avoiding retention of a competing CUDA context by the worker itself.

CPU preparation verifies the unchanged plan, all original frozen source hashes,
completed case0/1/4 and any other completed cases, full epoch histories and
checkpoint hashes. Active claims must have scheduler-visible owners and are
untouched. Any other failed/paused/interrupted claim blocks recovery.

Only the diagnosed failed case2 with exactly a train.log and no saved model/run
state may be archived. Both original log directory and claim directory are
moved to explicit subdirectories of recoveries/gpu_step_v1; checksums and source
lineage are recorded. Nothing is deleted, and the original stop stays in place.
Case2 then becomes eligible for one explicitly authorized retry after preflight.

New worker files and launcher are separately hashed. New failures set a new
recovery stop; a changed original marker also blocks further claims.
Numerical nonfinite losses are retained according to the original protocol.
No hyperparameters, split, initialization, training epochs, trainer code or
selection rules change. The old three results are not rerun; the currently
running ResNet101 case is not stopped or copied.

The submit helper queues the CPU audit and workers for snapshot unallocated
eligible H200/A100/A40 capacity. If no capacity is visible, it explicitly
queues one A40 worker. Starts are subject to scheduler policy, not guaranteed.
The submission receipt prevents accidental repeats; audit failures block
dependent GPU jobs. These allocations also have a24-hour limit. This recovery
does not authorize deleting locks or resuming any later interrupted training.
