# GPU preflight isolation, execution revision2

Observed allocated srun probes passed on all four GPUs of one A40 node but
failed to create a CUDA context on a different node's GPU1. The shared
preflight stop made the passing workers exit without claiming cases.
The underlying busy-device cause is still not proven; do not change compute
modes, kill other users' processes, or treat hardware exclusion as repair.

Revision2 keeps the frozen trainer, plan, original worker and revision1 files
unchanged. Only a failure before claiming any case is local to the worker.
It writes a separate preflight_failures/job.json and exits without retry.
Actual unexpected training failures still create the shared recovery stop.

The rearm audit requires:

- The original revision1 recovery authority, source hashes and archived case2.
- No live revision1 GPU workers and no claims created by their job IDs.
- Completed checkpoint/history re-verification and untouched live legacy claims.
- Exactly identified probe logs: at least one passed; all failed probes have
  the diagnosed busy/unavailable error and their nodes are explicitly excluded.
- A shared stop that represents only the prior preflight failure, not another
  training, filesystem, source or numerical issue.

It archives only that exact shared preflight stop, preserving its hash and
source lineage in recoveries/gpu_step_v2/receipt.json. Original stop and
case2 failure archives remain intact. Nothing is deleted. Existing training
on the excluded node is not stopped. New jobs exclude it through Slurm and
the worker verifies its actual hostname against the receipt.

The submission command uses snapshot currently unallocated eligible capacity
outside the excluded nodes, with one queued A40 job if none is visible.
Each GPU job requests one device and24h. Scheduling acceptance is not proof
of execution, CUDA availability, successful model training or worker advancement.
The CPU audit must pass before dependent GPU jobs start.
