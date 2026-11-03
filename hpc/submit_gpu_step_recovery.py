"""Submit CPU audit and explicitly GPU-bound recovery workers for the existing grid."""
from __future__ import annotations
import argparse
from collections import Counter
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.clean_protocol import write_json
from hpc.gpu_step_recovery import recovery_paths
from hpc.scale_full_factorial import capacity
from hpc.submit_full_factorial import submit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--cpu-partition", default="cpu_sapphire")
    args = p.parse_args()
    for key in ("HPC_PYTHON", "SCRATCH_FLASH", "WANDB_ENTITY", "WANDB_PROJECT"):
        if not os.environ.get(key):
            p.error("Missing environment: " + key)
    if os.environ.get("SCRATCH_TRACKING_MODE") != "online":
        p.error("Set online tracking explicitly")
    root = Path(os.environ["SCRATCH_FLASH"]) / "clean_classifier_study/scratch32_full_factorial_v1"
    # Audit handles all state mutations in a CPU allocation, never on login node.
    marker = json.loads((root / "investigate_stop.json").read_text())
    if marker.get("case_id") != 2 or marker.get("returncode") != 1:
        p.error("This recovery is only for the diagnosed case2 CUDA access error")
    priorities = ["gpu_h200", "gpu_a100", "gpu_a40"]
    user = getpass.getuser()
    account = os.environ.get("HPC_ACCOUNT", user)
    associations = subprocess.check_output(
        ["scontrol", "show", "assoc_mgr", "flags=assoc", "users="+user, "accounts="+account],
        text=True)
    from hpc.scale_full_factorial import fields
    allowed = {fields(line).get("Partition") for line in associations.splitlines()
               if "UserName=" + user + "(" in line or "UserName=" + user + " " in line}
    priorities = [name for name in priorities if name in allowed]
    if "gpu_a40" not in priorities:
        p.error("Could not verify GPU partition association")
    nodes = subprocess.check_output(["scontrol", "show", "nodes", "-o"], text=True)
    slots, details = capacity(nodes, priorities)
    requests = [name for name in priorities for _ in range(slots[name])]
    if not requests:
        requests = ["gpu_a40"]  # One queued request, explicitly disclosed, never a busy wait.
    print(json.dumps(dict(request_counts=dict(Counter(requests)), snapshot=details,
                         worker_count=len(requests),
                         note="Snapshot, not guaranteed allocation; old in-flight cases remain untouched."), indent=2))
    if not args.submit:
        print("Dry run only")
        return
    recovery = recovery_paths(root)
    recovery.mkdir(parents=True, exist_ok=True)
    path = recovery / "submission.json"
    receipt = dict(status="submitting", jobs=[], requests=dict(Counter(requests)))
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream)
    command = shlex.join([sys.executable, str(ROOT/"hpc/gpu_step_recovery.py"),
                          "--study-dir", str(root)])
    exclude = ["--exclude="+os.environ["FULL_GRID_EXCLUDE_CPU"]] if os.environ.get("FULL_GRID_EXCLUDE_CPU") else []
    try:
        prep = submit(["sbatch", "--parsable", "--partition="+args.cpu_partition,
                       *exclude, "--job-name=grid-recovery-audit", "--nodes=1", "--ntasks=1",
                       "--cpus-per-task=2", "--mem=4G", "--time=00:20:00", "--export=ALL",
                       "--output=grid-recovery-audit-%j.log",
                       "--wrap=export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2; exec " + command])
        receipt["jobs"].append(dict(role="audit", job_id=prep))
        write_json(path, receipt)
        for index, partition in enumerate(requests):
            jid = submit(["sbatch", "--parsable", "--partition="+partition,
                          "--dependency=afterok:"+prep, "--kill-on-invalid-dep=yes",
                          "--export=ALL", f"--job-name=grid-gpu-step-{index}",
                          "hpc/full_factorial_gpu_step.sbatch"])
            receipt["jobs"].append(dict(role="worker", job_id=jid, partition=partition))
            write_json(path, receipt)
        receipt["status"] = "submitted"
        write_json(path, receipt)
    except Exception:
        receipt["status"] = "partial_submission_inspect"
        write_json(path, receipt)
        raise
    print(json.dumps(receipt, indent=2))
    print("Do not repeat submission. Original stop marker retained for legacy workers. "
          "After audit, new workers check CUDA before claiming cases. Allocations24h.")


if __name__ == "__main__":
    main()
