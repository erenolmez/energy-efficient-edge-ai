"""Request currently unallocated eligible GPU capacity for an EXISTING grid."""
from __future__ import annotations
import argparse
from collections import Counter
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.clean_protocol import file_hash, write_json
from hpc.full_factorial_protocol import checked_plan


def fields(line):
    return dict(re.findall(r"(\w+)=([^\s]*)", line))


def gpu_count(tres):
    found = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
    return int(found.group(1)) if found else 0


def capacity(node_output, partitions):
    result = Counter()
    details = []
    for line in node_output.splitlines():
        node = fields(line)
        state = node.get("State", "")
        if state not in ("IDLE", "MIXED"):
            continue
        selected = next((p for p in partitions if p in node.get("Partitions", "").split(",")), None)
        if selected is None:
            continue
        free_gpu = max(0, gpu_count(node.get("CfgTRES", "")) - gpu_count(node.get("AllocTRES", "")))
        if not free_gpu:
            continue
        free_cpu = max(0, int(node.get("CPUEfctv", node.get("CPUTot", "0"))) - int(node.get("CPUAlloc", "0")))
        free_mem = max(0, int(node.get("RealMemory", "0")) - int(node.get("AllocMem", "0")))
        slots = min(free_gpu, free_cpu // 4, free_mem // (16 * 1024))
        if slots:
            result[selected] += slots
            details.append(dict(node=node["NodeName"], partition=selected, slots=slots))
    return result, details


def read_command(args):
    return subprocess.check_output(args, text=True, cwd=ROOT).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", default="gpu_h200,gpu_a100,gpu_a40")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.batch_id):
        parser.error("Use an alphanumeric batch ID")
    partitions = args.partitions.split(",")
    if not partitions or len(set(partitions)) != len(partitions) or any(
        p not in ("gpu_h200", "gpu_a100", "gpu_a40") for p in partitions
    ):
        parser.error("Use the explicitly authorized open H200/A100/A40 partitions")
    root = Path(os.environ["SCRATCH_FLASH"]) / "clean_classifier_study/scratch32_full_factorial_v1"
    original = json.loads((root / "submission.json").read_text())
    plan = checked_plan(root / "plan.json")
    ready = json.loads((root / "data_ready.json").read_text())
    if plan["sha256"] != original["plan_sha256"] or ready["plan_sha256"] != plan["sha256"]:
        raise RuntimeError("Existing preparation/submission/plan mismatch")
    if (root / "investigate_stop.json").exists():
        raise RuntimeError("Investigation stop exists; inspect before adding workers")
    for name, expected in original["source_hashes"].items():
        if file_hash(ROOT / "hpc" / name) != expected:
            raise RuntimeError("Frozen source mismatch: " + name)
    user = getpass.getuser()
    account = os.environ.get("HPC_ACCOUNT", user)
    associations = read_command(["scontrol", "show", "assoc_mgr", "flags=assoc",
                                  "users=" + user, "accounts=" + account])
    allowed = set()
    for line in associations.splitlines():
        if re.search(r"UserName=" + re.escape(user) + r"(?:\(|\s)", line):
            allowed.add(fields(line).get("Partition"))
    if any(p not in allowed for p in partitions):
        raise RuntimeError("Could not confirm partition associations; inspect account settings")
    slots, details = capacity(read_command(["scontrol", "show", "nodes", "-o"]), partitions)
    # Existing pending workers may already consume this capacity when dispatched.
    queue = read_command(["squeue", "--noheader", "--user=" + user, "--format=%P|%T|%j"])
    for line in queue.splitlines():
        p, state, name = line.split("|", 2)
        if state == "PENDING" and name.startswith("full-grid"):
            slots[p] = max(0, slots[p] - 1)
    unclaimed = sum(not (root / "claims" / f"case_{case['id']:04d}").exists()
                    for case in plan["trials"])
    requests = []
    for p in partitions:
        number = min(slots[p], max(0, unclaimed-len(requests)))
        requests.extend([p]*number)
    print(json.dumps(dict(snapshot_capacity=details, request_counts=dict(Counter(requests)),
                         additional_workers=len(requests),
                         note="Snapshot of unallocated resources, not guaranteed starts or a personal maximum"), indent=2))
    if not args.submit:
        return
    if not requests:
        print("No additional snapshot capacity; no jobs submitted")
        return
    if (os.environ.get("SCRATCH_TRACKING_MODE") != "online" or
        not all(os.environ.get(k) for k in ("HPC_PYTHON", "WANDB_ENTITY", "WANDB_PROJECT"))):
        parser.error("Set project Python and online W&B entity/project environment")
    target = root / "extensions" / args.batch_id
    # An exclusive directory prevents accidental repetition of this expansion.
    target.mkdir(parents=True, exist_ok=False)
    receipt = dict(batch_id=args.batch_id, plan_sha256=plan["sha256"],
                   status="submitting", jobs=[], snapshot=details)
    path = target / "receipt.json"
    write_json(path, receipt)
    try:
        for index, p in enumerate(requests):
            output = read_command(["sbatch", "--parsable", "--partition=" + p,
                                   f"--job-name=full-grid-add{index}", "--export=ALL",
                                   "hpc/full_factorial.sbatch"])
            job_id = output.split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError("Unexpected sbatch response; inspect expansion receipt")
            receipt["jobs"].append(dict(job_id=job_id, partition=p))
            write_json(path, receipt)
        receipt["status"] = "submitted"
        write_json(path, receipt)
    except Exception:
        receipt["status"] = "partial_submission_inspect_before_retry"
        write_json(path, receipt)
        raise
    print(json.dumps(receipt, indent=2))
    print("Existing jobs/claims/checkpoints untouched. New jobs have24-hour allocations. "
          "Do not repeat expansion to chase fluctuations; inspect scheduler status.")


if __name__ == "__main__":
    main()
