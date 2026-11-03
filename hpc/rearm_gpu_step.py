"""Audited worker-local preflight recovery; never change the frozen trainer."""
from __future__ import annotations
import argparse
from collections import Counter
import getpass
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.clean_protocol import file_hash, write_json
from hpc.full_factorial_protocol import checked_plan
from hpc.gpu_step_recovery import recovery_paths, verify_authority, audit_complete
from hpc.scale_full_factorial import capacity, fields
from hpc.submit_full_factorial import submit as sbatch

SOURCES = ("rearm_gpu_step.py", "full_factorial_worker_gpu_step_v2.py",
           "full_factorial_gpu_step_v2.sbatch")


def rearm_paths(root):
    return Path(root) / "recoveries/gpu_step_v2"


def verify_rearm(root):
    verify_authority(root)
    base = rearm_paths(root)
    receipt = json.loads((base/"receipt.json").read_text())
    if receipt["status"] != "verified":
        raise ValueError("Worker-local preflight recovery is not verified")
    if file_hash(base/"archived_preflight_stop.json") != receipt["archived_stop_sha256"]:
        raise ValueError("Archived preflight stop changed")
    for name, expected in receipt["source_hashes"].items():
        if file_hash(ROOT/"hpc"/name) != expected:
            raise ValueError("Recovery source changed: "+name)
    if socket.gethostname().split(".")[0] in receipt["excluded_nodes"]:
        raise ValueError("Worker landed on explicitly excluded node")
    return receipt


def prepare(root, excluded):
    root = Path(root).resolve()
    authority = verify_authority(root)
    plan = checked_plan(root/"plan.json")
    base = rearm_paths(root)
    if (base/"receipt.json").exists():
        raise ValueError("Rearm already attempted; inspect receipt before any repetition")
    stop_path = recovery_paths(root)/"stop.json"
    stop = json.loads(stop_path.read_text())
    if stop != dict(reason="CUDA preflight failed before claiming any case", returncode=1):
        raise ValueError("This recovery is not authorized for a training/source/other failure")
    queue = subprocess.check_output(["squeue", "--noheader", "--user="+getpass.getuser(),
                                     "--format=%i"], text=True)
    active = set(queue.split())
    old_submission = json.loads((recovery_paths(root)/"submission.json").read_text())
    old_jobs = {x["job_id"] for x in old_submission["jobs"] if x["role"] == "worker"}
    if old_jobs & active:
        raise ValueError("Prior recovery workers still active; do not rearm yet")
    # This incident must not have reached training or claimed any new case.
    completed, in_flight = [], []
    for claim in sorted((root/"claims").glob("case_*")):
        cid = int(claim.name.removeprefix("case_"))
        owner = json.loads((claim/"owner.json").read_text())
        if str(owner["job"]) in old_jobs:
            raise ValueError("Prior recovery claimed a case; inspect explicitly before rearm")
        result_path = claim/"result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["status"] != "complete":
                raise ValueError("Failed/paused case needs investigation")
            completed.append(audit_complete(root, plan, plan["trials"][cid], result))
        elif str(owner["job"]) in active:
            in_flight.append(dict(case_id=cid, job=owner["job"]))
        else:
            raise ValueError("Interrupted claim without a live scheduler owner")
    if not {0,1,4}.issubset({r["case_id"] for r in completed}):
        raise ValueError("Original completed cases missing")
    archive = recovery_paths(root)/"failed_case_0002/train.log"
    if file_hash(archive) != authority["failed_log_sha256"]:
        raise ValueError("Original failed attempt archive changed")
    proof, failed_nodes = [], set()
    for jid in sorted(old_jobs):
        path = ROOT/f"grid-gpu-step-{jid}.log"
        text = path.read_text(errors="replace")
        events = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                events.append(row)
        if any(r.get("event") == "case_claimed" for r in events):
            raise ValueError("Prior worker log contains training claim")
        probes = [r for r in events if r.get("event") == "cuda_preflight"]
        if len(probes) != 1:
            raise ValueError("Ambiguous/missing prior preflight identity")
        probe = probes[0]
        if str(probe["job"]) != jid:
            raise ValueError("Preflight log job identity mismatch")
        node = probe["node"].split(".")[0]
        passed = any(r.get("event") == "cuda_preflight_passed" for r in events)
        if not passed:
            if "CUDA-capable device(s) is/are busy or unavailable" not in text:
                raise ValueError("Preflight failure has a different cause")
            failed_nodes.add(node)
        proof.append(dict(job_id=jid, node=node, passed=passed, log_sha256=file_hash(path)))
    if not failed_nodes or not failed_nodes.issubset(set(excluded)) or not any(x["passed"] for x in proof):
        raise ValueError("Failed node must be excluded; at least one healthy preflight is required")
    base.mkdir(parents=True, exist_ok=True)
    for target in (base, stop_path):
        if not target.resolve().is_relative_to(root):
            raise ValueError("Archive target escapes study")
    receipt = dict(status="preparing", excluded_nodes=excluded,
                   archived_stop_sha256=file_hash(stop_path), preflight_proof=proof,
                   plan_sha256=plan["sha256"], completed_verified=completed,
                   untouched_in_flight=in_flight,
                   source_hashes={name:file_hash(ROOT/"hpc"/name) for name in SOURCES})
    write_json(base/"receipt.json", receipt)
    stop_path.rename(base/"archived_preflight_stop.json")
    receipt["status"] = "verified"
    write_json(base/"receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


def submit_jobs(args):
    for key in ("SCRATCH_FLASH","HPC_PYTHON","WANDB_ENTITY","WANDB_PROJECT"):
        if not os.environ.get(key):
            raise ValueError("Missing "+key)
    if os.environ.get("SCRATCH_TRACKING_MODE") != "online":
        raise ValueError("Set online tracking explicitly")
    root = Path(os.environ["SCRATCH_FLASH"])/"clean_classifier_study/scratch32_full_factorial_v1"
    verify_authority(root)
    user = getpass.getuser()
    associations = subprocess.check_output(["scontrol","show","assoc_mgr","flags=assoc",
                     "users="+user,"accounts="+os.environ.get("HPC_ACCOUNT",user)], text=True)
    allowed = {fields(line).get("Partition") for line in associations.splitlines()
               if "UserName="+user+"(" in line or "UserName="+user+" " in line}
    priorities = [p for p in ("gpu_h200","gpu_a100","gpu_a40") if p in allowed]
    if "gpu_a40" not in priorities:
        raise ValueError("Could not verify GPU association")
    excluded = args.exclude_nodes.split(",")
    nodes = subprocess.check_output(["scontrol","show","nodes","-o"], text=True)
    nodes = "\n".join(line for line in nodes.splitlines() if fields(line).get("NodeName") not in excluded)
    slots, details = capacity(nodes, priorities)
    requests = [p for p in priorities for _ in range(slots[p])] or ["gpu_a40"]
    print(json.dumps(dict(excluded_nodes=excluded, requests=dict(Counter(requests)),
                         snapshot=details), indent=2))
    if not args.submit:
        return
    base = rearm_paths(root)
    base.mkdir(parents=True, exist_ok=True)
    path = base/"submission.json"
    receipt = dict(status="submitting", jobs=[], excluded_nodes=excluded)
    with path.open("x") as stream:
        json.dump(receipt, stream)
    command = shlex.join([sys.executable,str(ROOT/"hpc/rearm_gpu_step.py"),"prepare",
                          "--study-dir",str(root),"--exclude-nodes",args.exclude_nodes])
    cpu_exclude = ["--exclude="+os.environ["FULL_GRID_EXCLUDE_CPU"]] if os.environ.get("FULL_GRID_EXCLUDE_CPU") else []
    try:
        prep = sbatch(["sbatch","--parsable","--partition="+args.cpu_partition,*cpu_exclude,
                       "--job-name=grid-rearm-audit","--nodes=1","--ntasks=1","--cpus-per-task=2",
                       "--mem=4G","--time=00:20:00","--export=ALL","--output=grid-rearm-audit-%j.log",
                       "--wrap=export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2; exec "+command])
        receipt["jobs"].append(dict(role="audit",job_id=prep))
        write_json(path,receipt)
        for i,p in enumerate(requests):
            jid = sbatch(["sbatch","--parsable","--partition="+p,"--exclude="+args.exclude_nodes,
                          "--dependency=afterok:"+prep,"--kill-on-invalid-dep=yes","--export=ALL",
                          f"--job-name=grid-gpu-v2-{i}","hpc/full_factorial_gpu_step_v2.sbatch"])
            receipt["jobs"].append(dict(role="worker",job_id=jid,partition=p))
            write_json(path,receipt)
        receipt["status"]="submitted"
        write_json(path,receipt)
    except Exception:
        receipt["status"]="partial_submission_inspect"
        write_json(path,receipt)
        raise
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("submit","prepare"))
    p.add_argument("--study-dir",type=Path)
    p.add_argument("--exclude-nodes",required=True)
    p.add_argument("--cpu-partition",default="cpu_sapphire")
    p.add_argument("--submit",action="store_true")
    args=p.parse_args()
    if args.action=="prepare":
        if args.study_dir is None or not os.environ.get("SLURM_JOB_ID"):
            p.error("Preparation requires a CPU allocation and study directory")
        prepare(args.study_dir,args.exclude_nodes.split(","))
    else:
        submit_jobs(args)
