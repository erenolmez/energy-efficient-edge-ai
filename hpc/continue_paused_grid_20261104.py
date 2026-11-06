"""One explicit continuation of four audited paused cases; frozen trainer unchanged.

Default is a read-only dry run. Submission is explicit, exclusive, and never retried.
Run from the same project Python/environment on HPC, not on a login-node GPU.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.clean_protocol import file_hash, write_json
from hpc.full_factorial_protocol import checked_plan, trial_directory
from hpc.gpu_step_recovery import audit_complete, recovery_paths
from hpc.rearm_gpu_step import verify_rearm
from hpc.scale_full_factorial import fields

EXPECTED = {154: 178, 155: 94, 158: 32, 159: 9}
PLAN = "9b3173c990036c44f2a1b7b09d1d38000a82cf2c03f796ff4d58047b8153f50c"
BATCH = "paused_continuation_20261104_v1"


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def latch(root, cid, reason):
    """Preserve the first safety-stop evidence, including another worker's stop."""
    try:
        with (recovery_paths(root) / "stop.json").open("x") as stream:
            json.dump(dict(case_id=cid, reason=reason), stream)
    except FileExistsError:
        pass


def check_case(root, plan, cid):
    """Read-only input gate, repeated inside the actual allocation."""
    case = plan["trials"][cid]
    directory = trial_directory(root, cid, case["seed"])
    claim = root / "claims" / f"case_{cid:04d}"
    saved, run = read(directory / "summary.json"), read(directory / "run.json")
    owner, result = read(claim / "owner.json"), read(claim / "result.json")
    require(not (directory / "active.lock").exists(), "Trainer lock exists; inspect, never remove blindly")
    require(saved["status"] == result["status"] == "paused", "Case is not paused")
    require(saved["epochs_complete"] == result["epochs_complete"] == EXPECTED[cid], "Saved epoch changed")
    require(result["returncode"] == 0 and owner["case_id"] == cid, "Unexpected claim/result")
    require(saved["trial"] == case and saved["seed"] == case["seed"] == 45, "Trial/seed changed")
    require(saved["plan_sha256"] == owner["plan_sha256"] == PLAN, "Plan changed")
    require(not saved["test_evaluated"], "Unexpected test evaluation")
    require(saved["source_hashes"] == run["source_hashes"], "Source receipt mismatch")
    for name, digest in saved["source_hashes"].items():
        require(file_hash(ROOT / name) == digest, "Frozen trainer source changed: " + name)
    require(file_hash(directory / "latest.pt") == saved["checkpoint_sha256"], "Checkpoint changed")
    identity = read(directory / "wandb_identity.json")
    require(identity["entity"] == "erenolmez-politecnico-di-torino" and
            identity["project"] == "energy-efficient-edge-ai", "Tracking destination changed")
    require(read(directory / "wandb_status.json")["run_id"] == identity["id"], "W&B run identity mismatch")
    return directory, claim, saved


def gate(root):
    verify_rearm(root)
    require(not (recovery_paths(root) / "stop.json").exists(), "Shared safety stop exists")
    plan = checked_plan(root / "plan.json")
    require(plan["sha256"] == PLAN, "Unexpected study")
    return plan


def tracking_gate():
    require(os.environ.get("SCRATCH_TRACKING_MODE") == "online", "Set SCRATCH_TRACKING_MODE=online")
    require(os.environ.get("WANDB_ENTITY") == "erenolmez-politecnico-di-torino" and
            os.environ.get("WANDB_PROJECT") == "energy-efficient-edge-ai", "Set original public W&B destination")
    require(Path(os.environ["HPC_PYTHON"]).resolve() == Path(sys.executable).resolve(), "Use HPC_PYTHON itself")
    import wandb
    require(wandb.__version__ == "0.23.0", "Pinned W&B SDK mismatch; do not upgrade")


def submit(root, partition, execute):
    tracking_gate()
    plan = gate(root)
    queue = subprocess.check_output(["squeue", "--noheader", "--user=" + getpass.getuser(), "--format=%i"], text=True)
    require(not queue.strip(), "Queue is no longer empty; inspect before continuation")
    association = subprocess.check_output([
        "scontrol", "show", "assoc_mgr", "flags=assoc", "users=" + getpass.getuser(),
        "accounts=" + os.environ.get("HPC_ACCOUNT", getpass.getuser())], text=True)
    allowed = {fields(line).get("Partition") for line in association.splitlines()
               if "UserName=" + getpass.getuser() + "(" in line or
               "UserName=" + getpass.getuser() + " " in line}
    require(partition in allowed, "Partition association not confirmed")
    for cid in EXPECTED:
        check_case(root, plan, cid)
    base = root / "recoveries" / BATCH
    require(not base.exists(), "Continuation already attempted; inspect receipt, never repeat")
    receipt = dict(status="dry_run_passed", plan_sha256=PLAN, cases=EXPECTED,
                   helper_sha256=file_hash(__file__), partition=partition,
                   excluded_nodes=["compute-5-11"], jobs=[], created_unix=time.time())
    print(json.dumps(receipt, indent=2), flush=True)
    if not execute:
        print("READ-ONLY DRY RUN: no jobs submitted or files changed", flush=True)
        return
    base.mkdir(exist_ok=False)
    receipt["status"] = "submitting"
    write_json(base / "submission.json", receipt)
    try:
        for cid in EXPECTED:
            command = shlex.join(["srun", "--ntasks=1", "--gres=gpu:1", "--gpu-bind=single:1",
                                  sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--case-id", str(cid)])
            output = subprocess.check_output([
                "sbatch", "--parsable", "--hold", "--partition=" + partition, "--exclude=compute-5-11",
                "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--gres=gpu:1", "--mem=16G",
                "--time=1-00:00:00", "--signal=USR1@180", "--export=ALL",
                f"--job-name=grid-cont-{cid}", "--output=grid-cont-%j.log",
                "--wrap=exec " + command], cwd=ROOT, text=True).strip()
            jid = output.split(";")[0]
            require(jid.isdigit(), "Ambiguous submission response; inspect queue, never resubmit")
            receipt["jobs"].append(dict(case_id=cid, job_id=jid))
            write_json(base / "submission.json", receipt)
        # Record all exact scheduler identities before any worker can execute.
        receipt["status"] = "releasing_verified_jobs"
        write_json(base / "submission.json", receipt)
        for job in receipt["jobs"]:
            subprocess.check_call(["scontrol", "release", job["job_id"]])
            job["released"] = True
            write_json(base / "submission.json", receipt)
        receipt["status"] = "submitted"
    except Exception:
        receipt["status"] = "partial_submission_inspect_do_not_retry"
        raise
    finally:
        write_json(base / "submission.json", receipt)
        print(json.dumps(receipt, indent=2), flush=True)


def worker(root, cid):
    require(cid in EXPECTED and os.environ.get("SLURM_JOB_ID"), "Explicit allocated case required")
    tracking_gate()
    plan = gate(root)
    base = root / "recoveries" / BATCH
    authority = read(base / "submission.json")
    require(authority["helper_sha256"] == file_hash(__file__), "Continuation helper changed")
    require(authority["status"] in ("releasing_verified_jobs", "submitted"), "Submission requires inspection")
    require(any(job["case_id"] == cid and job["job_id"] == os.environ["SLURM_JOB_ID"]
                for job in authority["jobs"]), "Job/case not in explicit submission receipt")
    # The exclusive per-case directory is the single-attempt lock, never removed.
    attempt = base / f"case_{cid:04d}"
    attempt.mkdir(exist_ok=False)
    status = dict(status="checking", case_id=cid, job=os.environ["SLURM_JOB_ID"], started_unix=time.time())
    write_json(attempt / "receipt.json", status)
    stopped, child = [], None

    def stop(signum, frame):
        stopped.append(signum)
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signal.SIGUSR1)
            except ProcessLookupError:
                pass

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    try:
        directory, claim, saved = check_case(root, plan, cid)
        require(shutil.disk_usage(root).free > (10 * 1024**3 + (directory / "latest.pt").stat().st_size),
                "Insufficient space for preservation plus safety reserve")
        if stopped:
            status["status"] = "interrupted_before_training"
            write_json(attempt / "receipt.json", status)
            return
        status["status"] = "preflight"
        write_json(attempt / "receipt.json", status)
        child = subprocess.Popen([sys.executable, "-B", str(ROOT / "hpc/gpu_step_preflight.py")], cwd=ROOT)
        preflight = child.wait()
        child = None
        if stopped:
            status["status"] = "interrupted_before_training"
            write_json(attempt / "receipt.json", status)
            return
        require(preflight == 0, "Preflight failed; no training or retry")
        status["status"] = "preserving"
        gate(root)
        # Preserve prior checkpoint/summary/run/log/claims before trainer overwrites mutable outputs.
        prior = attempt / "prior"
        prior.mkdir()
        files = {"latest.pt": directory / "latest.pt", "summary.json": directory / "summary.json",
                 "run.json": directory / "run.json", "train.log": directory / "train.log",
                 "owner.json": claim / "owner.json", "result.json": claim / "result.json",
                 "wandb_identity.json": directory / "wandb_identity.json"}
        hashes = {}
        for name, source in files.items():
            hashes[name] = file_hash(source)
            shutil.copyfile(source, prior / name)
            require(file_hash(prior / name) == hashes[name] == file_hash(source), "Preservation mismatch")
        require(hashes["latest.pt"] == saved["checkpoint_sha256"], "Checkpoint changed during preservation")
        status.update(status="resume_prepared", preserved_sha256=hashes, first_epoch=EXPECTED[cid] + 1)
        write_json(attempt / "receipt.json", status)
        if stopped:
            status["status"] = "interrupted_before_training"
            write_json(attempt / "receipt.json", status)
            return
        gate(root)
        command = [sys.executable, "-B", str(ROOT / "hpc/train_full_factorial.py"), "train",
                   "--study-dir", str(root), "--data-dir", str(root.parents[1] / "datasets/cifar100"),
                   "--split-manifest", str(root.parent / "split_seed42.json"),
                   "--case-id", str(cid), "--resume", "--tracking", "online"]
        status["status"] = "resuming"
        write_json(attempt / "receipt.json", status)
        with (attempt / "resume.log").open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if stopped and child.poll() is None:
                child.send_signal(signal.SIGUSR1)
            code = child.wait()
        child = None
        result = read(directory / "summary.json")
        state = "complete" if code == 0 and result["status"] == "complete" else (
            "paused" if code == 0 and stopped and result["status"] == "paused" else "failed")
        if state == "complete":
            audit_complete(root, plan, plan["trials"][cid], dict(epochs_complete=200))
        status.update(status=state, returncode=code, epochs_complete=result["epochs_complete"],
                      finished_unix=time.time())
        write_json(attempt / "receipt.json", status)
        # Keep original owner immutable; link the new execution in the result and archived evidence.
        write_json(claim / "result.json", dict(status=state, returncode=code, case_id=cid,
                   epochs_complete=result["epochs_complete"], finished_unix=time.time(),
                   continuation_job=os.environ["SLURM_JOB_ID"], continuation_receipt=str(attempt / "receipt.json")))
        if state == "failed":
            latch(root, cid, "Continuation failure; inspect preserved evidence")
            raise RuntimeError("Continuation failed; no retry")
        if state == "paused" or stopped or (recovery_paths(root) / "stop.json").exists():
            return
        gate(root)
        status["handoff"] = "unchanged_revision2_worker"
        write_json(attempt / "receipt.json", status)
        os.execv(sys.executable, [sys.executable, "-B", str(ROOT / "hpc/full_factorial_worker_gpu_step_v2.py"),
                 "--study-dir", str(root), "--data-dir", str(root.parents[1] / "datasets/cifar100"),
                 "--split-manifest", str(root.parent / "split_seed42.json")])
    except Exception:
        previous = status["status"]
        status.update(status="stopped_error_inspect_no_retry", failed_phase=previous, finished_unix=time.time())
        write_json(attempt / "receipt.json", status)
        if previous != "preflight":
            latch(root, cid, "Continuation exception; inspect preserved evidence")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "worker"))
    parser.add_argument("--case-id", type=int)
    parser.add_argument("--partition", choices=("gpu_a40", "gpu_a100", "gpu_h200"), default="gpu_a40")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ["SCRATCH_FLASH"]) / "clean_classifier_study/scratch32_full_factorial_v1"
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    if args.action == "worker":
        worker(root, args.case_id)
    else:
        submit(root, args.partition, args.submit)
