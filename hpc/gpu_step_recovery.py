"""Explicit recovery of the observed case2 pre-training CUDA access failure."""
from __future__ import annotations
import argparse
import getpass
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.clean_protocol import file_hash, write_json
from hpc.full_factorial_protocol import checked_plan, trial_directory

NEW_SOURCES = ("gpu_step_preflight.py", "gpu_step_recovery.py",
               "full_factorial_worker_gpu_step.py", "full_factorial_gpu_step.sbatch")


def recovery_paths(root):
    return Path(root) / "recoveries" / "gpu_step_v1"


def check_sources(hashes):
    for name, expected in hashes.items():
        if file_hash(ROOT / "hpc" / name) != expected:
            raise ValueError("Source lineage mismatch: " + name)


def verify_authority(root):
    root = Path(root)
    receipt = json.loads((recovery_paths(root) / "receipt.json").read_text())
    original = json.loads((root / "submission.json").read_text())
    if receipt["status"] != "verified" or receipt["plan_sha256"] != checked_plan(root / "plan.json")["sha256"]:
        raise ValueError("Recovery has not been verified")
    if file_hash(root / "investigate_stop.json") != receipt["retained_stop_sha256"]:
        raise ValueError("Original stop marker changed; investigate before continuing")
    check_sources(original["source_hashes"])
    check_sources(receipt["new_source_hashes"])
    return receipt


def audit_complete(root, plan, case, result):
    directory = trial_directory(root, case["id"], case["seed"])
    saved = json.loads((directory / "summary.json").read_text())
    original = json.loads((Path(root) / "submission.json").read_text())
    trainer_sources = {k: v for k, v in original["source_hashes"].items()
                       if k in ("train_full_factorial.py", "clean_protocol.py",
                                "scratch_recipes.py", "train_classifier_pilot.py",
                                "wandb_tracking.py", "full_factorial_protocol.py")}
    expected_sources = {"hpc/" + k: v for k, v in trainer_sources.items()}
    if (saved["status"] != "complete" or saved["epochs_complete"] != 200
        or result["epochs_complete"] != 200 or saved["trial"] != case
        or saved["seed"] != case["seed"] or saved["plan_sha256"] != plan["sha256"]
        or saved["test_evaluated"] or saved["source_hashes"] != expected_sources):
        raise ValueError("Incompatible completed result; preserve and inspect")
    if file_hash(directory / "latest.pt") != saved["checkpoint_sha256"]:
        raise ValueError("Completed checkpoint hash mismatch")
    history = saved["history"]
    if [r["epoch"] for r in history] != list(range(1, 201)):
        raise ValueError("Incomplete epoch history")
    for row in history:
        for role, count in (("train", 40000), ("validation", 5000)):
            value = row[role]
            if (value["images"] != count or not 0 <= value["accuracy"] <= 100
                or not all(math.isfinite(value[k]) for k in ("accuracy", "loss", "seconds"))):
                raise ValueError("Invalid completed training metrics")
    return dict(case_id=case["id"], summary_sha256=file_hash(directory / "summary.json"),
                checkpoint_sha256=saved["checkpoint_sha256"])


def prepare(root):
    root = Path(root).resolve()
    plan = checked_plan(root / "plan.json")
    original = json.loads((root / "submission.json").read_text())
    ready = json.loads((root / "data_ready.json").read_text())
    if original["plan_sha256"] != plan["sha256"] or ready["plan_sha256"] != plan["sha256"]:
        raise ValueError("Preparation/plan mismatch")
    check_sources(original["source_hashes"])
    stop_path = root / "investigate_stop.json"
    stop = json.loads(stop_path.read_text())
    if stop.get("case_id") != 2 or stop.get("returncode") != 1:
        raise ValueError("Recovery authorizes only the diagnosed case2 failure")
    case = plan["trials"][2]
    if case["architecture"] != "resnet50" or case["seed"] != 42:
        raise ValueError("Unexpected case2 identity")
    queue = subprocess.check_output(["squeue", "--noheader", "--user=" + getpass.getuser(),
                                     "--format=%i"], text=True)
    active_jobs = set(queue.split())
    completed, in_flight = [], []
    for claim in sorted((root / "claims").glob("case_*")):
        cid = int(claim.name.removeprefix("case_"))
        owner = json.loads((claim / "owner.json").read_text())
        if (owner["case_id"] != cid or owner["plan_sha256"] != plan["sha256"]
            or cid not in range(len(plan["trials"]))):
            raise ValueError("Invalid claim identity")
        path = claim / "result.json"
        result = json.loads(path.read_text()) if path.exists() else None
        if result is None:
            if str(owner["job"]) not in active_jobs:
                raise ValueError("Interrupted claim without active owner; explicit audit required")
            in_flight.append(dict(case_id=cid, job=owner["job"]))
        elif result["status"] == "complete":
            completed.append(audit_complete(root, plan, plan["trials"][cid], result))
        elif cid == 2 and result["status"] == "failed" and result["epochs_complete"] is None:
            if str(owner["job"]) in active_jobs:
                raise ValueError("Failed-case owning job still active")
        else:
            raise ValueError("Another failed/paused case needs investigation")
    if not {0, 1, 4}.issubset({r["case_id"] for r in completed}):
        raise ValueError("Previously completed cases0/1/4 must all be present and verified")
    claim = root / "claims/case_0002"
    failed = json.loads((claim / "result.json").read_text())
    if failed["status"] != "failed" or failed["epochs_complete"] is not None or failed["returncode"] != 1:
        raise ValueError("Case2 is not the diagnosed pre-training failure")
    directory = trial_directory(root, 2, 42)
    if {p.name for p in directory.iterdir()} != {"train.log"}:
        raise ValueError("Failed case has additional state/checkpoint; refusing automatic archival")
    text = (directory / "train.log").read_text(errors="replace")
    if ("CUDA-capable device(s) is/are busy or unavailable" not in text
        or "model = model.cuda()" not in text or '"epoch_saved"' in text):
        raise ValueError("Failed attempt does not match observed CUDA access failure")
    recovery = recovery_paths(root)
    recovery.mkdir(parents=True, exist_ok=True)
    if (recovery / "receipt.json").exists():
        raise ValueError("Recovery already prepared or interrupted; do not repeat")
    # Both exact targets must be descendants of the validated study root.
    for target in (claim, directory, recovery):
        if not target.resolve().is_relative_to(root):
            raise ValueError("Archive target escapes study")
    receipt = dict(status="preparing", plan_sha256=plan["sha256"],
                   retained_stop_sha256=file_hash(stop_path),
                   completed_verified=completed, untouched_in_flight=in_flight,
                   failed_log_sha256=file_hash(directory / "train.log"),
                   failed_claim_hashes={p.name: file_hash(p) for p in claim.iterdir() if p.is_file()},
                   new_source_hashes={name: file_hash(ROOT / "hpc" / name) for name in NEW_SOURCES},
                   qualification="CUDA context check through srun passed on one allocation; "
                   "original node/device root cause is not proven. Frozen training protocol unchanged.")
    write_json(recovery / "receipt.json", receipt)
    directory.rename(recovery / "failed_case_0002")
    claim.rename(recovery / "failed_claim_0002")
    receipt["status"] = "verified"
    write_json(recovery / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study-dir", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Run recovery audit in a CPU scheduler allocation")
    prepare(args.study_dir)
