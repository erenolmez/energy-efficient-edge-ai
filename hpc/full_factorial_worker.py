"""Claim independent frozen cases atomically; never steal or retry failed claims."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.clean_protocol import file_hash, write_json
from hpc.full_factorial_protocol import checked_plan, trial_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU workers require a scheduler allocation")
    plan = checked_plan(args.study_dir / "plan.json")
    receipt = json.loads((args.study_dir / "submission.json").read_text())
    if receipt["plan_sha256"] != plan["sha256"]:
        raise RuntimeError("Submission/plan mismatch")
    claims = args.study_dir / "claims"
    claims.mkdir(exist_ok=True)
    stop = []
    child = None
    def checkpoint_stop(signum, frame):
        stop.append(signum)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGUSR1)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, checkpoint_stop)
    for case in plan["trials"]:
        if stop or (args.study_dir / "investigate_stop.json").exists():
            break
        for name, expected in receipt["source_hashes"].items():
            if file_hash(ROOT / "hpc" / name) != expected:
                write_json(args.study_dir / "investigate_stop.json",
                           dict(reason="Source changed after submission; preserve and inspect", file=name))
                raise RuntimeError("Frozen source changed")
        if shutil.disk_usage(args.study_dir).free < 10 * 1024**3:
            write_json(args.study_dir / "investigate_stop.json",
                       dict(reason="Less than10GiB filesystem space available; inspect quota/storage"))
            raise RuntimeError("Low filesystem space; user quota must also be checked separately")
        claim = claims / f"case_{case['id']:04d}"
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        write_json(claim / "owner.json", dict(job=os.environ["SLURM_JOB_ID"],
                   case_id=case["id"], claimed_unix=time.time(), plan_sha256=plan["sha256"]))
        print(json.dumps(dict(event="case_claimed", case=case)), flush=True)
        directory = trial_directory(args.study_dir, case["id"], case["seed"])
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "hpc/train_full_factorial.py"), "train",
                   "--study-dir", str(args.study_dir), "--data-dir", str(args.data_dir),
                   "--split-manifest", str(args.split_manifest), "--case-id", str(case["id"])]
        # No auto-resume, forced deletion, test-based pruning or early-success stopping.
        with (directory / "train.log").open("x", encoding="utf-8") as logfile:
            child = subprocess.Popen(command, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT)
            code = child.wait()
        child = None
        summary = directory / "summary.json"
        result = json.loads(summary.read_text()) if summary.exists() else {}
        state = "complete" if code == 0 and result.get("status") == "complete" else (
            "paused" if stop and result.get("status") == "paused" else "failed")
        write_json(claim / "result.json", dict(status=state, returncode=code,
                   epochs_complete=result.get("epochs_complete"), case_id=case["id"],
                   finished_unix=time.time()))
        print(json.dumps(dict(event="case_finished", case_id=case["id"], status=state)), flush=True)
        if state == "failed":
            logtext = (directory / "train.log").read_text(errors="replace")
            if "Non-finite loss; previous checkpoint remains intact" not in logtext:
                write_json(args.study_dir / "investigate_stop.json",
                           dict(case_id=case["id"], returncode=code,
                                reason="Unexpected failure; inspect train.log before restarting"))
                raise RuntimeError("Unexpected case failure; other workers stop after current case")
        if stop:
            break
    print(json.dumps(dict(event="worker_exit", checkpoint_stop_requested=bool(stop))), flush=True)


if __name__ == "__main__":
    main()
