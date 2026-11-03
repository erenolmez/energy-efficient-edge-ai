"""Read-only progress, not an accuracy-selection tool for incomplete studies."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.full_factorial_protocol import trial_directory


def main():
    root = Path(os.environ["SCRATCH_FLASH"]) / "clean_classifier_study/scratch32_full_factorial_v1"
    if (root / "submission.json").exists():
        receipt = json.loads((root / "submission.json").read_text())
        print("Submission:", receipt["status"], "| Jobs:",
              ", ".join(x["job_id"] for x in receipt["jobs"]))
    if not (root / "plan.json").exists():
        print("Waiting for CPU preparation")
        return
    plan = json.loads((root / "plan.json").read_text())
    counts = dict(complete=0, failed=0, paused=0, active_or_interrupted=0, unclaimed=0)
    completed_epochs = 0
    for case in plan["trials"]:
        claim = root / "claims" / f"case_{case['id']:04d}"
        result_path = claim / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            counts[result["status"]] += 1
            completed_epochs += result.get("epochs_complete") or 0
        elif claim.exists():
            counts["active_or_interrupted"] += 1
            log = trial_directory(root, case["id"], case["seed"]) / "train.log"
            latest = {}
            if log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 24000))
                    for line in stream.read().decode(errors="replace").splitlines():
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if row.get("event") in ("batch", "epoch_saved"):
                            latest = row
            epoch = latest.get("epoch", 0)
            fraction = latest.get("images", 0) / 40000 if latest.get("event") == "batch" else 1
            progress = max(0, epoch - 1 + fraction)
            completed_epochs += int(progress)
            print(f"Case {case['id']:04d} {case['architecture']} {case['optimizer']} "
                  f"lr={case['lr']} seed={case['seed']}: epoch {epoch}/200, "
                  f"training progress ~{100*progress/200:.1f}% (claim may be interrupted)")
        else:
            counts["unclaimed"] += 1
    print("Cases:", counts)
    print(f"Completed cases: {counts['complete']}/{len(plan['trials'])} "
          f"({100*counts['complete']/len(plan['trials']):.2f}%)")
    print(f"Recorded epoch progress: ~{completed_epochs}/{plan['maximum_training_epochs']} "
          f"({100*completed_epochs/plan['maximum_training_epochs']:.2f}%)")
    if (root / "investigate_stop.json").exists():
        from hpc.clean_protocol import file_hash
        recovery = root / "recoveries/gpu_step_v1"
        receipt = json.loads((recovery/"receipt.json").read_text()) if (recovery/"receipt.json").exists() else {}
        if (receipt.get("status") == "verified"
            and receipt.get("retained_stop_sha256") == file_hash(root/"investigate_stop.json")):
            print("Legacy stop retained. GPU-step recovery audit verified; new worker starts still require scheduler evidence.")
            print("Case2 failed attempt preserved in recoveries/gpu_step_v1; not counted as a successful run.")
            if (recovery/"submission.json").exists():
                submitted = json.loads((recovery/"submission.json").read_text())
                print("Recovery jobs:", ", ".join(x["job_id"] for x in submitted["jobs"]))
            if (recovery/"stop.json").exists():
                print("STOP: GPU-step recovery hit a new error; inspect recoveries/gpu_step_v1/stop.json")
            v2 = root / "recoveries/gpu_step_v2"
            if (v2/"receipt.json").exists():
                rearm = json.loads((v2/"receipt.json").read_text())
                print("Worker-local preflight audit:", rearm["status"],
                      "| New-job node exclusions:", rearm.get("excluded_nodes"))
                if (v2/"submission.json").exists():
                    queued = json.loads((v2/"submission.json").read_text())
                    print("V2 recovery jobs:", ", ".join(x["job_id"] for x in queued["jobs"]))
                failures = list((v2/"preflight_failures").glob("*.json"))
                if failures:
                    print("Worker-local preflight failures:", len(failures),
                          "(no case claimed by these attempts; inspect per-job logs)")
        else:
            print("STOP: unexpected failure needs investigation; no automatic retry")


if __name__ == "__main__":
    main()
