"""One explicit submission, four independent ordinary GPU jobs, no arrays."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.clean_protocol import file_hash, write_json
from hpc.full_factorial_protocol import make_plan


def submit(command):
    value = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    if not value.isdigit():
        raise RuntimeError("Unexpected scheduler response; inspect receipt, do not resubmit")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partition", default="gpu_a40")
    p.add_argument("--cpu-partition", default="cpu_sapphire")
    p.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    plan = make_plan()
    print(json.dumps(dict(cases=len(plan["trials"]), epochs_per_case=plan["epochs"],
                         maximum_gpu_jobs=args.workers, total_epochs=plan["maximum_training_epochs"])))
    if not args.submit:
        print("Dry run only; add --submit to submit the frozen study")
        return
    if not os.environ.get("SCRATCH_FLASH") or not os.environ.get("HPC_PYTHON"):
        p.error("Set SCRATCH_FLASH and HPC_PYTHON")
    if os.environ.get("SCRATCH_TRACKING_MODE") != "online":
        p.error("Set SCRATCH_TRACKING_MODE=online for this authorized tracked study")
    if not os.environ.get("WANDB_ENTITY") or not os.environ.get("WANDB_PROJECT"):
        p.error("Set W&B entity/project, never an API key in this command")
    root = Path(os.environ["SCRATCH_FLASH"]) / "clean_classifier_study/scratch32_full_factorial_v1"
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "submission.json"
    # Exclusive receipt prevents duplicate submission, including a partially failed submission.
    receipt = dict(status="submitting", plan_sha256=plan["sha256"], jobs=[],
                   source_hashes={name: file_hash(ROOT / "hpc" / name) for name in
                       ("full_factorial_protocol.py", "train_full_factorial.py",
                        "full_factorial_worker.py", "full_factorial.sbatch",
                        "clean_protocol.py", "scratch_recipes.py",
                        "train_classifier_pilot.py", "wandb_tracking.py")})
    try:
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream)
    except FileExistsError:
        p.error("Existing submission receipt: refusing duplicates. Inspect status before any continuation.")
    common = ["--study-dir", str(root), "--data-dir",
              str(Path(os.environ["SCRATCH_FLASH"]) / "datasets/cifar100"),
              "--split-manifest", str(Path(os.environ["SCRATCH_FLASH"]) /
                                      "clean_classifier_study/split_seed42.json")]
    prepare = shlex.join([sys.executable, str(ROOT / "hpc/train_full_factorial.py"),
                          "prepare", *common])
    try:
        exclude = (["--exclude=" + os.environ["FULL_GRID_EXCLUDE_CPU"]]
                   if os.environ.get("FULL_GRID_EXCLUDE_CPU") else [])
        job = submit(["sbatch", "--parsable", f"--partition={args.cpu_partition}", *exclude,
                      "--job-name=full-grid-prepare",
                      "--nodes=1", "--ntasks=1", "--cpus-per-task=2",
                      "--mem=4G", "--time=00:20:00", "--export=ALL",
                      "--output=full-grid-prepare-%j.log",
                      "--wrap=export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2; exec " + prepare])
        receipt["jobs"].append(dict(role="prepare", job_id=job))
        write_json(receipt_path, receipt)
        for worker in range(args.workers):
            jid = submit(["sbatch", "--parsable", f"--partition={args.partition}",
                          f"--dependency=afterok:{job}", "--kill-on-invalid-dep=yes",
                          f"--job-name=full-grid-w{worker}", "--export=ALL",
                          "hpc/full_factorial.sbatch"])
            receipt["jobs"].append(dict(role=f"worker{worker}", job_id=jid))
            write_json(receipt_path, receipt)
        receipt["status"] = "submitted"
        write_json(receipt_path, receipt)
    except Exception:
        receipt["status"] = "submission_interrupted_inspect_jobs"
        write_json(receipt_path, receipt)
        raise
    print(json.dumps(receipt, indent=2))
    print("Each GPU job has a 24-hour limit. Unclaimed cases remain pending; "
          "paused/failed claims are retained, never blindly restarted. "
          "Existing small-study jobs are untouched and are additional GPU jobs.")


if __name__ == "__main__":
    main()
