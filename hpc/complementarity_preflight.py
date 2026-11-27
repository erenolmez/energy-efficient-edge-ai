"""Login-node preflight for the complementarity jobs (CPU only, reads nothing large).

Imports every module both jobs import, builds all eight architectures on CPU with the
timm overlay, and checks the files and hashes the jobs verify. Prints one line per check
and exits non-zero on the first missing piece. Optional: pynvml (energy readings).
"""
import importlib
import json
import os
from pathlib import Path
import sys

ok = True


def check(label, fn, optional=False):
    global ok
    try:
        detail = fn()
        print(f"OK    {label}" + (f": {detail}" if detail else ""))
    except Exception as exc:  # report every failure, not just the first
        ok = ok and optional
        print(f"{'WARN' if optional else 'FAIL'}  {label}: {exc!r}")


base = Path(os.environ.get("STUDY_BASE", "/mnt/beegfs/eolmez/clean_classifier_study"))
data = Path(os.environ.get("CIFAR_DATA", "/mnt/beegfs/eolmez/datasets/cifar100"))
print("python", sys.version.split()[0], sys.executable)
print("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
for name in ("numpy", "torch", "torchvision", "timm"):
    check(f"import {name}", lambda n=name: getattr(importlib.import_module(n), "__version__", ""))
check("timm is 1.0.28", lambda: (importlib.import_module("timm").__version__ == "1.0.28") or (_ for _ in ()).throw(
    RuntimeError(importlib.import_module("timm").__version__)))
check("import pynvml (required for GPU energy)", lambda: importlib.import_module("pynvml").__file__)
check("torch and numpy still come from the env, not an overlay",
      lambda: "/envs/edge-ai/" in importlib.import_module("torch").__file__
      and "/envs/edge-ai/" in importlib.import_module("numpy").__file__ and "yes"
      or (_ for _ in ()).throw(RuntimeError(importlib.import_module("torch").__file__)))
for mod in ("hpc.clean_protocol", "hpc.full_factorial_protocol", "hpc.train_imagenet_comparison",
            "model_pipeline.models", "hpc.complementarity_analysis", "hpc.complementarity_execution"):
    check(f"import {mod}", lambda m=mod: importlib.import_module(m) and "")


def build_all():
    from model_pipeline.models import create_classifier
    from hpc.full_factorial_protocol import checked_plan
    from hpc.train_imagenet_comparison import CASES
    plan = checked_plan(base / "scratch32_full_factorial_v1" / "plan.json")
    names = [plan["trials"][c]["architecture"] for c in CASES]
    for n in names:
        create_classifier(n, pretrained=False)
    return ", ".join(names)


def files():
    from hpc.clean_protocol import file_hash
    from hpc.train_imagenet_comparison import CASES, STUDY
    prepared = json.loads((base / STUDY / "prepared.json").read_text())
    want = prepared["spec"]["source_hashes"]["model_pipeline/models.py"]
    if file_hash(Path("model_pipeline/models.py")) != want:
        raise RuntimeError("model_pipeline/models.py differs from the training registry")
    missing = [str(p) for c in CASES for p in (base / STUDY / f"case_{c:04d}_128px" / "summary.json",
                                               base / STUDY / f"case_{c:04d}_128px" / "latest.pt") if not p.exists()]
    for p in (base / "split_seed42.json", base / "scratch32_full_factorial_v1" / "plan.json",
              data / "cifar-100-python" / "train"):
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise RuntimeError("missing: " + ", ".join(missing))
    return "registry hash, split, plan, 8 checkpoints, CIFAR train file"


check("build all eight architectures (CPU, no weights)", build_all)
check("study files and registry hash", files)
for out in ("complementarity_v1", "debatch_execution_v1"):
    check(f"output dir {out} does not exist yet", lambda o=out: (not (base / o).exists()) or (_ for _ in ()).throw(
        RuntimeError(f"{base / o} exists; the job refuses to overwrite it")))
print("PREFLIGHT", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
