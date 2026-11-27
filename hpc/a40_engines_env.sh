#!/bin/bash
# Job 3 environment: edge-ai python + two read-only --no-deps overlays on PYTHONPATH.
#   timm 1.0.28 overlay ($SCRATCH_FLASH/cache/imgcmp-deps.*) the 128px checkpoints were trained with
#   TensorRT/ONNX/NVML overlay ($SCRATCH_FLASH/envs/trt-overlay)
# The edge-ai env itself is never modified. Sourced by hpc/a40_engines.sbatch; run it directly on
# the login node (bash hpc/a40_engines_env.sh) to check every import the job needs before submitting.
set -euo pipefail
SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/eolmez}"
HPC_PYTHON="${HPC_PYTHON:-${SCRATCH_FLASH}/envs/edge-ai/bin/python}"
TRT_OVERLAY="${TRT_OVERLAY:-${SCRATCH_FLASH}/envs/trt-overlay}"
test -x "$HPC_PYTHON" || { echo "Missing python: $HPC_PYTHON" >&2; exit 1; }
test -d "$TRT_OVERLAY/tensorrt" || { echo "Missing TensorRT overlay: $TRT_OVERLAY" >&2; exit 1; }
if [[ -z "${IMGCMP_DEPS:-}" ]]; then
    for d in "$SCRATCH_FLASH"/cache/imgcmp-deps.*; do
        if [[ -d "$d/timm" && -n "$(ls -d "$d"/timm-1.0.28.dist-info 2>/dev/null)" ]]; then IMGCMP_DEPS="$d"; break; fi
    done
fi
: "${IMGCMP_DEPS:?No timm 1.0.28 overlay found under $SCRATCH_FLASH/cache; set IMGCMP_DEPS}"
export PYTHONPATH="${IMGCMP_DEPS}:${TRT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export SCRATCH_FLASH HPC_PYTHON TRT_OVERLAY IMGCMP_DEPS
echo "PYTHONPATH=$PYTHONPATH"
# Every third-party and repository module hpc/a40_engines.py imports, run from the repo root.
"$HPC_PYTHON" -B - <<'PY'
import importlib, sys
mods = ["numpy", "torch", "torch.nn.functional", "torch.onnx", "torch.utils.data", "torchvision",
        "torchvision.transforms", "torchvision.datasets", "PIL", "timm", "onnx", "tensorrt", "pynvml",
        "hpc.clean_protocol", "hpc.full_factorial_protocol", "hpc.scratch_recipes",
        "hpc.train_imagenet_comparison", "model_pipeline.models", "hpc.a40_engines", "hpc.a40_routers", "hpc.a40_policy"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        if "." not in m:
            print(f"ok  {m:12s} {getattr(mod, '__version__', ''):14s} {getattr(mod, '__file__', '')}")
    except Exception as exc:
        bad.append(m)
        print(f"BAD {m}: {exc!r}")
try:
    import timm, torch
    if timm.__version__ != "1.0.28":
        bad.append(f"timm=={timm.__version__} (need 1.0.28)")
    if not torch.version.cuda or not torch.version.cuda.startswith("12."):
        bad.append(f"torch CUDA {torch.version.cuda} (TensorRT overlay is cu12)")
    from model_pipeline.models import create_classifier
    for a in ("resnet18", "resnet34", "resnet50", "resnet101", "mobilenetv3_small", "mobilenetv3_large",
              "efficientnet_b0", "shufflenetv2_x1_0"):
        create_classifier(a, pretrained=False)
except Exception as exc:
    bad.append(f"version/architecture check: {exc!r}")
print("FAILED: " + ", ".join(bad) if bad else "All job 3 imports OK; all eight architectures construct.")
sys.exit(1 if bad else 0)
PY
