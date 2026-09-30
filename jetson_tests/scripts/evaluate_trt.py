#!/usr/bin/env python3
"""
TensorRT GPU evaluation + plotting script for Jetson Orin Nano.

Updated for CIFAR-100 ResNet-18 / distilled ResNet-18 experiments.
NO PyCUDA required.

Default project layout:

/home/project/Desktop/brand_new_pearl/
    all_precision_models_cifar100/
        fp32/
            baseline_fp32.pth or baseline_fp32_fp32.pth
            structured_10_fp32.pth
            structured_20_fp32.pth
            ...
    all_precision_models_cifar100_distilled/
        fp32/
            distilled_resnet18_fp32_fp32.pth
            structured_10_fp32.pth
            structured_20_fp32.pth
            ...
    plots_updated.py

This script:
    1. Loads FP32 PyTorch full-model checkpoints.
    2. Exports them to ONNX.
    3. Builds TensorRT FP32 / FP16 / INT8 GPU engines.
    4. Evaluates TensorRT engines on CIFAR-100.
    5. Measures:
        - top-1 accuracy
        - TensorRT pure-forward throughput / latency
        - end-to-end throughput / latency
        - TensorRT engine size
        - tegrastats power
        - optional Shelly wall power
    6. Saves CSV summaries and plots.

Default outputs:
    /home/project/Desktop/brand_new_pearl/tensorrt_outputs_cifar100/
    /home/project/Desktop/brand_new_pearl/tensorrt_results_cifar100/

Examples:

Normal CIFAR-100 models:
    python3 plots_updated.py \
      --fp32-dir all_precision_models_cifar100/fp32 \
      --trt-dir tensorrt_outputs_cifar100 \
      --results-dir tensorrt_results_cifar100

Distilled CIFAR-100 models:
    python3 plots_updated.py \
      --fp32-dir all_precision_models_cifar100_distilled/fp32 \
      --trt-dir tensorrt_outputs_cifar100_distilled \
      --results-dir tensorrt_results_cifar100_distilled
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Needed for loading full timm ResNet checkpoints saved with torch.save(model)
import timm  # noqa: F401

import tensorrt as trt


SCRIPT_VERSION = "cifar100_trt_power_eval_v3"

# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1]

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# These are populated from argparse in main(). Keeping globals makes the rest
# of the original script structure simple and stable.
BASE_DIR = DEFAULT_BASE_DIR
FP32_MODEL_DIR = None
TRT_DIR = None
ONNX_DIR = None
ENGINE_DIR = None
RESULTS_DIR = None
CIFAR_ROOT = None

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 4
N_SUBSETS = 5
SUBSET_SIZE = 2000
WARMUP_BATCHES = 10
CALIB_BATCHES = 200
PRECISIONS = ["fp32", "fp16", "int8"]
MODEL_PATTERN = None
FORCE_EXPORT_ONNX = False
FORCE_BUILD_ENGINE = False
ALLOW_TF32_FOR_FP32 = False
INT8_USE_FP16_FALLBACK = True
CI_LEVEL = 95
DOWNLOAD_DATA = True

ENABLE_SHELLY_POWER = False
SHELLY_POLL_INTERVAL_S = 0.5
SHELLY_IP = "192.168.8.50"
PASSWORD = "pearl"
USERNAME = "admin"

ENABLE_JETSON_STABILIZE = False
NVP_MODEL_SET = None

calib_loader = None
subset_loaders = None


# ============================================================
# ARGUMENTS / PATHS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR-100 TensorRT FP32/FP16/INT8 evaluation and plotting on Jetson."
    )

    parser.add_argument("--base-dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument(
        "--fp32-dir",
        type=str,
        default="all_precision_models_cifar100/fp32",
        help="FP32 .pth directory, relative to --base-dir unless absolute.",
    )
    parser.add_argument(
        "--trt-dir",
        type=str,
        default="tensorrt_outputs_cifar100",
        help="TensorRT output directory, relative to --base-dir unless absolute.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="tensorrt_results_cifar100",
        help="CSV/plot output directory, relative to --base-dir unless absolute.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="CIFAR-100 root directory, relative to --base-dir unless absolute.",
    )

    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-subsets", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=2000)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--calib-batches", type=int, default=200)

    parser.add_argument(
        "--precisions",
        type=str,
        default="fp32,fp16,int8",
        help="Comma-separated precision list. Example: fp32,fp16,int8",
    )
    parser.add_argument(
        "--model-pattern",
        default=None,
        help="Optional regular expression applied to normalized model names.",
    )

    parser.add_argument("--force-export-onnx", action="store_true")
    parser.add_argument("--force-build-engine", action="store_true")
    parser.add_argument("--allow-tf32-for-fp32", action="store_true")
    parser.add_argument("--no-int8-fp16-fallback", dest="int8_fp16_fallback", action="store_false")
    parser.set_defaults(int8_fp16_fallback=True)

    parser.add_argument("--ci-level", type=int, default=95)
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.set_defaults(download=True)

    parser.add_argument("--enable-shelly-power", action="store_true")
    parser.add_argument("--shelly-ip", type=str, default="192.168.8.50")
    parser.add_argument("--shelly-username", type=str, default="admin")
    parser.add_argument(
        "--shelly-password",
        type=str,
        default=os.environ.get("SHELLY_PASSWORD", ""),
        help="Read from SHELLY_PASSWORD by default; never store this secret in source control.",
    )
    parser.add_argument("--shelly-poll-interval", type=float, default=0.5)

    parser.add_argument("--stabilize-jetson", action="store_true")
    parser.add_argument("--nvpmodel", type=int, default=None)

    return parser.parse_args()


def resolve_path(path_like, base_dir):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return base_dir / path


def apply_args(args):
    global BASE_DIR, FP32_MODEL_DIR, TRT_DIR, ONNX_DIR, ENGINE_DIR, RESULTS_DIR, CIFAR_ROOT
    global IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, N_SUBSETS, SUBSET_SIZE, WARMUP_BATCHES, CALIB_BATCHES
    global PRECISIONS, MODEL_PATTERN, FORCE_EXPORT_ONNX, FORCE_BUILD_ENGINE, ALLOW_TF32_FOR_FP32, INT8_USE_FP16_FALLBACK
    global CI_LEVEL, DOWNLOAD_DATA
    global ENABLE_SHELLY_POWER, SHELLY_POLL_INTERVAL_S, SHELLY_IP, PASSWORD, USERNAME
    global ENABLE_JETSON_STABILIZE, NVP_MODEL_SET

    BASE_DIR = Path(args.base_dir).resolve()
    FP32_MODEL_DIR = resolve_path(args.fp32_dir, BASE_DIR)
    TRT_DIR = resolve_path(args.trt_dir, BASE_DIR)
    ONNX_DIR = TRT_DIR / "onnx"
    ENGINE_DIR = TRT_DIR / "engines"
    RESULTS_DIR = resolve_path(args.results_dir, BASE_DIR)
    CIFAR_ROOT = resolve_path(args.data_dir, BASE_DIR)

    IMAGE_SIZE = args.image_size
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    N_SUBSETS = args.n_subsets
    SUBSET_SIZE = args.subset_size
    WARMUP_BATCHES = args.warmup_batches
    CALIB_BATCHES = args.calib_batches

    PRECISIONS = [p.strip().lower() for p in args.precisions.split(",") if p.strip()]
    MODEL_PATTERN = args.model_pattern
    invalid = [p for p in PRECISIONS if p not in {"fp32", "fp16", "int8"}]
    if invalid:
        raise ValueError(f"Invalid precision(s): {invalid}. Use fp32, fp16, int8.")

    FORCE_EXPORT_ONNX = args.force_export_onnx
    FORCE_BUILD_ENGINE = args.force_build_engine
    ALLOW_TF32_FOR_FP32 = args.allow_tf32_for_fp32
    INT8_USE_FP16_FALLBACK = args.int8_fp16_fallback
    CI_LEVEL = args.ci_level
    DOWNLOAD_DATA = args.download

    ENABLE_SHELLY_POWER = args.enable_shelly_power
    SHELLY_IP = args.shelly_ip
    USERNAME = args.shelly_username
    PASSWORD = args.shelly_password
    SHELLY_POLL_INTERVAL_S = args.shelly_poll_interval

    ENABLE_JETSON_STABILIZE = args.stabilize_jetson
    NVP_MODEL_SET = args.nvpmodel

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CIFAR_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# OPTIONAL JETSON STABILIZATION
# ============================================================

def stabilize_jetson():
    if not ENABLE_JETSON_STABILIZE:
        return

    def run_cmd(cmd):
        try:
            subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            pass

    run_cmd(["sudo", "nvpmodel", "-q"])

    if NVP_MODEL_SET is not None:
        run_cmd(["sudo", "nvpmodel", "-m", str(NVP_MODEL_SET)])

    run_cmd(["sudo", "jetson_clocks"])


# ============================================================
# TENSORRT SETUP
# ============================================================

def trt_dtype_to_torch(dtype):
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int8:
        return torch.int8
    if hasattr(trt, "bool") and dtype == trt.bool:
        return torch.bool

    raise RuntimeError(f"Unsupported TensorRT dtype: {dtype}")


def print_environment():
    print("Script version:", SCRIPT_VERSION)
    print("TensorRT version:", trt.__version__)
    print("PyTorch version:", torch.__version__)
    print("CUDA available in PyTorch:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    print("Base dir:", BASE_DIR)
    print("Dataset: CIFAR-100")
    print("CIFAR root:", CIFAR_ROOT)
    print("FP32 model dir:", FP32_MODEL_DIR)
    print("ONNX dir:", ONNX_DIR)
    print("Engine dir:", ENGINE_DIR)
    print("Results dir:", RESULTS_DIR)
    print("Precisions:", PRECISIONS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Subsets:", N_SUBSETS, "x", SUBSET_SIZE)
    print("Calibration batches:", CALIB_BATCHES)
    print("CIFAR-100 mean:", CIFAR100_MEAN)
    print("CIFAR-100 std:", CIFAR100_STD)
    print("Shelly power enabled:", ENABLE_SHELLY_POWER)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in PyTorch. TensorRT GPU evaluation cannot run.")


# ============================================================
# POWER MONITOR: TEGRASTATS
# ============================================================

class PowerMonitor:
    def __init__(self, interval_ms=100):
        self.interval_ms = int(interval_ms)
        self.power_samples = deque(maxlen=10000)
        self._running = False
        self._process = None
        self._thread = None

    def _parse_power(self, line):
        # Orin Nano exposes total module input and a combined CPU/GPU/CV rail.
        total_match = re.search(r"VDD_IN\s+(\d+)mW", line)
        compute_match = re.search(r"VDD_CPU_GPU_CV\s+(\d+)mW", line)
        total_mw = int(total_match.group(1)) if total_match else None
        compute_mw = int(compute_match.group(1)) if compute_match else None
        gpu_temp_match = re.search(r"gpu@([0-9.]+)C", line)
        gpu_temp_c = float(gpu_temp_match.group(1)) if gpu_temp_match else None

        if total_mw is not None:
            return {
                "vdd_in_mW": total_mw,
                "cpu_gpu_cv_mW": compute_mw,
                "gpu_temp_c": gpu_temp_c,
            }

        # Fallback for some tegrastats variants.
        m = re.search(r"POM_5V_IN\s+(\d+)mW", line)
        if m:
            return {
                "vdd_in_mW": int(m.group(1)),
                "cpu_gpu_cv_mW": None,
                "gpu_temp_c": gpu_temp_c,
            }

        return None

    def _loop(self):
        try:
            self._process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            print("⚠ tegrastats not found. Power monitoring disabled.")
            self._running = False
            return

        for line in self._process.stdout:
            if not self._running:
                break

            rails = self._parse_power(line)

            if rails is not None:
                self.power_samples.append({"timestamp": time.time(), **rails})

        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass

    def start(self):
        r = subprocess.run(["which", "tegrastats"], capture_output=True, text=True)

        if r.returncode != 0:
            print("⚠ tegrastats not available. Skipping tegrastats power.")
            return False

        self.power_samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        time.sleep(0.4)
        return True

    def stop_window(self, t_start, t_end):
        self._running = False

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                pass

        if self._thread:
            self._thread.join(timeout=2)

        total_window = [
            s["vdd_in_mW"]
            for s in self.power_samples
            if t_start <= s["timestamp"] <= t_end
        ]
        compute_window = [
            s["cpu_gpu_cv_mW"]
            for s in self.power_samples
            if t_start <= s["timestamp"] <= t_end and s["cpu_gpu_cv_mW"] is not None
        ]
        temperature_window = [
            s["gpu_temp_c"]
            for s in self.power_samples
            if t_start <= s["timestamp"] <= t_end and s["gpu_temp_c"] is not None
        ]

        def integrate_joules(key):
            points = [
                (s["timestamp"], s[key] / 1000.0)
                for s in self.power_samples
                if t_start <= s["timestamp"] <= t_end and s[key] is not None
            ]
            if len(points) < 2:
                return 0.0
            if points[0][0] > t_start:
                points.insert(0, (t_start, points[0][1]))
            if points[-1][0] < t_end:
                points.append((t_end, points[-1][1]))
            times = np.asarray([point[0] for point in points], dtype=float)
            watts = np.asarray([point[1] for point in points], dtype=float)
            return float(np.trapz(watts, times))

        total_w = 0.0
        compute_w = 0.0
        if len(total_window) >= 2:
            total_w = float(np.mean(np.asarray(total_window, dtype=float)) / 1000.0)
        if len(compute_window) >= 2:
            compute_w = float(np.mean(np.asarray(compute_window, dtype=float)) / 1000.0)

        total_energy_j = integrate_joules("vdd_in_mW")
        compute_energy_j = integrate_joules("cpu_gpu_cv_mW")
        gpu_temp_mean_c = float(np.mean(temperature_window)) if temperature_window else float("nan")
        gpu_temp_max_c = float(np.max(temperature_window)) if temperature_window else float("nan")

        return (
            total_w,
            compute_w,
            total_energy_j,
            compute_energy_j,
            gpu_temp_mean_c,
            gpu_temp_max_c,
        )


# ============================================================
# POWER MONITOR: SHELLY
# ============================================================

def call_shelly_rpc(method, params=None):
    if params is None:
        params = {}

    url = f"http://{SHELLY_IP}/rpc"
    payload = {"id": 1, "method": method, "params": params}

    try:
        resp = requests.post(url, json=payload, timeout=1.0)
    except Exception as e:
        return {"error": str(e)}

    if resp.status_code == 401:
        auth_header = resp.headers.get("WWW-Authenticate")

        if not auth_header:
            return {"error": "No auth header received"}

        try:
            items = auth_header.replace("Digest ", "").split(", ")
            parts = {}

            for item in items:
                k, v = item.split("=", 1)
                parts[k] = v.strip('"')

        except Exception as e:
            return {"error": f"Failed to parse auth header: {e}"}

        realm = parts.get("realm")
        nonce = parts.get("nonce")
        cnonce = str(random.randint(100000, 999999))

        ha1_raw = f"{USERNAME}:{realm}:{PASSWORD}"
        ha1 = hashlib.sha256(ha1_raw.encode()).hexdigest()

        ha2_raw = "dummy_method:dummy_uri"
        ha2 = hashlib.sha256(ha2_raw.encode()).hexdigest()

        response_raw = f"{ha1}:{nonce}:1:{cnonce}:auth:{ha2}"
        response_hash = hashlib.sha256(response_raw.encode()).hexdigest()

        payload["auth"] = {
            "realm": realm,
            "username": USERNAME,
            "nonce": nonce,
            "cnonce": cnonce,
            "response": response_hash,
            "nc": 1,
            "algorithm": "SHA-256",
        }

        try:
            resp = requests.post(url, json=payload, timeout=1.0)
        except Exception as e:
            return {"error": str(e)}

    try:
        return resp.json()
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON received. Status: {resp.status_code}"}


class ShellyPowerMonitor:
    def __init__(self, interval_s=0.5):
        self.interval_s = float(interval_s)
        self.samples = deque(maxlen=20000)
        self._running = False
        self._thread = None

    def _loop(self):
        while self._running:
            t = time.time()
            data = call_shelly_rpc("Shelly.GetStatus")

            if "error" not in data:
                result = data.get("result", {})
                switch = result.get("switch:0", {})
                p = switch.get("apower", None)

                if isinstance(p, (int, float)):
                    self.samples.append({"timestamp": t, "power_w": float(p)})

            time.sleep(self.interval_s)

    def start(self):
        self.samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        time.sleep(0.2)
        return True

    def stop_window(self, t_start, t_end):
        self._running = False

        if self._thread:
            self._thread.join(timeout=2)

        window = [
            s["power_w"]
            for s in self.samples
            if t_start <= s["timestamp"] <= t_end
        ]

        if len(window) < 2:
            return 0.0

        return float(np.mean(np.asarray(window, dtype=float)))


# ============================================================
# DATA
# ============================================================

def make_loader(dataset, batch_size, shuffle, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
    )


def create_subsets(dataset, n_subsets, subset_size):
    total_needed = n_subsets * subset_size
    if total_needed > len(dataset):
        raise ValueError(
            f"Requested {n_subsets} subsets x {subset_size} = {total_needed} images, "
            f"but dataset only has {len(dataset)} images."
        )

    subsets = []
    indices = list(range(len(dataset)))

    np.random.seed(42)
    np.random.shuffle(indices)

    for i in range(n_subsets):
        start_idx = i * subset_size
        end_idx = start_idx + subset_size

        subset_indices = indices[start_idx:end_idx]
        subset = Subset(dataset, subset_indices)

        loader = make_loader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )

        subsets.append((subset, loader))

    return subsets


def build_data():
    global calib_loader, subset_loaders

    eval_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    test_ds_full = torchvision.datasets.CIFAR100(
        CIFAR_ROOT,
        train=False,
        download=DOWNLOAD_DATA,
        transform=eval_transform,
    )

    calib_ds = torchvision.datasets.CIFAR100(
        CIFAR_ROOT,
        train=True,
        download=DOWNLOAD_DATA,
        transform=eval_transform,
    )

    calib_loader = make_loader(
        calib_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    subset_loaders = create_subsets(test_ds_full, N_SUBSETS, SUBSET_SIZE)

    print(f"Created {N_SUBSETS} CIFAR-100 test subsets of {SUBSET_SIZE} images each.")


# ============================================================
# MODEL METADATA
# ============================================================

def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def clean_model_name(path):
    stem = Path(path).stem

    # Possible baseline names produced by your scripts.
    if stem in {"baseline_fp32", "baseline_fp32_fp32", "baseline"}:
        return "baseline"

    if stem in {"distilled_resnet18_fp32", "distilled_resnet18_fp32_fp32", "distilled_resnet18"}:
        return "distilled_resnet18"

    stem = stem.replace("_fp32_finetuned", "")
    stem = stem.replace("_fp32", "")
    stem = stem.replace("_finetuned", "")

    return stem


def infer_prune_from_name(name):
    if name in {"baseline", "distilled_resnet18"}:
        return 0

    m = re.search(r"structured_(\d+)", name)

    if m:
        return int(m.group(1))

    return None


def find_fp32_models():
    if not FP32_MODEL_DIR.exists():
        raise RuntimeError(f"FP32_MODEL_DIR does not exist: {FP32_MODEL_DIR}")

    paths = sorted(FP32_MODEL_DIR.glob("*.pth"))
    if MODEL_PATTERN:
        matcher = re.compile(MODEL_PATTERN)
        paths = [path for path in paths if matcher.search(clean_model_name(path))]

    def sort_key(p):
        name = clean_model_name(p)
        prune = infer_prune_from_name(name)

        if prune is None:
            prune = 999

        return prune, str(p)

    return sorted(paths, key=sort_key)


# ============================================================
# ONNX EXPORT
# ============================================================

def export_to_onnx(model_path, onnx_path):
    if onnx_path.exists() and not FORCE_EXPORT_ONNX:
        print(f"ONNX exists, skipping: {onnx_path}")
        return

    print(f"Exporting ONNX: {model_path.name} -> {onnx_path.name}")

    model = safe_torch_load(model_path, map_location="cpu")

    if not isinstance(model, nn.Module):
        raise RuntimeError(f"{model_path} did not load as torch.nn.Module")

    model.cpu()
    model.float()
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch"},
                "output": {0: "batch"},
            },
            opset_version=13,
            do_constant_folding=True,
        )

    print(f"Saved ONNX: {onnx_path}")


# ============================================================
# INT8 CALIBRATOR WITHOUT PYCUDA
# ============================================================

class EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibrator using PyTorch CUDA tensors instead of PyCUDA.
    """

    def __init__(self, loader, cache_file, max_batches):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.loader = loader
        self.cache_file = str(cache_file)
        self.max_batches = int(max_batches)

        self.batch_iter = iter(self.loader)
        self.current_batch = 0
        self.current_input = None  # keep tensor alive while TRT reads pointer

    def get_batch_size(self):
        return BATCH_SIZE

    def get_batch(self, names):
        if self.current_batch >= self.max_batches:
            return None

        try:
            images, _ = next(self.batch_iter)
        except StopIteration:
            return None

        if images.shape[0] != BATCH_SIZE:
            return None

        self.current_input = images.contiguous().to(
            device="cuda",
            dtype=torch.float32,
            non_blocking=False,
        )

        torch.cuda.synchronize()

        self.current_batch += 1

        if self.current_batch == 1 or self.current_batch % 25 == 0:
            print(f"    INT8 calibration batch {self.current_batch}/{self.max_batches}")

        return [int(self.current_input.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"    Using existing calibration cache: {self.cache_file}")

            with open(self.cache_file, "rb") as f:
                return f.read()

        return None

    def write_calibration_cache(self, cache):
        print(f"    Writing calibration cache: {self.cache_file}")

        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ============================================================
# TENSORRT ENGINE BUILDING
# ============================================================

def set_workspace(config, workspace_gb=2):
    workspace_bytes = int(workspace_gb * (1024 ** 3))

    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def build_engine(onnx_path, engine_path, precision):
    if engine_path.exists() and not FORCE_BUILD_ENGINE:
        print(f"Engine exists, skipping: {engine_path}")
        return

    print("\n" + "-" * 80)
    print(f"Building TensorRT engine: {engine_path.name}")
    print(f"Precision: {precision.upper()}")
    print("-" * 80)

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    set_workspace(config, workspace_gb=2)

    with open(onnx_path, "rb") as f:
        model_bytes = f.read()

    if not parser.parse(model_bytes):
        errors = []

        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))

        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    if network.num_inputs != 1:
        raise RuntimeError(f"Expected 1 network input, got {network.num_inputs}")

    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    profile = builder.create_optimization_profile()

    profile.set_shape(
        input_name,
        min=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        opt=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
        max=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
    )

    config.add_optimization_profile(profile)

    if precision == "fp32":
        if not ALLOW_TF32_FOR_FP32:
            try:
                config.clear_flag(trt.BuilderFlag.TF32)
                print("    TF32 disabled for FP32 engine.")
            except Exception:
                pass

    elif precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("    WARNING: platform_has_fast_fp16 is False.")

        config.set_flag(trt.BuilderFlag.FP16)

    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("    WARNING: platform_has_fast_int8 is False.")

        config.set_flag(trt.BuilderFlag.INT8)

        if INT8_USE_FP16_FALLBACK and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("    INT8 engine uses FP16 fallback for unsupported layers.")

        cache_file = engine_path.with_suffix(".calib.cache")

        calibrator = EntropyCalibrator2(
            loader=calib_loader,
            cache_file=cache_file,
            max_batches=CALIB_BATCHES,
        )

        config.int8_calibrator = calibrator

    else:
        raise ValueError(f"Unknown precision: {precision}")

    serialized_engine = None

    if hasattr(builder, "build_serialized_network"):
        serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("TensorRT engine build failed.")

        serialized_engine = engine.serialize()

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine serialization failed.")

    with open(engine_path, "wb") as f:
        f.write(bytes(serialized_engine))

    print(f"Saved engine: {engine_path}")


# ============================================================
# TENSORRT INFERENCE WITHOUT PYCUDA
# ============================================================

class TRTInfer:
    """
    TensorRT inference using PyTorch CUDA tensors instead of PyCUDA.
    Works with TensorRT Python APIs that expose binding methods.
    """

    def __init__(self, engine_path):
        self.engine_path = str(engine_path)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        if self.engine is None:
            raise RuntimeError(f"Could not deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context.")

        self.input_idx = None
        self.output_idx = None

        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                self.input_idx = idx
            else:
                self.output_idx = idx

        if self.input_idx is None or self.output_idx is None:
            raise RuntimeError("Could not identify TensorRT input/output bindings.")

        self.input_dtype = trt_dtype_to_torch(
            self.engine.get_binding_dtype(self.input_idx)
        )

        self.output_dtype = trt_dtype_to_torch(
            self.engine.get_binding_dtype(self.output_idx)
        )

    def prepare_input(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")

        return x.contiguous().to(
            device="cuda",
            dtype=self.input_dtype,
            non_blocking=True,
        )

    def infer(self, x_cpu_or_gpu):
        x_gpu = self.prepare_input(x_cpu_or_gpu)
        return self.infer_prepared(x_gpu)

    def infer_prepared(self, x_gpu):
        if not x_gpu.is_cuda:
            raise RuntimeError("infer_prepared expects a CUDA tensor.")

        if x_gpu.dtype != self.input_dtype:
            raise RuntimeError(
                f"infer_prepared expected dtype {self.input_dtype}, got {x_gpu.dtype}"
            )

        x_gpu = x_gpu.contiguous()

        self.context.set_binding_shape(self.input_idx, tuple(x_gpu.shape))

        output_shape = tuple(self.context.get_binding_shape(self.output_idx))

        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(f"Invalid dynamic output shape: {output_shape}")

        output_gpu = torch.empty(
            output_shape,
            device="cuda",
            dtype=self.output_dtype,
        )

        bindings = [0] * self.engine.num_bindings
        bindings[self.input_idx] = int(x_gpu.data_ptr())
        bindings[self.output_idx] = int(output_gpu.data_ptr())

        stream_handle = int(torch.cuda.current_stream().cuda_stream)

        ok = self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=stream_handle,
        )

        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed.")

        return output_gpu


# ============================================================
# EVALUATION
# ============================================================

def warmup_runner(runner, loader):
    stabilize_jetson()

    count = 0

    for x, _ in loader:
        _ = runner.infer(x)

        count += 1

        if count >= WARMUP_BATCHES:
            break

    torch.cuda.synchronize()


@torch.no_grad()
def evaluate_runner_on_subset(runner, subset, loader):
    n_images = len(subset)

    # ---------------- Accuracy pass ----------------
    correct = 0
    total = 0

    for x, y in loader:
        logits_gpu = runner.infer(x)

        pred_cpu = torch.argmax(logits_gpu, dim=1).detach().cpu()
        y_cpu = y.cpu()

        correct += pred_cpu.eq(y_cpu).sum().item()
        total += y_cpu.size(0)

    torch.cuda.synchronize()

    acc = 100.0 * correct / total

    # ---------------- Timed pass + power ----------------
    tegra_monitor = PowerMonitor(interval_ms=5)
    tegra_monitor.start()

    shelly_monitor = None

    if ENABLE_SHELLY_POWER:
        shelly_monitor = ShellyPowerMonitor(interval_s=SHELLY_POLL_INTERVAL_S)
        shelly_monitor.start()

    torch.cuda.synchronize()

    t0_e2e = time.time()
    total_pure_inference_time = 0.0

    for x, _ in loader:
        # End-to-end timer includes host-to-device transfer.
        x_gpu = runner.prepare_input(x)

        # Pure TensorRT timer excludes host-to-device transfer.
        torch.cuda.synchronize()
        t0 = time.time()

        _ = runner.infer_prepared(x_gpu)

        torch.cuda.synchronize()
        t1 = time.time()

        total_pure_inference_time += t1 - t0

    torch.cuda.synchronize()

    t1_e2e = time.time()
    elapsed_e2e = t1_e2e - t0_e2e

    time.sleep(0.2)

    (
        power_tegra,
        power_cpu_gpu_cv,
        energy_tegra_j,
        energy_cpu_gpu_cv_j,
        gpu_temp_mean_c,
        gpu_temp_max_c,
    ) = tegra_monitor.stop_window(t0_e2e, t1_e2e)

    if ENABLE_SHELLY_POWER and shelly_monitor is not None:
        power_shelly = shelly_monitor.stop_window(t0_e2e, t1_e2e)
    else:
        power_shelly = 0.0

    if total_pure_inference_time > 0:
        throughput_trt = n_images / total_pure_inference_time
        latency_trt = 1000.0 / throughput_trt
    else:
        throughput_trt = 0.0
        latency_trt = float("inf")

    if elapsed_e2e > 0:
        throughput_e2e = n_images / elapsed_e2e
        latency_e2e = 1000.0 / throughput_e2e
    else:
        throughput_e2e = 0.0
        latency_e2e = float("inf")

    return (
        acc,
        throughput_trt,
        latency_trt,
        throughput_e2e,
        latency_e2e,
        power_tegra,
        power_cpu_gpu_cv,
        1000.0 * energy_tegra_j / n_images,
        1000.0 * energy_cpu_gpu_cv_j / n_images,
        gpu_temp_mean_c,
        gpu_temp_max_c,
        power_shelly,
    )


def evaluate_engine(engine_path):
    runner = TRTInfer(engine_path)

    print("    Warming up...", end=" ", flush=True)
    warmup_runner(runner, subset_loaders[0][1])
    print("done.")

    acc_list = []
    throughput_trt_list = []
    latency_trt_list = []
    throughput_e2e_list = []
    latency_e2e_list = []
    power_tegra_list = []
    power_cpu_gpu_cv_list = []
    energy_tegra_list = []
    energy_cpu_gpu_cv_list = []
    gpu_temp_mean_list = []
    gpu_temp_max_list = []
    power_shelly_list = []

    for i, (subset, loader) in enumerate(subset_loaders):
        print(f"    Subset {i + 1}/{N_SUBSETS}...", end=" ", flush=True)

        (
            acc,
            throughput_trt,
            latency_trt,
            throughput_e2e,
            latency_e2e,
            power_tegra,
            power_cpu_gpu_cv,
            energy_tegra_mj_per_image,
            energy_cpu_gpu_cv_mj_per_image,
            gpu_temp_mean_c,
            gpu_temp_max_c,
            power_shelly,
        ) = evaluate_runner_on_subset(runner, subset, loader)

        acc_list.append(acc)
        throughput_trt_list.append(throughput_trt)
        latency_trt_list.append(latency_trt)
        throughput_e2e_list.append(throughput_e2e)
        latency_e2e_list.append(latency_e2e)
        power_tegra_list.append(power_tegra)
        power_cpu_gpu_cv_list.append(power_cpu_gpu_cv)
        energy_tegra_list.append(energy_tegra_mj_per_image)
        energy_cpu_gpu_cv_list.append(energy_cpu_gpu_cv_mj_per_image)
        gpu_temp_mean_list.append(gpu_temp_mean_c)
        gpu_temp_max_list.append(gpu_temp_max_c)
        power_shelly_list.append(power_shelly)

        print(
            f"ACC={acc:.2f}% | "
            f"TRT_THR={throughput_trt:.1f} img/s | "
            f"TRT_LAT={latency_trt:.2f} ms | "
            f"E2E_THR={throughput_e2e:.1f} img/s | "
            f"E2E_LAT={latency_e2e:.2f} ms | "
            f"PWR_VDD_IN={power_tegra:.2f} W | "
            f"PWR_CPU_GPU_CV={power_cpu_gpu_cv:.2f} W | "
            f"ENERGY={energy_tegra_mj_per_image:.3f} mJ/img | "
            f"GPU_TEMP={gpu_temp_mean_c:.1f}/{gpu_temp_max_c:.1f} C mean/max | "
            f"PWR_shelly={power_shelly:.2f} W"
        )

    return (
        acc_list,
        throughput_trt_list,
        latency_trt_list,
        throughput_e2e_list,
        latency_e2e_list,
        power_tegra_list,
        power_cpu_gpu_cv_list,
        energy_tegra_list,
        energy_cpu_gpu_cv_list,
        gpu_temp_mean_list,
        gpu_temp_max_list,
        power_shelly_list,
    )


# ============================================================
# PLOTTING
# ============================================================

def save_plots(df):
    sns.set_theme(style="whitegrid")

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
        }
    )

    palette = {
        "FP32": sns.color_palette("tab10")[0],
        "FP16": sns.color_palette("tab10")[1],
        "INT8": sns.color_palette("tab10")[2],
    }

    plot_df = df.copy()
    plot_df["precision_label"] = plot_df["precision"].str.upper()

    def lineplot(metric, ylabel, title, filename):
        plt.figure(figsize=(10, 6))

        kwargs = dict(
            data=plot_df,
            x="prune",
            y=metric,
            hue="precision_label",
            style="precision_label",
            markers=True,
            dashes=False,
            palette=palette,
            linewidth=2,
            markersize=8,
        )

        try:
            sns.lineplot(**kwargs, errorbar=("ci", CI_LEVEL), err_style="band")
        except Exception:
            sns.lineplot(**kwargs, ci=CI_LEVEL)

        plt.xlabel("Structured Pruning Ratio (%)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(title="Precision", loc="best")
        plt.tight_layout()

        out_path = RESULTS_DIR / filename
        plt.savefig(out_path)
        plt.close()

        print(f"Saved plot: {out_path}")

    lineplot(
        "acc",
        "Top-1 Accuracy (%)",
        f"CIFAR-100 TensorRT Top-1 Accuracy vs Pruning Ratio [95% CI, n={N_SUBSETS}]",
        "accuracy_trt.png",
    )

    lineplot(
        "throughput_trt",
        "Throughput (img/s)",
        f"CIFAR-100 TensorRT Pure Forward Throughput vs Pruning Ratio [95% CI, n={N_SUBSETS}]",
        "throughput_trt.png",
    )

    lineplot(
        "latency_trt",
        "Latency (ms/img)",
        f"CIFAR-100 TensorRT Pure Forward Latency vs Pruning Ratio [95% CI, n={N_SUBSETS}]",
        "latency_trt.png",
    )

    lineplot(
        "throughput_e2e",
        "Throughput (img/s)",
        f"CIFAR-100 TensorRT End-to-End Throughput vs Pruning Ratio [95% CI, n={N_SUBSETS}]",
        "throughput_e2e_trt.png",
    )

    lineplot(
        "latency_e2e",
        "Latency (ms/img)",
        f"CIFAR-100 TensorRT End-to-End Latency vs Pruning Ratio [95% CI, n={N_SUBSETS}]",
        "latency_e2e_trt.png",
    )

    lineplot(
        "power_tegra",
        "Power (W)",
        f"CIFAR-100 TensorRT tegrastats Power vs Pruning Ratio [95% CI, n={N_SUBSETS}]",
        "power_tegra_trt.png",
    )

    if ENABLE_SHELLY_POWER:
        lineplot(
            "power_shelly",
            "Power (W)",
            f"CIFAR-100 TensorRT Shelly Power vs Pruning Ratio [95% CI, n={N_SUBSETS}]",
            "power_shelly_trt.png",
        )

    size_df = (
        df.groupby(["precision", "prune"])
        .agg(size_mb=("engine_size_mb", "mean"))
        .reset_index()
    )

    size_df["precision_label"] = size_df["precision"].str.upper()

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=size_df,
        x="prune",
        y="size_mb",
        hue="precision_label",
        style="precision_label",
        markers=True,
        dashes=False,
        palette=palette,
        linewidth=2,
        markersize=8,
    )

    plt.xlabel("Structured Pruning Ratio (%)")
    plt.ylabel("TensorRT Engine Size (MB)")
    plt.title("CIFAR-100 TensorRT Engine Size vs Pruning Ratio")
    plt.legend(title="Precision", loc="best")
    plt.tight_layout()

    out_path = RESULTS_DIR / "engine_size_trt.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    apply_args(args)
    print_environment()
    build_data()

    model_paths = find_fp32_models()

    if not model_paths:
        raise RuntimeError(f"No FP32 .pth models found in {FP32_MODEL_DIR}")

    print("\nModels found:")

    for p in model_paths:
        print(f"  - {p}")

    results = []
    engine_jobs = []

    # 1. Export ONNX and build TensorRT engines.
    for model_path in model_paths:
        model_name = clean_model_name(model_path)
        prune = infer_prune_from_name(model_name)

        if prune is None:
            print(f"Skipping unknown model name: {model_path}")
            continue

        onnx_path = ONNX_DIR / f"{model_name}.onnx"

        export_to_onnx(model_path, onnx_path)

        for precision in PRECISIONS:
            precision_engine_dir = ENGINE_DIR / precision
            precision_engine_dir.mkdir(parents=True, exist_ok=True)

            engine_path = precision_engine_dir / f"{model_name}_{precision}.engine"

            build_engine(
                onnx_path=onnx_path,
                engine_path=engine_path,
                precision=precision,
            )

            engine_jobs.append(
                {
                    "model_name": model_name,
                    "prune": prune,
                    "precision": precision,
                    "engine_path": engine_path,
                }
            )

    # 2. Evaluate TensorRT engines.
    for job in engine_jobs:
        model_name = job["model_name"]
        prune = job["prune"]
        precision = job["precision"]
        engine_path = job["engine_path"]

        print("\n" + "=" * 100)
        print(f"Evaluating TensorRT engine: {engine_path}")
        print(f"Model: {model_name} | Precision: {precision.upper()} | Prune: {prune}%")
        print("=" * 100)

        try:
            (
                acc_list,
                throughput_trt_list,
                latency_trt_list,
                throughput_e2e_list,
                latency_e2e_list,
                power_tegra_list,
                power_cpu_gpu_cv_list,
                energy_tegra_list,
                energy_cpu_gpu_cv_list,
                gpu_temp_mean_list,
                gpu_temp_max_list,
                power_shelly_list,
            ) = evaluate_engine(engine_path)

        except Exception as e:
            print(f"FAILED TO EVALUATE: {type(e).__name__}: {e}")
            continue

        engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)

        for run_idx in range(N_SUBSETS):
            results.append(
                {
                    "dataset": "CIFAR-100",
                    "model_name": model_name,
                    "precision": precision,
                    "prune": prune,
                    "run": run_idx + 1,
                    "acc": acc_list[run_idx],
                    "throughput_trt": throughput_trt_list[run_idx],
                    "latency_trt": latency_trt_list[run_idx],
                    "throughput_e2e": throughput_e2e_list[run_idx],
                    "latency_e2e": latency_e2e_list[run_idx],
                    "power_tegra": power_tegra_list[run_idx],
                    "power_cpu_gpu_cv": power_cpu_gpu_cv_list[run_idx],
                    "energy_e2e_mj_per_image": energy_tegra_list[run_idx],
                    "energy_cpu_gpu_cv_mj_per_image": energy_cpu_gpu_cv_list[run_idx],
                    "gpu_temp_mean_c": gpu_temp_mean_list[run_idx],
                    "gpu_temp_max_c": gpu_temp_max_list[run_idx],
                    "power_shelly": power_shelly_list[run_idx],
                    "engine_size_mb": engine_size_mb,
                    "engine_path": str(engine_path),
                }
            )

        print(f"  Summary across {N_SUBSETS} subsets:")
        print(f"    ACC:          {np.mean(acc_list):.2f}% ± {np.std(acc_list):.2f}")
        print(f"    TRT THR:      {np.mean(throughput_trt_list):.1f} img/s ± {np.std(throughput_trt_list):.1f}")
        print(f"    TRT LAT:      {np.mean(latency_trt_list):.2f} ms ± {np.std(latency_trt_list):.2f}")
        print(f"    E2E THR:      {np.mean(throughput_e2e_list):.1f} img/s ± {np.std(throughput_e2e_list):.1f}")
        print(f"    E2E LAT:      {np.mean(latency_e2e_list):.2f} ms ± {np.std(latency_e2e_list):.2f}")
        print(f"    POWER tegra:  {np.mean(power_tegra_list):.2f} W ± {np.std(power_tegra_list):.2f}")
        print(f"    CPU/GPU/CV:   {np.mean(power_cpu_gpu_cv_list):.2f} W ± {np.std(power_cpu_gpu_cv_list):.2f}")
        if ENABLE_SHELLY_POWER:
            print(f"    POWER shelly: {np.mean(power_shelly_list):.2f} W ± {np.std(power_shelly_list):.2f}")
        print(f"    ENGINE SIZE:  {engine_size_mb:.2f} MB")

        torch.cuda.empty_cache()

    if not results:
        print("\nNo successful TensorRT evaluations.")
        return

    df = pd.DataFrame(results)

    csv_path = RESULTS_DIR / "tensorrt_results_all_runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    summary_df = (
        df.groupby(["precision", "prune", "model_name"])
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            throughput_trt_mean=("throughput_trt", "mean"),
            throughput_trt_std=("throughput_trt", "std"),
            latency_trt_mean=("latency_trt", "mean"),
            latency_trt_std=("latency_trt", "std"),
            throughput_e2e_mean=("throughput_e2e", "mean"),
            throughput_e2e_std=("throughput_e2e", "std"),
            latency_e2e_mean=("latency_e2e", "mean"),
            latency_e2e_std=("latency_e2e", "std"),
            power_tegra_mean=("power_tegra", "mean"),
            power_tegra_std=("power_tegra", "std"),
            power_cpu_gpu_cv_mean=("power_cpu_gpu_cv", "mean"),
            power_cpu_gpu_cv_std=("power_cpu_gpu_cv", "std"),
            energy_e2e_mj_per_image_mean=("energy_e2e_mj_per_image", "mean"),
            energy_e2e_mj_per_image_std=("energy_e2e_mj_per_image", "std"),
            energy_cpu_gpu_cv_mj_per_image_mean=("energy_cpu_gpu_cv_mj_per_image", "mean"),
            energy_cpu_gpu_cv_mj_per_image_std=("energy_cpu_gpu_cv_mj_per_image", "std"),
            gpu_temp_mean_c=("gpu_temp_mean_c", "mean"),
            gpu_temp_max_c=("gpu_temp_max_c", "max"),
            power_shelly_mean=("power_shelly", "mean"),
            power_shelly_std=("power_shelly", "std"),
            engine_size_mb=("engine_size_mb", "mean"),
        )
        .reset_index()
    )

    summary_csv = RESULTS_DIR / "tensorrt_summary_by_model.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV: {summary_csv}")

    save_plots(df)

    print("\n✓ TensorRT CIFAR-100 evaluation complete.")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
