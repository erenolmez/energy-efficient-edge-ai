"""Job 3, stage A: A40 TensorRT FP16 engines and per-model cost profiles.

  run        (GPU, Slurm) Verify and load the eight 128px HPC checkpoints (hash-checked as in
             hpc/evaluate_imagenet_comparison.py), export ONNX, build one TensorRT FP16 engine per
             model with a static optimization profile for every probed batch size plus one dynamic
             1-128 profile (for debatch sub-batches), check engine accuracy against PyTorch on the
             three non-test roles, time cold engine loads, check that all eight engines fit in VRAM
             at batch 128 with headroom, and profile latency and GPU energy per model and batch.
  summarize  (CPU) Markdown tables from the run's JSON files.

Two timing levels, both with all eight engines resident:
  engine    preprocessed FP32 128px input already on the GPU; engine call only.
  pipeline  pinned host uint8 32px batch -> H2D -> GPU bilinear resize to 128 -> normalize ->
            engine -> argmax -> class-ID-only readback to pinned host memory.
Every timed iteration ends with a stream synchronize (one batch in flight, as in deployment).
GPU energy comes from the NVML total-energy counter of the allocated GPU, matched by UUID
(PCI bus id as the only fallback; never index 0), totalled over windows of at least
--min-seconds. Energy is whole-GPU (including idle draw); the idle draw is also reported.

Roles used: classifier_validation, router_train, router_calibration. The test set is never
opened. These are cost measurements on the target GPU (A40) for policy locking in stage B,
not routed results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROLES = ("classifier_validation", "router_train", "router_calibration")
PROBE = (1, 8, 16, 32, 64, 128)
MAX_BATCH = 128
SIZE = 128
REFERENCE = "resnet101"
RESERVE_BYTES = 1024 ** 3  # same 1 GiB inference reserve as runtime/routed_residency.py
ENGINE_TOL_PP = 0.3        # TensorRT vs PyTorch FP32, same CPU preprocessing
ENGINE_MIN_AGREE = 99.0    # % top-1 agreement, TensorRT vs PyTorch FP32
RESIZE_TOL_PP = 0.5        # GPU resize vs CPU (PIL) resize, same engine


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n", encoding="utf-8")
    tmp.replace(path)


def log(**row):
    print(json.dumps(row, default=float), flush=True)


# ------------------------------------------------------------------ pure helpers (tested)

def canonical_uuid(value):
    if isinstance(value, bytes):
        value = value.decode()
    value = str(value).strip().lower()
    for prefix in ("gpu-", "mig-"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value


def canonical_bus(value):
    """'0000:3B:00.0' or '00000000:3b:00.0' -> '00000000:3b:00.0'."""
    if isinstance(value, bytes):
        value = value.decode()
    parts = str(value).strip().lower().split(":")
    if len(parts) == 2:
        parts = ["0"] + parts
    domain, bus, devfn = parts
    dev, _, fn = devfn.partition(".")
    return f"{int(domain, 16):08x}:{int(bus, 16):02x}:{int(dev, 16):02x}.{fn or '0'}"


def match_device(nvml_devices, torch_uuid=None, torch_bus=None):
    """nvml_devices: [(index, uuid, pci_bus_id)]. Returns (index, method)."""
    if torch_uuid:
        want = canonical_uuid(torch_uuid)
        hits = [d for d in nvml_devices if canonical_uuid(d[1]) == want]
        if len(hits) == 1:
            return hits[0][0], "uuid"
    if torch_bus:
        want = canonical_bus(torch_bus)
        hits = [d for d in nvml_devices if canonical_bus(d[2]) == want]
        if len(hits) == 1:
            return hits[0][0], "pci_bus_id"
    raise RuntimeError("Could not match the CUDA device to an NVML device by UUID or PCI bus id")


def window_metrics(times_s, energy_mj, images, idle_w):
    """Latency and energy of one measurement window (one cell, one repeat)."""
    times = np.asarray(times_s, dtype=float)
    total_s = float(times.sum())
    out = dict(iters=len(times), images=images, window_s=total_s,
               latency_ms=1000 * float(np.median(times)),
               latency_p90_ms=1000 * float(np.percentile(times, 90)),
               latency_mean_ms=1000 * float(times.mean()))
    out["ms_per_image"] = 1000 * total_s / images
    if energy_mj is not None and energy_mj > 0:
        out["energy_mj_per_image"] = energy_mj / images
        out["avg_power_w"] = energy_mj / 1000 / total_s
        if idle_w is not None:
            out["energy_above_idle_mj_per_image"] = (energy_mj - 1000 * idle_w * total_s) / images
    else:
        out["energy_mj_per_image"] = None
    return out


def aggregate(rows):
    """Median over repeats per (variant, model, level, batch)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["variant"], r["model"], r["level"], r["batch"]), []).append(r)
    keys = ("latency_ms", "latency_p90_ms", "ms_per_image", "energy_mj_per_image",
            "energy_above_idle_mj_per_image", "avg_power_w")
    out = []
    for (variant, model, level, batch), rs in sorted(groups.items()):
        agg = dict(variant=variant, model=model, level=level, batch=batch, repeats=len(rs))
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            agg[k] = float(np.median(vals)) if vals else None
            if vals and len(vals) > 1:
                agg[k + "_range"] = [float(min(vals)), float(max(vals))]
        # Cost of one whole batch, in the schema of hpc/complementarity_analysis.load_costs.
        agg["energy_mj"] = None if agg["energy_mj_per_image"] is None else agg["energy_mj_per_image"] * batch
        out.append(agg)
    return out


def accuracy_checks(acc, names):
    """acc[variant][model][role] = (accuracy %, agreement-with-torch_fp32_cpu %)."""
    checks = []
    for m in names:
        for role in ROLES:
            ref = acc["torch_fp32_cpu"][m][role][0]
            trt_cpu, agree_cpu = acc["trt_fp16_cpu"][m][role]
            trt_gpu, agree_gpu = acc["trt_fp16_gpu"][m][role]
            engine_ok = abs(trt_cpu - ref) <= ENGINE_TOL_PP and agree_cpu >= ENGINE_MIN_AGREE
            resize_ok = abs(trt_gpu - trt_cpu) <= RESIZE_TOL_PP
            checks.append(dict(model=m, role=role, torch_fp32_cpu=ref, trt_fp16_cpu=trt_cpu,
                               trt_fp16_gpu=trt_gpu, agreement_cpu=agree_cpu, agreement_gpu=agree_gpu,
                               engine_ok=engine_ok, resize_ok=resize_ok))
    return checks


# --------------------------------------------------------------------------------- run

def run(args):
    import inspect
    import os
    import random

    import torch
    import torch.nn.functional as F
    from torchvision import transforms as T
    from torchvision.datasets import CIFAR100
    from torch.utils.data import DataLoader, Subset
    import tensorrt as trt
    import pynvml

    from hpc.clean_protocol import file_hash, fingerprint, validate_split
    from hpc.full_factorial_protocol import checked_plan
    from hpc.train_imagenet_comparison import CASES, SOURCE_PLAN, STUDY
    from model_pipeline.models import create_classifier

    require(os.environ.get("SLURM_JOB_ID"), "Run on a Slurm allocation, not a login node")
    require(torch.cuda.is_available(), "CUDA allocation required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "engines").mkdir()
    probe = tuple(sorted(set(args.probe)))
    require(all(1 <= b <= MAX_BATCH for b in probe) and MAX_BATCH in probe, "Probe must include 128")
    rng = random.Random(args.seed)

    # ---- device identity and NVML handle matched by UUID
    props = torch.cuda.get_device_properties(0)
    torch_uuid = getattr(props, "uuid", None)
    torch_bus = None
    if all(hasattr(props, a) for a in ("pci_domain_id", "pci_bus_id", "pci_device_id")):
        torch_bus = f"{props.pci_domain_id:08x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    pynvml.nvmlInit()
    devices = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        devices.append((i, pynvml.nvmlDeviceGetUUID(h), pynvml.nvmlDeviceGetPciInfo(h).busId))
    nvml_index, method = match_device(devices, torch_uuid, torch_bus)
    handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
    energy = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # mJ
    energy()
    device = dict(name=torch.cuda.get_device_name(0), torch_uuid=str(torch_uuid), torch_pci=torch_bus,
                  nvml_index=nvml_index, nvml_uuid=str(devices[nvml_index][1]), match=method,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  driver=pynvml.nvmlSystemGetDriverVersion(), torch=torch.__version__,
                  cuda=torch.version.cuda, tensorrt=trt.__version__,
                  power_limit_w=pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000,
                  memory_total_bytes=pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                  host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"))
    require("A40" in device["name"] or args.allow_other_gpu,
            f"Expected an A40, got {device['name']} (pass --allow-other-gpu to profile anyway)")
    log(event="device", **device)

    def gpu_state():
        return dict(temp_c=pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                    sm_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                    mem_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))

    # ---- verified checkpoints (same checks as evaluate_imagenet_comparison.py)
    scratch = checked_plan(args.base / "scratch32_full_factorial_v1" / "plan.json")
    study = args.base / STUDY
    prepared = load_json(study / "prepared.json")
    spec = prepared["spec"]
    require(scratch["sha256"] == spec["scratch_plan_sha256"] == SOURCE_PLAN, "Plan mismatch")
    registry = "model_pipeline/models.py"
    require(file_hash(Path(registry)) == spec["source_hashes"][registry], "Model registry mismatch")
    split = load_json(args.base / "split_seed42.json")
    require(split["sha256"] == spec["split_sha256"], "Split differs from the training split")
    mean, std = scratch["normalization_mean"], scratch["normalization_std"]

    models, names, receipts = {}, [], []
    for cid in CASES:
        trial = scratch["trials"][cid]
        name = trial["architecture"]
        if args.models and name not in args.models:
            continue
        directory = study / f"case_{cid:04d}_{SIZE}px"
        summary = load_json(directory / "summary.json")
        recipe = summary["recipe"]
        require(summary["status"] == "complete" and summary["epochs_complete"] == 200
                and recipe["trial"] == trial and recipe["image_size"] == SIZE
                and recipe["plan_sha256"] == fingerprint(prepared)
                and recipe["initial"] == prepared["weights"][str(cid)], f"Summary mismatch: {directory}")
        path = directory / "latest.pt"
        digest = file_hash(path)
        require(digest == summary["checkpoint_sha256"], f"Hash mismatch: {path}")
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ck = torch.load(path, map_location="cpu", weights_only=True)
        require(ck["epoch"] == 200 and ck["recipe"] == recipe
                and ck["source_hashes"] == spec["source_hashes"]
                and ck["split_sha256"] == spec["split_sha256"]
                and ck["best_accuracy"] == summary["best_validation_accuracy"]
                and bool(ck["best_state"]), f"Best checkpoint mismatch: {path}")
        model = create_classifier(name, pretrained=False)
        model.load_state_dict(ck["best_state"], strict=True)
        del ck
        models[name] = model.eval()
        names.append(name)
        receipts.append(dict(case=cid, architecture=name, checkpoint=str(path), checkpoint_sha256=digest,
                             best_validation_accuracy=summary["best_validation_accuracy"]))
    require(names, "No models selected")
    log(event="checkpoints", models=names)

    # ---- ONNX export and TensorRT FP16 engine build
    logger = trt.Logger(trt.Logger.WARNING)
    export_kw = dict(input_names=["input"], output_names=["logits"], opset_version=17,
                     dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kw["dynamo"] = False
    profiles = [(b, b, b) for b in probe] + [(1, 32, MAX_BATCH)]  # last = dynamic
    manifest = []
    for name in names:
        onnx_path = out / "engines" / f"{name}.onnx"
        engine_path = out / "engines" / f"{name}_fp16.engine"
        t0 = time.perf_counter()
        with torch.no_grad():
            torch.onnx.export(models[name], torch.zeros(1, 3, SIZE, SIZE), str(onnx_path), **export_kw)
        export_s = time.perf_counter() - t0
        builder = trt.Builder(logger)
        flags = 0
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH") and int(trt.__version__.split(".")[0]) < 10:
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        require(parser.parse(onnx_path.read_bytes()),
                "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 ** 2)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_flag(trt.BuilderFlag.FP16)
        input_name = network.get_input(0).name
        for lo, opt, hi in profiles:
            profile = builder.create_optimization_profile()
            profile.set_shape(input_name, (lo, 3, SIZE, SIZE), (opt, 3, SIZE, SIZE), (hi, 3, SIZE, SIZE))
            config.add_optimization_profile(profile)
        t0 = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        require(serialized is not None, f"TensorRT build failed: {name}")
        engine_path.write_bytes(bytes(serialized))
        build_s = time.perf_counter() - t0
        del serialized, network, parser, config, builder
        manifest.append(dict(model=name, onnx=str(onnx_path), onnx_sha256=file_hash(onnx_path),
                             engine=str(engine_path), engine_sha256=file_hash(engine_path),
                             engine_bytes=engine_path.stat().st_size, export_s=export_s, build_s=build_s,
                             profiles=[list(p) for p in profiles],
                             precision="TensorRT FP16 enabled (TF32 off; builder may keep layers FP32); "
                                       "FP32 input and logits"))
        log(event="engine_built", **manifest[-1])
        write(out / "engines.json", dict(device=device, receipts=receipts, engines=manifest))

    stream = torch.cuda.Stream()

    def device_memory(engine, k):
        for attr in ("get_device_memory_size_for_profile_v2", "get_device_memory_size_for_profile"):
            if hasattr(engine, attr):
                return int(getattr(engine, attr)(k))
        return int(getattr(engine, "device_memory_size_v2", getattr(engine, "device_memory_size", 0)))

    # Allocate each context's activation memory for its own profile only (TensorRT >= 10).
    ALLOC = getattr(trt, "ExecutionContextAllocationStrategy", None)
    if ALLOC is not None and not hasattr(ALLOC, "ON_PROFILE_CHANGE"):
        ALLOC = None

    class Engine:
        """One deserialized engine with one execution context per optimization profile."""
        def __init__(self, blob):
            self.runtime = trt.Runtime(logger)
            self.engine = self.runtime.deserialize_cuda_engine(blob)
            require(self.engine is not None, "Engine deserialization failed")
            io = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.inp = next(n for n in io if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
            self.outp = next(n for n in io if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
            require(self.engine.get_tensor_dtype(self.inp) == trt.float32, "Expected FP32 input")
            self.output = torch.empty((MAX_BATCH, 100), device="cuda", dtype=torch.float32)
            self.ctx, self.shape = {}, {}
            for k, (lo, _, hi) in enumerate(profiles):
                ctx = (self.engine.create_execution_context(ALLOC.ON_PROFILE_CHANGE) if ALLOC
                       else self.engine.create_execution_context())
                require(ctx is not None, "Execution context allocation failed")
                require(ctx.set_optimization_profile_async(k, stream.cuda_stream), "Profile select failed")
                ctx.set_tensor_address(self.outp, self.output.data_ptr())
                self.ctx[lo if lo == hi else "dynamic"] = ctx
                self.shape[id(ctx)] = None
            stream.synchronize()
            self.device_bytes = sum(device_memory(self.engine, k) for k in range(len(profiles)))

        def __call__(self, x):
            """Enqueue on the current stream; x is contiguous FP32 CUDA (n,3,128,128)."""
            n = x.shape[0]
            ctx = self.ctx.get(n, self.ctx["dynamic"])
            if self.shape[id(ctx)] != n:
                require(ctx.set_input_shape(self.inp, (n, 3, SIZE, SIZE)), "Input shape rejected")
                self.shape[id(ctx)] = n
            ctx.set_tensor_address(self.inp, x.data_ptr())
            require(ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream), "TensorRT execution failed")
            return self.output[:n]

    # ---- data: non-test roles only
    transform = T.Compose([T.Resize((SIZE, SIZE), antialias=True), T.ToTensor(), T.Normalize(mean, std)])
    full = CIFAR100(args.data, train=True, transform=transform, download=False)  # never the test set
    validate_split(split, full.targets)
    indices = np.concatenate([np.asarray(split["indices"][r], dtype=np.int64) for r in ROLES])
    role_id = np.concatenate([np.full(len(split["indices"][r]), i, dtype=np.int8) for i, r in enumerate(ROLES)])
    labels = np.asarray(full.targets, dtype=np.int16)[indices]
    raw = np.ascontiguousarray(full.data[indices])  # (n,32,32,3) uint8, native CIFAR pixels
    n = len(indices)

    mean_t = torch.tensor(mean, device="cuda").view(1, 3, 1, 1)
    std_t = torch.tensor(std, device="cuda").view(1, 3, 1, 1)
    raw_dev = torch.empty((MAX_BATCH, 32, 32, 3), dtype=torch.uint8, device="cuda")
    prep_dev = torch.empty((MAX_BATCH, 3, SIZE, SIZE), dtype=torch.float32, device="cuda")
    ids_host = torch.empty(MAX_BATCH, dtype=torch.int64, pin_memory=True)

    def gpu_prepare(host):
        """Pinned uint8 NHWC 32px -> normalized FP32 NCHW 128px, on the current stream.
        Bilinear (half-pixel centres) then rounding to uint8 levels, to mirror PIL Resize."""
        b = host.shape[0]
        r = raw_dev[:b]
        r.copy_(host, non_blocking=True)
        y = F.interpolate(r.permute(0, 3, 1, 2).float(), size=(SIZE, SIZE), mode="bilinear",
                          align_corners=False)
        y = y.round_().clamp_(0, 255).div_(255)
        x = prep_dev[:b]
        torch.sub(y, mean_t, out=x)
        return x.div_(std_t)

    # ---- accuracy: TensorRT vs PyTorch FP32, CPU (training) and GPU preprocessing
    variants = ["torch_fp32_cpu", "trt_fp16_cpu", "torch_fp32_gpu", "trt_fp16_gpu"]
    ref_variants = ["torch_autocast_gpu", "torch_static_fp16_gpu"] if REFERENCE in names else []
    pred = {v: np.zeros((len(names), n), dtype=np.int16) for v in variants}
    pred.update({v: np.zeros(n, dtype=np.int16) for v in ref_variants})
    engines = {nm: Engine(Path(m["engine"]).read_bytes()) for nm, m in zip(names, manifest)}
    cuda_models = {nm: models[nm].cuda().eval() for nm in names}
    half_ref = None
    if ref_variants:
        import copy
        half_ref = copy.deepcopy(models[REFERENCE]).cuda().half().eval()
    loader = DataLoader(Subset(full, indices.tolist()), batch_size=MAX_BATCH, shuffle=False,
                        num_workers=4, pin_memory=True)
    with torch.inference_mode(), torch.cuda.stream(stream):
        offset = 0
        for x, _ in loader:
            x = x.cuda(non_blocking=True).contiguous()
            sl = slice(offset, offset + len(x))
            for j, nm in enumerate(names):
                lt = cuda_models[nm](x).float()
                le = engines[nm](x)
                require(bool(lt.isfinite().all()) and bool(le.isfinite().all()), f"Non-finite logits: {nm}")
                pred["torch_fp32_cpu"][j, sl] = lt.argmax(1).cpu().numpy()
                pred["trt_fp16_cpu"][j, sl] = le.argmax(1).cpu().numpy()
            offset += len(x)
        require(offset == n, "Incomplete CPU-preprocessed evaluation")
        for start in range(0, n, MAX_BATCH):
            host = torch.from_numpy(raw[start:start + MAX_BATCH]).pin_memory()
            x = gpu_prepare(host)
            sl = slice(start, start + len(host))
            for j, nm in enumerate(names):
                pred["torch_fp32_gpu"][j, sl] = cuda_models[nm](x).argmax(1).cpu().numpy()
                le = engines[nm](x)
                require(bool(le.isfinite().all()), f"Non-finite engine logits: {nm}")
                pred["trt_fp16_gpu"][j, sl] = le.argmax(1).cpu().numpy()
            if ref_variants:
                with torch.autocast("cuda", dtype=torch.float16):
                    pred["torch_autocast_gpu"][sl] = cuda_models[REFERENCE](x).float().argmax(1).cpu().numpy()
                pred["torch_static_fp16_gpu"][sl] = half_ref(x.half()).float().argmax(1).cpu().numpy()
    torch.cuda.synchronize()
    acc = {}
    for v in variants + ref_variants:
        acc[v] = {}
        for j, nm in enumerate(names):
            if pred[v].ndim == 1 and nm != REFERENCE:
                continue
            p = pred[v][j] if pred[v].ndim == 2 else pred[v]
            ref = pred["torch_fp32_cpu"][j]
            acc[v][nm] = {role: (100 * float((p[role_id == i] == labels[role_id == i]).mean()),
                                 100 * float((p[role_id == i] == ref[role_id == i]).mean()))
                          for i, role in enumerate(ROLES)}
    checks = accuracy_checks(acc, names)
    for c in checks:
        log(event="accuracy", **c)
    np.savez_compressed(out / "predictions.npz", models=np.array(names), roles=np.array(ROLES),
                        index=indices, role=role_id, label=labels,
                        **{f"pred_{v}": pred[v] for v in pred},
                        correct=pred["trt_fp16_gpu"] == labels[None, :])
    write(out / "accuracy.json", dict(accuracy=acc, checks=checks, tolerances=dict(
        engine_pp=ENGINE_TOL_PP, engine_min_agreement=ENGINE_MIN_AGREE, resize_pp=RESIZE_TOL_PP),
        note="Roles: classifier_validation 5k, router_train 3k, router_calibration 2k. Test not opened. "
             "*_cpu = torchvision PIL Resize as in training; *_gpu = the timed GPU pipeline."))

    # ---- cold load cost (host bytes -> engine + all contexts + one batch-128 call)
    for nm in names:
        engines[nm] = None
    for nm in names:
        cuda_models[nm].cpu()
    if half_ref is not None:
        half_ref.cpu()
    engines.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    loads = []
    x128 = torch.randn(MAX_BATCH, 3, SIZE, SIZE, device="cuda")
    for _ in range(args.load_repeats):
        for nm, m in zip(names, manifest):
            t0 = time.perf_counter()
            blob = Path(m["engine"]).read_bytes()
            t1 = time.perf_counter()
            with torch.cuda.stream(stream):
                e = Engine(blob)
                t2 = time.perf_counter()
                e(x128)
                stream.synchronize()
            t3 = time.perf_counter()
            loads.append(dict(model=nm, read_s=t1 - t0, deserialize_contexts_s=t2 - t1,
                              first_batch128_s=t3 - t2, total_s=t3 - t0))
            del e
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    log(event="loads", rows=loads)

    # ---- residency: all engines + buffers at batch 128
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free0, total = torch.cuda.mem_get_info()
    used0 = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    for nm, m in zip(names, manifest):
        engines[nm] = Engine(Path(m["engine"]).read_bytes())
    with torch.inference_mode(), torch.cuda.stream(stream):
        for nm in names:
            for b in probe:
                engines[nm](x128[:b])
        stream.synchronize()
    free1, _ = torch.cuda.mem_get_info()
    used1 = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    residency = dict(models=names, total_bytes=total, free_before_bytes=free0, free_after_bytes=free1,
                     engines_resident_bytes=free0 - free1, nvml_used_before=used0, nvml_used_after=used1,
                     profile_device_bytes={nm: engines[nm].device_bytes for nm in names},
                     context_allocation="ON_PROFILE_CHANGE" if ALLOC else "default (STATIC)",
                     engine_file_bytes={m["model"]: m["engine_bytes"] for m in manifest},
                     reserve_bytes=RESERVE_BYTES, all_fit_with_reserve=bool(free1 >= RESERVE_BYTES),
                     note="Each engine holds one context per profile (static 1/8/16/32/64/128 + dynamic), "
                          "all allocated up front; inference buffers are shared across models.")
    log(event="residency", **residency)
    write(out / "residency.json", dict(device=device, residency=residency, loads=loads))
    require(residency["all_fit_with_reserve"], "Engines do not fit with the 1 GiB reserve")

    # ---- profiling with all engines resident
    ref_model = ref_half = None
    if ref_variants:
        ref_model = models[REFERENCE].cuda().eval()
        ref_half = half_ref.cuda().eval()
    cells = [("trt_fp16", nm) for nm in names]
    if ref_variants:
        cells += [("torch_autocast", REFERENCE), ("torch_static_fp16", REFERENCE)]

    def forward(variant, nm, x):
        if variant == "trt_fp16":
            return engines[nm](x)
        if variant == "torch_autocast":
            with torch.autocast("cuda", dtype=torch.float16):
                return ref_model(x)
        return ref_half(x if x.dtype == torch.float16 else x.half())

    cal = np.flatnonzero(role_id == ROLES.index("router_calibration"))
    pools = {}
    for b in probe:
        k = max(4, min(16, len(cal) // b))
        pools[b] = [torch.from_numpy(raw[cal[i * b:(i + 1) * b]]).pin_memory() for i in range(k)]
    static_x = {b: torch.randn(b, 3, SIZE, SIZE, device="cuda") for b in probe}
    static_half = {b: x.half() for b, x in static_x.items()}

    def measure(variant, nm, level, b):
        pool = pools[b]

        def step(i):
            if level == "engine":
                forward(variant, nm, static_half[b] if variant == "torch_static_fp16" else static_x[b])
            else:
                logits = forward(variant, nm, gpu_prepare(pool[i % len(pool)]))
                ids_host[:b].copy_(logits.argmax(1), non_blocking=True)
            stream.synchronize()

        with torch.inference_mode(), torch.cuda.stream(stream):
            t_end = time.perf_counter() + args.warmup_seconds
            i = 0
            while i < args.warmup_iters or time.perf_counter() < t_end:
                step(i)
                i += 1
            times = []
            e0 = energy()
            start = time.perf_counter()
            while len(times) < args.min_iters or time.perf_counter() - start < args.min_seconds:
                t0 = time.perf_counter()
                step(len(times))
                times.append(time.perf_counter() - t0)
            e1 = energy()
        return window_metrics(times, e1 - e0, b * len(times), idle_w)

    # idle draw with engines resident
    torch.cuda.synchronize()
    time.sleep(2)
    e0, t0 = energy(), time.perf_counter()
    time.sleep(args.idle_seconds)
    idle_w = (energy() - e0) / 1000 / (time.perf_counter() - t0)
    log(event="idle", watts=idle_w, **gpu_state())

    # global warm-up: cycle every cell at batch 128 for --global-warmup seconds
    t_end = time.perf_counter() + args.global_warmup
    with torch.inference_mode(), torch.cuda.stream(stream):
        while time.perf_counter() < t_end:
            for variant, nm in cells:
                forward(variant, nm, static_x[MAX_BATCH])
            stream.synchronize()

    rows = []
    for repeat in range(args.repeats):
        order = [(v, nm, level, b) for v, nm in cells for level in ("engine", "pipeline") for b in probe]
        rng.shuffle(order)
        for variant, nm, level, b in order:
            before = gpu_state()
            row = dict(repeat=repeat, variant=variant, model=nm, level=level, batch=b,
                       **measure(variant, nm, level, b))
            after = gpu_state()
            row.update(temp_start_c=before["temp_c"], temp_end_c=after["temp_c"], sm_mhz_end=after["sm_mhz"])
            rows.append(row)
            log(event="cell", **row)
        write(out / "profile_raw.json", dict(device=device, idle_w=idle_w, rows=rows))

    summary = aggregate(rows)
    meta = dict(gpu=device["name"], device=device, idle_w=idle_w, probe=list(probe),
                warmup_seconds=args.warmup_seconds, warmup_iters=args.warmup_iters,
                min_seconds=args.min_seconds, min_iters=args.min_iters, repeats=args.repeats,
                levels=dict(engine="preprocessed FP32 128px input on GPU; model call only",
                            pipeline="pinned uint8 32px host batch, H2D, GPU resize + normalize, model, "
                                     "argmax, class-ID-only readback"),
                label="A40 COST PROFILE (stage A): per-model costs for policy locking, not routed results",
                test_evaluated=False)
    write(out / "profile.json", dict(meta=meta, rows=summary))
    for level in ("engine", "pipeline"):
        rows_l = [dict(r, precision="trt_fp16" if r["variant"] == "trt_fp16" else r["variant"])
                  for r in summary if r["level"] == level]
        write(out / f"costs_{level}.json", dict(meta=dict(meta, level=level), rows=rows_l))
    summarize(argparse.Namespace(output=out))


# ---------------------------------------------------------------------------- summarize

def summarize(args):
    out = args.output
    acc = load_json(out / "accuracy.json")
    res = load_json(out / "residency.json")
    prof = load_json(out / "profile.json")
    meta, rows = prof["meta"], prof["rows"]
    dev = meta["device"]
    gib = lambda v: f"{v / 1024 ** 3:.2f} GiB"
    lines = ["# Job 3 stage A: A40 TensorRT FP16 cost profile", "",
             f"{meta['label']}. GPU {dev['name']} (NVML device matched by {dev['match']}), driver "
             f"{dev['driver']}, TensorRT {dev['tensorrt']}, torch {dev['torch']}, job {dev['job']}. "
             f"Idle draw with engines resident: {meta['idle_w']:.1f} W. Medians over {meta['repeats']} "
             f"repeats; each window >= {meta['min_seconds']} s and >= {meta['min_iters']} batches.",
             "", "## Engine accuracy (non-test roles)", "",
             "| model | role | torch FP32 | TRT FP16 (CPU resize) | agree | TRT FP16 (GPU resize) | agree | engine | resize |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in acc["checks"]:
        lines.append(f"| {c['model']} | {c['role']} | {c['torch_fp32_cpu']:.2f} | {c['trt_fp16_cpu']:.2f} | "
                     f"{c['agreement_cpu']:.2f} | {c['trt_fp16_gpu']:.2f} | {c['agreement_gpu']:.2f} | "
                     f"{'ok' if c['engine_ok'] else 'FAIL'} | {'ok' if c['resize_ok'] else 'FAIL'} |")
    r = res["residency"]
    lines += ["", "## VRAM residency at batch 128", "",
              f"All {len(r['models'])} engines resident: {gib(r['engines_resident_bytes'])} used, "
              f"{gib(r['free_after_bytes'])} free of {gib(r['total_bytes'])} "
              f"({'fits' if r['all_fit_with_reserve'] else 'DOES NOT FIT'} with the 1 GiB reserve).", "",
              "| model | engine file | activation memory (sum over profiles) | cold load median (s) |", "|---|---|---|---|"]
    for nm in r["models"]:
        ls = [x["total_s"] for x in res["loads"] if x["model"] == nm]
        lines.append(f"| {nm} | {r['engine_file_bytes'][nm] / 1024 ** 2:.1f} MiB | "
                     f"{r['profile_device_bytes'][nm] / 1024 ** 2:.1f} MiB | {np.median(ls):.3f} |")
    fmt = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
    for level in ("pipeline", "engine"):
        lines += ["", f"## {level} level: {meta['levels'][level]}", "",
                  "| B | variant | model | ms/batch (median) | p90 | ms/img | mJ/img | mJ/img above idle | W | "
                  "latency vs RN101 TRT | energy vs RN101 TRT |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in meta["probe"]:
            sel = [x for x in rows if x["level"] == level and x["batch"] == b]
            ref = next((x for x in sel if x["variant"] == "trt_fp16" and x["model"] == REFERENCE), None)
            for x in sorted(sel, key=lambda x: (x["variant"] != "trt_fp16", x["ms_per_image"])):
                dl = de = "-"
                if ref:
                    dl = f"{100 * (x['ms_per_image'] / ref['ms_per_image'] - 1):+.0f}%"
                    if x["energy_mj_per_image"] and ref["energy_mj_per_image"]:
                        de = f"{100 * (x['energy_mj_per_image'] / ref['energy_mj_per_image'] - 1):+.0f}%"
                lines.append(f"| {b} | {x['variant']} | {x['model']} | {fmt(x['latency_ms'])} | "
                             f"{fmt(x['latency_p90_ms'])} | {fmt(x['ms_per_image'], 4)} | "
                             f"{fmt(x['energy_mj_per_image'], 2)} | {fmt(x['energy_above_idle_mj_per_image'], 2)} | "
                             f"{fmt(x['avg_power_w'], 0)} | {dl} | {de} |")
    text = "\n".join(lines) + "\n"
    (out / "profile.md").write_text(text, encoding="utf-8")
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", type=Path, required=True)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--models", nargs="*", default=None, help="Subset of architectures (smoke tests)")
    r.add_argument("--probe", type=int, nargs="*", default=list(PROBE))
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--min-seconds", type=float, default=4.0)
    r.add_argument("--min-iters", type=int, default=50)
    r.add_argument("--warmup-seconds", type=float, default=1.0)
    r.add_argument("--warmup-iters", type=int, default=20)
    r.add_argument("--global-warmup", type=float, default=30.0)
    r.add_argument("--idle-seconds", type=float, default=10.0)
    r.add_argument("--load-repeats", type=int, default=3)
    r.add_argument("--workspace-mib", type=int, default=2048)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--allow-other-gpu", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    run(args) if args.action == "run" else summarize(args)


if __name__ == "__main__":
    sys.exit(main())
