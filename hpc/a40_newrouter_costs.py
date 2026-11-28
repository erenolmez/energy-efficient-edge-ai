"""Job 3: A40 TensorRT cost of the new cross-fit routers (batch-set, smallest-sufficient, disagreement).

GPU only (gpu_a40). Same method as the cost phase of hpc.a40_routers: each router's deployment copy
(deploy/<network>-s42.pt from hpc.a40_crossfit, trained on all 5k pool images) is exported as one
TensorRT FP32 engine (TF32 off) with a static profile per batch size, its output checked against
PyTorch (max abs error < 1e-4), and its marginal cost per batch measured on top of the classifier
pipeline: engine (16x16 pooling of the shared normalized 128px batch, encoder, head, batch
aggregation), cost-adjusted argmax, one-ID readback. Energy = NVML total-energy counter over windows
of at least --min-seconds on the NVML device matched to torch's GPU by UUID.

Engines:
  set       pooled (mean, max) embeddings -> head -> 1 x 8 predicted batch accuracies
  ordinal   per-image cumulative P(cheap enough) -> batch mean, 1 x 8
  disagree  per-image predicted disagreement -> batch mean -> 8 per-classifier logistic maps, 1 x 8
Cost depends on the architecture, not on the training seed, so seed 42's copy is timed.

Writes router_costs_new.json and rewrites the score files of these families with the measured A40
cost (replacing the direct-tiny16 estimate). Never reads the test set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from hpc.a40_crossfit import CF_ROUTERS, COST_ESTIMATED, NETS, SEEDS, require, router_model, sha256
from hpc.a40_engines import match_device, write

BATCHES = (8, 16, 32, 64, 128)


def run(args):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision.datasets import CIFAR100
    import tensorrt as trt
    import pynvml
    from hpc.scratch_recipes import make_plan

    require(os.environ.get("SLURM_JOB_ID"), "Run on a Slurm allocation, not a login node")
    require(torch.cuda.is_available(), "The cost measurement needs the A40")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu = torch.cuda.get_device_name(0)
    require("A40" in gpu or args.allow_other_gpu, f"Expected an A40, got {gpu}")
    stage_a = json.loads((args.stage_a / "engines.json").read_text())
    require(stage_a["device"]["name"] == gpu, "GPU model differs from stage A")
    cf = args.crossfit
    meta = json.loads((cf / "crossfit.json").read_text())
    split = json.loads((args.base / "split_seed42.json").read_text())
    require(meta["split_sha256"] == split["sha256"], "Cross-fit split differs")
    out = cf / "engines_new"
    out.mkdir(exist_ok=True)
    plan = make_plan()
    mu = torch.tensor(plan["normalization_mean"], device="cuda").view(1, 3, 1, 1)
    sd = torch.tensor(plan["normalization_std"], device="cuda").view(1, 3, 1, 1)
    ds = CIFAR100(args.data, train=True, download=False)  # never the test set
    cal = np.asarray(split["indices"]["router_calibration"])
    raw = torch.as_tensor(ds.data[cal[:512]], device="cuda")

    def prepare128(u8):
        y = F.interpolate(u8.permute(0, 3, 1, 2).float(), size=(128, 128), mode="bilinear", align_corners=False)
        return y.round_().clamp_(0, 255).div_(255).sub_(mu).div_(sd).contiguous()

    x128 = torch.cat([prepare128(c) for c in raw.split(128)])

    class Engine(nn.Module):
        def __init__(self, net, kind, coef):
            super().__init__()
            self.net, self.kind = net, kind
            self.register_buffer("a", torch.as_tensor(coef[0], dtype=torch.float32))
            self.register_buffer("b", torch.as_tensor(coef[1], dtype=torch.float32))

        def forward(self, x):
            h = self.net["encoder"](F.adaptive_avg_pool2d(x, (16, 16)))
            if self.kind == "set":
                return self.net["rho"](torch.cat([h.mean(0, keepdim=True), h.amax(0, keepdim=True)], 1)).sigmoid()
            if self.kind == "ordinal":
                return self.net["ord"](h).sigmoid().mean(0, keepdim=True)
            d = self.net["dis"](h).sigmoid().mean(0, keepdim=True)  # 1 x 1
            return torch.sigmoid(self.a * d + self.b)               # 1 x 8

    logger = trt.Logger(trt.Logger.WARNING)
    import inspect
    export_kw = dict(input_names=["input"], output_names=["score"], opset_version=17,
                     dynamic_axes={"input": {0: "batch"}})
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kw["dynamo"] = False
    stream = torch.cuda.Stream()
    pynvml.nvmlInit()
    props = torch.cuda.get_device_properties(0)
    devices = [(i, pynvml.nvmlDeviceGetUUID(h), pynvml.nvmlDeviceGetPciInfo(h).busId)
               for i in range(pynvml.nvmlDeviceGetCount()) for h in [pynvml.nvmlDeviceGetHandleByIndex(i)]]
    nvml_index, method = match_device(devices, getattr(props, "uuid", None))
    handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
    energy = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    rel = torch.linspace(0.1, 1.0, 8, device="cuda")
    pinned_id = torch.empty(1, dtype=torch.int64, pin_memory=True)
    rows, manifest = [], []
    for fam in COST_ESTIMATED:
        _, net_name, kind, _ = CF_ROUTERS[fam]
        task, arch, _ = NETS[net_name]
        ck_path = cf / "deploy" / f"{net_name}-s{SEEDS[0]}.pt"
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        net = router_model(arch, task)
        net.load_state_dict(ck["state"], strict=True)
        coef = (np.linspace(-2, 2, 8), np.linspace(1, 3, 8))  # timing stand-in: 8 multiply-adds
        wrapper = Engine(net, kind, coef).cuda().eval()
        onnx_path, eng_path = out / f"{fam}.onnx", out / f"{fam}.engine"
        with torch.no_grad():
            torch.onnx.export(wrapper, x128[:8], str(onnx_path), **export_kw)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        require(parser.parse(onnx_path.read_bytes()),
                "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 ** 2)
        config.clear_flag(trt.BuilderFlag.TF32)
        for b in BATCHES:
            prof = builder.create_optimization_profile()
            prof.set_shape("input", (b, 3, 128, 128), (b, 3, 128, 128), (b, 3, 128, 128))
            config.add_optimization_profile(prof)
        blob = builder.build_serialized_network(network, config)
        require(blob is not None, f"Engine build failed: {fam}")
        eng_path.write_bytes(bytes(blob))
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(eng_path.read_bytes())
        outbuf = torch.empty((1, 8), device="cuda")
        alloc = getattr(trt, "ExecutionContextAllocationStrategy", None)
        alloc = alloc if alloc is not None and hasattr(alloc, "ON_PROFILE_CHANGE") else None
        ctx = {}
        for k, b in enumerate(BATCHES):
            c = engine.create_execution_context(alloc.ON_PROFILE_CHANGE) if alloc else engine.create_execution_context()
            require(c.set_optimization_profile_async(k, stream.cuda_stream), "Profile select failed")
            require(c.set_input_shape("input", (b, 3, 128, 128)), "Input shape rejected")
            c.set_tensor_address("score", outbuf.data_ptr())
            ctx[b] = c
        stream.synchronize()

        def run_engine(x):
            c = ctx[x.shape[0]]
            c.set_tensor_address("input", x.data_ptr())
            require(c.execute_async_v3(torch.cuda.current_stream().cuda_stream), "Engine execution failed")
            return outbuf

        worst = 0.0
        rng = np.random.default_rng(0)
        with torch.inference_mode(), torch.cuda.stream(stream):
            for b in BATCHES:
                for _ in range(5):
                    idx = torch.as_tensor(rng.choice(len(x128), b, replace=False), device="cuda")
                    x = x128[idx].contiguous()
                    got = run_engine(x).clone()
                    stream.synchronize()
                    worst = max(worst, float((got - wrapper(x)).abs().max()))
        require(worst < 1e-4, f"Engine parity failed for {fam}: {worst}")
        for b in BATCHES:
            x_ready = x128[:b].contiguous()

            def step():
                s = run_engine(x_ready)
                pinned_id.copy_((s[0] - 0.5 * rel).argmax().view(1), non_blocking=True)
                stream.synchronize()

            with torch.inference_mode(), torch.cuda.stream(stream):
                t_end = time.perf_counter() + 1.0
                while time.perf_counter() < t_end:
                    step()
                reps = []
                for _ in range(args.repeats):
                    times = []
                    e0, start = energy(), time.perf_counter()
                    while len(times) < 200 or time.perf_counter() - start < args.min_seconds:
                        t0 = time.perf_counter()
                        step()
                        times.append(time.perf_counter() - t0)
                    reps.append((1000 * float(np.median(times)), (energy() - e0) / len(times)))
            row = dict(router=fam, kind=kind, batch=b, latency_ms=float(np.median([r[0] for r in reps])),
                       energy_mj=float(np.median([r[1] for r in reps])),
                       latency_ms_range=[min(r[0] for r in reps), max(r[0] for r in reps)])
            rows.append(row)
            print(json.dumps(dict(event="router_cost", **row)), flush=True)
        manifest.append(dict(router=fam, kind=kind, checkpoint=str(ck_path), checkpoint_sha256=sha256(ck_path),
                             engine=str(eng_path), engine_sha256=sha256(eng_path), parity_max_abs_error=worst))
        del ctx, engine, runtime, builder, network, parser, config, blob
    cost_of = {(r["router"], r["batch"]): r for r in rows}
    for fam in COST_ESTIMATED:
        for seed in SEEDS:
            path = cf / "scores" / f"{fam}-s{seed}.npz"
            data = dict(np.load(path, allow_pickle=False))
            data.update(router_cost_batches=np.array(BATCHES),
                        router_latency_ms=np.array([cost_of[(fam, b)]["latency_ms"] for b in BATCHES]),
                        router_energy_mj=np.array([cost_of[(fam, b)]["energy_mj"] for b in BATCHES]))
            np.savez_compressed(path, **data)
    write(cf / "router_costs_new.json", dict(gpu=gpu, nvml_match=method, job=os.environ.get("SLURM_JOB_ID"),
                                             rows=rows, routers=manifest,
                                             label="A40 TensorRT FP32 cost of the new routers; test not opened"))
    meta_path = cf / "crossfit.json"
    meta["new_router_costs"] = str(cf / "router_costs_new.json")
    for r in meta["routers"]:
        if r["router"] in COST_ESTIMATED:
            r.update(cost_estimated=False, cost_from="A40 measured (hpc.a40_newrouter_costs)")
    write(meta_path, meta)
    print("New router costs measured:", ", ".join(COST_ESTIMATED), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crossfit", type=Path, required=True)
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--min-seconds", type=float, default=4.0)
    p.add_argument("--allow-other-gpu", action="store_true")
    run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
