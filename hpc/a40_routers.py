"""Job 3, stage B1: router scores on router-calibration and router costs on the A40.

GPU (Slurm). Never opens the test set. Reuses the frozen routers from the development screen
(router_arch_screen_v1: results/<task>-<arch>-s<seed>.pt, trained on router-train 3k against
the bank's PyTorch autocast correctness in evidence/) and trains one new variant, the direct
TinyCNN 16x16 on val+3k (classifier-validation 5k + router-train 3k), with the screen's recipe
(Adam 1e-3, 20 epochs, same mini-batch size, seeds 42/45/48).

Routers (each for seeds 42, 45, 48):
  direct-tiny16-3k        correctness head, TinyCNN 16x16 (screened)
  direct-tiny16-val3k     same architecture trained on val+3k (new)
  direct-wide32-3k        correctness head, WideCNN 32x32 (screened; kept only if it beats TinyCNN)
  blended-tiny16-t3k      shared TinyCNN 16x16, learned mix of correctness head and class
                          probabilities x 3k table (as trained)
  blended-tiny16-tval3k   same weights and mix with the val+3k table swapped in (ablation)
  lookup-res32-t3k / -tval3k / -t48k   Residual32 class probabilities x per-class table
Tables are 8x100 per-class accuracies from the screen's evidence (autocast predictions), on
router-train (3k), val+3k, or classifier-train+val+3k (48k, control). Never the calibration set.

Deployment preprocessing (same as stage A's timed pipeline): 16x16 routers average-pool the
GPU-resized normalized 128px batch the classifiers consume; 32x32 routers normalize native 32px
pixels. Each router is exported as one TensorRT FP32 engine (TF32 off) that outputs the batch
mean of P(correct) over the eight classifiers (1x8); the policy then subtracts lam x relative
cost and reads back one model ID. Parity of engine vs PyTorch batch means is checked.

Outputs: scores/<router>-s<seed>.npz (per-image calibration P(correct) for hpc.a40_policy, plus
the router's measured A40 cost per batch), engines/, routers.json, router_costs.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from hpc.a40_engines import match_device, write

BATCHES = (8, 16, 32, 64, 128)
SEEDS = (42, 45, 48)
SCREEN_RECIPE = dict(lr=1e-3, epochs=20, splits_per_3k=48)
ROUTERS = {
    # name: (checkpoint task-arch, kind, table)
    "direct-tiny16-3k": ("correctness-tiny16", "direct", None),
    "direct-tiny16-val3k": ("NEW:correctness-tiny16-val3k", "direct", None),
    "direct-wide32-3k": ("correctness-wide32", "direct", None),
    "blended-tiny16-t3k": ("shared-tiny16", "blended", "3k"),
    "blended-tiny16-tval3k": ("shared-tiny16", "blended", "val3k"),
    "lookup-res32-t3k": ("class-res32", "lookup", "3k"),
    "lookup-res32-tval3k": ("class-res32", "lookup", "val3k"),
    "lookup-res32-t48k": ("class-res32", "lookup", "48k"),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    import os
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision.datasets import CIFAR100
    import tensorrt as trt
    import pynvml

    from hpc.clean_protocol import validate_split
    from hpc.scratch_recipes import make_plan

    require(os.environ.get("SLURM_JOB_ID"), "Run on a Slurm allocation, not a login node")
    require(torch.cuda.is_available(), "CUDA allocation required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for d in ("scores", "engines", "trained"):
        (out / d).mkdir()
    screen, evidence, stage_a = args.screen, args.screen / "evidence", args.stage_a

    # ---- router architectures, copied from the screen's code/hpc_router_arch_sweep.py
    class Residual(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.path = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                                      nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c))

        def forward(self, x):
            return (x + self.path(x)).relu()

    def model(arch, task):
        if arch.startswith("tiny") or arch == "wide32":
            a, b, c = (32, 48, 64) if arch == "wide32" else (16, 24, 32)
            trunk = [nn.Conv2d(3, a, 3, padding=1), nn.ReLU()]
            for m, n in [(a, b), (b, c)]:
                trunk += [nn.Conv2d(m, m, 3, 2, 1, groups=m), nn.Conv2d(m, n, 1), nn.ReLU()]
        else:
            c = 48
            trunk = [nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), Residual(24),
                     nn.Conv2d(24, c, 3, 2, 1), nn.BatchNorm2d(c), nn.ReLU(), Residual(c),
                     nn.AvgPool2d(2), Residual(c)]
        return nn.ModuleDict(dict(encoder=nn.Sequential(*trunk, nn.AdaptiveAvgPool2d(1), nn.Flatten()),
                                  **({"classes": nn.Linear(c, 100)} if task != "correctness" else {}),
                                  **({"correct": nn.Linear(c, 8)} if task != "class" else {})))

    # ---- provenance: split, bank, stage A, screen code
    stage_a_engines = json.loads((stage_a / "engines.json").read_text())
    names = [m["model"] for m in stage_a_engines["engines"]]
    require(len(names) == 8, "Stage A must hold all eight engines")
    split = json.loads((args.base / "split_seed42.json").read_text())
    ev_split = json.loads((evidence / "split.json").read_text())
    require(ev_split["sha256"] == split["sha256"]
            == json.loads((evidence / "development_complete.json").read_text())["split"],
            "Screen evidence split differs from the training split")
    ds = CIFAR100(args.data, train=True, download=False)  # never the test set
    validate_split(split, ds.targets)
    labels = np.load(evidence / "labels.npy", allow_pickle=False)
    require(np.array_equal(labels, ds.targets), "Evidence labels differ from CIFAR-100 train")
    bank = json.loads((evidence / "bank.json").read_text())
    bank_names = [m["architecture"] for m in bank]
    require(sorted(bank_names) == sorted(names), "Screen bank and stage A models differ")
    # Same weights in the screen bank and in stage A's engines.
    receipts = {r["architecture"]: r for r in stage_a_engines["receipts"]}
    for m in bank:
        bank_file = screen / "bank" / m["file"]
        require(sha256(bank_file) == m["sha256"], f"Bank hash mismatch: {bank_file}")
        sd_bank = torch.load(bank_file, map_location="cpu", weights_only=True)
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ck = torch.load(receipts[m["architecture"]]["checkpoint"], map_location="cpu", weights_only=True)
        require(sha256(receipts[m["architecture"]]["checkpoint"]) == receipts[m["architecture"]]["checkpoint_sha256"],
                "Stage A checkpoint changed since stage A")
        same = sd_bank.keys() == ck["best_state"].keys() and all(
            torch.equal(sd_bank[k], ck["best_state"][k]) for k in sd_bank)
        require(same, f"Screen bank weights differ from stage A checkpoint: {m['architecture']}")
        del sd_bank, ck
    code = screen / "code" / "hpc_router_arch_sweep.py"
    code_sha = sha256(code)

    ids = split["indices"]
    tr, cal = np.array(ids["router_train"]), np.array(ids["router_calibration"])
    val, ctrain = np.array(ids["classifier_validation"]), np.array(ids["classifier_train"])
    preds = [np.load(evidence / f"{a}.npy", allow_pickle=False) for a in bank_names]
    correct = np.stack([p == labels for p in preds], axis=-1).astype(np.float32)  # n x 8, bank order
    perm = [bank_names.index(n) for n in names]  # bank order -> stage A order
    correct = correct[:, perm]

    def table(ix):
        return np.stack([correct[ix[labels[ix] == c]].mean(0) for c in range(100)])  # 100 x 8

    tables = {"3k": table(tr), "val3k": table(np.concatenate([val, tr])),
              "48k": table(np.concatenate([ctrain, val, tr]))}
    saved_sha = sha256(screen / "classifier_lookup_val5k_router3k.json")

    plan = make_plan()
    from hpc.full_factorial_protocol import checked_plan
    scratch = checked_plan(args.base / "scratch32_full_factorial_v1" / "plan.json")
    require(all(np.allclose(plan[k], scratch[k]) for k in ("normalization_mean", "normalization_std")),
            "Router normalization differs from the classifiers' (stage A) normalization")
    mu = torch.tensor(plan["normalization_mean"], device="cuda").view(1, 3, 1, 1)
    sd = torch.tensor(plan["normalization_std"], device="cuda").view(1, 3, 1, 1)

    # ---- the new direct val+3k TinyCNN 16x16, screen recipe
    ev_images = np.load(evidence / "images.npy", mmap_mode="r", allow_pickle=False)

    def screen_inputs_16(ix):
        outp = []
        for idx in np.array_split(ix, max(1, len(ix) // 75)):
            x = torch.as_tensor(np.asarray(ev_images[idx]), device="cuda").permute(0, 3, 1, 2).float()
            outp.append(F.adaptive_avg_pool2d(x.div_(255).sub_(mu).div_(sd), (16, 16)))
        return torch.cat(outp)

    fit = np.concatenate([val, tr])
    xfit = screen_inputs_16(fit)
    tfit = torch.as_tensor(correct[fit][:, np.argsort(perm)], device="cuda")  # heads use bank order
    splits = int(round(SCREEN_RECIPE["splits_per_3k"] * len(fit) / len(tr)))
    trained = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        net = model("tiny16", "correctness").cuda()
        opt = torch.optim.Adam(net.parameters(), lr=SCREEN_RECIPE["lr"])
        for _ in range(SCREEN_RECIPE["epochs"]):
            for idx in np.array_split(rng.permutation(len(fit)), splits):
                q = net["correct"](net["encoder"](xfit[idx])).sigmoid()
                loss = F.binary_cross_entropy(q, tfit[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        path = out / "trained" / f"correctness-tiny16-val3k-s{seed}.pt"
        torch.save(dict(state=net.state_dict(), mix=None,
                        meta=dict(task="correctness", architecture="tiny16", seed=seed, fit="val+3k",
                                  images=len(fit), minibatches_per_epoch=splits, **SCREEN_RECIPE,
                                  split_sha256=split["sha256"])), path)
        trained[seed] = path
    del xfit, tfit

    # ---- deployment preprocessing (identical to stage A's gpu_prepare)
    def prepare128(raw_u8):
        y = F.interpolate(raw_u8.permute(0, 3, 1, 2).float(), size=(128, 128), mode="bilinear",
                          align_corners=False)
        return y.round_().clamp_(0, 255).div_(255).sub_(mu).div_(sd).contiguous()

    perm_index = torch.as_tensor(perm, device="cuda")  # stage A column i <- bank column perm[i]

    class Score(nn.Module):
        """Per-image P(correct) for the eight classifiers, in stage A order. Heads were trained
        with columns in bank order, so they are re-indexed; tables are already in stage A order
        (a column permutation commutes with the blend)."""
        def __init__(self, net, kind, arch, tab=None, mix=None):
            super().__init__()
            self.net, self.kind, self.pool16 = net, kind, arch == "tiny16"
            self.register_buffer("mu", mu.clone())
            self.register_buffer("sd", sd.clone())
            self.register_buffer("perm", perm_index.clone())
            if tab is not None:
                self.register_buffer("table", torch.as_tensor(tab, dtype=torch.float32, device="cuda"))
            if mix is not None:
                self.register_buffer("mix", mix.softmax(0).to("cuda"))

        def forward(self, x):
            # 16x16: x is the normalized GPU-resized 128px batch; 32x32: float 0..255 NCHW 32px.
            x = F.adaptive_avg_pool2d(x, (16, 16)) if self.pool16 else (x / 255 - self.mu) / self.sd
            h = self.net["encoder"](x)
            if self.kind == "lookup":
                return self.net["classes"](h).softmax(-1) @ self.table
            q = self.net["correct"](h).sigmoid().index_select(1, self.perm)
            if self.kind == "blended":
                q = self.mix[0] * q + self.mix[1] * (self.net["classes"](h).softmax(-1) @ self.table)
            return q

    class BatchMean(nn.Module):
        def __init__(self, score):
            super().__init__()
            self.score = score

        def forward(self, x):
            return self.score(x).mean(0, keepdim=True)

    # ---- build all routers
    raw_cal = torch.as_tensor(ds.data[cal], device="cuda")  # (2000,32,32,3) uint8
    x128 = torch.cat([prepare128(c) for c in raw_cal.split(128)])
    x32 = raw_cal.permute(0, 3, 1, 2).float().contiguous()
    logger = trt.Logger(trt.Logger.WARNING)
    import inspect
    export_kw = dict(input_names=["input"], output_names=["score"], opset_version=17,
                     dynamic_axes={"input": {0: "batch"}})
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kw["dynamo"] = False
    manifest, scores = [], {}
    for rname, (src, kind, tab_key) in ROUTERS.items():
        task_arch = src.replace("NEW:", "")
        task, arch = task_arch.split("-")[0], task_arch.split("-")[1]
        for seed in SEEDS:
            if src.startswith("NEW:"):
                path = trained[seed]
                meta_json = None
            else:
                path = screen / "results" / f"{task_arch}-s{seed}.pt"
                meta_json = json.loads(path.with_suffix(".json").read_text())
            ck = torch.load(path, map_location="cpu", weights_only=True)
            if meta_json is not None:
                require(ck["meta"] == meta_json, f"Checkpoint meta differs from its JSON: {path}")
                require(meta_json["split_sha256"] == split["sha256"], f"Split mismatch: {path}")
                require(meta_json["code_sha256"] == code_sha, f"Screen code changed since {path}")
            net = model(arch, task)
            net.load_state_dict(ck["state"], strict=True)
            net = net.cuda().eval()
            tab = tables[tab_key] if kind in ("lookup", "blended") else None
            require(kind != "blended" or ck.get("mix") is not None, f"Blended router without mix: {path}")
            score = Score(net, kind, arch, tab, ck.get("mix") if kind == "blended" else None).eval()
            with torch.inference_mode():
                inp = x128 if arch == "tiny16" else x32
                prob = torch.cat([score(c) for c in inp.split(128)]).float().cpu().numpy()
            require(np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1 + 1e-6,
                    f"Router probabilities out of range: {rname}")
            key = f"{rname}-s{seed}"
            scores[key] = prob.clip(0, 1)
            # TensorRT FP32 engine: batch-mean scores, one static profile per batch size
            onnx_path, eng_path = out / "engines" / f"{key}.onnx", out / "engines" / f"{key}.engine"
            shape = (1, 3, 128, 128) if arch == "tiny16" else (1, 3, 32, 32)
            wrapper = BatchMean(score).eval()
            with torch.no_grad():
                torch.onnx.export(wrapper, inp[:8], str(onnx_path), **export_kw)
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
                prof.set_shape("input", (b, *shape[1:]), (b, *shape[1:]), (b, *shape[1:]))
                config.add_optimization_profile(prof)
            blob = builder.build_serialized_network(network, config)
            require(blob is not None, f"Router engine build failed: {key}")
            eng_path.write_bytes(bytes(blob))
            manifest.append(dict(router=rname, seed=seed, kind=kind, arch=arch, table=tab_key,
                                 checkpoint=str(path), checkpoint_sha256=sha256(path),
                                 parameters=sum(p.numel() for p in net.parameters()),
                                 mix=None if ck.get("mix") is None else ck["mix"].softmax(0).tolist(),
                                 engine=str(eng_path), engine_sha256=sha256(eng_path),
                                 input="normalized GPU-resized 128px, avg-pooled to 16x16 in-engine"
                                       if arch == "tiny16" else "native 32px, normalized in-engine"))
            print(json.dumps(dict(event="router_built", **manifest[-1])), flush=True)
            del builder, network, parser, config, blob

    # ---- TensorRT runner, parity, and A40 router cost
    stream = torch.cuda.Stream()

    class RouterEngine:
        def __init__(self, path, shape):
            self.runtime = trt.Runtime(logger)
            self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
            require(self.engine is not None, "Router engine deserialization failed")
            self.out = torch.empty((1, 8), device="cuda")
            alloc = getattr(trt, "ExecutionContextAllocationStrategy", None)
            alloc = alloc if alloc is not None and hasattr(alloc, "ON_PROFILE_CHANGE") else None
            self.ctx = {}
            for k, b in enumerate(BATCHES):
                ctx = (self.engine.create_execution_context(alloc.ON_PROFILE_CHANGE) if alloc
                       else self.engine.create_execution_context())
                require(ctx.set_optimization_profile_async(k, stream.cuda_stream), "Profile select failed")
                require(ctx.set_input_shape("input", (b, *shape[1:])), "Input shape rejected")
                ctx.set_tensor_address("score", self.out.data_ptr())
                self.ctx[b] = ctx
            stream.synchronize()

        def __call__(self, x):
            ctx = self.ctx[x.shape[0]]
            ctx.set_tensor_address("input", x.data_ptr())
            require(ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream), "Router execution failed")
            return self.out

    pynvml.nvmlInit()
    props = torch.cuda.get_device_properties(0)
    devices = [(i, pynvml.nvmlDeviceGetUUID(h), pynvml.nvmlDeviceGetPciInfo(h).busId)
               for i in range(pynvml.nvmlDeviceGetCount()) for h in [pynvml.nvmlDeviceGetHandleByIndex(i)]]
    nvml_index, method = match_device(devices, getattr(props, "uuid", None))
    handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
    energy = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    gpu = torch.cuda.get_device_name(0)
    require("A40" in gpu or args.allow_other_gpu, f"Expected an A40, got {gpu}")
    stage_a_gpu = stage_a_engines["device"]
    require(stage_a_gpu["name"] == gpu, "GPU model differs from stage A")

    rng = np.random.default_rng(0)
    parity, costs = [], []
    rel = torch.linspace(0.1, 1.0, 8, device="cuda")  # stand-in relative costs for the decision step
    pinned_id = torch.empty(1, dtype=torch.int64, pin_memory=True)
    for m in manifest:
        key = f"{m['router']}-s{m['seed']}"
        shape = (1, 3, 128, 128) if m["arch"] == "tiny16" else (1, 3, 32, 32)
        eng = RouterEngine(m["engine"], shape)
        inp = x128 if m["arch"] == "tiny16" else x32
        worst = 0.0
        with torch.inference_mode(), torch.cuda.stream(stream):
            for b in BATCHES:
                for _ in range(10):
                    idx = torch.as_tensor(rng.choice(len(cal), b, replace=False), device="cuda")
                    x = inp[idx].contiguous()
                    got = eng(x).clone()
                    stream.synchronize()
                    want = torch.as_tensor(scores[key], device="cuda")[idx].mean(0, keepdim=True)
                    worst = max(worst, float((got - want).abs().max()))
        require(worst < 1e-4, f"Router engine parity failed for {key}: {worst}")
        parity.append(dict(router=m["router"], seed=m["seed"], max_abs_error=worst))
        m["parity_max_abs_error"] = worst
        if m["seed"] != SEEDS[0]:
            continue  # cost depends on the architecture and table size, not the seed
        raw_dev = raw_cal[:128].contiguous()
        for b in BATCHES:
            x_ready = x128[:b].contiguous()

            def step(eng=eng, x_ready=x_ready, arch=m["arch"], b=b):
                # Router's marginal cost on top of the classifier pipeline: (32px) permute/float,
                # engine (pooling/normalization, network, batch mean), cost-adjusted argmax,
                # one-ID readback. The 128px resize is shared with the classifier and costed there.
                x = x_ready if arch == "tiny16" else raw_dev[:b].permute(0, 3, 1, 2).float().contiguous()
                s = eng(x)
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
            row = dict(router=m["router"], arch=m["arch"], kind=m["kind"], batch=b,
                       latency_ms=float(np.median([r[0] for r in reps])),
                       energy_mj=float(np.median([r[1] for r in reps])),
                       latency_ms_range=[min(r[0] for r in reps), max(r[0] for r in reps)])
            costs.append(row)
            print(json.dumps(dict(event="router_cost", **row)), flush=True)
        del eng
    cost_of = {(c["router"], c["batch"]): c for c in costs}
    for key, prob in scores.items():
        rname, seed = key.rsplit("-s", 1)
        np.savez_compressed(out / "scores" / f"{key}.npz", index=cal, models=np.array(names),
                            correct_prob=prob, router_cost_batches=np.array(BATCHES),
                            router_latency_ms=np.array([cost_of[(rname, b)]["latency_ms"] for b in BATCHES]),
                            router_energy_mj=np.array([cost_of[(rname, b)]["energy_mj"] for b in BATCHES]))
    write(out / "routers.json", dict(
        gpu=gpu, nvml_match=method, job=os.environ.get("SLURM_JOB_ID"), split_sha256=split["sha256"],
        screen_code_sha256=code_sha, routers=manifest, parity=parity,
        tables={k: v.tolist() for k, v in tables.items()}, table_columns=names,
        saved_val3k_table_sha256=saved_sha,
        label="CALIBRATION scores and A40 router costs (stage B1); test not opened", test_evaluated=False))
    write(out / "router_costs.json", dict(gpu=gpu, rows=costs))
    print("Stage B1 complete:", len(scores), "score files", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--min-seconds", type=float, default=2.0)
    p.add_argument("--allow-other-gpu", action="store_true")
    run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
