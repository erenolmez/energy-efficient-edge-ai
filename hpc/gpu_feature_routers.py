"""GPU (A40) versions of the hand-made-feature and Taylor et al. routers: parity, cost, routed report.

Development screening and policy locking on router-calibration (2k); the test set is never read.

1. Refit on CPU with the exact recipes of hpc.feature_router_ablation / hpc.taylor_premodel, on the
   CPU features those runs saved (same fit sets, same correctness labels):
     texture_xgb_cond, compact_xgb_cond   val3k and 3k, seeds 42/45/48
     compact_logistic, compact_knn        val3k and 3k
     taylor-knn8-paper7, -sel-s42/45/48   val3k
   (HistGradientBoosting is not ported; compact_xgb_cond is the same model family.)
2. Port each to the GPU: features with hpc.gpu_features (torch), logistic as a matmul, k-NN as
   cdist + top-k, XGBoost as a vectorised walk of the booster's own JSON trees.
3. Parity: GPU vs CPU features on all fit + calibration images, and GPU vs CPU probabilities on
   calibration; the report says where they differ.
4. Cost on the A40, the same way stage A timed the classifiers: one batch in flight, stream
   synchronize per iteration, NVML total-energy counter of the allocated GPU (matched by UUID)
   over windows of >= --min-seconds, whole-GPU energy including idle draw. The router starts from
   the uint8 32px batch already on the GPU (the classifier pipeline's H2D is in its own cost) and
   ends with the 8 batch-mean P(correct) values read back to pinned host memory.
5. Writes a40_policy-format score files (GPU probabilities + measured A40 router latency/energy)
   and runs hpc.routed_report on them (plus --extra-scores, e.g. the TinyCNN routers).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from hpc.a40_policy import BATCHES, load_stage_a
from hpc.feature_router_ablation import XGB_RECIPE, XGB_SIZE, load_evidence, require

SEEDS = (42, 45, 48)


# ------------------------------------------------------------------ CPU fits (same recipes)
def conditional_np(x, m=8):
    return np.hstack([np.repeat(x, m, axis=0), np.tile(np.eye(m, dtype=np.float32), (len(x), 1))])


def fit_xgb(x, y, feat, seed, threads):
    from xgboost import XGBClassifier
    depth, trees = XGB_SIZE[feat]
    model = XGBClassifier(n_estimators=trees, max_depth=depth, objective="binary:logistic",
                          tree_method="hist", n_jobs=threads, random_state=seed, **XGB_RECIPE)
    model.fit(conditional_np(x), y.ravel())
    return model


def fit_logistic(x, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return [make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=4000)).fit(x, c) for c in y.T]


def fit_knn(x, y, threads):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=100, weights="distance",
                                                                n_jobs=threads)).fit(x, y)


# ------------------------------------------------------------------ GPU predictors
class TreesGPU:
    """XGBoost binary:logistic booster evaluated from its JSON model (exact float32 splits)."""

    def __init__(self, model, device, check_x):
        import torch
        raw = json.loads(model.get_booster().save_raw(raw_format="json"))
        trees = raw["learner"]["gradient_booster"]["model"]["trees"]
        n = max(len(t["left_children"]) for t in trees)
        T = len(trees)
        feat = np.zeros((T, n), np.int64)
        thr = np.zeros((T, n), np.float32)
        left = np.zeros((T, n), np.int64)
        right = np.zeros((T, n), np.int64)
        leaf = np.zeros((T, n), np.float32)
        isleaf = np.ones((T, n), bool)
        depth = 0
        for i, t in enumerate(trees):
            lc, rc = np.array(t["left_children"]), np.array(t["right_children"])
            k = len(lc)
            feat[i, :k], thr[i, :k] = t["split_indices"], t["split_conditions"]
            left[i, :k], right[i, :k] = np.where(lc < 0, 0, lc), np.where(rc < 0, 0, rc)
            isleaf[i, :k] = lc < 0
            leaf[i, :k] = np.where(lc < 0, t["split_conditions"], 0)
            level, frontier = 0, [0]
            while frontier:
                frontier = [c for nd in frontier for c in (lc[nd], rc[nd]) if c >= 0]
                level += bool(frontier)
            depth = max(depth, level)
        self.depth = depth
        to = lambda a: torch.as_tensor(a, device=device)
        self.feat, self.thr, self.left, self.right, self.leaf, self.isleaf = map(to, (feat, thr, left, right, leaf, isleaf))
        self.T, self.n = T, n
        self.eye = torch.eye(8, device=device)
        # intercept = booster margin minus the leaf sum (checked constant on real rows)
        z = conditional_np(check_x[:64])
        margin = model.predict(z, output_margin=True)
        mine = self.leafsum(torch.as_tensor(z, device=device)).cpu().numpy()
        off = margin - mine
        require(np.ptp(off) < 1e-4, f"Tree port does not reproduce XGBoost margins (spread {np.ptp(off)})")
        self.offset = float(np.median(off))

    def leafsum(self, z):
        import torch
        R = z.shape[0]
        node = torch.zeros(R, self.T, dtype=torch.int64, device=z.device)
        base = (torch.arange(self.T, device=z.device) * self.n)[None]
        for _ in range(self.depth + 1):
            idx = base + node
            f = self.feat.view(-1)[idx]
            v = torch.gather(z, 1, f)
            go = v < self.thr.view(-1)[idx]
            nxt = torch.where(go, self.left.view(-1)[idx], self.right.view(-1)[idx])
            node = torch.where(self.isleaf.view(-1)[idx], node, nxt)
        return self.leaf.view(-1)[base + node].sum(1)

    def __call__(self, x):
        import torch
        B = x.shape[0]
        z = torch.cat([x.repeat_interleave(8, 0), self.eye.repeat(B, 1)], 1)
        return torch.sigmoid(self.leafsum(z) + self.offset).view(B, 8)


class LogisticGPU:
    def __init__(self, heads, device):
        import torch
        sc = [h[0] for h in heads]
        lr = [h[-1] for h in heads]
        t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
        self.mean, self.scale = t([s.mean_ for s in sc]), t([s.scale_ for s in sc])  # 8 x F
        self.coef, self.b = t([m.coef_[0] for m in lr]), t([m.intercept_[0] for m in lr])

    def __call__(self, x):
        import torch
        z = (x[:, None, :] - self.mean[None]) / self.scale[None]
        return torch.sigmoid((z * self.coef[None]).sum(2) + self.b[None])


class KnnGPU:
    """k-NN vote share/regression on scaled features: 'distance' (sklearn weights) or 'uniform'."""

    def __init__(self, train_x, train_y, lo, span, k, weights, device):
        import torch
        t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
        self.lo, self.span = t(lo), t(span)
        self.train, self.y = (t(train_x) - self.lo) / self.span, t(train_y)
        self.k, self.weights = k, weights

    def __call__(self, x):
        import torch
        z = (x - self.lo) / self.span
        d = torch.cdist(z, self.train)
        dist, idx = d.topk(self.k, dim=1, largest=False)
        y = self.y[idx]  # B, k, 8
        if self.weights == "uniform":
            return y.mean(1)
        zero = dist == 0
        w = torch.where(zero.any(1, keepdim=True), zero.float(), 1.0 / dist.clamp_min(1e-30))
        return (w[..., None] * y).sum(1) / w.sum(1, keepdim=True)


# ------------------------------------------------------------------ run
def features_by_index(path):
    z = np.load(path, allow_pickle=False)
    return {int(k): i for i, k in enumerate(z["index"])}, z["x"]


def run(args):
    import torch
    from hpc import gpu_features as gf

    dev = torch.device(args.device)
    if dev.type == "cpu" and not args.dry_run and not os.environ.get("GPU_ROUTERS_CPU_TEST"):
        raise RuntimeError("Without --dry-run this measures A40 costs: use --device cuda on a Slurm GPU")
    if dev.type == "cuda":
        require(os.environ.get("SLURM_JOB_ID"), "Run the GPU phase on a Slurm allocation")
        require(torch.cuda.is_available(), "CUDA allocation required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    ev = load_evidence(args.base, args.evidence, args.data)
    a = load_stage_a(args.stage_a)
    names = a["names"]
    require(sorted(names) == sorted(ev["names"]), "Stage A and evidence models differ")
    correct = ev["correct"][:, [ev["names"].index(n) for n in names]]  # stage A order
    images, cal, fits = ev["images"], ev["cal"], ev["fits"]
    require(np.array_equal(np.sort(a["index"]), np.sort(cal)), "Stage A calibration images differ")

    fa, ta = args.feature_ablation, args.taylor
    fa_res = json.loads((fa / "results.json").read_text())
    ta_res = json.loads((ta / "results.json").read_text())
    require(fa_res["split_sha256"] == ev["split"]["sha256"] == ta_res["split_sha256"], "Split differs")
    pos, X = {}, {}
    for feat in ("compact", "texture"):
        pos[feat], X[feat] = features_by_index(fa / f"features_{feat}.npz")
    pos["taylor"], X["taylor"] = features_by_index(ta / "features29.npz")
    rows = lambda feat, ix: X[feat][[pos[feat][int(i)] for i in ix]]
    kp_thr = ta_res["keypoint_threshold"]
    taylor = gf.Taylor29(dev, kp_thr)

    def gpu_feats(xb, which=("compact", "texture", "taylor")):
        out = {}
        if "texture" in which:
            out["compact"], out["texture"] = gf.compact_texture(xb)
        elif "compact" in which:
            out["compact"], _ = gf.compact_texture(xb, texture=False)
        if "taylor" in which:
            out["taylor"] = taylor(xb)
        return out

    from hpc.taylor_premodel import FEATURES
    every = cal[:64] if args.dry_run else np.concatenate([fits["val3k"], cal])
    parity = {}
    with torch.inference_mode():
        got = {k: [] for k in ("compact", "texture", "taylor")}
        for chunk in np.array_split(every, max(1, len(every) // 128)):
            out = gpu_feats(torch.as_tensor(images[chunk], device=dev))
            for k in got:
                got[k].append(out[k].cpu().numpy())
    for k in got:
        g, c = np.concatenate(got[k]), rows(k, every)
        d = np.abs(g - c.astype(np.float32))
        bad = (d > 1e-4) & (d / np.maximum(np.abs(c), 1e-6) > 1e-4)
        label = (lambda j: FEATURES[j]) if k == "taylor" else (lambda j: f"{k}[{j}]")
        parity[k] = dict(images=len(every), max_abs=float(d.max()), images_any_mismatch=int(bad.any(1).sum()),
                         features_any_mismatch={label(int(j)): int(bad[:, j].sum()) for j in np.flatnonzero(bad.any(0))})
        print(f"feature parity {k}: max abs {d.max():.3g}, images with a mismatch "
              f"{parity[k]['images_any_mismatch']}/{len(every)} {parity[k]['features_any_mismatch']}", flush=True)

    # fit (CPU) and port (GPU)
    routers = {}  # name -> (feat, gpu predictor, cpu prob on cal from CPU features, timing key)
    seeds = SEEDS[:1] if args.dry_run else SEEDS
    fit_names = ("val3k",) if args.dry_run else ("val3k", "3k")
    for fit_name in fit_names:
        fit = fits[fit_name]
        y = correct[fit].astype(np.float32)
        for feat in ("texture", "compact"):
            xf, xc = rows(feat, fit), rows(feat, cal)
            for seed in seeds:
                t = time.time()
                m = fit_xgb(xf, y, feat, seed, args.threads)
                cpu = m.predict_proba(conditional_np(xc))[:, 1].reshape(-1, 8)
                routers[f"gpu-{feat}_xgb_cond-{fit_name}-s{seed}"] = (feat, TreesGPU(m, dev, xc), cpu, f"{feat}_xgb")
                print(f"fit {feat}_xgb {fit_name} s{seed} in {time.time() - t:.1f}s", flush=True)
        xf, xc = rows("compact", fit), rows("compact", cal)
        heads = fit_logistic(xf, y)
        cpu = np.column_stack([h.predict_proba(xc)[:, 1] for h in heads])
        routers[f"gpu-compact_logistic-{fit_name}"] = ("compact", LogisticGPU(heads, dev), cpu, "compact_logistic")
        knn = fit_knn(xf, y, args.threads)
        sc = knn[0]
        routers[f"gpu-compact_knn-{fit_name}"] = ("compact", KnnGPU(xf, y, sc.mean_, sc.scale_, 100, "distance", dev),
                                                  knn.predict(xc), f"compact_knn_{fit_name}")
    fit = fits["val3k"]
    y = correct[fit].astype(np.float32)
    from hpc.taylor_premodel import PAPER7, MinMax, neighbours
    sets = {"paper7": PAPER7}
    for s, v in ta_res["feature_sets"].items():
        if s.startswith("sel"):
            sets[s] = v["selected"]
    for s, sel in sets.items():
        cols = [FEATURES.index(f) for f in sel]
        xf, xc = rows("taylor", fit)[:, cols], rows("taylor", cal)[:, cols]
        mm = MinMax(xf)
        cpu = correct[fit][neighbours(mm(xf), mm(xc))].mean(1)
        cols_t = torch.as_tensor(cols, device=dev)
        routers[f"gpu-taylor-knn8-{s}"] = ("taylor", (cols_t, KnnGPU(xf, y, mm.lo, mm.span, 5, "uniform", dev)), cpu, "taylor_knn8")

    def predict(name, xb, feats=None):
        feat, pred, _, _ = routers[name]
        f = (feats if feats is not None else gpu_feats(xb, (feat,)))[feat]  # only this router's features
        if feat == "taylor":
            cols_t, knn = pred
            return knn(f[:, cols_t])
        return pred(f)

    # GPU probabilities on calibration and parity vs the CPU pipeline
    gprob, prob_parity = {}, {}
    with torch.inference_mode():
        feats_cal = [gpu_feats(torch.as_tensor(images[ch], device=dev)) for ch in np.array_split(cal, len(cal) // 125)]
        for name in routers:
            p = torch.cat([predict(name, None, f) for f in feats_cal]).cpu().numpy().astype(np.float64)
            cpu = np.asarray(routers[name][2], dtype=np.float64)
            d = np.abs(p - cpu)
            gprob[name] = p.clip(0, 1)
            prob_parity[name] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                                     images_over_1e3=int((d.max(1) > 1e-3).sum()))
            print(f"prob parity {name}: max abs {d.max():.3g}, images > 1e-3: {prob_parity[name]['images_over_1e3']}", flush=True)

    result = dict(stage="development screening / policy locking on router-calibration; test never read",
                  feature_parity=parity, probability_parity=prob_parity, device=str(dev),
                  routers=sorted(routers), seeds=list(seeds))
    if args.dry_run:
        print(json.dumps(result, indent=1))
        print("DRY RUN OK (nothing written)")
        return

    # ---- A40 cost: NVML energy, same method as stage A
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "scores").mkdir()
    import contextlib
    if dev.type == "cuda":
        import pynvml
        from hpc.a40_engines import match_device
        pynvml.nvmlInit()
        props = torch.cuda.get_device_properties(0)
        torch_uuid = getattr(props, "uuid", None)  # same identity logic as hpc.a40_engines
        torch_bus = None
        if all(hasattr(props, k) for k in ("pci_domain_id", "pci_bus_id", "pci_device_id")):
            torch_bus = f"{props.pci_domain_id:08x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        devices = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            devices.append((i, pynvml.nvmlDeviceGetUUID(h), pynvml.nvmlDeviceGetPciInfo(h).busId))
        nvml_index, method = match_device(devices, torch_uuid, torch_bus)
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
        energy = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        stream = torch.cuda.Stream()
        host = torch.empty(8, pin_memory=True)
    else:  # CPU path only for the synthetic end-to-end test: no NVML, energy reported as 0
        method, energy = "none (cpu test)", (lambda: 0)
        stream = type("S", (), {"synchronize": staticmethod(lambda: None)})()
        host = torch.empty(8)
    pools = {b: [torch.as_tensor(images[cal[i * b:(i + 1) * b]], device=dev) for i in range(max(4, min(16, len(cal) // b)))]
             for b in BATCHES}

    def measure(fn, b):
        pool = pools[b]

        def step(i):
            p = fn(pool[i % len(pool)])
            host.copy_(p.mean(0) if p.shape[1] == 8 else p.reshape(-1)[:8], non_blocking=True)
            stream.synchronize()

        ctx = torch.cuda.stream(stream) if dev.type == "cuda" else contextlib.nullcontext()
        with torch.inference_mode(), ctx:
            t_end, i = time.perf_counter() + args.warmup_seconds, 0
            while i < 10 or time.perf_counter() < t_end:
                step(i)
                i += 1
            times, e0, start = [], energy(), time.perf_counter()
            while len(times) < 20 or time.perf_counter() - start < args.min_seconds:
                t0 = time.perf_counter()
                step(len(times))
                times.append(time.perf_counter() - t0)
            e1 = energy()
        return dict(latency_ms=1000 * float(np.median(times)), energy_mj=(e1 - e0) / len(times), iters=len(times))

    if dev.type == "cuda":
        torch.cuda.synchronize()
    time.sleep(2 if dev.type == "cuda" else 0)
    e0, t0 = energy(), time.perf_counter()
    time.sleep(args.idle_seconds)
    idle_w = (energy() - e0) / 1000 / (time.perf_counter() - t0)
    timed = {}
    reps = {}
    for name, (_, _, _, key) in routers.items():
        reps.setdefault(key, name)  # same architecture and size -> same cost (seed/fit only change weights)
    cells = [(k, lambda xb, n=n: predict(n, xb)) for k, n in reps.items()]
    cells += [(f"features_{f}", lambda xb, f=f: gpu_feats(xb, (f,))[f]) for f in ("compact", "texture", "taylor")]
    rng = np.random.default_rng(0)
    raw = []
    for rep in range(args.repeats):
        order = [(k, fn, b) for k, fn in cells for b in BATCHES]
        rng.shuffle(order)
        for k, fn, b in order:
            row = dict(repeat=rep, cell=k, batch=b, **measure(fn, b))
            raw.append(row)
            print(json.dumps(row), flush=True)
    for k, _ in cells:
        timed[k] = {b: dict(latency_ms=float(np.median([r["latency_ms"] for r in raw if r["cell"] == k and r["batch"] == b])),
                            energy_mj=float(np.median([r["energy_mj"] for r in raw if r["cell"] == k and r["batch"] == b])))
                    for b in BATCHES}
    files = []
    for name, (_, _, _, key) in routers.items():
        path = out / "scores" / f"{name}.npz"
        np.savez_compressed(path, index=cal, models=np.array(names), correct_prob=gprob[name].astype(np.float32),
                            router_cost_batches=np.array(BATCHES),
                            router_latency_ms=np.array([timed[key][b]["latency_ms"] for b in BATCHES]),
                            router_energy_mj=np.array([timed[key][b]["energy_mj"] for b in BATCHES]))
        files.append(str(path))
    result.update(gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu (test)", nvml_match=method, idle_w=idle_w, job=os.environ.get("SLURM_JOB_ID"),
                  cost_scope="uint8 32px batch already on the GPU -> features -> predictor -> 8 batch-mean "
                             "P(correct) to pinned host; whole-GPU NVML energy incl. idle, per batch",
                  timed={k: {str(b): v for b, v in d.items()} for k, d in timed.items()}, raw=raw)
    (out / "results.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    lines = ["# GPU (A40) hand-made-feature and Taylor routers: parity and router cost", "",
             f"GPU {result['gpu']}, idle {idle_w:.1f} W. Router cost per batch, median of {args.repeats} repeats "
             "(whole-GPU energy incl. idle, as for the classifiers).", "",
             "| cell | " + " | ".join(f"B={b} ms / mJ" for b in BATCHES) + " |", "|---|" + "---|" * len(BATCHES)]
    for k in timed:
        lines.append(f"| {k} | " + " | ".join(f"{timed[k][b]['latency_ms']:.2f} / {timed[k][b]['energy_mj']:.0f}"
                                             for b in BATCHES) + " |")
    lines += ["", "Feature parity (GPU float32 vs CPU):", ""]
    for k, v in parity.items():
        lines.append(f"- {k}: max abs {v['max_abs']:.3g}; images with any mismatch (>1e-4 abs and rel) "
                     f"{v['images_any_mismatch']}/{v['images']}; features: {v['features_any_mismatch'] or 'none'}")
    lines += ["", "Probability parity (GPU pipeline vs CPU features + CPU model), calibration:", ""]
    for k, v in prob_parity.items():
        lines.append(f"- {k}: max abs {v['max_abs']:.3g}, images > 1e-3: {v['images_over_1e3']}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    from hpc import routed_report
    extra = [str(p) for p in args.extra_scores if Path(p).exists()]
    routed_report.main(["--stage-a", str(args.stage_a), "--scores", *files, *extra, "--output", str(out / "report")])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--feature-ablation", type=Path, required=True)
    p.add_argument("--taylor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extra-scores", type=Path, nargs="*", default=[])
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--min-seconds", type=float, default=2.0)
    p.add_argument("--warmup-seconds", type=float, default=1.0)
    p.add_argument("--idle-seconds", type=float, default=10.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="CPU-safe: check inputs, ports and parity on 64 images")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
