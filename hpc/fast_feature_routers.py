"""Fast GPU hand-made-feature and Taylor routers, 5-fold cross-fit on the 5k router pool (A40).

For the final router comparison (job 3). Test never read; classifier-validation never used.

Pool and folds: router_train 3k + router_calibration 2k, the same 5 stratified folds as
hpc/a40_crossfit.py (make_folds, fold seed 42: 10 images per class per fold). For each fold a
router is fit on the other 4 folds (4k) and scores the held-out 1k, so every pool image gets one
out-of-fold P(correct) for each of the eight classifiers. Training targets are the screen
evidence correctness (bank PyTorch autocast), as for every other router.

Features are computed on the GPU (hpc.gpu_features) for fitting and for scoring, i.e. exactly what
deploys. Parity against the earlier CPU features is reported (Taylor is now float64 with numpy's
binning rules, so it should match bit for bit except at rare ulp-level edges).

Routers (A40 deployable; recipes as in hpc.feature_router_ablation / hpc.taylor_premodel):
  feat-compact-logistic-4k        46 compact features, one L2 logistic per classifier
  feat-compact-xgb-4k-s{42,45,48} compact, XGBoost on [features, one-hot model] (depth 3, 300)
  feat-texture-xgb-4k-s{...}      450 texture features (adds histograms, HOG, LBP; depth 4, 400)
  feat-compact-knn-4k             compact, standardised, 100-NN distance-weighted
  taylor-knn8-paper7-4k           Taylor et al.'s 7 features, min-max, 5-NN vote share; the
                                  keypoint threshold is the upper quartile of Hessian peak
                                  responses on the 4k training folds (the old run used val+3k)
Speed: the whole router (features + predictor + batch mean) is captured as one CUDA graph per
batch size, which removes the per-kernel launch overhead that dominated the eager version.
Connected components (Taylor) run a fixed number of propagation + pointer-jumping steps so they
can be captured; the count is calibrated on the pool + classifier-validation images and the
fixed-count features are checked equal to the converged ones on every pool image.

Cost: A40, same method as stage A: one batch in flight, stream synchronize per iteration, NVML
total-energy counter of the allocated GPU (UUID match) over windows >= --min-seconds, whole-GPU
energy incl. idle, from the uint8 32px batch already on the GPU to the 8 batch-mean P(correct)
values in pinned host memory. Measured for the CUDA graph (used in the score files) and eager.
Cost does not depend on the training fold, so it is measured with the fold-0 models.

Output: <output>/scores/<router>.npz in the a40_policy format (index = 5k pool, models,
correct_prob, fold, router_cost_batches, router_latency_ms, router_energy_mj), results.json,
summary.md. Lock with: python -m hpc.a40_policy --roles router_train router_calibration ...
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import time

import numpy as np

from hpc.feature_router_ablation import XGB_RECIPE, load_evidence, require
from hpc.gpu_feature_routers import KnnGPU, LogisticGPU, TreesGPU, conditional_np, fit_knn, fit_logistic, fit_xgb

BATCHES = (8, 16, 32, 64, 128)
SEEDS = (42, 45, 48)
FOLDS = 5
POOL_ROLES = ("router_train", "router_calibration")


def make_folds(pool, labels, k=FOLDS, seed=42):
    """Copy of hpc/a40_crossfit.make_folds (branch claude/project-thread-1kcwzo): stratified,
    each class's images shuffled and dealt round-robin."""
    pool = np.asarray(pool)
    lab = np.asarray(labels)[pool]
    rng = np.random.default_rng(seed)
    fold = np.full(len(pool), -1, dtype=np.int8)
    for c in np.unique(lab):
        members = np.flatnonzero(lab == c)
        require(len(members) % k == 0, f"class {c}: {len(members)} pool images do not split into {k} folds")
        fold[rng.permutation(members)] = np.arange(len(members)) % k
    return fold


def stage_a_names(stage_a):
    z = np.load(Path(stage_a) / "predictions.npz", allow_pickle=False)
    return [str(m) for m in z["models"]]


def run(args):
    import torch
    from hpc import gpu_features as gf
    from hpc.taylor_premodel import FEATURES, PAPER7

    dev = torch.device(args.device)
    if dev.type == "cuda":
        require(os.environ.get("SLURM_JOB_ID"), "Run the GPU phase on a Slurm allocation")
        require(torch.cuda.is_available(), "CUDA allocation required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif not args.dry_run and not os.environ.get("GPU_ROUTERS_CPU_TEST"):
        raise RuntimeError("Without --dry-run this measures A40 costs: use --device cuda on a Slurm GPU")
    t_start = time.time()
    ev = load_evidence(args.base, args.evidence, args.data)
    names = stage_a_names(args.stage_a)
    require(sorted(names) == sorted(ev["names"]), "Stage A and evidence models differ")
    correct = ev["correct"][:, [ev["names"].index(n) for n in names]].astype(np.float32)  # stage A order
    ids = {k: np.asarray(v) for k, v in ev["split"]["indices"].items()}
    pool = np.concatenate([ids[r] for r in POOL_ROLES])
    require(not np.intersect1d(pool, ids["classifier_validation"]).size, "Pool overlaps classifier-validation")
    fold = make_folds(pool, ev["labels"], FOLDS, args.fold_seed)
    images = ev["images"]
    layout = dict(pool=len(pool), folds=FOLDS, fold_seed=args.fold_seed,
                  fold_sizes=[int((fold == f).sum()) for f in range(FOLDS)])
    print(json.dumps(dict(event="folds", **layout)), flush=True)

    # ---------------- GPU features on the pool (eager, converged components)
    ta_res = json.loads((args.taylor / "results.json").read_text())
    old_thr = ta_res["keypoint_threshold"]
    taylor_old = gf.Taylor29(dev, old_thr)          # same threshold as the CPU run, for parity
    taylor_noKp = gf.Taylor29(dev, None)
    work = pool[:640] if args.dry_run else pool  # dry run: enough for 100-NN on 4 folds
    chunks = np.array_split(np.arange(len(work)), max(1, len(work) // 128))
    feats = {k: [] for k in ("compact", "texture", "taylor_old", "taylor")}
    peaks, iters_needed = [], 0
    t = time.time()
    with torch.inference_mode():
        for ch in chunks:
            xb = torch.as_tensor(images[work[ch]], device=dev)
            c, tx = gf.compact_texture(xb)
            feats["compact"].append(c.cpu().numpy())
            feats["texture"].append(tx.cpu().numpy())
            f_old, n = taylor_old(xb, return_iters=True)
            iters_needed = max(iters_needed, n)
            feats["taylor_old"].append(f_old.cpu().numpy())
            feats["taylor"].append(taylor_noKp(xb).cpu().numpy())
            peaks.append(taylor_noKp.peak_responses(xb).cpu().numpy())
        # calibrate the fixed component-iteration count on pool + classifier-validation images
        if not args.dry_run:
            for ch in np.array_split(ids["classifier_validation"], 40):
                _, n = taylor_noKp(torch.as_tensor(images[ch], device=dev), return_iters=True)
                iters_needed = max(iters_needed, n)
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    peaks = np.concatenate(peaks)
    cc_iters = iters_needed + 2
    print(f"GPU features for {len(work)} images in {time.time() - t:.1f}s; components converge in "
          f"<= {iters_needed} steps, fixed count {cc_iters}", flush=True)

    # fixed-count components reproduce the converged features exactly
    taylor_fixed = gf.Taylor29(dev, old_thr, cc_iters=cc_iters)
    with torch.inference_mode():
        fixed = np.concatenate([taylor_fixed(torch.as_tensor(images[work[ch]], device=dev)).cpu().numpy() for ch in chunks])
    cc_check = dict(steps_needed=iters_needed, fixed_steps=cc_iters,
                    images_differing=int((np.abs(fixed - feats["taylor_old"]).max(1) > 0).sum()), images=len(work))
    require(cc_check["images_differing"] == 0, f"Fixed-step components differ from converged ones: {cc_check}")

    # parity vs the earlier CPU features (fa: val3k+cal; taylor features29: val3k+cal, val+3k threshold)
    parity = {}
    for key, path, gkey in (("compact", args.feature_ablation / "features_compact.npz", "compact"),
                            ("texture", args.feature_ablation / "features_texture.npz", "texture"),
                            ("taylor", args.taylor / "features29.npz", "taylor_old")):
        z = np.load(path, allow_pickle=False)
        pos = {int(k): i for i, k in enumerate(z["index"])}
        cpu = z["x"][[pos[int(i)] for i in work]].astype(np.float64)
        g = feats[gkey].astype(np.float64)
        d = np.abs(g - cpu)
        bad = (d > 1e-4) & (d / np.maximum(np.abs(cpu), 1e-6) > 1e-4)
        label = (lambda j: FEATURES[j]) if key == "taylor" else (lambda j, k=key: f"{k}[{j}]")
        parity[key] = dict(images=len(work), max_abs=float(d.max()), images_any_mismatch=int(bad.any(1).sum()),
                           features_any_mismatch={label(int(j)): int(bad[:, j].sum()) for j in np.flatnonzero(bad.any(0))})
        print(f"feature parity {key}: max abs {d.max():.3g}, images with a mismatch "
              f"{parity[key]['images_any_mismatch']}/{len(work)} {parity[key]['features_any_mismatch']}", flush=True)

    # ---------------- cross-fit
    y_all = correct[work]
    f_work = fold[: len(work)]
    cols7 = [FEATURES.index(f) for f in PAPER7]
    kp_col = FEATURES.index("n_keypoints")

    def taylor_x(rows, thr):
        x = feats["taylor"][rows][:, cols7].copy()
        x[:, PAPER7.index("n_keypoints")] = (peaks[rows] > thr).sum(1)
        return x

    def kp_threshold(rows):
        pooled = peaks[rows]
        pooled = pooled[np.isfinite(pooled)]
        return float(np.quantile(pooled, 0.75))

    from hpc.taylor_premodel import MinMax
    families = {"feat-compact-logistic-4k": None, "feat-compact-knn-4k": None, "taylor-knn8-paper7-4k": None}
    seeds = SEEDS[:1] if args.dry_run else SEEDS
    for seed in seeds:
        families[f"feat-compact-xgb-4k-s{seed}"] = seed
        families[f"feat-texture-xgb-4k-s{seed}"] = seed
    folds_run = (0,) if args.dry_run else range(FOLDS)
    oof = {f: np.full((len(work), 8), np.nan) for f in families}
    port_check = {}
    deploy = {}  # fold-0 models, for timing
    with torch.inference_mode():
        for k in folds_run:
            tr, te = np.flatnonzero(f_work != k), np.flatnonzero(f_work == k)
            if not len(te):
                continue
            y = y_all[tr]
            xte_c = torch.as_tensor(feats["compact"][te], device=dev)
            xte_t = torch.as_tensor(feats["texture"][te], device=dev)
            heads = fit_logistic(feats["compact"][tr], y)
            lg = LogisticGPU(heads, dev)
            oof["feat-compact-logistic-4k"][te] = lg(xte_c).cpu().numpy()
            knn = fit_knn(feats["compact"][tr], y, args.threads)
            kg = KnnGPU(feats["compact"][tr], y, knn[0].mean_, knn[0].scale_, 100, "distance", dev)
            oof["feat-compact-knn-4k"][te] = kg(xte_c).cpu().numpy()
            thr = kp_threshold(tr)
            xtr7, xte7 = taylor_x(tr, thr), taylor_x(te, thr)
            mm = MinMax(xtr7)
            tg = KnnGPU(xtr7, y, mm.lo, mm.span, 5, "uniform", dev, dtype=np.float64)
            oof["taylor-knn8-paper7-4k"][te] = tg(torch.as_tensor(xte7, device=dev)).cpu().numpy()
            if k == 0:
                port_check["feat-compact-logistic-4k"] = float(np.abs(
                    oof["feat-compact-logistic-4k"][te] - np.column_stack([h.predict_proba(feats["compact"][te])[:, 1] for h in heads])).max())
                deploy.update({"feat-compact-logistic-4k": ("compact", lg), "feat-compact-knn-4k": ("compact", kg),
                               "taylor-knn8-paper7-4k": ("taylor", (tg, thr))})
            for seed in seeds:
                for feat, xte in (("compact", xte_c), ("texture", xte_t)):
                    t = time.time()
                    m = fit_xgb(feats[feat][tr], y, feat, seed, args.threads)
                    tgpu = TreesGPU(m, dev, feats[feat][te])
                    name = f"feat-{feat}-xgb-4k-s{seed}"
                    oof[name][te] = tgpu(xte).cpu().numpy()
                    if k == 0:
                        cpu_p = m.predict_proba(conditional_np(feats[feat][te]))[:, 1].reshape(-1, 8)
                        port_check[name] = float(np.abs(oof[name][te] - cpu_p).max())
                        deploy[name] = (feat, tgpu)
                    print(f"fold {k} {name}: fit + port {time.time() - t:.1f}s", flush=True)
            print(f"fold {k} done", flush=True)
    print("GPU-vs-CPU predictor check (fold 0, max abs P diff):", json.dumps(port_check), flush=True)

    result = dict(stage="cross-fit out-of-fold router scores on the 5k router pool (router-train + "
                        "router-calibration); classifier-validation not used; test not read",
                  layout=layout, models=names, feature_parity=parity, components=cc_check,
                  predictor_port_check=port_check, xgb_recipe=XGB_RECIPE, device=str(dev))
    if args.dry_run:
        print(json.dumps(result, indent=1))
        print("DRY RUN OK (nothing written)")
        return
    for f, p in oof.items():
        require(np.isfinite(p).all(), f"Missing out-of-fold scores for {f}")

    # ---------------- A40 cost: CUDA graph (deployed) and eager
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "scores").mkdir()
    taylor_dep = {}

    def router_fn(name):
        feat, pred = deploy[name]
        if feat == "taylor":
            knn, thr = pred
            tay = taylor_dep.setdefault(thr, gf.Taylor29(dev, thr, cc_iters=cc_iters))
            cols_t = torch.as_tensor(cols7, device=dev)
            return lambda xb: knn(tay(xb)[:, cols_t]).float()
        tex = feat == "texture"
        return lambda xb: pred(gf.compact_texture(xb, texture=tex)[1 if tex else 0])

    cells = {n: router_fn(n) for n in ("feat-compact-logistic-4k", "feat-compact-knn-4k", "taylor-knn8-paper7-4k",
                                       "feat-compact-xgb-4k-s42", "feat-texture-xgb-4k-s42")}
    tay_cost = gf.Taylor29(dev, deploy["taylor-knn8-paper7-4k"][1][1], cc_iters=cc_iters)
    cells["features-compact"] = lambda xb: gf.compact_texture(xb, texture=False)[0]
    cells["features-texture"] = lambda xb: gf.compact_texture(xb)[1]
    cells["features-taylor29"] = lambda xb: tay_cost(xb)
    # every 4k-fold model of a family has the same structure/size; seeds 45/48 reuse s42's cost
    cost_key = {f: ("feat-compact-xgb-4k-s42" if "compact-xgb" in f else "feat-texture-xgb-4k-s42" if "texture-xgb" in f else f)
                for f in families}

    if dev.type == "cuda":
        import pynvml
        from hpc.a40_engines import match_device
        pynvml.nvmlInit()
        props = torch.cuda.get_device_properties(0)
        torch_uuid = getattr(props, "uuid", None)
        torch_bus = None
        if all(hasattr(props, a) for a in ("pci_domain_id", "pci_bus_id", "pci_device_id")):
            torch_bus = f"{props.pci_domain_id:08x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        devices = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            devices.append((i, pynvml.nvmlDeviceGetUUID(h), pynvml.nvmlDeviceGetPciInfo(h).busId))
        nvml_index, method = match_device(devices, torch_uuid, torch_bus)
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
        energy = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        stream = torch.cuda.Stream()
        host = torch.empty(64, pin_memory=True)
        ctx = lambda: torch.cuda.stream(stream)
        sync = stream.synchronize
    else:  # synthetic CPU test only
        method, energy, host = "none (cpu test)", (lambda: 0), torch.empty(64)
        ctx, sync = contextlib.nullcontext, (lambda: None)

    cal_pool = pool[np.isin(pool, ids["router_calibration"])]
    batches_in = {b: [torch.as_tensor(images[cal_pool[i * b:(i + 1) * b]], device=dev)
                      for i in range(max(4, min(16, len(cal_pool) // b)))] for b in BATCHES}

    def reduce(p):
        return p.mean(0) if p.shape[1] == 8 else p.reshape(-1)[:8]

    def make_step(fn, b, mode):
        pool_b = batches_in[b]
        if mode == "eager":
            def step(i):
                host[:8].copy_(reduce(fn(pool_b[i % len(pool_b)])).float(), non_blocking=True)
                sync()
            return step, None
        static = torch.empty_like(pool_b[0])
        static.copy_(pool_b[0])
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn(static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            res = reduce(fn(static)).float()

        def step(i):
            static.copy_(pool_b[i % len(pool_b)], non_blocking=True)
            graph.replay()
            host[:8].copy_(res, non_blocking=True)
            sync()
        return step, res

    def measure(step):
        with torch.inference_mode(), ctx():
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
        time.sleep(2)
    e0, t0 = energy(), time.perf_counter()
    time.sleep(args.idle_seconds)
    idle_w = (energy() - e0) / 1000 / max(time.perf_counter() - t0, 1e-9)
    modes = ("graph", "eager") if dev.type == "cuda" else ("eager",)
    steps, graph_errors, graph_check = {}, {}, {}
    with torch.inference_mode():
        for name, fn in cells.items():
            for b in BATCHES:
                for mode in modes:
                    try:
                        step, res = make_step(fn, b, mode)
                        steps[(name, b, mode)] = step
                        if mode == "graph":  # graph output equals eager output on the same batch
                            step(0)
                            graph_check[f"{name}|{b}"] = float((res.cpu() - reduce(fn(batches_in[b][0])).float().cpu()).abs().max())
                    except Exception as exc:  # capture failed: keep eager for this cell
                        graph_errors[f"{name}|{b}|{mode}"] = repr(exc)[:300]
                        print(f"capture failed {name} B={b} {mode}: {exc!r}"[:400], flush=True)
    rng = np.random.default_rng(0)
    raw = []
    for rep in range(args.repeats):
        order = list(steps)
        rng.shuffle(order)
        for key in order:
            name, b, mode = key
            row = dict(repeat=rep, cell=name, batch=b, mode=mode, **measure(steps[key]))
            raw.append(row)
            print(json.dumps(row), flush=True)
    med = lambda name, b, mode, k: float(np.median([r[k] for r in raw if (r["cell"], r["batch"], r["mode"]) == (name, b, mode)]))
    timed = {}
    for name in cells:
        timed[name] = {}
        for b in BATCHES:
            row = {}
            for mode in modes:
                if (name, b, mode) in steps:
                    row[mode] = dict(latency_ms=med(name, b, mode, "latency_ms"), energy_mj=med(name, b, mode, "energy_mj"))
            row["used"] = "graph" if "graph" in row else "eager"
            timed[name][b] = row

    files = []
    for fam, p in oof.items():
        ck = cost_key[fam]
        lat = [timed[ck][b][timed[ck][b]["used"]]["latency_ms"] for b in BATCHES]
        en = [timed[ck][b][timed[ck][b]["used"]]["energy_mj"] for b in BATCHES]
        path = out / "scores" / f"{fam}.npz"
        np.savez_compressed(path, index=pool, models=np.array(names), correct_prob=p.clip(0, 1).astype(np.float32),
                            fold=fold, router_cost_batches=np.array(BATCHES), router_latency_ms=np.array(lat),
                            router_energy_mj=np.array(en))
        files.append(str(path))
    result.update(gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu (test)", nvml_match=method,
                  idle_w=idle_w, job=os.environ.get("SLURM_JOB_ID"), graph_errors=graph_errors, graph_check=graph_check,
                  cost_scope="uint8 32px batch already on the GPU -> features -> predictor -> 8 batch-mean P(correct) "
                             "to pinned host; whole-GPU NVML energy incl. idle; per batch; fold-0 models",
                  cost_key=cost_key, timed={n: {str(b): v for b, v in d.items()} for n, d in timed.items()},
                  score_files=files, raw=raw, seconds=round(time.time() - t_start, 1))
    (out / "results.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    L = ["# Fast GPU hand-made-feature and Taylor routers (cross-fit, A40 cost)", "",
         f"GPU {result['gpu']}, idle {idle_w:.1f} W. Out-of-fold scores on the 5k router pool (5 folds, 4k fit / 1k "
         "scored; no classifier-validation images; test not read). Router cost per batch: median over "
         f"{args.repeats} repeats, whole-GPU energy incl. idle (as for the classifiers). 'graph' = one CUDA graph per "
         "batch size (used in the score files), 'eager' = plain PyTorch launches.", "",
         "| cell | mode | " + " | ".join(f"B={b} ms / mJ" for b in BATCHES) + " |", "|---|---|" + "---|" * len(BATCHES)]
    for name in cells:
        for mode in modes:
            vals = [timed[name][b].get(mode) for b in BATCHES]
            L.append(f"| {name} | {mode} | " + " | ".join(f"{v['latency_ms']:.3f} / {v['energy_mj']:.1f}" if v else "-"
                                                         for v in vals) + " |")
    L += ["", f"Connected components: converge in <= {cc_check['steps_needed']} steps; fixed {cc_check['fixed_steps']} "
          f"steps reproduce the converged features on {cc_check['images'] - cc_check['images_differing']}/"
          f"{cc_check['images']} pool images.", "", "Feature parity, GPU vs earlier CPU features (pool images):", ""]
    for k, v in parity.items():
        L.append(f"- {k}: max abs {v['max_abs']:.3g}; images with any mismatch (>1e-4 abs and rel) "
                 f"{v['images_any_mismatch']}/{v['images']}; {v['features_any_mismatch'] or 'no feature differs'}")
    L += ["", "GPU vs CPU predictor on fold 0 (max abs P difference): " + ", ".join(f"{k} {v:.2g}" for k, v in port_check.items()),
          "", "Graph capture failures: " + (", ".join(graph_errors) if graph_errors else "none") + "", "",
          f"Score files ({len(files)}): {out / 'scores'}", ""]
    (out / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--feature-ablation", type=Path, required=True, help="earlier CPU features, for parity only")
    p.add_argument("--taylor", type=Path, required=True, help="earlier Taylor run, for parity only")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--min-seconds", type=float, default=2.0)
    p.add_argument("--warmup-seconds", type=float, default=0.5)
    p.add_argument("--idle-seconds", type=float, default=10.0)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--dry-run", action="store_true", help="CPU-safe: inputs, parity, one fold, no timing, writes nothing")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
