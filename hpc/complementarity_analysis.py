"""How much multi-model complementarity survives one-classifier-per-batch routing?

Development analysis only: it reads the 5k classifier-validation, 3k router-training and
2k router-calibration roles of the official CIFAR-100 *training* set. It never constructs
the official test set.

  collect  (GPU, Slurm)  Verify and load the eight 128px ImageNet-initialized checkpoints,
                         save per-image predictions for the three roles, and run a short
                         latency/energy probe on the allocated GPU.
  analyze  (CPU)         Per-image and per-batch oracles vs fixed ResNet101 at batch sizes
                         1..128 with shuffled (and optionally class-coherent) batches,
                         cost-aware oracles under the 1pp budget, oracle model usage, and
                         per-class winners.

All oracles see the true labels. They are upper bounds on what any router could reach on
this split, not router results. The probe costs are screening placeholders, not the
A40 end-to-end benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROLES = ("classifier_validation", "router_train", "router_calibration")
BATCHES = (1, 8, 16, 32, 64, 128)
# Probe sizes also cover the sub-batch sizes that batch-debatch routing produces.
PROBE = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
REFERENCE = "resnet101"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- collect

def collect(args):
    import os
    import time
    import torch
    from torchvision import transforms as T
    from torchvision.datasets import CIFAR100
    from torch.utils.data import DataLoader, Subset

    from hpc.clean_protocol import file_hash, fingerprint, validate_split
    from hpc.full_factorial_protocol import checked_plan
    from hpc.train_imagenet_comparison import CASES, SOURCE_PLAN, STUDY
    from model_pipeline.models import create_classifier

    require(os.environ.get("SLURM_JOB_ID"), "Run on a Slurm allocation, not a login node")
    require(torch.cuda.is_available(), "CUDA allocation required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)

    scratch = checked_plan(args.base / "scratch32_full_factorial_v1" / "plan.json")
    study = args.base / STUDY
    prepared = load_json(study / "prepared.json")
    spec = prepared["spec"]
    require(scratch["sha256"] == spec["scratch_plan_sha256"] == SOURCE_PLAN, "Plan mismatch")
    require(file_hash(Path("model_pipeline/models.py")) == spec["source_hashes"]["model_pipeline/models.py"],
            "Model registry mismatch")
    split = load_json(args.base / "split_seed42.json")
    require(split["sha256"] == spec["split_sha256"], "Split differs from the training split")

    size = 128
    transform = T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(),
                           T.Normalize(scratch["normalization_mean"], scratch["normalization_std"])])
    # train=True only: the official test set is never instantiated here.
    full = CIFAR100(args.data, train=True, transform=transform, download=False)
    validate_split(split, full.targets)
    indices = np.concatenate([np.asarray(split["indices"][r], dtype=np.int64) for r in ROLES])
    role_id = np.concatenate([np.full(len(split["indices"][r]), i, dtype=np.int8)
                              for i, r in enumerate(ROLES)])
    labels = np.asarray(full.targets, dtype=np.int16)[indices]
    loader = DataLoader(Subset(full, indices.tolist()), batch_size=128, shuffle=False,
                        num_workers=4, pin_memory=True)

    models, names, receipts = [], [], []
    for cid in CASES:
        trial = scratch["trials"][cid]
        directory = study / f"case_{cid:04d}_{size}px"
        summary = load_json(directory / "summary.json")
        recipe = summary["recipe"]
        require(summary["status"] == "complete" and summary["epochs_complete"] == 200
                and recipe["trial"] == trial and recipe["image_size"] == size
                and recipe["plan_sha256"] == fingerprint(prepared), f"Summary mismatch: {directory}")
        path = directory / "latest.pt"
        digest = file_hash(path)
        require(digest == summary["checkpoint_sha256"], f"Hash mismatch: {path}")
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ck = torch.load(path, map_location="cpu", weights_only=True)
        require(ck["epoch"] == 200 and ck["split_sha256"] == spec["split_sha256"]
                and ck["best_accuracy"] == summary["best_validation_accuracy"]
                and bool(ck["best_state"]), f"Best checkpoint mismatch: {path}")
        model = create_classifier(trial["architecture"], pretrained=False)
        model.load_state_dict(ck["best_state"], strict=True)
        del ck
        models.append(model.cuda().eval())
        names.append(trial["architecture"])
        receipts.append(dict(case=cid, architecture=trial["architecture"], seed=trial["seed"],
                             checkpoint_sha256=digest,
                             best_validation_accuracy=summary["best_validation_accuracy"]))
    require(REFERENCE in names, "ResNet101 checkpoint missing")

    n, m = len(indices), len(models)
    pred = np.zeros((m, n), dtype=np.int16)
    conf = np.zeros((m, n), dtype=np.float16)
    logits_all = np.zeros((m, n, 100), dtype=np.float16)
    offset = 0
    with torch.inference_mode():
        for x, _ in loader:
            x = x.cuda(non_blocking=True)
            for j, model in enumerate(models):
                logits = model(x).float()
                require(bool(logits.isfinite().all()), "Non-finite logits")
                p = logits.softmax(1)
                c, k = p.max(1)
                pred[j, offset:offset + len(x)] = k.cpu().numpy()
                conf[j, offset:offset + len(x)] = c.cpu().numpy()
                logits_all[j, offset:offset + len(x)] = logits.cpu().numpy()
            offset += len(x)
    require(offset == n, "Incomplete evaluation")
    correct = pred == labels[None, :]
    # Sanity check against the trainer's own best classifier-validation accuracy
    # (trainer used FP16 autocast; FP32 here can differ by a few hundredths of a point).
    val = role_id == 0
    for j, r in enumerate(receipts):
        r["recomputed_validation_accuracy"] = round(100 * float(correct[j, val].mean()), 3)
        print(json.dumps(r), flush=True)
    np.savez_compressed(args.output / "predictions.npz", models=np.array(names), roles=np.array(ROLES),
                        index=indices, role=role_id, label=labels, pred=pred, conf=conf,
                        correct=correct, logits=logits_all)

    # --------------------------------------------------- latency / energy screening probe
    energy_fn = None
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        energy_fn = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # mJ
    except Exception as exc:  # energy is optional for this screening pass
        print(json.dumps(dict(event="energy_unavailable", reason=repr(exc))), flush=True)
    gpu = torch.cuda.get_device_name()
    rows = []
    for b in PROBE:
        x = torch.randn(b, 3, size, size, device="cuda")
        for name, model in zip(names, models):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for _ in range(args.warmup):
                    model(x)
                torch.cuda.synchronize()
                times = []
                e0 = energy_fn() if energy_fn else None
                t_all = time.perf_counter()
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    model(x)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                wall = time.perf_counter() - t_all
                e1 = energy_fn() if energy_fn else None
            # NVML's counter has coarse (~ms, ~mJ) granularity; total over many iterations.
            energy = (e1 - e0) / args.iters if energy_fn else None
            rows.append(dict(model=name, batch=b, latency_ms=1000 * float(np.median(times)),
                             wall_ms=1000 * wall / args.iters, energy_mj=energy))
            print(json.dumps(rows[-1]), flush=True)
    meta = dict(gpu=gpu, precision="PyTorch CUDA FP16 autocast, all eight models resident, "
                "synthetic input, model forward only", warmup=args.warmup, iters=args.iters,
                label="SCREENING PROXY: not the A40 end-to-end benchmark", receipts=receipts,
                job=os.environ.get("SLURM_JOB_ID"), test_evaluated=False)
    (args.output / "costs.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=2))


# ----------------------------------------------------------------------------- analyze

def load_costs(path, names):
    """Return ({objective: {batch: per-image cost over models}},
               {objective: (sizes, batch-cost matrix models x sizes)}, metadata)."""
    data = load_json(path)
    out, curves = {}, {}
    for objective, key in (("latency", "latency_ms"), ("energy", "energy_mj")):
        sizes, matrix = [], []
        for b in sorted({r["batch"] for r in data["rows"]}):
            vals = []
            for name in names:
                row = [r for r in data["rows"] if r["model"] == name and r["batch"] == b]
                vals.append(None if not row or row[0].get(key) is None else row[0][key])
            if all(v is not None for v in vals):
                sizes.append(b)
                matrix.append(vals)
        if not sizes:
            continue
        matrix = np.asarray(matrix, dtype=float).T  # models x sizes, cost of one batch
        curves[objective] = (np.asarray(sizes), matrix)
        table = {b: matrix[:, sizes.index(b)] / b for b in BATCHES if b in sizes}
        if table:
            out[objective] = table
    return out, curves, data.get("meta", {})


def batch_cost(curve, model, n):
    """Cost of one sub-batch of n images on `model`, linear interpolation between probes."""
    sizes, matrix = curve
    return float(np.interp(n, sizes, matrix[model]))


def make_batches(labels, b, rng, coherent=None):
    """Index arrays of full batches. Shuffled, or drawn from `coherent` classes each."""
    n = len(labels)
    if coherent is None:
        perm = rng.permutation(n)
        return perm[: (n // b) * b].reshape(-1, b)
    per = b // coherent
    if per * coherent != b:
        return None
    pools = {c: list(rng.permutation(np.flatnonzero(labels == c))) for c in np.unique(labels)}
    if per > min(len(v) for v in pools.values()):
        return None
    batches = []
    while True:
        alive = [c for c, v in pools.items() if len(v) >= per]
        if len(alive) < coherent:
            break
        chosen = rng.choice(alive, coherent, replace=False)
        batch = []
        for c in chosen:
            batch += pools[c][:per]
            pools[c] = pools[c][per:]
        batches.append(batch)
    return np.asarray(batches)


def oracle_choice(counts, batch_cost, lam):
    """Per batch, maximize correct - lam*cost; ties go to the cheaper model."""
    score = counts - lam * batch_cost[:, None]
    best = score.max(0, keepdims=True)
    tied = np.isclose(score, best, rtol=0, atol=1e-9)
    return np.where(tied, batch_cost[:, None], np.inf).argmin(0)


def cost_oracle(counts, batch_cost, target_correct):
    """Cheapest Lagrangian oracle whose total correct >= target. Feasible, so its savings
    are a lower bound on the exact (knapsack) oracle's savings."""
    diffs = np.abs(batch_cost[:, None] - batch_cost[None, :])
    diffs = diffs[diffs > 0]
    lo, hi = 0.0, (counts.max() + 1) / (diffs.min() if diffs.size else 1.0)
    best = oracle_choice(counts, batch_cost, 0.0)
    cols = np.arange(counts.shape[1])
    if counts[best, cols].sum() < target_correct:
        return None
    for _ in range(60):
        lam = (lo + hi) / 2
        choice = oracle_choice(counts, batch_cost, lam)
        if counts[choice, cols].sum() >= target_correct:
            lo, best = lam, choice
        else:
            hi = lam
    return best


def analyze_role(correct, labels, names, costs, b, repeats, rng, coherent, budget_pp):
    m = len(names)
    ref = names.index(REFERENCE)
    stats = dict(fixed=[], acc_oracle=[], acc_oracle_share=[], cost=dict())
    for objective in costs:
        stats["cost"][objective] = dict(zero=[], budget=[], zero_share=[], budget_share=[],
                                        zero_acc=[], budget_acc=[])
    for _ in range(repeats):
        batches = make_batches(labels, b, rng, coherent)
        if batches is None or len(batches) == 0:
            return None
        counts = correct[:, batches].sum(2).astype(float)  # m x n_batches
        nimg = batches.size
        cols = np.arange(counts.shape[1])
        fixed = 100 * counts.sum(1) / nimg
        stats["fixed"].append(fixed)
        # Accuracy-maximizing oracle (ties -> cheapest by latency if known, else ref first).
        tie_cost = costs["latency"][b] if "latency" in costs else np.where(np.arange(m) == ref, 0, 1.0)
        choice = oracle_choice(counts, tie_cost * b, 0.0)
        stats["acc_oracle"].append(100 * counts[choice, cols].sum() / nimg)
        stats["acc_oracle_share"].append(np.bincount(choice, minlength=m) / len(choice))
        for objective, table in costs.items():
            c = table[b] * b
            ref_cost = c[ref] * len(cols)
            ref_correct = counts[ref].sum()
            for tag, target in (("zero", ref_correct), ("budget", ref_correct - budget_pp / 100 * nimg)):
                pick = cost_oracle(counts, c, target)
                s = stats["cost"][objective]
                s[tag].append(100 * (1 - c[pick].sum() / ref_cost))
                s[tag + "_acc"].append(100 * counts[pick, cols].sum() / nimg)
                s[tag + "_share"].append(np.bincount(pick, minlength=m) / len(pick))
    q = lambda v: dict(mean=float(np.mean(v)), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))
    fixed = np.mean(stats["fixed"], 0)
    result = dict(batch=b, repeats=repeats, fixed={n: float(f) for n, f in zip(names, fixed)},
                  reference_accuracy=float(fixed[ref]), acc_oracle=q(stats["acc_oracle"]),
                  acc_oracle_gain_pp=q(np.asarray(stats["acc_oracle"]) - np.asarray(stats["fixed"])[:, ref]),
                  acc_oracle_share=dict(zip(names, np.mean(stats["acc_oracle_share"], 0).round(4).tolist())),
                  cost={})
    for objective, s in stats["cost"].items():
        result["cost"][objective] = {
            tag: dict(savings_pct=q(s[tag]), accuracy=q(s[tag + "_acc"]),
                      share=dict(zip(names, np.mean(s[tag + "_share"], 0).round(4).tolist())))
            for tag in ("zero", "budget")}
    return result


def pick_subset(correct, k, unit, names, rule):
    """Choose k models on the fit split. 'coverage': the k-subset whose per-image oracle
    is right most often (ties: cheaper). 'usage': the k models the all-model per-image
    oracle uses most when it routes each image to its cheapest correct model."""
    from itertools import combinations
    m = len(names)
    if rule == "coverage":
        best = max(combinations(range(m), k),
                   key=lambda s: (correct[list(s)].any(0).sum(), -unit[list(s)].sum()))
        return sorted(best)
    choice = per_image_choice(correct, np.arange(m), unit, 0.0)
    has = correct.any(0)
    counts = np.bincount(choice[has], minlength=m)
    return sorted(np.argsort(-counts, kind="stable")[:k].tolist())


def per_image_choice(correct, subset, unit, lam):
    """Per image, maximize correct - lam*unit cost within `subset`; ties go to the cheaper model."""
    subset = np.asarray(subset)
    u = unit[subset] / unit.max()
    score = correct[subset].astype(float) - (lam + 1e-6) * u[:, None]
    return subset[score.argmax(0)]


def debatch_eval(correct, labels, subset, costs, curves, b, repeats, rng, budget_pp, ref):
    """Batch-debatch: route each image of an incoming batch of b to its own model, run one
    sub-batch per chosen model (sequentially), reassemble. Cost of an incoming batch =
    sum of each model's probed cost at its sub-batch size; scatter/gather and model
    switching are not included (all eight resident)."""
    if not costs:
        return None
    unit = costs["latency"][b] if "latency" in costs else next(iter(costs.values()))[b]
    m = correct.shape[0]
    acc = {"oracle": [], "budget": []}
    save = {(o, t): [] for o in curves for t in acc}
    models_used = {t: [] for t in acc}
    share = {t: [] for t in acc}
    for _ in range(repeats):
        batches = make_batches(labels, b, rng)
        flat = batches.ravel()
        c = correct[:, flat]
        ref_correct = c[ref].sum()
        target = ref_correct - budget_pp / 100 * len(flat)
        picks = {"oracle": per_image_choice(c, subset, unit, 0.0)}
        idx = np.arange(len(flat))
        oracle_correct = c[picks["oracle"], idx].sum()
        if oracle_correct >= target:
            # Exact for per-image linear costs: an extra error is only worth taking by sending
            # a correctly routed image to the subset's cheapest model; take the largest savings.
            cheapest = min(subset, key=lambda j: unit[j])
            ch = picks["oracle"].copy()
            saving = np.where(c[ch, idx], unit[ch] - unit[cheapest], 0.0)
            allowed = int(np.floor(oracle_correct - target + 1e-9))
            order = np.argsort(-saving, kind="stable")[:allowed]
            order = order[saving[order] > 0]
            ch[order] = cheapest
            picks["budget"] = ch
        for tag, ch in picks.items():
            acc[tag].append(100 * c[ch, np.arange(len(flat))].mean())
            per_batch = ch.reshape(batches.shape)
            sizes = np.stack([(per_batch == j).sum(1) for j in range(m)], 1)  # batches x models
            models_used[tag].append((sizes > 0).sum(1).mean())
            share[tag].append(sizes.sum(0) / sizes.sum())
            for o, curve in curves.items():
                total = sum(batch_cost(curve, j, n) for row in sizes for j, n in enumerate(row) if n)
                save[(o, tag)].append(100 * (1 - total / (len(batches) * batch_cost(curve, ref, b))))
    out = {}
    for tag in acc:
        if not acc[tag]:
            out[tag] = None
            continue
        out[tag] = dict(accuracy=float(np.mean(acc[tag])), models_per_batch=float(np.mean(models_used[tag])),
                        share=np.mean(share[tag], 0).round(4).tolist(),
                        savings_pct={o: float(np.mean(save[(o, tag)])) for o in curves})
    return out


def per_class(correct, labels, names):
    ref = names.index(REFERENCE)
    acc = np.stack([[correct[j, labels == c].mean() for c in range(100)] for j in range(len(names))]) * 100
    return acc, ref


def analyze(args):
    data = np.load(args.predictions, allow_pickle=False)
    names = [str(x) for x in data["models"]]
    roles = [str(x) for x in data["roles"]]
    require(REFERENCE in names, "ResNet101 missing from predictions")
    correct, labels, role = data["correct"].astype(bool), data["label"].astype(int), data["role"]
    costs, curves, meta = load_costs(args.costs, names) if args.costs else ({}, {}, {})
    ref = names.index(REFERENCE)
    rng = np.random.default_rng(args.seed)
    views = {
        "router_calibration (2k)": role == roles.index("router_calibration"),
        "val+router_train (8k, context)": np.isin(role, [roles.index("classifier_validation"),
                                                           roles.index("router_train")]),
    }
    report = dict(meta=dict(costs=meta, budget_pp=args.budget_pp, repeats=args.repeats, seed=args.seed,
                            note="Oracles use true labels: upper bounds, development analysis only."),
                  views={})
    lines = ["# Complementarity by batch size (development analysis, no test data)", ""]
    lines.append(f"Models: {', '.join(names)}. Reference: {REFERENCE}. Budget: {args.budget_pp} pp. "
                 f"Shuffle repeats: {args.repeats}.")
    if meta:
        lines.append(f"Costs: {meta.get('label')} on {meta.get('gpu')} ({meta.get('precision')}).")
    else:
        lines.append("Costs: none given, so no savings columns.")
    lines.append("Savings are vs fixed ResNet101 at the same batch size; switching cost ignored "
                 "(all eight resident). Cost oracle = Lagrangian, a feasible lower bound on the exact oracle.")
    for view, mask in views.items():
        c, y = correct[:, mask], labels[mask]
        any_correct = 100 * c.any(0).mean()
        fixed = 100 * c.mean(1)
        out = dict(images=int(mask.sum()), fixed_accuracy=dict(zip(names, fixed.round(3).tolist())),
                   per_image_any_correct=float(any_correct), batches={})
        lines += ["", f"## {view}", "",
                  "Fixed accuracy: " + ", ".join(f"{n} {a:.2f}" for n, a in sorted(zip(names, fixed), key=lambda t: -t[1])),
                  f"Per-image oracle (any model right): {any_correct:.2f}% "
                  f"(+{any_correct - fixed[ref]:.2f} pp over ResNet101).",
                  f"ResNet101 wrong but some other model right: {100 * (~c[ref] & c.any(0)).mean():.2f}% of images; "
                  f"ResNet101 uniquely right: {100 * (c[ref] & (c.sum(0) == 1)).mean():.2f}%.", ""]
        pair = {n: float(100 * (~c[ref] & c[j]).mean()) for j, n in enumerate(names) if j != ref}
        out["rescues_over_resnet101_pct"] = pair
        lines.append("Images each model gets right that ResNet101 misses: " +
                     ", ".join(f"{n} {v:.2f}%" for n, v in sorted(pair.items(), key=lambda t: -t[1])))
        scenarios = [("shuffled", None)] + [(f"{k}-class batches", k) for k in args.coherent]
        for scen, k in scenarios:
            header = "| B | ResNet101 | acc-oracle | gain pp (p5..p95) |"
            sep = "|---|---|---|---|"
            for o in costs:
                header += f" {o} save @0pp | {o} save @-{args.budget_pp}pp |"
                sep += "---|---|"
            lines += ["", f"### {scen}", "", header, sep]
            usage = []
            for b in BATCHES:
                if k is not None and b == 1:
                    continue
                r = analyze_role(c, y, names, costs, b, 1 if b == 1 and k is None else args.repeats,
                                 rng, k, args.budget_pp)
                if r is None:
                    continue
                out["batches"][f"{scen}|{b}"] = r
                g = r["acc_oracle_gain_pp"]
                row = (f"| {b} | {r['reference_accuracy']:.2f} | {r['acc_oracle']['mean']:.2f} | "
                       f"{g['mean']:+.2f} ({g['p5']:+.2f}..{g['p95']:+.2f}) |")
                for o in costs:
                    z, bb = r["cost"][o]["zero"], r["cost"][o]["budget"]
                    row += (f" {z['savings_pct']['mean']:.1f}% |"
                            f" {bb['savings_pct']['mean']:.1f}% (acc {bb['accuracy']['mean']:.2f}) |")
                lines.append(row)
                share = r["acc_oracle_share"]
                txt = f"B={b} acc-oracle: " + ", ".join(f"{n} {100*v:.0f}%" for n, v in share.items() if v >= 0.005)
                for o in costs:
                    s = r["cost"][o]["budget"]["share"]
                    txt += f" | {o}@-{args.budget_pp}pp: " + ", ".join(
                        f"{n} {100*v:.0f}%" for n, v in s.items() if v >= 0.005)
                usage.append(txt)
            lines += ["", "Model usage (share of batches):", ""] + [f"- {u}" for u in usage]
        report["views"][view] = out

    # Per-class winners: found on val+router_train, checked on calibration (out-of-sample).
    fit = views["val+router_train (8k, context)"]
    cal = views["router_calibration (2k)"]
    acc_fit, _ = per_class(correct[:, fit], labels[fit], names)
    acc_cal, _ = per_class(correct[:, cal], labels[cal], names)
    lines += ["", "## Per-class winners", "",
              "Winners chosen on val+router_train (80 images/class), then scored on calibration "
              "(20 images/class). Ties go to the cheaper model when latency costs exist."]
    lat = costs.get("latency", {}).get(64)
    order = np.argsort(lat) if lat is not None else np.arange(len(names))
    winners = []
    for cls in range(100):
        best_val = acc_fit[:, cls].max()
        cand = [j for j in order if acc_fit[j, cls] >= best_val - 1e-9]
        winners.append(cand[0])
    winners = np.asarray(winners)
    lines.append("Classes won (fit split): " + ", ".join(
        f"{names[j]} {int((winners == j).sum())}" for j in range(len(names)) if (winners == j).any()))
    beats = [cls for cls in range(100) if winners[cls] != ref]
    lines.append(f"Classes where a non-ResNet101 model is best on the fit split: {len(beats)}")
    # Class-oracle: route by true class with the fit-split winner table, score on calibration.
    cls_pick = winners[labels[cal]]
    cal_idx = np.flatnonzero(cal)
    class_oracle = 100 * correct[cls_pick, cal_idx].mean()
    lines.append(f"True-class lookup oracle on calibration (per image, table from fit split): "
                 f"{class_oracle:.2f}% vs ResNet101 {100*correct[ref, cal].mean():.2f}%. "
                 "This bounds what an ideal class-lookup router could gain at batch 1.")
    gaps = sorted(((acc_fit[winners[c], c] - acc_fit[ref, c], c) for c in beats), reverse=True)[:args.top]
    lines += ["", "| class | winner | fit acc | ResNet101 fit | cal acc | ResNet101 cal |", "|---|---|---|---|---|---|"]
    for gap, cls in gaps:
        j = winners[cls]
        lines.append(f"| {cls} | {names[j]} | {acc_fit[j, cls]:.1f} | {acc_fit[ref, cls]:.1f} | "
                     f"{acc_cal[j, cls]:.1f} | {acc_cal[ref, cls]:.1f} |")
    held = [acc_cal[winners[c], c] - acc_cal[ref, c] for c in beats]
    if held:
        lines.append(f"\nOn calibration, those {len(beats)} winners beat ResNet101 in "
                     f"{sum(h > 0 for h in held)}, tie in {sum(h == 0 for h in held)}, lose in "
                     f"{sum(h < 0 for h in held)} classes (mean {np.mean(held):+.2f} pp).")
    weakest = np.argsort(acc_fit[ref])[:10]
    lines.append("ResNet101's 10 weakest classes on the fit split (class: R101 acc / best model acc): " +
                 ", ".join(f"{c}: {acc_fit[ref, c]:.0f}/{acc_fit[:, c].max():.0f} {names[acc_fit[:, c].argmax()]}"
                           for c in weakest))
    # Batch-debatch routing (co-advisor proposal). Subsets chosen on val+router_train only.
    fit_mask = views["val+router_train (8k, context)"]
    unit_fit = (costs["latency"][1] if "latency" in costs else
                next(iter(costs.values()))[1] if costs else np.ones(len(names)))
    subsets = {}
    for rule in ("coverage", "usage"):
        for k in args.debatch_k:
            subsets[f"top-{k} by {rule}"] = pick_subset(correct[:, fit_mask], k, unit_fit, names, rule)
    subsets["all 8"] = list(range(len(names)))
    lines += ["", "## Batch-debatch routing (upper bound)", "",
              "Each incoming batch is split; every image goes to its own model; one sub-batch per "
              "chosen model runs in sequence; outputs are reassembled. Oracle = cheapest model in the "
              "subset that is right (labels known). Budget variant = per-image cost-aware oracle "
              f"within -{args.budget_pp}pp of ResNet101 (cost per image approximated at the incoming "
              "batch size, then costed exactly). Cost = sum of probed cost at each sub-batch size, "
              "interpolated between probe sizes. Scatter/gather and launch gaps between sub-batches "
              "beyond what the probe captures, and the router itself, are NOT included. Subsets picked on val+router_train, "
              "scored on calibration.", ""]
    report["debatch"] = {}
    if not costs:
        lines.append("No costs given, so batch-debatch costs are skipped.")
    cal_c, cal_y = correct[:, views["router_calibration (2k)"]], labels[views["router_calibration (2k)"]]
    for label, subset in subsets.items():
        sub_names = ", ".join(names[j] for j in subset)
        lines += [f"### {label}: {sub_names}", "",
                  f"Per-image oracle accuracy on calibration: {100 * cal_c[subset].any(0).mean():.2f}% "
                  f"(ResNet101 {100 * cal_c[ref].mean():.2f}%).", ""]
        if not costs:
            continue
        header = "| B | oracle acc | models/batch |" + "".join(f" {o} save |" for o in curves)
        header += f" budget acc |" + "".join(f" {o} save @-{args.budget_pp}pp |" for o in curves)
        lines += [header, "|---" * (header.count("|") - 1) + "|"]
        report["debatch"][label] = dict(models=[names[j] for j in subset], batches={})
        for b in BATCHES:
            r = debatch_eval(cal_c, cal_y, subset, costs, curves, b, 1 if b == 1 else args.repeats,
                             rng, args.budget_pp, ref)
            report["debatch"][label]["batches"][b] = r
            o_, bu = r["oracle"], r["budget"]
            row = f"| {b} | {o_['accuracy']:.2f} | {o_['models_per_batch']:.2f} |"
            row += "".join(f" {o_['savings_pct'][o]:.1f}% |" for o in curves)
            if bu:
                row += f" {bu['accuracy']:.2f} |" + "".join(f" {bu['savings_pct'][o]:.1f}% |" for o in curves)
            else:
                row += " infeasible |" + " - |" * len(curves)
            lines.append(row)
        lines.append("")
    report["per_class"] = dict(fit_accuracy=acc_fit.round(2).tolist(), cal_accuracy=acc_cal.round(2).tolist(),
                               winners=[names[j] for j in winners], class_lookup_oracle_cal=float(class_oracle))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "complementarity.json").write_text(json.dumps(report, indent=1))
    (args.output / "complementarity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--base", type=Path, required=True, help="clean_classifier_study directory")
    c.add_argument("--data", type=Path, required=True, help="CIFAR-100 root")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--warmup", type=int, default=20)
    c.add_argument("--iters", type=int, default=100)
    a = sub.add_parser("analyze")
    a.add_argument("--predictions", type=Path, required=True)
    a.add_argument("--costs", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--repeats", type=int, default=50)
    a.add_argument("--budget-pp", type=float, default=1.0)
    a.add_argument("--coherent", type=int, nargs="*", default=[4, 8],
                   help="also test batches drawn from k classes each (stream locality)")
    a.add_argument("--top", type=int, default=15)
    a.add_argument("--debatch-k", type=int, nargs="*", default=[3])
    a.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    collect(args) if args.action == "collect" else analyze(args)


if __name__ == "__main__":
    sys.exit(main())
