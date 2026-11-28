"""Job 3: 5-fold cross-fitting of the routers on the 5k router pool (CPU; test never read).

The fixed 3k/2k cut (router-train / router-calibration) trains routers on 3k and locks lam on
2k, so the lock rests on 2k images and a lucky or unlucky cut moves it by ~1 pp. Cross-fitting
uses the same 5k pool more fully, without touching any other split:

  * The pool (router_train 3k + router_calibration 2k, 50 images per class) is cut into 5
    stratified folds, 10 images per class each (fold seed --fold-seed, the same folds for every
    router seed).
  * For each fold, every router is trained on the other 4 folds (4k) with the screen's recipe
    (hpc_router_arch_sweep.py: Adam 1e-3, 20 epochs, ~62 images per mini-batch, BCE on
    correctness, + CE on classes (x0.2 for the shared model), learned mix for blended, seeds
    42/45/48) and scores the held-out 1k. Every pool image gets one out-of-fold P(correct).
  * Per-class tables (lookup/blended) are built from the same 4k (+ classifier-validation for
    the tval ablations, + classifier-train for the t49k control), never from the held-out fold.
  * hpc.a40_policy then locks lam on all 5k out-of-fold scores (--roles router_train
    router_calibration) with the usual rule (1 pp, mean and >= 80% of 50 repeats). The same
    scores restricted to the 2k calibration images give a like-for-like comparison with the
    current 3k/2k lock.
  * A deployment copy of every router is also trained on all 5k (saved under deploy/, never
    scored here): lam locked on out-of-fold scores is applied to the router fit on all 5k, the
    usual cross-fitting convention. It has 25% more training data than the fold models.

Families (old 3k/2k name in brackets):
  direct-tiny16-4k [direct-tiny16-3k], direct-tiny16-val4k [direct-tiny16-val3k, ablation],
  direct-wide32-4k [direct-wide32-3k], blended-tiny16-t4k [blended-tiny16-t3k],
  blended-tiny16-tval4k [blended-tiny16-tval3k], lookup-res32-t4k [lookup-res32-t3k],
  lookup-res32-tval4k [lookup-res32-tval3k], lookup-res32-t49k [lookup-res32-t48k, control].
The val ablations add classifier-validation images, which also chose the classifier checkpoints,
so they are side ablations, not the main design.

Training targets are the screen evidence (bank PyTorch autocast correctness), as for the
screened routers; the policy lock scores routers against the deployed TensorRT FP16 predictions
from stage A. Scoring uses deployment preprocessing (16x16 routers pool the resized 128px batch;
32x32 routers normalize native 32px), as in hpc.a40_routers. Router costs: none (accuracy-only),
or --router-costs <a40_routers_v1/router_costs.json> (A40 cost depends on the architecture and
table size, not on the training images, so the measured cost of the old family is reused).

--phase train  : folds, training, out-of-fold scores (scores/*.npz), deploy/ checkpoints.
--phase report : bias check (classifier accuracy on classifier-validation 5k vs the router pool)
                 and the comparison of the cross-fit locks with the 3k/2k lock and random mix.
--dry-run      : (train) all provenance checks, fold layout, one timed epoch per network; writes
                 nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

SEEDS = (42, 45, 48)
RECIPE = dict(lr=1e-3, epochs=20, minibatches_per_3k=48, shared_class_weight=0.2)
POOL_ROLES = ("router_train", "router_calibration")
FOLDS = 5
# network: (task, arch, adds classifier-validation images to its training set)
NETS = {
    "correctness-tiny16": ("correctness", "tiny16", False),
    "correctness-tiny16-val": ("correctness", "tiny16", True),
    "correctness-wide32": ("correctness", "wide32", False),
    "shared-tiny16": ("shared", "tiny16", False),
    "class-res32": ("class", "res32", False),
}
# cross-fit family: (old 3k/2k family, network, kind, table)
#   table "fold" = training folds (4k), "valfold" = classifier-validation + training folds,
#   "all" = classifier-train + classifier-validation + training folds (control)
CF_ROUTERS = {
    "direct-tiny16-4k": ("direct-tiny16-3k", "correctness-tiny16", "direct", None),
    "direct-tiny16-val4k": ("direct-tiny16-val3k", "correctness-tiny16-val", "direct", None),
    "direct-wide32-4k": ("direct-wide32-3k", "correctness-wide32", "direct", None),
    "blended-tiny16-t4k": ("blended-tiny16-t3k", "shared-tiny16", "blended", "fold"),
    "blended-tiny16-tval4k": ("blended-tiny16-tval3k", "shared-tiny16", "blended", "valfold"),
    "lookup-res32-t4k": ("lookup-res32-t3k", "class-res32", "lookup", "fold"),
    "lookup-res32-tval4k": ("lookup-res32-tval3k", "class-res32", "lookup", "valfold"),
    "lookup-res32-t49k": ("lookup-res32-t48k", "class-res32", "lookup", "all"),
}
CONTROLS = ("lookup-res32-t49k",)
ABLATIONS = ("direct-tiny16-val4k", "blended-tiny16-tval4k", "lookup-res32-tval4k")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- pure helpers (tested)
def make_folds(pool, labels, k=FOLDS, seed=42):
    """Stratified fold id per pool image: each class's images shuffled and dealt round-robin,
    so every fold holds the same number of images of every class."""
    pool = np.asarray(pool)
    lab = np.asarray(labels)[pool]
    rng = np.random.default_rng(seed)
    fold = np.full(len(pool), -1, dtype=np.int8)
    for c in np.unique(lab):
        members = np.flatnonzero(lab == c)
        require(len(members) % k == 0, f"class {c}: {len(members)} pool images do not split into {k} folds")
        fold[rng.permutation(members)] = np.arange(len(members)) % k
    return fold


def class_table(correct, labels, ix, classes=100):
    """classes x models mean correctness over images `ix` (per-class accuracy table)."""
    ix = np.asarray(ix)
    lab = np.asarray(labels)[ix]
    rows = []
    for c in range(classes):
        sel = ix[lab == c]
        require(len(sel) > 0, f"class {c} has no images for the table")
        rows.append(correct[sel].mean(0))
    return np.stack(rows)


def minibatches(n):
    """Same mini-batch size as the screen (3k images in 48 mini-batches)."""
    return max(1, int(round(RECIPE["minibatches_per_3k"] * n / 3000)))


def brier(prob, correct):
    return float(np.mean((np.asarray(prob, float) - np.asarray(correct, float)) ** 2))


# ---------------------------------------------------------------- torch parts
def router_model(arch, task):
    """The screen's router architectures (identical to hpc.a40_routers)."""
    from torch import nn

    class Residual(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.path = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                                      nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c))

        def forward(self, x):
            return (x + self.path(x)).relu()

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


def train_net(task, arch, x, t, y, table, seed, epochs=RECIPE["epochs"]):
    """The screen's training loop. x inputs, t (n x 8) correctness targets, y (n,) labels,
    table (100 x 8) per-class table for the shared model's mix. Returns (net, mix)."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = router_model(arch, task)
    opt = torch.optim.Adam(net.parameters(), lr=RECIPE["lr"])
    mix = nn.Parameter(torch.zeros(2)) if task == "shared" else None
    if mix is not None:
        opt.add_param_group(dict(params=[mix]))
    tab = torch.as_tensor(table, dtype=torch.float32) if task == "shared" else None
    net.train()
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(x)), minibatches(len(x))):
            h = net["encoder"](x[idx])
            log = net["classes"](h) if "classes" in net else None
            q = net["correct"](h).sigmoid() if "correct" in net else None
            if mix is not None:
                mw = mix.softmax(0)
                q = mw[0] * q + mw[1] * (log.softmax(-1) @ tab)
            loss = 0
            if q is not None:
                loss = loss + F.binary_cross_entropy(q, t[idx])
            if log is not None:
                loss = loss + (RECIPE["shared_class_weight"] if mix is not None else 1.0) * F.cross_entropy(log, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net, (None if mix is None else mix.detach().clone())


def score_net(net, mix, kind, x, table=None):
    """Per-image P(correct), n x 8 in the network's (bank) column order."""
    import torch
    out = []
    with torch.inference_mode():
        for xx in x.split(256):
            h = net["encoder"](xx)
            if kind == "lookup":
                q = net["classes"](h).softmax(-1) @ torch.as_tensor(table, dtype=torch.float32)
            else:
                q = net["correct"](h).sigmoid()
                if kind == "blended":
                    mw = mix.softmax(0)
                    q = mw[0] * q + mw[1] * (net["classes"](h).softmax(-1) @ torch.as_tensor(table, dtype=torch.float32))
            out.append(q)
    return torch.cat(out).numpy().astype(np.float64).clip(0, 1)


def crossfit(data, seeds=SEEDS, epochs=RECIPE["epochs"], folds_to_run=None, log=print, deploy_dir=None):
    """Core loop on prepared tensors. `data` keys:
       pool, val, ctrain (CIFAR train indices), labels (50k), correct (50k x 8, bank order),
       fold (per pool image), train_x {arch: tensor indexed by row of `rows`}, rows (index -> row),
       score_x {arch: tensor for the pool, in pool order}.
    Returns {family: {seed: n_pool x 8 out-of-fold probabilities (bank order)}} and a timing log."""
    import torch
    pool, val, ctrain = data["pool"], data["val"], data["ctrain"]
    labels, correct, fold, rows = data["labels"], data["correct"], data["fold"], data["rows"]
    oof = {f: {s: np.full((len(pool), 8), np.nan) for s in seeds} for f in CF_ROUTERS}
    timing = []
    fold_ids = list(range(FOLDS)) if folds_to_run is None else list(folds_to_run)
    for seed in seeds:
        for f in fold_ids + (["full"] if deploy_dir is not None else []):
            held = np.zeros(len(pool), bool) if f == "full" else fold == f
            fit_pool = pool[~held]
            tables = {"fold": class_table(correct, labels, fit_pool),
                      "valfold": class_table(correct, labels, np.concatenate([val, fit_pool])),
                      "all": class_table(correct, labels, np.concatenate([ctrain, val, fit_pool]))}
            for net_name, (task, arch, add_val) in NETS.items():
                fit = np.concatenate([val, fit_pool]) if add_val else fit_pool
                r = torch.as_tensor(np.array([rows[int(i)] for i in fit]))
                x = data["train_x"][arch][r]
                t = torch.as_tensor(correct[fit], dtype=torch.float32)
                y = torch.as_tensor(labels[fit], dtype=torch.long)
                t0 = time.perf_counter()
                net, mix = train_net(task, arch, x, t, y, tables["fold"], seed, epochs)
                dt = time.perf_counter() - t0
                timing.append(dict(seed=seed, fold=f, net=net_name, images=len(fit), seconds=dt))
                log(json.dumps(dict(event="trained", **timing[-1])))
                if f == "full":
                    torch.save(dict(state=net.state_dict(), mix=mix,
                                    meta=dict(task=task, architecture=arch, seed=seed, fit="pool5k" + ("+val5k" if add_val else ""),
                                              images=len(fit), minibatches_per_epoch=minibatches(len(fit)),
                                              epochs=epochs, **{k: v for k, v in RECIPE.items() if k != "epochs"},
                                              tables={k: v.tolist() for k, v in tables.items()})),
                               Path(deploy_dir) / f"{net_name}-s{seed}.pt")
                    continue
                xs = data["score_x"][arch][torch.as_tensor(np.flatnonzero(held))]
                for fam, (_, nname, kind, tkey) in CF_ROUTERS.items():
                    if nname == net_name:
                        oof[fam][seed][held] = score_net(net, mix, kind, xs, tables.get(tkey))
    return oof, timing


# ---------------------------------------------------------------- train phase
def run_train(args):
    import torch
    from torch.nn import functional as F
    from torchvision.datasets import CIFAR100
    from hpc.clean_protocol import validate_split
    from hpc.scratch_recipes import make_plan
    from hpc.full_factorial_protocol import checked_plan

    require(os.environ.get("SLURM_JOB_ID"), "Run on a Slurm allocation, not a login node")
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    torch.set_num_threads(threads)
    torch.backends.cudnn.allow_tf32 = False
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=False)
        for d in ("scores", "deploy"):
            (args.output / d).mkdir()
    screen, evidence, stage_a = args.screen, args.screen / "evidence", args.stage_a

    # ---- provenance (as hpc.a40_routers): split, evidence, bank vs stage A, normalization
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
    for m in bank:
        require(sha256(screen / "bank" / m["file"]) == m["sha256"], f"Bank hash mismatch: {m['file']}")
    code_sha = sha256(screen / "code" / "hpc_router_arch_sweep.py")
    plan = make_plan()
    scratch = checked_plan(args.base / "scratch32_full_factorial_v1" / "plan.json")
    require(all(np.allclose(plan[k], scratch[k]) for k in ("normalization_mean", "normalization_std")),
            "Router normalization differs from the classifiers' normalization")
    mu = torch.tensor(plan["normalization_mean"]).view(1, 3, 1, 1)
    sd = torch.tensor(plan["normalization_std"]).view(1, 3, 1, 1)
    preds = [np.load(evidence / f"{a}.npy", allow_pickle=False) for a in bank_names]
    correct = np.stack([p == labels for p in preds], axis=-1).astype(np.float32)  # 50k x 8, bank order
    perm = [bank_names.index(n) for n in names]  # stage A column i <- bank column perm[i]

    ids = split["indices"]
    pool = np.concatenate([np.asarray(ids[r]) for r in POOL_ROLES])
    val, ctrain = np.asarray(ids["classifier_validation"]), np.asarray(ids["classifier_train"])
    require(len(set(pool.tolist()) & set(val.tolist())) == 0 and len(set(pool.tolist()) & set(ctrain.tolist())) == 0,
            "Router pool overlaps the classifier splits")
    fold = make_folds(pool, labels, FOLDS, args.fold_seed)
    per_class = {int(f): int(np.bincount(labels[pool[fold == f]], minlength=100).min()) for f in range(FOLDS)}
    require(len(set(per_class.values())) == 1, f"Folds are not stratified: {per_class}")
    layout = dict(pool=len(pool), folds=FOLDS, fold_seed=args.fold_seed,
                  fold_sizes=[int((fold == f).sum()) for f in range(FOLDS)], per_class_per_fold=per_class[0],
                  calibration_images_per_fold=[int(np.isin(pool[fold == f], ids["router_calibration"]).sum())
                                               for f in range(FOLDS)])
    print(json.dumps(dict(event="folds", **layout)), flush=True)

    # ---- inputs: training as in the screen, scoring as deployed (as in hpc.a40_routers)
    ev_images = np.load(evidence / "images.npy", mmap_mode="r", allow_pickle=False)
    train_ix = np.concatenate([val, pool])
    rows = {int(i): r for r, i in enumerate(train_ix)}

    def norm(u8_nhwc):
        return torch.as_tensor(np.asarray(u8_nhwc)).permute(0, 3, 1, 2).float().div_(255).sub_(mu).div_(sd)

    def chunks(ix, fn):
        return torch.cat([fn(c) for c in np.array_split(ix, max(1, len(ix) // 200))])

    def prepare128(raw_u8):
        y = F.interpolate(torch.as_tensor(np.asarray(raw_u8)).permute(0, 3, 1, 2).float(), size=(128, 128),
                          mode="bilinear", align_corners=False)
        return F.adaptive_avg_pool2d(y.round_().clamp_(0, 255).div_(255).sub_(mu).div_(sd), (16, 16))

    t0 = time.perf_counter()
    x32 = chunks(train_ix, lambda c: norm(ds.data[c]))
    train_x = {"tiny16": chunks(train_ix, lambda c: F.adaptive_avg_pool2d(norm(ev_images[c]), (16, 16))),
               "wide32": x32, "res32": x32}
    s32 = chunks(pool, lambda c: norm(ds.data[c]))
    score_x = {"tiny16": chunks(pool, lambda c: prepare128(ds.data[c])), "wide32": s32, "res32": s32}
    print(json.dumps(dict(event="inputs_ready", seconds=time.perf_counter() - t0,
                          shapes={k: list(v.shape) for k, v in train_x.items()})), flush=True)
    data = dict(pool=pool, val=val, ctrain=ctrain, labels=labels, correct=correct, fold=fold, rows=rows,
                train_x=train_x, score_x=score_x)

    if args.dry_run:
        _, timing = crossfit(data, seeds=(SEEDS[0],), epochs=1, folds_to_run=[0])
        per_epoch = {t["net"]: t["seconds"] for t in timing}
        # per seed: 5 folds + 1 deployment fit, each 20 epochs; the full fit is 25% larger
        est = sum(per_epoch.values()) * RECIPE["epochs"] * (FOLDS + 1.25) * len(SEEDS)
        print(json.dumps(dict(event="dry_run", threads=threads, seconds_per_epoch=per_epoch,
                              estimated_total_minutes=est / 60)), flush=True)
        print(f"DRY RUN OK: provenance, {FOLDS} stratified folds of {layout['fold_sizes'][0]} "
              f"({layout['per_class_per_fold']} per class), inputs, one epoch per network. "
              f"Estimated full run: {est / 60:.0f} min on {threads} threads. Nothing written.", flush=True)
        return

    oof, timing = crossfit(data, epochs=args.epochs, deploy_dir=args.output / "deploy",
                           log=lambda s: print(s, flush=True))
    costs = None
    if args.router_costs is not None:
        rows_c = json.loads(args.router_costs.read_text())["rows"]
        costs = {(r["router"], int(r["batch"])): r for r in rows_c}
    batches = (8, 16, 32, 64, 128)
    manifest = []
    for fam, (old, net_name, kind, tkey) in CF_ROUTERS.items():
        for seed in SEEDS:
            prob = oof[fam][seed]
            require(np.isfinite(prob).all(), f"Missing out-of-fold scores: {fam} s{seed}")
            prob = prob[:, perm]  # bank order -> stage A order
            extra = {}
            if costs is not None:
                require(all((old, b) in costs for b in batches), f"No A40 cost for {old}")
                extra = dict(router_cost_batches=np.array(batches),
                             router_latency_ms=np.array([costs[(old, b)]["latency_ms"] for b in batches]),
                             router_energy_mj=np.array([costs[(old, b)]["energy_mj"] for b in batches]))
            path = args.output / "scores" / f"{fam}-s{seed}.npz"
            np.savez_compressed(path, index=pool, models=np.array(names), correct_prob=prob, fold=fold, **extra)
            manifest.append(dict(router=fam, seed=seed, old_family=old, network=net_name, kind=kind, table=tkey,
                                 scores=str(path), cost_from=old if costs is not None else None))
    from hpc.a40_engines import write
    write(args.output / "crossfit.json", dict(
        label="CROSS-FIT out-of-fold router scores on the 5k router pool (router-train + router-calibration); "
              "test not opened", test_evaluated=False, job=os.environ.get("SLURM_JOB_ID"), threads=threads,
        split_sha256=split["sha256"], screen_code_sha256=code_sha, recipe=dict(RECIPE, epochs=args.epochs), seeds=list(SEEDS),
        layout=layout, routers=manifest, controls=list(CONTROLS), ablations=list(ABLATIONS),
        router_costs=None if args.router_costs is None else str(args.router_costs),
        training_targets="screen evidence (bank PyTorch autocast correctness)", timing=timing,
        deploy="deploy/<network>-s<seed>.pt trained on all 5k (not scored here)"))
    print(f"Cross-fit complete: {len(manifest)} out-of-fold score files", flush=True)


# ---------------------------------------------------------------- report phase
def family_of(key):
    head, _, tail = key.rpartition("-s")
    return head if tail.isdigit() else key


def ranking_scores(policy_json):
    sel = json.loads(Path(policy_json).read_text()).get("selection") or {}
    return {r["family"]: r for r in sel.get("ranking", [])}, sel.get("winner")


def cell_savings(policy_json):
    """{(family, objective, B): median saving over seeds if every seed routes else None}"""
    locked = json.loads(Path(policy_json).read_text())["locked"]
    cells = {}
    for key, v in locked.items():
        rname, objective, b = key.split("|")
        cells.setdefault((family_of(rname), objective, int(b)), []).append(
            v.get("calibration_saving_pct") if v["decision"] == "route" else None)
    return {k: (float(np.median(v)) if all(x is not None for x in v) else None) for k, v in cells.items()}


def run_report(args):
    from hpc.a40_policy import load_scores, load_stage_a
    cf = args.crossfit
    out_lines = ["# Job 3: cross-fitted routers on the 5k router pool", "",
                 "CALIBRATION only; test not read. Out-of-fold = each router scored only on images its fold "
                 "model never trained on. All locks: 1 pp budget vs fixed ResNet101, mean and >= 80% of 50 "
                 "shuffled batchings, A40 TensorRT FP16 pipeline costs.", ""]
    meta = json.loads((cf / "crossfit.json").read_text())
    costed = meta.get("router_costs") is not None
    out_lines += [f"Folds: {meta['layout']['folds']} x {meta['layout']['fold_sizes'][0]} images "
                  f"({meta['layout']['per_class_per_fold']} per class), fold seed {meta['layout']['fold_seed']}. "
                  f"Router cost {'included (A40, measured on the old families)' if costed else 'NOT included (accuracy-only)'}.", ""]

    # ---- bias check: classifier accuracy on classifier-validation vs the router pool
    val_a = load_stage_a(args.stage_a, ("classifier_validation",))
    pool_a = load_stage_a(args.stage_a, POOL_ROLES)
    out_lines += ["## Bias check: deployed TensorRT accuracy, classifier-validation (5k) vs router pool (5k)", "",
                  "classifier-validation chose the classifier checkpoints, so accuracy there may be optimistic. "
                  "Difference = validation - pool; 95% interval from a normal approximation (two independent "
                  "image sets).", "",
                  "| classifier | validation acc | pool acc | difference (pp) | 95% interval |", "|---|---|---|---|---|"]
    bias = {}
    for j, n in enumerate(val_a["names"]):
        p1, p2 = val_a["correct"][j].mean(), pool_a["correct"][pool_a["names"].index(n)].mean()
        se = np.sqrt(p1 * (1 - p1) / val_a["correct"].shape[1] + p2 * (1 - p2) / pool_a["correct"].shape[1])
        d = 100 * (p1 - p2)
        bias[n] = dict(validation=100 * p1, pool=100 * p2, diff_pp=d, ci=[d - 196 * se, d + 196 * se])
        out_lines.append(f"| {n} | {100 * p1:.2f} | {100 * p2:.2f} | {d:+.2f} | {d - 196 * se:+.2f}..{d + 196 * se:+.2f} |")
    out_lines.append("")

    # ---- probability quality: Brier score vs deployed correctness (lower is better)
    cal_a = load_stage_a(args.stage_a, ("router_calibration",))
    out_lines += ["## Router probability quality (Brier score vs deployed TensorRT correctness; lower is better)", "",
                  "Same 2k calibration images for both columns: old = trained on router-train 3k; cross-fit = "
                  "out-of-fold models trained on 4k. 5k column: cross-fit on the whole pool. Median over seeds.", "",
                  "| cross-fit family | old family | old 3k on cal 2k | cross-fit on cal 2k | cross-fit on pool 5k |",
                  "|---|---|---|---|---|"]
    quality = {}
    for fam, (old, *_rest) in CF_ROUTERS.items():
        b_old, b_cf2, b_cf5 = [], [], []
        for seed in SEEDS:
            p_cf = cf / "scores" / f"{fam}-s{seed}.npz"
            prob5, _ = load_scores(p_cf, pool_a["names"], pool_a["index"])
            b_cf5.append(brier(prob5, pool_a["correct"]))
            prob2, _ = load_scores(p_cf, cal_a["names"], cal_a["index"])
            b_cf2.append(brier(prob2, cal_a["correct"]))
            if args.old_scores is not None and (args.old_scores / f"{old}-s{seed}.npz").exists():
                probo, _ = load_scores(args.old_scores / f"{old}-s{seed}.npz", cal_a["names"], cal_a["index"])
                b_old.append(brier(probo, cal_a["correct"]))
        med = lambda v: f"{np.median(v):.4f}" if v else "-"
        quality[fam] = dict(old_cal2k=np.median(b_old) if b_old else None, cf_cal2k=float(np.median(b_cf2)),
                            cf_pool5k=float(np.median(b_cf5)))
        out_lines.append(f"| {fam} | {old} | {med(b_old)} | {med(b_cf2)} | {med(b_cf5)} |")
    out_lines.append("")

    # ---- policy comparison: winner-rule scores and per-cell savings
    locks = [("old 3k/2k lock (cal 2k)", args.old_policy), ("cross-fit, cal 2k only", cf / "policy_cal2k" / "policy.json"),
             ("cross-fit, pool 5k", cf / "policy_pool5k" / "policy.json")]
    locks = [(n, Path(p)) for n, p in locks if p is not None and Path(p).exists()]
    ranks = {n: ranking_scores(p) for n, p in locks}
    out_lines += ["## Pre-registered winner rule on each lock", "",
                  "Score = mean over (objective, B) cells of the median-over-seeds saving (a cell counts 0 unless "
                  "all 3 seeds route). Old families are matched to their cross-fit counterparts.", "",
                  "| cross-fit family | old family | " + " | ".join(n for n, _ in locks) + " |",
                  "|---|---|" + "---|" * len(locks)]
    for fam, (old, *_rest) in CF_ROUTERS.items():
        cols = []
        for n, _ in locks:
            r = ranks[n][0].get(old if n.startswith("old") else fam)
            cols.append("-" if r is None else f"{r['score']:.2f} ({r['cells_routed_all_seeds']}/{r['cells']})")
        tag = " (control)" if fam in CONTROLS else " (ablation)" if fam in ABLATIONS else ""
        out_lines.append(f"| {fam}{tag} | {old} | " + " | ".join(cols) + " |")
    out_lines.append("| WINNER | | " + " | ".join(str(ranks[n][1]) for n, _ in locks) + " |")
    out_lines.append("")

    pool_policy = cf / "policy_pool5k" / "policy.json"
    base_json = cf / "baselines_pool5k" / "baselines.json"
    if pool_policy.exists() and base_json.exists():
        cells = cell_savings(pool_policy)
        base = json.loads(base_json.read_text())
        mix = {(c["objective"], c["batch"]): c["random_mix"] for c in base["cells"] if abs(c["budget_pp"] - 1) < 1e-9}
        fams = [f for f in CF_ROUTERS if f not in CONTROLS]
        out_lines += ["## Cross-fit pool 5k lock vs random mix at 1 pp (median saving over seeds; '-' = a seed falls back)", "",
                      "| objective | B | random mix | " + " | ".join(fams) + " |", "|---|---|---|" + "---|" * len(fams)]
        for (objective, b) in sorted(mix, key=lambda k: (k[0], k[1])):
            m = mix[(objective, b)]
            mtxt = "-" if m is None else f"{m['saving']:.1f}%"
            cols = ["-" if cells.get((f, objective, b)) is None else f"{cells[(f, objective, b)]:.1f}%" for f in fams]
            out_lines.append(f"| {objective} | {b} | {mtxt} | " + " | ".join(cols) + " |")
        out_lines.append("")
    out_lines += ["The cross-fit lock uses out-of-fold scores from fold models (4k each); the deployed router "
                  "would be the copy trained on all 5k (deploy/), which has slightly more data than the models "
                  "the lock was measured on.", ""]
    (cf / "crossfit_report.md").write_text("\n".join(out_lines) + "\n")
    (cf / "crossfit_report.json").write_text(json.dumps(dict(bias=bias, brier=quality,
                                                              rankings={n: r[0] for n, r in ranks.items()},
                                                              winners={n: r[1] for n, r in ranks.items()}),
                                                         indent=1, default=float) + "\n")
    print("\n".join(out_lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("train", "report"), required=True)
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--base", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--screen", type=Path)
    p.add_argument("--output", type=Path, help="train: new output directory")
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--router-costs", type=Path, default=None,
                   help="train: a40_routers_v1/router_costs.json to attach A40 router costs (else accuracy-only)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=int, default=RECIPE["epochs"], help="train: keep the screen's 20 (smoke tests only)")
    p.add_argument("--crossfit", type=Path, help="report: the train phase output")
    p.add_argument("--old-policy", type=Path, default=None, help="report: the 3k/2k policy.json to compare with")
    p.add_argument("--old-scores", type=Path, default=None, help="report: the 3k/2k scores/ directory")
    args = p.parse_args(argv)
    if args.phase == "train":
        for k in ("base", "data", "screen"):
            require(getattr(args, k) is not None, f"--{k} is required for --phase train")
        require(args.dry_run or args.output is not None, "--output is required")
        run_train(args)
    else:
        require(args.crossfit is not None, "--crossfit is required for --phase report")
        run_report(args)


if __name__ == "__main__":
    sys.exit(main())
