"""Job 2: how should debatch routing be executed? (development analysis, no test data)

Compares, for each model set and incoming batch size, against fixed ResNet101:
  whole-batch          one model per batch (best-in-set oracle)
  debatch sequential   each image to its own model, one sub-batch per chosen model, in turn
  debatch concurrent   same routing, sub-batches on separate CUDA streams (measured)
  per-image (batch 1)  same routing, every image runs alone at batch 1

  run      (GPU, Slurm) Verify and load the eight 128px checkpoints, save per-image
           predictions for classifier-validation, router-train and router-calibration,
           probe per-model cost curves, then measure sequential vs concurrent execution
           of a representative sub-batch split for every model set and batch size.
  analyze  (CPU)        Four-strategy tables on router-calibration.

Routing target is per-model correctness (the cheapest model that is right; labels known),
so all accuracies are upper bounds. Model sets are chosen on val+router_train. Costs are
screening measurements on the allocated GPU, not the A40 end-to-end benchmark.
This module leaves hpc/complementarity_analysis.py (job 1) unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from hpc.complementarity_analysis import (BATCHES, PROBE, REFERENCE, ROLES, batch_cost, cost_oracle,
                                          load_costs, load_json, make_batches, oracle_choice,
                                          pick_subset, per_image_choice, require)

STRATEGIES = ("whole-batch", "debatch sequential", "debatch concurrent", "per-image (batch 1)")


def define_subsets(correct_fit, unit1, names, ks):
    subsets = {}
    for rule in ("coverage", "usage"):
        for k in ks:
            subsets[f"top-{k} by {rule}"] = [int(j) for j in pick_subset(correct_fit, k, unit1, names, rule)]
    subsets["all 8"] = list(range(len(names)))
    return subsets


def representative_split(correct_fit, subset, unit_b, b):
    """Mean sub-batch sizes of the cheapest-correct oracle for an incoming batch of b."""
    choice = per_image_choice(correct_fit, subset, unit_b, 0.0)
    share = np.array([(choice == j).mean() for j in subset]) * b
    split = np.floor(share).astype(int)
    split[np.argsort(-(share - split))[: b - split.sum()]] += 1
    return split


# --------------------------------------------------------------------------------- run

def run(args):
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
    full = CIFAR100(args.data, train=True, transform=transform, download=False)  # never the test set
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
        receipts.append(dict(case=cid, architecture=trial["architecture"], checkpoint_sha256=digest))

    n = len(indices)
    pred = np.zeros((len(models), n), dtype=np.int16)
    offset = 0
    with torch.inference_mode():
        for x, _ in loader:
            x = x.cuda(non_blocking=True)
            for j, model in enumerate(models):
                logits = model(x).float()
                require(bool(logits.isfinite().all()), "Non-finite logits")
                pred[j, offset:offset + len(x)] = logits.argmax(1).cpu().numpy()
            offset += len(x)
    require(offset == n, "Incomplete evaluation")
    correct = pred == labels[None, :]
    for j, r in enumerate(receipts):
        r["validation_accuracy"] = round(100 * float(correct[j, role_id == 0].mean()), 3)
        print(json.dumps(r), flush=True)
    np.savez_compressed(args.output / "predictions.npz", models=np.array(names), roles=np.array(ROLES),
                        index=indices, role=role_id, label=labels, pred=pred, correct=correct)

    energy_fn = None
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        energy_fn = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # mJ
    except Exception as exc:
        print(json.dumps(dict(event="energy_unavailable", reason=repr(exc))), flush=True)

    def measure(fn):
        # NVML's energy counter is coarse; total it over many iterations.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(args.warmup):
                fn()
            torch.cuda.synchronize()
            times = []
            e0 = energy_fn() if energy_fn else None
            for _ in range(args.iters):
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            e1 = energy_fn() if energy_fn else None
        return 1000 * float(np.median(times)), ((e1 - e0) / args.iters if energy_fn else None)

    rows = []
    for b in PROBE:
        x = torch.randn(b, 3, size, size, device="cuda")
        for name, model in zip(names, models):
            ms, mj = measure(lambda: model(x))
            rows.append(dict(precision="autocast", model=name, batch=b, latency_ms=ms, energy_mj=mj))
            print(json.dumps(rows[-1]), flush=True)

    fit = np.isin(role_id, [0, 1])
    unit = {b: np.array([[r["latency_ms"] for r in rows if r["model"] == nm and r["batch"] == b][0]
                         for nm in names]) / b for b in BATCHES}
    subsets = define_subsets(correct[:, fit], unit[1], names, args.debatch_k)
    concurrent = []
    for label, subset in subsets.items():
        for b in BATCHES[1:]:
            parts = representative_split(correct[:, fit], subset, unit[b], b)
            jobs = [(models[j], torch.randn(k, 3, size, size, device="cuda"))
                    for j, k in zip(subset, parts) if k]
            streams = [torch.cuda.Stream() for _ in jobs]
            for st in streams:
                st.wait_stream(torch.cuda.current_stream())

            def sequential():
                for model, x in jobs:
                    model(x)

            def parallel():
                for st, (model, x) in zip(streams, jobs):
                    with torch.cuda.stream(st):
                        model(x)
                for st in streams:
                    torch.cuda.current_stream().wait_stream(st)

            seq_ms, seq_mj = measure(sequential)
            par_ms, par_mj = measure(parallel)
            concurrent.append(dict(subset=label, models=[names[j] for j in subset], batch=b,
                                   split=[int(k) for k in parts], seq_latency_ms=seq_ms, seq_energy_mj=seq_mj,
                                   conc_latency_ms=par_ms, conc_energy_mj=par_mj))
            print(json.dumps(concurrent[-1]), flush=True)
    meta = dict(gpu=torch.cuda.get_device_name(),
                precision="PyTorch CUDA FP16 autocast, all eight models resident, synthetic input, "
                          "model forward only; concurrent = one CUDA stream per chosen model",
                warmup=args.warmup, iters=args.iters, receipts=receipts,
                label="SCREENING PROXY: not the A40 end-to-end benchmark",
                job=os.environ.get("SLURM_JOB_ID"), test_evaluated=False)
    (args.output / "costs.json").write_text(json.dumps(dict(meta=meta, rows=rows, concurrent=concurrent),
                                                       indent=2))


# ----------------------------------------------------------------------------- analyze

def evaluate(correct, labels, subset, costs, curves, b, repeats, rng, budget_pp, ref, conc=None):
    """Accuracy and savings vs fixed ResNet101 for the four execution strategies.
    Concurrent cost = sequential cost scaled by the measured concurrent/sequential ratio
    at this set's representative split. Router, scatter/gather and model switching are
    not costed (all eight resident)."""
    unit = costs["latency"][b] if "latency" in costs else next(iter(costs.values()))[b]
    m = correct.shape[0]
    sub = np.asarray(subset)
    ratio = {}
    for o in curves:
        key = "latency_ms" if o == "latency" else "energy_mj"
        if conc and conc.get("seq_" + key) and conc.get("conc_" + key) is not None:
            ratio[o] = conc["conc_" + key] / conc["seq_" + key]
    res = {(st, t): dict(acc=[], used=[], **{o: [] for o in curves})
           for st in STRATEGIES for t in ("oracle", "budget")}
    for _ in range(repeats):
        batches = make_batches(labels, b, rng)
        nb, flat = len(batches), batches.ravel()
        c = correct[:, flat]
        idx = np.arange(len(flat))
        target = c[ref].sum() - budget_pp / 100 * len(flat)
        ref_cost = {o: nb * batch_cost(curve, ref, b) for o, curve in curves.items()}
        counts = correct[:, batches].sum(2).astype(float)[sub]
        whole = {"oracle": sub[oracle_choice(counts, unit[sub] * b, 0.0)]}
        pick = cost_oracle(counts, unit[sub] * b, target)
        if pick is not None:
            whole["budget"] = sub[pick]
        for t, ch in whole.items():
            r = res[("whole-batch", t)]
            r["acc"].append(100 * correct[ch[:, None], batches].mean())
            r["used"].append(1.0)
            for o, curve in curves.items():
                r[o].append(100 * (1 - sum(batch_cost(curve, j, b) for j in ch) / ref_cost[o]))
        picks = {"oracle": per_image_choice(c, sub, unit, 0.0)}
        oracle_correct = c[picks["oracle"], idx].sum()
        if oracle_correct >= target:
            cheapest = min(subset, key=lambda j: unit[j])
            ch = picks["oracle"].copy()
            saving = np.where(c[ch, idx], unit[ch] - unit[cheapest], 0.0)
            order = np.argsort(-saving, kind="stable")[:int(np.floor(oracle_correct - target + 1e-9))]
            ch[order[saving[order] > 0]] = cheapest
            picks["budget"] = ch
        for t, ch in picks.items():
            sizes = np.stack([(ch.reshape(batches.shape) == j).sum(1) for j in range(m)], 1)
            accuracy = 100 * c[ch, idx].mean()
            used = (sizes > 0).sum(1).mean()
            seq = {o: sum(batch_cost(curve, j, k) for row in sizes for j, k in enumerate(row) if k)
                   for o, curve in curves.items()}
            single = {o: sum(batch_cost(curve, j, 1) * k for j, k in enumerate(sizes.sum(0)))
                      for o, curve in curves.items()}
            for st, total in (("debatch sequential", seq), ("per-image (batch 1)", single),
                              ("debatch concurrent", {o: seq[o] * ratio[o] for o in ratio})):
                r = res[(st, t)]
                r["acc"].append(accuracy)
                r["used"].append(used)
                for o in curves:
                    if o in total:
                        r[o].append(100 * (1 - total[o] / ref_cost[o]))
    out = {}
    for (st, t), r in res.items():
        out.setdefault(st, {})[t] = None if not r["acc"] else dict(
            accuracy=float(np.mean(r["acc"])), models_per_batch=float(np.mean(r["used"])),
            savings_pct={o: (float(np.mean(r[o])) if r[o] else None) for o in curves})
    return out


def analyze(args):
    data = np.load(args.predictions, allow_pickle=False)
    names = [str(x) for x in data["models"]]
    roles = [str(x) for x in data["roles"]]
    correct, labels, role = data["correct"].astype(bool), data["label"].astype(int), data["role"]
    ref = names.index(REFERENCE)
    costs, curves, meta = load_costs(args.costs, names)
    raw = load_json(args.costs)
    conc = {(r["subset"], r["batch"]): r for r in raw.get("concurrent", [])}
    fit = np.isin(role, [roles.index("classifier_validation"), roles.index("router_train")])
    cal = role == roles.index("router_calibration")
    subsets = define_subsets(correct[:, fit], costs["latency"][1], names, args.debatch_k)
    rng = np.random.default_rng(args.seed)
    fmt = lambda v: "-" if v is None else f"{v:.1f}%"
    lines = ["# Debatch execution strategies (development analysis, no test data)", "",
             f"Costs: {meta.get('label')} on {meta.get('gpu')} ({meta.get('precision')}).",
             "Routing = cheapest correct model (labels known): upper bounds. Savings vs fixed ResNet101 "
             "at the same batch size, same GPU. Model sets chosen on val+router_train, scored on "
             f"router-calibration. Budget = within -{args.budget_pp}pp of ResNet101. Router, "
             "scatter/gather and model switching not costed.",
             f"ResNet101 calibration accuracy: {100 * correct[ref, cal].mean():.2f}%."]
    report = dict(meta=meta, sets={})
    for label, subset in subsets.items():
        lines += ["", f"## {label}: {', '.join(names[j] for j in subset)}", "",
                  "| B | strategy | acc | models/batch | latency save | energy save | "
                  f"acc @-{args.budget_pp}pp | latency save | energy save |",
                  "|---|---|---|---|---|---|---|---|---|"]
        report["sets"][label] = dict(models=[names[j] for j in subset], batches={})
        for b in BATCHES:
            r = evaluate(correct[:, cal], labels[cal], subset, costs, curves, b,
                         1 if b == 1 else args.repeats, rng, args.budget_pp, ref, conc.get((label, b)))
            report["sets"][label]["batches"][b] = r
            if b in (c_["batch"] for c_ in raw.get("concurrent", []) if c_["subset"] == label):
                cr = conc[(label, b)]
                report["sets"][label]["batches"][b]["concurrent_probe"] = cr
            for st in STRATEGIES:
                o_, bu = r[st]["oracle"], r[st]["budget"]
                if o_ is None or (st == "debatch concurrent" and b == 1):
                    continue
                row = (f"| {b} | {st} | {o_['accuracy']:.2f} | {o_['models_per_batch']:.2f} | "
                       f"{fmt(o_['savings_pct'].get('latency'))} | {fmt(o_['savings_pct'].get('energy'))} |")
                row += (f" {bu['accuracy']:.2f} | {fmt(bu['savings_pct'].get('latency'))} | "
                        f"{fmt(bu['savings_pct'].get('energy'))} |" if bu else " infeasible | - | - |")
                lines.append(row)
        probes = [c_ for c_ in raw.get("concurrent", []) if c_["subset"] == label]
        if probes:
            lines += ["", "Measured representative splits (sequential vs concurrent, ms):", ""]
            lines += [f"- B={c_['batch']} split {c_['split']}: {c_['seq_latency_ms']:.2f} vs "
                      f"{c_['conc_latency_ms']:.2f} ms ({100 * (c_['conc_latency_ms'] / c_['seq_latency_ms'] - 1):+.0f}%)"
                      for c_ in probes]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "execution.json").write_text(json.dumps(report, indent=1, default=float))
    (args.output / "execution.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", type=Path, required=True)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--iters", type=int, default=100)
    r.add_argument("--debatch-k", type=int, nargs="*", default=[3])
    a = sub.add_parser("analyze")
    a.add_argument("--predictions", type=Path, required=True)
    a.add_argument("--costs", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--repeats", type=int, default=50)
    a.add_argument("--budget-pp", type=float, default=1.0)
    a.add_argument("--debatch-k", type=int, nargs="*", default=[3])
    a.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run(args) if args.action == "run" else analyze(args)


if __name__ == "__main__":
    sys.exit(main())
