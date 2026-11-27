"""Job 3, stage B2: lock routing policies on router-calibration (2k) with A40 TensorRT costs.

CPU only. Never reads test data. Inputs:
  --stage-a   job 3 stage A output (predictions.npz with the deployed TRT FP16 GPU-pipeline
              predictions, costs_pipeline.json with A40 per-batch latency/energy).
  --scores    zero or more router score files (npz), one per router variant, holding for the
              router-calibration images:
                index        (n,) CIFAR train indices, any order
                models       (8,) classifier names in score column order
                correct_prob (n, 8) predicted P(classifier j is right on image i)
              and optionally router_ms / router_mj per batch size (router_cost_batches,
              router_latency_ms, router_energy_mj) measured on the A40. Without them the
              router is costed at zero and the report says so.

Policy per router, objective (latency | energy) and batch size B:
  choose argmax_j  mean_{i in batch} correct_prob[i, j] - lam * cost_j(B) / cost_ResNet101(B)
lam is locked on calibration: the cheapest lam whose mean accuracy over --repeats shuffled
batchings is >= ResNet101's accuracy on the same batchings minus --budget-pp. The locked
policy must also be cheaper than the cheapest eligible fixed classifier (a fixed model within
the budget), router cost included; otherwise it is locked as FALLBACK to that fixed model.
Also reported, for context only: every fixed model, a labels-known oracle (upper bound), and
a matched-random control that draws models with the locked router's calibration shares.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from hpc.complementarity_analysis import cost_oracle, make_batches

BATCHES = (8, 16, 32, 64, 128)
OBJECTIVES = {"latency": "latency_ms", "energy": "energy_mj"}
REFERENCE = "resnet101"
LAMBDAS = np.concatenate([[0.0], np.geomspace(1e-4, 5.0, 160)])


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_stage_a(stage_a):
    data = np.load(stage_a / "predictions.npz", allow_pickle=False)
    names = [str(x) for x in data["models"]]
    roles = [str(x) for x in data["roles"]]
    cal = data["role"] == roles.index("router_calibration")
    correct = (data["pred_trt_fp16_gpu"] == data["label"][None, :])[:, cal]
    costs_raw = load_json(stage_a / "costs_pipeline.json")
    require(costs_raw["meta"]["test_evaluated"] is False, "Stage A cost file is not test-free")
    costs = {}
    for objective, key in OBJECTIVES.items():
        costs[objective] = {}
        for b in BATCHES:
            row = {r["model"]: r[key] for r in costs_raw["rows"]
                   if r["variant"] == "trt_fp16" and r["batch"] == b and r.get(key) is not None}
            require(set(row) == set(names), f"Missing {objective} costs at B={b}")
            costs[objective][b] = np.array([row[n] for n in names], dtype=float)
    return dict(names=names, index=data["index"][cal], label=data["label"][cal].astype(int),
                correct=correct, costs=costs, meta=costs_raw["meta"])


def load_scores(path, names, index):
    data = np.load(path, allow_pickle=False)
    order = [str(x) for x in data["models"]]
    require(sorted(order) == sorted(names), f"{path}: classifier set differs from stage A")
    prob = data["correct_prob"].astype(float)
    require(prob.shape == (len(data["index"]), len(order)), f"{path}: bad correct_prob shape")
    require(np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1, f"{path}: probabilities out of range")
    pos = {int(k): i for i, k in enumerate(data["index"])}
    require(set(pos) == set(int(k) for k in index), f"{path}: images differ from router-calibration")
    rows = np.array([pos[int(k)] for k in index])
    prob = prob[rows][:, [order.index(n) for n in names]].T  # models x n, stage A order
    router = {o: {} for o in OBJECTIVES}
    if "router_cost_batches" in data.files:
        for o, key in (("latency", "router_latency_ms"), ("energy", "router_energy_mj")):
            for b, v in zip(data["router_cost_batches"].tolist(), data[key].tolist()):
                router[o][int(b)] = float(v)
    return prob, router


def evaluate_choices(correct, batchings, choices, costs_b):
    """Mean accuracy (%) and mean total cost over batchings for per-batch model choices."""
    acc, cost = [], []
    for batches, ch in zip(batchings, choices):
        acc.append(100 * correct[ch[:, None], batches].mean())
        cost.append(costs_b[ch].sum())
    return float(np.mean(acc)), float(np.mean(cost)), np.asarray(acc)


def lock_router(prob, correct, batchings, costs_b, ref, target_acc, lambdas=LAMBDAS):
    """Cheapest lam meeting target_acc on calibration. Returns dict or None if infeasible."""
    norm = costs_b / costs_b[ref]
    mean_prob = [prob[:, batches].mean(2) for batches in batchings]  # models x n_batches
    best = None
    for lam in lambdas:
        choices = [(mp - lam * norm[:, None]).argmax(0) for mp in mean_prob]
        acc, cost, per = evaluate_choices(correct, batchings, choices, costs_b)
        if acc >= target_acc and (best is None or cost < best["cost"] - 1e-12):
            share = np.bincount(np.concatenate(choices), minlength=len(costs_b))
            best = dict(lam=float(lam), accuracy=acc, cost=cost, per_repeat_accuracy=per,
                        share=share / share.sum())
    return best


def matched_random(share, correct, batchings, costs_b, rng):
    choices = [rng.choice(len(share), size=len(b), p=share) for b in batchings]
    return evaluate_choices(correct, batchings, choices, costs_b)


def lock(args):
    a = load_stage_a(args.stage_a)
    names, correct, labels = a["names"], a["correct"], a["label"]
    ref = names.index(REFERENCE)
    routers = {}
    for path in args.scores:
        routers[Path(path).stem] = load_scores(path, names, a["index"])
    rng = np.random.default_rng(args.seed)
    policy = dict(meta=dict(stage="B2 policy lock on router-calibration (2k); test not read",
                            budget_pp=args.budget_pp, repeats=args.repeats, seed=args.seed,
                            classifier_costs="A40 TensorRT FP16 pipeline (stage A, job %s)" % a["meta"]["device"]["job"],
                            models=names, reference=REFERENCE, lambdas=len(LAMBDAS)),
                  locked={})
    lines = ["# Job 3 stage B2: policies locked on router-calibration (2k)", "",
             "CALIBRATION (policy locking), not held-out results. Classifier correctness = deployed TensorRT "
             f"FP16 GPU-pipeline predictions; costs = A40 pipeline medians from stage A. Budget: within "
             f"-{args.budget_pp} pp of fixed ResNet101 on the same {args.repeats} shuffled batchings per B. "
             "Savings are vs fixed ResNet101 TensorRT FP16 (not PyTorch autocast).", ""]
    summary = {}
    for b in BATCHES:
        batchings = [make_batches(labels, b, rng) for _ in range(args.repeats)]
        nb = len(batchings[0])
        fixed_acc = np.array([np.mean([100 * correct[j][bt].mean() for bt in batchings]) for j in range(len(names))])
        target = fixed_acc[ref] - args.budget_pp
        eligible = [j for j in range(len(names)) if fixed_acc[j] >= target]
        lines += [f"## B={b} ({nb} batches per repeat)", "",
                  "Fixed accuracy: " + ", ".join(f"{names[j]} {fixed_acc[j]:.2f}" for j in np.argsort(-fixed_acc)),
                  f"Target accuracy: {target:.2f}. Eligible fixed: " + ", ".join(names[j] for j in eligible), "",
                  "| objective | policy | lam | accuracy | repeats meeting target | saving vs RN101 | "
                  "router cost included | decision | selections |", "|---|---|---|---|---|---|---|---|---|"]
        for objective in OBJECTIVES:
            cb = a["costs"][objective][b]
            ref_cost = cb[ref] * nb
            cheapest_fixed = min(eligible, key=lambda j: cb[j])
            fixed_cost = cb[cheapest_fixed] * nb
            save = lambda c: 100 * (1 - c / ref_cost)
            # labels-known oracle (upper bound, context only)
            oracle_acc, oracle_cost = [], []
            for bt in batchings:
                counts = correct[:, bt].sum(2).astype(float)
                pick = cost_oracle(counts, cb, counts[ref].sum() - args.budget_pp / 100 * bt.size)
                oracle_acc.append(100 * counts[pick, np.arange(len(bt))].sum() / bt.size)
                oracle_cost.append(cb[pick].sum())
            lines.append(f"| {objective} | oracle (labels known, bound) | - | {np.mean(oracle_acc):.2f} | - | "
                         f"{save(np.mean(oracle_cost)):.1f}% | no | context only | - |")
            lines.append(f"| {objective} | fixed {names[cheapest_fixed]} | - | {fixed_acc[cheapest_fixed]:.2f} | - | "
                         f"{save(fixed_cost):.1f}% | - | fallback baseline | - |")
            for rname, (prob, rcost) in routers.items():
                res = lock_router(prob, correct, batchings, cb, ref, target)
                r_overhead = rcost[objective].get(b)
                key = f"{rname}|{objective}|{b}"
                family = rname.rsplit("-s", 1)[0] if rname.rsplit("-s", 1)[-1].isdigit() else rname
                srow = summary.setdefault((family, objective, b), [])
                if res is None:
                    srow.append(None)
                    policy["locked"][key] = dict(decision="fallback", fixed=names[cheapest_fixed],
                                                 reason="no lam meets the calibration budget")
                    lines.append(f"| {objective} | {rname} | - | infeasible | - | - | - | "
                                 f"FALLBACK {names[cheapest_fixed]} | - |")
                    continue
                total = res["cost"] + (r_overhead or 0.0) * nb
                routed = total < fixed_cost
                srow.append(dict(routed=routed, saving=save(total), accuracy=res["accuracy"]))
                meet = float((res["per_repeat_accuracy"] >= target).mean())
                share = {names[j]: round(float(s), 4) for j, s in enumerate(res["share"]) if s > 0}
                policy["locked"][key] = dict(
                    decision="route" if routed else "fallback", fixed=names[cheapest_fixed], lam=res["lam"],
                    calibration_accuracy=res["accuracy"], calibration_saving_pct=save(total),
                    repeats_meeting_target=meet, router_cost_per_batch=r_overhead,
                    router_cost_included=r_overhead is not None, selection_share=share,
                    reason=None if routed else "not cheaper than the cheapest eligible fixed classifier")
                sel = ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(share.items(), key=lambda kv: -kv[1]))
                lines.append(f"| {objective} | {rname} | {res['lam']:.4g} | {res['accuracy']:.2f} | "
                             f"{100 * meet:.0f}% | {save(total):.1f}% | {'yes' if r_overhead is not None else 'NO (zero)'} | "
                             f"{'ROUTE' if routed else 'FALLBACK ' + names[cheapest_fixed]} | {sel} |")
                if routed:
                    racc, rcost_, _ = matched_random(res["share"], correct, batchings, cb, rng)
                    lines.append(f"| {objective} | matched random ({rname}) | - | {racc:.2f} | - | "
                                 f"{save(rcost_ + (r_overhead or 0.0) * nb):.1f}% | as router | control | same shares |")
        lines.append("")
    if summary:
        costed = [bool(rc["latency"]) for _, rc in routers.values()]
        cost_note = ("router cost included" if all(costed) else
                     "router cost NOT included (counted as zero; accuracy-only lock)" if not any(costed) else
                     "router cost included only where measured; see the per-B tables")
        lines += ["## Summary across training seeds", "",
                  "Per router family: seeds locked as ROUTE (others fall back), and median calibration "
                  f"accuracy and saving over the seeds that met the budget ({cost_note}).", "",
                  "| router | objective | B | route / seeds | median accuracy | median saving |",
                  "|---|---|---|---|---|---|"]
        for (family, objective, b), rows in sorted(summary.items()):
            ok = [r for r in rows if r is not None]
            routed = sum(r["routed"] for r in ok)
            med = lambda k: f"{np.median([r[k] for r in ok]):.2f}" if ok else "-"
            lines.append(f"| {family} | {objective} | {b} | {routed}/{len(rows)} | {med('accuracy')} | "
                         f"{med('saving')}{'%' if ok else ''} |")
        lines.append("")
    if not routers:
        lines += ["No router score files given: only fixed classifiers and the labels-known bound are shown.", ""]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "policy.json").write_text(json.dumps(policy, indent=1, default=float) + "\n")
    (args.output / "policy.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--scores", type=Path, nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget-pp", type=float, default=1.0)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    lock(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
