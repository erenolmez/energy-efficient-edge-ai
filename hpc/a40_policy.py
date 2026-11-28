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


def batchings_for(labels, b, rng, repeats, coherent=None, groups=None):
    """`repeats` batchings of size b: shuffled (i.i.d. stream), or each batch drawn from `coherent`
    classes (a non-i.i.d. stream). With `groups` (e.g. cross-fit fold ids), every batch is drawn
    within one group, so a batch never mixes images scored by different fold models (needed by
    set routers, whose batch score is not an average of per-image scores). All repeats are cut to
    the same number of batches. None if batches of this size cannot be formed."""
    def one():
        if groups is None:
            return make_batches(labels, b, rng, coherent)
        parts = []
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            mb = make_batches(labels[idx], b, rng, coherent)
            if mb is None or len(mb) == 0:
                return None
            parts.append(idx[mb])
        return rng.permutation(np.concatenate(parts))
    out = [one() for _ in range(repeats)]
    if any(x is None or len(x) == 0 for x in out):
        return None
    n = min(len(x) for x in out)
    return [x[:n] for x in out]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_stage_a(stage_a, roles=("router_calibration",)):
    """Stage A correctness and costs for the images of `roles` (default router-calibration).
    ("router_train", "router_calibration") gives the 5k router pool used by cross-fitting."""
    data = np.load(stage_a / "predictions.npz", allow_pickle=False)
    names = [str(x) for x in data["models"]]
    stored = [str(x) for x in data["roles"]]
    require(roles and all(r in stored and r != "test" for r in roles), f"Unknown or test role in {roles}")
    cal = np.isin(data["role"], [stored.index(r) for r in roles])
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
                correct=correct, costs=costs, meta=costs_raw["meta"], roles=tuple(roles),
                pool=pool_label(roles, int(cal.sum())))


def restrict(a, models):
    """Stage A data limited to a classifier subset (the router may pick only these). The reference
    model must stay in. None or empty = all eight."""
    if not models:
        return a
    require(REFERENCE in models and all(m in a["names"] for m in models), f"Bad classifier subset {models}")
    keep = [a["names"].index(m) for m in models]
    return dict(a, names=list(models), correct=a["correct"][keep],
                costs={o: {b: v[keep] for b, v in by_b.items()} for o, by_b in a["costs"].items()})


def pool_label(roles, n):
    return " + ".join(r.replace("_", "-") for r in roles) + f" ({n / 1000:g}k)"


class SetScorer:
    """Batch-set router: per-image embeddings (from the fold model that never saw the image) and
    one small head per fold mapping the batch's pooled embeddings (mean and max) to the predicted
    batch accuracy of each classifier. Batches must lie within one fold (--group-by-fold)."""

    def __init__(self, embed, fold, heads, cols):
        self.embed, self.fold, self.heads, self.cols = embed, fold, heads, cols

    def batch(self, bt):
        bt = np.asarray(bt)
        f = self.fold[bt]
        require((f == f[:, :1]).all(), "Set router batches must stay within one fold (use --group-by-fold)")
        e = self.embed[bt]
        pooled = np.concatenate([e.mean(1), e.max(1)], axis=1)
        out = np.empty((len(bt), len(self.cols)))
        for g in np.unique(f[:, 0]):
            rows = f[:, 0] == g
            w1, b1, w2, b2 = self.heads[int(g)]
            h = np.maximum(pooled[rows] @ w1.T + b1, 0)
            out[rows] = (1 / (1 + np.exp(-(h @ w2.T + b2))))[:, self.cols]
        return out.T  # models x n_batches


def batch_scores(prob, bt):
    """models x n_batches predicted batch accuracy: the batch mean of per-image P(correct), or a
    set router's batch score."""
    if isinstance(prob, SetScorer):
        return prob.batch(bt)
    return prob[:, bt].mean(2)


def load_folds(paths, index):
    """Cross-fit fold id per requested image, from the score files (they must agree)."""
    fold = None
    for path in paths:
        data = np.load(path, allow_pickle=False)
        if "fold" not in data.files:
            continue
        pos = {int(k): i for i, k in enumerate(data["index"])}
        f = np.array([int(data["fold"][pos[int(k)]]) for k in index])
        require(fold is None or np.array_equal(fold, f), f"{path}: fold ids differ between score files")
        fold = f
    require(fold is not None, "--group-by-fold needs cross-fit score files (with fold ids)")
    return fold


def load_scores(path, names, index):
    data = np.load(path, allow_pickle=False)
    order = [str(x) for x in data["models"]]
    require(set(names) <= set(order), f"{path}: score file lacks some of the classifiers")
    if "set_embed" in data.files:
        pos = {int(k): i for i, k in enumerate(data["index"])}
        require(set(int(k) for k in index) <= set(pos), f"{path}: score file lacks some of the requested images")
        rows = np.array([pos[int(k)] for k in index])
        heads = {int(g): (data["set_w1"][g], data["set_b1"][g], data["set_w2"][g], data["set_b2"][g])
                 for g in range(len(data["set_w1"]))}
        prob = SetScorer(data["set_embed"][rows].astype(float), data["fold"][rows].astype(int), heads,
                         [order.index(n) for n in names])
        return prob, router_costs(data)
    prob = data["correct_prob"].astype(float)
    require(prob.shape == (len(data["index"]), len(order)), f"{path}: bad correct_prob shape")
    require(np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1, f"{path}: probabilities out of range")
    pos = {int(k): i for i, k in enumerate(data["index"])}
    # A score file may cover more images (e.g. cross-fit out-of-fold scores on the 5k pool);
    # only the requested images are used.
    require(set(int(k) for k in index) <= set(pos), f"{path}: score file lacks some of the requested images")
    rows = np.array([pos[int(k)] for k in index])
    prob = prob[rows][:, [order.index(n) for n in names]].T  # models x n, stage A order
    return prob, router_costs(data)


def router_costs(data):
    router = {o: {} for o in OBJECTIVES}
    if "router_cost_batches" in data.files:
        for o, key in (("latency", "router_latency_ms"), ("energy", "router_energy_mj")):
            for b, v in zip(data["router_cost_batches"].tolist(), data[key].tolist()):
                router[o][int(b)] = float(v)
    return router


def evaluate_choices(correct, batchings, choices, costs_b):
    """Mean accuracy (%) and mean total cost over batchings for per-batch model choices."""
    acc, cost = [], []
    for batches, ch in zip(batchings, choices):
        acc.append(100 * correct[ch[:, None], batches].mean())
        cost.append(costs_b[ch].sum())
    return float(np.mean(acc)), float(np.mean(cost)), np.asarray(acc)


def lock_router(prob, correct, batchings, costs_b, ref, target_acc, lambdas=LAMBDAS, min_share=0.0):
    """Cheapest lam whose mean calibration accuracy is >= target_acc AND whose accuracy is >=
    target_acc in at least `min_share` of the shuffled batchings. None if infeasible."""
    norm = costs_b / costs_b[ref]
    mean_prob = [batch_scores(prob, batches) for batches in batchings]  # models x n_batches
    best = None
    for lam in lambdas:
        choices = [(mp - lam * norm[:, None]).argmax(0) for mp in mean_prob]
        acc, cost, per = evaluate_choices(correct, batchings, choices, costs_b)
        if acc >= target_acc and (per >= target_acc).mean() >= min_share - 1e-12 and (best is None or cost < best["cost"] - 1e-12):
            share = np.bincount(np.concatenate(choices), minlength=len(costs_b))
            best = dict(lam=float(lam), accuracy=acc, cost=cost, per_repeat_accuracy=per,
                        share=share / share.sum())
    return best


def matched_random(share, correct, batchings, costs_b, rng):
    choices = [rng.choice(len(share), size=len(b), p=share) for b in batchings]
    return evaluate_choices(correct, batchings, choices, costs_b)


def lock(args):
    a = restrict(load_stage_a(args.stage_a, tuple(args.roles)), args.models)
    names, correct, labels = a["names"], a["correct"], a["label"]
    ref = names.index(REFERENCE)
    routers = {}
    for path in args.scores:
        routers[Path(path).stem] = load_scores(path, names, a["index"])
    rng = np.random.default_rng(args.seed)
    groups = load_folds(args.scores, a["index"]) if args.group_by_fold else None
    policy = dict(meta=dict(stage=f"B2 policy lock on {a['pool']}; test not read", roles=list(a["roles"]),
                            budget_pp=args.budget_pp, repeats=args.repeats, seed=args.seed,
                            classifier_costs="A40 TensorRT FP16 pipeline (stage A, job %s)" % a["meta"]["device"]["job"],
                            models=names, reference=REFERENCE, lambdas=len(LAMBDAS),
                            min_repeat_share=args.min_repeat_share,
                            classifiers=names, batches_within_fold=bool(args.group_by_fold),
                            batches="shuffled" if args.coherent is None else f"class-coherent ({args.coherent} classes per batch)"),
                  locked={})
    lines = [f"# Job 3 stage B2: policies locked on {a['pool']}", "",
             "CALIBRATION (policy locking), not held-out results. Classifier correctness = deployed TensorRT "
             f"FP16 GPU-pipeline predictions; costs = A40 pipeline medians from stage A. Budget: within "
             f"-{args.budget_pp} pp of fixed ResNet101 on the same {args.repeats} "
             + ("shuffled" if args.coherent is None else f"class-coherent ({args.coherent} classes per batch)")
             + " batchings per B. "
             "Savings are vs fixed ResNet101 TensorRT FP16 (not PyTorch autocast).", ""]
    summary = {}
    for b in BATCHES:
        batchings = batchings_for(labels, b, rng, args.repeats, args.coherent, groups)
        if batchings is None:
            lines += [f"## B={b}: skipped (cannot form batches of {args.coherent} classes x {b // args.coherent} images)", ""]
            continue
        nb = len(batchings[0])
        fixed_acc = np.array([np.mean([100 * correct[j][bt].mean() for bt in batchings]) for j in range(len(names))])
        target = fixed_acc[ref] - args.budget_pp
        eligible = [j for j in range(len(names)) if fixed_acc[j] >= target]
        policy.setdefault("fixed_accuracy", {})[str(b)] = {n: float(fixed_acc[j]) for j, n in enumerate(names)}
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
            policy.setdefault("oracle", {})[f"{objective}|{b}"] = dict(accuracy=float(np.mean(oracle_acc)),
                                                                     saving=float(save(np.mean(oracle_cost))))
            lines.append(f"| {objective} | oracle (labels known, bound) | - | {np.mean(oracle_acc):.2f} | - | "
                         f"{save(np.mean(oracle_cost)):.1f}% | no | context only | - |")
            lines.append(f"| {objective} | fixed {names[cheapest_fixed]} | - | {fixed_acc[cheapest_fixed]:.2f} | - | "
                         f"{save(fixed_cost):.1f}% | - | fallback baseline | - |")
            for rname, (prob, rcost) in routers.items():
                res = lock_router(prob, correct, batchings, cb, ref, target, min_share=args.min_repeat_share)
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
                srow.append(dict(routed=routed, saving=save(total), accuracy=res["accuracy"],
                                 router_latency=rcost["latency"].get(b)))
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
        # Pre-registered winner rule (fixed before the costed lock was seen).
        fams = sorted({f for f, _, _ in summary} - set(args.exclude_from_selection))
        ranking = []
        for f in fams:
            cells = [rows for (ff, _, _), rows in summary.items() if ff == f]
            per_cell = [float(np.median([r["saving"] if (r and r["routed"]) else 0.0 for r in rows]))
                        if all(r and r["routed"] for r in rows) else 0.0 for rows in cells]
            full = sum(all(r and r["routed"] for r in rows) for rows in cells)
            rl = [r["router_latency"] for rows in cells for r in rows if r and r.get("router_latency") is not None]
            ranking.append(dict(family=f, score=float(np.mean(per_cell)), cells_routed_all_seeds=full,
                                cells=len(cells), mean_router_ms=float(np.mean(rl)) if rl else None))
        ranking.sort(key=lambda r: (-r["score"], r["mean_router_ms"] if r["mean_router_ms"] is not None else 0.0))
        qualified = [r for r in ranking if r["cells_routed_all_seeds"] > 0]
        policy["selection"] = dict(
            rule="score = mean over (objective, B) cells of the median-over-seeds calibration saving, "
                 f"{cost_note}; a cell counts 0 unless all seeds ROUTE; feasibility requires the "
                 f"mean and >= {args.min_repeat_share:.0%} of repeats within budget; ties -> cheaper router; "
                 f"excluded (controls): {', '.join(args.exclude_from_selection) or 'none'}",
            ranking=ranking, winner=qualified[0]["family"] if qualified else None,
            runners_up=[r["family"] for r in qualified[1:]])
        lines += ["## Pre-registered winner selection", "", policy["selection"]["rule"] + ".", "",
                  "| rank | router | score (mean saving %) | cells routed on all seeds | mean router ms |",
                  "|---|---|---|---|---|"]
        for i, r in enumerate(ranking, 1):
            ms = "-" if r["mean_router_ms"] is None else f"{r['mean_router_ms']:.3f}"
            lines.append(f"| {i} | {r['family']} | {r['score']:.2f} | {r['cells_routed_all_seeds']}/{r['cells']} | {ms} |")
        win = policy["selection"]["winner"]
        lines += ["", f"WINNER (whole-batch): {win}" if win else "No router qualifies: FALLBACK to fixed ResNet101.",
                  "Runners-up go to the held-out run labelled as runners-up." if win else "", ""]
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
    p.add_argument("--roles", nargs="+", default=["router_calibration"],
                   help="stage A roles to lock on; router_train router_calibration = the 5k cross-fit pool")
    p.add_argument("--min-repeat-share", type=float, default=0.8,
                   help="locked lam must meet the budget in at least this share of repeats")
    p.add_argument("--coherent", type=int, default=None,
                   help="class-coherent batches: each batch holds images of this many classes (non-i.i.d. stream)")
    p.add_argument("--models", nargs="*", default=None,
                   help="restrict the router (and fixed baselines) to these classifiers; must include resnet101")
    p.add_argument("--group-by-fold", action="store_true",
                   help="draw every batch within one cross-fit fold (required for set routers)")
    p.add_argument("--exclude-from-selection", nargs="*", default=["lookup-res32-t48k"],
                   help="control variants reported but never selected as the winner")
    lock(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
