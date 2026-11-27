"""Job 3: routers vs random mixing of fixed classifiers, budget sweep, bootstrap CIs (CPU).

CALIBRATION only (router-calibration 2k); never reads test data. Inputs are stage A
(TensorRT GPU-pipeline predictions, A40 pipeline costs) and router score files from stage B1
(accuracy-only scores, or the costed ones once the A40 cost phase has run).

Random mix (no router): each batch goes to one fixed classifier drawn at random with fixed
mixing probabilities. Its expected (cost, accuracy) lies on the segment between fixed models, so
the best mix within a budget mixes at most two models (lower convex hull). Every pair is searched
on a 0.5% weight grid with common random numbers, and locked with the same rule as the routers:
mean accuracy within the budget of fixed ResNet101 on the same shuffled batchings AND within it
in >= --min-repeat-share of them; the cheapest feasible mix wins.

Routers are re-locked here on the same batchings as the random mix (so numbers can differ slightly
from hpc.a40_policy's report, which draws its own batchings), at budgets --budgets. A router cell
counts as ROUTE only if it is cheaper than the cheapest eligible fixed classifier at that budget,
router cost included when the score file carries it.

Bootstrap (--boot reps): resample calibration images with replacement, re-batch, and evaluate the
policies locked at the 1 pp budget (lam or mix fixed). 95% percentile intervals of the saving vs
ResNet101 and of the accuracy difference vs ResNet101. They cover image sampling, not lam selection.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

from hpc.a40_policy import BATCHES, LAMBDAS, OBJECTIVES, REFERENCE, load_scores, load_stage_a
from hpc.complementarity_analysis import make_batches

WEIGHTS = np.linspace(0, 1, 201)


def counts_for(correct, batchings):
    """Per batching: models x n_batches correct counts."""
    return [correct[:, bt].sum(2).astype(float) for bt in batchings]


def router_path(prob, counts, batchings, norm):
    """Accuracy per repeat (L x R) and per-repeat choices' cost index along the lam grid."""
    mean_prob = [prob[:, bt].mean(2) for bt in batchings]
    acc = np.zeros((len(LAMBDAS), len(batchings)))
    choices = []
    for li, lam in enumerate(LAMBDAS):
        ch_l = []
        for r, (mp, cnt) in enumerate(zip(mean_prob, counts)):
            ch = (mp - lam * norm[:, None]).argmax(0)
            acc[li, r] = 100 * cnt[ch, np.arange(len(ch))].sum() / batchings[r].size
            ch_l.append(ch)
        choices.append(ch_l)
    return acc, choices


def cost_of(choices_list, cb):
    return np.array([np.mean([cb[ch].sum() for ch in chs]) for chs in choices_list])


def lock_index(acc, cost, target, min_share):
    ok = (acc.mean(1) >= target) & ((acc >= target).mean(1) >= min_share - 1e-12)
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    return int(idx[np.argmin(cost[idx])])


def random_mix(counts, batchings, cb, target, min_share, uniforms, m):
    """Cheapest feasible two-model (or single-model) random mix. Returns dict or None."""
    best = None
    pairs = [(i, i) for i in range(m)] + list(combinations(range(m), 2))
    for i, j in pairs:
        # weight w on model i, 1 - w on model j; common random numbers per batch
        acc = np.zeros((len(WEIGHTS), len(batchings)))
        cost = np.zeros(len(WEIGHTS))
        for r, (cnt, u) in enumerate(zip(counts, uniforms)):
            take_i = u[None, :] < WEIGHTS[:, None]  # W x nb
            ch = np.where(take_i, i, j)
            acc[:, r] = 100 * cnt[ch, np.arange(cnt.shape[1])[None, :]].sum(1) / batchings[r].size
            cost += cb[ch].sum(1) / len(batchings)
        k = lock_index(acc, cost, target, min_share)
        if k is not None and (best is None or cost[k] < best["cost"] - 1e-12):
            w = float(WEIGHTS[k])
            best = dict(models=(i, j), weight_i=w, cost=float(cost[k]), accuracy=float(acc[k].mean()),
                        repeats_meeting=float((acc[k] >= target).mean()))
    return best


def run(args):
    a = load_stage_a(args.stage_a)
    names, correct, labels = a["names"], a["correct"], a["label"]
    m, ref = len(names), names.index(REFERENCE)
    routers = {Path(p).stem: load_scores(p, names, a["index"]) for p in args.scores}
    family = lambda k: k.rsplit("-s", 1)[0] if k.rsplit("-s", 1)[-1].isdigit() else k
    costed = all(bool(rc["latency"]) for _, rc in routers.values()) if routers else False
    report = dict(meta=dict(label="CALIBRATION baselines (router-calibration 2k); test not read",
                            budgets=args.budgets, repeats=args.repeats, min_repeat_share=args.min_repeat_share,
                            boot=args.boot, router_cost_included=costed, models=names), cells=[])
    lines = ["# Job 3: routers vs random mixing of fixed classifiers (calibration)", "",
             "CALIBRATION, not held-out. Random mix = each batch goes to a fixed classifier drawn at random "
             "(best two-model mix on the lower convex hull), no router. Same shuffled batchings, budget rule "
             f"(mean and >= {args.min_repeat_share:.0%} of {args.repeats} repeats) and A40 TensorRT pipeline "
             "costs for every policy. Savings vs fixed ResNet101. Router cost "
             + ("included." if costed else "NOT included (accuracy-only scores; counted as zero)."), ""]
    for b in BATCHES:
        rng = np.random.default_rng([args.seed, b])
        batchings = [make_batches(labels, b, rng) for _ in range(args.repeats)]
        nb = len(batchings[0])
        counts = counts_for(correct, batchings)
        uniforms = [rng.random(nb) for _ in batchings]
        fixed_acc = np.array([np.mean([100 * c[j].sum() / bt.size for c, bt in zip(counts, batchings)])
                              for j in range(m)])
        for objective in OBJECTIVES:
            cb = a["costs"][objective][b]
            norm = cb / cb[ref]
            ref_cost = cb[ref] * nb
            save = lambda c: 100 * (1 - c / ref_cost)
            rpaths = {}
            for k, (prob, rc) in routers.items():
                acc, chs = router_path(prob, counts, batchings, norm)
                overhead = (rc[objective].get(b) or 0.0) * nb
                rpaths[k] = (acc, cost_of(chs, cb) + overhead, chs)
            for budget in args.budgets:
                target = fixed_acc[ref] - budget
                eligible = [j for j in range(m) if fixed_acc[j] >= target]
                best_fixed = min(eligible, key=lambda j: cb[j])
                mix = random_mix(counts, batchings, cb, target, args.min_repeat_share, uniforms, m)
                cell = dict(batch=b, objective=objective, budget_pp=budget, target=float(target),
                            best_fixed=names[best_fixed], best_fixed_saving=save(cb[best_fixed] * nb),
                            random_mix=None if mix is None else dict(
                                models=[names[mix["models"][0]], names[mix["models"][1]]],
                                weight_first=mix["weight_i"], saving=save(mix["cost"]),
                                accuracy=mix["accuracy"], repeats_meeting=mix["repeats_meeting"]),
                            routers={})
                for k, (acc, cost, chs) in rpaths.items():
                    li = lock_index(acc, cost, target, args.min_repeat_share)
                    if li is None:
                        cell["routers"][k] = dict(decision="fallback", saving=None)
                        continue
                    routed = cost[li] < cb[best_fixed] * nb
                    share = np.bincount(np.concatenate(chs[li]), minlength=m) / sum(len(c) for c in chs[li])
                    cell["routers"][k] = dict(decision="route" if routed else "fallback", lam=float(LAMBDAS[li]),
                                              saving=save(cost[li]) if routed else None,
                                              accuracy=float(acc[li].mean()),
                                              repeats_meeting=float((acc[li] >= target).mean()),
                                              share={names[j]: round(float(s), 4) for j, s in enumerate(share) if s})
                # bootstrap at the 1 pp budget only
                if args.boot and abs(budget - 1.0) < 1e-9:
                    boot_rng = np.random.default_rng([args.seed, b, 7 if objective == "latency" else 11])
                    locked = {k: v.get("lam") for k, v in cell["routers"].items() if v["decision"] == "route"}
                    samples = {k: [] for k in list(locked) + (["random_mix"] if mix else [])}
                    for _ in range(args.boot):
                        draw = boot_rng.integers(0, len(labels), len(labels))
                        bl = labels[draw]
                        bts = [draw[make_batches(bl, b, boot_rng)] for _ in range(args.boot_batchings)]
                        bcounts = counts_for(correct, bts)
                        ref_acc = np.mean([100 * c[ref].sum() / bt.size for c, bt in zip(bcounts, bts)])
                        for k, lam in locked.items():
                            prob, rc = routers[k]
                            over = (rc[objective].get(b) or 0.0) * nb
                            accs, costs = [], []
                            for c, bt in zip(bcounts, bts):
                                ch = (prob[:, bt].mean(2) - lam * norm[:, None]).argmax(0)
                                accs.append(100 * c[ch, np.arange(len(ch))].sum() / bt.size)
                                costs.append(cb[ch].sum() + over)
                            samples[k].append((save(np.mean(costs)), np.mean(accs) - ref_acc))
                        if mix:
                            i, j = mix["models"]
                            accs, costs = [], []
                            for c, bt in zip(bcounts, bts):
                                ch = np.where(boot_rng.random(len(bt)) < mix["weight_i"], i, j)
                                accs.append(100 * c[ch, np.arange(len(ch))].sum() / bt.size)
                                costs.append(cb[ch].sum())
                            samples["random_mix"].append((save(np.mean(costs)), np.mean(accs) - ref_acc))
                    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                    cell["bootstrap"] = {k: dict(saving_ci=ci([s for s, _ in v]), acc_diff_ci=ci([d for _, d in v]))
                                         for k, v in samples.items()}
                report["cells"].append(cell)
    # ---------------- markdown
    fams = sorted({family(k) for k in routers})
    fmt = lambda v: "-" if v is None else f"{v:.1f}%"
    cells = report["cells"]
    get = lambda b, o, bud: next(c for c in cells if c["batch"] == b and c["objective"] == o and c["budget_pp"] == bud)

    def fam_stats(cell, f):
        rows = [v for k, v in cell["routers"].items() if family(k) == f]
        routed = [v for v in rows if v["decision"] == "route"]
        med = float(np.median([v["saving"] for v in routed])) if len(routed) == len(rows) and rows else None
        return med, len(routed), len(rows)

    lines += ["## Routers vs random mix at the 1 pp budget", "",
              "Router saving = median over training seeds, shown only when every seed routes. "
              "\"vs mix\" = router minus random-mix saving (percentage points); BEATS when positive and the "
              "router's bootstrap saving interval for every seed lies above the mix's point saving.", ""]
    for objective in OBJECTIVES:
        lines += [f"### {objective}", "",
                  "| B | random mix (models, weight) | mix saving | mix 95% CI | " + " | ".join(fams) + " |",
                  "|---|---|---|---|" + "---|" * len(fams)]
        for b in BATCHES:
            c = get(b, objective, 1.0)
            mx = c["random_mix"]
            mtxt = "-" if mx is None else f"{mx['models'][0]} {mx['weight_first']:.2f} / {mx['models'][1]}"
            msave = None if mx is None else mx["saving"]
            mci = c.get("bootstrap", {}).get("random_mix")
            cols = []
            for f in fams:
                med, r, n = fam_stats(c, f)
                if med is None:
                    cols.append(f"fallback ({r}/{n})")
                    continue
                d = med - (msave or 0.0)
                cis = [c["bootstrap"][k]["saving_ci"] for k in c.get("bootstrap", {}) if family(k) == f]
                beats = d > 0 and len(cis) == n and all(lo > (msave or 0.0) for lo, _ in cis)
                cols.append(f"{med:.1f}% ({d:+.1f}{' BEATS' if beats else ''})")
            ci_txt = "-" if not mci else "{:.1f}..{:.1f}".format(*mci["saving_ci"])
            lines.append(f"| {b} | {mtxt} | {fmt(msave)} | {ci_txt} | " + " | ".join(cols) + " |")
        lines.append("")
    lines += ["## Saving vs budget (median over seeds; '-' = a seed falls back to the best-fixed row at that budget)", ""]
    for objective in OBJECTIVES:
        lines += [f"### {objective}", "", "| policy | B | " + " | ".join(f"{x:g} pp" for x in args.budgets) + " |",
                  "|---|---|" + "---|" * len(args.budgets)]
        for b in BATCHES:
            lines.append(f"| best fixed | {b} | " + " | ".join(
                f"{get(b, objective, x)['best_fixed']} {get(b, objective, x)['best_fixed_saving']:.1f}%"
                for x in args.budgets) + " |")
            lines.append(f"| random mix | {b} | " + " | ".join(
                fmt(None if get(b, objective, x)["random_mix"] is None else get(b, objective, x)["random_mix"]["saving"])
                for x in args.budgets) + " |")
            for f in fams:
                lines.append(f"| {f} | {b} | " + " | ".join(fmt(fam_stats(get(b, objective, x), f)[0])
                                                          for x in args.budgets) + " |")
        lines.append("")
    lines += ["## Bootstrap 95% intervals at 1 pp (per router seed; image resampling, lam fixed)", "",
              "| objective | B | policy | saving CI | accuracy diff vs ResNet101 CI (pp) |", "|---|---|---|---|---|"]
    for c in cells:
        for k, v in sorted(c.get("bootstrap", {}).items()):
            lines.append(f"| {c['objective']} | {c['batch']} | {k} | {v['saving_ci'][0]:.1f}..{v['saving_ci'][1]:.1f}% | "
                         f"{v['acc_diff_ci'][0]:+.2f}..{v['acc_diff_ci'][1]:+.2f} |")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baselines.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    (args.output / "baselines.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--scores", type=Path, nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budgets", type=float, nargs="*", default=[0.5, 1.0, 2.0, 3.0])
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--min-repeat-share", type=float, default=0.8)
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--boot-batchings", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
