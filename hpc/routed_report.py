"""Routed accuracy, latency and energy on router-calibration (2k), CPU only, test never read.

Every router is locked with the hpc.a40_policy rule: per batch choose
argmax_j mean_i P(correct_ij) - lam * cost_j(B) / cost_ResNet101(B), with lam the cheapest value
whose mean accuracy over shuffled batchings stays within --budget-pp of fixed TensorRT ResNet101.
If no lam meets the budget, or the routed policy (router cost included) is not cheaper than the
cheapest eligible fixed classifier, the decision is FALLBACK and the fixed classifier's numbers
are what deploys. Accuracy = deployed TensorRT FP16 GPU-pipeline predictions (stage A); costs =
A40 pipeline medians per batch (stage A). For each locked policy both latency and energy are
reported, whichever objective it was locked for.

Router inputs:
  --scores           a40_policy-format npz files (TinyCNN/blended/lookup from a40_routers,
                     taylor_premodel_v1/scores). Router cost per batch is read from the file when
                     present; otherwise the router is costed at zero and the table says so.
  --feature-ablation feature_router_ablation_v1 directory: calibration probabilities of the
                     hand-made-feature routers. Router latency per batch = measured single-thread
                     CPU feature extraction (median ms/image x B) + predictor time per batch (both
                     from that run's results.json, HPC Sapphire Rapids CPU).
CPU routers (feature routers, and score files with router latency but zero router energy, e.g.
Taylor) are charged an estimated CPU energy = router ms x --cpu-watts (default 9.4 W: the Xeon
Gold 6442Y's 225 W TDP / 24 cores, one single-threaded core at full load; not measured). Any
policy that routes for energy while its latency is worse than fixed ResNet101 is labelled
ROUTE (energy only, latency +X%), because it would not save end-to-end time. This is calibration (policy
locking), not a held-out result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hpc.a40_policy import BATCHES, LAMBDAS, REFERENCE, evaluate_choices, load_scores, load_stage_a, require
from hpc.complementarity_analysis import make_batches

FEATURE_METHODS = ("texture_xgb_cond", "compact_hgb_cond", "compact_logistic")


def feature_routers(directory, names, index, methods, cpu_w):
    """name -> (models x n prob in stage A order, {'latency': {B: ms}, 'energy': {B: 0}}, note)."""
    directory = Path(directory)
    res = json.loads((directory / "results.json").read_text())
    z = np.load(directory / "calibration_probabilities.npz", allow_pickle=False)
    order = [str(m) for m in z["models"]]
    require(sorted(order) == sorted(names), "Feature-ablation models differ from stage A")
    pos = {int(k): i for i, k in enumerate(z["index"])}
    require(set(pos) == set(int(k) for k in index), "Feature-ablation images differ from router-calibration")
    rows = np.array([pos[int(k)] for k in index])
    cols = [order.index(n) for n in names]
    timing = res.get("timing", {})
    out = {}
    for key in z.files:
        if "__" not in key:
            continue
        method, rest = key.split("__", 1)
        fit, seed = rest.rsplit("-s", 1)
        if method not in methods:
            continue
        feat = res["methods"][method]["features"]
        per_img = timing["features"][feat]["median_ms_per_image"]
        pred = timing["predictor"][method]  # timed on the val+3k fit; same model size for 3k
        lat = {b: per_img * b + pred[str(b)]["median_ms_per_batch"] for b in BATCHES}
        name = f"{method}-{fit}" + ("" if seed == "None" else f"-s{seed}")
        out[name] = (z[key][rows][:, cols].T.astype(float).clip(0, 1),
                     {"latency": lat, "energy": {b: lat[b] * cpu_w for b in BATCHES}},
                     f"CPU features, energy est. {cpu_w:g} W")
    return out


def family(name):
    head, _, tail = name.rpartition("-s")
    return head if tail.isdigit() else name


def run(args):
    a = load_stage_a(args.stage_a)
    names, correct, labels, costs = a["names"], a["correct"], a["label"], a["costs"]
    ref = names.index(REFERENCE)
    routers = {}
    for path in args.scores:
        prob, rcost = load_scores(path, names, a["index"])
        has = bool(rcost["latency"])
        note = "A40 router cost" if has else "router cost NOT included"
        if has and not any(rcost["energy"].values()):  # CPU router: latency measured, energy not
            rcost["energy"] = {b: ms * args.cpu_watts for b, ms in rcost["latency"].items()}
            note = f"CPU router, energy est. {args.cpu_watts:g} W"
        routers[Path(path).stem] = (prob, rcost, note)
    if args.feature_ablation and Path(args.feature_ablation).is_dir():
        routers.update(feature_routers(args.feature_ablation, names, a["index"], args.feature_methods, args.cpu_watts))
    require(routers, "No routers given")
    rng = np.random.default_rng(args.seed)
    L = ["# Routed accuracy, latency and energy (router-calibration 2k, policy locking)", "",
         "CALIBRATION, not held out; test never read. Accuracy: deployed TensorRT FP16 predictions. "
         "Costs: A40 TensorRT pipeline medians (stage A) plus router cost. Savings are vs fixed TensorRT "
         f"ResNet101 on the same {args.repeats} shuffled batchings; budget {args.budget_pp} pp. FALLBACK rows "
         "show what deploys (the cheapest fixed classifier within budget), not a routing gain.", ""]
    summary = {}
    records = []
    for b in BATCHES:
        bts = [make_batches(labels, b, rng) for _ in range(args.repeats)]
        nb = len(bts[0])
        fixed_acc = np.array([np.mean([100 * correct[j][bt].mean() for bt in bts]) for j in range(len(names))])
        target = fixed_acc[ref] - args.budget_pp
        eligible = [j for j in range(len(names)) if fixed_acc[j] >= target]
        ref_cost = {o: costs[o][b][ref] * nb for o in costs}
        L += [f"## Batch {b}", "",
              f"ResNet101 accuracy {fixed_acc[ref]:.2f}%, target {target:.2f}%, eligible fixed: "
              + ", ".join(names[j] for j in eligible), "",
              "| router | locked for | decision | accuracy % | vs RN101 pp | latency saving | energy saving | "
              "repeats in budget | selections | router cost |", "|---|---|---|---|---|---|---|---|---|---|"]
        for o in ("latency", "energy"):
            j = min(eligible, key=lambda k: costs[o][b][k])
            L.append(f"| fixed {names[j]} (fallback) | {o} | fixed | {fixed_acc[j]:.2f} | {fixed_acc[j] - fixed_acc[ref]:+.2f} | "
                     f"{100 * (1 - costs['latency'][b][j] / costs['latency'][b][ref]):.1f}% | "
                     f"{100 * (1 - costs['energy'][b][j] / costs['energy'][b][ref]):.1f}% | - | - | - |")
        for rname, (prob, rcost, note) in routers.items():
            mean_prob = [prob[:, bt].mean(2) for bt in bts]
            for o in ("latency", "energy"):
                cb = costs[o][b]
                fixed = min(eligible, key=lambda k: cb[k])
                norm = cb / cb[ref]
                best = None
                for lam in LAMBDAS:
                    ch = [(mp - lam * norm[:, None]).argmax(0) for mp in mean_prob]
                    acc, cost, per = evaluate_choices(correct, bts, ch, cb)
                    if acc >= target and (best is None or cost < best[1] - 1e-12):
                        best = (lam, cost, acc, per, ch)
                rec = dict(router=rname, family=family(rname), objective=o, batch=b, fallback_model=names[fixed])
                if best is None:
                    rec.update(decision="fallback", reason="no lam meets the budget")
                else:
                    lam, _, acc, per, ch = best
                    over = {k: (rcost[k].get(b) or 0.0) * nb for k in ("latency", "energy")}
                    tot = {k: evaluate_choices(correct, bts, ch, costs[k][b])[1] + over[k] for k in ("latency", "energy")}
                    routed = tot[o] < cb[fixed] * nb
                    share = np.bincount(np.concatenate(ch), minlength=len(names)) / sum(len(c) for c in ch)
                    rec.update(decision="route" if routed else "fallback", lam=float(lam), accuracy=acc,
                               delta_pp=acc - fixed_acc[ref], repeats_in_budget=float((per >= target).mean()),
                               latency_saving=100 * (1 - tot["latency"] / ref_cost["latency"]),
                               energy_saving=100 * (1 - tot["energy"] / ref_cost["energy"]),
                               router_ms_per_batch=rcost["latency"].get(b),
                               share={names[k]: float(s) for k, s in enumerate(share) if s > 0},
                               reason=None if routed else "not cheaper than the fixed fallback, router cost included",
                               latency_regression=bool(routed and tot["latency"] > ref_cost["latency"]))
                records.append(rec)
                summary.setdefault((rec["family"], o, b), []).append(rec)
                if "accuracy" in rec:
                    sel = ", ".join(f"{m} {100 * s:.0f}%" for m, s in sorted(rec["share"].items(), key=lambda kv: -kv[1]))
                    dec = "ROUTE" if rec["decision"] == "route" else f"FALLBACK {names[fixed]}"
                    if rec["decision"] == "route" and o == "energy" and rec["latency_saving"] < 0:
                        dec = f"ROUTE (energy only, latency +{-rec['latency_saving']:.0f}%)"
                    L.append(f"| {rname} | {o} | {dec} | {rec['accuracy']:.2f} | {rec['delta_pp']:+.2f} | "
                             f"{rec['latency_saving']:.1f}% | {rec['energy_saving']:.1f}% | "
                             f"{100 * rec['repeats_in_budget']:.0f}% | {sel} | {note} |")
                else:
                    L.append(f"| {rname} | {o} | FALLBACK {names[fixed]} (budget infeasible) | - | - | - | - | - | - | {note} |")
        L.append("")
    L += ["## Summary per router family (median over training seeds; routed numbers only where it routes)", "",
          "| router | locked for | B | routes / seeds | accuracy % | vs RN101 pp | latency saving | energy saving | note |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (fam, o, b), rows in sorted(summary.items()):
        ok = [r for r in rows if r["decision"] == "route"]
        med = lambda k, f="{:.2f}": f.format(np.median([r[k] for r in ok])) if ok else "-"
        L.append(f"| {fam} | {o} | {b} | {len(ok)}/{len(rows)} | {med('accuracy')} | {med('delta_pp', '{:+.2f}')} | "
                 f"{med('latency_saving', '{:.1f}%')} | {med('energy_saving', '{:.1f}%')} | "
                 f"{'energy only: slower than RN101' if any(r.get('latency_regression') for r in ok) else ''} |")
    L += ["", f"CPU routers are charged router ms x {args.cpu_watts:g} W of estimated CPU energy (TDP share of one "
          "core, not measured); GPU idle power while the router runs is not added, so energy savings stay "
          "optimistic. 'Energy only' rows save GPU+CPU energy on paper but are slower than fixed ResNet101 "
          "end to end, so they are not a routing win. Footnote: correctness Brier is in each router's own "
          "results file.", ""]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "routed_report.json").write_text(json.dumps(records, indent=1, default=float) + "\n")
    (args.output / "routed_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--scores", type=Path, nargs="*", default=[])
    p.add_argument("--feature-ablation", type=Path)
    p.add_argument("--feature-methods", nargs="*", default=list(FEATURE_METHODS))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget-pp", type=float, default=1.0)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu-watts", type=float, default=225 / 24,
                   help="estimated CPU power of a single-threaded router (W); default Gold 6442Y TDP/cores")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
