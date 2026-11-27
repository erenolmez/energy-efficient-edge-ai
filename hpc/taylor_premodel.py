"""Development screening (CPU): reproduction of the Taylor et al. premodel as a baseline router.

Papers: B. Taylor, V. Sanz Marco, W. Wolff, Y. Elkhatib, Z. Wang, "Adaptive Deep Learning Model
Selection on Embedded Systems", LCTES 2018 (arXiv 1805.04252), and the journal extension
"Optimizing Deep Learning Inference on Embedded Systems Through Adaptive Model Selection",
ACM TECS 2019 (arXiv 1911.04946). Authors' artefact: github.com/qwerybot/Adaptive_Deep_Learning
(premodel chain in code/server/features.py; it ships precomputed features, not the extractor).

What the papers do (per image, ImageNet, Jetson TX2):
  * 29 candidate features (Table 2): n_keypoints, avg_brightness, brightness_rms,
    avg_perceived_brightness, perceived_brightness_rms, contrast, edge_length{1-7},
    edge_angle{1-7}, area_by_perim and aspect_ratio of the main object, hue{1-7}.
  * Feature selection: drop features with |Pearson r| > 0.75 to a kept one (29 -> 17), then
    greedy backward elimination by K-fold cross-validated premodel accuracy down to 7 features.
    Paper's final 7: n_keypoints, avg_perceived_brightness, hue1, contrast, area_by_perim,
    edge_length1, aspect_ratio. Min-max scaling with training-set min/max.
  * DNN set (Algorithm 1, theta = 0.5 pp): first the model that is optimal (cheapest correct) for
    the most training images, then repeatedly the model most accurate on the images the current
    set gets wrong, while the gain is >= theta. ImageNet result: MobileNet -> Inception_v4 ->
    ResNet_v1_152.
  * Premodel: a chain of binary KNN classifiers (k=5, Euclidean, uniform vote), one per DNN in
    cost order. KNN-j decides whether DNN j should classify the image; if no, go to KNN-(j+1).
    If every level says no: use a pre-specified model (or report failure).
  * Target: the optimal model = the one meeting the accuracy goal (top-1) in least time. In the
    artefact, each level is trained on all training images with its own 0/1 column.

Minimal adaptations to our setup (stated in the outputs):
  * CIFAR-100 32x32 instead of ImageNet: the extractor is re-implemented with numpy/scipy because
    the authors' OpenCV/SimpleCV extractor is not released. Keypoints = scale-normalised
    determinant-of-Hessian maxima (the SURF detector, Gaussian instead of box filters) with a
    response threshold taken from the fit set; edges = gradient-magnitude pixels (> 0.1), edge
    length = 8-connected segment size, edge angle = gradient orientation; main object = largest
    Otsu component on the side touching fewer border pixels; hue histogram over pixels with
    saturation > 0.1. All thresholds and bin edges are fixed here or fit on val+3k only.
  * Costs: "least time" uses the A40 TensorRT FP16 per-batch latency (or energy) at B=64 from stage
    A, since we run batches, not single images. The chain is built per objective.
  * Level target: "correct" (model j right; artefact-consistent, primary) and "optimal" (model j
    is the cheapest correct model) are both run. All-reject -> ResNet101 (pre-specified model).
  * Fit on val+3k (classifier-validation 5k + router-train 3k); CV for feature selection inside
    val+3k only (5 folds, fold shuffles for seeds 42/45/48). Scored on router-calibration 2k.
  * Per batch (our deployment): (a) chain-plurality: each image runs the chain, the batch uses the
    model most images chose (ties -> costlier); (b) chain-max: the costliest model any image
    chose; (c) the same k=5 KNN on all eight models gives per-image P(correct) = vote share, which
    is written in the hpc.a40_policy score format and locked with the same 1 pp rule as the other
    routers. Router latency = measured single-thread CPU feature extraction + KNN per batch on this
    node; router energy is written as 0 (GPU energy excludes the CPU router, a lower bound).

Images are native 32x32 CIFAR-100 train pixels (cifar-100-python pickle under --data; the
evidence images.npy is the 128px resize and is not used). Correctness targets for fitting come
from router_arch_screen_v1/evidence (as for every other
router); calibration accuracy for policies comes from stage A's deployed TensorRT predictions.
The official test set is never opened (the split's namespace is official_train only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from hpc.a40_policy import BATCHES, REFERENCE, evaluate_choices, load_stage_a
from hpc.complementarity_analysis import make_batches
from hpc.feature_router_ablation import load_evidence, require, score, sha256

STAGE = "development screening"
SEEDS = (42, 45, 48)
K = 5
THETA_PP = 0.5
PCC_MAX = 0.75
KEEP = 7
FOLDS = 5
COST_B = 64
EDGE_LEN_BINS = np.array([1, 2, 3, 5, 8, 13, 21, np.inf])
FEATURES = (["n_keypoints", "avg_brightness", "brightness_rms", "avg_perceived_brightness",
             "perceived_brightness_rms", "contrast"] + [f"edge_length{i}" for i in range(1, 8)]
            + [f"edge_angle{i}" for i in range(1, 8)] + ["area_by_perim", "aspect_ratio"]
            + [f"hue{i}" for i in range(1, 8)])
PAPER7 = ["n_keypoints", "avg_perceived_brightness", "hue1", "contrast", "area_by_perim",
          "edge_length1", "aspect_ratio"]


# ---------------------------------------------------------------- features (29, Table 2)
def hessian_maxima(gray):
    """Scale-normalised det-of-Hessian responses at 3x3x3 local maxima (SURF-style detector)."""
    from scipy import ndimage
    stack = []
    for s in (1.2, 1.6, 2.0, 2.8):
        dxx = ndimage.gaussian_filter(gray, s, order=(0, 2))
        dyy = ndimage.gaussian_filter(gray, s, order=(2, 0))
        dxy = ndimage.gaussian_filter(gray, s, order=(1, 1))
        stack.append((dxx * dyy - dxy ** 2) * s ** 4)
    r = np.stack(stack)
    peak = (r == ndimage.maximum_filter(r, size=3, mode="nearest")) & (r > 0)
    peak[:, :2], peak[:, -2:], peak[:, :, :2], peak[:, :, -2:] = False, False, False, False
    return r[peak]


def otsu(values):
    hist, edges = np.histogram(values, bins=64, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    between = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return edges[1:][np.argmax(between)]


def raw_features(image, kp_threshold):
    from scipy import ndimage
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    plain = rgb.mean(2)
    perceived = np.sqrt(0.241 * r ** 2 + 0.691 * g ** 2 + 0.068 * b ** 2)
    kp = int((hessian_maxima(gray) > kp_threshold).sum()) if kp_threshold is not None else 0

    dy, dx = np.gradient(gray)
    mag = np.hypot(dx, dy)
    edges = mag > 0.1
    seg, n_seg = ndimage.label(edges, structure=np.ones((3, 3)))
    lengths = np.bincount(seg.ravel())[1:] if n_seg else np.zeros(0)
    len_hist = np.histogram(lengths, bins=EDGE_LEN_BINS)[0] / max(n_seg, 1)
    ang = np.mod(np.arctan2(dy, dx), np.pi)[edges]
    ang_hist = np.histogram(ang, bins=7, range=(0, np.pi))[0] / max(ang.size, 1)

    # main object: largest Otsu component on the side that touches fewer border pixels
    fg = gray > otsu(gray)
    border = np.ones_like(fg)
    border[1:-1, 1:-1] = False
    if fg[border].mean() > 0.5:
        fg = ~fg
    lab, n = ndimage.label(fg, structure=np.ones((3, 3)))
    if n:
        main = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        area = main.sum()
        inner = ndimage.binary_erosion(main, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], border_value=0)
        perim = max((main & ~inner).sum(), 1)
        ys, xs = np.nonzero(main)
        aspect = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        area_by_perim = area / perim
    else:
        area_by_perim, aspect = 0.0, 1.0

    mx, mn = rgb.max(2), rgb.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-12), 0)
    delta = np.maximum(mx - mn, 1e-12)
    hue = np.select([mx == r, mx == g], [((g - b) / delta) % 6, (b - r) / delta + 2], (r - g) / delta + 4) / 6
    hue = hue[sat > 0.1]
    hue_hist = np.histogram(hue, bins=7, range=(0, 1))[0] / max(hue.size, 1)

    return np.concatenate([[kp, plain.mean(), np.sqrt((plain ** 2).mean()), perceived.mean(),
                            np.sqrt((perceived ** 2).mean()), gray.std()], len_hist, ang_hist,
                           [area_by_perim, aspect], hue_hist])


def keypoint_threshold(images, index, quantile=0.75):
    """Hessian response threshold = upper quartile of all local-maximum responses on the fit set."""
    pooled = np.concatenate([hessian_maxima(np.asarray(images[i], dtype=np.float64).dot(
        [0.299, 0.587, 0.114]) / 255.0) for i in index])
    return float(np.quantile(pooled, quantile)) if pooled.size else None


# ---------------------------------------------------------------- premodel
class MinMax:
    def __init__(self, x):
        self.lo, span = x.min(0), np.ptp(x, 0)
        self.span = np.where(span > 0, span, 1.0)

    def __call__(self, x):
        return (x - self.lo) / self.span


def neighbours(train_x, query_x, exclude_self=False):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=K + int(exclude_self), metric="euclidean", n_jobs=1).fit(train_x)
    idx = nn.kneighbors(query_x, return_distance=False)
    return idx[:, 1:] if exclude_self else idx


def level_targets(correct, chain, target):
    """n x len(chain) binary targets for each chain level."""
    y = correct[:, chain]
    if target == "optimal":  # chain is cost ordered: first correct level is optimal
        first = np.where(y.any(1), y.argmax(1), -1)
        y = (first[:, None] == np.arange(len(chain))[None, :]).astype(np.float32)
    return y


def chain_decide(votes, chain, fallback):
    """votes: n x len(chain) vote shares. First level with a majority 'yes' (>= 3 of 5) wins."""
    yes = votes > 0.5
    pick = np.where(yes.any(1), np.asarray(chain)[yes.argmax(1)], fallback)
    return pick, ~yes.any(1)


def build_chain(correct, cost, theta_pp=THETA_PP):
    """Algorithm 1: greedy, on fit-set correctness. Returns models in cost order and the trace."""
    order = np.argsort(cost)
    any_ok = correct.any(1)
    optimal = np.where(any_ok, order[correct[:, order].argmax(1)], -1)
    first = int(np.bincount(optimal[optimal >= 0], minlength=correct.shape[1]).argmax())
    chosen, trace = [first], [dict(model=first, coverage=float(correct[:, first].mean()))]
    while len(chosen) < correct.shape[1]:
        missed = ~correct[:, chosen].any(1)
        if not missed.any():
            break
        cand = [j for j in range(correct.shape[1]) if j not in chosen]
        best = max(cand, key=lambda j: correct[missed, j].mean())
        gain = 100 * (correct[:, chosen + [best]].any(1).mean() - correct[:, chosen].any(1).mean())
        if gain < theta_pp:
            break
        chosen.append(best)
        trace.append(dict(model=best, gain_pp=float(gain), coverage=float(correct[:, chosen].any(1).mean())))
    return sorted(chosen, key=lambda j: cost[j]), trace


def cv_premodel_accuracy(x, correct, chain, fallback, folds):
    hits = []
    for k in range(len(folds)):
        te, tr = folds[k], np.concatenate([f for i, f in enumerate(folds) if i != k])
        scale = MinMax(x[tr])
        nb = neighbours(scale(x[tr]), scale(x[te]))
        votes = level_targets(correct[tr], chain, "correct")[nb].mean(1)
        pick, _ = chain_decide(votes, chain, fallback)
        hits.append(correct[te, pick])
    return float(np.concatenate(hits).mean())


def select_features(x, correct, chain, fallback, seed):
    """|PCC| > 0.75 filter in Table 2 order, then greedy backward elimination to KEEP features."""
    std = x.std(0)
    keep = []
    for j in range(x.shape[1]):
        if std[j] == 0:
            continue
        if all(abs(np.corrcoef(x[:, j], x[:, i])[0, 1]) <= PCC_MAX for i in keep):
            keep.append(j)
    after_pcc = list(keep)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(x)), FOLDS)
    trace = []
    while len(keep) > KEEP:
        scores = {j: cv_premodel_accuracy(x[:, [i for i in keep if i != j]], correct, chain, fallback, folds)
                  for j in keep}
        drop = max(scores, key=scores.get)  # removing it hurts least
        keep.remove(drop)
        trace.append(dict(dropped=FEATURES[drop], cv_accuracy=scores[drop]))
    return keep, after_pcc, trace


# ---------------------------------------------------------------- run
def run(args):
    t0 = time.time()
    ev = load_evidence(args.base, args.evidence, args.data)
    names, correct, images, cal = ev["names"], ev["correct"].astype(bool), ev["images"], ev["cal"]
    fit = ev["fits"]["val3k"]
    a = load_stage_a(args.stage_a)
    require(sorted(a["names"]) == sorted(names), "Stage A and evidence models differ")
    perm = [names.index(n) for n in a["names"]]  # evidence order -> stage A order
    correct = correct[:, perm]
    names = a["names"]
    ref = names.index(REFERENCE)
    require(np.array_equal(np.sort(a["index"]), np.sort(cal)), "Stage A calibration images differ")
    require(np.array_equal(a["label"], ev["labels"][a["index"]]), "Stage A labels differ from evidence labels")
    if args.dry_run:
        print("DRY RUN (nothing written)")
        print("stage A models:", names, "calibration images:", len(a["index"]))
        for o in ("latency", "energy"):
            print(f"A40 {o} at B={COST_B}:", {n: round(float(c), 3) for n, c in zip(names, a["costs"][o][COST_B])})
        print("stage A TRT calibration accuracy:", {n: round(float(c), 4) for n, c in zip(names, a["correct"].mean(1))})
        print("evidence calibration accuracy:", {n: round(float(c), 4) for n, c in zip(names, correct[cal].mean(0))})
        thr = keypoint_threshold(images, fit[:200])
        x = np.stack([raw_features(images[i], thr) for i in cal[:20]])
        print("29 features on 20 images:", x.shape, "finite", bool(np.isfinite(x).all()))
        print("feature means:", dict(zip(FEATURES, np.round(x.mean(0), 4).tolist())))
        print("DRY RUN OK")
        return
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "scores").mkdir()

    kp_thr = keypoint_threshold(images, fit)
    t = time.time()
    every = np.concatenate([fit, cal])
    x_all = np.stack([raw_features(images[i], kp_thr) for i in every])
    require(np.isfinite(x_all).all(), "Nonfinite features")
    x_fit, x_cal = x_all[:len(fit)], x_all[len(fit):]
    np.savez_compressed(out / "features29.npz", index=every, x=x_all, names=np.array(FEATURES))
    print(f"29 features for {len(every)} images in {time.time() - t:.1f}s (keypoint thr {kp_thr})", flush=True)

    result = dict(stage=STAGE, test_set="never opened (split namespace official_train)",
                  fit="val3k", fit_size=len(fit), score_set="router_calibration", score_size=len(cal),
                  models=names, reference=REFERENCE, k=K, theta_pp=THETA_PP, pcc_max=PCC_MAX, keep=KEEP,
                  folds=FOLDS, cost_batch=COST_B, keypoint_threshold=kp_thr,
                  evidence_sha256=ev["hashes"], split_sha256=ev["split"]["sha256"],
                  paper="arXiv 1805.04252 / 1911.04946; artefact github.com/qwerybot/Adaptive_Deep_Learning",
                  feature_stats={n: dict(mean=float(x_fit[:, j].mean()), std=float(x_fit[:, j].std()))
                                 for j, n in enumerate(FEATURES)},
                  chains={}, feature_sets={}, brier={}, per_image={}, per_batch={})

    # chains per objective (Algorithm 1 on fit-set correctness, cost at B=64)
    chains = {}
    for objective in ("latency", "energy"):
        cost = a["costs"][objective][COST_B]
        chain, trace = build_chain(correct[fit], cost)
        chains[objective] = chain
        result["chains"][objective] = dict(models=[names[j] for j in chain],
                                           trace=[{**s, "model": names[s["model"]]} for s in trace])
        print(f"chain ({objective}): {' -> '.join(names[j] for j in chain)}", flush=True)

    # feature sets: paper's 7, and our selection per seed (latency chain, as the paper optimised time)
    sets = {"paper7": [FEATURES.index(f) for f in PAPER7]}
    for seed in SEEDS:
        t = time.time()
        keep, after_pcc, trace = select_features(x_fit, correct[fit], chains["latency"], ref, seed)
        sets[f"sel-s{seed}"] = keep
        result["feature_sets"][f"sel-s{seed}"] = dict(after_pcc=[FEATURES[j] for j in after_pcc],
                                                      selected=[FEATURES[j] for j in keep], trace=trace,
                                                      seconds=round(time.time() - t, 1))
        print(f"seed {seed}: {len(after_pcc)} after PCC -> {[FEATURES[j] for j in keep]}", flush=True)
    result["feature_sets"]["paper7"] = dict(selected=PAPER7)

    rng = np.random.default_rng(0)
    batchings = {b: [make_batches(a["label"], b, rng) for _ in range(args.repeats)] for b in BATCHES}
    pos = {int(k): i for i, k in enumerate(a["index"])}
    cal_rows = np.array([pos[int(k)] for k in cal])  # cal order -> stage A column
    stage_correct = a["correct"][:, cal_rows]  # models x n, in our cal order
    batchings = {b: [cal_rows.argsort()[bt] for bt in v] for b, v in batchings.items()}  # stage-A cols -> cal order

    for sname, cols in sets.items():
        scale = MinMax(x_fit[:, cols])
        zf, zc = scale(x_fit[:, cols]), scale(x_cal[:, cols])
        nb = neighbours(zf, zc)
        # (c) all-eight KNN vote shares -> P(correct), scored and written for hpc.a40_policy
        prob8 = correct[fit][nb].mean(1)
        result["brier"][f"taylor-knn8-{sname}"] = score(prob8, correct[cal].astype(np.float32), names)
        lat = router_latency(images, cal, cols, kp_thr, scale, zf)
        np.savez_compressed(out / "scores" / f"taylor-knn8-{sname}.npz", index=cal, models=np.array(names),
                            correct_prob=prob8.astype(np.float32), router_cost_batches=np.array(BATCHES),
                            router_latency_ms=np.array([lat[b] for b in BATCHES]),
                            router_energy_mj=np.zeros(len(BATCHES)))
        result.setdefault("router_latency_ms", {})[sname] = lat
        for objective, chain in chains.items():
            for target in ("correct", "optimal"):
                key = f"{sname}|{objective}|{target}"
                votes = level_targets(correct[fit], chain, target)[nb].mean(1)
                pick, none = chain_decide(votes, chain, ref)
                share = np.bincount(pick, minlength=len(names)) / len(pick)
                result["per_image"][key] = dict(
                    accuracy=float(100 * correct[cal, pick].mean()), all_reject=float(none.mean()),
                    oracle_chain=float(100 * correct[cal][:, chain].any(1).mean()),
                    share={names[j]: float(s) for j, s in enumerate(share) if s > 0})
                result["per_batch"][key] = batch_policies(pick, stage_correct, batchings, a["costs"][objective],
                                                          ref, lat, objective, names, args.budget_pp)
        print(f"{sname}: Brier knn8 {result['brier'][f'taylor-knn8-{sname}']['mean_brier']:.5f}", flush=True)

    result["seconds"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(result, indent=1, allow_nan=False, default=float) + "\n")
    (out / "summary.md").write_text(markdown(result))
    print(markdown(result))


def router_latency(images, cal, cols, kp_thr, scale, zf, reps=5):
    """Median single-thread CPU ms per batch: 29-feature extraction (all computed, then the subset)
    + scaling + KNN query, on calibration images."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=K, n_jobs=1).fit(zf)
    out = {}
    for b in BATCHES:
        ims = [np.asarray(images[i]).copy() for i in cal[:b]]
        ms = []
        for _ in range(reps):
            t = time.perf_counter_ns()
            x = np.stack([raw_features(im, kp_thr) for im in ims])[:, cols]
            nn.kneighbors(scale(x), return_distance=False)
            ms.append((time.perf_counter_ns() - t) / 1e6)
        out[b] = float(np.median(ms))
    return out


def batch_policies(pick, correct, batchings, costs, ref, lat, objective, names, budget_pp):
    """Chain-plurality and chain-max per batch on stage-A calibration correctness (models x n)."""
    res = {}
    for b, bts in batchings.items():
        cb = costs[b]
        nb = len(bts[0])
        fixed = np.array([np.mean([100 * correct[j][bt].mean() for bt in bts]) for j in range(len(names))])
        target = fixed[ref] - budget_pp
        eligible = [j for j in range(len(names)) if fixed[j] >= target]
        cheapest = min(eligible, key=lambda j: cb[j])
        ref_cost, fixed_cost = cb[ref] * nb, cb[cheapest] * nb
        overhead = (lat[b] if objective == "latency" else 0.0) * nb
        row = {"fixed_fallback": names[cheapest], "target_accuracy": float(target),
               "ref_accuracy": float(fixed[ref])}
        for rule in ("plurality", "max"):
            choices = []
            for bt in bts:
                p = pick[bt]
                if rule == "max":
                    choices.append(np.array([max(r, key=lambda j: cb[j]) for r in p]))
                else:
                    cnt = np.stack([np.bincount(r, minlength=len(names)) for r in p])
                    score_ = cnt + 1e-6 * (cb / cb.max())[None, :]  # ties -> costlier model
                    choices.append(score_.argmax(1))
            acc, cost, per = evaluate_choices(correct, bts, choices, cb)
            total = cost + overhead
            ok = acc >= target
            share = np.bincount(np.concatenate(choices), minlength=len(names))
            row[rule] = dict(accuracy=acc, repeats_meeting_target=float((per >= target).mean()),
                             saving_vs_rn101_pct=float(100 * (1 - total / ref_cost)),
                             decision="route" if ok and total < fixed_cost else "fallback",
                             router_cost_included=objective == "latency",
                             share={names[j]: float(s / share.sum()) for j, s in enumerate(share) if s})
        res[b] = row
    return res


def markdown(r):
    L = [f"# Taylor et al. premodel on the HPC bank ({STAGE})", "",
         "Fit on val+3k, scored on router-calibration 2k; test never opened. KNN k=5, Euclidean, "
         "min-max scaled. Minimal adaptations are listed in hpc/taylor_premodel.py.", "",
         "## Chains (Algorithm 1, theta 0.5 pp, A40 cost at B=64)", ""]
    for o, c in r["chains"].items():
        L.append(f"- {o}: " + " -> ".join(c["models"]))
    L += ["", "## Selected features", ""]
    for s, f in r["feature_sets"].items():
        L.append(f"- {s}: " + ", ".join(f["selected"]))
    L += ["", "## Correctness Brier, all-eight KNN vote shares (lower is better)", "",
          "| feature set | mean Brier |", "|---|---|"]
    for k, v in sorted(r["brier"].items(), key=lambda kv: kv[1]["mean_brier"]):
        L.append(f"| {k} | {v['mean_brier']:.5f} |")
    L += ["", "## Per-image chain (paper's native setting; accuracy on the 2k, evidence correctness)", "",
          "| feature set / objective / level target | accuracy % | all-reject % | chain oracle % | shares |",
          "|---|---|---|---|---|"]
    for k, v in r["per_image"].items():
        sh = ", ".join(f"{m} {100 * s:.0f}%" for m, s in sorted(v["share"].items(), key=lambda kv: -kv[1]))
        L.append(f"| {k} | {v['accuracy']:.2f} | {100 * v['all_reject']:.1f} | {v['oracle_chain']:.2f} | {sh} |")
    L += ["", "## Per batch (stage A TensorRT correctness and A40 costs; router latency included, "
          "router energy not)", "",
          "| feature set / objective / target | B | rule | accuracy | target | repeats meeting | saving vs RN101 | decision |",
          "|---|---|---|---|---|---|---|---|"]
    for k, rows in r["per_batch"].items():
        for b, row in rows.items():
            for rule in ("plurality", "max"):
                v = row[rule]
                dec = "ROUTE" if v["decision"] == "route" else f"FALLBACK {row['fixed_fallback']}"
                L.append(f"| {k} | {b} | {rule} | {v['accuracy']:.2f} | {row['target_accuracy']:.2f} | "
                         f"{100 * v['repeats_meeting_target']:.0f}% | {v['saving_vs_rn101_pct']:.1f}% | {dec} |")
    L += ["", "The all-eight KNN score files in scores/ are locked by hpc.a40_policy (policy/policy.md).", ""]
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="CIFAR-100 root (holds cifar-100-python/)")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="load and check every input, write nothing")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--budget-pp", type=float, default=1.0)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
