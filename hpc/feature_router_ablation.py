"""Development screening (CPU only): hand-made image features vs the TinyCNN correctness router.

Question: do cheap hand-made descriptors predict, image by image, whether each of the eight
frozen HPC classifiers (ResNet18/34/50/101, MobileNetV3 Small/Large, EfficientNet-B0,
ShuffleNetV2) is correct as well as the TinyCNN 16x16 direct-correctness router does?

Data (all from router_arch_screen_v1/evidence, the same correctness targets the routers used):
  images.npy  (50000, 32, 32, 3) uint8, official CIFAR-100 *train* order
  labels.npy  (50000,)
  <architecture>.npy  per-model predicted class for all 50k train images
  split.json / bank.json, checked against <base>/split_seed42.json
The split namespace is official_train; the official 10k test set is never opened.

Fit sets:   val3k = classifier-validation 5k + router-train 3k (primary)
            3k    = router-train only (control, matches the screened TinyCNN's fit set)
Score set:  router-calibration 2k only.

Descriptors are the PC study's, unchanged (jetson_tests/scripts/router_features.py):
  compact  46 values: RGB/gray statistics, gradient/Laplacian, saturation, quadrant colour
           mean/std, center vs border
  texture 450 values: compact + RGB/gray histograms, HOG and LBP

Methods (binary correctness of each of the eight models):
  constant              each model's mean accuracy on the fit set (the floor to beat)
  <feat>_xgb_cond       PC recipe: one XGBClassifier on [features, one-hot model id]
                        (compact: depth 3 / 300 trees, texture: depth 4 / 400 trees)
  <feat>_hgb_cond       sklearn HistGradientBoosting on the same inputs (fallback / cross-check)
  compact_logistic      StandardScaler + one L2 logistic regression per model
  compact_knn           StandardScaler + 100-NN distance-weighted regression (PC recipe)
XGBoost (row/column subsampling) runs for seeds 42, 45, 48; HistGradientBoosting also runs all
three seeds but has no subsampling here, so its seeds should agree; logistic and kNN run once.

Score: mean correctness Brier over 2000 images x 8 models (lower is better), per-model Brier
and AUC. If --router-scores points at a40_routers_v1/scores, every router's calibration
P(correct) there (TinyCNN direct, blended, lookup) is scored with the same code and data.

Timing (optional, on by default): single-thread CPU time per image for feature extraction
(batch 1, numpy only) and predictor time per batch, measured on the Slurm node that runs this.
That is HPC CPU time, not an edge or A40 cost.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 45, 48)
BATCHES = (8, 16, 32, 64, 128)
STAGE = "development screening"
XGB_RECIPE = dict(learning_rate=0.04, min_child_weight=15, reg_lambda=10, reg_alpha=0.1,
                  subsample=0.85, colsample_bytree=0.85)
XGB_SIZE = {"compact": (3, 300), "texture": (4, 400)}  # (max_depth, trees), PC recipe
TINYCNN_SCREEN_REFERENCE = {"router": "TinyCNN 16x16 direct correctness, router-train 3k",
                            "mean_brier": 0.1596, "source": "router_arch_screen_v1 summary "
                            "(quoted, not recomputed here)"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_feature_module():
    path = ROOT / "jetson_tests" / "scripts" / "router_features.py"
    spec = importlib.util.spec_from_file_location("router_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def load_evidence(base, evidence):
    sys.path.insert(0, str(ROOT))
    from hpc.clean_protocol import validate_split

    split = json.loads((base / "split_seed42.json").read_text())
    ev_split = json.loads((evidence / "split.json").read_text())
    require(ev_split["sha256"] == split["sha256"], "Evidence split differs from split_seed42.json")
    labels = np.load(evidence / "labels.npy", allow_pickle=False)
    validate_split(split, labels.tolist())  # official_train namespace, no test indices exist
    bank = json.loads((evidence / "bank.json").read_text())
    names = [m["architecture"] for m in bank]
    require(len(names) == 8, f"Expected eight bank models, found {names}")
    preds = np.stack([np.load(evidence / f"{n}.npy", allow_pickle=False) for n in names], axis=-1)
    require(preds.shape == (len(labels), 8), f"Prediction shape {preds.shape}")
    correct = (preds == labels[:, None]).astype(np.float32)
    images = np.load(evidence / "images.npy", mmap_mode="r", allow_pickle=False)
    require(images.shape == (len(labels), 32, 32, 3) and images.dtype == np.uint8,
            f"Unexpected images.npy {images.shape} {images.dtype}")
    ids = {k: np.asarray(v, dtype=np.int64) for k, v in split["indices"].items()}
    fits = {"val3k": np.concatenate([ids["classifier_validation"], ids["router_train"]]),
            "3k": ids["router_train"]}
    cal = ids["router_calibration"]
    for f in fits.values():
        require(not np.intersect1d(f, cal).size, "Fit and calibration sets overlap")
    hashes = {f"{n}.npy": sha256(evidence / f"{n}.npy") for n in names}
    hashes["labels.npy"] = sha256(evidence / "labels.npy")
    return dict(split=split, names=names, correct=correct, images=images, fits=fits, cal=cal,
                hashes=hashes)


def extract(images, index, feature_set, fm):
    return np.stack([fm.extract_feature_vector(np.asarray(images[i]), feature_set) for i in index])


def conditional(x, m=8):
    return np.hstack([np.repeat(x, m, axis=0), np.tile(np.eye(m, dtype=np.float32), (len(x), 1))])


def make_methods(have_xgb, threads, quick):
    """name -> (feature_set, seeded, fit(x, y, seed) -> predict(x) -> n x 8 probabilities)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def xgb_cond(depth, trees):
        def fit(x, y, seed):
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=trees, max_depth=depth, objective="binary:logistic",
                                  tree_method="hist", n_jobs=threads, random_state=seed, **XGB_RECIPE)
            model.fit(conditional(x), y.ravel())
            model.set_params(n_jobs=1)
            return lambda z: model.predict_proba(conditional(z))[:, 1].reshape(-1, 8)
        return fit

    def hgb_cond(depth, trees):
        def fit(x, y, seed):
            model = HistGradientBoostingClassifier(
                max_iter=trees, max_depth=depth, learning_rate=XGB_RECIPE["learning_rate"],
                l2_regularization=XGB_RECIPE["reg_lambda"], min_samples_leaf=15,
                early_stopping=False, random_state=seed)
            model.fit(conditional(x), y.ravel())
            return lambda z: model.predict_proba(conditional(z))[:, 1].reshape(-1, 8)
        return fit

    def logistic(x, y, seed):
        heads = [make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=4000)).fit(x, c)
                 if 0 < c.mean() < 1 else float(c.mean()) for c in y.T]
        return lambda z: np.column_stack([np.full(len(z), h) if isinstance(h, float)
                                          else h.predict_proba(z)[:, 1] for h in heads])

    def knn(x, y, seed):
        model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=100, weights="distance",
                                                                     n_jobs=threads)).fit(x, y)
        return lambda z: model.predict(z)

    sizes = {k: ((d, 30) if quick else (d, t)) for k, (d, t) in XGB_SIZE.items()}
    methods = {}
    for feat in (("compact",) if quick else ("compact", "texture")):
        if have_xgb:
            methods[f"{feat}_xgb_cond"] = (feat, True, xgb_cond(*sizes[feat]))
        methods[f"{feat}_hgb_cond"] = (feat, True, hgb_cond(*sizes[feat]))
    methods["compact_logistic"] = ("compact", False, logistic)
    methods["compact_knn"] = ("compact", False, knn)
    return methods


def score(prob, target, names):
    from sklearn.metrics import roc_auc_score
    prob = np.clip(np.asarray(prob, dtype=np.float64), 0, 1)
    require(prob.shape == target.shape and np.isfinite(prob).all(), "Bad probability matrix")
    per = ((prob - target) ** 2).mean(0)
    auc = {n: (float(roc_auc_score(target[:, j], prob[:, j]))
               if 0 < target[:, j].mean() < 1 and np.ptp(prob[:, j]) > 0 else None)
           for j, n in enumerate(names)}
    return {"mean_brier": float(per.mean()), "per_model_brier": dict(zip(names, map(float, per))),
            "per_model_auc": auc}


def time_features(images, index, feature_set, fm, n=500):
    for i in index[:20]:
        fm.extract_feature_vector(np.asarray(images[i]), feature_set)
    sample = [np.asarray(images[i]).copy() for i in index[:n]]
    ms = []
    for im in sample:
        t = time.perf_counter_ns()
        fm.extract_feature_vector(im, feature_set)
        ms.append((time.perf_counter_ns() - t) / 1e6)
    return {"images": len(ms), "median_ms_per_image": float(np.median(ms)),
            "p95_ms_per_image": float(np.percentile(ms, 95)), "mean_ms_per_image": float(np.mean(ms))}


def time_predictor(predict, x, reps=30):
    out = {}
    for b in BATCHES:
        z = x[:b]
        for _ in range(3):
            predict(z)
        ms = []
        for _ in range(reps):
            t = time.perf_counter_ns()
            predict(z)
            ms.append((time.perf_counter_ns() - t) / 1e6)
        out[str(b)] = {"median_ms_per_batch": float(np.median(ms))}
    return out


def score_router_files(directory, names, cal, correct):
    rows = {}
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            require(np.array_equal(z["index"], cal), f"{path.name}: index is not router-calibration")
            order = [list(z["models"]).index(n) for n in names]
            rows[path.stem] = score(z["correct_prob"][:, order], correct[cal], names)
    return rows


def summarize(rows):
    """rows: key 'method@fit' -> list of per-seed score dicts."""
    out = {}
    for key, runs in rows.items():
        v = np.array([r["mean_brier"] for r in runs])
        out[key] = {"seeds": len(v), "mean_brier": float(v.mean()), "min": float(v.min()),
                    "max": float(v.max())}
    return out


def markdown(result):
    lines = [f"# Hand-made features vs TinyCNN correctness router ({STAGE})", "",
             "Fit on classifier-validation 5k + router-train 3k (val3k) or router-train only (3k); "
             "scored on the separate router-calibration 2k. Mean correctness Brier over 2000 images "
             "x 8 models; lower is better. The official test set is never opened.", "",
             "| Router | Fit | Seeds | Mean Brier | Seed range | Feature CPU ms/img |", "|---|---|---|---|---|---|"]
    timing = result.get("timing", {}).get("features", {})
    for key, s in sorted(result["summary"].items(), key=lambda kv: kv[1]["mean_brier"]):
        method, fit = key.split("@")
        feat = result["methods"].get(method, {}).get("features")
        t = f"{timing[feat]['median_ms_per_image']:.2f}" if feat in timing else ""
        rng = f"{s['min']:.5f}-{s['max']:.5f}" if s["seeds"] > 1 else ""
        lines.append(f"| {method} | {fit} | {s['seeds']} | {s['mean_brier']:.5f} | {rng} | {t} |")
    if result.get("router_scores"):
        lines += ["", "Neural routers, scored here from a40_routers calibration scores (same targets, "
                  "deployment preprocessing):", "", "| Router | Seeds | Mean Brier | Seed range |", "|---|---|---|---|"]
        for key, s in sorted(result["router_summary"].items(), key=lambda kv: kv[1]["mean_brier"]):
            lines.append(f"| {key} | {s['seeds']} | {s['mean_brier']:.5f} | {s['min']:.5f}-{s['max']:.5f} |")
    ref = result["tinycnn_screen_reference"]
    lines += ["", f"Reference: {ref['router']}: {ref['mean_brier']} ({ref['source']}).", ""]
    return "\n".join(lines)


def run(args):
    t0 = time.time()
    fm, fm_path = load_feature_module()
    try:
        import xgboost
        have_xgb, xgb_version = True, xgboost.__version__
    except ImportError:
        have_xgb, xgb_version = False, None
    import sklearn
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    ev = load_evidence(args.base, args.evidence)
    names, correct, images, cal = ev["names"], ev["correct"], ev["images"], ev["cal"]
    methods = make_methods(have_xgb, args.threads, args.quick)
    feature_sets = sorted({m[0] for m in methods.values()})

    features = {}
    every = np.concatenate([ev["fits"]["val3k"], cal])
    for feat in feature_sets:
        t = time.time()
        x = extract(images, every, feat, fm)
        require(np.isfinite(x).all(), f"Nonfinite {feat} features")
        features[feat] = dict(zip(every.tolist(), x))
        np.savez_compressed(out / f"features_{feat}.npz", index=every, x=x)
        print(f"{feat}: {x.shape[1]} features for {len(every)} images in {time.time() - t:.1f}s", flush=True)

    def X(feat, index):
        return np.stack([features[feat][i] for i in index.tolist()])

    rows, details, probs = {}, {}, {}
    for fit_name, fit in ev["fits"].items():
        base_rate = correct[fit].mean(0)
        rows[f"constant@{fit_name}"] = [score(np.tile(base_rate, (len(cal), 1)), correct[cal], names)]
        for name, (feat, seeded, fitter) in methods.items():
            key = f"{name}@{fit_name}"
            rows[key] = []
            for seed in (SEEDS if seeded else (None,)):
                t = time.time()
                predict = fitter(X(feat, fit), correct[fit], seed if seed is not None else 0)
                p = predict(X(feat, cal))
                s = score(p, correct[cal], names)
                s.update(seed=seed, fit_seconds=round(time.time() - t, 2))
                rows[key].append(s)
                probs[f"{key}-s{seed}"] = p.astype(np.float32)
                print(f"{key} seed {seed}: mean Brier {s['mean_brier']:.5f} ({s['fit_seconds']}s)", flush=True)
                if args.timing and fit_name == "val3k" and seed in (None, SEEDS[0]):
                    details.setdefault("predictor", {})[name] = time_predictor(predict, X(feat, cal))
    np.savez_compressed(out / "calibration_probabilities.npz", index=cal, models=np.array(names),
                        **{k.replace("@", "__"): v for k, v in probs.items()})

    result = {
        "stage": STAGE, "test_set": "never opened (split namespace official_train)",
        "fit_sets": {k: len(v) for k, v in ev["fits"].items()}, "score_set": "router_calibration",
        "score_size": len(cal), "models": names, "split_sha256": ev["split"]["sha256"],
        "evidence_sha256": ev["hashes"], "feature_code": str(fm_path.relative_to(ROOT)),
        "feature_code_sha256": sha256(fm_path),
        "methods": {n: {"features": m[0], "seeded": m[1]} for n, m in methods.items()},
        "xgb_recipe": {**XGB_RECIPE, "size": XGB_SIZE, "quick": args.quick},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "xgboost": xgb_version},
        "host": platform.node(), "job": os.environ.get("SLURM_JOB_ID"),
        "calibration_accuracy": dict(zip(names, map(float, correct[cal].mean(0)))),
        "runs": rows, "summary": summarize(rows), "tinycnn_screen_reference": TINYCNN_SCREEN_REFERENCE,
    }
    if args.router_scores and Path(args.router_scores).is_dir():
        r = score_router_files(args.router_scores, names, cal, correct)
        result["router_scores"] = r
        grouped = {}
        for k, v in r.items():
            grouped.setdefault(k.rsplit("-s", 1)[0], []).append(v)
        result["router_summary"] = summarize(grouped)
    if args.timing:
        result["timing"] = {"scope": "single-thread CPU on this Slurm node; batch-1 numpy feature "
                            "extraction; predictor per batch; not an edge or A40 cost",
                            "cpu": platform.processor() or platform.machine(),
                            "features": {f: time_features(images, cal, f, fm) for f in feature_sets},
                            **details}
    result["seconds"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (out / "summary.md").write_text(markdown(result))
    print(markdown(result))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True, help="clean_classifier_study (holds split_seed42.json)")
    p.add_argument("--evidence", type=Path, required=True, help="router_arch_screen_v1/evidence")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--router-scores", type=Path, help="optional a40_routers_v1/scores to score alongside")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--no-timing", dest="timing", action="store_false")
    p.add_argument("--quick", action="store_true", help="smoke test: compact only, 30 trees")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
