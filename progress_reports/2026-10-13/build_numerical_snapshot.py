"""Read saved experiments only; export numerical supplements without GPU work."""
import collections
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPORTS = ROOT / "pc_tests/model_pipeline/reports"


def mean(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.mean(values) if values else None


def main():
    snapshot = {"cutoff": "12 October 2026; presentation 13 October", "sources": {}, "experiments": {}}
    for name in ("gpu_static.json", "precision_accuracy.json", "pc_routes.json", "pc_routes_global.json", "pc_routes_correctness.json"):
        path = REPORTS / name
        data = json.loads(path.read_text(encoding="utf-8"))
        snapshot["sources"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        result = []
        for row in data["results"]:
            item = {k: v for k, v in row.items() if k not in ("measurements", "comparator_measurements", "models", "timings", "integer_compute_layers")}
            if "models" in row:
                item["logical_candidate_count"] = len(row["models"])
            if "integer_compute_layers" in row:
                item["integer_compute_layer_count"] = len(row["integer_compute_layers"])
            if "timings" in row:
                item["timings_summary"] = {b: {k: mean(ms, k) for k in ms[0] if isinstance(ms[0][k], (int, float))} | {"repeats": len(ms)} for b, ms in row["timings"].items() if ms}
                item["static_repeat_measurements"] = row["timings"]
            if "measurements" in row:
                ms = row["measurements"]
                item["measurement_summary"] = {k: mean(ms, k) for k in ms[0] if isinstance(ms[0][k], (int, float))} | {"repeats": len(ms)}
                models = row.get("models", [row.get("model", "unknown")])
                item["selected_images_per_repeat"] = {m: statistics.mean(r["selected_images"][i] for r in ms) for i, m in enumerate(models) if any(r.get("selected_images", [])[i] for r in ms)}
                item["repeat_numeric_measurements"] = [{k: v for k, v in r.items() if isinstance(v, (int, float, str))} for r in ms]
            if "comparator_measurements" in row:
                item["comparator_summaries"] = {m: {k: mean(ms, k) for k in ms[0] if isinstance(ms[0][k], (int, float))} | {"repeats": len(ms)} for m, ms in row["comparator_measurements"].items()}
                item["comparator_repeat_numeric_measurements"] = {m: [{k: v for k, v in r.items() if isinstance(v, (int, float, str))} for r in ms] for m, ms in row["comparator_measurements"].items()}
            result.append(item)
        snapshot["experiments"][name] = result
    audit = REPORTS / "complementarity/validation_complementarity.json"
    snapshot["complementarity"] = json.loads(audit.read_text(encoding="utf-8"))
    snapshot["sources"]["validation_complementarity.json"] = {"sha256": hashlib.sha256(audit.read_bytes()).hexdigest()}
    training = REPORTS / "interim/20261011T180318Z/training_times.json"
    snapshot["classifier_and_scalar_training_stages"] = json.loads(training.read_text(encoding="utf-8"))
    snapshot["sources"]["training_times.json"] = {"sha256": hashlib.sha256(training.read_bytes()).hexdigest()}
    snapshot["correctness_training"] = {}
    for marker in sorted((ROOT / "pc_tests/model_pipeline/artifacts/pc_routers_correctness").glob("global_correctness_seed*.json")):
        snapshot["correctness_training"][marker.name] = json.loads(marker.read_text(encoding="utf-8"))
        snapshot["sources"][marker.name] = {"sha256": hashlib.sha256(marker.read_bytes()).hexdigest()}
    snapshot["jetson_checkpoint"] = (ROOT / "jetson_tests/STAGE_CHECKPOINT.md").read_text(encoding="utf-8")
    (OUT / "numerical_results.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    compact = {"cutoff": snapshot["cutoff"], "sources": snapshot["sources"], "experiments": {}}
    kept = {"name", "id", "status", "architecture", "prune", "precision", "test_accuracy", "validation_accuracy", "integer_compute_layer_count", "router", "seed", "objective", "batch_size", "budget_pp", "logical_candidate_count", "baseline", "best_eligible_static", "accuracy_loss_vs_p0_pp", "test_budget_passed", "latency_saving_vs_p0_fp32_percent", "energy_saving_vs_p0_fp32_percent", "latency_saving_vs_best_eligible_static_percent", "energy_saving_vs_best_eligible_static_percent", "measurement_summary", "selected_images_per_repeat", "timings_summary"}
    for name, rows in snapshot["experiments"].items():
        compact["experiments"][name] = [{k: v for k, v in row.items() if k in kept} for row in rows]
    compact["classifier_and_scalar_training_stages"] = snapshot["classifier_and_scalar_training_stages"]
    compact["jetson_checkpoint"] = snapshot["jetson_checkpoint"]
    compact["note"] = "Per-configuration summaries only; full repeats/provenance remain in local numerical_results.json. No engines or weights published."
    (OUT / "numerical_summary.json").write_text(json.dumps(compact, separators=(",", ":")), encoding="utf-8")
    print("SAVED", OUT / "numerical_results.json")
    for name in ("pc_routes.json", "pc_routes_global.json", "pc_routes_correctness.json"):
        rows = [r for r in snapshot["experiments"][name] if "router" in r]
        print("STUDY", name, len(rows), "budgetpass", sum(r.get("test_budget_passed", False) for r in rows))
        for method in sorted(set(r["router"].split("_b")[0] if r["router"].startswith("image_tree_") else r["router"] for r in rows)):
            group = [r for r in rows if (r["router"].split("_b")[0] if r["router"].startswith("image_tree_") else r["router"]) == method]
            good = [r for r in group if r.get("test_budget_passed")]
            print(method, "n", len(group), "pass", len(good), "L+", sum(r["latency_saving_vs_best_eligible_static_percent"] > 0 for r in good), "E+", sum(r["energy_saving_vs_best_eligible_static_percent"] > 0 for r in good))
        if name == "pc_routes_correctness.json":
            for b in (8, 16, 32, 64, 128):
                for obj in ("latency", "energy"):
                    group = [r for r in rows if r["batch_size"] == b and r["objective"] == obj]
                    good = [r for r in group if r["test_budget_passed"]]
                    key = obj + "_saving_vs_best_eligible_static_percent"
                    print("CORRECT", b, obj, "pass", len(good), "wins", sum(r[key] > 0 for r in good), "range", round(min(r[key] for r in group), 2), round(max(r[key] for r in group), 2))
            for seed, obj, b, budget in ((42, "latency", 128, 2), (45, "latency", 128, 1), (48, "latency", 128, 1), (48, "latency", 128, 1.5), (48, "latency", 128, 2), (42, "energy", 128, 1.5)):
                r = next(r for r in rows if (r["seed"], r["objective"], r["batch_size"], r["budget_pp"]) == (seed, obj, b, budget))
                print("EXAMPLE", {k: r[k] for k in ("seed", "objective", "batch_size", "budget_pp", "accuracy_loss_vs_p0_pp", "best_eligible_static", "latency_saving_vs_best_eligible_static_percent", "energy_saving_vs_best_eligible_static_percent", "measurement_summary", "selected_images_per_repeat")})


if __name__ == "__main__":
    main()
