"""Read saved choices/order only; count exact switches and simulate memory budgets.

No image/label access, inference, policy fitting, hardware changes or network access.
"""
import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np

PIPE = Path(__file__).resolve().parents[1]
OUT = PIPE / "reports/six_model_switching_20261105"
GIB = 1024**3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transitions(sequence, models=6):
    sequence = np.asarray(sequence)
    if sequence.ndim != 1 or not len(sequence) or sequence.dtype.kind not in "iu" or (sequence < 0).any() or (sequence >= models).any():
        raise ValueError("Nonempty integer model-ID sequence required")
    matrix = np.zeros((models, models), dtype=np.int64)
    np.add.at(matrix, (sequence[:-1], sequence[1:]), 1)
    switches = int(matrix.sum() - np.trace(matrix))
    counts = np.bincount(sequence, minlength=models)
    assert matrix.sum() == len(sequence) - 1
    assert switches == np.count_nonzero(sequence[:-1] != sequence[1:])
    return matrix, counts, switches


def replay(sequence, sizes, budget, capacity=6, initial=()):
    """Byte-limited LRU, fixed decisions; no simulated Jetson time or energy."""
    sizes = list(map(int, sizes))
    if budget <= 0 or capacity < 1 or min(sizes) <= 0:
        raise ValueError("Positive memory sizes/budget/capacity required")
    cache = OrderedDict((int(i), None) for i in initial)
    if len(cache) > capacity or sum(sizes[i] for i in cache) > budget:
        raise ValueError("Initial cache does not fit")
    hits = loads = evictions = peak = blocked = 0
    for value in sequence:
        index = int(value)
        required = sizes[index]
        if required > budget:
            blocked += 1
            continue
        if index in cache:
            hits += 1
            cache.move_to_end(index)
        else:
            while cache and (len(cache) >= capacity or sum(sizes[i] for i in cache) + required > budget):
                cache.popitem(last=False)
                evictions += 1
            cache[index] = None
            loads += 1
        peak = max(peak, sum(sizes[i] for i in cache))
    assert hits + loads + blocked == len(sequence) and peak <= budget
    return dict(hits=hits, cold_and_reload_loads=loads, evictions=evictions,
                infeasible_requests=blocked, peak_model_proxy_bytes=peak,
                final_resident=list(cache), hit_rate=hits / len(sequence))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    reports = ("compact_resident_screen_20261028", "compact_switching_confirmation_20261028")
    models = None
    entries, pairs, simulations, provenance = [], [], [], {}
    profile = json.loads((PIPE / "reports/compact_runtime_profile_20261028/audited_summary.json").read_text())
    assert profile["audit_passed"]
    measured_sizes = {r["name"]: r["file_resident_bytes"] for r in profile["models"]}
    for study_name in reports:
        directory = PIPE / "reports" / study_name
        report_path, audit_path = directory / "study.json", directory / "audited_summary.json"
        report, audit = [json.loads(p.read_text()) for p in (report_path, audit_path)]
        assert report["status"] == "complete" and audit["audit_passed"]
        assert sha(report_path) == audit["report_sha256"]
        if models is None:
            models = report["pool"]
        assert models == report["pool"] and len(models) == 6
        provenance[str(report_path.relative_to(PIPE))] = sha(report_path)
        provenance[str(audit_path.relative_to(PIPE))] = sha(audit_path)
        for row in report["full_sweeps"]:
            path = Path(row["runtime_file"])
            assert path.resolve().is_relative_to((PIPE / "artifacts" / study_name).resolve())
            assert sha(path) == row["runtime_sha256"]
            with np.load(path, allow_pickle=False) as saved:
                # Saved arrays are indexed by IMAGE position, not wall-clock execution order.
                choices, order = saved["choices"], saved["order"]
                assert choices.shape == order.shape == (10000,)
                assert order.dtype.kind in "iu" and np.array_equal(np.sort(order), np.arange(10000))
                sequence = choices[order]
            matrix, counts, switches = transitions(sequence)
            assert counts.tolist() == row["selected_images"]
            method = row.get("method", row.get("variant"))
            key = dict(study=study_name, method=method, seed=row["seed"], repeat=row.get("repeat", 0))
            entries.append(dict(**key, images=10000, possible_transitions=9999,
                switches=switches, switch_percent=100 * switches / 9999,
                first_model=models[int(sequence[0])], mean_run_length=10000 / (switches + 1),
                **{f"selected_model_{i}": int(counts[i]) for i in range(6)},
                runtime_sha256=sha(path), trace_path=str(path.relative_to(PIPE))))
            for start in range(6):
                for end in range(6):
                    if start != end:
                        pairs.append(dict(**key, from_model=models[start], to_model=models[end],
                                          switches=int(matrix[start, end])))
            sizes = [measured_sizes[name] for name in models]
            if study_name == reports[1] and row["capacity"] == 2:
                check = replay(sequence, sizes, sum(sizes), capacity=2)
                assert check["cold_and_reload_loads"] == row["runtime_loads"]
                assert check["evictions"] == row["evictions"]
            # Use the observed host store from Study B for all frozen-trace scenarios.
            # These are hypothetical deployment budgets, not measurements on Jetson.
            b_report = report if study_name == reports[1] else json.loads((PIPE / "reports" / reports[1] / "study.json").read_text())
            host_bytes = b_report["host_store"]["host_bytes"]
            io_bytes = 128 * 3 * 128**2 * (1 + 4 + 4)  # raw + two FP32 batch buffers
            for system_gib in (2, 3, 4):
                budget = 8_000_000_000 - int((system_gib + 1 + .5) * GIB) - host_bytes - io_bytes
                assert budget > 0
                for multiplier in (1, 1.5, 2):
                    proxy_sizes = [int(size * multiplier) for size in sizes]
                    for capacity in (2, 6):
                        simulated = replay(sequence, proxy_sizes, budget, capacity=capacity)
                        simulations.append(dict(**key, os_application_reserve_gib=system_gib,
                            memory_multiplier=multiplier, max_resident_models=capacity,
                            model_budget_bytes=budget, **{k:v for k,v in simulated.items() if k != "final_resident"}))
    assert len(entries) == 102 and len(pairs) == 102 * 30 and len(simulations) == 102 * 18
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "switch_counts.csv", entries)
    write_csv(OUT / "from_to_transitions.csv", pairs)
    write_csv(OUT / "jetson_8gb_cache_simulation.csv", simulations)
    summary = dict(status="complete_audited_saved_trace_analysis", model_ids=dict(enumerate(models)),
        report_hashes=provenance, profile_sha256=sha(PIPE / "reports/compact_runtime_profile_20261028/audited_summary.json"),
        sweeps=len(entries), images_per_sweep=10000, transitions_per_sweep=9999,
        switch_definition="Consecutive chronological requests select different classifier IDs; first request is NOT a switch.",
        no_gpu_execution=True, no_images_or_labels_opened=True, no_policies_changed=True,
        simulation="Fixed saved policy traces; byte-LRU scenarios, NOT Jetson hardware/time/energy emulation. Cache-aware policies are NOT recalibrated for new caches.",
        memory_assumptions=dict(total_bytes=8_000_000_000, os_application_gib=[2,3,4],
            safety_gib=1, runtime_router_gib=.5, retained_host_store_bytes=host_bytes,
            io_buffers_bytes=io_bytes, pc_memory_multipliers=[1,1.5,2]),
        limits=["Repeated sweeps are not independent input samples; retain per-seed/per-repeat results.",
                "Historical held-out decision traces are descriptive only; no test-based policy selection or tuning.",
                "Memory multipliers/reserves are assumptions, not confidence bounds or verified Jetson capacities.",
                "Serialized host store, CPU+GPU allocations share the same assumed budget; build/load transient peaks unmeasured."],
        results=entries)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Six-model switch counts and Jetson8GB PC simulation", "",
        "Completed CPU-only analysis of102 saved10,000-request sweeps. No model inference, images/labels, Jetson/SSH or hardware access.", "",
        "A switch means a different model on the next request; cold first load is not a switch. A model switch is not necessarily a cache miss.", "",
        "Choices are reordered using each saved execution-order permutation. All trace/report hashes and selection totals checked; original capacity2 LRU load/eviction counts reproduced exactly.", "",
        "## Study A: compact routers, all six engines resident", "",
        "| Router | Seed42 switches | Seed45 switches | Seed48 switches |", "|---|---:|---:|---:|"]
    for method in sorted({r["method"] for r in entries if r["study"] == reports[0]}):
        selected = [next(r for r in entries if r["study"] == reports[0] and r["method"] == method and r["seed"] == seed) for seed in (42,45,48)]
        lines.append("| " + method + " | " + " | ".join(str(r["switches"]) for r in selected) + " |")
    lines += ["", "## Study B: TinyCNN switching confirmation", "",
        "Ranges span three repeats per seed; no pooling as independent observations.", "",
        "| Variant | Seed42 switch range | Seed45 switch range | Seed48 switch range |", "|---|---:|---:|---:|"]
    for method in sorted({r["method"] for r in entries if r["study"] == reports[1]}):
        values = []
        for seed in (42,45,48):
            v = [r["switches"] for r in entries if r["study"] == reports[1] and r["method"] == method and r["seed"] == seed]
            values.append(str(min(v)) if min(v) == max(v) else f"{min(v)}–{max(v)}")
        lines.append("| " + method + " | " + " | ".join(values) + " |")
    lines += ["", "## Directional transitions: Study A TinyCNN, seed42", "",
        "| From | To | Switches |", "|---|---|---:|"]
    for row in pairs:
        if row["study"] == reports[0] and row["method"] == "tinycnn" and row["seed"] == 42 and row["switches"]:
            lines.append(f"| {row['from_model']} | {row['to_model']} | {row['switches']} |")
    lines += ["", "## TinyCNN selected destinations (images, not switches)", "",
        "| Model | Seed42 | Seed45 | Seed48 |", "|---|---:|---:|---:|"]
    for index, name in enumerate(models):
        selected = [next(r for r in entries if r["study"] == reports[0] and r["method"] == "tinycnn" and r["seed"] == seed) for seed in (42,45,48)]
        lines.append("| " + name + " | " + " | ".join(str(r[f"selected_model_{index}"]) for r in selected) + " |")
    lines += ["", "Full exact directional counts (including zeros) for every router/seed/repeat are in from_to_transitions.csv.", "",
        "## PC-based Jetson8GB simulation", "",
        summary["simulation"], "", "8e9-byte budget;2/3/4GiB OS+applications,1GiB safety,0.5GiB runtime/router; original host-store bytes and128-image raw+twoFP32 buffers subtracted. Allocation deltas multiplied by1/1.5/2; capacity2 versus up-to6 LRU. Cold-start loads included. No timing/energy predictions.", ""]
    center = [r for r in simulations if r["os_application_reserve_gib"] == 3 and r["memory_multiplier"] == 1.5 and r["max_resident_models"] == 6]
    lines.append(f"Central scenario across102 traces: loads range {min(r['cold_and_reload_loads'] for r in center)}-{max(r['cold_and_reload_loads'] for r in center)}, evictions range {min(r['evictions'] for r in center)}-{max(r['evictions'] for r in center)}, infeasible requests {sum(r['infeasible_requests'] for r in center)}. Demand-loaded cache, NOT preloading all six. Not a measured Jetson result.")
    lines += ["", "## Qualifications", ""] + ["- " + x for x in summary["limits"]]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
