"""CPU-only budget arithmetic, NOT hardware emulation or Jetson measurement."""
import hashlib
import json
from pathlib import Path

PIPE = Path(__file__).resolve().parents[1]
GIB = 1024**3
OUT = PIPE / "reports/jetson_8gb_pc_approximation_20261105"
SOURCES = {
    "original_module_specs": "https://developer.nvidia.com/blog/solving-entry-level-edge-ai-challenges-with-nvidia-jetson-orin-nano/",
    "shared_memory": "https://docs.pytorch.org/TensorRT/tutorials/resource_memory/resource_management.html",
    "current_module_specs": "https://developer.nvidia.com/embedded/jetson-modules/",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix_fit(rows, budget, multiplier):
    """Replay ONLY the measured loading prefix; never extrapolate unseen models."""
    cost, names = 0, []
    for row in rows:
        # Separate-memory PC GPU allocation delta plus a hypothetical retained CPU blob.
        candidate = int(multiplier * row["incremental_used_bytes"]) + row["engine_file_bytes"]
        if cost + candidate > budget:
            break
        cost += candidate
        names.append(row["added_model"])
    return dict(accepted=len(names), measured_prefix_exhausted=len(names) == len(rows),
                used_proxy_bytes=cost, names=names)


def main():
    residency_path = PIPE / "reports/gpu_residency_20261028/study.json"
    audit_path = PIPE / "reports/gpu_residency_20261028/audited_summary.json"
    images_path = PIPE / "artifacts/image_capacity_20261103/study.json"
    residency, audit, images = [json.loads(p.read_text()) for p in (residency_path, audit_path, images_path)]
    assert audit["audit_passed"] and sha(residency_path) == audit["report_sha256"]
    assert residency["status"] == images["status"] == "complete"
    rows = residency["results"]
    assert all(r["status"] == "complete" and r["incremental_used_bytes"] >= 0 for r in rows)
    # Explicit conservative decimal interpretation; this is not measured usable RAM.
    total = 8_000_000_000
    safety, runtime = GIB, GIB // 2
    report = dict(status="complete_cpu_only_scenario_analysis", sources=SOURCES,
                  input_sha256={str(p.relative_to(PIPE)): sha(p) for p in (residency_path, audit_path, images_path)},
                  documented_target="Jetson Orin Nano 8GB, CPU/GPU shared physical memory",
                  assumed_total_bytes=total, safety_bytes=safety, runtime_router_reserve_bytes=runtime,
                  assumptions="8e9-byte conservative interpretation; OS/application reserve2/3/4GiB; "
                  "runtime/router0.5GiB and safety1GiB are unmeasured scenario allowances. "
                  "PC allocation multipliers1/1.5/2 are sensitivity scenarios, NOT statistical bounds.",
                  residency=[], image_arrays=[], batch_input_arrays=[],
                  no_gpu_execution=True, no_dataset_access=True, no_latency_or_energy_prediction=True)
    for os_gib in (2, 3, 4):
        budget = max(0, total - os_gib * GIB - safety - runtime)
        for multiplier in (1, 1.5, 2):
            for order in ("smallest_first", "largest_first", "architecture_round_robin"):
                prefix = sorted((r for r in rows if r["order"] == order), key=lambda r: r["count"])
                assert [r["count"] for r in prefix] == list(range(1, len(prefix) + 1))
                report["residency"].append(dict(os_application_reserve_gib=os_gib,
                    model_image_budget_bytes=budget, allocation_multiplier=multiplier, order=order,
                    **prefix_fit(prefix, budget, multiplier)))
        unloaded = next(r for r in images["rows"] if not r["models_loaded"] and r["dtype"] == "torch.float32")
        loaded = next(r for r in images["rows"] if r["models_loaded"] and r["dtype"] == "torch.float32")
        gpu_delta = unloaded["start_free_bytes"] - loaded["start_free_bytes"]
        assert gpu_delta > 0
        host_blob_bytes = sum(Path(p).stat().st_size for p, _ in images["loaded_engines"])
        for models in (0, 2):
            # Two-model observation is R34/R101 FP16 plus router at maxbatch128,
            # not two arbitrary models or an eight-best-model result.
            model_proxy = int(1.5 * gpu_delta) + host_blob_bytes if models else 0
            available = max(0, budget - model_proxy)
            for side in (32, 128):
                for representation, bytes_per_channel in (("uint8_only", 1), ("fp16_only", 2),
                        ("fp32_only", 4), ("uint8_plus_fp32", 5)):
                    per_image = 3 * side * side * bytes_per_channel
                    count = available // per_image
                    report["image_arrays"].append(dict(os_application_reserve_gib=os_gib,
                        classifier_models=models, model_proxy_bytes=model_proxy, side=side,
                        representation=representation, bytes_per_image=per_image, images=count,
                        full_batches128=count // 128, available_for_arrays_bytes=available))
    for side in (32, 128):
        for batch in (1, 8, 16, 32, 64, 128):
            report["batch_input_arrays"].append(dict(side=side, batch=batch,
                fp32_input_bytes=batch * 3 * side**2 * 4,
                qualification="Input tensor only; NOT activation/context/whole-runtime feasibility"))
    report["limits"] = [
        "No measured Jetson capacity or guaranteed safe model count; no timing/energy emulation.",
        "Only measured PC loading prefixes replayed, not a measured all213-target catalog.",
        "GPU allocation deltas are order/allocator-dependent; serialized bytes do not equal resident bytes.",
        "One retained serialized CPU copy is an explicit scenario, not an observed Jetson allocation.",
        "Image counts cover specified contiguous arrays only, no Python/PIL object overhead or extra copies.",
        "Temporary engine-load/build peaks and actual free Jetson RAM remain unmeasured; reserve is not proof.",
        "Original128px213-target models and new HPC32px models remain distinct;32px image arithmetic is not a model-capacity measurement.",
        "Original68GB/s and newer Super-mode102GB/s memory-bandwidth profiles do not change this arithmetic; neither is used to scale PC latency.",
    ]
    assert len(report["residency"]) == 27 and len(report["image_arrays"]) == 48
    assert len(report["batch_input_arrays"]) == 12
    for row in report["image_arrays"]:
        assert row["images"] * row["bytes_per_image"] <= row["available_for_arrays_bytes"]
        assert (row["images"] + 1) * row["bytes_per_image"] > row["available_for_arrays_bytes"]
    for row in report["residency"]:
        assert row["used_proxy_bytes"] <= row["model_image_budget_bytes"]
        for stricter in report["residency"]:
            if stricter["order"] == row["order"] and stricter["os_application_reserve_gib"] >= row["os_application_reserve_gib"] and stricter["allocation_multiplier"] >= row["allocation_multiplier"]:
                assert stricter["accepted"] <= row["accepted"]
    report["arithmetic_checks_passed"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "capacity_scenarios.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Jetson Orin Nano8GB: PC-based capacity scenarios", "",
        "CPU-only arithmetic completed. **Not Jetson measurement or full hardware emulation.**", "",
        report["assumptions"], "", "## Model residency:1.5x PC allocation scenario", "",
        "Counts are prefixes of previously measured PC loading sequences, not validated Jetson maxima.", "",
        "| OS/application reserve | Smallest first | Largest first | Round-robin |",
        "|---|---:|---:|---:|"]
    for os_gib in (2, 3, 4):
        selected = [r for r in report["residency"] if r["os_application_reserve_gib"] == os_gib and r["allocation_multiplier"] == 1.5]
        values = [str(r["accepted"]) + ("+ (prefix exhausted)" if r["measured_prefix_exhausted"] else "") for r in selected]
        lines.append(f"| {os_gib}GiB | " + " | ".join(values) + " |")
    lines += ["", "##128px image arrays:3GiB OS/application scenario", "",
        "One representation at a time, except the explicit uint8+FP32 pair. Models compete with images for the SAME budget.", "",
        "Two-model proxy uses the existing R34/R101 FP16+router observation,1.5x GPU-memory delta plus one serialized CPU copy.", "",
        "| Classifier models | Representation | Array-capacity estimate |", "|---|---|---:|"]
    for r in report["image_arrays"]:
        if r["os_application_reserve_gib"] == 3 and r["side"] == 128:
            lines.append(f"| {r['classifier_models']} | {r['representation']} | {r['images']:,} |")
    lines += ["", "## Limitations", ""] + ["- " + item for item in report["limits"]]
    lines += ["", "## Documentation", ""] + [f"- [{name}]({url})" for name, url in SOURCES.items()]
    lines += ["", "Inputs and SHA256 identities are recorded in capacity_scenarios.json. No original results, training code, automations or hardware were changed."]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
