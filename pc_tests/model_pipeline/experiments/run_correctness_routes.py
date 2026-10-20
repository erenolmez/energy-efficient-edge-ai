"""Frozen correctness policies: matched, resumable all-213 GPU deployments."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import make_batches, split_validation
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import require_idle_gpu, require_guard
from pc_tests.model_pipeline.experiments.run_precision_experiments import file_hash, write_json


def verified_markers(directory, provenance, names, baseline, labels):
    markers = []
    for seed in (42, 45, 48):
        path = directory / f'global_correctness_seed{seed}.json'
        marker = json.loads(path.read_text())
        fit, calibration = split_validation(labels, seed)
        if (marker.get('status') != 'complete' or marker.get('models') != names
                or marker.get('baseline') != baseline or marker.get('provenance') != provenance
                or marker['settings']['seed'] != seed or marker['settings']['epochs'] != 20
                or marker['fit_positions'] != fit.tolist()
                or marker['calibration_positions'] != calibration.tolist()
                or len(marker['training_loss']) != 20):
            raise RuntimeError('Incomplete/stale correctness training: ' + str(path))
        if file_hash(Path(marker['weights'])) != marker['weights_sha256']:
            raise RuntimeError('Weights hash mismatch')
        for engine in marker['engines'].values():
            if file_hash(Path(engine['engine'])) != engine['sha256'] or engine['torch_max_error'] > 1e-5:
                raise RuntimeError('Engine hash/parity failure')
        if set(marker['policies']) != {'latency', 'energy'}:
            raise RuntimeError('Both objectives required')
        for policies in marker['policies'].values():
            if set(policies) != {'8', '16', '32', '64', '128'}:
                raise RuntimeError('Incomplete batch coverage')
            for item in policies.values():
                if len(item['costs']) != 213 or set(item['budgets']) != {'0.5', '1.0', '1.5', '2.0'}:
                    raise RuntimeError('Incomplete candidate/budget coverage')
        markers.append((path, marker))
    return markers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('static-report', 'static-dir', 'cache-dir', 'router-dir', 'report-dir', 'thermal-state'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resident-capacity', type=int, default=2)
    parser.add_argument('--guard-startup-seconds', type=int, default=120)
    args = parser.parse_args()
    require_idle_gpu()
    records, validation_labels, _, baseline, provenance = load_validation(args.static_report, args.static_dir, args.cache_dir)
    names = [r['name'] for r in records]
    by_name = {r['name']: r for r in records}
    markers = verified_markers(args.router_dir, provenance, names, baseline, validation_labels)
    # Reuse guard verification, but require the measured runner identity/script.
    def guarded():
        from datetime import datetime, timezone
        state = json.loads(args.thermal_state.read_text(encoding='utf-8-sig'))
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(state['utc'].replace('Z', '+00:00'))).total_seconds()
        if state['status'] != 'monitoring':
            raise RuntimeError('Thermal stop latched: do not resume')
        if not 0 <= age <= 120 or state['expected_script'] != Path(__file__).name or os.getpid() not in state['tracked_process_ids']:
            raise RuntimeError('Fresh exact runtime thermal guard required')
    deadline = time.monotonic() + args.guard_startup_seconds
    print(f'Awaiting verified runtime thermal guard for PID {os.getpid()}', flush=True)
    while True:
        try:
            guarded()
            break
        except RuntimeError:
            if json.loads(args.thermal_state.read_text(encoding='utf-8-sig'))['status'] != 'monitoring' or time.monotonic() >= deadline:
                raise
            time.sleep(1)
    import torch
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner
    from pc_tests.model_pipeline.runtime.global_candidate_pool import EngineCache
    from pc_tests.model_pipeline.runtime.measure_pc_routes import measure
    from pc_tests.model_pipeline.experiments.run_pc_routes import eligible_static
    images = np.load(args.cache_dir / 'test_rgb128.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(args.cache_dir / 'test_labels.npy', allow_pickle=False)
    if images.shape != (10000, 3, 128, 128) or len(labels) != 10000:
        raise RuntimeError('Held-out data identity shape mismatch')
    test_hashes = {name: file_hash(args.cache_dir / name) for name in ('test_rgb128.npy', 'test_labels.npy')}
    static_hash = file_hash(args.static_report)
    report_path = args.report_dir / 'pc_routes_correctness.json'
    protocol = dict(protocol_version=1, scope='global_correctness', models=names,
                    static_report_sha256=static_hash, test_hashes=test_hashes,
                    resident_capacity=args.resident_capacity, repeats=5, expected_configurations=120,
                    training_markers={p.name: file_hash(p) for p, _ in markers},
                    timing_boundary='Pinned RGB128 -> transfer/normalization -> fresh TensorRT 213 probabilities -> host selection -> lazy cache hash/load/eviction -> selected inference -> host predictions. No cached routing outputs; true partial shapes. Decode/resize/pinning excluded for every path.',
                    energy_scope='NVML GPU only; includes idle/display, excludes CPU/rest of PC')
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(**protocol, results=[])
    if any(report.get(k) != v for k, v in protocol.items()):
        raise RuntimeError('Existing report protocol changed; preserve and inspect')
    saved = {r['id']: r for r in report['results']}
    def persist(row):
        saved[row['id']] = row
        report['results'] = list(saved.values())
        write_json(report_path, report)
    def load(record):
        if file_hash(Path(record['engine_path'])) != record['engine_sha256']:
            raise RuntimeError('Classifier engine hash mismatch')
        return EngineRunner(record['engine_path'])
    def release():
        gc.collect()
        torch.cuda.empty_cache()
    for marker_path, metadata in markers:
        seed = metadata['settings']['seed']
        routers = {int(size): RouterRunner(item['engine']) for size, item in metadata['engines'].items()}
        for runner in routers.values():
            if tuple(runner.output.shape) != (1, 213):
                raise RuntimeError('Actual TensorRT engine does not expose 213 probabilities')
        for objective, sizes in metadata['policies'].items():
            for size_text, item in sizes.items():
                size = int(size_text)
                batches = make_batches(np.arange(len(labels)), size, seed + size)
                if not {len(b) for b in batches} <= set(routers):
                    raise RuntimeError('Missing actual partial-batch engine')
                for budget_text, policy in item['budgets'].items():
                    identity = f'correctness_{objective}_seed{seed}_b{size}_loss{budget_text}'
                    if saved.get(identity, {}).get('status') == 'complete':
                        continue
                    guarded()
                    row = dict(id=identity, router='tiny213_correctness', seed=seed, objective=objective,
                               batch_size=size, budget_pp=float(budget_text), models=names,
                               router_metadata_sha256=file_hash(marker_path), policy=policy)
                    engines = EngineCache(records, args.resident_capacity, load, torch.cuda.synchronize, release)
                    try:
                        chosen = eligible_static(records, np.asarray(metadata['calibration_positions']),
                                                 validation_labels, args.static_dir, by_name[baseline], size, objective, float(budget_text))
                        comparators = {}
                        for name in dict.fromkeys((baseline, chosen)):
                            cache = EngineCache([by_name[name]], 1, load, torch.cuda.synchronize, release)
                            try:
                                comparators[name] = measure(images, labels, batches, cache, 0)
                            finally:
                                cache.clear()
                        route_policy = dict(policy, costs=item['costs'])
                        measurements = measure(images, labels, batches, engines, names.index(baseline),
                                               routers, route_policy, 'correctness_images')
                        row.update(status='complete', measurements=measurements, comparator_measurements=comparators,
                                   baseline=baseline, best_eligible_static=chosen,
                                   accuracy_loss_vs_p0_pp=float(np.mean([r['test_accuracy'] for r in comparators[baseline]]) - np.mean([r['test_accuracy'] for r in measurements])))
                        row['test_budget_passed'] = row['accuracy_loss_vs_p0_pp'] <= float(budget_text) + 1e-10
                        for label, name in (('p0_fp32', baseline), ('best_eligible_static', chosen)):
                            for metric, key in (('latency', 'rgb_input_to_prediction_seconds'), ('energy', 'gpu_energy_j_per_image')):
                                values = comparators[name] + measurements
                                if all(r.get(key) is not None for r in values):
                                    base = np.median([r[key] for r in comparators[name]])
                                    actual = np.median([r[key] for r in measurements])
                                    row[f'{metric}_saving_vs_{label}_percent'] = float(100 * (base - actual) / base)
                                else:
                                    row[f'{metric}_saving_vs_{label}_percent'] = None
                    except Exception:
                        row.update(status='failed', traceback=traceback.format_exc())
                        persist(row)
                        raise
                    finally:
                        engines.clear()
                    persist(row)
                    print(f'Finished {identity}: {row["status"]}', flush=True)
        del routers
        release()


if __name__ == '__main__':
    main()
