"""Resumable end-to-end PC router tests against matched static deployments."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import argparse
import gc
import json
import traceback
from pathlib import Path
import numpy as np
import torch

from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import make_batches
from pc_tests.model_pipeline.runtime.measure_pc_routes import measure
from pc_tests.model_pipeline.experiments.run_precision_experiments import file_hash, write_json
from pc_tests.model_pipeline.runtime.global_candidate_pool import EngineCache, global_candidates


def eligible_static(records, positions, labels, directory, baseline, size, objective, budget):
    """Choose comparator using calibration positions, never test correctness."""
    base_predictions = np.load(directory / f"{baseline['name']}_validation_predictions.npy")
    minimum = 100 * (base_predictions[positions] == labels[positions]).mean() - budget
    eligible = []
    for record in records:
        predictions = np.load(directory / f"{record['name']}_validation_predictions.npy")
        if 100 * (predictions[positions] == labels[positions]).mean() + 1e-10 < minimum:
            continue
        key = 'normalized_host_input_to_logits_batch_ms' if objective == 'latency' else 'gpu_energy_j_per_image'
        values = [row.get(key) for row in record['timings'][str(size)]]
        if all(value is not None and value > 0 for value in values):
            eligible.append((float(np.median(values)), record['name']))
    if not eligible:
        raise RuntimeError('No eligible measured static comparator')
    return min(eligible)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static-report', type=Path, required=True)
    parser.add_argument('--static-dir', type=Path, required=True)
    parser.add_argument('--router-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--scope', choices=('global', 'within-architecture'), default='global')
    parser.add_argument('--resident-capacity', type=int, default=2)
    args = parser.parse_args()
    static = json.loads(args.static_report.read_text())
    if len(static['results']) != 213 or any(row['status'] != 'complete' for row in static['results']):
        raise RuntimeError('GPU static sweep must be complete')
    by_name = {row['name']: row for row in static['results']}
    static_hash = file_hash(args.static_report)
    groups = ['global'] if args.scope == 'global' else {row['architecture'] for row in static['results']}
    required_markers = {f'{architecture}_{objective}_seed{seed}.json'
                        for architecture in groups
                        for objective in ('latency','energy') for seed in (42,45,48)}
    discovered_markers = {path.name for path in args.router_dir.glob('*_seed*.json')}
    if not required_markers <= discovered_markers:
        raise RuntimeError(f'Router training has not finished all required configurations: {sorted(required_markers-discovered_markers)}')
    images = np.load(args.cache_dir / 'test_rgb128.npy', mmap_mode='r')
    labels = np.load(args.cache_dir / 'test_labels.npy')
    validation_labels = np.load(args.cache_dir / 'validation_labels.npy')
    report_path = args.report_dir / ('pc_routes_global.json' if args.scope == 'global' else 'pc_routes.json')
    report = json.loads(report_path.read_text()) if report_path.exists() else {'results': []}
    saved = {row['id']: row for row in report['results']}
    report.update(protocol_version=2, scope=args.scope, resident_capacity=args.resident_capacity,
                  static_report_sha256=static_hash, test_images=10000,
                  timing_boundary='Pinned RGB128 -> shared CUDA normalization -> fresh GPU features/router -> dispatch -> runtime cache miss verification/loading/eviction when needed -> TensorRT inference -> host prediction. Decode/resize and pinning excluded. Matched global static comparators use the same cold-cache boundary.',
                  repeats=5, energy_scope='NVML total reported GPU energy; includes display/idle load, excludes CPU/rest of PC.')

    def persist(row):
        saved[row['id']] = row
        report['results'] = list(saved.values())
        write_json(report_path, report)

    for marker in sorted(args.router_dir.glob('*_seed*.json')):
        if marker.name not in required_markers:
            continue
        metadata = json.loads(marker.read_text())
        if metadata.get('status') == 'unavailable':
            continue
        if metadata.get('status') != 'complete' or metadata['static_report_sha256'] != static_hash:
            raise RuntimeError(f'Incomplete or stale router training: {marker}')
        if len(metadata['routers']) != 40 or set(metadata['settings']['batch_sizes']) != {8,16,32,64,128}:
            raise RuntimeError('Incomplete direct/aggregation router or batch-size coverage')
        architecture, seed = metadata['architecture'], metadata['seed']
        full = static['results'] if args.scope == 'global' else [row for row in static['results'] if row['architecture'] == architecture]
        baseline = by_name[metadata.get('baseline_name', f'{architecture}__p0__fp32')]
        positions = np.asarray(metadata['calibration_positions'])
        names = metadata['models']
        if args.scope == 'global':
            expected, reference = global_candidates(full)
            if names != [r['name'] for r in expected] or baseline['name'] != reference or metadata.get('scope') != 'global':
                raise RuntimeError('Global routing must include all 213 candidates and validation-selected reference')
        engines = []
        try:
            def load(record):
                if file_hash(Path(record['engine_path'])) != record['engine_sha256']:
                    raise RuntimeError(f"Engine checksum mismatch: {record['name']}")
                return EngineRunner(record['engine_path'])
            def release():
                gc.collect()
                torch.cuda.empty_cache()
            if args.scope == 'global':
                engines = EngineCache([by_name[n] for n in names], args.resident_capacity,
                                      load, torch.cuda.synchronize, release)
            else:
                engines = [load(by_name[name]) for name in names]
        except Exception:
            # No engine loading is moved inside timing to conceal resident-pool
            # capacity failures. Keep the full failure record for diagnosis.
            persist({'id': marker.stem + '_resident_pool', 'status': 'failed',
                     'static_report_sha256': static_hash, 'traceback': traceback.format_exc()})
            raise
        saved.pop(marker.stem + '_resident_pool', None)
        for item in metadata['routers']:
            if file_hash(Path(item['engine'])) != item['engine_sha256']:
                raise RuntimeError('Router engine checksum mismatch')
            router = RouterRunner(item['engine'])
            for size_text, policies in item['policies'].items():
                size = int(size_text)
                batches = make_batches(np.arange(len(labels)), size, seed + size)
                for budget_text, policy in policies.items():
                    identity = f"{marker.stem}_{item['method']}_b{size}_loss{budget_text}"
                    prior = saved.get(identity, {})
                    if prior.get('status') == 'complete' and prior.get('router_metadata_sha256') == file_hash(marker):
                        continue
                    row = {'id': identity, 'architecture': architecture, 'seed': seed,
                           'router': item['method'], 'batch_size': size, 'budget_pp': float(budget_text),
                           'objective': metadata['objective'], 'models': names, 'scope': args.scope,
                           'router_metadata_sha256': file_hash(marker), 'static_report_sha256': static_hash}
                    try:
                        chosen = eligible_static(full, positions, validation_labels, args.static_dir,
                                                 baseline, size, metadata['objective'], float(budget_text))
                        comparator_results = {}
                        for comparator in (baseline['name'], chosen):
                            comparator_id = f'static_{architecture}_seed{seed}_b{size}_{comparator}'
                            previous = saved.get(comparator_id, {})
                            if args.scope == 'global' or previous.get('status') != 'complete' or previous.get('static_report_sha256') != static_hash:
                                if args.scope == 'global':
                                    static_engine = EngineCache([by_name[comparator]], 1, load, torch.cuda.synchronize, release)
                                elif comparator in names:
                                    static_engine = engines[names.index(comparator)]
                                else:
                                    static_engine = EngineRunner(by_name[comparator]['engine_path'])
                                measurements = measure(images, labels, batches, static_engine if args.scope == 'global' else [static_engine], 0)
                                previous = {'id': comparator_id, 'status': 'complete',
                                            'static_report_sha256': static_hash, 'model': comparator,
                                            'seed': seed, 'batch_size': size, 'measurements': measurements}
                                persist(previous)
                                if args.scope == 'global':
                                    static_engine.clear()
                                del static_engine
                            comparator_results[comparator] = previous['measurements']
                        measurements = measure(images, labels, batches, engines, metadata['baseline_index'],
                                               router, policy, item['input'])
                        if args.scope == 'global':
                            engines.clear()
                        row['comparator_measurements'] = comparator_results
                        row['comparison_protocol'] = 'Fresh static comparators immediately before each global routed configuration; five cold-cache repeats per path.' if args.scope == 'global' else 'Legacy cached comparators'
                        p0_rows = comparator_results[baseline['name']]
                        best_rows = comparator_results[chosen]
                        row.update(status='complete', measurements=measurements, best_eligible_static=chosen,
                                   accuracy_loss_vs_p0_pp=float(np.mean([r['test_accuracy'] for r in p0_rows]) - np.mean([r['test_accuracy'] for r in measurements])))
                        row['test_budget_passed'] = row['accuracy_loss_vs_p0_pp'] <= float(budget_text) + 1e-10
                        for label, comparison in [('p0_fp32', p0_rows), ('best_eligible_static', best_rows)]:
                            for objective, key in [('latency', 'rgb_input_to_prediction_seconds'), ('energy', 'gpu_energy_j_per_image')]:
                                if all(r.get(key) is not None for r in comparison + measurements):
                                    base = float(np.median([r[key] for r in comparison]))
                                    actual = float(np.median([r[key] for r in measurements]))
                                    row[f'{objective}_saving_vs_{label}_percent'] = 100 * (base - actual) / base
                                else:
                                    row[f'{objective}_saving_vs_{label}_percent'] = None
                    except Exception as exc:
                        row.update(status='failed', error=str(exc), traceback=traceback.format_exc())
                    persist(row)
                    print(f"Finished {identity}: {row['status']}", flush=True)
            del router
        if args.scope == 'global':
            engines.clear()
        del engines
        gc.collect()
        torch.cuda.empty_cache()
    if any(row.get('status') == 'failed' for row in saved.values()):
        raise SystemExit('Recorded routing failures remain; inspect and resume, not overall completion')


if __name__ == '__main__':
    main()
