"""Fit batch routers on validation images using measured PC TensorRT costs."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(PIPELINE.parent / 'router_benchmark'))
from train_batch_routers import train_tinycnn
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation, make_batches, batch_targets, calibrate
from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
from pc_tests.model_pipeline.runtime.gpu_router_models import TreeRouter, LogisticRouter, XGBRouter, ConstantRouter, AggregatedTreeRouter
from pc_tests.model_pipeline.experiments.run_precision_experiments import file_hash, write_json
from pc_tests.model_pipeline.runtime.router_tensorrt import build_router, RouterRunner
from pc_tests.model_pipeline.runtime.router_candidate_pool import shortlist
from pc_tests.model_pipeline.runtime.global_candidate_pool import global_candidates


def extract(inputs, images, batches):
    features, grids = [], []
    with torch.inference_mode():
        for batch in batches:
            host = torch.from_numpy(np.array(images[batch])).pin_memory()
            normalized = inputs.prepare(host)
            features.append(inputs.features(normalized).cpu().numpy())
            grids.append(inputs.grid(normalized).cpu())
    return np.asarray(features), torch.cat(grids)


def costs_for(records, size, objective):
    key = ('normalized_host_input_to_logits_batch_ms' if objective == 'latency'
           else 'gpu_energy_j_per_image')
    costs = []
    for record in records:
        values = [row.get(key) for row in record['timings'][str(size)]]
        if any(value is None or not np.isfinite(value) or value <= 0 for value in values):
            return None
        cost = float(np.median(values))
        costs.append(cost if objective == 'latency' else cost * size)
    return np.asarray(costs)


def fit_models(features, targets, grids, seed, epochs):
    stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=20, random_state=seed)
    stump.fit(features[:, [8]], targets)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=seed).fit(features, targets)
    models = {'gradient_stump': TreeRouter(stump.tree_, [8]), 'batch_tree_d3': TreeRouter(tree.tree_)}
    binary = targets >= np.median(targets)
    if len(np.unique(binary)) == 1:
        models['batch_logistic'] = ConstantRouter(float(targets.mean()))
    else:
        scaler = StandardScaler().fit(features)
        classifier = LogisticRegression(C=.25, max_iter=2000, random_state=seed).fit(scaler.transform(features), binary)
        models['batch_logistic'] = LogisticRouter(scaler, classifier)
    xgb = XGBRegressor(n_estimators=120, max_depth=2, learning_rate=.04,
                       min_child_weight=20, reg_lambda=15, reg_alpha=.2,
                       subsample=.85, colsample_bytree=.85, tree_method='hist',
                       device='cuda', n_jobs=4, random_state=seed).fit(features, targets)
    models['batch_xgb_d2'] = XGBRouter(xgb.get_booster())
    tiny, _ = train_tinycnn(grids, targets.astype('float32'), seed, epochs, 'cuda')
    models['batch_tinycnn'] = tiny
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static-report', type=Path, required=True)
    parser.add_argument('--static-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', default='42,45,48')
    parser.add_argument('--batch-sizes', default='8,16,32,64,128')
    parser.add_argument('--budgets-pp', default='.5,1,1.5,2')
    parser.add_argument('--training-shuffles', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--maximum-candidates', type=int, default=8)
    parser.add_argument('--scope', choices=('global', 'within-architecture'), default='global')
    args = parser.parse_args()
    report = json.loads(args.static_report.read_text())
    records = report['results']
    if len(records) != 213 or any(record['status'] != 'complete' for record in records):
        raise RuntimeError('All 213 GPU static records must finish before router training')
    if any(record.get('validation_images') != 5000 for record in records):
        raise RuntimeError('Full validation predictions are required')
    labels = np.load(args.cache_dir / 'validation_labels.npy')
    images = np.load(args.cache_dir / 'validation_rgb128.npy', mmap_mode='r')
    data_marker = json.loads((args.cache_dir / 'router_data.json').read_text())
    if report['validation_indices_sha256'] != data_marker['validation_indices_sha256']:
        raise RuntimeError('Validation split fingerprint mismatch')
    sizes = [int(value) for value in args.batch_sizes.split(',')]
    budgets = [float(value) for value in args.budgets_pp.split(',')]
    settings = {'batch_sizes': sizes, 'budgets_pp': budgets,
                'training_shuffles': args.training_shuffles, 'epochs': args.epochs,
                'maximum_candidates': args.maximum_candidates, 'scope': args.scope}
    if not set(sizes) <= set(report['batch_sizes']):
        raise ValueError('Missing measured batch costs')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = SharedBatchInputs()
    image_features = []
    with torch.inference_mode():
        for start in range(0, len(images), 128):
            host = torch.from_numpy(np.array(images[start:start+128])).pin_memory()
            image_features.append(inputs.image_features(inputs.prepare(host)).cpu().numpy())
    image_features = np.concatenate(image_features)
    fingerprint = file_hash(args.static_report)
    architectures = (['global'] if args.scope == 'global' else
                     sorted({record['architecture'] for record in records}))
    for seed in map(int, args.seeds.split(',')):
        fit, calibration = split_validation(labels, seed)
        fit_batches = {size: make_batches(fit, size, seed + size, args.training_shuffles) for size in sizes}
        cal_batches = {size: make_batches(calibration, size, seed + size) for size in sizes}
        # This offline cache accelerates fitting only. Timed inference recomputes
        # features and grids from RGB input for every dispatched batch.
        fit_inputs = {size: extract(inputs, images, fit_batches[size]) for size in sizes}
        cal_inputs = {size: extract(inputs, images, cal_batches[size]) for size in sizes}
        x = np.concatenate([fit_inputs[size][0] for size in sizes])
        grids = torch.cat([fit_inputs[size][1] for size in sizes])
        for architecture in architectures:
            full_candidates = records if architecture == 'global' else [record for record in records if record['architecture'] == architecture]
            if architecture == 'global':
                candidates, baseline_name = global_candidates(full_candidates)
            else:
                candidates = shortlist(full_candidates, args.maximum_candidates)
                baseline_name = f'{architecture}__p0__fp32'
            names = [record['name'] for record in candidates]
            baseline = names.index(baseline_name)
            predictions = np.stack([np.load(args.static_dir / f'{name}_validation_predictions.npy') for name in names], axis=1)
            correct = predictions == labels[:, None]
            for objective in ('latency', 'energy'):
                marker = args.output_dir / f'{architecture}_{objective}_seed{seed}.json'
                prior = json.loads(marker.read_text()) if marker.exists() else {}
                if (prior.get('static_report_sha256') == fingerprint
                        and prior.get('settings') == settings and prior.get('status') == 'complete'):
                    if all(Path(item['artifact']).exists()
                           and file_hash(Path(item['artifact'])) == item['sha256']
                           and Path(item.get('engine', '')).is_file()
                           and file_hash(Path(item['engine'])) == item['engine_sha256']
                           for item in prior['routers']):
                        print(f'Skip {marker.stem}', flush=True)
                        continue
                costs = {size: costs_for(candidates, size, objective) for size in sizes}
                if any(value is None for value in costs.values()):
                    write_json(marker, {'status': 'unavailable', 'objective': objective,
                                        'reason': 'Candidate energy/cost telemetry unavailable',
                                        'static_report_sha256': fingerprint})
                    continue
                targets = np.concatenate([batch_targets(fit_batches[size], correct, costs[size], baseline)[0] for size in sizes])
                started = time.perf_counter()
                models = fit_models(x, targets, grids, seed, args.epochs)
                metadata = {'status': 'complete', 'architecture': architecture, 'objective': objective,
                            'seed': seed, 'models': names, 'baseline_index': baseline,
                            'all_static_models': [record['name'] for record in full_candidates],
                            'scope': args.scope, 'baseline_name': baseline_name,
                            'pool_policy': ('All 213 configurations across eight architectures; bounded runtime cache does not restrict choices.' if architecture == 'global' else 'Legacy within-architecture shortlist.'),
                            'static_report_sha256': fingerprint, 'validation_indices_sha256': data_marker['validation_indices_sha256'],
                            'fit_positions': fit.tolist(), 'calibration_positions': calibration.tolist(),
                            'training_shuffles': args.training_shuffles, 'epochs': args.epochs,
                            'settings': settings,
                            'routers': [], 'cost_boundary': report['timing_boundary'],
                            'note': 'Calibration costs exclude router overhead and runtime cache misses. End-to-end testing includes both; estimates are not savings claims.'}
                for method, model in models.items():
                    model = model.cuda().eval()
                    path = args.output_dir / f'{architecture}_{objective}_{method}_seed{seed}.pth'
                    torch.save(model.cpu(), path)
                    kind = 'grid' if method == 'batch_tinycnn' else 'features'
                    engine_path, parity_error = build_router(model, kind, args.output_dir / 'engines', path.stem)
                    deployed = RouterRunner(engine_path)
                    policies = {}
                    with torch.inference_mode():
                        for size in sizes:
                            _, counts, lengths = batch_targets(cal_batches[size], correct, costs[size], baseline)
                            values = cal_inputs[size][1] if method == 'batch_tinycnn' else torch.from_numpy(cal_inputs[size][0])
                            # Calibrate the deployed runtime, including its actual
                            # floating-point behavior near policy thresholds.
                            scores = np.asarray([float(deployed.run(value[None].cuda().contiguous()).item()) for value in values])
                            policies[str(size)] = {str(budget): calibrate(scores, counts, lengths, costs[size], baseline, budget) for budget in budgets}
                    metadata['routers'].append({'method': method, 'artifact': str(path.resolve()),
                                                'sha256': file_hash(path), 'policies': policies,
                                                'input': kind, 'engine': str(engine_path.resolve()),
                                                'engine_sha256': file_hash(engine_path),
                                                'engine_torch_max_error': parity_error})
                    del deployed
                metadata['training_seconds'] = time.perf_counter() - started
                for size in sizes:
                    order = np.argsort(costs[size], kind='stable')
                    ranked_correct = correct[:, order]
                    ranks = ranked_correct.argmax(1)
                    ranks[~ranked_correct.any(1)] = int(np.flatnonzero(order == baseline)[0])
                    image_targets = ranks / max(1, len(order)-1)
                    estimator = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20,
                                                       random_state=seed).fit(image_features[fit], image_targets[fit])
                    _, counts, lengths = batch_targets(cal_batches[size], correct, costs[size], baseline)
                    for percentile in (10, 25, 50, 70, 80, 90, None):
                        aggregation = 'mean' if percentile is None else f'p{percentile}'
                        method = f'image_tree_{aggregation}_b{size}'
                        model = AggregatedTreeRouter(TreeRouter(estimator.tree_), size, percentile)
                        path = args.output_dir / f'{architecture}_{objective}_{method}_seed{seed}.pth'
                        torch.save(model, path)
                        engine_path, parity_error = build_router(model, 'image_features', args.output_dir / 'engines',
                                                                path.stem, input_shape=(size, 13))
                        deployed = RouterRunner(engine_path)
                        scores = []
                        with torch.inference_mode():
                            for batch in cal_batches[size]:
                                values = image_features[batch]
                                if len(values) < size:
                                    values = np.concatenate((values, np.repeat(values[-1:], size-len(values), axis=0)))
                                scores.append(float(deployed.run(torch.from_numpy(values).cuda().contiguous()).item()))
                        policies = {str(size): {str(budget): calibrate(scores, counts, lengths, costs[size], baseline, budget)
                                               for budget in budgets}}
                        metadata['routers'].append({'method': method, 'artifact': str(path.resolve()),
                                                    'sha256': file_hash(path), 'policies': policies,
                                                    'input': 'image_features', 'aggregation': aggregation,
                                                    'tail_policy': 'Duplicate last feature row to fixed batch size; count only real images.',
                                                    'engine': str(engine_path.resolve()), 'engine_sha256': file_hash(engine_path),
                                                    'engine_torch_max_error': parity_error})
                        del deployed, model
                metadata['training_seconds'] = time.perf_counter() - started
                write_json(marker, metadata)
                print(f'Finished {marker.stem}', flush=True)
                del models
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
