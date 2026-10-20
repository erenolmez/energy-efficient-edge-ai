"""CPU-only validation complementarity audit. Never opens test predictions/labels."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation, make_batches
from pc_tests.model_pipeline.runtime.global_candidate_pool import global_candidates


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_validation(report_path, static_dir, cache_dir):
    report = json.loads(Path(report_path).read_text())
    records, baseline = global_candidates(report['results'])
    labels_path = Path(cache_dir) / 'validation_labels.npy'
    labels = np.load(labels_path, allow_pickle=False)
    marker_path = Path(cache_dir) / 'router_data.json'
    marker = json.loads(marker_path.read_text())
    identity = report['validation_indices_sha256']
    if len(labels) != 5000 or labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or ((labels < 0) | (labels >= 100)).any():
        raise ValueError('Expected 5000 integer CIFAR-100 validation labels')
    if marker['validation_indices_sha256'] != identity:
        raise ValueError('Validation split identity mismatch')
    columns, hashes = [], {}
    for row in records:
        if row.get('validation_images') != len(labels) or row['validation_indices_sha256'] != identity:
            raise ValueError('Incomplete or mismatched validation record')
        path = Path(static_dir) / (row['name'] + '_validation_predictions.npy')
        predictions = np.load(path, allow_pickle=False)
        if predictions.shape != labels.shape or not np.issubdtype(predictions.dtype, np.integer) or ((predictions < 0) | (predictions >= 100)).any():
            raise ValueError('Invalid validation predictions: ' + row['name'])
        correct = predictions == labels
        if not np.isclose(100 * correct.mean(), row['validation_accuracy'], atol=1e-8):
            raise ValueError('Recorded accuracy disagrees with predictions')
        columns.append(correct)
        hashes[row['name']] = digest(path)
    provenance = dict(static_report_sha256=digest(report_path), validation_labels_sha256=digest(labels_path),
                      router_data_sha256=digest(marker_path), validation_indices_sha256=identity,
                      validation_prediction_sha256=hashes, test_data_accessed=False)
    return records, labels, np.stack(columns, 1), baseline, provenance


def audit_partition(correct, labels, positions, names, baseline, seed):
    q, y = correct[positions], labels[positions]
    integers = q.astype(np.int32)
    # entry [a,b]: candidate a rescues an error made by candidate b.
    rescues = integers.T @ (1 - integers)
    accuracies = 100 * q.mean(0)
    best = int(accuracies.argmax())
    architecture_oracles = {}
    for architecture in sorted({name.split('__')[0] for name in names}):
        columns = [i for i, name in enumerate(names) if name.split('__')[0] == architecture]
        architecture_oracles[architecture] = 100 * float(q[:, columns].any(1).mean())
    p0_columns = [i for i, name in enumerate(names) if name.endswith('__p0__fp32')]
    batches = {}
    for size in (1, 8, 16, 32, 64, 128):
        partitions = make_batches(np.arange(len(q)), size, seed + size)
        counts = np.stack([q[b].sum(0) for b in partitions])
        batches[str(size)] = dict(images=len(q), batches=len(partitions),
            ideal_one_model_per_batch_accuracy=100 * float(counts.max(1).sum()) / len(q),
            best_static_accuracy=float(accuracies[best]))
    class_summary = []
    for label in np.unique(y):
        subset = q[y == label]
        ca = 100 * subset.mean(0)
        winner = int(ca.argmax())
        class_summary.append(dict(class_id=int(label), images=len(subset), best_candidate=names[winner],
            best_accuracy=float(ca[winner]), reference_accuracy=float(ca[baseline]),
            per_image_oracle_accuracy=100 * float(subset.any(1).mean())))
    return dict(images=len(q), reference_accuracy=float(accuracies[baseline]),
        best_static_model=names[best], best_static_accuracy=float(accuracies[best]),
        per_image_oracle_accuracy=100 * float(q.any(1).mean()),
        architecture_oracle_accuracy=architecture_oracles,
        eight_unpruned_fp32_oracle_accuracy=100 * float(q[:, p0_columns].any(1).mean()),
        unsolved_by_any=int((~q.any(1)).sum()),
        accuracy_by_candidate=accuracies.tolist(), pairwise_rescue_counts=rescues.tolist(),
        per_class=class_summary, batch_oracles=batches,
        reference_rescues=[dict(model=name, rescued=int(rescues[i, baseline]),
                               introduced=int(rescues[baseline, i])) for i, name in enumerate(names)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--static-report', type=Path, required=True)
    p.add_argument('--static-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    records, labels, correct, baseline, provenance = load_validation(args.static_report, args.static_dir, args.cache_dir)
    names = [r['name'] for r in records]
    baseline_index = names.index(baseline)
    result = dict(protocol_version=1, scope='validation_only', models=names,
                  baseline=baseline, provenance=provenance, partitions={},
                  note='Oracle uses labels offline only: not a runtime accuracy or savings claim. Class winners are descriptive, not runtime ground-truth dispatch.')
    result['descriptive_all_validation'] = audit_partition(correct, labels, np.arange(len(labels)), names, baseline_index, 42)
    for seed in (42, 45, 48):
        fit, calibration = split_validation(labels, seed)
        result['partitions'][str(seed)] = dict(fit=audit_partition(correct, labels, fit, names, baseline_index, seed),
            calibration=audit_partition(correct, labels, calibration, names, baseline_index, seed),
            fit_positions=fit.tolist(), calibration_positions=calibration.tolist())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'validation_complementarity.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(path)
    all_rows = result['descriptive_all_validation']
    text = ['# Validation-only complementarity audit', '',
            f'All {len(names)} candidates from eight architectures; 5,000 validation images. No test labels or predictions opened.',
            f"Reference {baseline}: {all_rows['reference_accuracy']:.2f}%.",
            f"Best static: {all_rows['best_static_accuracy']:.2f}%; ideal per-image selector: {all_rows['per_image_oracle_accuracy']:.2f}%.",
            f"Eight unpruned FP32 architectures alone: {all_rows['eight_unpruned_fp32_oracle_accuracy']:.2f}% ideal selector accuracy.",
            '', 'Oracle numbers are unattainable-label-access upper bounds, not deployable results or energy savings.',
            'Separate fit/calibration results for seeds 42/45/48 and provenance hashes are retained in JSON.', '',
            '| Batch | Ideal one-model-per-batch accuracy |', '|---:|---:|']
    for size, row in all_rows['batch_oracles'].items():
        text.append(f"| {size} | {row['ideal_one_model_per_batch_accuracy']:.2f}% |")
    text += ['', '## Largest error rescues relative to reference', '', '| Candidate | Rescued | Introduced |', '|---|---:|---:|']
    for row in sorted(all_rows['reference_rescues'], key=lambda r: -r['rescued'])[:12]:
        text.append(f"| {row['model']} | {row['rescued']} | {row['introduced']} |")
    (args.output_dir / 'validation_complementarity.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({k: all_rows[k] for k in ('reference_accuracy', 'best_static_accuracy', 'per_image_oracle_accuracy', 'unsolved_by_any', 'batch_oracles')}))


if __name__ == '__main__':
    main()
