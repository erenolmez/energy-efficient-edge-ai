"""Audit saved GPU execution, predictions, repeat coverage and engine hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_record(record):
    if record['status'] != 'complete':
        raise ValueError('Record is not complete')
    if record['test_images'] != 10000 or record['validation_images'] != 5000:
        raise ValueError('Incomplete accuracy coverage')
    if record['precision'] == 'int8' and not record['integer_compute_layers']:
        raise ValueError('No inspector evidence of integer compute')
    if set(record['timings']) != {'1','8','16','32','64','128'}:
        raise ValueError('Incomplete static batch-size coverage')
    count = 0
    for size, repeats in record['timings'].items():
        if len(repeats) != 5 or {row['repeat'] for row in repeats} != set(range(5)):
            raise ValueError('Incomplete or duplicate repeat coverage')
        for row in repeats:
            for key in ('forward_batch_ms','normalized_host_input_to_logits_batch_ms'):
                if not np.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError('Invalid measured latency')
            if row.get('gpu_energy_j_per_image') is not None:
                if row['gpu_energy_j_per_image'] <= 0 or row['energy_window_seconds'] < 3:
                    raise ValueError('Invalid measured energy window')
                if row['energy_window_images'] != row['timed_batches'] * int(size):
                    raise ValueError('Energy/image denominator mismatch')
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--static-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report = json.loads((args.report_dir / 'gpu_static.json').read_text())
    complete = [row for row in report['results'] if row['status'] == 'complete']
    if not args.allow_partial and len(complete) != 213:
        raise RuntimeError('Full audit requires 213 complete GPU targets')
    names = [row['name'] for row in complete]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate completed target')
    labels = {'test': np.load(args.cache_dir / 'test_labels.npy'),
              'validation': np.load(args.cache_dir / 'validation_labels.npy')}
    audited = []
    for record in complete:
        repeat_count = validate_record(record)
        engine = Path(record['engine_path'])
        if file_hash(engine) != record['engine_sha256']:
            raise ValueError(f"Engine checksum mismatch: {record['name']}")
        hashes = {}
        for split, truth in labels.items():
            path = args.static_dir / f"{record['name']}_{split}_predictions.npy"
            predicted = np.load(path)
            if predicted.shape != truth.shape or not np.isin(predicted, np.arange(100)).all():
                raise ValueError(f'Invalid {split} predictions')
            actual = 100 * float((predicted == truth).mean())
            if abs(actual - record[f'{split}_accuracy']) > 1e-9:
                raise ValueError('Saved prediction accuracy mismatch')
            hashes[split + '_predictions_sha256'] = file_hash(path)
        audited.append({'name': record['name'], 'engine_sha256': record['engine_sha256'],
                        'timing_repeats': repeat_count, **hashes})
    output = {'complete_targets': len(complete), 'expected_targets': 213,
              'partial': len(complete) != 213, 'timing_repeats': sum(row['timing_repeats'] for row in audited),
              'records': audited}
    (args.report_dir / 'gpu_static_audit.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(f'Audited {len(complete)} targets, {output["timing_repeats"]} timing repeats; engine and prediction hashes verified')


if __name__ == '__main__':
    main()
