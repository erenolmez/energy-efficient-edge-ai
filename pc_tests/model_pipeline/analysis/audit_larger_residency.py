"""Verify the larger residency experiment and summarize paired cache windows (CPU only)."""
import hashlib
import itertools
import json
from pathlib import Path
from statistics import median

import numpy as np

from pc_tests.model_pipeline.paths import PIPELINE


OUT = PIPELINE / 'reports/larger_residency_20261105'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def expected_loads(route, replays, capacity):
    """Count LRU misses from reuse distance, independently of the worker cache."""
    requests = route * (replays + 1)
    misses = 0
    for pos in range(len(route), len(requests)):
        previous = max(i for i in range(pos) if requests[i] == requests[pos])
        misses += len(set(requests[previous+1:pos])) >= capacity
    return misses


def main():
    study = json.loads((OUT / 'study.json').read_text())
    assert study['status'] == 'complete'
    plan = json.loads((OUT / 'plan.json').read_text())
    assert study['plan'] == plan
    for path, digest in study['protected_sha256'].items():
        assert sha(path) == digest, path
    orders = ('smallest_first', 'largest_first', 'architecture_round_robin')
    assert {(r['batch'], r['order']) for r in study['capacity']} == set(itertools.product(plan['batches'], orders))
    assert len(study['capacity']) == 9
    assert len(set(study['pool'])) == 8 and set(plan['route']) == set(range(8))
    reserve = plan['reserve_bytes']
    counts = {}
    for row in study['capacity']:
        assert row['count'] == len({r['name'] for r in row['models']}) == len(row['models'])
        assert row['status'] == 'verified' and row['verified_free_bytes'] >= reserve
        assert all(r['free_bytes'] >= reserve for r in row['models'])
        counts[row['batch'], row['order']] = row['count']
    telemetry = [json.loads(line) for line in (OUT / 'telemetry.jsonl').read_text().splitlines()]
    assert telemetry and all('error' not in r and r['temperature_c'] < plan['peak_stop_c'] for r in telemetry)
    skipped = {r['batch'] for r in study['skipped']}
    measured = set((1, *plan['batches'])) - skipped
    expected = set(itertools.product(measured, range(plan['repeats']), plan['capacities']))
    windows = {(r['batch'], r['repeat'], r['capacity']):r for r in study['windows']}
    assert set(windows) == expected and len(study['windows']) == len(expected)
    for (batch, repeat, capacity), row in windows.items():
        for key in ('prediction', 'reference'):
            path = Path(row[key + '_path']).resolve()
            assert path.parent == OUT.resolve() and sha(path) == row[key + '_sha256']
        actual = np.load(row['prediction_path'], allow_pickle=False)
        reference = np.load(row['reference_path'], allow_pickle=False)
        assert actual.shape == reference.shape == (len(plan['route'])*plan['timed_replays'], batch)
        assert actual.size == row['images'] and np.array_equal(actual, reference)
        assert np.issubdtype(actual.dtype, np.integer) and np.all((actual >= 0) & (actual < 100))
        loads = expected_loads(plan['route'], plan['timed_replays'], capacity)
        assert row['timed_loads'] == row['timed_evictions'] == loads
        phase = f'b{batch}:r{repeat}:c{capacity}'
        readings = [row['start'], row['end'], *(r for r in telemetry if r['phase'] == phase)]
        assert len(readings) > 2 and all(not r['clock_reasons'] & 32 for r in readings)
        assert row['accepted'] and row['prediction_mismatches'] == 0
        assert np.isfinite(row['elapsed_s']) and row['elapsed_s'] > 0
        energy = row['energy']
        assert energy['energy_window_images'] == actual.size
        assert np.isfinite(energy['gpu_energy_j']) and energy['gpu_energy_j'] > 0
        assert np.isclose(energy['gpu_energy_j_per_image'] * actual.size, energy['gpu_energy_j'])
        assert np.isclose(row['ms_per_image'] * actual.size, 1000*row['elapsed_s'])
    lines = [
        '# Larger-batch residency and eight-model cache replay',
        '',
        f"GPU: {study['gpu']}. Independent CPU audit passed for {len(windows)} timed windows.",
        '',
        '## Capacity: original engines, dedicated buffers, 1 GiB reserve',
        '',
        'All retained engines were re-executed on synthetic inputs while the entire pool remained resident. '
        'The original batch-1 TinyCNN was also kept resident. Counts are demonstrated for these loading orders and this session, not GPU maxima.',
        '',
        '| Batch | Smallest first | Largest first | Architecture round-robin |',
        '|---:|---:|---:|---:|',
    ]
    for batch in plan['batches']:
        lines.append(f"| {batch} | " + ' | '.join(str(counts[batch, o]) for o in orders) + ' |')
    lines += [
        '',
        '## Same eight-model route, six-slot LRU versus eight resident',
        '',
        'Eight unpruned FP16 engines, one per architecture. Fixed synthetic CPU-issued route, one selected engine per batch. '
        'Each window replays 96 batches after a separate warm-up trace. Cold setup and host file reads are excluded; '
        'timed misses deserialize already-loaded RAM bytes and recreate the GPU context and buffers. '
        'Both modes use the same synchronized batch boundary and device predictions.',
        '',
        '| Batch | c6 / c8 median ms/image | c6 / c8 median GPU mJ/image | Paired latency reduction | Paired GPU-energy reduction |',
        '|---:|---:|---:|---:|---:|',
    ]
    for batch in sorted(measured):
        a = [windows[batch,r,6] for r in range(plan['repeats'])]
        b = [windows[batch,r,8] for r in range(plan['repeats'])]
        t = lambda row: row['ms_per_image']
        e = lambda row: row['energy']['gpu_energy_j_per_image']*1000
        med = lambda rows, fn: median(map(fn, rows))
        gain = lambda fn: median(100*(1-fn(y)/fn(x)) for x,y in zip(a,b))
        lines.append(f"| {batch} | {med(a,t):.3f} / {med(b,t):.3f} | {med(a,e):.3f} / {med(b,e):.3f} | {gain(t):.2f}% | {gain(e):.2f}% |")
    lines += [
        '',
        f"Timed loads and evictions per window: {expected_loads(plan['route'], plan['timed_replays'], 6)} each at c6, zero at c8. "
        f"All {sum(r['images'] for r in windows.values()):,} timed predictions matched. "
        f"Peak sampled temperature: {max(r['temperature_c'] for r in telemetry)} °C. "
        'Energy is gross NVML GPU energy, not whole-PC energy.',
        '',
        'This measures the cost of cache misses for one deliberately switch-heavy sequence. '
        'It does not validate routing accuracy, natural cache hit rates, CUDA Graph dispatch, HPC or Jetson performance. '
        'Three paired repeats are descriptive, not confidence intervals. Raw timing, cold trace, telemetry and predictions remain in this directory.',
    ]
    if skipped:
        lines.append(f'No timed windows for batches {sorted(skipped)}: the eight-model pool did not pass reserve admission.')
    report = '\n'.join(lines) + '\n'
    with (OUT / 'RESULTS.md').open('x', encoding='utf-8') as handle:
        handle.write(report)
    print(report)


if __name__ == '__main__':
    main()
