"""Summarize frequency-ranked residency, excluding thermally invalid windows."""
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median
import sys


def summarize(root):
    root = Path(root)
    state = json.loads((root/'study.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    assert plan['reject_timed_thermal_flags'] and not plan['cache_reset_after_warmup']
    mode = 'fixed_resnet101' if 'fixed_classifier' in plan else 'route'
    reserve = plan.get('min_gpu_free_bytes',plan.get('minimum_gpu_free_bytes'))
    traces = defaultdict(list)
    for line in (root/'telemetry.jsonl').read_text().splitlines():
        reading=json.loads(line)
        traces[reading['window']].append(reading)
    results = defaultdict(list)
    rejected = list(state['rejected'])
    temperatures = []
    for row in state['rows']:
        b, i = row['batch'], row['repeat']
        assert row['mode'] == mode
        w = f'r{i}_b{b}_{mode}:timed'
        samples = traces[w]
        if not samples or any(r['temperature_c'] is None or r['throttle_reasons'] is None
                              or r['temperature_c']>=plan['hard_stop_c'] or r['throttle_reasons']&96 for r in samples):
            rejected.append(dict(batch=b,repeat=i,error='Timed telemetry missing or thermal flag detected in audit'))
            continue
        assert row['gpu_at_timed_start']['temperature_c'] <= plan.get('timing_start_max_c',65)
        assert not (row['gpu_at_timed_start']['throttle_reasons'] | row['gpu_at_timed_end']['throttle_reasons'])&96
        assert row['warmup_s'] >= 30 and row['warmup_batches'] >= 50
        assert row['min_gpu_free_bytes'] >= reserve
        if mode == 'route':
            assert row['pinned_models'] == state['calibration_hot_sets'][str(b)][:len(row['pinned_models'])]
        else:
            assert row['pinned_models'] == [plan['fixed_classifier']]
            assert set(row['selections']) == {plan['fixed_classifier']} and row['loads'] == 0 and row['timed_disk_reads'] == 0
        assert row['images'] == 10_000 and row['batches'] == math.ceil(10_000/b)
        assert math.isclose(row['ms_per_batch']*row['batches'],row['ms_per_image']*row['images'],rel_tol=1e-12)
        assert sum(v['images'] for v in row['selections'].values())==row['images']
        assert sum(v['batches'] for v in row['selections'].values())==row['batches']
        temperatures.extend(r['temperature_c'] for r in samples)
        results[b].append(row)
    summary=[]
    for b in plan['batches']:
        g=results[b]
        assert len({r['repeat'] for r in g})==len(g)
        if g:
            assert len({r['predictions_sha256'] for r in g})==1
        summary.append(dict(batch=b,qualified_passes=len(g),calibration_ranked=state.get('calibration_hot_sets',{}).get(str(b),[]),
                            pinned=[len(r['pinned_models']) for r in g],timed_loads=[r['loads'] for r in g],
                            disk_reads=[r['timed_disk_reads'] for r in g],
                            **{k:median(r[k] for r in g) if g else None for k in
                               ('test_accuracy','ms_per_image','ms_per_batch','gpu_mj_per_image')}))
    return dict(status=state['status'],batches=summary,rejected=rejected,
                temperature_range_c=[min(temperatures),max(temperatures)] if temperatures else None)


def main():
    print(json.dumps(summarize(sys.argv[1]),indent=2))


if __name__ == '__main__':
    main()
