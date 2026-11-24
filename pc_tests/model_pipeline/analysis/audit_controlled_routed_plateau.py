"""Verify the router-only rerun and summarize without pooling with the original replay."""
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median
import sys


def main():
    root = Path(sys.argv[1])
    state = json.loads((root/'study.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    if state['status'] != 'complete_pending_audit' or len(state['rows']) != 15 or plan['expected_windows'] != 15:
        raise ValueError(f"Incomplete controlled replay: {state['status']} ({len(state['rows'])}/15)")
    telemetry = defaultdict(list)
    with (root/'telemetry.jsonl').open() as f:
        for line in f:
            t = json.loads(line)
            telemetry[t['window']].append(t)
    groups = defaultdict(list)
    thresholds = plan['minimum_warmup_seconds'], plan['minimum_warmup_batches']
    assert thresholds == (30, 50) and plan['cool_start_max_c'] == 65
    assert plan['cache_reset_after_warmup'] and plan['capacity'] == 2
    thermal_windows, timed_temperatures, cooldown_temperatures, warmups = [], [], [], []
    for r in state['rows']:
        b, repeat = r['batch'], r['repeat']
        assert r['mode']=='route' and b in plan['batches']
        assert r['warmup_s'] >= 30 and r['warmup_batches'] >= 50
        c = r['cooldown']
        assert c['temperature_c'] <= 65 and c['seconds'] >= 15 and not c['throttle_reasons'] & 96
        warmups.append(r['warmup_s'])
        cooldown_temperatures.append(c['temperature_c'])
        assert r['images'] == 10_000 and r['batches'] == math.ceil(r['images']/b)
        assert sum(x['images'] for x in r['selections'].values()) == r['images']
        assert sum(x['batches'] for x in r['selections'].values()) == r['batches']
        assert math.isclose(r['ms_per_batch']*r['batches'],r['ms_per_image']*r['images'],rel_tol=1e-12)
        window = f'r{repeat}_b{b}_route'
        assert telemetry[window+':warmup'] and telemetry[window+':timed']
        timed = telemetry[window+':timed']+[r['gpu_at_timed_start'],r['gpu_at_timed_end']]
        assert all(t['temperature_c'] is not None and t['temperature_c'] < plan['hard_stop_c']
                   and t['throttle_reasons'] is not None for t in timed)
        timed_temperatures.extend(t['temperature_c'] for t in timed)
        if any(t['throttle_reasons'] & 96 for t in timed):
            thermal_windows.append(window)
        groups[b].append(r)
    result = []
    for b in plan['batches']:
        g = groups[b]
        assert len(g)==3 and {r['repeat'] for r in g}=={0,1,2}
        assert len({r['predictions_sha256'] for r in g})==1
        result.append(dict(batch=b,loads=[r['loads'] for r in g],
                           **{k:median(r[k] for r in g) for k in
                              ('test_accuracy','ms_per_batch','ms_per_image','gpu_mj_per_image')}))
    print(json.dumps(dict(rows=result,timed_temperature_range_c=[min(timed_temperatures),max(timed_temperatures)],
                          cooldown_c=[min(cooldown_temperatures),max(cooldown_temperatures)],
                          warmup_seconds=[min(warmups),max(warmups)],
                          thermal_windows=thermal_windows),indent=2))


if __name__=='__main__':
    main()
