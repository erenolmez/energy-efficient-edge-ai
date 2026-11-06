"""Saved full-routing evidence only; no CUDA, images, labels or policy tuning."""
from pathlib import Path
import hashlib
import json
import math
from statistics import median

REPO = Path(__file__).resolve().parents[3]
OUTPUT = REPO / 'pc_tests/model_pipeline/reports/batch_plateau_20261103'


def summarize(row):
    measurements = row['measurements']
    if row['status'] != 'complete' or len(measurements) != 5:
        raise ValueError('Expected complete five-repeat historical row')
    n = measurements[0]['test_images']
    if any(m['test_images'] != n for m in measurements):
        raise ValueError('Inconsistent image denominator')
    def med(key, scale=1):
        values = [float(m[key])*scale for m in measurements]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite saved measurement')
        return median(values)
    batch = row['batch_size']
    return dict(id=row['id'], seed=row['seed'], objective=row['objective'],
                budget_pp=row['budget_pp'], batch=batch,
                ms_image=med('rgb_input_to_prediction_seconds', 1000/n),
                gpu_mj_image=med('gpu_energy_j_per_image', 1000),
                decision_ms_batch=med('mean_gpu_decision_ms'),
                selected_ms_batch=med('mean_gpu_selected_inference_ms'),
                loads=med('runtime_engine_loads'), evictions=med('runtime_engine_evictions'),
                loads_per_1000_images=med('runtime_engine_loads', 1000/n),
                gpu_average_watts=med('gpu_average_watts'),
                end_temperature_c=med('gpu_temperature_c'),
                accuracy=med('test_accuracy'), eligible_static=row['best_eligible_static'],
                selection_counts=measurements[0]['selected_images'],
                selection_repeat_consistent=all(m['selected_images']==measurements[0]['selected_images'] for m in measurements))


def percent_change(after, before):
    if before <= 0:
        raise ValueError('Positive denominator required')
    return 100*(after/before-1)


def main():
    paths = [REPO/'pc_tests/model_pipeline/reports/pc_routes_correctness.json',
             REPO/'progress_reports/2026-10-29/report_data.json']
    raw, publication = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    rows = [summarize(r) for r in raw['results']]
    if len(rows) != 120 or len({r['id'] for r in rows}) != 120:
        raise ValueError('Expected 120 unique historical settings')
    report_rows = [r for r in publication['fixed1pp_latency_rows'] if r['batch'] != 1]
    fixed = [r for r in rows if r['objective']=='latency' and r['budget_pp']==1]
    for r in fixed:
        published = next(p for p in report_rows if p['seed']==r['seed'] and p['batch']==r['batch'])
        if not math.isclose(r['ms_image'], published['ms_image'], rel_tol=1e-8):
            raise ValueError('Raw measurements disagree with October29 latency')
        if not math.isclose(r['gpu_mj_image']/1000, published['j_image'], rel_tol=1e-8):
            raise ValueError('Raw measurements disagree with October29 energy')
    medians=[]
    for b in (8,16,32,64,128):
        group=[r for r in fixed if r['batch']==b]
        medians.append(dict(batch=b, **{k:median(r[k] for r in group)
                       for k in ('ms_image','gpu_mj_image','decision_ms_batch','loads','evictions')}))
    paired=[]
    for r in rows:
        if r['batch'] not in (64,128):
            continue
        base=next(x for x in rows if x['batch']==32 and
                  (x['seed'],x['objective'],x['budget_pp'])==(r['seed'],r['objective'],r['budget_pp']))
        paired.append(dict(seed=r['seed'],objective=r['objective'],budget_pp=r['budget_pp'],
                           batch=r['batch'],latency_change_pct=percent_change(r['ms_image'],base['ms_image']),
                           energy_change_pct=percent_change(r['gpu_mj_image'],base['gpu_mj_image']),
                           eligible_changed=r['eligible_static']!=base['eligible_static']))
    result=dict(status='saved_data_audited', sources_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                boundary=raw['timing_boundary'], energy_scope=raw['energy_scope'],
                fixed_latency_1pp_rows=fixed, fixed_latency_1pp_seed_medians=medians,
                all_120_rows=rows, paired_changes_from_b32=paired,
                causal_power_limit_established=False, test_images_accessed=False,
                qualification='Saved test-summary analysis, not test-based selection. Batch-dependent frozen policies/model mixes and sequential sessions differ. CUDA selected-event span may include host/cache waits; not pure kernel time. End temperature and average power do not establish clocks or throttling.')
    OUTPUT.mkdir(exist_ok=True,parents=True)
    (OUTPUT/'routed_analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Full routed-system batch plateau: saved-data audit', '',
           'Reproduces the October29 latency/1pp figure from five-repeat historical medians.',
           'Batch1 is excluded from scaling comparisons: its six-resident-model setup differs from historical213/cache2.', '',
           '| Batch | Median ms/image | Median GPU mJ/image | Decision ms/batch | Loads /10000 images | Evictions |',
           '|---|---:|---:|---:|---:|---:|']
    for r in medians:
        lines.append(f"| {r['batch']} | {r['ms_image']:.4f} | {r['gpu_mj_image']:.3f} | {r['decision_ms_batch']:.3f} | {r['loads']:.0f} | {r['evictions']:.0f} |")
    base=next(r for r in medians if r['batch']==32)
    lines+=['', 'These are medians across the same three seeds, not paired confidence intervals.', '']
    for r in medians[-2:]:
        lines.append(f"Batch{r['batch']} versus32: latency {percent_change(r['ms_image'],base['ms_image']):+.2f}%; GPU energy {percent_change(r['gpu_mj_image'],base['gpu_mj_image']):+.2f}% per image (positive means worse).")
    lines+=['', '## Interpretation and missing telemetry', '',
            'The reported routed-system curve has diminishing benefits after32. This is not a universal fixed-classifier hardware limit.',
            'Batch-dependent selection/mix, cache misses, and session variation remain confounders. All120 settings, negatives and accuracy failures are retained in the JSON; no settings were selected using test results.',
            'Saved GPU decision and selected-engine event durations are not a complete causal decomposition: the latter may include GPU idle time during host loading or synchronization.',
            'Existing evidence contains end-window temperature and counter-derived mean power, not time-aligned clock, utilization or throttle traces.',
            'New diagnostics need UTC/monotonic window markers; GPU temperature/power/utilization/memory; SM and memory clocks; performance state; supported throttle/event reason flags; per-window model choice/load/eviction counts; and separately timed host transfer, selection, loading and classifier execution.',
            'Log unsupported fields as unavailable. GPU energy excludes CPU and the rest of the PC. Never infer throttling merely from temperature or power.', '',
            'Next: isolated validation-only frozen-policy diagnostic with same-session static controls, balanced batch order, fixed warm-up and telemetry. No retraining, policy recalibration or test-based tuning. No old worker is resumed.']
    (OUTPUT/'ROUTED_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],rows=len(rows),seed_medians=medians),indent=2))


if __name__=='__main__':
    main()
