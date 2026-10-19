"""Audit saved routing triplets; never launch inference or change policies."""
import hashlib
import json
import statistics
from pathlib import Path

folder = Path(__file__).resolve().parent
source = folder.parents[1] / 'pc_tests/model_pipeline/reports/pc_routes_correctness.json'
raw = source.read_bytes()
rows = json.loads(raw)['results']
lookup = {(r['objective'], r['seed'], r['batch_size'], r['budget_pp']): r for r in rows}
records = []
for objective in ('latency', 'energy'):
    for seed in (42, 45, 48):
        for batch in (8, 16, 32, 64, 128):
            triplet = []
            for budget in (1., 1.5, 2.):
                row = lookup[objective, seed, batch, budget]
                measured = row['measurements']
                static = row['comparator_measurements'][row['best_eligible_static']]
                triplet.append(dict(budget=budget, comparator=row['best_eligible_static'],
                    penalty=row['policy']['penalty'],
                    route_ms=statistics.median(m['rgb_input_to_prediction_seconds'] / m['test_images'] * 1000 for m in measured),
                    static_ms=statistics.median(m['rgb_input_to_prediction_seconds'] / m['test_images'] * 1000 for m in static),
                    median_loads=statistics.median(m['runtime_engine_loads'] for m in measured),
                    median_evictions=statistics.median(m['runtime_engine_evictions'] for m in measured),
                    accuracy=statistics.mean(m['test_accuracy'] for m in measured),
                    accuracy_pass=row['test_budget_passed'],
                    latency_saving=row['latency_saving_vs_best_eligible_static_percent'],
                    energy_saving=row['energy_saving_vs_best_eligible_static_percent']))
            records.append(dict(objective=objective, seed=seed, batch=batch, triplet=triplet))
flips = [r for r in records if r['triplet'][0]['latency_saving'] > 0 > r['triplet'][1]['latency_saving']]
recoveries = [r for r in flips if r['triplet'][2]['latency_saving'] > 0]
assert len(flips) == 8 and len(recoveries) == 3
assert all(r['seed'] == 48 for r in flips)
assert all(not r['triplet'][2]['accuracy_pass'] for r in recoveries)
report = dict(source_sha256=hashlib.sha256(raw).hexdigest(),
              positive_1_negative_1_5_cases=len(flips), positive_again_at_2_cases=len(recoveries),
              all_recoveries_fail_accuracy=True, cases=records)
example = [lookup['latency',48,128,b] for b in (1.,1.5,2.)]
fixed = example[0]['comparator_measurements'][example[0]['best_eligible_static']]
fixed_l = statistics.median(m['rgb_input_to_prediction_seconds']/m['test_images']*1000 for m in fixed)
fixed_e = statistics.median(m['gpu_energy_j_per_image']*1000 for m in fixed)
report['fixed_reference_example'] = dict(name=example[0]['best_eligible_static'],
    source_budget=1, latency_ms_per_image=fixed_l, energy_mj_per_image=fixed_e,
    interpretation='Counterfactual renormalization against one held-constant measured benchmark, not fresh paired controls',
    rows=[dict(budget=r['budget_pp'],
        fixed_latency_saving=100*(fixed_l-statistics.median(m['rgb_input_to_prediction_seconds']/m['test_images']*1000 for m in r['measurements']))/fixed_l,
        fixed_energy_saving=100*(fixed_e-statistics.median(m['gpu_energy_j_per_image']*1000 for m in r['measurements']))/fixed_e)
        for r in example])
(folder / 'allowance_change_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print('Verified 8 positive-to-negative cases; 3 positive-at-2 recoveries, all fail accuracy.')
