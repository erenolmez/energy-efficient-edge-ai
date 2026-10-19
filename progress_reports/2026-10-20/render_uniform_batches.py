"""Combine audited batch64 and frozen batch128 uniform comparisons only."""
import hashlib
import json
from pathlib import Path

folder = Path(__file__).resolve().parent
reports = folder.parents[1] / 'pc_tests/model_pipeline/reports'
paths = {64: reports / 'random_b64/random_comparison_summary.json',
         128: folder / 'numerical_snapshot.json'}
data64 = json.loads(paths[64].read_text())
assert data64['complete'] and len(data64['rows']) == 72
data = {64: data64['rows'], 128: json.loads(paths[128].read_text())['random128']['rows']}
original = json.loads((reports / 'pc_routes_correctness.json').read_text())
old = {r['id']: r for r in original['results']}
lookups = {b: {(r['objective'], r['seed'], r['budget_pp'], r['policy']): r
               for r in rows} for b, rows in data.items()}
names = {'resnet101__p0__fp16': 'R101 FP16', 'resnet50__p0__fp16': 'R50 FP16'}
lines = [r'\begin{tabular}{@{}lrlrrrrrr@{}}', r'\toprule',
         r'& & & \multicolumn{3}{c}{Batch 64} & \multicolumn{3}{c}{Batch 128}\\',
         r'\cmidrule(lr){4-6}\cmidrule(l){7-9}',
         r'Obj./seed & $\delta$ & Benchmark & Acc. ML / R & ML: L / E & R: L / E & Acc. ML / R & ML: L / E & R: L / E\\\midrule']
differences = []
for objective in ('latency', 'energy'):
    for seed in (42, 45, 48):
        for budget in (.5, 1., 1.5, 2.):
            key = (objective, seed, budget)
            candidates = [lookups[b][(*key, 'learned')]['eligible_static'] for b in (64, 128)]
            assert candidates[0] == candidates[1]
            cells = [('L' if objective == 'latency' else 'E') + f'/{seed}',
                     f'{budget:g}', names[candidates[0]]]
            for batch in (64, 128):
                learned = lookups[batch][(*key, 'learned')]
                uniform = lookups[batch][(*key, 'uniform')]
                assert learned['eligible_static'] == uniform['eligible_static']
                accuracy = f"{learned['accuracy']:.2f}" + (r'$^*$' if not learned['mean_accuracy_pass'] else '')
                cells.append(accuracy + f" / {uniform['accuracy']:.2f}")
                for row in (learned, uniform):
                    cells.append(f"{row['latency_saving_percent']:+.1f} / {row['energy_saving_percent']:+.1f}")
                if batch == 64:
                    prior = old[learned['case']]
                    old_accuracy = sum(m['test_accuracy'] for m in prior['measurements']) / len(prior['measurements'])
                    differences.append({'case': learned['case'], 'previous_accuracy': old_accuracy,
                                        'fresh_accuracy': learned['accuracy'],
                                        'difference_pp': learned['accuracy'] - old_accuracy})
            lines.append(' & '.join(cells) + r'\\')
lines.extend([r'\bottomrule', r'\end{tabular}'])
(folder / 'uniform_batches_savings.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
audit = {'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
         'batch64_original_accuracy_comparison': differences,
         'batch64_max_absolute_accuracy_difference_pp': max(abs(r['difference_pp']) for r in differences),
         'comparison_cases_per_batch': 24, 'table_rows': 24}
(folder / 'uniform_batches_table_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
