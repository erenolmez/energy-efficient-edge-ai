"""Render savings with explicitly preserved experimental comparator identities."""
import json
from pathlib import Path

folder = Path(__file__).resolve().parent
snapshot = json.loads((folder / 'numerical_snapshot.json').read_text())
rows = snapshot['random128']['rows']
lookup = {(r['objective'], r['seed'], r['budget_pp'], r['policy']): r for r in rows}
names = {'resnet101__p0__fp16': 'R101 FP16', 'resnet50__p0__fp16': 'R50 FP16'}
lines = [r'\begin{tabular}{@{}lrllrrrrr@{}}', r'\toprule',
         r'Obj./seed & $\delta$ & Benchmark & Acc. ML & Acc. Rnd $\pm$s & L\% ML & L\% Rnd & E\% ML & E\% Rnd\\\midrule']
for objective in ('latency', 'energy'):
    for seed in (42, 45, 48):
        for budget in (.5, 1., 1.5, 2.):
            learned = lookup[objective, seed, budget, 'learned']
            uniform = lookup[objective, seed, budget, 'uniform']
            assert learned['eligible_static'] == uniform['eligible_static']
            accuracy = f"{learned['accuracy']:.2f}" + (r'$^*$' if not learned['mean_accuracy_pass'] else '')
            cells = [('L' if objective == 'latency' else 'E') + f'/{seed}', f'{budget:g}',
                     names[learned['eligible_static']], accuracy,
                     f"{uniform['accuracy']:.2f}" + r'$\pm$' + f"{uniform['accuracy_sd']:.2f}",
                     f"{learned['latency_saving_percent']:+.1f}", f"{uniform['latency_saving_percent']:+.1f}",
                     f"{learned['energy_saving_percent']:+.1f}", f"{uniform['energy_saving_percent']:+.1f}"]
            lines.append(' & '.join(cells) + r'\\')
lines.extend([r'\bottomrule', r'\end{tabular}'])
(folder / 'uniform_savings.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

costs = {(r['name'], r['batch_size']): r for r in snapshot['most_pruned_other']['rows']}
costs.update({(r['name'], 128): r for r in snapshot['most_pruned128']['rows']})
display = {r['architecture']: r['display_name'] for r in snapshot['fp32_classes']['models']}
lines = [r'\begin{tabular}{@{}llrrrrr@{}}', r'\toprule',
         r'Architecture & Precision & Acc.\% & b8: L\% / E\% & b32: L\% / E\% & b64: L\% / E\% & b128: L\% / E\%\\\midrule']
for row in snapshot['most_pruned128']['rows']:
    arch, _, precision = row['name'].split('__')
    cells = [display[arch], precision.upper(), f"{row['accuracy']:.2f}"]
    for batch in (8, 32, 64, 128):
        measured = costs[row['name'], batch]
        cells.append(f"{measured['latency_saving_percent']:+.1f} / {measured['energy_saving_percent']:+.1f}")
    lines.append(' & '.join(cells) + r'\\')
lines.extend([r'\bottomrule', r'\end{tabular}'])
(folder / 'pruned_savings.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Saved 24 uniform comparisons versus original eligible-static identities; 21 p90 rows versus fresh R101 FP32.')
