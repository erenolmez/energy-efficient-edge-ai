"""Render audited p80 measurements; retain every prior p90 artifact."""
import json
from pathlib import Path

folder=Path(__file__).resolve().parent
source=folder.parents[1]/'pc_tests/model_pipeline/reports/p80_b8_32_64_128'
audit=json.loads((source/'completion_audit.json').read_text())
summary=json.loads((source/'p80_summary.json').read_text())
assert audit['complete'] and summary['complete'] and len(summary['rows'])==84
lookup={(r['name'],r['batch_size']):r for r in summary['rows']}
snapshot=json.loads((folder/'numerical_snapshot.json').read_text())
display={r['architecture']:r['display_name'] for r in snapshot['fp32_classes']['models']}
lines=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',
       r'Architecture & Precision & Acc.\% & b8: L\% / E\% & b32: L\% / E\% & b64: L\% / E\% & b128: L\% / E\%\\\midrule']
for item in (r for r in summary['rows'] if r['batch_size']==128):
    name=item['name']
    arch,prune,precision=name.split('__')
    assert prune=='p80'
    assert all(abs(lookup[name,b]['accuracy']-item['accuracy'])<1e-8 for b in (8,32,64,128))
    cells=[display[arch],precision.upper(),f"{item['accuracy']:.2f}"]
    for batch in (8,32,64,128):
        row=lookup[name,batch]
        cells.append(f"{row['latency_saving_percent']:+.1f} / {row['energy_saving_percent']:+.1f}")
    lines.append(' & '.join(cells)+r'\\')
lines.extend([r'\bottomrule',r'\end{tabular}'])
(folder/'p80_savings.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Saved 21 p80 rows; accuracy identical across all four batch sizes.')
