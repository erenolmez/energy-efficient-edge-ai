"""Freeze saved reports and generate numerical LaTeX tables; no GPU work."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
REPORTS=ROOT/'pc_tests/model_pipeline/reports'


def load(relative):
    return json.loads((REPORTS/relative).read_text(encoding='utf-8'))


def esc(value):
    return str(value).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def main():
    paths=['class_statistics/fp32_class_statistics.json',
           'class_statistics/all_213_class_statistics.json',
           'random_b128/random_routes_b128.json','random_b128/random_comparison_summary.json',
           'most_pruned_b128/most_pruned_routes.json','most_pruned_b128/most_pruned_summary.json',
           'most_pruned_b8_32_64/most_pruned_routes.json','most_pruned_b8_32_64/most_pruned_summary.json',
           'random_b64/random_routes_b64.json','pc_routes_correctness.json','gpu_static.json']
    data={p:load(p) for p in paths}
    snapshot=dict(captured_utc=datetime.now(timezone.utc).isoformat(),
                  sources={p:hashlib.sha256((REPORTS/p).read_bytes()).hexdigest() for p in paths},
                  random128=data['random_b128/random_comparison_summary.json'],
                  most_pruned128=data['most_pruned_b128/most_pruned_summary.json'],
                  most_pruned_other=data['most_pruned_b8_32_64/most_pruned_summary.json'],
                  fp32_classes=data['class_statistics/fp32_class_statistics.json'])
    ongoing=data['random_b64/random_routes_b64.json']
    snapshot['random64']=dict(status=ongoing.get('status','in progress'),
                             completed=len(ongoing['results']),expected=ongoing['expected_blocks'],
                             last_completed_utc=ongoing['results'][-1]['completed_utc'] if ongoing['results'] else None)
    assert snapshot['random128']['complete'] and len(snapshot['random128']['rows'])==72
    assert snapshot['most_pruned128']['complete'] and len(snapshot['most_pruned128']['rows'])==21
    assert snapshot['most_pruned_other']['complete'] and len(snapshot['most_pruned_other']['rows'])==63
    tables=[]
    def table(name,header,rows,columns):
        tables.extend(['\\newcommand{\\'+name+r'}{%', r'\begin{tabular}{'+columns+'}',r'\toprule',
                       header+r'\\\midrule', *[r' & '.join(row)+r'\\' for row in rows],
                       r'\bottomrule',r'\end{tabular}}'])
    def class_cell(classes):
        if len({c['accuracy_percent'] for c in classes})==1:
            return ' / '.join(esc(c['class_name'].replace('_',' ')) for c in classes)+f" ({classes[0]['accuracy_percent']:.0f}\\%)"
        return '; '.join(esc(c['class_name'].replace('_',' '))+f" ({c['accuracy_percent']:.0f}\\%)" for c in classes)
    rows=[]
    for r in snapshot['fp32_classes']['models']:
        rows.append([esc(r['display_name']),f"{r['overall_accuracy_percent']:.2f}\\%",
                     class_cell(r['worst']),class_cell(r['best']),class_cell(r['middle_ranked_classes'])])
    table('ClassTable',r'Architecture & Overall & Worst class & Best class(es) & Median-ranked classes',rows,
          '@{}p{2.65cm}rp{1.65cm}p{4.0cm}p{3.75cm}@{}')
    summary=snapshot['random128']['rows']
    lookup={(r['objective'],r['seed'],r['budget_pp'],r['policy']):r for r in summary}
    rows=[]
    for objective in ('latency','energy'):
        for seed in (42,45,48):
            for budget in (.5,1.,1.5,2.):
                a=lookup[objective,seed,budget,'learned'];b=lookup[objective,seed,budget,'matched']
                def acc(r,sd=False):
                    value=f"{r['accuracy']:.2f}"
                    if sd:value+=r'\,$\pm$\,'+f"{r['accuracy_sd']:.2f}"
                    if not r['mean_accuracy_pass']:value+=r'$^*$'
                    return value
                rows.append([('L' if objective=='latency' else 'E')+f'/{seed}',f'{budget:g}',
                             acc(a),acc(b,True),f"{a['accuracy']-b['accuracy']:+.2f}",
                             f"{a['median_latency_ms_per_image']:.3f}",f"{b['median_latency_ms_per_image']:.3f}",
                             f"{a['median_energy_mj_per_image']:.2f}",f"{b['median_energy_mj_per_image']:.2f}"])
    table('RandomTable',r'Obj./seed & $\delta$ & Acc. Lrn & Acc. Rnd $\pm$s & $\Delta$pp & ms Lrn & ms Rnd & mJ Lrn & mJ Rnd',rows,'@{}lrrrrrrrr@{}')
    rows=[]
    for seed in (42,45,48):
        r=lookup['latency',seed,2.,'uniform']
        rows.append([str(seed),f"{r['accuracy']:.2f} $\\pm$ {r['accuracy_sd']:.2f}",
                     f"{r['median_latency_ms_per_image']:.3f}",f"{r['median_energy_mj_per_image']:.2f}"])
    table('UniformTable',r'Seed & Uniform accuracy (\%) & ms/image & mJ/image',rows,'@{}rrrr@{}')
    costs={(r['name'],r['batch_size']):r for r in snapshot['most_pruned_other']['rows']}
    costs.update({(r['name'],128):r for r in snapshot['most_pruned128']['rows']})
    display={r['architecture']:r['display_name'] for r in snapshot['fp32_classes']['models']}
    rows=[]
    for r in snapshot['most_pruned128']['rows']:
        arch,prune,precision=r['name'].split('__')
        values=[esc(display[arch]),precision.upper(),f"{r['accuracy']:.2f}"]
        for size in (8,32,64,128):
            c=costs[r['name'],size]
            assert abs(c['accuracy']-r['accuracy'])<1e-8
            values.append(f"{c['latency_ms_per_image']:.3f} / {c['energy_mj_per_image']:.2f}")
        rows.append(values)
    table('PrunedTable',r'Architecture & Precision & Acc.\% & b8: ms/mJ & b32: ms/mJ & b64: ms/mJ & b128: ms/mJ',rows,'@{}llrrrrr@{}')
    old=data['pc_routes_correctness.json']['results']
    snapshot['threshold_examples']=[r for r in old if r['seed']==48 and r['objective']=='latency' and r['batch_size']==128]
    stat={r['name']:r for r in data['gpu_static.json']['results']}
    snapshot['forest_examples']=[dict(name=n,forest_accuracy=next(c['accuracy_percent'] for c in snapshot['fp32_classes']['models'][i]['classes'] if c['class_name']=='forest'),
         overall=snapshot['fp32_classes']['models'][i]['overall_accuracy_percent'],
         batch32_fp32_median_ms=statistics.median(t['normalized_host_input_to_logits_batch_ms'] for t in stat[n+'__p0__fp32']['timings']['32']))
         for i,n in ((3,'resnet101'),(0,'resnet18'),(5,'mobilenetv3_large'))]
    # Snapshot report JSON atomically so its recorded hash always matches its bytes.
    snapshot['sources']['random_b64/random_routes_b64.json']=hashlib.sha256(json.dumps(ongoing,sort_keys=True).encode()).hexdigest()
    snapshot['source_note']='random64 hash is canonical parsed JSON for the captured in-memory snapshot; other sources use file bytes.'
    snapshot['random64_captured_report']=ongoing
    (OUT/'numerical_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n',encoding='utf-8')
    tables.extend([r'\newcommand{\RandomProgress}{'+str(snapshot['random64']['completed'])+'}',
                   r'\newcommand{\CutoffUTC}{'+snapshot['captured_utc'][:16].replace('T',' ')+r' UTC}'])
    (OUT/'tables.tex').write_text('\n'.join(tables)+'\n',encoding='utf-8')
    # Preserve the larger 71-row table separately, outside the five-page PDF.
    for name in ('all_213_class_table.md','all_213_class_table.tex'):
        (OUT/name).write_bytes((REPORTS/'class_statistics'/name).read_bytes())
    print('Saved frozen numerical snapshot, 8-class summary rows,24paired random rows,21pruned rows and71-row supplement.')
    print('Random64 snapshot:',snapshot['random64'])


if __name__=='__main__':
    main()
