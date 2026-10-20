"""End-to-end routing summaries; retain negative savings and accuracy failures."""
import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def validate_measurements(row):
    values = row['measurements']
    if len(values) != 5 or {value['repeat'] for value in values} != set(range(5)):
        raise ValueError('Every route requires five distinct repeats')
    expected_batches = int(np.ceil(10000 / row['batch_size']))
    for value in values:
        if value['test_images'] != 10000:
            raise ValueError('Route does not cover all held-out images')
        if sum(value['selected_images']) != 10000 or sum(value['selected_batches']) != expected_batches:
            raise ValueError('Selection counts do not match held-out batch/image coverage')
        if value['rgb_input_to_prediction_seconds'] <= 0 or not np.isfinite(value['test_accuracy']):
            raise ValueError('Invalid end-to-end measurements')


def aggregate(report):
    grouped = defaultdict(list)
    for row in report['results']:
        if 'router' not in row or row['status'] != 'complete':
            continue
        validate_measurements(row)
        key = (row['architecture'], row['router'], row['batch_size'], row['objective'], row['budget_pp'])
        grouped[key].append(row)
    rows = []
    for key, cases in sorted(grouped.items()):
        seeds = [case['seed'] for case in cases]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Duplicate seed within a routing configuration')
        row = dict(zip(('architecture','router','batch_size','objective','budget_pp'),key))
        row.update(seeds=sorted(seeds), seeds_completed=len(seeds),
                   accuracy_matched_static_models=sorted({case['best_eligible_static'] for case in cases if case.get('best_eligible_static')}),
                   test_budget_passed_seeds=sum(case['test_budget_passed'] for case in cases),
                   accuracy_loss_vs_p0_pp=float(np.mean([case['accuracy_loss_vs_p0_pp'] for case in cases])),
                   test_accuracy=float(np.mean([value['test_accuracy'] for case in cases for value in case['measurements']])),
                   mean_gpu_decision_ms=float(np.mean([value['mean_gpu_decision_ms'] for case in cases for value in case['measurements']])))
        for objective in ('latency','energy'):
            for comparator in ('p0_fp32','best_eligible_static'):
                name = f'{objective}_saving_vs_{comparator}_percent'
                values = [case.get(name) for case in cases]
                row[name] = float(np.median(values)) if all(value is not None for value in values) else None
                row[name+'_seed_min'] = float(min(values)) if row[name] is not None else None
                row[name+'_seed_max'] = float(max(values)) if row[name] is not None else None
        rows.append(row)
    return rows


def plot(rows, objective, comparator, path, size=32):
    subset = [row for row in rows if row['objective'] == objective and row['batch_size'] == size]
    architectures = sorted({row['architecture'] for row in subset})
    if not architectures:
        return
    methods = sorted({row['router'] for row in subset})
    colors = ['#0067a5','#c75100','#2e7d32','#8e44ad','#c62828','#007f7f',
              '#8c6d31','#ad1457','#455a64','#5246a8','#558b2f','#6d4c41']
    key = f'{objective}_saving_vs_{comparator}_percent'
    valid = [row[key] for row in subset if row[key] is not None]
    if not valid:
        return
    lower, upper = min(-5,min(valid)-5), max(5,max(valid)+5)
    width, height = 1440, 180 + int(np.ceil(len(architectures)/4))*300
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial,sans-serif" fill="#222">',
           f'<text x="20" y="27" font-size="18">End-to-end {objective} saving versus {comparator}, batch {size}</text>',
           '<text x="20" y="49" font-size="12">All methods shown; no test-based policy selection. Hollow markers: at least one measured seed violates the test accuracy budget.</text>']
    for index, method in enumerate(methods):
        x, y = 20 + (index%4)*355, 76+(index//4)*22
        color = colors[index%len(colors)]
        out += [f'<line x1="{x}" x2="{x+22}" y1="{y-4}" y2="{y-4}" stroke="{color}" stroke-width="2"/>',
                f'<text x="{x+28}" y="{y}" font-size="11">{html.escape(method)}</text>']
    for panel, architecture in enumerate(architectures):
        left, top = 52+(panel%4)*355, 175+(panel//4)*300
        chart_width, chart_height = 280, 210
        def position(budget, saving):
            return left+(budget-.5)/1.5*chart_width, top+chart_height-(saving-lower)/(upper-lower)*chart_height
        out += [f'<text x="{left}" y="{top-16}" font-size="13">{html.escape(architecture)}</text>',
                f'<path d="M{left},{top} V{top+chart_height} H{left+chart_width}" fill="none" stroke="#444"/>']
        for budget in (.5,1,1.5,2):
            x,_=position(budget,0)
            out.append(f'<text x="{x-7}" y="{top+chart_height+17}" font-size="10">{budget:g}</text>')
        for saving in np.linspace(lower,upper,5):
            _,y=position(.5,saving)
            out += [f'<line x1="{left}" x2="{left+chart_width}" y1="{y}" y2="{y}" stroke="#ddd"/>',
                    f'<text x="{left-37}" y="{y+3}" font-size="9">{saving:.0f}%</text>']
        out.append(f'<text x="{left+30}" y="{top+chart_height+36}" font-size="10">Validation policy accuracy allowance (pp)</text>')
        for index, method in enumerate(methods):
            points = sorted([row for row in subset if row['architecture']==architecture and row['router']==method and row[key] is not None],key=lambda row:row['budget_pp'])
            color=colors[index%len(colors)]
            coordinates=[position(row['budget_pp'],row[key]) for row in points]
            if coordinates:
                encoded=' '.join(f'{x:.2f},{y:.2f}' for x,y in coordinates)
                out.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            for row,(x,y) in zip(points,coordinates):
                fill=color if row['test_budget_passed_seeds']==row['seeds_completed'] else 'white'
                out.append(f'<circle cx="{x}" cy="{y}" r="3" stroke="{color}" fill="{fill}"/>')
    out.append('</g></svg>')
    path.write_text('\n'.join(out),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--scope', choices=('global','within-architecture'), default='global')
    args=parser.parse_args()
    prefix = 'pc_routes_global' if args.scope == 'global' else 'pc_routes'
    report=json.loads((args.report_dir/f'{prefix}.json').read_text())
    if args.scope == 'global' and report.get('scope') != 'global':
        raise ValueError('Global summary requires a global report')
    cases=[row for row in report['results'] if 'router' in row]
    completed=[row for row in cases if row['status']=='complete']
    failures=[row for row in cases if row['status']=='failed']
    # 8 architectures * 2 objectives * 3 seeds * (5 direct + 7 aggregated)
    # methods * 5 batch sizes * 4 accuracy budgets.
    expected=1440 if args.scope == 'global' else 11520
    if not args.allow_partial and (len(completed)!=expected or failures):
        raise RuntimeError(f'Full route report requires {expected} complete cases; energy-unavailable cases need explicit scope accounting')
    rows=aggregate(report)
    (args.report_dir/f'{prefix}_comparison.json').write_text(json.dumps({'scope':args.scope,'completed_cases':len(completed),'expected_cases':expected,
        'partial':len(completed)!=expected,'rows':rows},indent=2)+'\n',encoding='utf-8')
    lines=['# PC end-to-end batch routing','',f'Completed configurations: {len(completed)}/{expected}. Execution failures: {len(failures)}.','',
           'Partial results; not overall PC completion.' if len(completed)!=expected else 'The configured route measurements are complete; final audits still apply.','',
           'The measured boundary includes RGB transfer, shared GPU normalization, fresh feature/grid construction, TensorRT router execution, dispatch synchronization, selected-model inference and host predictions. Decode/resize and batch pinning are excluded equally from static and routed paths. NVML energy covers the GPU, not the whole PC.','',
           '## Two static baselines','',
           '1. Accuracy reference: the unpruned FP32 architecture with highest validation accuracy (global PC: ResNet-101 p0 FP32). This measures the trade-off against the high-accuracy reference, not evidence of a dynamic advantage by itself.',
           '2. Accuracy-matched static alternative: search all 213 candidates on calibration accuracy, then select the lowest prior measured cost for the objective and batch size within the accuracy allowance. The selected model identity is retained below. Selection uses normalized-input static costs, so this is a validation-selected comparator, not a proven end-to-end optimum. Test labels never select either baseline.','',
           'Global static comparator measurements are fresh before each routed configuration, with five repeats and matching input/cold-cache boundaries. Measurements are sequential, not interleaved. Legacy within-architecture reports use cached comparators. Negative savings and test-budget failures remain visible. Positive savings when only the comparator classifier is selected do not establish a dynamic selection benefit; investigate timing nonstationarity.','',
           '## Batch 32 snapshot','',
           '| Scope | Router | Objective | Budget pp | Seeds | Test budget passes | Accuracy | Matched static model(s) | Saving vs accuracy reference % | Saving vs matched static % |',
           '|---|---|---|---:|---:|---:|---:|---|---:|---:|']
    for row in rows:
        if row['batch_size']!=32:
            continue
        def format_saving(comparator):
            value=row[f"{row['objective']}_saving_vs_{comparator}_percent"]
            return 'unavailable' if value is None else f'{value:.2f}'
        lines.append(f"| {row['architecture']} | {row['router']} | {row['objective']} | {row['budget_pp']:g} | {row['seeds_completed']} | {row['test_budget_passed_seeds']} | {row['test_accuracy']:.2f}% | {', '.join(row['accuracy_matched_static_models'])} | {format_saving('p0_fp32')} | {format_saving('best_eligible_static')} |")
    for objective in ('latency','energy'):
        for comparator in ('p0_fp32','best_eligible_static'):
            path=args.report_dir/f'{prefix}_{objective}_vs_{comparator}.svg'
            plot(rows,objective,comparator,path)
            if path.exists():
                lines+=['',f'[{objective.capitalize()} trade-off curves versus {comparator}]({path.name})']
    lines+=['','Full batch sizes, repeat data and selection counts remain in the machine-readable reports. Curves show seed-median savings of every configured method, not a test-selected envelope.','']
    (args.report_dir/f'{prefix}_results.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Summarized {len(completed)} completed route cases')


if __name__=='__main__':
    main()
