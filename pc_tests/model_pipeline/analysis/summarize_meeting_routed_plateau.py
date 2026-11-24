"""Verify a completed routed replay and emit a LaTeX table fragment."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('run',type=Path)
    cli.add_argument('out',type=Path)
    args=cli.parse_args()
    state=json.loads((args.run/'study.json').read_text())
    plan=json.loads((args.run/'plan.json').read_text())
    if state['status']!='complete_pending_audit' or len(state['rows'])!=plan['expected_windows']:
        raise ValueError('The full set of windows is not complete')
    traces=defaultdict(list)
    for line in (args.run/'telemetry.jsonl').read_text().splitlines():
        row=json.loads(line)
        traces[row['window']].append(row)
    rows=defaultdict(list)
    for r in state['rows']:
        b,mode,repeat=r['batch'],r['mode'],r['repeat']
        if (r['images']!=10000 or r['batches']!=math.ceil(10000/b) or r['warmup_s']<10
            or sum(v['batches'] for v in r['selections'].values())!=r['batches']
            or sum(v['images'] for v in r['selections'].values())!=r['images']
            or not traces[f'r{repeat}_b{b}_{mode}']
            or any(v['temperature_c'] is None or v['temperature_c']>=85 for v in traces[f'r{repeat}_b{b}_{mode}'])):
            raise ValueError(f'Invalid measurements: {b}, {mode}, {repeat}')
        rows[b,mode].append(r)
    out=[]
    for b in plan['batches']:
        fields=[str(b)]
        for mode in ('route','eligible_static'):
            group=rows[b,mode]
            if {r['repeat'] for r in group}!={0,1,2} or len(group)!=3 or len({r['predictions_sha256'] for r in group})!=1:
                raise ValueError(f'Incomplete or non-reproducible group: {b}, {mode}')
            vals=[f"{median(r[k] for r in group):.{p}f}{suffix}"
                  for k,p,suffix in [('test_accuracy',2,r'\%'),('ms_per_batch',2,''),
                                     ('ms_per_image',3,''),('gpu_mj_per_image',2,'')]]
            fields.extend(vals)
        out.append(' & '.join(fields)+r'\\')
    with args.out.open('x',encoding='utf-8') as f:
        f.write('\\def\\RoutedPlateauRows{%\n'+'%\n'.join(out)+'%\n}\n')
    print('\n'.join(out))


if __name__=='__main__':
    main()
