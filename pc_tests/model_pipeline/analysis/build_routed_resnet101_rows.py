"""Write a report fragment from audited router and fixed-ResNet101 results."""
from pathlib import Path
import sys

from pc_tests.model_pipeline.analysis.audit_optimized_routed_plateau import summarize


def main():
    args = sys.argv[1:]
    interrupted_fixed = '--include-interrupted-fixed' in args
    paths = [arg for arg in args if arg != '--include-interrupted-fixed']
    router_path, retry_path, fixed_path, output, *fixed_retries = paths
    router, retry, fixed = [summarize(path) for path in (router_path,retry_path,fixed_path)]
    if (any(run['status'] in ('preflight','running') for run in (router,retry)) or
        fixed['status']=='preflight' or (fixed['status']=='running' and not interrupted_fixed)):
        raise ValueError('Wait for all three runs to stop before building the report')
    routed = {r['batch']:r for r in router['batches']}
    baseline = {r['batch']:r for r in fixed['batches']}
    for path in fixed_retries:
        supplements = summarize(path)
        if supplements['status'] in ('running','preflight'):
            raise ValueError('Wait for the fixed-classifier retry to finish')
        for row in supplements['batches']:
            if row['qualified_passes']:
                if baseline[row['batch']]['qualified_passes']:
                    raise ValueError('Do not pool fixed-classifier runs with different temperature gates')
                baseline[row['batch']] = row
    for r in retry['batches']:
        if r['qualified_passes']:
            if routed[r['batch']]['qualified_passes']:
                raise ValueError('Do not pool runs with different temperature gates')
            routed[r['batch']] = r
    fields = [('test_accuracy',2,r'\%'),('ms_per_batch',2,''),('ms_per_image',3,''),('gpu_mj_per_image',2,'')]
    lines = []
    for b in (8,16,32,64,128):
        rows = [routed[b],baseline[b]]
        cells = [f'{row[key]:.{n}f}{suffix}' if row[key] is not None else r'\missing'
                 for row in rows for key,n,suffix in fields]
        lines.append(' & '.join([str(b),*cells])+r'\\')
        print(f"Batch {b}: router={routed[b]['qualified_passes']} qualified; ResNet101={baseline[b]['qualified_passes']} qualified")
    Path(output).write_text('\\def\\RoutedResnetRows{%\n'+'%\n'.join(lines)+'%\n}\n',encoding='utf-8')


if __name__ == '__main__':
    main()
