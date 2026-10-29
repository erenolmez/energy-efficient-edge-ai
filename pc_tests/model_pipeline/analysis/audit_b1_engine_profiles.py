"""CPU-only audit and descriptive old/fixed-batch1 profile comparison."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
import math
from pathlib import Path
import statistics
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json


def main():
    output=PIPELINE/'reports/b1_engine_profiles_20261028'
    path=output/'study.json'
    report=json.loads(path.read_text())
    assert report['status']=='complete' and report['test_data_opened'] is False
    assert len(report['pool'])==6 and len(set(report['pool']))==6
    assert len(report['results'])==14 and len({r['id'] for r in report['results']})==14
    assert digest(PIPELINE/'experiments/profile_b1_engines.py')==report['source_sha256']
    assert digest(PIPELINE/'reports/gpu_static.json')==report['static_report_sha256']
    assert digest(PIPELINE/'artifacts/quantization/deployment_manifest.json')==report['manifest_sha256']
    assert set(report['calibration_probe_positions']).isdisjoint(report['fit_positions'])
    rows=[]
    for name in report['pool']:
        built=next(r for r in report['results'] if r['kind']=='build' and r['name']==name)
        pair=next(r for r in report['results'] if r['kind']=='paired_profile' and r['name']==name)
        assert digest(Path(built['engine_path']))==built['engine_sha256']==pair['engine_sha256']
        row=dict(name=name,build_seconds=built['build_seconds'],
                 logits_max_abs_difference=pair['logits_max_abs_difference'],
                 argmax_disagreements_on_100=pair['argmax_disagreements_on_100'])
        for variant in ('original','batch1'):
            values=pair['measurements'][variant]
            assert len(values)==5 and sorted(v['repeat'] for v in values)==list(range(5))
            for value in values:
                assert value['energy_window_seconds']>=2. and value['images']>0
                assert all(math.isfinite(value[k]) and value[k]>0 for k in ('ms_per_image','gpu_energy_j_per_image'))
            row[variant]=dict(ms_per_image=statistics.median(v['ms_per_image'] for v in values),
                gpu_j_per_image=statistics.median(v['gpu_energy_j_per_image'] for v in values),
                observed_resident_mib=statistics.median(v['observed_resident_bytes'] for v in pair['load_profiles'][variant])/1024**2,
                deserialize_context_seconds=statistics.median(v['deserialize_context_seconds'] for v in pair['load_profiles'][variant]))
        row['latency_saving_percent']=100*(row['original']['ms_per_image']-row['batch1']['ms_per_image'])/row['original']['ms_per_image']
        row['gpu_energy_saving_percent']=100*(row['original']['gpu_j_per_image']-row['batch1']['gpu_j_per_image'])/row['original']['gpu_j_per_image']
        rows.append(row)
    pools=[r for r in report['results'] if r['kind']=='pool_memory']
    assert len(pools)==2 and all(r['count']==6 and r['free_bytes']>=1024**3 for r in pools)
    summary=dict(audit_passed=True,report_sha256=digest(path),rows=rows,pool_memory=pools,
        audit_scope='Structural/provenance/reserve audit, not numerical-equivalence acceptance',
        candidates_with_argmax_changes=[r['name'] for r in rows if r['argmax_disagreements_on_100']>0],
        heldout_accuracy_evaluated=False,router_speedup_evaluated=False,
        caveat='Five paired >=2s validation-input windows; new profiles may change logits/argmax. No full-test routing gain, no whole-PC energy or unobserved peak-memory claim.')
    atomic_json(output/'audited_summary.json',summary)
    lines=['# Paired dedicated batch-one engine profiles','',summary['caveat'],'',
        '| Model | Original ms/image | Batch1 ms/image | Latency saving | GPU-energy saving | Argmax changes /100 |',
        '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['name']} | {r['original']['ms_per_image']:.3f} | {r['batch1']['ms_per_image']:.3f} | {r['latency_saving_percent']:+.1f}% | {r['gpu_energy_saving_percent']:+.1f}% | {r['argmax_disagreements_on_100']} |" for r in rows]
    lines += ['','Same6models held resident:']
    lines += [f"- {r['variant']}: observed incremental {r['observed_increment_bytes']/1024**2:.0f} MiB; {r['free_bytes']/1024**2:.0f} MiB free." for r in pools]
    lines += ['','Selection used seed42fit positions only. Preserve all negative savings and discrepancies. Dedicated-engine validation correctness must be refreshed before policy calibration.']
    lines += ['', 'IMPORTANT: ResNet34 p0 INT8 changed40/100probeclasses. Do not treat that engine as numerically interchangeable or count its speed change as equivalent-model optimization; diagnose before downstream use. Structural audit passed does not mean numerical acceptance.']
    (output/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
