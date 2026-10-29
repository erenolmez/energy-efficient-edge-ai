"""Audit preliminary resident batch-one results; preserve all seeds/failures."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import csv,json,math
from pathlib import Path
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json


def main():
    folder=PIPELINE/'reports/compact_resident_screen_20261028';path=folder/'study.json'
    s=json.loads(path.read_text());a=PIPELINE/'artifacts/compact_resident_screen_20261028'
    assert s['status']=='complete' and len(s['full_sweeps'])==21 and len(s['results'])==424
    assert len({r['id'] for r in s['results']})==424
    assert s['source_sha256']==digest(PIPELINE/'experiments/run_compact_resident_screen.py')
    assert s['subset_source_sha256']==digest(PIPELINE/'runtime/compact_subset.py')
    assert s['preparation_sha256']==digest(PIPELINE/'artifacts/compact_b1_20261028/preparation.json')
    assert s['profile_sha256']==digest(PIPELINE/'reports/compact_runtime_profile_20261028/audited_summary.json')
    assert s['frozen_plan_sha256']==digest(a/'frozen_plan.json')
    for p,h in s['protected_hashes'].items():assert digest(p)==h
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    for name,key in [('test_rgb128.npy','test_rgb_sha256'),('test_labels.npy','test_labels_sha256')]:assert digest(cache/name)==s[key]
    labels=np.load(cache/'test_labels.npy',allow_pickle=False)
    plan=json.loads((a/'frozen_plan.json').read_text())
    prep=json.loads((PIPELINE/'artifacts/compact_b1_20261028/preparation.json').read_text())
    for p in plan['plans']:
        seed=p['seed']
        assert digest(PIPELINE/f'artifacts/correctness_b1_20261028/engines/correctness_seed{seed}_b1.engine')==p['router_engine_sha256']
        for method,hashes in p['models'].items():
            record=next(r for r in prep['results'] if r['seed']==seed and r['method']==method)
            assert digest(record['model_file'])==hashes['model_sha256']
            assert digest(record['calibration_file'])==hashes['calibration_sha256']
    fits=json.loads((PIPELINE/'reports/correctness_training_energy_20261028/audited_summary.json').read_text())
    fit_j={r['seed']:r['joules'] for r in fits['rows']}
    results=[]
    for seed in (42,45,48):
        group=[r for r in s['full_sweeps'] if r['seed']==seed]
        assert len(group)==7 and {r['method'] for r in group}=={'reference','eligible_static',*s['methods']}
        ref=next(r for r in group if r['method']=='reference');static=next(r for r in group if r['method']=='eligible_static')
        for row in group:
            chunks=[r for r in s['results'] if r.get('kind')=='test_chunk' and r['seed']==seed and r['method']==row['method']]
            assert len(chunks)==20 and sorted(r['completed_images'] for r in chunks)==list(range(500,10001,500))
            assert all(r['images']==500 and r['seconds']>0 and r['gpu_energy']['gpu_energy_j']>0 for r in chunks)
            assert np.isclose(sum(r['seconds'] for r in chunks),row['seconds'],atol=1e-9)
            assert np.isclose(sum(r['gpu_energy']['gpu_energy_j'] for r in chunks),row['gpu_energy_j'],atol=1e-8)
            assert digest(row['runtime_file'])==row['runtime_sha256']
            with np.load(row['runtime_file'],allow_pickle=False) as data:
                pred=data['predictions'];choices=data['choices'];order=data['order']
                assert pred.shape==choices.shape==order.shape==(10000,)
                assert np.array_equal(np.sort(order),np.arange(10000))
                assert np.isin(pred,np.arange(100)).all() and np.isin(choices,np.arange(6)).all()
                assert np.array_equal(np.bincount(choices,minlength=6),row['selected_images'])
                assert np.isclose(100*(pred==labels).mean(),row['accuracy'],atol=1e-10)
            loss=ref['accuracy']-row['accuracy'];passed=loss<=1.+1e-10
            value=dict(seed=seed,method=row['method'],accuracy=row['accuracy'],accuracy_loss_pp=loss,accuracy_pass=passed,
                ms_per_image=row['ms_per_image'],gpu_j_per_image=row['gpu_j_per_image'],decision_ms=row['decision_ms_per_image'],
                latency_saving_vs_reference_pct=100*(1-row['ms_per_image']/ref['ms_per_image']),
                energy_saving_vs_reference_pct=100*(1-row['gpu_j_per_image']/ref['gpu_j_per_image']),
                latency_saving_vs_eligible_pct=100*(1-row['ms_per_image']/static['ms_per_image']),
                energy_saving_vs_eligible_pct=100*(1-row['gpu_j_per_image']/static['gpu_j_per_image']),
                selected_images=row['selected_images'])
            for key,comparator in [('reference',ref),('eligible',static)]:
                saving=comparator['gpu_j_per_image']-row['gpu_j_per_image']
                applicable=row['method']=='tinycnn' and passed and saving>0
                value['fit_gpu_payback_images_vs_'+key]=math.ceil(fit_j[seed]/saving) if applicable else None
                value['payback_reason_vs_'+key]='GPU-only fit numerator, warm inference denominator; excludes setup/calibration/export, separate measurement sessions' if applicable else ('CPU fit energy unmeasured' if row['method'] not in ('tinycnn','reference','eligible_static') else 'Not applicable, accuracy fails or energy saving nonpositive')
            results.append(value)
    summary=dict(audit_passed=True,report_sha256=digest(path),sweeps=21,chunks=420,results=results,
        scope='One-repeat preliminary latency1pp same6originalresidentpool, not213choice/capacity2. No test tuning. No seed selection.',
        payback_scope='TinyCNN only, seed-specific measured GPU fit energy /positive matched warm inference saving; accuracy gate. CPU training energy unavailable. Original fit/new inference sessions, no full-PC/completepipeline payback.',
        caveat='No repeated-sweep uncertainty estimate. Static references sampled in same session but not temporally interleaved perimage. Observed performance variation may affect small savings; preserve negatives. No final optimized routing claim.')
    atomic_json(folder/'audited_summary.json',summary)
    fields=[k for k in results[0] if k!='selected_images']
    with (folder/'comparison.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in results)
    text=['# Preliminary batch-one resident comparison','',summary['scope'],'',
        'Accuracy pass means loss no more than1percentage point versus the same-seed fresh R101FP32 reference. Savings below use the fresh validation-selected static comparator; positive means lower latency/energy. All21sweeps use10000testimages.','',
        '| Seed | Method | Accuracy % | Loss pp | Pass | ms/image | Latency saving % | GPUenergy saving % |',
        '|---:|---|---:|---:|:---:|---:|---:|---:|']
    for r in results:text.append(f"| {r['seed']} | {r['method']} | {r['accuracy']:.2f} | {r['accuracy_loss_pp']:.2f} | {'yes' if r['accuracy_pass'] else 'NO'} | {r['ms_per_image']:.3f} | {r['latency_saving_vs_eligible_pct']:+.2f} | {r['energy_saving_vs_eligible_pct']:+.2f} |")
    text+=['',summary['caveat'],'',summary['payback_scope'],'',
        'The four compact routers are CPU implementations: GPU-only energy does not include their CPU energy. Independent logistic/boosted heads evaluate only6available candidates; sharedMLP/tree/CNN retain full213outputcompute. Startup pool loading is separate from warm inference.']
    (folder/'results.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit_passed=True,sweeps=21,router_passes=sum(r['accuracy_pass'] for r in results if r['method'] in s['methods']),router_cases=15,
        tinycnn_payback=[{k:v for k,v in r.items() if k.startswith('fit_gpu_payback') or k in ('seed','accuracy_pass')} for r in results if r['method']=='tinycnn']),indent=2))

if __name__=='__main__':main()
