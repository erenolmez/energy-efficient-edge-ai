"""Audit per-image grouping costs and true batch8 comparisons, all seeds."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import csv,json
from pathlib import Path
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

def main():
    path=PIPELINE/'reports/batch8_grouping_20261029/study.json';s=json.loads(path.read_text());a=PIPELINE/'artifacts/batch8_grouping_20261029'
    assert s['status']=='complete' and len(s['full_sweeps'])==36 and len(s['results'])==903 and len({r['id'] for r in s['results']})==903
    assert s['source_sha256']==digest(PIPELINE/'experiments/compare_batch8_grouping.py')
    assert s['predecessor_sha256']==digest(PIPELINE/'reports/router_supervision_screen_20261029/audited_summary.json')
    assert s['frozen_plan_sha256']==digest(a/'frozen_plan.json')
    for group in ('test_hashes','protected_hashes'):
        for p,h in s[group].items():assert digest(p)==h
    plan=json.loads((a/'frozen_plan.json').read_text());labels=np.load(PIPELINE/'artifacts/pc_precision/data_cache/test_labels.npy',allow_pickle=False)
    for p in plan['plans']:
        assert digest(p['new_router_engine'])==p['new_router_sha256'] and p['random_parity_error']<=1e-5 and p['real_mean_parity_error']<=1e-5
        marker=json.loads((PIPELINE/f'artifacts/pc_routers_correctness/global_correctness_seed{p["seed"]}.json').read_text())
        assert digest(marker['weights'])==p['original_weights_sha256']
    assert len({(r['seed'],r['repeat'],r['mode']) for r in s['full_sweeps']})==36
    for row in s['full_sweeps']:
        chunks=[r for r in s['results'] if r['kind']=='test_chunk' and (r['seed'],r['repeat'],r['mode'])==(row['seed'],row['repeat'],row['mode'])]
        assert len(chunks)==25 and sorted(r['completed_images'] for r in chunks)==list(range(400,10001,400))
        assert all(r['images']==400 and r['gpu_energy']['gpu_energy_j']>0 and r['seconds']>0 and r['observed_free_bytes']>=1024**3 for r in chunks)
        assert np.isclose(sum(r['seconds'] for r in chunks),row['seconds']) and np.isclose(sum(r['gpu_energy']['gpu_energy_j'] for r in chunks),row['gpu_energy_j'])
        assert digest(row['runtime_file'])==row['runtime_sha256'] and sum(row['selected_images'])==10000
        with np.load(row['runtime_file'],allow_pickle=False) as data:
            assert data['predictions'].shape==data['choices'].shape==data['order'].shape==(10000,)
            assert np.array_equal(np.sort(data['order']),np.arange(10000))
            assert np.array_equal(np.bincount(data['choices'],minlength=6),row['selected_images'])
            assert np.isclose(100*(data['predictions']==labels).mean(),row['accuracy'],atol=1e-10)
            # One-choice mode must assign one model to each true8-image batch.
            if row['mode']!='per_image':
                choices=data['choices'][data['order']].reshape(1250,8)
                assert np.all(choices==choices[:,:1]) and row['classifier_invocations']==1250
            else:
                assert 1250<=row['classifier_invocations']<=10000
        assert np.isclose(row['ms_per_batch'],8*row['ms_per_image'])
    out=[]
    for seed in (42,45,48):
      for mode in s['modes']:
        rows=sorted([r for r in s['full_sweeps'] if r['seed']==seed and r['mode']==mode],key=lambda r:r['repeat'])
        assert [r['repeat'] for r in rows]==[0,1,2]
        ref=sorted([r for r in s['full_sweeps'] if r['seed']==seed and r['mode']=='reference'],key=lambda r:r['repeat'])
        static=[r for r in s['full_sweeps'] if r['seed']==seed and r['mode']=='eligible_static']
        row=dict(seed=seed,mode=mode,accuracy=float(np.mean([r['accuracy'] for r in rows])),
            accuracy_loss_pp=float(np.mean([r['accuracy']-v['accuracy'] for r,v in zip(ref,rows)])),
            accuracy_pass=all(r['accuracy']-v['accuracy']<=1.+1e-10 for r,v in zip(ref,rows)),
            median_ms_per_image=float(np.median([r['ms_per_image'] for r in rows])),median_ms_per_batch=float(np.median([r['ms_per_batch'] for r in rows])),
            latency_range_ms_per_image=[min(r['ms_per_image'] for r in rows),max(r['ms_per_image'] for r in rows)],
            median_gpu_j_per_image=float(np.median([r['gpu_j_per_image'] for r in rows])),median_classifier_invocations=int(np.median([r['classifier_invocations'] for r in rows])))
        row['latency_saving_vs_eligible_pct']=100*(1-row['median_ms_per_image']/float(np.median([r['ms_per_image'] for r in static])))
        row['gpu_energy_saving_vs_eligible_pct']=100*(1-row['median_gpu_j_per_image']/float(np.median([r['gpu_j_per_image'] for r in static])))
        row['validation_fallback']=next(p for p in plan['plans'] if p['seed']==seed)['policies'].get(mode,{}).get('fallback')
        out.append(row)
    summary=dict(audit_passed=True,report_sha256=digest(path),full_sweeps=36,full_chunks=900,results=out,
        note='Actualfulltest, original3seeds3repeats, same6residentclassifiers, truebatch8. FreshperimageGPUgather/groupclassification/scatter/hostoutput charged. Independentmodepolicies calibratedbeforetest; fallbackflags retained. GPU-only energy excludesCPU. Differentfixedbatch1andgroupedclassifiershapes maydiffernumerically. Three-repeat ranges notconfidence intervals; no generalcross-session speedupclaim.')
    atomic_json(path.parent/'audited_summary.json',summary)
    fields=[k for k in out[0] if k!='latency_range_ms_per_image']
    with (path.parent/'comparison.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in out)
    lines=['# Per-image grouping versus one classifier perbatch8','',summary['note'],'',
        '| Seed | Mode | Accuracy % | Pass1pp | ms/image | ms/batch | Classifier calls /10000 | L saving % | GPU-E saving % |','|---:|---|---:|:---:|---:|---:|---:|---:|---:|']
    for r in out:lines.append(f"| {r['seed']} | {r['mode']} | {r['accuracy']:.2f} | {'yes' if r['accuracy_pass'] else 'NO'} | {r['median_ms_per_image']:.3f} | {r['median_ms_per_batch']:.3f} | {r['median_classifier_invocations']} | {r['latency_saving_vs_eligible_pct']:+.2f} | {r['gpu_energy_saving_vs_eligible_pct']:+.2f} |")
    (path.parent/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
