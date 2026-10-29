"""Audit resumed confirmation, all seeds/variants and mode-matched comparators."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    folder=PIPELINE/'reports/compact_switching_confirmation_20261028';path=folder/'study.json';s=json.loads(path.read_text())
    a=PIPELINE/'artifacts/compact_switching_confirmation_20261028';receipt=json.loads((a/'resume_receipt.json').read_text())
    old=json.loads((a/'recovery_checkpoint41/study_failed.json').read_text());revision=s['code_revisions'][-1]
    assert s['status']=='complete' and len(s['full_sweeps'])==81 and len(s['results'])==1623
    assert len({r['id'] for r in s['results']})==1623 and sum(r.get('kind')=='test_chunk' for r in s['results'])==1620
    assert canonical(s['full_sweeps'][:41])==canonical(old['full_sweeps'])==revision['preserved_full_sweeps_sha256']
    assert s['results'][:823]==[r for r in old['results'] if r.get('kind')!='test_chunk' or (r['seed'],r['repeat'],r['variant']) in {(v['seed'],v['repeat'],v['variant']) for v in old['full_sweeps']}]
    assert len(s['interrupted_attempts'][0]['results'])==5
    assert revision==receipt['revision'] and s['recovery_receipt_sha256']==digest(a/'resume_receipt.json')
    assert revision['original_failed_report_sha256']==digest(a/'recovery_checkpoint41/study_failed.json')
    assert revision['old_source_sha256']==s['source_sha256']==digest(a/'recovery_checkpoint41/confirm_compact_switching_original.py')
    assert revision['new_source_sha256']==digest(PIPELINE/'experiments/confirm_compact_switching.py')
    assert revision['writer_sha256']==digest(PIPELINE/'runtime/atomic_checkpoint.py')
    assert s['frozen_plan_sha256']==digest(a/'frozen_plan.json')==digest(a/'recovery_checkpoint41/frozen_plan.json')
    for group in ('protected_hashes','dependencies','test_hashes'):
        for p,h in s[group].items():assert digest(p)==h
    assert s['predecessor_sha256']==digest(PIPELINE/'reports/compact_switching_pilot_20261028/audited_summary.json')
    profile_path=PIPELINE/'reports/compact_runtime_profile_20261028/study.json';assert s['profile_sha256']==digest(profile_path)
    prof=json.loads(profile_path.read_text());plan=json.loads((a/'frozen_plan.json').read_text())
    for p in plan['plans']:
        seed=p['seed'];assert digest(PIPELINE/f'artifacts/correctness_b1_20261028/engines/correctness_seed{seed}_b1.engine')==p['router_engine_sha256']
        prepared=json.loads((PIPELINE/f'artifacts/correctness_b1_20261028/prepared_seed{seed}.json').read_text())
        marker=PIPELINE/f'artifacts/pc_routers_correctness/global_correctness_seed{seed}.json'
        assert digest(marker)==prepared['marker_sha256']
        original=json.loads(marker.read_text());assert digest(original['weights'])==prepared['weights_sha256']
    for name,h in plan['fresh_validation_hashes'].items():assert digest(prof['fresh_original_validation'][name]['predictions_path'])==h
    cache=PIPELINE/'artifacts/pc_precision/data_cache';labels=np.load(cache/'test_labels.npy',allow_pickle=False)
    validation_labels=np.load(cache/'validation_labels.npy',allow_pickle=False)
    static=json.loads((PIPELINE/'reports/gpu_static.json').read_text())
    from pc_tests.model_pipeline.runtime.global_candidate_pool import global_candidates
    records,_=global_candidates(static['results']);byname={r['name']:r for r in records}
    for name in s['pool']:assert digest(byname[name]['engine_path'])==byname[name]['engine_sha256']
    assert len({(r['seed'],r['repeat'],r['variant']) for r in s['full_sweeps']})==81
    for row in s['full_sweeps']:
        group=[r for r in s['results'] if r.get('kind')=='test_chunk' and (r['seed'],r['repeat'],r['variant'])==(row['seed'],row['repeat'],row['variant'])]
        assert len(group)==20 and sorted(r['completed_images'] for r in group)==list(range(500,10001,500))
        assert all(r['images']==500 and r['seconds']>0 and r['gpu_energy']['gpu_energy_j']>0 and r['observed_free_bytes']>=1024**3 for r in group)
        assert np.isclose(sum(r['seconds'] for r in group),row['seconds'],atol=1e-9)
        assert np.isclose(sum(r['gpu_energy']['gpu_energy_j'] for r in group),row['gpu_energy_j'],atol=1e-8)
        assert digest(row['runtime_file'])==row['runtime_sha256']
        with np.load(row['runtime_file'],allow_pickle=False) as data:
            assert data['predictions'].shape==data['choices'].shape==data['order'].shape==(10000,)
            assert np.isin(data['predictions'],np.arange(100)).all() and np.isin(data['choices'],np.arange(6)).all()
            assert np.array_equal(np.sort(data['order']),np.arange(10000))
            assert np.array_equal(np.bincount(data['choices'],minlength=6),row['selected_images'])
            assert np.isclose(100*(data['predictions']==labels).mean(),row['accuracy'],atol=1e-10)
        if row['capacity']==6:assert row['runtime_loads']==row['evictions']==0
    training=json.loads((PIPELINE/'reports/correctness_training_energy_20261028/audited_summary.json').read_text())
    fits={r['seed']:r['joules'] for r in training['rows']};output=[]
    for seed in s['seeds']:
      for variant in s['variants']:
        rows=sorted([r for r in s['full_sweeps'] if r['seed']==seed and r['variant']==variant],key=lambda r:r['repeat'])
        assert len(rows)==3 and [r['repeat'] for r in rows]==[0,1,2]
        mode=rows[0]['comparator_mode']
        reference=sorted([r for r in s['full_sweeps'] if r['seed']==seed and r['variant']=='reference_'+mode],key=lambda r:r['repeat'])
        eligible=sorted([r for r in s['full_sweeps'] if r['seed']==seed and r['variant']=='eligible_'+mode],key=lambda r:r['repeat'])
        loss=[r['accuracy']-v['accuracy'] for r,v in zip(reference,rows)];passed=max(loss)<=1.+1e-10
        routed=not variant.startswith(('reference_','eligible_'))
        row=dict(seed=seed,variant=variant,comparator_mode=mode,accuracy=float(np.mean([r['accuracy'] for r in rows])),
            accuracy_loss_pp=float(np.mean(loss)),accuracy_pass=passed,all_repeats_pass=passed,
            median_ms_per_image=float(np.median([r['ms_per_image'] for r in rows])),latency_range_ms=[min(r['ms_per_image'] for r in rows),max(r['ms_per_image'] for r in rows)],
            median_gpu_j_per_image=float(np.median([r['gpu_j_per_image'] for r in rows])),gpu_energy_range_j_per_image=[min(r['gpu_j_per_image'] for r in rows),max(r['gpu_j_per_image'] for r in rows)],
            median_loads=float(np.median([r['runtime_loads'] for r in rows])),median_evictions=float(np.median([r['evictions'] for r in rows])),
            median_setup_seconds=float(np.median([r['setup_seconds'] for r in rows])),is_routed=routed)
        used=sorted({i for r in rows for i,count in enumerate(r['selected_images']) if count})
        row['used_models']=[s['pool'][i] for i in used]
        row['constant_classifier_selection']=routed and len(used)==1
        for key,controls in [('reference',reference),('eligible',eligible)]:
            ms=float(np.median([r['ms_per_image'] for r in controls]));energy=float(np.median([r['gpu_j_per_image'] for r in controls]))
            saving=energy-row['median_gpu_j_per_image'];crossing=math.ceil(fits[seed]/saving) if saving>0 and routed else None
            row[key+'_comparison']=dict(latency_saving_percent=100*(1-row['median_ms_per_image']/ms),gpu_energy_saving_percent=100*(1-row['median_gpu_j_per_image']/energy),
                matched_repeat_latency_saving_range_percent=[min(100*(1/r2['ms_per_image'])*(r2['ms_per_image']-r1['ms_per_image']) for r1,r2 in zip(rows,controls)),max(100*(1/r2['ms_per_image'])*(r2['ms_per_image']-r1['ms_per_image']) for r1,r2 in zip(rows,controls))],
                mathematical_fit_crossing_images=crossing,valid_gpu_fit_payback_images=crossing if passed else None,
                no_valid_payback_reason=None if crossing is not None and passed else ('accuracy allowance fails' if not passed else 'no positive inference energy saving or staticcontrol'))
        output.append(row)
    summary=dict(audit_passed=True,report_sha256=digest(path),source_lineage_verified=True,original41rows_unchanged=True,archived_partial_chunks=5,
        full_sweeps=81,full_chunks=1620,results=output,
        scope='Three repeats, all3seeds, six original candidates, latency1pp. Mode-matched eager/graph controls; no test-based tuning. Compare only matchingpool/configuration.',
        payback_scope='Seed-specific TinyCNN GPUfit-only numerator, positive matchedGPUinferencesaving denominator, allrepeataccuracygate. Excludes CPU,supervision,calibration,export,startup. Training/inference separate sessions; measurements straddle saved recovery, notwholePC energy.',
        limitations='Three repeats are descriptive ranges, not confidence intervals. Sequential caseorders/sessionthermals may affect differences: eligible-eager timings varied substantially across repeats/sessions. All host-aware cases collapsed to the same staticR101FP16; faster same-classifier measurements are not proof of dynamic-selection benefit. Keep rawcoldloads/setup separate; resident warm inference excludes pool startup.')
    atomic_json(folder/'audited_summary.json',summary)
    fields=['seed','variant','accuracy','accuracy_loss_pp','accuracy_pass','median_ms_per_image','median_gpu_j_per_image','median_loads','median_evictions','comparator_mode']
    with (folder/'comparison.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+['latency_saving_vs_eligible_pct','gpu_energy_saving_vs_eligible_pct','valid_gpu_fit_payback_vs_eligible'])
        writer.writeheader()
        for r in output:writer.writerow(dict({k:r[k] for k in fields},latency_saving_vs_eligible_pct=r['eligible_comparison']['latency_saving_percent'],gpu_energy_saving_vs_eligible_pct=r['eligible_comparison']['gpu_energy_saving_percent'],valid_gpu_fit_payback_vs_eligible=r['eligible_comparison']['valid_gpu_fit_payback_images']))
    lines=['# Batch-one switching confirmation','',summary['scope'],'',
        '| Seed | Variant | Accuracy % | Loss pp | Pass | ms/image | Latency saving % | GPU-energy saving % |','|---:|---|---:|---:|:---:|---:|---:|---:|']
    for r in output:
        if r['is_routed']:lines.append(f"| {r['seed']} | {r['variant']} | {r['accuracy']:.2f} | {r['accuracy_loss_pp']:.2f} | {'yes' if r['accuracy_pass'] else 'NO'} | {r['median_ms_per_image']:.3f} | {r['eligible_comparison']['latency_saving_percent']:+.2f} | {r['eligible_comparison']['gpu_energy_saving_percent']:+.2f} |")
    lines+=['','Savings use fresh same-mode eligible-static comparators; positive means lower cost. Reference-FP32 comparisons andseed-specific fit payback are inJSON.','',summary['payback_scope'],'',summary['limitations'],'','All41pre-repair fullrows and5archivedpartialchunks preserved. Sourcehashlineage/frozenplan andactualpredictions checked.']
    (folder/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    routed=[r for r in output if r['is_routed']]
    print(json.dumps(dict(audit_passed=True,sweeps=81,router_passes=sum(r['accuracy_pass'] for r in routed),router_cases=15,
       pass_and_both_savings=[dict(seed=r['seed'],variant=r['variant'],latency_saving=r['eligible_comparison']['latency_saving_percent'],energy_saving=r['eligible_comparison']['gpu_energy_saving_percent']) for r in routed if r['accuracy_pass'] and r['eligible_comparison']['latency_saving_percent']>0 and r['eligible_comparison']['gpu_energy_saving_percent']>0]),indent=2))

if __name__=='__main__':main()
