"""Audit fixed seed42 validation-only supervision screen, never test performance."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
from pathlib import Path
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

def main():
    path=PIPELINE/'reports/router_supervision_screen_20261029/study.json';s=json.loads(path.read_text())
    assert s['status']=='complete' and not s['test_data_opened'] and len(s['fits'])==5 and len(s['results'])==185
    assert len({r['id'] for r in s['results']})==185 and sum(r['kind']=='epoch' for r in s['results'])==180
    assert s['source_sha256']==digest(PIPELINE/'experiments/screen_router_supervision.py')
    assert s['predecessor_sha256']==digest(PIPELINE/'reports/compact_switching_confirmation_20261028/audited_summary.json')
    for p,h in s['protected_hashes'].items():assert digest(p)==h
    labels=np.load(PIPELINE/'artifacts/pc_precision/data_cache/validation_labels.npy',allow_pickle=False)
    assert len(s['fit_positions'])==3000 and len(s['calibration_positions'])==2000 and not set(s['fit_positions'])&set(s['calibration_positions'])
    results=[]
    for fit in s['fits']:
        pos=fit['fit_positions'];assert len(pos)==fit['fit_images'] and len(set(pos))==len(pos) and set(pos)<=set(s['fit_positions'])
        assert np.all(np.bincount(labels[pos],minlength=100)==len(pos)//100)
        epochs=[r for r in s['results'] if r['kind']=='epoch' and r['fit']==fit['name']]
        assert len(epochs)==fit['epochs'] and [r['epoch'] for r in epochs]==list(range(1,fit['epochs']+1))
        assert np.isfinite(fit['losses']).all() and len(fit['losses'])==fit['epochs']
        assert digest(epochs[-1]['checkpoint'])==epochs[-1]['checkpoint_sha256']
        for snapshot in fit['snapshots'].values():assert digest(snapshot['path'])==snapshot['sha256']
        assert digest(fit['weights'])==fit['weights_sha256']
        if fit['internal_tune_metrics']:
            assert not set(pos)&set(s['internal_tune_positions']) and len(pos)==2400
            chosen=min(fit['internal_tune_metrics'],key=lambda e:(fit['internal_tune_metrics'][e]['pool_brier'],int(e)))
            assert int(chosen)==fit['selected_epoch']
        assert fit['gpu_fit_energy']['gpu_energy_j']>0 and fit['gpu_fit_energy']['energy_window_seconds']>0
        assert digest(fit['calibration_file'])==fit['calibration_sha256']
        with np.load(fit['calibration_file'],allow_pickle=False) as d:
            assert np.array_equal(d['positions'],s['calibration_positions'])
            assert d['probabilities'].shape==d['correctness'].shape==(2000,213)
            assert np.isfinite(d['probabilities']).all() and ((d['probabilities']>=0)&(d['probabilities']<=1)).all()
            assert np.isin(d['correctness'],[0,1]).all()
        assert sorted(p['calibration_images'] for p in fit['policies'])==[500,1000,2000]
        for p in fit['policies']:
            cp=p['calibration_positions'];assert len(cp)==p['calibration_images'] and set(cp)<=set(s['calibration_positions'])
            assert np.all(np.bincount(labels[cp],minlength=100)==len(cp)//100)
            assert p['heldout_validation_images']==2000-len(cp)
            if len(cp)==2000:assert p['heldout_validation_accuracy'] is None
        results.append(dict(name=fit['name'],fit_images=len(pos),epochs=fit['epochs'],selected_epoch=fit['selected_epoch'],pool_brier=fit['pool_brier'],
            gpu_fit_joules=fit['gpu_fit_energy']['gpu_energy_j'],calibration_subset_findings=fit['policies']))
    summary=dict(audit_passed=True,report_sha256=digest(path),fits=5,epoch_records=180,results=results,
        note='Seed42validation-onlyscreen, notheldouttest oroptimizeddeploymentclaim. SmallBrierdifferences are descriptive. Internal20/60epochselection uses600fit-heldoutimages only. Latestepochfiles overwrittenbydesign, seven20/60snapshots andfivefinalepochfiles verified. GPUenergy scopeincludesinternalvalidation/checkpoints andexcludesCPU/fullpreparation. No newengineexport/parity/runtime yet.')
    atomic_json(path.parent/'audited_summary.json',summary)
    text=['# Router supervision screen','',summary['note'],'','| Fit | Images | Epochs run | Selected epoch | Pool Brier | GPU fit J |','|---|---:|---:|---:|---:|---:|']
    for r in results:text.append(f"| {r['name']} | {r['fit_images']} | {r['epochs']} | {r['selected_epoch']} | {r['pool_brier']:.6f} | {r['gpu_fit_joules']:.2f} |")
    (path.parent/'results.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('AUDITED5fits/180epochs/15policy-calibration groups; no test-data ordeployment claim.')

if __name__=='__main__':main()
