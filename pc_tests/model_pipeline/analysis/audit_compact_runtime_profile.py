"""Audit measured setup/warm windows and refreshed original validation arrays."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
import numpy as np
from pathlib import Path
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json

def main():
    path=PIPELINE/'reports/compact_runtime_profile_20261028/study.json'
    s=json.loads(path.read_text());rows=s['results']
    assert s['status']=='complete' and not s['test_data_opened'] and len(rows)==157 and len({r['id'] for r in rows})==157
    assert s['source_sha256']==digest(PIPELINE/'experiments/profile_compact_runtime.py')
    for p,h in s['dependencies'].items(): assert digest(p)==h
    labels=np.load(PIPELINE/'artifacts/pc_precision/data_cache/validation_labels.npy',allow_pickle=False)
    summary=[];unresolved_setup_energy=[]
    for name in s['pool']:
        values={}
        for kind,modes,repeat_count in [('setup',('file','hostbytes'),3),('warm',('eager','cuda_graph'),5)]:
            for mode in modes:
                group=[r for r in rows if r.get('name')==name and r['kind']==kind and r['mode']==mode]
                assert len(group)==repeat_count and {r['repeat'] for r in group}==set(range(repeat_count))
                for r in group:
                    energy=r['gpu_energy'].get('gpu_energy_j')
                    if kind=='setup' and energy is None:
                        assert r['gpu_energy'].get('energy_unavailable_reason')=='NVML energy counter did not advance'
                        unresolved_setup_energy.append(r['id'])
                    else:
                        assert energy is not None and energy>0 and np.isfinite(energy)
                    if kind=='warm': assert r['seconds']>=2 and r['images']>0 and r['ms_per_image']>0
                if kind=='setup':
                    values[mode+'_setup_seconds']=float(np.median([r['setup_seconds'] for r in group]))
                    values[mode+'_resident_bytes']=int(np.median([r['observed_resident_bytes'] for r in group]))
                else:
                    values[mode+'_ms']=float(np.median([r['ms_per_image'] for r in group]))
                    values[mode+'_j_per_image']=float(np.median([r['gpu_energy']['gpu_energy_j_per_image'] for r in group]))
        fresh=s['fresh_original_validation'][name]
        assert digest(fresh['predictions_path'])==fresh['predictions_sha256']
        pred=np.load(fresh['predictions_path'],allow_pickle=False)
        assert pred.shape==(5000,) and np.isclose(100*(pred==labels).mean(),fresh['accuracy'],atol=1e-10)
        values.update(name=name,validation_accuracy=fresh['accuracy'],differences_vs_cached=fresh['differences_vs_cached'],
                      graph_latency_saving_percent=100*(1-values['cuda_graph_ms']/values['eager_ms']),
                      graph_gpu_energy_saving_percent=100*(1-values['cuda_graph_j_per_image']/values['eager_j_per_image']))
        summary.append(values)
    result=dict(audit_passed=True,report_sha256=digest(path),rows=157,models=summary,
                unresolved_setup_energy_rows=unresolved_setup_energy,cold_energy_comparison_accepted=False,
                note='Classifier-only resident microbenchmarks, fresh GPU normalization to host class; no router or heldout accuracy, not end-to-end routing speedup. GPU-only energy excludes CPU. Some short setup intervals did not advance NVML; retain unavailable values, not zero energy. Warm windows all have finite positive energy. Aggregate cold-energy measurement remains pending.')
    atomic_json(path.parent/'audited_summary.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
