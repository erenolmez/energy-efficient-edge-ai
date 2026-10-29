"""Audit paired dispatch pilot; retain unbalanced fixed-duration input tails."""
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
    path=PIPELINE/'reports/gpu_dispatch_20261028/study.json';s=json.loads(path.read_text())
    assert s['status']=='complete' and len(s['results'])==13 and len({r['id'] for r in s['results']})==13
    assert s['source_sha256']==digest(PIPELINE/'experiments/measure_gpu_dispatch.py') and not s['test_data_opened']
    assert s['router_engine_sha256']==digest(PIPELINE/'artifacts/correctness_b1_20261028/engines/correctness_seed42_b1.engine')
    assert s['profile_sha256']==digest(PIPELINE/'reports/compact_runtime_profile_20261028/audited_summary.json')
    assert s['final_class_parity_images']==100 and next(r for r in s['results'] if r['id']=='selection_parity')['choice_mismatches']==0
    summary={}
    for mode in ('cpu_vector','gpu_id'):
        rows=[r for r in s['results'] if r.get('kind')=='window' and r['mode']==mode]
        assert len(rows)==5 and {r['repeat'] for r in rows}==set(range(5))
        for r in rows:
            assert r['seconds']>=2 and r['images']==sum(r['selected_images']) and r['gpu_energy']['gpu_energy_j']>0
        summary[mode]=dict(median_ms_per_image=float(np.median([r['ms_per_image'] for r in rows])),
            median_gpu_j_per_image=float(np.median([r['gpu_energy']['gpu_energy_j_per_image'] for r in rows])),
            images_per_window=[r['images'] for r in rows])
    result=dict(audit_passed=True,report_sha256=digest(path),summary=summary,
        gpu_vs_cpu_latency_saving_percent=100*(1-summary['gpu_id']['median_ms_per_image']/summary['cpu_vector']['median_ms_per_image']),
        gpu_vs_cpu_gpu_energy_saving_percent=100*(1-summary['gpu_id']['median_gpu_j_per_image']/summary['cpu_vector']['median_gpu_j_per_image']),
        note='Preliminary seed42 FP64/Torch selector implementation, same6residentpool/100inputcycle. Fixed2s windows have different imagecounts/final cycle tails, so not identical fullsweeps or a general causal overhead estimate. Preserve negative savings. Fullclassifier output parity checked on100validationimages, not heldout accuracy. Fused FP32/GPUconditional deployment not tested.')
    atomic_json(path.parent/'audited_summary.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
