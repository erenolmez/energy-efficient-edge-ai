"""Audit final validation arrays and same-original-engine adapter parity."""
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
    folder=PIPELINE/'reports/compact_validation_20261028'
    path=folder/'study.json'
    state=json.loads(path.read_text())
    assert state['status']=='complete' and not state['test_data_opened']
    rows=state['results'];assert len(rows)==84 and len({r['id'] for r in rows})==84
    assert state['source_sha256']==digest(PIPELINE/'experiments/refresh_compact_validation.py')
    assert state['adapter_sha256']==digest(PIPELINE/'runtime/compact_trt.py')
    assert state['profile_report_sha256']==digest(PIPELINE/'reports/b1_engine_profiles_20261028/study.json')
    labels=np.load(PIPELINE/'artifacts/pc_precision/data_cache/validation_labels.npy',allow_pickle=False)
    assert digest(PIPELINE/'artifacts/pc_precision/data_cache/validation_labels.npy')==state['provenance']['validation_labels_sha256']
    results=[]
    for name in state['pool']:
        adapters=[r for r in rows if r.get('kind')=='adapter_parity' and r['name']==name]
        assert {r['mode'] for r in adapters}=={'file','hostbytes','shared_buffers','cuda_graph'}
        for row in adapters:
            if row['supported']: assert row['argmax_changes']==0 and row['max_abs_logit_difference']<=1e-4 and row['observed_free_bytes']>=1024**3
        chunks=[r for r in rows if r.get('kind')=='validation_chunk' and r['name']==name]
        assert sorted(r['images'] for r in chunks)==list(range(500,5001,500))
        final=state['dedicated_validation'][name]
        prediction=np.load(final['predictions_path'],allow_pickle=False)
        assert prediction.shape==(5000,) and np.issubdtype(prediction.dtype,np.integer) and ((prediction>=0)&(prediction<100)).all()
        assert digest(final['predictions_path'])==final['predictions_sha256']==next(r['predictions_sha256'] for r in chunks if r['images']==5000)
        assert np.isclose(100*(prediction==labels).mean(),final['accuracy'],atol=1e-10)
        cached=np.load(PIPELINE/f'artifacts/gpu_static/{name}_validation_predictions.npy',allow_pickle=False)
        results.append(dict(name=name,dedicated_validation_accuracy=final['accuracy'],
            original_cached_validation_accuracy=100*float((cached==labels).mean()),
            changes_vs_original_cached=int((cached!=prediction).sum()),quarantined=final['quarantined'],
            supported_adapter_modes=[r['mode'] for r in adapters if r['supported']]))
    summary=dict(audit_passed=True,report_sha256=digest(path),rows=84,validation_images_per_engine=5000,models=results,
                 note='Intermediate partial files were renamed/replaced; only final arrays are durable/hash-audited. Cached-vs-new differences are descriptive, not isolated profile effects. No heldout/timing gain claim.')
    atomic_json(folder/'audited_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
