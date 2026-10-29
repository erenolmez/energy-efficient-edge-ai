"""Matched validation-only INT8 profile/shape diagnostic; never modifies engines."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import json, time, os, gc
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json

OUT = PIPELINE/'reports/b1_int8_diagnostic_20261028/study.json'

def thermal():
    path = PIPELINE/'artifacts/pc_routers/thermal_guard.json'
    for attempt in range(20):
        try:
            s = json.loads(path.read_text(encoding='utf-8-sig')); break
        except (OSError, json.JSONDecodeError):
            if attempt == 19: raise
            time.sleep(.025)
    if s['status'] != 'monitoring' or s.get('boundary_start_count') is not None:
        raise RuntimeError('Thermal latch: preserve checkpoint, never restart')
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(s['utc'].replace('Z','+00:00'))).total_seconds()
    if not 0 <= age <= 130 or s['expected_script'] != Path(__file__).name or os.getpid() not in s['tracked_process_ids']:
        raise RuntimeError('Fresh exact diagnostic guard required')

def main():
    if OUT.exists(): raise RuntimeError('Existing diagnostic retained; no blind restart')
    pause=json.loads((PIPELINE/'reports/experiment_pause.json').read_text())
    assert pause['status']=='resumed_by_user'
    deadline=time.monotonic()+120
    while True:
        try: thermal(); break
        except RuntimeError:
            s=json.loads((PIPELINE/'artifacts/pc_routers/thermal_guard.json').read_text(encoding='utf-8-sig'))
            if s['status']!='monitoring' or time.monotonic()>deadline: raise
            time.sleep(1)
    import torch
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    torch.set_num_threads(1)
    old=json.loads((PIPELINE/'reports/b1_engine_profiles_20261028/study.json').read_text())
    static=json.loads((PIPELINE/'reports/gpu_static.json').read_text())
    # Static reports may wrap model records; locate the exact saved identity.
    def find(value):
        if isinstance(value, dict):
            if value.get('name')=='resnet34__p0__int8' and 'engine_path' in value: return value
            for v in value.values():
                r=find(v)
                if r is not None: return r
        elif isinstance(value,list):
            for v in value:
                r=find(v)
                if r is not None: return r
    record=find(static)
    assert record is not None
    build=next(r for r in old['results'] if r['id']=='build_resnet34__p0__int8')
    positions=old['calibration_probe_positions']
    images=np.load(PIPELINE/'artifacts/pc_precision/data_cache/validation_rgb128.npy',mmap_mode='r')
    report=dict(status='running',results=[],test_data_opened=False,source_sha256=digest(Path(__file__)),
                validation_positions=positions,scope='Matched normalized validation inputs; original engine b1/b32 vs dedicated b1. Not an accuracy-preserving speedup claim.')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT,report)
    logits={}
    try:
        for variant,batch,path,expected in [('original_b1',1,record['engine_path'],record['engine_sha256']),
                ('original_b32',32,record['engine_path'],record['engine_sha256']),
                ('dedicated_b1',1,build['engine_path'],build['engine_sha256'])]:
            thermal(); assert digest(Path(path))==expected
            engine=EngineRunner(Path(path),maximum_batch=batch)
            buffers=SharedBatchInputs(maximum_batch=batch)
            rows=[]
            with torch.inference_mode():
                for pos in positions:
                    host=torch.from_numpy(np.repeat(np.array(images[pos:pos+1]),batch,axis=0)).pin_memory()
                    result=engine.run(buffers.prepare(host),copy_input=False).cpu().numpy().copy()
                    assert np.isfinite(result).all()
                    rows.append(result[0])
            logits[variant]=np.stack(rows)
            free,_=torch.cuda.mem_get_info()
            if free<1024**3: raise RuntimeError('1GiB GPU reserve violated')
            report['results'].append(dict(id=variant,router='int8_diagnostic',status='complete',engine_sha256=expected,
                predictions=logits[variant].argmax(1).tolist(),observed_free_bytes=free))
            atomic_json(OUT,report); thermal()
            del engine,buffers;gc.collect();torch.cuda.empty_cache()
        report['comparisons']={}
        for a,b in [('original_b1','original_b32'),('original_b1','dedicated_b1'),('original_b32','dedicated_b1')]:
            report['comparisons'][a+'_vs_'+b]=dict(argmax_changes=int((logits[a].argmax(1)!=logits[b].argmax(1)).sum()),
                max_abs_logit_difference=float(np.abs(logits[a]-logits[b]).max()))
        report['status']='complete'
        report['note']='Shape dependence is diagnostic evidence only; root cause and full validation correctness remain separate gates.'
        atomic_json(OUT,report)
        print(json.dumps(report['comparisons']),flush=True)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(OUT,report);raise

if __name__=='__main__': main()
