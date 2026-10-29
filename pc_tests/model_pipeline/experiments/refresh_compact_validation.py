"""Refresh real selected-pool validation outcomes; compare same-engine adapters.

No test images opened. Dedicated INT8 remains quarantined, even after this run.
"""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import gc
import json
import time
from pathlib import Path
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation,digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json
from pc_tests.model_pipeline.runtime.experiment_guard import check,wait

REPORT=PIPELINE/'reports/compact_validation_20261028/study.json'
ARTIFACTS=PIPELINE/'artifacts/compact_validation_20261028'


def main():
    if REPORT.exists(): raise RuntimeError('Saved validation refresh exists; no blind restart')
    wait(__file__)
    import torch
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.compact_trt import create_runner,UnsupportedGraph
    torch.set_num_threads(1)
    profiles=json.loads((PIPELINE/'reports/b1_engine_profiles_20261028/study.json').read_text())
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,_,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
    records={r['name']:r for r in records}
    images=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    probes=profiles['calibration_probe_positions']
    report=dict(status='running',results=[],pool=profiles['pool'],expected_validation_chunks=60,
        expected_adapter_rows=24,test_data_opened=False,provenance=provenance,baseline=baseline,
        source_sha256=digest(Path(__file__)),adapter_sha256=digest(PIPELINE/'runtime/compact_trt.py'),
        profile_report_sha256=digest(PIPELINE/'reports/b1_engine_profiles_20261028/study.json'),
        scope='5000 real validation outcomes for each dedicated engine; original-engine adapter parity on100fixed validation probes; no timing/savings claims.',
        quarantined=['resnet34__p0__int8'],created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='compact_validation',status='complete',**values))
        atomic_json(REPORT,report);check(__file__)
    atomic_json(REPORT,report)
    def release():
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    try:
        for name in profiles['pool']:
            check(__file__)
            record=records[name]
            assert digest(Path(record['engine_path']))==record['engine_sha256']
            buffers=SharedBatchInputs(maximum_batch=1)
            probehost=[torch.from_numpy(np.array(images[i:i+1])).pin_memory() for i in probes]
            reference=EngineRunner(Path(record['engine_path']),maximum_batch=1)
            with torch.inference_mode():
                expected=np.concatenate([reference.run(buffers.prepare(h),copy_input=False).cpu().numpy().copy() for h in probehost])
            del reference;release()
            for mode in ('file','hostbytes','shared_buffers','cuda_graph'):
                check(__file__)
                shared=(torch.empty((1,3,128,128),device='cuda'),torch.empty((1,100),device='cuda')) if mode=='shared_buffers' else None
                value=Path(record['engine_path']).read_bytes() if mode=='hostbytes' else None
                if value is not None:
                    import hashlib
                    assert hashlib.sha256(value).hexdigest()==record['engine_sha256']
                try:
                    runner=create_runner(record,verified_bytes=value,shared_buffers=shared,cuda_graph=mode=='cuda_graph')
                    with torch.inference_mode():
                        actual=np.concatenate([runner.run(buffers.prepare(h),copy_input=False).cpu().numpy().copy() for h in probehost])
                    if not np.isfinite(actual).all(): raise RuntimeError('Nonfinite adapter output')
                    delta=float(np.max(np.abs(expected-actual)))
                    changes=int((expected.argmax(1)!=actual.argmax(1)).sum())
                    if changes or not np.allclose(expected,actual,rtol=1e-4,atol=1e-4):
                        raise RuntimeError(f'Adapter changed same-engine outputs: {name} {mode} {delta} {changes}')
                    free,_=torch.cuda.mem_get_info()
                    if free<1024**3: raise RuntimeError('GPU reserve below1GiB')
                    done(f'adapter_{name}_{mode}',kind='adapter_parity',name=name,mode=mode,supported=True,
                         max_abs_logit_difference=delta,argmax_changes=changes,observed_free_bytes=free)
                    del runner
                except UnsupportedGraph as exc:
                    done(f'adapter_{name}_{mode}',kind='adapter_parity',name=name,mode=mode,supported=False,reason=str(exc))
                del shared,value;release()
            del probehost
            build=next(r for r in profiles['results'] if r['id']=='build_'+name)
            assert digest(Path(build['engine_path']))==build['engine_sha256']
            runner=EngineRunner(Path(build['engine_path']),maximum_batch=1)
            predictions=np.empty(5000,dtype=np.int64)
            start=time.perf_counter()
            with torch.inference_mode():
                for first in range(0,5000,500):
                    check(__file__)
                    for pos in range(first,first+500):
                        host=torch.from_numpy(np.array(images[pos:pos+1])).pin_memory()
                        predictions[pos]=int(runner.run(buffers.prepare(host),copy_input=False).argmax(1).item())
                    output=ARTIFACTS/f'{name}_validation_partial.npy'
                    temporary=output.with_suffix('.tmp')
                    with temporary.open('wb') as handle: np.save(handle,predictions[:first+500],allow_pickle=False)
                    temporary.replace(output)
                    done(f'validation_{name}_{first+500}',kind='validation_chunk',name=name,images=first+500,
                         predictions_path=str(output),predictions_sha256=digest(output),engine_sha256=build['engine_sha256'])
            output=ARTIFACTS/f'{name}_validation_predictions.npy'
            partial=ARTIFACTS/f'{name}_validation_partial.npy';partial.replace(output)
            report.setdefault('dedicated_validation',{})[name]=dict(images=5000,accuracy=100*float((predictions==labels).mean()),
                predictions_path=str(output),predictions_sha256=digest(output),engine_sha256=build['engine_sha256'],
                elapsed_seconds=time.perf_counter()-start,quarantined=name in report['quarantined'])
            atomic_json(REPORT,report)
            print('Validated '+name+' '+str(report['dedicated_validation'][name]['accuracy']),flush=True)
            del runner,buffers;release()
        report['status']='complete';atomic_json(REPORT,report)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(REPORT,report);raise

if __name__=='__main__': main()
