"""Validation-only original-engine switch/setup and resident execution profiles."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import gc,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation,digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json
from pc_tests.model_pipeline.runtime.experiment_guard import check,wait

REPORT=PIPELINE/'reports/compact_runtime_profile_20261028/study.json'
ARTIFACTS=PIPELINE/'artifacts/compact_runtime_profile_20261028'


def main():
    if REPORT.exists(): raise RuntimeError('Existing profile preserved; no blind restart')
    wait(__file__)
    import torch
    from pc_tests.model_pipeline.runtime.compact_trt import create_runner
    from pc_tests.model_pipeline.runtime.compact_switching import VerifiedHostStore
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    torch.set_num_threads(1)
    previous=PIPELINE/'reports/compact_validation_20261028/audited_summary.json'
    assert json.loads(previous.read_text())['audit_passed']
    profiles=json.loads((PIPELINE/'reports/b1_engine_profiles_20261028/study.json').read_text())
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,_,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
    by_name={r['name']:r for r in records};pool=[by_name[n] for n in profiles['pool']]
    images=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    host=[torch.from_numpy(np.array(images[i:i+1])).pin_memory() for i in profiles['calibration_probe_positions']]
    buffers=SharedBatchInputs(maximum_batch=1)
    report=dict(status='running',results=[],pool=[r['name'] for r in pool],baseline=baseline,provenance=provenance,
        source_sha256=digest(Path(__file__)),test_data_opened=False,original_engines_only=True,
        expected_rows=157,expected_setup_rows=36,expected_warm_rows=60,expected_validation_chunks=60,
        protocol='All6 frozen original engines. Verified host store once.3 cold loads perfile/host mode, five alternating paired2-second windows eager/graph with fresh RGB preprocessing and final host class. Original5000validation predictions refreshed; no heldout tests or router. GPU-only energy excludes CPU.',
        dependencies={str(p):digest(p) for p in [PIPELINE/'runtime/compact_trt.py',PIPELINE/'runtime/compact_switching.py',PIPELINE/'runtime/gpu_batch_inputs.py',previous]},
        created_utc=datetime.now(timezone.utc).isoformat())
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='compact_runtime_profile',status='complete',**values))
        atomic_json(REPORT,report);check(__file__)
    def release():
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    atomic_json(REPORT,report)
    meter=EnergyCounter()
    if not meter.available: raise RuntimeError('NVML GPU energy unavailable')
    try:
        check(__file__);meter.start();start=time.perf_counter();store=VerifiedHostStore(pool)
        done('host_store',kind='host_store',host_bytes=store.host_bytes,setup_seconds=time.perf_counter()-start,
             gpu_energy=meter.finish(1),note='CPU memory occupied; GPU energy does not include CPU hash/read energy')
        for record in pool:
            name=record['name'];check(__file__)
            if digest(Path(record['engine_path']))!=record['engine_sha256']: raise RuntimeError('Engine hash changed')
            for repeat in range(3):
                for mode in (('file','hostbytes') if repeat%2==0 else ('hostbytes','file')):
                    release();check(__file__);free_before,_=torch.cuda.mem_get_info()
                    meter.start();start=time.perf_counter()
                    runner=create_runner(record,verified_bytes=store.get(name) if mode=='hostbytes' else None)
                    torch.cuda.synchronize();seconds=time.perf_counter()-start;energy=meter.finish(1)
                    free,_=torch.cuda.mem_get_info()
                    if free<1024**3: raise RuntimeError('1GiB reserve violated')
                    done(f'setup_{name}_{mode}_{repeat}',kind='setup',name=name,mode=mode,repeat=repeat,
                        setup_seconds=seconds,gpu_energy=energy,observed_resident_bytes=max(0,free_before-free),engine_sha256=record['engine_sha256'])
                    cleanup=time.perf_counter();del runner;release()
                    report['results'][-1]['cleanup_seconds']=time.perf_counter()-cleanup;atomic_json(REPORT,report)
            for repeat in range(5):
                for mode in (('eager','cuda_graph') if repeat%2==0 else ('cuda_graph','eager')):
                    check(__file__);release()
                    setup=time.perf_counter();runner=create_runner(record,verified_bytes=store.get(name),cuda_graph=mode=='cuda_graph')
                    setup_seconds=time.perf_counter()-setup
                    with torch.inference_mode():
                        for h in host[:3]: runner.run(buffers.prepare(h),copy_input=False).argmax(1).item()
                        torch.cuda.synchronize();meter.start();start=time.perf_counter();count=0
                        while time.perf_counter()-start<2.:
                            runner.run(buffers.prepare(host[count%len(host)]),copy_input=False).argmax(1).item();count+=1
                        torch.cuda.synchronize();elapsed=time.perf_counter()-start;energy=meter.finish(count)
                    done(f'warm_{name}_{mode}_{repeat}',kind='warm',name=name,mode=mode,repeat=repeat,
                         images=count,seconds=elapsed,ms_per_image=1000*elapsed/count,gpu_energy=energy,
                         setup_excluded_seconds=setup_seconds)
                    del runner;release()
            runner=create_runner(record,verified_bytes=store.get(name));predictions=np.empty(5000,dtype=np.int64)
            with torch.inference_mode():
                for first in range(0,5000,500):
                    check(__file__)
                    for pos in range(first,first+500):
                        h=torch.from_numpy(np.array(images[pos:pos+1])).pin_memory()
                        predictions[pos]=runner.run(buffers.prepare(h),copy_input=False).argmax(1).item()
                    output=ARTIFACTS/f'{name}_validation_predictions.npy'
                    temp=output.with_suffix('.tmp')
                    with temp.open('wb') as handle: np.save(handle,predictions[:first+500],allow_pickle=False)
                    temp.replace(output)
                    done(f'validation_{name}_{first+500}',kind='validation_chunk',name=name,images=first+500,predictions_path=str(output),
                         predictions_sha256=digest(output),engine_sha256=record['engine_sha256'])
            cached=np.load(PIPELINE/f'artifacts/gpu_static/{name}_validation_predictions.npy',allow_pickle=False)
            report.setdefault('fresh_original_validation',{})[name]=dict(images=5000,accuracy=100*float((predictions==labels).mean()),
                differences_vs_cached=int((predictions!=cached).sum()),predictions_path=str(output),predictions_sha256=digest(output),
                engine_sha256=record['engine_sha256'])
            atomic_json(REPORT,report);del runner;release();print('PROFILED '+name,flush=True)
        assert len(report['results'])==report['expected_rows']
        report['status']='complete';atomic_json(REPORT,report)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(REPORT,report);raise

if __name__=='__main__': main()
