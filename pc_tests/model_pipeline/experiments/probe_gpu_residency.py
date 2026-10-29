"""Retain real batch-one classifier engines/contexts to measure safe GPU residency.

Synthetic inputs only; no training/test-label use, no model-selection tuning.
"""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

from collections import defaultdict
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import os
import time
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json

OUTPUT=PIPELINE/'reports/gpu_residency_20261028'
STATE=PIPELINE/'artifacts/pc_routers/thermal_guard.json'
RESERVE=1024*1024**2


def order_records(records):
    def size(record):
        return Path(record['engine_path']).stat().st_size
    groups=defaultdict(list)
    for record in sorted(records,key=lambda r:r['name']):
        groups[record['architecture']].append(record)
    round_robin=[]
    for position in range(max(map(len,groups.values()))):
        for name in sorted(groups):
            if position<len(groups[name]):
                round_robin.append(groups[name][position])
    return {'smallest_first':sorted(records,key=lambda r:(size(r),r['name'])),
            'largest_first':sorted(records,key=lambda r:(-size(r),r['name'])),
            'architecture_round_robin':round_robin}


def admission_estimate(file_bytes,observed_extra):
    if file_bytes<=0 or observed_extra<0:
        raise ValueError('Positive file size and nonnegative observed overhead required')
    return file_bytes+max(64*1024**2,observed_extra)


def thermal():
    for attempt in range(20):
        try:
            state=json.loads(STATE.read_text(encoding='utf-8-sig'))
            break
        except (OSError,json.JSONDecodeError):
            if attempt==19:
                raise RuntimeError('Thermal state persistently unreadable')
            time.sleep(.025)
    if state.get('status')!='monitoring' or state.get('boundary_start_count') is not None:
        raise RuntimeError('Thermal stop latched: preserve results, never restart')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(state['utc'].replace('Z','+00:00'))).total_seconds()
    if not 0<=age<=130 or state.get('expected_script')!=Path(__file__).name or os.getpid() not in state.get('tracked_process_ids',[]):
        raise RuntimeError('Fresh exact residency worker guard required')


def main():
    path=OUTPUT/'study.json'
    if path.exists():
        raise RuntimeError('Existing residency study preserved: no blind restart')
    deadline=time.monotonic()+120
    print(f'Waiting for exact guard PID {os.getpid()}',flush=True)
    while True:
        try:
            thermal();break
        except RuntimeError:
            state=json.loads(STATE.read_text(encoding='utf-8-sig'))
            if state.get('status')!='monitoring' or time.monotonic()>=deadline:
                raise
            time.sleep(1)
    import torch
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner
    torch.set_num_threads(1)
    stopped=PIPELINE/'reports/correctness_b1_20261028/stop_receipt.json'
    receipt=json.loads(stopped.read_text())
    predecessor=PIPELINE/'reports/correctness_b1_20261028/study.json'
    if receipt['status']!='stopped_by_user' or digest(predecessor)!=receipt['study_sha256']:
        raise RuntimeError('Stopped-study receipt/provenance mismatch')
    static_path=PIPELINE/'reports/gpu_static.json'
    static=json.loads(static_path.read_text())
    records=static['results']
    if len(records)!=213 or any(r['status']!='complete' for r in records):
        raise RuntimeError('213 completed static targets required')
    prepared=json.loads((PIPELINE/'reports/correctness_b1_20261028/study.json').read_text())['prepared']['42']
    if digest(Path(prepared['engine']))!=prepared['engine_sha256']:
        raise RuntimeError('Original batch1 router engine hash mismatch')
    # Tiny router remains resident in every pass to approximate the intended deployment.
    router=RouterRunner(prepared['engine'])
    torch.manual_seed(42)
    synthetic=torch.randn((1,3,128,128),device='cuda')
    with torch.inference_mode():
        q=router.run(synthetic).cpu().numpy()
    if q.shape!=(1,213) or not np.isfinite(q).all():
        raise RuntimeError('Resident router output invalid')
    orders=order_records(records)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',results=[],orders={},reserve_bytes=RESERVE,batch_size=1,
        gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,gpu_total_bytes=torch.cuda.mem_get_info()[1],
        source_sha256=digest(Path(__file__)),engine_runner_sha256=digest(PIPELINE/'runtime/gpu_engine_preflight.py'),
        static_report_sha256=digest(static_path),stopped_study_sha256=digest(predecessor),
        router_sha256=prepared['engine_sha256'],test_labels_opened=False,training_performed=False,
        note='Safe demonstrated residency for each frozen order under 1GiB free reserve and conservative admission; NOT an absolute maximum or speedup/energy result. Counts include classifier engine+context+b1buffers, with tiny router also retained. Driver-reported free/used includes other device consumers. No eviction within each pass.',
        created_utc=datetime.now(timezone.utc).isoformat())
    def save():
        atomic_json(path,report)
    save()
    resident=[]
    try:
        for order_name,ordered in orders.items():
            thermal()
            resident=[];expected=[];names=[];extra=64*1024**2
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
            initial_free,total=torch.cuda.mem_get_info()
            summary=dict(initial_free_bytes=initial_free,initial_used_bytes=total-initial_free,
                         resident_models=[],count=0,all_resident_verified=False,stop_reason=None)
            report['orders'][order_name]=summary
            for record in ordered:
                thermal()
                before_free,total=torch.cuda.mem_get_info()
                file_size=Path(record['engine_path']).stat().st_size
                estimate=admission_estimate(file_size,extra)
                if before_free-estimate<RESERVE:
                    summary.update(stop_reason='conservative admission would cross 1GiB reserve',next_candidate=record['name'],next_estimated_bytes=estimate)
                    break
                if digest(Path(record['engine_path']))!=record['engine_sha256']:
                    raise RuntimeError('Classifier engine hash mismatch')
                started=time.perf_counter()
                engine=EngineRunner(record['engine_path'],maximum_batch=1)
                loaded=time.perf_counter()
                with torch.inference_mode():
                    logits=engine.run(synthetic).cpu().numpy().copy()
                torch.cuda.synchronize()
                if logits.shape!=(1,100) or not np.isfinite(logits).all():
                    raise RuntimeError('Classifier cannot execute finite synthetic batch1')
                after_free,total=torch.cuda.mem_get_info()
                if after_free<RESERVE:
                    del engine
                    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
                    summary.update(stop_reason='candidate rolled back to preserve 1GiB reserve',next_candidate=record['name'])
                    break
                resident.append(engine);expected.append(logits);names.append(record['name'])
                del engine
                extra=max(extra,max(0,before_free-after_free-file_size))
                summary.update(resident_models=list(names),count=len(resident),free_bytes=after_free,used_bytes=total-after_free)
                report['results'].append(dict(id=f'{order_name}_{len(resident)}',router='residency_probe',status='complete',
                    order=order_name,count=len(resident),added_model=record['name'],engine_sha256=record['engine_sha256'],
                    engine_file_bytes=file_size,free_bytes=after_free,used_bytes=total-after_free,
                    incremental_used_bytes=before_free-after_free,deserialize_context_seconds=loaded-started,
                    load_and_first_execution_seconds=time.perf_counter()-started,synthetic_logits_sha256=__import__('hashlib').sha256(logits.tobytes()).hexdigest()))
                save()
                print(f'{order_name}: {len(resident)} resident, free={after_free/1024**2:.0f}MiB',flush=True)
            else:
                summary['stop_reason']='all213 candidates loaded within reserve'
            # Keep ALL engines alive while validating them, not just the most recent.
            errors=[]
            for retained,reference in zip(resident,expected):
                thermal()
                with torch.inference_mode():
                    actual=retained.run(synthetic).cpu().numpy().copy()
                if not np.allclose(actual,reference,atol=1e-5,rtol=1e-4):
                    raise RuntimeError('Retained engine parity changed')
                errors.append(float(np.max(np.abs(actual-reference))))
            if resident:
                del retained
            summary['all_resident_verified']=True
            summary['maximum_retained_logits_difference']=max(errors,default=0.)
            summary['free_after_verification_bytes']=torch.cuda.mem_get_info()[0]
            if summary['free_after_verification_bytes']<RESERVE:
                raise RuntimeError('Verification crossed memory reserve')
            save()
            print(f'ORDER COMPLETE {order_name}: {len(resident)} usable resident models',flush=True)
            resident.clear();expected.clear();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        if digest(predecessor)!=receipt['study_sha256'] or digest(static_path)!=report['static_report_sha256']:
            raise RuntimeError('Protected source report changed')
        report['status']='complete';report['completed_utc']=datetime.now(timezone.utc).isoformat();save()
        print('COMPLETE safe GPU-residency probe, not a full routing performance test',flush=True)
    except Exception as error:
        report['status']='failed';report['error']=repr(error);save()
        raise
    finally:
        resident.clear();gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    main()
