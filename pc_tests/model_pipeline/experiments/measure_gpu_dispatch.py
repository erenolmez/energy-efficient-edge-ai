"""Matched validation-only full-vector readback vs GPU selection+ID readback."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation,digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
from pc_tests.model_pipeline.routers.correctness.correctness_policy import calibrate_correctness,select_correctness
from pc_tests.model_pipeline.runtime.experiment_guard import check,wait

REPORT=PIPELINE/'reports/gpu_dispatch_20261028/study.json'

def main():
    if REPORT.exists(): raise RuntimeError('Saved dispatch experiment exists; no blind restart')
    wait(__file__)
    import torch
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner
    from pc_tests.model_pipeline.runtime.compact_trt import create_runner
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    from pc_tests.model_pipeline.routers.correctness.device_selection import DeviceCostSelection
    torch.set_num_threads(1)
    profile_path=PIPELINE/'reports/compact_runtime_profile_20261028/audited_summary.json'
    profile=json.loads(profile_path.read_text());assert profile['audit_passed']
    cost_by_name={r['name']:r for r in profile['models']}
    runtime=json.loads((profile_path.parent/'study.json').read_text())
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,_,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
    names=[r['name'] for r in records];pool=runtime['pool'];columns=np.array([names.index(n) for n in pool]);base=pool.index(baseline)
    by_name={r['name']:r for r in records}
    costs=np.array([cost_by_name[n]['eager_ms'] for n in pool],dtype=np.float64)
    _,cal=split_validation(labels,42)
    images=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    engine_path=PIPELINE/'artifacts/correctness_b1_20261028/engines/correctness_seed42_b1.engine'
    router=RouterRunner(engine_path);buffers=SharedBatchInputs(maximum_batch=1)
    report=dict(status='preparing',results=[],seed=42,pool=pool,logical_candidates=6,router_outputs=213,
        test_data_opened=False,source_sha256=digest(Path(__file__)),router_engine_sha256=digest(engine_path),
        profile_sha256=digest(profile_path),provenance=provenance,expected_rows=13,
        protocol='Validation-only engineering comparison seed42, same6 original classifiers resident. Fresh213-output CNN. CPU copies213 probabilities then selects among6; GPU gathers6/scores inFP64 and copies1int64ID. Same normalizedGPUinput, hostlaunch and finalhostclass. Five alternating paired2second windows, same input cycle. Not full213model routing or heldout accuracy.',
        created_utc=datetime.now(timezone.utc).isoformat())
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='gpu_dispatch',status='complete',**values));atomic_json(REPORT,report);check(__file__)
    atomic_json(REPORT,report)
    meter=EnergyCounter()
    if not meter.available: raise RuntimeError('NVML energy unavailable')
    try:
        q=[]
        with torch.inference_mode():
            for i,pos in enumerate(cal):
                if i%100==0:check(__file__)
                h=torch.from_numpy(np.array(images[pos:pos+1])).pin_memory()
                q.append(router.run(buffers.prepare(h)).cpu().numpy().copy()[0,columns])
        correct=np.column_stack([np.load(runtime['fresh_original_validation'][n]['predictions_path'],allow_pickle=False)[cal]==labels[cal] for n in pool])
        policy=calibrate_correctness(np.array(q),correct.astype(float),np.ones(len(cal)),costs,base,1.)
        report['policy']=policy;report['calibration_positions']=cal.tolist();report['calibration_costs_ms']=costs.tolist()
        done('calibration',kind='calibration',images=2000,policy=policy)
        indices=torch.tensor(columns,device='cuda',dtype=torch.int64)
        selector=DeviceCostSelection(costs,policy['penalty'],base if policy['fallback'] else None).cuda().eval()
        def select(prob,mode):
            if mode=='cpu_vector':
                value=prob.cpu().numpy().copy()[:,columns]
                return base if policy['fallback'] else int(select_correctness(value,costs,policy['penalty'])[0])
            return int(selector(prob.index_select(1,indices)).item())
        host=[torch.from_numpy(np.array(images[p:p+1])).pin_memory() for p in cal[:100]]
        with torch.inference_mode():
            for h in host:
                prob=router.run(buffers.prepare(h));assert select(prob,'cpu_vector')==select(prob,'gpu_id')
        done('selection_parity',kind='selection_parity',images=100,choice_mismatches=0)
        meter.start();start=time.perf_counter();engines=[create_runner(by_name[n]) for n in pool]
        torch.cuda.synchronize();free,_=torch.cuda.mem_get_info()
        if free<1024**3: raise RuntimeError('Residentpool violates1GiBreserve')
        done('resident_setup',kind='setup',seconds=time.perf_counter()-start,gpu_energy=meter.finish(1),observed_free_bytes=free)
        def forward(h,mode):
            normalized=buffers.prepare(h);prob=router.run(normalized)
            index=select(prob,mode)
            return int(engines[index].run(normalized,copy_input=False).argmax(1).item()),index
        with torch.inference_mode():
            # Real final-class parity before timing; no cached choices in timedforward.
            for h in host:
                assert forward(h,'cpu_vector')==forward(h,'gpu_id')
            for repeat in range(5):
                for mode in (('cpu_vector','gpu_id') if repeat%2==0 else ('gpu_id','cpu_vector')):
                    check(__file__)
                    for h in host[:3]:forward(h,mode)
                    torch.cuda.synchronize();meter.start();start=time.perf_counter();count=0;choices=np.zeros(6,dtype=np.int64)
                    while time.perf_counter()-start<2.:
                        _,index=forward(host[count%100],mode);choices[index]+=1;count+=1
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    done(f'window_{mode}_{repeat}',kind='window',mode=mode,repeat=repeat,images=count,seconds=elapsed,
                        ms_per_image=1000*elapsed/count,gpu_energy=meter.finish(count),selected_images=choices.tolist())
        report['status']='complete';report['final_class_parity_images']=100;atomic_json(REPORT,report)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(REPORT,report);raise

if __name__=='__main__': main()
