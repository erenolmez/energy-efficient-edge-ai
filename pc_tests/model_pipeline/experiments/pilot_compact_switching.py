"""Fixed seed42 validation-only cache/switching pilot before any long sweeps."""
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
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
from pc_tests.model_pipeline.routers.correctness.correctness_policy import calibrate_correctness,select_correctness
from pc_tests.model_pipeline.runtime.compact_switching import calibrate_cache_aware,select_cache_aware,ProfiledLRU,VerifiedHostStore
from pc_tests.model_pipeline.runtime.experiment_guard import check,wait

REPORT=PIPELINE/'reports/compact_switching_pilot_20261028/study.json'

def main():
    if REPORT.exists():raise RuntimeError('Existing switching pilot retained; no blind restart')
    wait(__file__)
    import torch
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner
    from pc_tests.model_pipeline.runtime.compact_trt import create_runner
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    torch.set_num_threads(1)
    audited=PIPELINE/'reports/compact_resident_screen_20261028/audited_summary.json'
    assert json.loads(audited.read_text())['audit_passed']
    source=PIPELINE/'reports/compact_runtime_profile_20261028/study.json'
    profile=json.loads(source.read_text());pool=profile['pool']
    summary=json.loads((source.parent/'audited_summary.json').read_text())
    bycost={r['name']:r for r in summary['models']}
    cache_dir=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,_,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache_dir)
    names=[r['name'] for r in records];columns=[names.index(n) for n in pool];base=pool.index(baseline)
    byname={r['name']:r for r in records};records=[dict(byname[n],profiled_resident_bytes=bycost[n]['file_resident_bytes']) for n in pool]
    costs=np.array([bycost[n]['eager_ms']/1000 for n in pool])
    # Includes observed full cleanup in the estimated file-miss cost.
    miss=np.array([np.median([r['setup_seconds']+r['cleanup_seconds'] for r in profile['results'] if r['kind']=='setup' and r['name']==n and r['mode']=='file']) for n in pool])
    _,cal=split_validation(labels,42)
    images=np.load(cache_dir/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    correct=np.column_stack([np.load(profile['fresh_original_validation'][n]['predictions_path'],allow_pickle=False)[cal]==labels[cal] for n in pool])
    engine=PIPELINE/'artifacts/correctness_b1_20261028/engines/correctness_seed42_b1.engine'
    router=RouterRunner(engine);buffers=SharedBatchInputs(maximum_batch=1)
    report=dict(status='preparing',results=[],seed=42,pool=pool,test_data_opened=False,source_sha256=digest(Path(__file__)),
        predecessor_sha256=digest(audited),profile_sha256=digest(source),provenance=provenance,expected_pilots=7,
        router_engine_sha256=digest(engine),dependencies={str(p):digest(p) for p in [PIPELINE/'runtime/compact_switching.py',PIPELINE/'runtime/compact_trt.py']},
        protocol='Predeclared seed42 engineering screen on256calibration images only, policies calibrated on original2000cal positions. FreshCNN/selection/classification, finalhostclass, coldcapacity2 orfullyresident6. No test tuning. Pilotaccuracy descriptive oncalibration, not heldout accuracy. Never extrapolate to unmeasured fulltest savings.',
        created_utc=datetime.now(timezone.utc).isoformat())
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='compact_switching_pilot',status='complete',**values));atomic_json(REPORT,report);check(__file__)
    def cleanup():gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    atomic_json(REPORT,report)
    meter=EnergyCounter()
    if not meter.available:raise RuntimeError('NVML energy unavailable')
    try:
      with torch.inference_mode():
        q=[]
        for i,pos in enumerate(cal):
            if i%100==0:check(__file__)
            h=torch.from_numpy(np.array(images[pos:pos+1])).pin_memory()
            q.append(router.run(buffers.prepare(h)).cpu().numpy().copy()[0,columns])
        q=np.array(q)
        static=calibrate_correctness(q,correct.astype(float),np.ones(2000),costs,base,1.)
        aware=calibrate_cache_aware(q,correct.astype(float),np.ones(2000),costs,miss,2,base,1.)
        report.update(static_policy=static,cache_aware_policy=aware,costs_seconds=costs.tolist(),miss_costs_seconds=miss.tolist(),calibration_positions=cal.tolist())
        done('policy_freeze',kind='calibration',images=2000)
        specs=[('file_static_c2',2,False,False,False,False),('file_aware_c2',2,False,True,False,False),
            ('host_static_c2',2,True,False,False,False),('host_aware_c2',2,True,True,False,False),
            ('host_static_reuse_c2',2,True,False,False,True),('resident_static_c6',6,True,False,False,False),
            ('resident_graph_c6',6,True,False,True,False)]
        # Store verification/setup excluded from runtime and reported separately.
        start=time.perf_counter();store=VerifiedHostStore(records)
        report['host_store']=dict(seconds=time.perf_counter()-start,host_bytes=store.host_bytes)
        host=[torch.from_numpy(np.array(images[p:p+1])).pin_memory() for p in cal[:256]]
        report['status']='measuring';atomic_json(REPORT,report)
        for name,capacity,usehost,useaware,graph,reuse in specs:
            check(__file__);cleanup()
            factory=lambda r:create_runner(r,verified_bytes=store.get(r['name']) if usehost else None,cuda_graph=graph)
            release=(lambda:torch.cuda.synchronize()) if reuse else cleanup
            cache=ProfiledLRU(records,capacity,sum(r['profiled_resident_bytes'] for r in records),factory,torch.cuda.synchronize,release)
            setup=time.perf_counter()
            if capacity==6:
                for i in range(6):cache[i]
            setup_seconds=time.perf_counter()-setup
            startloads=cache.loads;policy=aware if useaware else static
            chosen=np.zeros(6,dtype=np.int64);right=0;decision_seconds=0.
            meter.start();start=time.perf_counter()
            for i,h in enumerate(host):
                normalized=buffers.prepare(h);decision=time.perf_counter()
                prob=router.run(normalized).cpu().numpy().copy()[0,columns]
                if policy['fallback']:index=base
                elif useaware:index=select_cache_aware(prob,costs,miss,list(cache.resident),policy['penalty'])
                else:index=int(select_correctness(prob[None,:],costs,policy['penalty'])[0])
                decision_seconds+=time.perf_counter()-decision
                result=cache[index].run(normalized,copy_input=False).argmax(1).item()
                right+=result==labels[cal[i]];chosen[index]+=1
                if (i+1)%64==0:
                    check(__file__)
                    free,_=torch.cuda.mem_get_info()
                    if free<1024**3:raise RuntimeError('1GiBreserve violated during switch pilot')
            torch.cuda.synchronize();elapsed=time.perf_counter()-start;energy=meter.finish(256)
            if energy.get('gpu_energy_j') is None:raise RuntimeError('Pilotenergy unavailable')
            done(name,kind='pilot',capacity=capacity,hostbytes=usehost,cache_aware=useaware,cuda_graph=graph,no_forced_cleanup=reuse,
                images=256,seconds=elapsed,ms_per_image=1000*elapsed/256,gpu_energy=energy,setup_seconds=setup_seconds,
                runtime_loads=cache.loads-startloads,evictions=cache.evictions,selected_images=chosen.tolist(),
                calibration_probe_accuracy=100*right/256,decision_ms=1000*decision_seconds/256,
                linear_10000image_minutes=elapsed/256*10000/60,note='Projection only, not fulltest result')
            cache.clear();del cache;cleanup();print('PILOT '+name+' '+str(elapsed),flush=True)
        report['status']='complete';atomic_json(REPORT,report)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(REPORT,report);raise

if __name__=='__main__':main()
