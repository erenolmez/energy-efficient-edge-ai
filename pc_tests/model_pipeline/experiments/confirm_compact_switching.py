"""Repeated full-test confirmation of a predefined small switching shortlist.

Three seeds, latency1pp,5routervariants+4matchedstaticcontrols,3repeats=81sweeps.
Original6pool, originalweights; no heldout labels used to change any policy.
"""
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
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
from pc_tests.model_pipeline.routers.correctness.correctness_policy import calibrate_correctness,select_correctness
from pc_tests.model_pipeline.runtime.compact_switching import calibrate_cache_aware,select_cache_aware,ProfiledLRU,VerifiedHostStore
from pc_tests.model_pipeline.runtime.experiment_guard import check,wait

REPORT=PIPELINE/'reports/compact_switching_confirmation_20261028/study.json'
ARTIFACTS=PIPELINE/'artifacts/compact_switching_confirmation_20261028'
VARIANTS=('file_static_c2','host_static_reuse_c2','host_aware_c2','resident_eager_c6','resident_graph_c6',
          'reference_eager','eligible_eager','reference_graph','eligible_graph')

def main():
    resuming='--resume' in __import__('sys').argv
    saved=json.loads(REPORT.read_text()) if REPORT.exists() else None
    if saved is not None and not resuming:raise RuntimeError('Existing confirmation preserved; explicit verified resume required')
    if resuming:
        if saved is None or saved['status']!='resume_authorized':raise RuntimeError('Verified human-authorized recovery receipt required')
        assert saved['code_revisions'][-1]['new_source_sha256']==digest(Path(__file__))
        assert digest(ARTIFACTS/'frozen_plan.json')==saved['frozen_plan_sha256']
    wait(__file__)
    import torch
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner
    from pc_tests.model_pipeline.runtime.compact_trt import create_runner
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    torch.set_num_threads(1)
    audit=PIPELINE/'reports/compact_switching_pilot_20261028/audited_summary.json'
    assert json.loads(audit.read_text())['audit_passed']
    profile_path=PIPELINE/'reports/compact_runtime_profile_20261028/study.json'
    prof=json.loads(profile_path.read_text());summary=json.loads((profile_path.parent/'audited_summary.json').read_text())
    pool=prof['pool'];bycost={r['name']:r for r in summary['models']}
    cache_dir=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,_,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache_dir)
    allnames=[r['name'] for r in records];columns=[allnames.index(n) for n in pool];base=pool.index(baseline)
    byname={r['name']:r for r in records};records=[dict(byname[n],profiled_resident_bytes=bycost[n]['file_resident_bytes']) for n in pool]
    costs=np.array([bycost[n]['eager_ms']/1000 for n in pool]);graphcosts=np.array([bycost[n]['cuda_graph_ms']/1000 for n in pool])
    # Host-aware deployment uses host-load plus measured fullcleanup, not filecost.
    miss=np.array([np.median([r['setup_seconds']+r['cleanup_seconds'] for r in prof['results'] if r['kind']=='setup' and r['name']==n and r['mode']=='hostbytes']) for n in pool])
    images=np.load(cache_dir/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    for n in pool:assert digest(prof['fresh_original_validation'][n]['predictions_path'])==prof['fresh_original_validation'][n]['predictions_sha256']
    correctness=np.column_stack([np.load(prof['fresh_original_validation'][n]['predictions_path'],allow_pickle=False)==labels for n in pool])
    report=dict(status='preparing',results=[],full_sweeps=[],seeds=[42,45,48],variants=list(VARIANTS),repeats=3,
        expected_full_sweeps=81,expected_test_chunks=1620,pool=pool,baseline=baseline,budget_pp=1.,objective='latency',
        source_sha256=digest(Path(__file__)),predecessor_sha256=digest(audit),profile_sha256=digest(profile_path),provenance=provenance,
        dependencies={str(p):digest(p) for p in [PIPELINE/'runtime/compact_switching.py',PIPELINE/'runtime/compact_trt.py',PIPELINE/'runtime/gpu_batch_inputs.py']},
        protected_hashes={str(p):digest(p) for p in [PIPELINE/'reports/pc_routes_correctness.json',PIPELINE/'reports/gpu_static.json',PIPELINE/'reports/correctness_b1_20261028/study.json']},
        test_data_opened=False,created_utc=datetime.now(timezone.utc).isoformat(),
        protocol='Predefined5mechanism/deployment variants plus eager/graph freshR101FP32 andvalidationeligible controls;3seeds3repeats. Original6models. Routerstaticpolicy same for graph/eager; graphcontrol sameexecutionmode. Host-aware policy uses measuredhostmiss+cleanup. Confirmatory comparison, no ranking/selecting policies by prior testperformance. No CPUcompact router retraining or cancelledmatrix restart.',
        timing_scope='Fresh pinnedRGBtransfer/GPU normalization, fresh213outputCNN andCPU6poolselection ifrouted, lazyhash/load/context/eviction whencapacity2, classifier/sync/hostclass. Startup separately.500image windows excludecheckpoint writes/checks; energyGPUonly includesCPUwait, excludesCPUenergy.')
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    if resuming:
        report=saved
        assert provenance==report['provenance'] and digest(profile_path)==report['profile_sha256']
        for path,expected in report['protected_hashes'].items():assert digest(path)==expected
        for path,expected in report['dependencies'].items():assert digest(path)==expected
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='compact_switching_confirmation',status='complete',**values));atomic_json(REPORT,report);check(__file__)
    def cleanup():gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    atomic_json(REPORT,report);buffers=SharedBatchInputs(maximum_batch=1)
    meter=EnergyCounter()
    if not meter.available:raise RuntimeError('NVML energy unavailable')
    routers={};plans=[]
    try:
      with torch.inference_mode():
        for seed in (42,45,48):
            check(__file__);_,cal=split_validation(labels,seed)
            path=PIPELINE/f'artifacts/correctness_b1_20261028/engines/correctness_seed{seed}_b1.engine'
            prepared=json.loads((PIPELINE/f'artifacts/correctness_b1_20261028/prepared_seed{seed}.json').read_text())
            assert digest(path)==prepared['engine_sha256'] and prepared['calibration_positions']==cal.tolist()
            routers[seed]=RouterRunner(path);q=[]
            if resuming:
                oldplan=json.loads((ARTIFACTS/'frozen_plan.json').read_text())
                p=next(p for p in oldplan['plans'] if p['seed']==seed)
                assert p['calibration_positions']==cal.tolist() and p['router_engine_sha256']==digest(path)
                plans.append(p)
                continue
            for i,pos in enumerate(cal):
                if i%100==0:check(__file__)
                h=torch.from_numpy(np.array(images[pos:pos+1])).pin_memory();q.append(routers[seed].run(buffers.prepare(h)).cpu().numpy().copy()[0,columns])
            q=np.array(q);target=correctness[cal].astype(float)
            static=calibrate_correctness(q,target,np.ones(2000),costs,base,1.)
            aware=calibrate_cache_aware(q,target,np.ones(2000),costs,miss,2,base,1.)
            minimum=100*target[:,base].mean()-1.;eligible=np.flatnonzero(100*target.mean(0)+1e-10>=minimum)
            p=dict(seed=seed,calibration_positions=cal.tolist(),static=static,aware=aware,router_engine_sha256=digest(path),
                   eligible_eager=int(min(eligible,key=lambda i:(costs[i],pool[i]))),eligible_graph=int(min(eligible,key=lambda i:(graphcosts[i],pool[i]))))
            plans.append(p);done(f'calibration_{seed}',kind='calibration',seed=seed)
        plan=dict(plans=plans,pool=pool,costs_seconds=costs.tolist(),host_miss_seconds=miss.tolist(),variants=list(VARIANTS),repeats=3,
            orders={f'{seed}_{repeat}':[str(v) for v in np.random.default_rng(seed+repeat*100).permutation(VARIANTS)] for seed in (42,45,48) for repeat in range(3)},
            fresh_validation_hashes={n:prof['fresh_original_validation'][n]['predictions_sha256'] for n in pool})
        plan_path=ARTIFACTS/'frozen_plan.json'
        if resuming:
            frozen=json.loads(plan_path.read_text())
            assert plan==frozen and digest(plan_path)==report['frozen_plan_sha256']
        else:
            atomic_json(plan_path,plan);report['frozen_plan_sha256']=digest(plan_path)
        # All models/policies/caseorders frozen before test access.
        test=np.load(cache_dir/'test_rgb128.npy',mmap_mode='r',allow_pickle=False);testlabels=np.load(cache_dir/'test_labels.npy',allow_pickle=False)
        assert test.shape==(10000,3,128,128) and testlabels.shape==(10000,)
        if resuming:
            for path,expected in report['test_hashes'].items():assert digest(path)==expected
        report.update(test_data_opened=True,test_hashes={str(cache_dir/f):digest(cache_dir/f) for f in ('test_rgb128.npy','test_labels.npy')},status='measuring')
        host=[torch.from_numpy(np.array(test[i:i+1])).pin_memory() for i in range(10000)]
        start=time.perf_counter();store=VerifiedHostStore(records);setup=dict(seconds=time.perf_counter()-start,host_bytes=store.host_bytes)
        if resuming:report.setdefault('resume_host_store_sessions',[]).append(setup)
        else:report['host_store']=setup
        atomic_json(REPORT,report)
        completed={(r['seed'],r['repeat'],r['variant']) for r in report['full_sweeps']}
        for p in plans:
          seed=p['seed'];order=np.random.default_rng(seed+1).permutation(10000)
          for repeat in range(3):
           for variant in plan['orders'][f'{seed}_{repeat}']:
            if (seed,repeat,variant) in completed:continue
            check(__file__);cleanup()
            control=variant.startswith(('reference_','eligible_'));graph='graph' in variant
            capacity=6 if graph or 'resident_' in variant or control else 2
            usehost=variant!='file_static_c2';reuse=variant=='host_static_reuse_c2';aware=variant=='host_aware_c2'
            factory=lambda r:create_runner(r,verified_bytes=store.get(r['name']) if usehost else None,cuda_graph=graph)
            cache=ProfiledLRU(records,capacity,sum(r['profiled_resident_bytes'] for r in records),factory,torch.cuda.synchronize,
                             (lambda:torch.cuda.synchronize()) if reuse else cleanup)
            setup=time.perf_counter()
            if capacity==6:
                for i in range(6):cache[i]
            torch.cuda.synchronize();setup_seconds=time.perf_counter()-setup
            free,_=torch.cuda.mem_get_info()
            if free<1024**3:raise RuntimeError('Reserve violated after setup')
            initial_loads=cache.loads;counts=np.zeros(6,dtype=np.int64);pred=np.empty(10000,dtype=np.int64);choices=np.empty(10000,dtype=np.int64)
            seconds=energy_total=decision_seconds=0.;policy=p['aware'] if aware else p['static']
            for first in range(0,10000,500):
                check(__file__);torch.cuda.synchronize();meter.start();start=time.perf_counter()
                for step in range(first,first+500):
                    pos=order[step];normalized=buffers.prepare(host[pos]);decision=time.perf_counter()
                    if control:index=base if variant.startswith('reference') else p['eligible_graph' if graph else 'eligible_eager']
                    else:
                        q=routers[seed].run(normalized).cpu().numpy().copy()[0,columns]
                        if policy['fallback']:index=base
                        elif aware:index=select_cache_aware(q,costs,miss,list(cache.resident),policy['penalty'])
                        else:index=int(select_correctness(q[None,:],costs,policy['penalty'])[0])
                    decision_seconds+=time.perf_counter()-decision
                    pred[pos]=cache[index].run(normalized,copy_input=False).argmax(1).item();choices[pos]=index;counts[index]+=1
                torch.cuda.synchronize();elapsed=time.perf_counter()-start;energy=meter.finish(500)
                if energy.get('gpu_energy_j') is None or energy['gpu_energy_j']<=0:raise RuntimeError('Chunkenergy unavailable')
                seconds+=elapsed;energy_total+=energy['gpu_energy_j']
                free,_=torch.cuda.mem_get_info()
                if free<1024**3:raise RuntimeError('1GiBreserve violated')
                done(f'test_{seed}_{repeat}_{variant}_{first+500}',kind='test_chunk',seed=seed,repeat=repeat,variant=variant,completed_images=first+500,
                     images=500,seconds=elapsed,gpu_energy=energy,observed_free_bytes=free)
            path=ARTIFACTS/f'{variant}_seed{seed}_repeat{repeat}.npz';np.savez_compressed(path,predictions=pred,choices=choices,order=order)
            row=dict(seed=seed,repeat=repeat,variant=variant,images=10000,accuracy=100*float((pred==testlabels).mean()),ms_per_image=1000*seconds/10000,
                seconds=seconds,gpu_energy_j=energy_total,gpu_j_per_image=energy_total/10000,decision_ms=1000*decision_seconds/10000,
                setup_seconds=setup_seconds,capacity=capacity,selected_images=counts.tolist(),runtime_loads=cache.loads-initial_loads,evictions=cache.evictions,
                runtime_file=str(path),runtime_sha256=digest(path),comparator_mode='graph' if graph else 'eager')
            report['full_sweeps'].append(row);atomic_json(REPORT,report);print(f'COMPLETE {len(report["full_sweeps"])}/81 seed{seed} repeat{repeat} {variant}',flush=True)
            cache.clear();del cache;cleanup()
        assert len(report['full_sweeps'])==81 and digest(plan_path)==report['frozen_plan_sha256']
        for p,h in report['protected_hashes'].items():assert digest(p)==h
        report['status']='complete';atomic_json(REPORT,report)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(REPORT,report);raise

if __name__=='__main__':main()
