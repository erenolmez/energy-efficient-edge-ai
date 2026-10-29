"""Frozen preliminary batch-one full-test screen, same six resident originals.

All3 seeds; latency1pp only, one10000-image sweep permethod plus fresh controls.
This intentionally is not the cancelled full213choice, capacity2,360sweep matrix.
"""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation,digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
from pc_tests.model_pipeline.routers.correctness.correctness_policy import calibrate_correctness,select_correctness
from pc_tests.model_pipeline.runtime.experiment_guard import check,wait

REPORT=PIPELINE/'reports/compact_resident_screen_20261028/study.json'
ARTIFACTS=PIPELINE/'artifacts/compact_resident_screen_20261028'
METHODS=('logistic','tree_d3','thumbnail_mlp16','boosted_d2','tinycnn')

def main():
    if REPORT.exists():raise RuntimeError('Existing screen preserved; no blind restart')
    wait(__file__)
    import torch,joblib
    from threadpoolctl import threadpool_limits
    from pc_tests.model_pipeline.runtime.compact_subset import predict
    from pc_tests.model_pipeline.runtime.compact_trt import create_runner
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    torch.set_num_threads(1)
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    original=PIPELINE/'reports/compact_runtime_profile_20261028/study.json'
    profile_path=original.parent/'audited_summary.json'
    summary=json.loads(profile_path.read_text());assert summary['audit_passed']
    prof=json.loads(original.read_text());pool=prof['pool'];costs=np.array([next(r['eager_ms'] for r in summary['models'] if r['name']==n) for n in pool])
    records,labels,_,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
    names=[r['name'] for r in records];columns=np.array([names.index(n) for n in pool]);base=pool.index(baseline)
    byname={r['name']:r for r in records}
    preparation_path=PIPELINE/'artifacts/compact_b1_20261028/preparation.json'
    prep=json.loads(preparation_path.read_text());assert prep['status']=='complete' and len(prep['results'])==12 and prep['models']==names
    assert prep['family_source_sha256']==digest(PIPELINE/'routers/correctness/compact_cpu.py')
    validation=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    for n in pool:
        assert digest(prof['fresh_original_validation'][n]['predictions_path'])==prof['fresh_original_validation'][n]['predictions_sha256']
    actual=np.column_stack([np.load(prof['fresh_original_validation'][n]['predictions_path'],allow_pickle=False)==labels for n in pool])
    report=dict(status='preparing',results=[],full_sweeps=[],pool=pool,baseline=baseline,seeds=[42,45,48],methods=list(METHODS),
        objective='latency',accuracy_allowance_pp=1.,repeats=1,expected_full_sweeps=21,expected_full_chunks=420,
        source_sha256=digest(Path(__file__)),subset_source_sha256=digest(PIPELINE/'runtime/compact_subset.py'),
        preparation_sha256=digest(preparation_path),profile_sha256=digest(profile_path),provenance=provenance,
        protected_hashes={str(p):digest(p) for p in [PIPELINE/'reports/gpu_static.json',PIPELINE/'reports/pc_routes_correctness.json',PIPELINE/'reports/correctness_b1_20261028/study.json']},
        protocol='Preliminary one full10000image pass percase. All3seeds, fourCPUrouters plus samepool original213outputTinyCNN. Frozen6resident original engines, eager inference. IndependentCPUheads evaluate6available heads; sharedMLP/tree/CNN retain full213outputs. Fresh CPUfeatures/prediction, GPUpreprocessing,selection,inference,hostclass timed; no cached testchoices/features. Setup separate, no eviction. Fresh R101FP32 and validationeligible static controls perseed. Not full213choice/capacity2 matrix; no test tuning.',
        energy_scope='Gross NVML GPU energy includes GPU idle during CPUwork, excludes CPU/restPC; no CPU training-energy payback',
        created_utc=datetime.now(timezone.utc).isoformat(),test_data_opened=False)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='compact_resident_screen',status='complete',**values));atomic_json(REPORT,report);check(__file__)
    atomic_json(REPORT,report)
    meter=EnergyCounter()
    if not meter.available:raise RuntimeError('NVML energy unavailable')
    cpu={};routers={};plans=[];buffers=SharedBatchInputs(maximum_batch=1)
    try:
      with threadpool_limits(limits=1),torch.inference_mode():
        for seed in (42,45,48):
            check(__file__);_,cal=split_validation(labels,seed)
            router_path=PIPELINE/f'artifacts/correctness_b1_20261028/engines/correctness_seed{seed}_b1.engine'
            historical=json.loads((PIPELINE/f'artifacts/correctness_b1_20261028/prepared_seed{seed}.json').read_text())
            assert digest(router_path)==historical['engine_sha256'] and historical['calibration_positions']==cal.tolist()
            routers[seed]=RouterRunner(router_path)
            qcnn=[]
            for i,pos in enumerate(cal):
                if i%100==0:check(__file__)
                h=torch.from_numpy(np.array(validation[pos:pos+1])).pin_memory()
                qcnn.append(routers[seed].run(buffers.prepare(h)).cpu().numpy().copy()[0,columns])
            minimum=100*actual[cal,base].mean()-1.
            eligible=[i for i in range(6) if 100*actual[cal,i].mean()+1e-10>=minimum]
            static=min(eligible,key=lambda i:(costs[i],pool[i]))
            seedplan=dict(seed=seed,calibration_positions=cal.tolist(),static_index=int(static),reference_index=base,
                router_engine_sha256=digest(router_path),policies={})
            for method in METHODS:
                if method=='tinycnn':q=np.array(qcnn)
                else:
                    saved=next(r for r in prep['results'] if r['seed']==seed and r['method']==method)
                    assert saved['calibration_positions']==cal.tolist()
                    assert digest(saved['model_file'])==saved['model_sha256'] and digest(saved['calibration_file'])==saved['calibration_sha256']
                    cpu[seed,method]=joblib.load(saved['model_file'])
                    with np.load(saved['calibration_file'],allow_pickle=False) as data:
                        assert np.array_equal(data['positions'],cal)
                        q=data['probabilities'][:,columns]
                    seedplan.setdefault('models',{})[method]=dict(model_sha256=saved['model_sha256'],calibration_sha256=saved['calibration_sha256'])
                seedplan['policies'][method]=calibrate_correctness(q,actual[cal].astype(float),np.ones(2000),costs,base,1.)
            plans.append(seedplan);done(f'calibration_{seed}',kind='calibration',seed=seed,images=2000)
        plan=dict(plans=plans,pool=pool,columns=columns.tolist(),costs_ms=costs.tolist(),objective='latency',budget_pp=1.,repeats=1,
            fresh_validation_hashes={n:prof['fresh_original_validation'][n]['predictions_sha256'] for n in pool},
            case_order={str(seed):[str(m) for m in np.random.default_rng(seed).permutation(['reference','eligible_static',*METHODS])] for seed in (42,45,48)})
        plan_path=ARTIFACTS/'frozen_plan.json';atomic_json(plan_path,plan)
        report['frozen_plan_sha256']=digest(plan_path);report['status']='measuring';atomic_json(REPORT,report)
        # Plan is immutable before any test arrays are opened.
        images=np.load(cache/'test_rgb128.npy',mmap_mode='r',allow_pickle=False)
        testlabels=np.load(cache/'test_labels.npy',allow_pickle=False)
        assert images.shape==(10000,3,128,128) and testlabels.shape==(10000,)
        report['test_data_opened']=True;report['test_rgb_sha256']=digest(cache/'test_rgb128.npy');report['test_labels_sha256']=digest(cache/'test_labels.npy')
        host=[torch.from_numpy(np.array(images[i:i+1])).pin_memory() for i in range(10000)]
        meter.start();start=time.perf_counter();engines=[create_runner(byname[n]) for n in pool];torch.cuda.synchronize()
        free,_=torch.cuda.mem_get_info()
        if free<1024**3:raise RuntimeError('Residentpool1GiBreserve violated')
        done('resident_setup',kind='setup',seconds=time.perf_counter()-start,gpu_energy=meter.finish(1),observed_free_bytes=free)
        for p in plans:
          seed=p['seed'];order=np.random.default_rng(seed+1).permutation(10000)
          for method in plan['case_order'][str(seed)]:
            check(__file__);correct=0;energy_total=0.;elapsed_total=0.;decision_total=0.;counts=np.zeros(6,dtype=np.int64)
            predictions=np.empty(10000,dtype=np.int64);decisions=np.empty(10000,dtype=np.int64)
            for first in range(0,10000,500):
              check(__file__);torch.cuda.synchronize();meter.start();start=time.perf_counter()
              for step in range(first,first+500):
                pos=order[step];h=host[pos];normalized=buffers.prepare(h)
                decision_start=time.perf_counter()
                if method in ('reference','eligible_static'):
                    index=base if method=='reference' else p['static_index']
                else:
                    policy=p['policies'][method]
                    if method=='tinycnn':q=routers[seed].run(normalized).cpu().numpy().copy()[:,columns]
                    else:q=predict(cpu[seed,method],h.numpy(),columns)
                    index=base if policy['fallback'] else int(select_correctness(q,costs,policy['penalty'])[0])
                decision_total+=time.perf_counter()-decision_start
                predictions[pos]=engines[index].run(normalized,copy_input=False).argmax(1).item();decisions[pos]=index;counts[index]+=1
              torch.cuda.synchronize();elapsed=time.perf_counter()-start;energy=meter.finish(500)
              if energy.get('gpu_energy_j') is None or energy['gpu_energy_j']<=0:raise RuntimeError('Full chunk GPU energy unavailable')
              elapsed_total+=elapsed;energy_total+=energy['gpu_energy_j']
              done(f'test_{seed}_{method}_{first+500}',kind='test_chunk',seed=seed,method=method,images=500,completed_images=first+500,
                   seconds=elapsed,gpu_energy=energy)
            path=ARTIFACTS/f'{method}_seed{seed}_runtime.npz';np.savez_compressed(path,predictions=predictions,choices=decisions,order=order)
            row=dict(seed=seed,method=method,images=10000,accuracy=100*float((predictions==testlabels).mean()),seconds=elapsed_total,
                ms_per_image=1000*elapsed_total/10000,gpu_energy_j=energy_total,gpu_j_per_image=energy_total/10000,
                decision_ms_per_image=1000*decision_total/10000,selected_images=counts.tolist(),runtime_loads_inside_sweep=0,runtime_evictions=0,
                runtime_file=str(path),runtime_sha256=digest(path))
            report['full_sweeps'].append(row);atomic_json(REPORT,report);print('COMPLETE '+str(seed)+' '+method+' '+str(row['accuracy'])+' '+str(row['ms_per_image']),flush=True)
        assert len(report['full_sweeps'])==21 and digest(plan_path)==report['frozen_plan_sha256']
        for path,h in report['protected_hashes'].items():assert digest(path)==h
        report['status']='complete';atomic_json(REPORT,report)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(REPORT,report);raise

if __name__=='__main__':main()
