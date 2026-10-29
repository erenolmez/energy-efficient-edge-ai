"""Same-pool batch8: one classifier perbatch vs perimage grouped dispatch."""
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
from pc_tests.model_pipeline.runtime.experiment_guard import check,wait
from pc_tests.model_pipeline.runtime.grouped_dispatch import groups
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
from pc_tests.model_pipeline.routers.correctness.correctness_policy import calibrate_correctness,select_correctness
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import costs_for

REPORT=PIPELINE/'reports/batch8_grouping_20261029/study.json'
ARTIFACTS=PIPELINE/'artifacts/batch8_grouping_20261029'
MODES=('batch_choice','per_image','reference','eligible_static')

def main():
    if REPORT.exists():raise RuntimeError('Existing grouping comparison preserved; no blind restart')
    wait(__file__)
    import torch
    from pc_tests.model_pipeline.routers.correctness.correctness_router import CorrectnessRouter
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner,build_router
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    torch.set_num_threads(1)
    audit=PIPELINE/'reports/router_supervision_screen_20261029/audited_summary.json';assert json.loads(audit.read_text())['audit_passed']
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,_,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
    profile=json.loads((PIPELINE/'reports/compact_runtime_profile_20261028/study.json').read_text());pool=profile['pool']
    names=[r['name'] for r in records];columns=[names.index(n) for n in pool];base=pool.index(baseline)
    byname={r['name']:r for r in records};records=[byname[n] for n in pool]
    summary=json.loads((PIPELINE/'reports/compact_runtime_profile_20261028/audited_summary.json').read_text())
    cost1=np.array([next(r['eager_ms'] for r in summary['models'] if r['name']==n) for n in pool]);cost8=costs_for(records,8,'latency')
    val=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    report=dict(status='preparing',results=[],full_sweeps=[],pool=pool,baseline=baseline,seeds=[42,45,48],modes=list(MODES),repeats=3,
        expected_full_sweeps=36,expected_test_chunks=900,source_sha256=digest(Path(__file__)),test_data_opened=False,
        predecessor_sha256=digest(audit),provenance=provenance,protected_hashes={str(p):digest(p) for p in [PIPELINE/'reports/pc_routes_correctness.json',PIPELINE/'reports/compact_switching_confirmation_20261028/study.json']},
        protocol='Threeoriginalseeds3repeats:batch8onechoice vs independentperimagechoices withGPUgather/classifiergroups/scatter/reassembly, plusfreshR101FP32 andeligible-static batch8controls. Same6originalresidentengines, true8images everybatch. Fresh213outputrouter, CPUprobabilities/selection; samefinalhostclasses. Newperimage-output b8exports reuseweights, realmeanparity vsoldbatchrouter checked beforetest. Policies use freshclassifiercalibration outcomes; actualgroupedcalvalidation1ppgate beforetest; failedpolicy fallsbackto reference usingvalidationonly. No test tuning.',
        timing_scope='Pinnedbatch ->GPUtransfer/normalization ->freshrouter/CPUdecision ->GPUgather/groupinfer/scatter ->host8classes. Setup excluded, GPUonlyenergy excludesCPU.400image chunks, no checkpointwrites inside timing.',created_utc=datetime.now(timezone.utc).isoformat())
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='batch8_grouping',status='complete',**values));atomic_json(REPORT,report);check(__file__)
    atomic_json(REPORT,report);buffers=SharedBatchInputs(maximum_batch=8)
    meter=EnergyCounter()
    if not meter.available:raise RuntimeError('NVML energy unavailable')
    routers={};plans=[];engines=[]
    try:
      with torch.inference_mode():
        for record in records:
            check(__file__);assert digest(record['engine_path'])==record['engine_sha256'];engines.append(EngineRunner(Path(record['engine_path']),maximum_batch=8))
        free,_=torch.cuda.mem_get_info()
        if free<1024**3:raise RuntimeError('1GiBreserve violated')
        def forward(h,mode,seed,policy,static_index):
            normalized=buffers.prepare(h)
            if mode in ('reference','eligible_static'):
                index=base if mode=='reference' else static_index
                return engines[index].run(normalized,copy_input=False).argmax(1).cpu().numpy(),np.full(8,index,dtype=np.int64),1
            q=routers[seed].run(normalized).cpu().numpy().copy()[:,columns]
            if mode=='batch_choice':
                index=base if policy['fallback'] else int(select_correctness(q.mean(0,keepdims=True),cost8,policy['penalty'])[0])
                return engines[index].run(normalized,copy_input=False).argmax(1).cpu().numpy(),np.full(8,index,dtype=np.int64),1
            selected=np.full(8,base,dtype=np.int64) if policy['fallback'] else select_correctness(q,cost1,policy['penalty'])
            output=torch.empty(8,device='cuda',dtype=torch.int64);parts=groups(selected,6)
            for index,positions in parts:
                indices=torch.tensor(positions,device='cuda',dtype=torch.int64)
                gathered=normalized.index_select(0,indices)
                predicted=engines[index].run(gathered,copy_input=False).argmax(1)
                output.index_copy_(0,indices,predicted)
            return output.cpu().numpy(),selected,len(parts)
        for seed in (42,45,48):
            check(__file__);_,cal=split_validation(labels,seed)
            marker_path=PIPELINE/f'artifacts/pc_routers_correctness/global_correctness_seed{seed}.json';marker=json.loads(marker_path.read_text())
            assert digest(marker['weights'])==marker['weights_sha256'] and marker['calibration_positions']==cal.tolist()
            model=CorrectnessRouter(213).cuda();model.load_state_dict(torch.load(marker['weights'],map_location='cuda',weights_only=True));model.eval()
            wrapped=torch.nn.Sequential(model,torch.nn.Sigmoid()).eval()
            path,error=build_router(wrapped,'normalized_images',ARTIFACTS/'engines',f'per_image_seed{seed}_b8',input_shape=(8,3,128,128))
            assert error<=1e-5;routers[seed]=RouterRunner(path)
            assert tuple(routers[seed].output.shape)==(8,213)
            old=marker['engines']['8'];assert digest(old['engine'])==old['sha256'];oldrouter=RouterRunner(old['engine'])
            hosts=[torch.from_numpy(np.array(val[cal[i:i+8]])).pin_memory() for i in range(0,2000,8)]
            q=[];actual=[];meanerror=0.
            for i,h in enumerate(hosts):
                if i%25==0:check(__file__)
                normalized=buffers.prepare(h)
                value=routers[seed].run(normalized).cpu().numpy().copy();expected=oldrouter.run(normalized).cpu().numpy().copy()
                meanerror=max(meanerror,float(np.abs(value.mean(0,keepdims=True)-expected).max()))
                q.append(value[:,columns]);actual.append(np.column_stack([e.run(normalized,copy_input=False).argmax(1).cpu().numpy() for e in engines]))
            assert meanerror<=1e-5
            q=np.concatenate(q);predicted=np.concatenate(actual);correct=predicted==labels[cal,None]
            p=dict(seed=seed,calibration_positions=cal.tolist(),new_router_engine=str(path),new_router_sha256=digest(path),original_weights_sha256=marker['weights_sha256'],
                random_parity_error=error,real_mean_parity_error=meanerror,policies={},reference_calibration_accuracy=100*float(correct[:,base].mean()))
            p['policies']['per_image']=calibrate_correctness(q,correct.astype(float),np.ones(2000),cost1,base,1.)
            p['policies']['batch_choice']=calibrate_correctness(q.reshape(250,8,6).mean(1),correct.reshape(250,8,6).sum(1),np.full(250,8),cost8,base,1.)
            minimum=p['reference_calibration_accuracy']-1.;eligible=np.flatnonzero(100*correct.mean(0)+1e-10>=minimum)
            p['eligible_index']=int(min(eligible,key=lambda i:(cost8[i],pool[i])))
            for mode in ('batch_choice','per_image'):
                output=np.concatenate([forward(h,mode,seed,p['policies'][mode],p['eligible_index'])[0] for h in hosts])
                accuracy=100*float((output==labels[cal]).mean());p.setdefault('actual_calibration_accuracy',{})[mode]=accuracy
                if accuracy+1e-10<minimum:
                    p['policies'][mode]=dict(fallback=True,baseline_index=base,penalty=0.,budget_pp=1.,reason='Actual groupedvalidation accuracy failed; referencefallback beforetest')
            plans.append(p);done(f'prepare_{seed}',kind='prepare',seed=seed,real_mean_parity_error=meanerror)
            del model,wrapped,oldrouter;gc.collect();torch.cuda.empty_cache()
        plan=dict(plans=plans,pool=pool,cost1=cost1.tolist(),cost8=cost8.tolist(),orders={f'{seed}_{rep}':[str(m) for m in np.random.default_rng(seed+rep*100).permutation(MODES)] for seed in (42,45,48) for rep in range(3)})
        planpath=ARTIFACTS/'frozen_plan.json';atomic_json(planpath,plan);report['frozen_plan_sha256']=digest(planpath)
        test=np.load(cache/'test_rgb128.npy',mmap_mode='r',allow_pickle=False);testlabels=np.load(cache/'test_labels.npy',allow_pickle=False)
        report.update(status='measuring',test_data_opened=True,test_hashes={str(cache/f):digest(cache/f) for f in ('test_rgb128.npy','test_labels.npy')});atomic_json(REPORT,report)
        for p in plans:
          seed=p['seed'];order=np.random.default_rng(seed+8).permutation(10000)
          host=[torch.from_numpy(np.array(test[order[i:i+8]])).pin_memory() for i in range(0,10000,8)]
          for rep in range(3):
           for mode in plan['orders'][f'{seed}_{rep}']:
            check(__file__);out=np.empty(10000,dtype=np.int64);choice=np.empty(10000,dtype=np.int64);seconds=joules=0.;invocations=0;counts=np.zeros(6,dtype=np.int64)
            for first in range(0,1250,50):
                check(__file__);torch.cuda.synchronize();meter.start();start=time.perf_counter()
                for batch in range(first,first+50):
                    pred,selected,n=forward(host[batch],mode,seed,p['policies'].get(mode),p['eligible_index'])
                    positions=order[batch*8:batch*8+8];out[positions]=pred;choice[positions]=selected;counts+=np.bincount(selected,minlength=6);invocations+=n
                torch.cuda.synchronize();elapsed=time.perf_counter()-start;energy=meter.finish(400)
                if energy.get('gpu_energy_j') is None:raise RuntimeError('Chunkenergy unavailable')
                seconds+=elapsed;joules+=energy['gpu_energy_j']
                free,_=torch.cuda.mem_get_info()
                if free<1024**3:raise RuntimeError('1GiBreserve violated duringgrouping')
                done(f'test_{seed}_{rep}_{mode}_{(first+50)*8}',kind='test_chunk',seed=seed,repeat=rep,mode=mode,images=400,completed_images=(first+50)*8,seconds=elapsed,gpu_energy=energy,observed_free_bytes=free)
            path=ARTIFACTS/f'{mode}_seed{seed}_repeat{rep}.npz';np.savez_compressed(path,predictions=out,choices=choice,order=order)
            report['full_sweeps'].append(dict(seed=seed,repeat=rep,mode=mode,images=10000,accuracy=100*float((out==testlabels).mean()),seconds=seconds,
                ms_per_image=1000*seconds/10000,ms_per_batch=1000*seconds/1250,gpu_energy_j=joules,gpu_j_per_image=joules/10000,
                classifier_invocations=invocations,selected_images=counts.tolist(),runtime_file=str(path),runtime_sha256=digest(path)))
            atomic_json(REPORT,report);print(f'COMPLETE {len(report["full_sweeps"])}/36 {seed} {rep} {mode}',flush=True)
        assert len(report['full_sweeps'])==36 and digest(planpath)==report['frozen_plan_sha256']
        for p,h in report['protected_hashes'].items():assert digest(p)==h
        report['status']='complete';atomic_json(REPORT,report)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(REPORT,report);raise

if __name__=='__main__':main()
