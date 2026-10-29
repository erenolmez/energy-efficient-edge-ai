"""ActualTRTparity/fulltest for fivepredeclaredseed42fit variants andcal sizes."""
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
from pc_tests.model_pipeline.routers.correctness.correctness_policy import calibrate_correctness,select_correctness

REPORT=PIPELINE/'reports/supervision_deployment_20261029/study.json'
ARTIFACTS=PIPELINE/'artifacts/supervision_deployment_20261029'

def main():
    if REPORT.exists():raise RuntimeError('Existing deploymentscreen retained; no blind restart')
    wait(__file__)
    import torch
    from pc_tests.model_pipeline.routers.correctness.correctness_router import CorrectnessRouter
    from pc_tests.model_pipeline.runtime.router_tensorrt import build_router,RouterRunner
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    torch.set_num_threads(1)
    prior=PIPELINE/'reports/gpu_conditional_20261029/audited_summary.json';assert json.loads(prior.read_text())['audit_passed']
    training_path=PIPELINE/'reports/router_supervision_screen_20261029/study.json'
    training=json.loads(training_path.read_text());assert json.loads((training_path.parent/'audited_summary.json').read_text())['audit_passed']
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,_,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
    names=[r['name'] for r in records];byname={r['name']:r for r in records}
    profile=json.loads((PIPELINE/'reports/compact_runtime_profile_20261028/study.json').read_text());pool=profile['pool'];columns=[names.index(n) for n in pool];base=pool.index(baseline)
    costs=np.array([next(r['eager_ms'] for r in json.loads((PIPELINE/'reports/compact_runtime_profile_20261028/audited_summary.json').read_text())['models'] if r['name']==n) for n in pool])
    cal=np.array(training['calibration_positions']);correct=np.column_stack([np.load(profile['fresh_original_validation'][n]['predictions_path'],allow_pickle=False)[cal]==labels[cal] for n in pool])
    validation=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    report=dict(status='preparing',results=[],full_sweeps=[],prepared=[],seed=42,pool=pool,source_sha256=digest(Path(__file__)),predecessor_sha256=digest(prior),training_sha256=digest(training_path),
        provenance=provenance,test_data_opened=False,expected_full_sweeps=20,expected_chunks=400,
        protected_hashes={str(p):digest(p) for p in [PIPELINE/'reports/pc_routes_correctness.json',PIPELINE/'reports/gpu_static.json',training_path,PIPELINE/'reports/compact_switching_confirmation_20261028/study.json']},
        protocol='Predeclaredseed42onlyscreen, all5fit variants xcal500/1000/2000=15cases plusoriginalCNNcontrolcal2000,freshR101reference,3cal-size-matchedeligiblestaticcontrols=20fullsweeps. One10000image sweep each, same6originalresidenteagerengines. No test-basedmodel/seed/policyselection. ActualnewTRTexports/random+real2000parity andfreshTRTcalibration beforetest. Internalcheckpointselection frozenfrom600fit-heldoutBrier, nottest. Balancedlogweightcorrectionsameastraining. FreshGPUpreprocessing/router/CPUselection/classifier/hostclass timed; GPUonlyenergy excludesCPU; startupseparate.',
        created_utc=datetime.now(timezone.utc).isoformat())
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='supervision_deployment',status='complete',**values));atomic_json(REPORT,report);check(__file__)
    atomic_json(REPORT,report);buffers=SharedBatchInputs(maximum_batch=1);runners={};settings={}
    meter=EnergyCounter()
    if not meter.available:raise RuntimeError('NVMLenergy unavailable')
    class Corrected(torch.nn.Module):
        def __init__(self,model,weight):
            super().__init__();self.model=model;self.register_buffer('logweight',weight.log())
        def forward(self,x):return torch.sigmoid(self.model(x)-self.logweight)
    try:
      with torch.inference_mode():
        oldpath=PIPELINE/'artifacts/correctness_b1_20261028/engines/correctness_seed42_b1.engine'
        oldmarker=json.loads((PIPELINE/'artifacts/correctness_b1_20261028/prepared_seed42.json').read_text());assert digest(oldpath)==oldmarker['engine_sha256']
        specs=[dict(name='original_control',engine=str(oldpath),weights_sha256=oldmarker['weights_sha256'])]+training['fits']
        calsets={p['calibration_images']:np.array(p['calibration_positions']) for p in training['fits'][0]['policies']}
        eligible={}
        for n,positions in calsets.items():
            mask=np.isin(cal,positions);minimum=100*correct[mask,base].mean()-1.
            valid=np.flatnonzero(100*correct[mask].mean(0)+1e-10>=minimum);eligible[n]=int(min(valid,key=lambda i:(costs[i],pool[i])))
        for spec in specs:
            name=spec['name'];check(__file__);expected=None;error=None
            if name=='original_control':path=oldpath
            else:
                assert digest(spec['weights'])==spec['weights_sha256']
                latest=next(r for r in training['results'] if r['kind']=='epoch' and r['fit']==name and r['epoch']==spec['epochs'])
                assert digest(latest['checkpoint'])==latest['checkpoint_sha256']
                checkpoint=torch.load(latest['checkpoint'],map_location='cpu',weights_only=False)
                weight=torch.from_numpy(np.asarray(checkpoint['weight'],dtype=np.float32)).cuda()
                model=CorrectnessRouter(213).cuda();model.load_state_dict(torch.load(spec['weights'],map_location='cuda',weights_only=True));wrapped=Corrected(model,weight).eval()
                path,error=build_router(wrapped,'normalized_images',ARTIFACTS/'engines',name,input_shape=(1,3,128,128));assert error<=1e-5
                assert digest(spec['calibration_file'])==spec['calibration_sha256']
                with np.load(spec['calibration_file'],allow_pickle=False) as data:expected=data['probabilities'].copy();assert np.array_equal(data['positions'],cal)
            runners[name]=RouterRunner(path);q=[]
            for i,pos in enumerate(cal):
                if i%100==0:check(__file__)
                h=torch.from_numpy(np.array(validation[pos:pos+1])).pin_memory();q.append(runners[name].run(buffers.prepare(h)).cpu().numpy().copy()[0])
            q=np.array(q);realerror=float(np.max(np.abs(q-expected))) if expected is not None else None
            if realerror is not None and realerror>1e-5:raise RuntimeError('Realvalidation routerparity exceedsbound')
            for n in ([2000] if name=='original_control' else [500,1000,2000]):
                mask=np.isin(cal,calsets[n]);policy=calibrate_correctness(q[mask][:,columns],correct[mask].astype(float),np.ones(n),costs,base,1.)
                settings[name+'__cal'+str(n)]=dict(method=name,calibration_images=n,policy=policy)
            report['prepared'].append(dict(name=name,engine=str(path),engine_sha256=digest(path),weights_sha256=spec['weights_sha256'],random_parity_error=error,real_parity_error=realerror))
            done('prepare_'+name,kind='prepare',name=name)
            if name!='original_control':del checkpoint,weight,model,wrapped;gc.collect();torch.cuda.empty_cache()
        settings['reference']=dict(method=None,index=base,calibration_images=2000)
        for n,index in eligible.items():settings['eligible_'+str(n)]=dict(method=None,index=index,calibration_images=n)
        order=[str(x) for x in np.random.default_rng(4242).permutation(list(settings))]
        plan=dict(settings=settings,case_order=order,pool=pool,costs=costs.tolist(),calibration_subsets={str(n):p.tolist() for n,p in calsets.items()},prepared=report['prepared'])
        planpath=ARTIFACTS/'frozen_plan.json';atomic_json(planpath,plan);report['frozen_plan_sha256']=digest(planpath)
        test=np.load(cache/'test_rgb128.npy',mmap_mode='r',allow_pickle=False);testlabels=np.load(cache/'test_labels.npy',allow_pickle=False)
        report.update(status='measuring',test_data_opened=True,test_hashes={str(cache/f):digest(cache/f) for f in ('test_rgb128.npy','test_labels.npy')})
        indices=np.random.default_rng(43).permutation(10000);hosts=[torch.from_numpy(np.array(test[i:i+1])).pin_memory() for i in indices]
        for n in pool:assert digest(byname[n]['engine_path'])==byname[n]['engine_sha256']
        start=time.perf_counter();engines=[EngineRunner(Path(byname[n]['engine_path']),maximum_batch=1) for n in pool]
        report['resident_setup_seconds']=time.perf_counter()-start;atomic_json(REPORT,report)
        for case in order:
            setting=settings[case];check(__file__);pred=np.empty(10000,dtype=np.int64);choices=np.empty(10000,dtype=np.int64);seconds=joules=0.;counts=np.zeros(6,dtype=np.int64)
            for first in range(0,10000,500):
                check(__file__);torch.cuda.synchronize();meter.start();start=time.perf_counter()
                for i in range(first,first+500):
                    normalized=buffers.prepare(hosts[i])
                    if setting['method'] is None:index=setting['index']
                    else:
                        q=runners[setting['method']].run(normalized).cpu().numpy().copy()[:,columns];p=setting['policy']
                        index=base if p['fallback'] else int(select_correctness(q,costs,p['penalty'])[0])
                    pos=indices[i];pred[pos]=engines[index].run(normalized,copy_input=False).argmax(1).item();choices[pos]=index;counts[index]+=1
                torch.cuda.synchronize();elapsed=time.perf_counter()-start;energy=meter.finish(500)
                if energy.get('gpu_energy_j') is None:raise RuntimeError('Deploymentchunkenergy unavailable')
                free,_=torch.cuda.mem_get_info()
                if free<1024**3:raise RuntimeError('1GiBreserve violated')
                seconds+=elapsed;joules+=energy['gpu_energy_j']
                done(f'test_{case}_{first+500}',kind='test_chunk',case=case,images=500,completed_images=first+500,seconds=elapsed,gpu_energy=energy,observed_free_bytes=free)
            path=ARTIFACTS/(case+'_runtime.npz');np.savez_compressed(path,predictions=pred,choices=choices,order=indices)
            report['full_sweeps'].append(dict(case=case,method=setting['method'],calibration_images=setting['calibration_images'],images=10000,accuracy=100*float((pred==testlabels).mean()),
                seconds=seconds,ms_per_image=1000*seconds/10000,gpu_energy_j=joules,gpu_j_per_image=joules/10000,selected_images=counts.tolist(),runtime_file=str(path),runtime_sha256=digest(path)))
            atomic_json(REPORT,report);print(f'COMPLETE {len(report["full_sweeps"])}/20 {case}',flush=True)
        assert len(report['full_sweeps'])==20 and digest(planpath)==report['frozen_plan_sha256']
        for p,h in report['protected_hashes'].items():assert digest(p)==h
        report['status']='complete';atomic_json(REPORT,report)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(REPORT,report);raise

if __name__=='__main__':main()
