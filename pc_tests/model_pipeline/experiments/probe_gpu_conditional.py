"""Bounded two-classifier nativeGPU If prototype, validation-onlyengineering."""
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
from pc_tests.model_pipeline.runtime.conditional_onnx import fixed_model,make_conditional
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
from pc_tests.model_pipeline.routers.correctness.correctness_policy import calibrate_correctness
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import costs_for

REPORT=PIPELINE/'reports/gpu_conditional_20261029/study.json'
ARTIFACTS=PIPELINE/'artifacts/gpu_conditional_20261029'

def main():
    if REPORT.exists():raise RuntimeError('Existing conditionalprobe retained; no blind restart')
    wait(__file__)
    import torch,onnx
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import trt,TRT_LOGGER,EngineRunner
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner
    from pc_tests.model_pipeline.runtime.fixed_gpu_runner import FixedGPURunner
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    torch.set_num_threads(1)
    prior=PIPELINE/'reports/dispatch_controls_20261029/audited_summary.json';assert json.loads(prior.read_text())['audit_passed']
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,_,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
    names=[r['name'] for r in records];pair=['resnet101__p0__fp32','efficientnet_b0__p0__fp32']
    byname={r['name']:r for r in records};records=[byname[n] for n in pair];columns=[names.index(n) for n in pair]
    cost=costs_for(records,1,'latency');_,cal=split_validation(labels,42)
    images=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    manifest=json.loads((PIPELINE/'artifacts/quantization/deployment_manifest.json').read_text())
    routergraph=PIPELINE/'artifacts/correctness_b1_20261028/engines/correctness_seed42_b1.onnx'
    routerengine=routergraph.with_suffix('.engine')
    report=dict(status='running',results=[],test_data_opened=False,seed=42,pair=pair,source_sha256=digest(Path(__file__)),
        helper_sha256=digest(PIPELINE/'runtime/conditional_onnx.py'),fixedrunner_sha256=digest(PIPELINE/'runtime/fixed_gpu_runner.py'),predecessor_sha256=digest(prior),
        provenance=provenance,router_graph_sha256=digest(routergraph),router_engine_sha256=digest(routerengine),
        protected_hashes={str(p):digest(p) for p in [PIPELINE/'reports/gpu_static.json',PIPELINE/'reports/pc_routes_correctness.json']},
        prototype_supported=False,branch_execution_verified=False,
        protocol='Boundedvalidation-onlyseed42prototype,twoFP32classifiers(R101/EffB0),sameoldCNNweights. Rebuildstandalonefixedb1FP32branches andnativeONNXIfcombinedgraph; no testdata. Calibratedbudgetpolicy recorded; predefinedmediancompetencescorebias fromcalibration ONLY ensuresbothbranches exercised forengineering, NOTanaccuracy-budgetdeploymentclaim. Requirelazyconditionalinputs/branchconvdependency,logit/choiceparity andprofilingbothbranchcases beforetiming. No Where/runboth surrogate. TypedGPUchoiceoutput isreadonly fordiagnostics, notintermediatedispatchintiming.',
        created_utc=datetime.now(timezone.utc).isoformat(),documentation='NVIDIA TensorRT10.x WorkingwithConditionals:conditionalinputboundaries determine lazyexecution; installedAPI/buildactualsupport tested')
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='gpu_conditional',status='complete',**values));atomic_json(REPORT,report);check(__file__)
    def unsupported(reason):
        report.update(status='complete',prototype_supported=False,unsupported_reason=reason);atomic_json(REPORT,report)
    def parse(model):
        builder=trt.Builder(TRT_LOGGER);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(network,TRT_LOGGER)
        if not parser.parse(model.SerializeToString()):raise RuntimeError('ONNXparse: '+';'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        return builder,network,parser
    def build(builder,network,path):
        config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,256*1024**2);config.clear_flag(trt.BuilderFlag.TF32);config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        value=builder.build_serialized_network(network,config)
        if value is None:raise RuntimeError('Conditional/standalonebuild returnedNone')
        path.write_bytes(bytes(value))
    def lazy_evidence(network):
        layers=[network.get_layer(i) for i in range(network.num_layers)]
        producers={layer.get_output(j).name:i for i,layer in enumerate(layers) for j in range(layer.num_outputs) if layer.get_output(j) is not None}
        memo={}
        def lazy(name,seen=()):
            if name in memo:return memo[name]
            if name not in producers:return False
            i=producers[name]
            if i in seen:return False
            layer=layers[i];kind=str(layer.type)
            if 'CONDITIONAL_INPUT' in kind:memo[name]=True;return True
            value=any(lazy(layer.get_input(j).name,seen+(i,)) for j in range(layer.num_inputs) if layer.get_input(j) is not None)
            memo[name]=value;return value
        evidence={}
        for prefix in ('A__','B__'):
            relevant=[l for l in layers if prefix in l.name and any(t in str(l.type) for t in ('CONVOLUTION','MATRIX_MULTIPLY'))]
            evidence[prefix]=dict(compute_layers=len(relevant),all_compute_lazy=bool(relevant) and all(lazy(l.get_output(0).name) for l in relevant))
        return dict(boundaries=[dict(name=l.name,type=str(l.type)) for l in layers if 'CONDITION' in str(l.type)],branches=evidence)
    atomic_json(REPORT,report);meter=EnergyCounter()
    if not meter.available:raise RuntimeError('NVMLenergy unavailable')
    buffers=SharedBatchInputs(maximum_batch=1);models=[];runners=[]
    try:
      with torch.inference_mode():
        for record in records:
            check(__file__);source=next(r for r in manifest['models'] if r['architecture']==record['architecture'] and r['prune']==0)
            path=Path(source['onnx']);assert digest(path)==record['source_sha256']
            model=fixed_model(onnx.load(str(path)));models.append(model)
            destination=ARTIFACTS/(record['name']+'_fixed_fp32.engine');start=time.perf_counter()
            builder,network,parser=parse(model);build(builder,network,destination)
            runners.append(EngineRunner(destination,maximum_batch=1));free,_=torch.cuda.mem_get_info()
            if free<1024**3:raise RuntimeError('1GiBreserve violated')
            done('branch_'+record['name'],kind='branch_build',name=record['name'],source_sha256=digest(path),engine=str(destination),engine_sha256=digest(destination),seconds=time.perf_counter()-start)
            del builder,network,parser;gc.collect()
        router=RouterRunner(routerengine);q=[];correct=[]
        for i,pos in enumerate(cal):
            if i%100==0:check(__file__)
            h=torch.from_numpy(np.array(images[pos:pos+1])).pin_memory();normalized=buffers.prepare(h)
            q.append(router.run(normalized).cpu().numpy().copy()[0,columns]);correct.append([int(r.run(normalized,copy_input=False).argmax(1).item())==labels[pos] for r in runners])
        q=np.array(q);correct=np.array(correct)
        budget=calibrate_correctness(q,correct.astype(float),np.ones(2000),cost,0,1.)
        penalty=float(budget['penalty']);offset=(penalty*cost/cost.max()).astype(np.float32)
        difference=(q[:,0]-offset[0])-(q[:,1]-offset[1]);bias=float(np.median(difference))
        report.update(budget_policy=budget,engineering_bias=bias,calibration_positions=cal.tolist(),costs=cost.tolist(),
                      engineering_policy_note='Medianbias intentionallyexercisebothbranches, notclaimed1ppaccuracy-compliant; no testdata used')
        done('policy',kind='policy',images=2000)
        conditional=make_conditional(onnx.load(str(routergraph)),models[0],models[1],columns,cost,penalty,bias)
        graphpath=ARTIFACTS/'conditional_fp32.onnx';onnx.save(conditional,str(graphpath))
        try:
            builder,network,parser=parse(conditional);evidence=lazy_evidence(network);report['lazy_structure']=evidence
            if not all(v['all_compute_lazy'] for v in evidence['branches'].values()):unsupported('Parserdidnotexposeprovablylazyclassifiercomputeboundaries');return
            path=ARTIFACTS/'conditional_fp32.engine';start=time.perf_counter();build(builder,network,path)
        except RuntimeError as exc:unsupported(str(exc));return
        combined=FixedGPURunner(path)
        done('conditional_build',kind='conditional_build',engine=str(path),engine_sha256=digest(path),graph_sha256=digest(graphpath),seconds=time.perf_counter()-start,lazy_structure=evidence)
        hosts=[torch.from_numpy(np.array(images[p:p+1])).pin_memory() for p in cal[:512]]
        probes={};choice_mismatches=0;maxerror=0.;classchanges=0
        def cpu_choice(value):
            scores=value.astype(np.float32)-offset
            return 0 if np.float32(scores[0]-scores[1])>=np.float32(bias) else 1
        for h in hosts:
            normalized=buffers.prepare(h);output=combined.run(normalized);chosen=int(output['choice'].item())
            logits=output['logits'].cpu().numpy().copy();expected=runners[chosen].run(normalized,copy_input=False).cpu().numpy().copy()
            maxerror=max(maxerror,float(np.max(np.abs(logits-expected))));classchanges+=int(logits.argmax()!=expected.argmax())
            if not np.allclose(logits,expected,atol=1e-3,rtol=1e-4):raise RuntimeError('Conditionalbranch FP32logitparity failed')
            oldq=router.run(normalized).cpu().numpy()[0,columns];choice_mismatches+=cpu_choice(oldq)!=chosen
            probes.setdefault(chosen,h)
        report['parity']=dict(images=512,max_abs_logit_error=maxerror,class_changes=classchanges,cpu_vs_gpu_choice_mismatches=choice_mismatches,branches_seen=sorted(probes))
        done('parity',kind='parity',**report['parity'])
        if classchanges:unsupported('Someconditionalclasspredictionsdifferfromchosenstandalonebranch;notinterchangeable');return
        if len(probes)!=2:unsupported('Built/paritypassed,butcalibrationprobesdidnotexercisebothbranches');return
        class LayerProfile(trt.IProfiler):
            def __init__(self):trt.IProfiler.__init__(self);self.times=[]
            def report_layer_time(self,name,ms):self.times.append(dict(name=name,ms=float(ms)))
        profiler=LayerProfile();combined.context.profiler=profiler;combined.context.enqueue_emits_profile=True
        trace=[]
        for chosen,h in sorted(probes.items()):
            profiler.times=[];combined.run(buffers.prepare(h))['logits'].argmax(1).item();torch.cuda.synchronize()
            if not profiler.times:combined.context.report_to_profiler()
            entries=list(profiler.times)
            masses={prefix:sum(r['ms'] for r in entries if prefix in r['name'] and any(k in r['name'].lower() for k in ('conv','gemm','matmul'))) for prefix in ('A__','B__')}
            trace.append(dict(chosen=chosen,times=entries,compute_ms_by_branch=masses))
        # Recreate context rather than depend on profiler-detachment API support.
        del combined;gc.collect();torch.cuda.empty_cache()
        combined=FixedGPURunner(path)
        verified=all(t['compute_ms_by_branch']['A__' if t['chosen']==0 else 'B__']>0 and t['compute_ms_by_branch']['B__' if t['chosen']==0 else 'A__']<=1e-7 for t in trace)
        report['prototype_supported']=True;report['branch_execution_verified']=verified
        done('branch_execution_trace',kind='branch_trace',verified=verified,traces=trace)
        if not verified:
            report.update(status='complete',timing_skipped_reason='Lazystructureandparitypassed,butsufficientperbranchruntimeprofileevidencenotavailable;nospeedupclaim');atomic_json(REPORT,report);return
        for h in hosts[:3]:
            normalized=buffers.prepare(h);combined.run(normalized)['logits'].argmax(1).item()
            router.run(normalized).cpu().numpy()
            for runner in runners:runner.run(normalized,copy_input=False).argmax(1).item()
        for repeat in range(3):
          modes=('cpu_dispatch','gpu_conditional','both_eager');modes=modes[repeat:]+modes[:repeat]
          for mode in modes:
            check(__file__);pred=[];meter.start();start=time.perf_counter()
            for h in hosts:
                normalized=buffers.prepare(h)
                if mode=='gpu_conditional':logits=combined.run(normalized)['logits']
                else:
                    selected=cpu_choice(router.run(normalized).cpu().numpy()[0,columns])
                    if mode=='both_eager':both=[r.run(normalized,copy_input=False) for r in runners];logits=both[selected]
                    else:logits=runners[selected].run(normalized,copy_input=False)
                pred.append(int(logits.argmax(1).item()))
            torch.cuda.synchronize();elapsed=time.perf_counter()-start;energy=meter.finish(512)
            if energy.get('gpu_energy_j') is None:raise RuntimeError('Conditionaltiming energy unavailable')
            done(f'window_{repeat}_{mode}',kind='window',repeat=repeat,mode=mode,images=512,seconds=elapsed,ms_per_image=1000*elapsed/512,gpu_energy=energy,predictions=pred,
                 note='No profiler attached;GPUconditionalselectedIDneverreadforintermediatedispatch;finalCPUclassonly')
        for p,h in report['protected_hashes'].items():assert digest(p)==h
        report['status']='complete';atomic_json(REPORT,report)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));atomic_json(REPORT,report);raise

if __name__=='__main__':main()
