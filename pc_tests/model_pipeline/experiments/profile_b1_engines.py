"""Validation-only paired old-vs-batch1 engine profiles for a fixed small pool."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

import gc
import json
import os
from pathlib import Path
import time
from datetime import datetime,timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation,digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json,costs_for
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation

OUTPUT=PIPELINE/'reports/b1_engine_profiles_20261028'
ARTIFACTS=PIPELINE/'artifacts/b1_engine_profiles_20261028'
STATE=PIPELINE/'artifacts/pc_routers/thermal_guard.json'


def choose_pool(records,correct,positions,baseline,limit=6):
    """Fit-only screen: reference, cheapest eligible, then complementary rescues."""
    names=[r['name'] for r in records]
    y=correct[positions].astype(bool)
    energy=costs_for(records,1,'energy')
    latency=costs_for(records,1,'latency')
    base=names.index(baseline)
    minimum=y[:,base].mean()-.01
    selected=[base]
    eligible=np.flatnonzero(y.mean(0)+1e-10>=minimum)
    for costs in (energy,latency):
        index=min(eligible,key=lambda i:(costs[i],names[i]))
        if index not in selected:
            selected.append(int(index))
    while len(selected)<min(limit,len(names)):
        covered=y[:,selected].any(1)
        remaining=[i for i in range(len(names)) if i not in selected]
        # Accuracy coverage is the first key; measured cost only resolves ties.
        best=min(remaining,key=lambda i:(-int((y[:,i]&~covered).sum()),energy[i],names[i]))
        selected.append(best)
    return [records[i] for i in selected]


def thermal():
    for attempt in range(20):
        try:
            state=json.loads(STATE.read_text(encoding='utf-8-sig'));break
        except (OSError,json.JSONDecodeError):
            if attempt==19:
                raise RuntimeError('Thermal state unreadable')
            time.sleep(.025)
    if state.get('status')!='monitoring' or state.get('boundary_start_count') is not None:
        raise RuntimeError('Thermal latch: never restart')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(state['utc'].replace('Z','+00:00'))).total_seconds()
    if not 0<=age<=130 or state.get('expected_script')!=Path(__file__).name or os.getpid() not in state.get('tracked_process_ids',[]):
        raise RuntimeError('Fresh exact profile-worker guard required')


def build_fixed(graph,destination,precision):
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import trt,TRT_LOGGER
    builder=trt.Builder(TRT_LOGGER)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,TRT_LOGGER)
    if not parser.parse(Path(graph).read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512*1024**2)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    if precision in ('fp16','int8'):
        config.set_flag(trt.BuilderFlag.FP16)
    if precision=='int8':
        config.set_flag(trt.BuilderFlag.INT8)
    profile=builder.create_optimization_profile()
    shape=(1,3,128,128)
    profile.set_shape(network.get_input(0).name,shape,shape,shape)
    config.add_optimization_profile(profile)
    value=builder.build_serialized_network(network,config)
    if value is None:
        raise RuntimeError('Fixed batch1 build failed')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.engine.tmp')
    temporary.write_bytes(bytes(value));temporary.replace(destination)


def main():
    path=OUTPUT/'study.json'
    if path.exists():
        raise RuntimeError('Existing profiles preserved; no blind restart')
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
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner,trt
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    from pc_tests.model_pipeline.runtime.engine_precision_audit import integer_compute_layers
    torch.set_num_threads(1)
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,correct,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
    fit,calibration=split_validation(labels,42)
    pool=choose_pool(records,correct,fit,baseline)
    images=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    positions=calibration[:100]
    host=[torch.from_numpy(np.array(images[i:i+1])).pin_memory() for i in positions]
    buffers=SharedBatchInputs(maximum_batch=1)
    manifest_path=PIPELINE/'artifacts/quantization/deployment_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    sources={(r['architecture'],r['prune']):r for r in manifest['models']}
    report=dict(status='running',results=[],pool=[r['name'] for r in pool],fit_positions=fit.tolist(),
        calibration_probe_positions=positions.tolist(),pool_selection='Frozen seed42 fit-only coverage/cost screen, no test data',
        provenance=provenance,source_sha256=digest(Path(__file__)),manifest_sha256=digest(manifest_path),
        static_report_sha256=digest(PIPELINE/'reports/gpu_static.json'),test_data_opened=False,
        expected_models=len(pool),gpu=torch.cuda.get_device_name(0),tensorrt_version=trt.__version__,
        timing_scope='Pinned RGB128 -> transfer/normalization -> classifier -> host argmax; no router; gross GPU energy only. Five paired alternating 2-second windows on fixed validation inputs.',
        note='No full-test accuracy claim. Dedicated engine numerical/argmax changes are reported and require refreshed validation correctness before policy calibration. Build/setup/warm execution are separate.',
        created_utc=datetime.now(timezone.utc).isoformat())
    OUTPUT.mkdir(parents=True,exist_ok=True)
    def save():
        atomic_json(path,report)
    def done(identity,**values):
        report['results'].append(dict(id=identity,router='b1_profile',status='complete',**values));save();thermal()
    save()
    def release():
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    try:
        for record in pool:
            thermal()
            name=record['name']
            source=sources[(record['architecture'],record['prune'])]
            original_graph=Path(source['onnx']) if record['precision']!='int8' else PIPELINE/f'artifacts/pc_precision/converted_models/{name}.onnx'
            if digest(original_graph)!=record['source_sha256']:
                raise RuntimeError('Original graph hash mismatch')
            graph=original_graph if record['precision']!='int8' else PIPELINE/f'artifacts/gpu_static/prepared_graphs/{name}.onnx'
            target=ARTIFACTS/f'engines/{name}.engine'
            print('BUILD batch1 '+name,flush=True)
            started=time.perf_counter();build_fixed(graph,target,record['precision'])
            done('build_'+name,kind='build',name=name,build_seconds=time.perf_counter()-started,
                 graph_sha256=digest(graph),source_graph_sha256=digest(original_graph),engine_path=str(target),engine_sha256=digest(target),profile=[1,1,1])
            logits_by_variant={};measurements={'original':[],'batch1':[]}
            load_profiles={'original':[],'batch1':[]}
            for repeat in range(5):
                for variant in (('original','batch1') if repeat%2==0 else ('batch1','original')):
                    thermal();release()
                    before,total=torch.cuda.mem_get_info()
                    engine_path=Path(record['engine_path']) if variant=='original' else target
                    if variant=='original' and digest(engine_path)!=record['engine_sha256']:
                        raise RuntimeError('Original engine changed')
                    meter=EnergyCounter()
                    if not meter.available:
                        raise RuntimeError('NVML energy unavailable')
                    meter.start();loaded=time.perf_counter()
                    engine=EngineRunner(engine_path,maximum_batch=1)
                    torch.cuda.synchronize()
                    deserialize_seconds=time.perf_counter()-loaded
                    load_energy=meter.finish(1)
                    with torch.inference_mode():
                        outputs=[engine.run(buffers.prepare(h),copy_input=False).cpu().numpy().copy() for h in host]
                    torch.cuda.synchronize()
                    if any(x.shape!=(1,100) or not np.isfinite(x).all() for x in outputs):
                        raise RuntimeError('Nonfinite/invalid real validation logits')
                    free,total=torch.cuda.mem_get_info()
                    if free<1024**3:
                        raise RuntimeError('Single engine violates 1GiB reserve')
                    load_profiles[variant].append(dict(repeat=repeat,observed_resident_bytes=max(0,before-free),
                        deserialize_context_seconds=deserialize_seconds,setup_gpu_energy=load_energy,
                        load_plus_100_probe_seconds=time.perf_counter()-loaded))
                    if repeat==0:
                        logits_by_variant[variant]=np.concatenate(outputs)
                    if record['precision']=='int8' and repeat==0:
                        inspector=engine.engine.create_engine_inspector();inspector.execution_context=engine.context
                        integer=integer_compute_layers(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
                        if not integer:
                            raise RuntimeError('INT8 build lost all integer compute paths')
                        del inspector
                    with torch.inference_mode():
                        for h in host[:20]:
                            engine.run(buffers.prepare(h),copy_input=False).argmax(1).cpu()
                        torch.cuda.synchronize();meter.start();started=time.perf_counter();iterations=0
                        while time.perf_counter()-started<2.:
                            h=host[iterations%len(host)]
                            engine.run(buffers.prepare(h),copy_input=False).argmax(1).cpu()
                            iterations+=1
                        torch.cuda.synchronize()
                        elapsed=time.perf_counter()-started
                        energy=meter.finish(iterations)
                    if energy.get('gpu_energy_j_per_image') is None or energy['gpu_energy_j_per_image']<=0:
                        raise RuntimeError('Invalid warm energy')
                    measurements[variant].append(dict(repeat=repeat,images=iterations,ms_per_image=elapsed*1000/iterations,**energy))
                    del engine;release()
            a,b=logits_by_variant['original'],logits_by_variant['batch1']
            done('paired_'+name,kind='paired_profile',name=name,measurements=measurements,load_profiles=load_profiles,
                logits_max_abs_difference=float(np.max(np.abs(a-b))),
                argmax_disagreements_on_100=int((a.argmax(1)!=b.argmax(1)).sum()),
                original_profile=[1,32,128],new_profile=[1,1,1],engine_sha256=digest(target))
            print('PAIRED COMPLETE '+name,flush=True)
        # Profile the SAME entire candidate pool resident, for each build type.
        for variant in ('original','batch1'):
            thermal();release();initial_free,total=torch.cuda.mem_get_info();resident=[]
            try:
                for record in pool:
                    thermal()
                    engine_path=Path(record['engine_path']) if variant=='original' else ARTIFACTS/f"engines/{record['name']}.engine"
                    if torch.cuda.mem_get_info()[0]-engine_path.stat().st_size-128*1024**2<1024**3:
                        raise RuntimeError('Pool admission violates reserve; do not force OOM')
                    resident.append(EngineRunner(engine_path,maximum_batch=1))
                    with torch.inference_mode():
                        resident[-1].run(buffers.prepare(host[0]),copy_input=False).cpu()
                    torch.cuda.synchronize()
                    if torch.cuda.mem_get_info()[0]<1024**3:
                        raise RuntimeError('Actual pool memory violates reserve')
                final_free,total=torch.cuda.mem_get_info()
                done('pool_'+variant,kind='pool_memory',variant=variant,count=len(resident),
                    observed_increment_bytes=initial_free-final_free,free_bytes=final_free,used_bytes=total-final_free)
            finally:
                resident.clear();release()
        if digest(PIPELINE/'reports/gpu_static.json')!=report['static_report_sha256']:
            raise RuntimeError('Protected static report changed')
        report['status']='complete';save()
        print('COMPLETE paired engine profiles; no held-out test evaluation',flush=True)
    except Exception as error:
        report['status']='failed';report['error']=repr(error);save();raise


if __name__=='__main__':
    main()
