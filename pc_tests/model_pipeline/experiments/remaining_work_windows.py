"""Measured windows for explicitly authorized finite PC follow-ups."""
import json
from pathlib import Path
import time
import numpy as np
from pc_tests.model_pipeline.experiments import compare_dispatch_residency_v2 as old
from pc_tests.model_pipeline.experiments.remaining_work_design import graph_windows
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json


class CampaignGuard(old.Guard):
    def reading(self):
        row=super().reading()
        for key,query in {
            'power_mw':lambda:self.n.nvmlDeviceGetPowerUsage(self.h),
            'memory_clock_mhz':lambda:self.n.nvmlDeviceGetClockInfo(self.h,self.n.NVML_CLOCK_MEM),
            'gpu_utilization_pct':lambda:self.n.nvmlDeviceGetUtilizationRates(self.h).gpu,
            'memory_utilization_pct':lambda:self.n.nvmlDeviceGetUtilizationRates(self.h).memory,
            'performance_state':lambda:self.n.nvmlDeviceGetPerformanceState(self.h),
            'enforced_power_limit_mw':lambda:self.n.nvmlDeviceGetEnforcedPowerLimit(self.h),
        }.items():
            try: row[key]=int(query())
            except Exception as exc: row[key]=None;row[key+'_unavailable']=str(exc)
        return row


def measure_graph(env):
    import torch
    from pc_tests.model_pipeline.runtime.fixed_gpu_runner import FixedGPURunner
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    out=env['OUT'] if 'OUT' in env else Path(env['REPORT']).parent
    state,guard,stream=env['state'],env['guard'],env['stream']
    stage,inputs=env['stage'],env['inputs']
    router,branches=env['router'],env['branches']
    native,handle,call=env['native'],env['handle'],env['call']
    result,choice,counts,pred=env['output'],env['choice'],env['counts'],env['prediction']
    ref_logits,ref_choices,expected=env['ref_logits'],env['ref_choices'],env['ref_pred']
    prior=env['prior'];host=env['host'].pin_memory()
    plan=dict(windows=graph_windows(),expected_windows=72,requests=128,repeats=6,warmup=100,
        cooldown_min_seconds=15,cooldown_max_seconds=180,cool_start_max_c=55,stop_c=80,
        scope='Batch1 two FP32 models and frozen engineering policy; 3 modes crossed input/output boundaries.',
        timing='Exactly128 requests; wall and CUDA span include host gaps; grossGPU-only energy.',
        policy='No per-request selection/prediction readback in GPUinput/deferred conditional or graph. CPU still submits/final sync.',
        qualification='Graph includes branch-counter instrumentation. Cool-start frequency ramp and short energy intervals qualified.')
    atomic_json(out/'plan.json',plan)
    conditional_row=next(r for r in prior['results'] if r['kind']=='conditional_build')
    conditional_path=Path(conditional_row['engine'])
    assert old.digest(conditional_path)==conditional_row['engine_sha256']
    conditional=FixedGPURunner(conditional_path)
    conditional.context.set_tensor_address(conditional.input,stage.data_ptr())
    assert torch.cuda.mem_get_info()[0]>=1024**3
    for p in (conditional_path,Path(__file__),Path(__file__).with_name('remaining_work_design.py'),out/'plan.json'):
        state['protected_hashes'][str(p)]=old.digest(p)
    state['plan_sha256']=old.digest(out/'plan.json')
    state['reference_sha256']=env['v2']['reference_sha256']
    offsets=env['offset'];ia,ib=env['ia'],env['ib'];bias=np.float32(prior['engineering_bias'])
    predictions=torch.empty(128,device='cuda',dtype=torch.int64)
    def enqueue(r):
        old.checked_enqueue(r.context,stream)
        return r.outputs
    def cpu_choose(q):
        scores=q[0,[ia,ib]].astype(np.float32)-offsets
        return 0 if np.float32(scores[0]-scores[1])>=bias else 1
    def request(mode,residence,egress,i):
        stage.copy_((host if residence=='host' else inputs)[i:i+1],non_blocking=True)
        if mode=='cuda_graph':
            call('graph_launch',handle,stream.cuda_stream)
            value=pred
        elif mode=='gpu_conditional':
            value=enqueue(conditional)['logits'].argmax(1)
        else:
            q=next(iter(enqueue(router).values())).cpu().numpy()
            selected=cpu_choose(q)
            value=next(iter(enqueue(branches[selected]).values())).argmax(1)
        predictions[i:i+1].copy_(value)
        return int(predictions[i].item()) if egress=='per_image_cpu' else None
    guard.phase='independent_parity'
    evidence=[]
    for mode in ('cuda_graph','gpu_conditional'):
        for i in range(128):
            guard.check();stage.copy_(inputs[i:i+1])
            if mode=='cuda_graph':
                result.fill_(float('nan'));choice.fill_(-1);pred.fill_(-1);counts.zero_()
                call('graph_launch',handle,stream.cuda_stream)
                observed=result.cpu().numpy().copy();selected=int(choice.item())
                branch_count=counts.cpu().numpy().tolist()
                wanted=[0,0];wanted[int(ref_choices[i])]=1
                assert branch_count==wanted
            else:
                conditional.outputs['logits'].fill_(float('nan'));conditional.outputs['choice'].fill_(-1)
                values=enqueue(conditional)
                observed=values['logits'].cpu().numpy().copy();selected=int(values['choice'].item())
            assert np.isfinite(observed).all() and np.allclose(observed,ref_logits[i:i+1],atol=1e-3,rtol=1e-4)
            assert selected==int(ref_choices[i]) and int(observed.argmax())==int(expected[i])
            evidence.append(dict(mode=mode,input_index=i,choice=selected,prediction=int(observed.argmax()),
                max_abs_logit_error=float(np.abs(observed-ref_logits[i:i+1]).max())))
    state['independent_parity']=evidence
    # Preserve established selected-only profile evidence; source hash makes
    # this provenance explicit rather than claiming a new profiler timing run.
    profile=env['PIPELINE']/'artifacts/dispatch_residency_v2_20261104/study.json'
    assert len(json.loads(profile.read_text())['branch_execution'])==2
    state['native_if_selected_only_profile_source']=str(profile)
    state['protected_hashes'][str(profile)]=old.digest(profile)
    meter=EnergyCounter();assert meter.available
    state['status']='measuring';state['rows']=[]
    atomic_json(out/'study.json',state)
    for index,spec in enumerate(plan['windows']):
        guard.check();guard.phase='warmup'
        for i in range(100):
            request(spec['mode'],spec['residence'],spec['egress'],i)
            if i%32==0: guard.check()
        stream.synchronize()
        state['current']=dict(index=index,**spec);state['phase']='cooldown'
        atomic_json(out/'study.json',state)
        cooling=guard.cool()
        predictions.fill_(-1);counts.zero_();stream.synchronize()
        phase=f'window:{index}';guard.phase=phase
        start_reading=guard.reading()
        with guard.lock: sample_start=len(guard.samples)
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        state['phase']='window';atomic_json(out/'study.json',state)
        meter.start();started=time.perf_counter();begin.record(stream)
        cpu_predictions=[]
        for i in range(128):
            value=request(spec['mode'],spec['residence'],spec['egress'],i)
            if value is not None: cpu_predictions.append(value)
            if i%32==0: guard.check()
        end.record(stream);stream.synchronize()
        seconds=time.perf_counter()-started;energy=meter.finish(128)
        end_reading=guard.reading();guard.phase='verification'
        observed=predictions.cpu().numpy().copy()
        mismatch=int(np.count_nonzero(observed!=expected))
        if cpu_predictions: mismatch+=int(np.count_nonzero(np.asarray(cpu_predictions)!=expected))
        observed_counts=counts.cpu().numpy().tolist() if spec['mode']=='cuda_graph' else None
        if observed_counts is not None: assert observed_counts==np.bincount(ref_choices,minlength=2).tolist()
        file=out/f'window_{index:02d}_predictions.npy';np.save(file,observed,allow_pickle=False)
        with guard.lock: samples=[r for r in guard.samples[sample_start:] if r['phase']==phase]
        readings=[start_reading,*samples,end_reading]
        thermal=sum(bool(r['clock_reasons']&32) for r in readings)
        powercap=sum(bool(r['clock_reasons']&4) for r in readings)
        positive=energy.get('gpu_energy_j') is not None and np.isfinite(energy['gpu_energy_j']) and energy['gpu_energy_j']>0
        row=dict(index=index,**spec,requests=128,seconds=seconds,ms_per_image=1000*seconds/128,
            stream_span_ms=begin.elapsed_time(end),gpu_energy=energy,cooldown=cooling,start_reading=start_reading,end_reading=end_reading,
            timed_telemetry_samples=len(samples),software_thermal_samples=thermal,software_powercap_samples=powercap,
            prediction_mismatches=mismatch,graph_branch_counts=observed_counts,predictions_file=str(file),
            predictions_file_sha256=old.digest(file),ended_utc=old.utc(),accepted=bool(mismatch==0 and thermal==0 and positive and samples))
        state['rows'].append(row);atomic_json(out/'study.json',state)
        print(json.dumps(dict(index=index,accepted=row['accepted'],mode=spec['mode'],ms_per_image=row['ms_per_image'])),flush=True)
        if not row['accepted']: raise RuntimeError('Invalid timed window; preserved, no retry')
        guard.check()
