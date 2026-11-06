"""Fixed-model batch/transfer follow-up; validation inputs, no training/rebuild."""
import sys
from pathlib import Path
if not __package__: sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
import os
import subprocess
import time
import gc
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
from pc_tests.model_pipeline.experiments import compare_dispatch_residency_v2 as common
from pc_tests.model_pipeline.experiments.remaining_work_windows import CampaignGuard
from pc_tests.model_pipeline.experiments.remaining_work_design import plateau_windows

OUT=PIPELINE/'artifacts/remaining_plateau_20261104'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('x') as f: json.dump(dict(pid=os.getpid(),utc=common.utc()),f)
    state=dict(status='preparing',rows=[],test_data_accessed=False,pid=os.getpid())
    guard=None
    try:
        raw=subprocess.check_output(['powershell.exe','-NoProfile','-Command',
            f'Get-CimInstance Win32_Process -Filter "ProcessId = {os.getpid()}" | Select-Object ProcessId,ExecutablePath,CommandLine,@{{n="creation_unix_ms";e={{[DateTimeOffset]::new($_.CreationDate).ToUnixTimeMilliseconds()}}}} | ConvertTo-Json'],text=True)
        atomic_json(OUT/'worker_identity.json',json.loads(raw))
        common.OUT=OUT;guard=CampaignGuard();state['start_telemetry']=guard.start()
        import torch
        from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
        from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
        torch.set_num_threads(1)
        plan=dict(windows=plateau_windows(),expected_windows=36,repeats=3,images_per_cycle=128,
            measured_minimum_seconds=1,maximum_seconds=10,warmup_minimum_seconds=1,max_cycles=1000,
            cooldown_minimum_seconds=15,cool_start_max_c=55,stop_c=80,
            models='Historical unpruned FP16 ResNet101 and EfficientNetB0, no engine rebuild.',
            boundaries='GPUresident normalized input to device logits/predictions vs pinned normalized host input to host logits; normalization excluded.',
            qualifications='Same128 validation inputs repeated; no routed mix/cache/model changes. Original opt32 dynamic profiles retained. No hardware changes; clocks/thermal observed, not manipulated. Short windows, grossGPU-only energy.')
        atomic_json(OUT/'plan.json',plan)
        prior_path=PIPELINE/'artifacts/dispatch_residency_v2_20261104/study.json'
        prior=json.loads(prior_path.read_text())
        cache=PIPELINE/'artifacts/pc_precision/data_cache/validation_rgb128.npy'
        static=PIPELINE/'reports/gpu_static.json'
        records=json.loads(static.read_text())['results']
        selected={name:next(r for r in records if r['name']==name) for name in {w['name'] for w in plan['windows']}}
        normalized=torch.from_numpy(np.array(np.load(cache,mmap_mode='r',allow_pickle=False)[prior['input_positions']],copy=True)).float()
        normalized.div_(255).sub_(torch.tensor([.5071,.4867,.4408])[None,:,None,None])
        normalized.div_(torch.tensor([.2675,.2565,.2761])[None,:,None,None])
        assert common.array_hash(normalized.numpy())==prior['input_sha256']
        host=normalized.pin_memory();device=host.to('cuda')
        protected=[Path(__file__),Path(__file__).with_name('remaining_work_design.py'),Path(__file__).with_name('remaining_work_windows.py'),
            OUT/'plan.json',prior_path,cache,static,PIPELINE/'runtime/gpu_engine_preflight.py',PIPELINE/'runtime/gpu_energy.py']
        for r in selected.values():
            p=Path(r['engine_path']);assert p.drive.upper()=='C:' and common.digest(p)==r['engine_sha256'];protected.append(p)
        state['protected_hashes']={str(p):common.digest(p) for p in protected}
        state['plan_sha256']=common.digest(OUT/'plan.json')
        state['input_sha256']=prior['input_sha256'];state['input_positions']=prior['input_positions']
        references={};state['reference_files']={};state['engine_profiles']={}
        # One model resident at a time: do not alter historical cache tests.
        runner=None;loaded=None
        def get_runner(name):
            nonlocal runner,loaded
            if loaded!=name:
                if runner is not None:
                    torch.cuda.synchronize();runner=None;gc.collect();torch.cuda.empty_cache()
                runner=EngineRunner(Path(selected[name]['engine_path']))
                loaded=name
                state['engine_profiles'][name]=[list(s) for s in runner.engine.get_tensor_profile_shape(runner.input_name,0)]
                assert torch.cuda.mem_get_info()[0]>=1024**3
            return runner
        meter=EnergyCounter();assert meter.available
        with torch.inference_mode():
            for index,spec in enumerate(plan['windows']):
                guard.check();r=get_runner(spec['name']);batch=spec['batch']
                key=(spec['name'],batch)
                if key not in references:
                    guard.phase='independent_reference'
                    parts=[]
                    for i in range(0,128,batch):
                        parts.append(r.run(device[i:i+batch],copy_input=False).cpu().numpy().copy())
                    reference=np.concatenate(parts);assert reference.shape==(128,100) and np.isfinite(reference).all()
                    file=OUT/f'{spec["name"]}_b{batch}_reference_logits.npy'
                    np.save(file,reference,allow_pickle=False)
                    references[key]=reference
                    state['reference_files'][f'{key[0]}_b{batch}']=dict(path=str(file),sha256=common.digest(file))
                expected=references[key].argmax(1)
                predictions=torch.empty((plan['max_cycles'],128),device='cuda',dtype=torch.int64)
                def cycle(n,copy_host):
                    for i in range(0,128,batch):
                        if copy_host:
                            r.input[:batch].copy_(host[i:i+batch],non_blocking=True)
                            value=r._enqueue(r.input[:batch],copy_input=False)
                        else: value=r._enqueue(device[i:i+batch],copy_input=False)
                        predictions[n,i:i+batch].copy_(value.argmax(1))
                        if copy_host: value.cpu().numpy().copy()
                state['current']=dict(index=index,**spec);state['phase']='cooldown';atomic_json(OUT/'study.json',state)
                cooling=guard.cool()
                # Unlike the earlier routed diagnostic, steady preconditioning
                # occurs AFTER cooldown, immediately before a marked timer.
                guard.phase='warmup';started=time.perf_counter();warm_cycles=0
                while time.perf_counter()-started<plan['warmup_minimum_seconds']:
                    cycle(0,spec['boundary']=='host_to_logits');torch.cuda.synchronize();warm_cycles+=1;guard.check()
                predictions.fill_(-1);torch.cuda.synchronize()
                phase=f'window:{index}';guard.phase=phase;start_reading=guard.reading()
                with guard.lock: sample_start=len(guard.samples)
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                state['phase']='window';atomic_json(OUT/'study.json',state)
                meter.start();started=time.perf_counter();begin.record();cycles=0
                while True:
                    cycle(cycles,spec['boundary']=='host_to_logits');cycles+=1
                    torch.cuda.synchronize();guard.check()
                    elapsed=time.perf_counter()-started
                    if elapsed>=plan['measured_minimum_seconds']: break
                    if cycles>=plan['max_cycles'] or elapsed>plan['maximum_seconds']: raise RuntimeError('Bound exceeded')
                end.record();torch.cuda.synchronize();seconds=time.perf_counter()-started
                energy=meter.finish(cycles*128);end_reading=guard.reading();guard.phase='verification'
                observed=predictions[:cycles].cpu().numpy().copy()
                mismatches=int(np.count_nonzero(observed!=expected[None,:]))
                file=OUT/f'window_{index:02d}_predictions.npy';np.save(file,observed,allow_pickle=False)
                with guard.lock: samples=[t for t in guard.samples[sample_start:] if t['phase']==phase]
                readings=[start_reading,*samples,end_reading]
                thermal=sum(bool(t['clock_reasons']&32) for t in readings)
                powercap=sum(bool(t['clock_reasons']&4) for t in readings)
                positive=energy.get('gpu_energy_j') is not None and np.isfinite(energy['gpu_energy_j']) and energy['gpu_energy_j']>0
                row=dict(index=index,**spec,cycles=cycles,requests=cycles*128,seconds=seconds,ms_per_image=seconds*1000/(cycles*128),
                    stream_span_ms=begin.elapsed_time(end),gpu_energy=energy,cooldown=cooling,warm_cycles=warm_cycles,
                    start_reading=start_reading,end_reading=end_reading,timed_telemetry_samples=len(samples),
                    software_thermal_samples=thermal,software_powercap_samples=powercap,prediction_mismatches=mismatches,
                    predictions_file=str(file),predictions_file_sha256=common.digest(file),ended_utc=common.utc(),
                    accepted=bool(mismatches==0 and thermal==0 and positive and samples and seconds<=plan['maximum_seconds']))
                state['rows'].append(row);state['status']='measuring';atomic_json(OUT/'study.json',state)
                print(json.dumps(dict(index=index,accepted=row['accepted'],ms_per_image=row['ms_per_image'])),flush=True)
                if not row['accepted']: raise RuntimeError('Invalid timed plateau window; preserve and stop, no retry')
        for p,h in state['protected_hashes'].items(): assert common.digest(p)==h,p
        state.update(status='complete_pending_audit',finished_utc=common.utc());atomic_json(OUT/'study.json',state)
    except BaseException as exc:
        state.update(status='stopped_error',error=repr(exc),finished_utc=common.utc());atomic_json(OUT/'study.json',state);raise
    finally:
        if guard: guard.close()


if __name__=='__main__': main()
