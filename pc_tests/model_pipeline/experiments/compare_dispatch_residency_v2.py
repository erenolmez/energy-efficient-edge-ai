"""Fresh corrected CPU/native-If repeat. Original study remains immutable."""
import sys
from pathlib import Path
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import hashlib
import itertools
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

OUT = PIPELINE/'artifacts/dispatch_residency_v2_20261104'
REPORT = OUT/'study.json'
PRIOR = PIPELINE/'reports/gpu_conditional_20261029/study.json'


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def checked_enqueue(context, stream):
    """A false return is failure, including inside a capture, never success."""
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError('TensorRT execute_async_v3 returned False')


def variant_order():
    """48 windows: each input/output stratum has exactly3 AB and3 BA pairs."""
    strata = list(itertools.product(('host','gpu'),('per_image_cpu','deferred_gpu')))
    rows = []
    for repeat in range(6):
        order = strata[repeat % 4:] + strata[:repeat % 4]
        if repeat % 2:
            order = list(reversed(order))
        for residence, egress in order:
            modes = ('cpu_dispatch','gpu_conditional') if repeat % 2 == 0 else ('gpu_conditional','cpu_dispatch')
            rows.extend(dict(repeat=repeat,mode=m,residence=residence,egress=egress) for m in modes)
    return rows


class Guard:
    def __init__(self):
        import pynvml as nv
        nv.nvmlInit()
        self.n, self.h = nv, nv.nvmlDeviceGetHandleByIndex(0)
        self.phase = 'setup'
        self.latch = None
        self.last = None
        self.last_mono = 0
        self.samples = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def reading(self):
        n,h = self.n,self.h
        return dict(utc=utc(),phase=self.phase,
            temperature_c=int(n.nvmlDeviceGetTemperature(h,n.NVML_TEMPERATURE_GPU)),
            clock_reasons=int(n.nvmlDeviceGetCurrentClocksThrottleReasons(h)),
            graphics_clock_mhz=int(n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_GRAPHICS)))

    def start(self):
        row=self.reading()
        if row['temperature_c']>70 or row['clock_reasons'] & 32:
            raise RuntimeError('Initial gate not met; no launch/retry of GPU timing')
        self.last,self.last_mono = row,time.monotonic()
        self.thread.start()
        return row

    def loop(self):
        with (OUT/'telemetry.jsonl').open('x',buffering=1) as f:
            while not self.stop.is_set():
                try:
                    row=self.reading()
                    if row['temperature_c']>=80:
                        self.latch='80C conservative safety latch'
                    with self.lock:
                        self.last,self.last_mono=row,time.monotonic()
                        self.samples.append(row)
                except Exception as exc:
                    self.latch='Missing telemetry: '+type(exc).__name__
                    row=dict(utc=utc(),phase=self.phase,error=self.latch)
                f.write(json.dumps(row)+'\n')
                self.stop.wait(.1)

    def check(self):
        if self.latch or time.monotonic()-self.last_mono>2:
            raise RuntimeError(self.latch or 'Telemetry stale')
        if json.loads((PIPELINE/'reports/experiment_pause.json').read_text())['status']=='paused_by_user':
            raise RuntimeError('User pause')

    def cool(self):
        self.phase='cooldown'
        started=time.monotonic()
        while True:
            self.check()
            row=self.reading()
            elapsed=time.monotonic()-started
            if elapsed>=15 and row['temperature_c']<=55 and not row['clock_reasons']&32:
                return dict(seconds=elapsed,reading=row)
            if elapsed>180:
                raise RuntimeError('Cooldown <=55C/no thermal flag not reached in180s; deferred, no retry')
            self.stop.wait(1)

    def close(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3)


def main():
    if REPORT.exists():
        raise RuntimeError('Existing corrected study preserved; no restart')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('x') as f:
        json.dump(dict(pid=os.getpid(),started_utc=utc()),f)
    plan=dict(batch=1,windows=variant_order(),expected_windows=48,requests_per_window=128,
        warmup_requests=100,cooldown_min_seconds=15,cooldown_max_seconds=180,start_temperature_c_max=55,
        telemetry_period_seconds=.1,conservative_stop_c=80,
        input_policy='Exactly original128 cached validation inputs, same engineering policy and engines. Host normalizedCPU pinned H2D vs GPU normalizedresident D2D staging. Normalization excluded equally.',
        output_policy='Per-imageCPU .item vs deferredGPU128-prediction tensor. Full timed prediction vector copied AFTER timer and compared elementwise to independent reference in every window.',
        gpu_only='Native If computes router/decision/selectedclassifier/prediction onGPU with no per-request decision/predictionreadback for GPUinput/deferredoutput. CPU still submits and endboundary sync remains.',
        graph_policy='Unsupported for this conditional TensorRTengine, per documented limitation and original runtime error. NOT timed, NOT relabelled as eager, NO speedup from invalid original graph rows.',
        thermal_policy='Warm100 then cooldown>=15s and<=55C/no softwarethermalflag before EVERYwindow. Retain and stop on any timed softwarethermalflag; do not repeat affected window automatically. Powercap flag recorded/qualified separately.',
        timing='Fixed one128request cycle; no time-target adaptivity. One final sync except intentionalCPUdispatch/per-imageCPUoutput. Guardcheck every32requests common instrumentation included.',
        energy='Gross NVML GPUenergy includingidle/display, excludingCPU/restPC; shortwindow duration andcounterresolution disclosed.',
        scope='Two FP32 classifiers, batch1, medianbiased engineeringpolicy; notaccuracy-budget policy, no testdata ortraining. No static/largerbatch/213resident claims.')
    atomic_json(OUT/'plan.json',plan)
    state=dict(status='preparing',started_utc=utc(),pid=os.getpid(),rows=[],plan=plan,
        test_data_accessed=False,graph_status='unsupported_not_timed',expected_windows=48)
    atomic_json(REPORT,state)
    guard=None
    try:
        identity=subprocess.check_output(['powershell.exe','-NoProfile','-Command',
            f'Get-CimInstance Win32_Process -Filter "ProcessId = {os.getpid()}" | Select-Object ProcessId,ExecutablePath,CommandLine,@{{n="creation_unix_ms";e={{[DateTimeOffset]::new($_.CreationDate).ToUnixTimeMilliseconds()}}}} | ConvertTo-Json'],text=True)
        atomic_json(OUT/'worker_identity.json',json.loads(identity))
        guard=Guard()
        state['start_telemetry']=guard.start()
        import torch
        from pc_tests.model_pipeline.runtime.fixed_gpu_runner import FixedGPURunner
        from pc_tests.model_pipeline.runtime.gpu_engine_preflight import trt
        from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
        from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation
        torch.set_num_threads(1)
        prior=json.loads(PRIOR.read_text())
        assert prior['branch_execution_verified']
        branches=[r for r in prior['results'] if r['kind']=='branch_build']
        combined_row=next(r for r in prior['results'] if r['kind']=='conditional_build')
        paths=[Path(r['engine']) for r in branches]+[Path(combined_row['engine'])]
        for r,p in zip(branches+[combined_row],paths):
            assert digest(p)==r['engine_sha256']
        router_path=PIPELINE/'artifacts/correctness_b1_20261028/engines/correctness_seed42_b1.engine'
        assert digest(router_path)==prior['router_engine_sha256']
        cache=PIPELINE/'artifacts/pc_precision/data_cache'
        records,_,_,_,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
        columns=[[r['name'] for r in records].index(n) for n in prior['pair']]
        data_path=cache/'validation_rgb128.npy'
        positions=prior['calibration_positions'][:128]
        rgb=np.load(data_path,mmap_mode='r',allow_pickle=False)
        normalized=torch.from_numpy(np.array(rgb[positions],copy=True)).float()
        normalized.div_(255).sub_(torch.tensor([.5071,.4867,.4408])[None,:,None,None])
        normalized.div_(torch.tensor([.2675,.2565,.2761])[None,:,None,None])
        host=normalized.pin_memory()
        device=host.to('cuda')
        stage=torch.empty((1,3,128,128),device='cuda')
        stream=torch.cuda.Stream()
        router=FixedGPURunner(router_path)
        classifiers=[FixedGPURunner(p) for p in paths[:2]]
        combined=FixedGPURunner(paths[2])
        for r in [router,*classifiers,combined]:
            assert r.shape==(1,3,128,128)
            r.context.set_tensor_address(r.input,stage.data_ptr())
        def enqueue(r):
            checked_enqueue(r.context,stream)
            return r.outputs
        def logits(r):
            return next(iter(enqueue(r).values()))
        costs=np.array(prior['costs'],dtype=np.float32)
        offset=(float(prior['budget_policy']['penalty'])*costs/costs.max()).astype(np.float32)
        bias=np.float32(prior['engineering_bias'])
        def choose(q):
            score=q[0,columns].astype(np.float32)-offset
            return 0 if np.float32(score[0]-score[1])>=bias else 1
        state.update(provenance=provenance,input_positions=positions,input_sha256=array_hash(normalized.numpy()),
                     torch_version=torch.__version__,tensorrt_version=trt.__version__,gpu=torch.cuda.get_device_name())
        protected=[Path(__file__),PRIOR,*paths,router_path,data_path,PIPELINE/'runtime/fixed_gpu_runner.py',PIPELINE/'runtime/gpu_engine_preflight.py',OUT/'plan.json',
            PIPELINE/'reports/dispatch_residency_20261104/audited_summary.json',PIPELINE/'artifacts/dispatch_residency_20261104/study.json']
        state['protected_hashes']={str(p):digest(p) for p in protected}
        state['free_bytes_after_setup']=torch.cuda.mem_get_info()[0]
        assert state['free_bytes_after_setup']>=1024**3
        guard.phase='independent_reference'
        expected_logits=[]
        expected_choices=[]
        with torch.inference_mode(),torch.cuda.stream(stream):
            for i in range(128):
                guard.check()
                stage.copy_(device[i:i+1])
                selected=choose(logits(router).cpu().numpy().copy())
                expected_choices.append(selected)
                expected_logits.append(logits(classifiers[selected]).cpu().numpy().copy())
        reference=np.concatenate(expected_logits)
        expected=np.argmax(reference,axis=1).astype(np.int64)
        assert set(expected_choices)=={0,1}
        np.save(OUT/'reference_logits.npy',reference,allow_pickle=False)
        np.save(OUT/'reference_predictions.npy',expected,allow_pickle=False)
        np.save(OUT/'reference_choices.npy',np.array(expected_choices,dtype=np.int32),allow_pickle=False)
        state['reference_sha256']={p.name:digest(p) for p in OUT.glob('reference_*.npy')}
        state['reference_prediction_sha256']=array_hash(expected)
        # No eager combined execution immediately before a check. Poison both
        # outputs first; then use independent saved classifier/reference values.
        guard.phase='independent_conditional_parity'
        maxerr=0
        with torch.inference_mode(),torch.cuda.stream(stream):
            for i in range(128):
                stage.copy_(device[i:i+1])
                combined.outputs['logits'].fill_(float('nan'))
                combined.outputs['choice'].fill_(-1)
                result=enqueue(combined)
                observed=result['logits'].cpu().numpy().copy()
                assert np.isfinite(observed).all(), 'Stale/unwritten conditional output'
                assert int(result['choice'].item())==expected_choices[i], 'Conditional choice mismatch'
                assert np.allclose(reference[i:i+1],observed,atol=1e-3,rtol=1e-4), 'Independent logit mismatch'
                assert int(observed.argmax())==int(expected[i]), 'Independent class mismatch'
                maxerr=max(maxerr,float(np.abs(reference[i:i+1]-observed).max()))
        state['parity']=dict(images=128,choice_mismatches=0,class_mismatches=0,max_abs_logit_error=maxerr,
                            both_branches=np.bincount(expected_choices).tolist(),poisoned_outputs=True,independent_reference=True)
        # Selected-only runtime evidence rechecked in this new session.
        class Profile(trt.IProfiler):
            def __init__(self):
                trt.IProfiler.__init__(self)
                self.times=[]
            def report_layer_time(self,name,ms):
                self.times.append(dict(name=name,ms=float(ms)))
        profile=Profile()
        combined.context.profiler=profile
        combined.context.enqueue_emits_profile=True
        traces=[]
        with torch.inference_mode(),torch.cuda.stream(stream):
            for selected in (0,1):
                i=expected_choices.index(selected)
                stage.copy_(device[i:i+1])
                profile.times=[]
                enqueue(combined)
                stream.synchronize()
                if not profile.times:
                    combined.context.report_to_profiler()
                masses={p:sum(r['ms'] for r in profile.times if p in r['name'] and any(k in r['name'].lower() for k in ('conv','gemm','matmul'))) for p in ('A__','B__')}
                assert masses['A__' if selected==0 else 'B__']>0 and masses['B__' if selected==0 else 'A__']<=1e-7
                traces.append(dict(choice=selected,compute_ms=masses,layers=list(profile.times)))
        state['branch_execution']=traces
        combined.context=combined.engine.create_execution_context()
        combined.context.set_tensor_address(combined.input,stage.data_ptr())
        for name,value in combined.outputs.items():
            combined.context.set_tensor_address(name,value.data_ptr())
        state['status']='measuring'
        atomic_json(REPORT,state)
        predictions=torch.empty(128,dtype=torch.int64,device='cuda')
        def request(mode,residence,egress,i):
            stage.copy_((host if residence=='host' else device)[i:i+1],non_blocking=True)
            if mode=='cpu_dispatch':
                out=logits(classifiers[choose(logits(router).cpu().numpy())])
            else:
                out=enqueue(combined)['logits']
            prediction=out.argmax(1)
            predictions[i:i+1].copy_(prediction)
            if egress=='per_image_cpu':
                return int(prediction.item())
            return None
        meter=EnergyCounter()
        assert meter.available
        with torch.inference_mode(),torch.cuda.stream(stream):
            for index,spec in enumerate(plan['windows']):
                guard.check()
                guard.phase='warmup'
                for i in range(100):
                    request(spec['mode'],spec['residence'],spec['egress'],i)
                    if i%32==0:
                        guard.check()
                stream.synchronize()
                state['current']=dict(index=index,**spec)
                state['phase']='cooldown'
                atomic_json(REPORT,state)
                cooling=guard.cool()
                # Poison the prediction buffer before EACH timed run. Result
                # copies are outside measurement; same writes in all variants.
                predictions.fill_(-1)
                stream.synchronize()
                phase=f"window:{index}"
                guard.phase=phase
                start_reading=guard.reading()
                with guard.lock:
                    sample_start=len(guard.samples)
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                state['phase']='window'
                atomic_json(REPORT,state)
                meter.start()
                start=time.perf_counter()
                begin.record(stream)
                cpu_predictions=[]
                for i in range(128):
                    value=request(spec['mode'],spec['residence'],spec['egress'],i)
                    if value is not None:
                        cpu_predictions.append(value)
                    if i%32==0:
                        guard.check()
                end.record(stream)
                stream.synchronize()
                seconds=time.perf_counter()-start
                energy=meter.finish(128)
                end_reading=guard.reading()
                guard.phase='output_verification'
                # Full elementwise verification of EVERY timed prediction,
                # not a checksum and not same-path eager-before-replay parity.
                observed=predictions.cpu().numpy().copy()
                mismatches=int(np.count_nonzero(observed!=expected))
                if cpu_predictions:
                    mismatches+=int(np.count_nonzero(np.array(cpu_predictions)!=expected))
                with guard.lock:
                    samples=[x for x in guard.samples[sample_start:] if x['phase']==phase]
                thermal=sum(bool(x['clock_reasons']&32) for x in [start_reading,*samples,end_reading])
                powercap=sum(bool(x['clock_reasons']&4) for x in [start_reading,*samples,end_reading])
                output=OUT/f'window_{index:02d}_predictions.npy'
                np.save(output,observed,allow_pickle=False)
                row=dict(index=index,**spec,phase=phase,requests=128,seconds=seconds,ms_per_image=seconds*1000/128,
                    stream_span_ms=begin.elapsed_time(end),gpu_energy=energy,cooldown=cooling,
                    start_reading=start_reading,end_reading=end_reading,timed_telemetry_samples=len(samples),
                    software_thermal_samples=thermal,software_powercap_samples=powercap,
                    prediction_mismatches=mismatches,predictions_file=str(output),predictions_file_sha256=digest(output),
                    prediction_array_sha256=array_hash(observed),ended_utc=utc(),accepted=False)
                valid_energy=energy.get('gpu_energy_j') is not None and np.isfinite(energy['gpu_energy_j']) and energy['gpu_energy_j']>0
                row['accepted']=bool(mismatches==0 and thermal==0 and valid_energy and samples)
                state['rows'].append(row)
                atomic_json(REPORT,state)
                print(json.dumps(dict(index=index,accepted=row['accepted'],mismatches=mismatches,thermal_samples=thermal,ms_per_image=row['ms_per_image'])),flush=True)
                if not row['accepted']:
                    raise RuntimeError('Window not accepted; preserve and stop without retry')
                guard.check()
        for p,h in state['protected_hashes'].items():
            assert digest(p)==h, 'Protected source/input changed: '+p
        state.update(status='complete_pending_audit',finished_utc=utc())
        atomic_json(REPORT,state)
    except BaseException as exc:
        state.update(status='stopped_error',error=repr(exc),stopped_utc=utc())
        atomic_json(REPORT,state)
        raise
    finally:
        if guard:
            guard.close()


if __name__=='__main__':
    main()
