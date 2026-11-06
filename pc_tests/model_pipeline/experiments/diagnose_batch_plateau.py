"""Bounded validation-only full-routing diagnostic, never historical resume."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from datetime import datetime, timezone

from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

OUT = PIPELINE/'artifacts/batch_plateau_20261103'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Telemetry:
    """Read-only one-second sampling; conservative85C stop, no shutdown."""
    def __init__(self):
        import pynvml as nv
        nv.nvmlInit()
        self.nv, self.handle = nv, nv.nvmlDeviceGetHandleByIndex(0)
        self.done, self.unsafe = threading.Event(), threading.Event()
        self.error, self.window = None, 'setup'
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        nv, h = self.nv, self.handle
        with (OUT/'telemetry.jsonl').open('x', encoding='utf-8') as stream:
            while not self.done.is_set():
                row = dict(utc=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic(), window=self.window)
                queries = dict(temperature_c=lambda:nv.nvmlDeviceGetTemperature(h,nv.NVML_TEMPERATURE_GPU),
                               power_mw=lambda:nv.nvmlDeviceGetPowerUsage(h),
                               sm_clock_mhz=lambda:nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_SM),
                               memory_clock_mhz=lambda:nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_MEM),
                               gpu_utilization_pct=lambda:nv.nvmlDeviceGetUtilizationRates(h).gpu,
                               memory_used_bytes=lambda:nv.nvmlDeviceGetMemoryInfo(h).used,
                               performance_state=lambda:nv.nvmlDeviceGetPerformanceState(h),
                               throttle_reasons=lambda:nv.nvmlDeviceGetCurrentClocksThrottleReasons(h))
                for key, query in queries.items():
                    try:
                        row[key] = query()
                    except Exception as exc:
                        row[key] = None
                        row[key+'_unavailable'] = str(exc)
                if row['temperature_c'] is None or row['temperature_c'] >= 85:
                    self.error = 'Temperature unavailable or conservative85C threshold reached'
                    self.unsafe.set()
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                self.done.wait(1)

    def check(self):
        if self.unsafe.is_set() or not self.thread.is_alive():
            raise RuntimeError(self.error or 'Telemetry thread stopped')


def main():
    # Presence is an exclusive run reservation; never automatically resume.
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'run.lock').open('x') as lock:
        lock.write(json.dumps(dict(pid=os.getpid(),executable=sys.executable,started_utc=datetime.now(timezone.utc).isoformat())))
    state = dict(status='preflight',results=[], energy_scope='NVML GPU only; CPU/restPC excluded',test_accessed=False)
    telemetry = None
    try:
        import numpy as np
        import torch
        from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
        from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner
        from pc_tests.model_pipeline.runtime.global_candidate_pool import EngineCache
        from pc_tests.model_pipeline.runtime.measure_pc_routes import measure
        from pc_tests.model_pipeline.routers.scalar.batch_router_policy import make_batches
        cache_dir=PIPELINE/'artifacts/pc_precision/data_cache'
        marker_path=PIPELINE/'artifacts/pc_routers_correctness/global_correctness_seed42.json'
        marker=json.loads(marker_path.read_text())
        historical_path=PIPELINE/'reports/pc_routes_correctness.json'
        historical=json.loads(historical_path.read_text())
        static_path=PIPELINE/'reports/gpu_static.json'
        records=sorted(json.loads(static_path.read_text())['results'],key=lambda r:r['name'])
        names=[r['name'] for r in records]
        if marker['status']!='complete' or marker['models']!=names or historical['models']!=names:
            raise RuntimeError('Frozen candidate identity mismatch')
        labels=np.load(cache_dir/'validation_labels.npy',allow_pickle=False)
        # Fixed class-balanced validation sample: five images/class, then12
        # additional predefined indices to obtain512. No test files opened.
        positions=np.concatenate([np.flatnonzero(labels==c)[:5] for c in range(100)])
        extra=np.setdiff1d(np.arange(len(labels)),positions)[:12]
        positions=np.concatenate((positions,extra))
        if len(positions)!=512 or len(np.unique(positions))!=512:
            raise RuntimeError('Validation sample malformed')
        sizes=(32,64,128)
        policies={}
        for b in sizes:
            item=marker['policies']['latency'][str(b)]
            policies[b]=dict(item['budgets']['1.0'],costs=item['costs'])
        original_sources=[PIPELINE/'runtime/measure_pc_routes.py',PIPELINE/'runtime/global_candidate_pool.py',Path(__file__)]
        plan=dict(seed=42,objective='latency',budget_pp=1,capacity=2,images=512,
                  validation_positions=positions.tolist(),batch_orders=[[32,64,128],[64,128,32],[128,32,64]],
                  mode_orders=[['eligible','route','reference'],['route','reference','eligible'],['reference','eligible','route']],
                  expected_windows=27,warmup_batches=20,minimum_energy_window_seconds=3,
                  policies=policies, selection_rule='Original frozen policies, no recalibration or test selection',
                  note='Validation engineering diagnostic only; sample is not perfectly balanced (512 images), no deployment accuracy claim. Cache reset after warm-up matches historical implementation.',
                  sources_sha256={str(p):digest(p) for p in [marker_path,historical_path,static_path,*original_sources]},
                  data_sha256={n:digest(cache_dir/n) for n in ('validation_rgb128.npy','validation_labels.npy')})
        atomic_json(OUT/'plan.json',plan)
        state['plan_sha256']=digest(OUT/'plan.json')
        telemetry=Telemetry()
        if telemetry.nv.nvmlDeviceGetTemperature(telemetry.handle,telemetry.nv.NVML_TEMPERATURE_GPU)>=80:
            raise RuntimeError('GPU too hot for initial diagnostic; not launched')
        telemetry.thread.start()
        images=np.array(np.load(cache_dir/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)[positions])
        selected_labels=labels[positions]
        def release():
            gc.collect()
            torch.cuda.empty_cache()
        def load(record):
            path=Path(record['engine_path'])
            if path.drive.upper()!='C:' or digest(path)!=record['engine_sha256']:
                raise RuntimeError('Engine identity/C-drive restriction failed')
            return EngineRunner(path)
        by_name={r['name']:r for r in records}
        routers={}
        state['status']='running'
        atomic_json(OUT/'study.json',state)
        for repeat, order in enumerate(plan['batch_orders']):
            for b in order:
                telemetry.check()
                spec=marker['engines'][str(b)]
                engine=Path(spec['engine'])
                if engine.drive.upper()!='C:' or digest(engine)!=spec['sha256']:
                    raise RuntimeError('Router identity failed')
                routers[b]=RouterRunner(engine)
                historical_row=next(r for r in historical['results'] if r['seed']==42 and r['objective']=='latency' and r['budget_pp']==1 and r['batch_size']==b)
                batches=make_batches(np.arange(512),b,42+b)
                for mode in plan['mode_orders'][repeat]:
                    telemetry.check()
                    identity=f'r{repeat}_b{b}_{mode}'
                    telemetry.window=identity
                    active=records if mode=='route' else [by_name[historical_row['best_eligible_static'] if mode=='eligible' else historical_row['baseline']]]
                    pool=EngineCache(active,2 if mode=='route' else 1,load,torch.cuda.synchronize,release)
                    started=time.monotonic()
                    try:
                        result=measure(images,selected_labels,batches,pool,names.index(marker['baseline']) if mode=='route' else 0,
                                       routers if mode=='route' else None,policies[b] if mode=='route' else None,
                                       'correctness_images',repeats=1,warmup=20,minimum_seconds=3)[0]
                        if not result.get('gpu_energy_j_per_image') or result['gpu_energy_j_per_image']<=0:
                            raise RuntimeError('Unavailable/nonpositive measured GPU energy')
                        state['results'].append(dict(id=identity,batch=b,mode=mode,repeat=repeat,start_monotonic=started,
                                                     end_monotonic=time.monotonic(),measurement=result))
                        atomic_json(OUT/'study.json',state)
                    finally:
                        pool.clear()
                    telemetry.check()
                del routers[b]
                release()
        if len(state['results'])!=27:
            raise RuntimeError('Incomplete diagnostic')
        state['status']='complete_pending_audit'
    except Exception as exc:
        state.update(status='stopped_preserve_no_restart',error=repr(exc))
        raise
    finally:
        if telemetry:
            telemetry.done.set()
            if telemetry.thread.ident is not None:
                telemetry.thread.join(timeout=3)
        atomic_json(OUT/'study.json',state)


if __name__=='__main__':
    main()
