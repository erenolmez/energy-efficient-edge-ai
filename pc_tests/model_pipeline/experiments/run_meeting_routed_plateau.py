"""End-to-end replay of the frozen 213-choice TinyCNN plateau experiment."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from pc_tests.model_pipeline.experiments import diagnose_batch_plateau as prior_diag
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.routers.correctness.correctness_policy import select_correctness
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import make_batches
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
from pc_tests.model_pipeline.runtime.global_candidate_pool import EngineCache
from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner

SIZES = (8,16,32,64,128)


class ThermalViolation(RuntimeError):
    pass


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def cool_for_replay(guard, window, max_c=65):
    guard.window = window + ':cooldown'
    start = time.monotonic()
    while True:
        guard.check()
        temp = guard.nv.nvmlDeviceGetTemperature(guard.handle,guard.nv.NVML_TEMPERATURE_GPU)
        flags = guard.nv.nvmlDeviceGetCurrentClocksThrottleReasons(guard.handle)
        elapsed = time.monotonic()-start
        if elapsed >= 15 and temp <= max_c and not flags & 96:
            return dict(seconds=elapsed,temperature_c=temp,throttle_reasons=flags)
        if elapsed >= 240:
            raise ThermalViolation(f'Repeat could not cool to {max_c} C without thermal throttling within 240 s')
        guard.done.wait(1)


def calibration_hot_set(calibration, image_bank, batch, routers, policy, guard):
    counts = np.zeros(len(policy['costs']),dtype=np.int64)
    inputs = SharedBatchInputs(maximum_batch=batch)
    for positions in make_batches(calibration,batch,42+batch):
        guard.check()
        host = torch.from_numpy(np.array(image_bank[positions])).pin_memory()
        p = routers[len(host)].run(inputs.prepare(host).contiguous()).cpu().numpy().copy()
        index = int(policy['baseline_index']) if policy['fallback'] else int(
            select_correctness(p,policy['costs'],policy['penalty'])[0])
        counts[index] += 1
    return sorted(np.flatnonzero(counts).tolist(),key=lambda i:(-counts[i],i))


def measure(hosts, batches, labels, pool, routers, policy, meter, guard, controlled=False, strict=False,
            optimized=False, timing_start_max_c=65):
    inputs = SharedBatchInputs(maximum_batch=max(map(len,hosts)) if optimized else 128)
    if optimized and policy is not None:
        costs = np.asarray(policy['costs'])
        if not np.isfinite(costs).all() or not (costs>0).all():
            raise ValueError('Invalid frozen routing costs')
        adjustment = policy['penalty'] * costs / costs.max()

    def forward(host):
        normalized = inputs.prepare(host)
        if policy is None:
            chosen = 0
        else:
            p = routers[len(normalized)].run(normalized.contiguous()).cpu().numpy().copy()
            if p.shape != (1,len(pool)) or not np.isfinite(p).all():
                raise ValueError('Invalid router output')
            if policy['fallback']:
                chosen = int(policy['baseline_index'])
            elif optimized:
                chosen = int(np.argmax(p[0] - adjustment))
            else:
                chosen = int(select_correctness(p, policy['costs'], policy['penalty'])[0])
        return pool[chosen].run(normalized,copy_input=False).argmax(1).cpu().numpy(), chosen

    window = guard.window
    cooling = cool_for_replay(guard,window) if controlled else None
    if controlled:
        guard.window = window + ':warmup'
    min_batches, min_seconds = (50,30) if controlled else (20,10)
    start = time.perf_counter()
    n = 0
    while n < min_batches or time.perf_counter()-start < min_seconds:
        guard.check()
        forward(hosts[n % len(hosts)])
        n += 1
    torch.cuda.synchronize()
    warmup_s = time.perf_counter()-start
    # Replay the historical measured cold-cache boundary after longer preconditioning.
    if not optimized:
        pool.clear()
    timing_cooldown = cool_for_replay(guard,window + ':postwarm',timing_start_max_c) if strict else None
    pred = np.empty(len(labels),dtype=np.int16)
    selected_batches = np.zeros(len(pool),dtype=np.int64)
    selected_images = np.zeros(len(pool),dtype=np.int64)
    elapsed = []
    loads, evictions = pool.loads, pool.evictions
    prior_disk_reads = pool.disk_reads if optimized else 0
    def gpu_state():
        return dict(temperature_c=guard.nv.nvmlDeviceGetTemperature(guard.handle,guard.nv.NVML_TEMPERATURE_GPU),
                    throttle_reasons=guard.nv.nvmlDeviceGetCurrentClocksThrottleReasons(guard.handle))
    def require_no_throttle(reading):
        if reading['throttle_reasons'] & 96:
            raise ThermalViolation(f"Thermal throttle at {reading['temperature_c']} C (flags {reading['throttle_reasons']})")
    measured_start = gpu_state() if controlled else None
    if strict:
        require_no_throttle(measured_start)
        if measured_start['temperature_c'] > timing_start_max_c:
            raise ThermalViolation(f"Timing start exceeded {timing_start_max_c} C: {measured_start['temperature_c']}")
    if controlled:
        guard.window = window + ':timed'
    meter.start()
    next_poll = time.monotonic()
    for idx,host in zip(batches,hosts):
        guard.check()
        if strict and time.monotonic() >= next_poll:
            require_no_throttle(gpu_state())
            next_poll = time.monotonic() + .25
        start = time.perf_counter()
        output, chosen = forward(host)
        elapsed.append(time.perf_counter()-start)
        pred[idx] = output
        selected_batches[chosen] += 1
        selected_images[chosen] += len(idx)
    energy = meter.finish(len(labels))
    measured_end = gpu_state() if controlled else None
    if strict:
        require_no_throttle(measured_end)
    if controlled:
        guard.window = window + ':finished'
        guard.check()
    seconds = sum(elapsed)
    joules = energy.get('gpu_energy_j')
    if not joules or not np.isfinite(joules) or joules <= 0 or energy['energy_window_seconds'] < 3:
        raise ValueError('Invalid GPU energy or full test sweep shorter than 3 seconds')
    return dict(test_accuracy=100*float(np.mean(pred==labels)), images=len(labels), batches=len(batches),
        ms_per_batch=1000*seconds/len(batches), ms_per_image=1000*seconds/len(labels),
        gpu_mj_per_batch=1000*joules/len(batches), gpu_mj_per_image=1000*joules/len(labels),
        warmup_s=warmup_s, warmup_batches=n, loads=pool.loads-loads, evictions=pool.evictions-evictions,
        cooldown=cooling, timing_cooldown=timing_cooldown, gpu_at_timed_start=measured_start,
        gpu_at_timed_end=measured_end,
        timed_disk_reads=pool.disk_reads-prior_disk_reads if optimized else None,
        min_gpu_free_bytes=pool.min_free if optimized else None,
        selected_batches=selected_batches.tolist(), selected_images=selected_images.tolist(), energy=energy,
        predictions_sha256=hashlib.sha256(pred.tobytes()).hexdigest())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--controlled-router-only', action='store_true')
    cli.add_argument('--strict-thermal', action='store_true', help='Reject any sampled thermal throttle during timing')
    cli.add_argument('--optimized-residency', action='store_true')
    cli.add_argument('--fixed-resnet101', action='store_true')
    cli.add_argument('--batch', action='append', choices=SIZES, type=int)
    cli.add_argument('--timing-start-max-c', choices=(62,63,64,65), type=int, default=65)
    cli.add_argument('--rounds', choices=(1,3), type=int, default=3)
    args=cli.parse_args()
    sizes=tuple(dict.fromkeys(args.batch or SIZES))
    if args.fixed_resnet101:
        if args.optimized_residency:
            cli.error('Select either optimized router or fixed ResNet101')
        args.strict_thermal=True
    if args.strict_thermal:
        args.controlled_router_only = True
    if args.optimized_residency and not args.strict_thermal:
        cli.error('Optimized residency requires the strict thermal protocol')
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    state={'status':'preflight','rows':[],'rejected':[]}
    guard=None
    try:
        cache=PIPELINE/'artifacts/pc_precision/data_cache'
        marker_path=PIPELINE/'artifacts/pc_routers_correctness/global_correctness_seed42.json'
        history_path=PIPELINE/'reports/pc_routes_correctness.json'
        static_path=PIPELINE/'reports/gpu_static.json'
        marker=json.loads(marker_path.read_text()); history=json.loads(history_path.read_text())
        records=sorted(json.loads(static_path.read_text())['results'],key=lambda r:r['name'])
        names=[r['name'] for r in records]; by_name={r['name']:r for r in records}
        if len(names)!=213 or names!=marker['models'] or names!=history['models'] or history['resident_capacity']!=2:
            raise ValueError('Historical configuration is not the 213-choice / cache-2 campaign')
        data_hashes={n:sha(cache/n) for n in ('test_rgb128.npy','test_labels.npy')}
        if data_hashes != history['test_hashes'] or sha(marker_path)!=history['training_markers'][marker_path.name]:
            raise ValueError('Frozen policy metadata or test inputs changed')
        runs={b:next(r for r in history['results'] if r['seed']==42 and r['objective']=='latency'
                    and r['budget_pp']==1 and r['batch_size']==b and r['status']=='complete') for b in SIZES}
        plan=dict(batches=sizes,seed=42,repeats=3,expected_windows=30,capacity=2,budget_pp=1,
            objective='latency',images=10000,minimum_warmup_seconds=10,minimum_warmup_batches=20,
            cache_reset_after_warmup=True,comparators={b:r['best_eligible_static'] for b,r in runs.items()},
            input_hashes=data_hashes,source_hashes={str(p):sha(p) for p in (marker_path,history_path,static_path,Path(__file__))})
        if args.controlled_router_only:
            plan.update(expected_windows=15,comparators={},minimum_warmup_seconds=30,
                        minimum_warmup_batches=50,cooldown_minimum_seconds=15,cool_start_max_c=65,
                        hard_stop_c=85,thermal_flags_are_qualification_not_rejection=True)
        if args.strict_thermal:
            plan.update(repeats=args.rounds,expected_windows=len(sizes)*args.rounds,
                        cool_after_warmup=True,thermal_flags_are_qualification_not_rejection=False,
                        reject_timed_thermal_flags=True,timing_start_max_c=args.timing_start_max_c)
        if args.fixed_resnet101:
            from pc_tests.model_pipeline.runtime.routed_residency import RoutedResidency,RESERVE,RAM_LIMIT
            plan.update(capacity=1,minimum_gpu_free_bytes=RESERVE,host_cache_limit_bytes=RAM_LIMIT,
                        fixed_classifier='resnet101__p0__fp16',cache_reset_after_warmup=False)
        if args.optimized_residency:
            from pc_tests.model_pipeline.runtime.routed_residency import RoutedResidency, RESERVE, RAM_LIMIT
            cal=np.asarray(marker['calibration_positions'],dtype=int)
            cal_path=cache/'validation_rgb128.npy'
            calibration_images=np.load(cal_path,mmap_mode='r',allow_pickle=False)
            if len(cal)!=2000 or len(set(cal.tolist()))!=2000 or np.any(cal<0) or np.any(cal>=len(calibration_images)):
                raise ValueError('Invalid validation-only calibration positions')
            reference_path=PIPELINE/'artifacts/meeting_routed_plateau_controlled_20261123/study.json'
            reference=json.loads(reference_path.read_text())
            expected={b:{r['predictions_sha256'] for r in reference['rows'] if r['batch']==b and r['mode']=='route'}
                      for b in SIZES}
            if reference['status']!='complete_pending_audit' or len(reference['rows'])!=15 or any(len(v)!=1 for v in expected.values()):
                raise ValueError('Cannot establish prediction parity with original replay')
            plan.update(capacity='adaptive: calibration frequency, measured GPU reserve',hot_slots=None,
                        transient_nonresident_engine=True,min_gpu_free_bytes=RESERVE,host_cache_limit_bytes=RAM_LIMIT,
                        hot_pool_source='selection frequency on 2,000 calibration images, never test data',
                        calibration_sha256=sha(cal_path),prediction_reference_sha256=sha(reference_path),
                        cache_reset_after_warmup=False,optimized_dispatch='precomputed frozen cost adjustment')
        atomic_json(out/'plan.json',plan)
        labels=np.load(cache/'test_labels.npy',allow_pickle=False)
        images=np.load(cache/'test_rgb128.npy',mmap_mode='r',allow_pickle=False)
        if len(labels)!=len(images) or len(labels)!=10000:
            raise ValueError('Expected the 10,000 held-out images')
        prior_diag.OUT=out
        guard=prior_diag.Telemetry()
        if not args.strict_thermal and guard.nv.nvmlDeviceGetTemperature(guard.handle,guard.nv.NVML_TEMPERATURE_GPU)>65:
            raise RuntimeError('Routed campaign start requires at most 65 C')
        guard.thread.start()
        meter=EnergyCounter()
        if not meter.available:
            raise RuntimeError('GPU energy counter unavailable')

        def load(record):
            path=Path(record['engine_path'])
            if sha(path)!=record['engine_sha256']:
                raise ValueError('Classifier hash mismatch: '+record['name'])
            return EngineRunner(path)

        def release():
            gc.collect(); torch.cuda.empty_cache()

        state['status']='running'
        hot_sets={}
        with torch.inference_mode():
            for repeat,offset in enumerate((0,2,4)[:args.rounds]):
                shift=offset%len(sizes)
                for b in sizes[shift:]+sizes[:shift]:
                    batches=make_batches(np.arange(len(labels)),b,42+b)
                    if sorted(np.concatenate(batches).tolist())!=list(range(len(labels))):
                        raise ValueError('Test batching omitted or repeated a sample')
                    routers={}
                    shapes={len(i) for i in batches}
                    if args.optimized_residency and len(cal)%b:
                        shapes.add(len(cal)%b)
                    for size in (() if args.fixed_resnet101 else sorted(shapes)):
                        spec=marker['engines'][str(size)]
                        if sha(spec['engine'])!=spec['sha256']:
                            raise ValueError('Router engine hash mismatch')
                        routers[size]=RouterRunner(spec['engine'])
                    hosts=[torch.from_numpy(np.array(images[i])).pin_memory() for i in batches]
                    config=marker['policies']['latency'][str(b)]
                    policy=dict(config['budgets']['1.0'],costs=config['costs'])
                    if args.optimized_residency and b not in hot_sets:
                        guard.window=f'b{b}:calibration'
                        hot_sets[b]=calibration_hot_set(cal,calibration_images,b,routers,policy,guard)
                        state['calibration_hot_sets']={k:[names[i] for i in v] for k,v in hot_sets.items()}
                        atomic_json(out/'study.json',state)
                    modes=('route','eligible_static') if (repeat+SIZES.index(b))%2==0 else ('eligible_static','route')
                    if args.controlled_router_only:
                        modes=('route',)
                    if args.fixed_resnet101:
                        modes=('fixed_resnet101',)
                    for mode in modes:
                        guard.window=f'r{repeat}_b{b}_{mode}'
                        route=mode=='route'
                        active=(records if route else
                                [by_name['resnet101__p0__fp16' if args.fixed_resnet101 else runs[b]['best_eligible_static']]])
                        pinned=args.optimized_residency or args.fixed_resnet101
                        pool=(RoutedResidency(active,hot_sets[b] if route else [0],b,release) if pinned else
                              EngineCache(active,2 if route else 1,load,torch.cuda.synchronize,release))
                        if pinned:
                            admitted=[active[i]['name'] for i in pool.pinned]
                        if args.optimized_residency:
                            print(f'Batch {b}, repeat {repeat+1}: pinned {len(admitted)}/{len(hot_sets[b])} calibration-selected engines',flush=True)
                        try:
                            row=measure(hosts,batches,labels,pool,routers,policy if route else None,meter,guard,
                                        controlled=args.controlled_router_only,strict=args.strict_thermal,
                                        optimized=pinned,timing_start_max_c=args.timing_start_max_c)
                            if args.optimized_residency and row['predictions_sha256'] not in expected[b]:
                                raise ValueError(f'Optimized cache or dispatch changed predictions at batch {b}')
                            if pinned:
                                row['pinned_models']=admitted
                            if args.fixed_resnet101 and not np.isclose(row['test_accuracy'],active[0]['test_accuracy'],rtol=0,atol=1e-9):
                                raise ValueError(f'Fixed classifier accuracy differs from archived test score at batch {b}')
                            counts=row.pop('selected_batches'); image_counts=row.pop('selected_images')
                            row.update(batch=b,repeat=repeat,mode=mode,selections={r['name']:{'batches':n,'images':m}
                                for r,n,m in zip(active,counts,image_counts) if n})
                            state['rows'].append(row)
                            atomic_json(out/'study.json',state)
                            print(json.dumps({k:row[k] for k in ('batch','repeat','mode','test_accuracy','ms_per_image','gpu_mj_per_image')}),flush=True)
                        except ThermalViolation as exc:
                            state['rejected'].append(dict(batch=b,repeat=repeat,error=str(exc)))
                            atomic_json(out/'study.json',state)
                            print(json.dumps(state['rejected'][-1]),flush=True)
                        finally:
                            pool.clear()
                    del hosts,routers
                    release()
        state['status']='complete_with_thermal_rejections' if state['rejected'] else 'complete_pending_audit'
    except BaseException as exc:
        state.update(status='stopped',error=repr(exc))
        raise
    finally:
        if guard and guard.thread.ident:
            guard.done.set(); guard.thread.join(timeout=3)
        atomic_json(out/'study.json',state)


if __name__=='__main__':
    main()
