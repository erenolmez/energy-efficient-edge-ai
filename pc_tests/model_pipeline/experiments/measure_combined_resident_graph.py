"""TinyCNN + eight resident engines + device-selected CUDA Graph dispatch."""
import ctypes as C
import gc
import json
import os
from pathlib import Path
from statistics import median
import time

import numpy as np

from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.experiments import compare_dispatch_residency_v2 as common
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

OUT = PIPELINE / 'reports/combined_resident_graph_20261105'
BUILD = PIPELINE / 'artifacts/combined_resident_graph_20261105'
BATCHES = (1, 8, 16, 32, 64, 128)
REPEATS = 3
RESERVE = 1024**3
SAMPLE = 1024


def run():
    import torch
    from pc_tests.model_pipeline.runtime.fixed_gpu_runner import FixedGPURunner
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'run.lock').open('x') as lock:
        json.dump(dict(pid=os.getpid(), utc=common.utc()), lock)
    common.OUT = OUT
    guard = common.Guard()
    state = dict(status='preparing', windows=[], batches={}, test_data_accessed=False)

    def save():
        atomic_json(OUT/'study.json', state)

    def memory():
        guard.check()
        free = torch.cuda.mem_get_info()[0]
        if free < RESERVE:
            raise RuntimeError('Below 1 GiB free VRAM; stop without retry')
        return free

    def cool():
        guard.phase = 'cooldown'
        started = time.monotonic()
        while True:
            reading = guard.reading()
            elapsed = time.monotonic()-started
            memory()
            if elapsed >= 5 and reading['temperature_c'] <= 60 and not reading['clock_reasons']&32:
                return reading
            if elapsed > 120:
                raise RuntimeError('Cooldown gate not reached')
            time.sleep(.5)

    def sha(path):
        return common.digest(Path(path))

    save()
    try:
        state['initial_telemetry'] = guard.start()
        torch.set_num_threads(1)
        b1_path = PIPELINE/'reports/correctness_b1_20261028/study.json'
        big_path = PIPELINE/'artifacts/pc_routers_correctness/global_correctness_seed42.json'
        static = PIPELINE/'reports/gpu_static.json'
        cache = PIPELINE/'artifacts/pc_precision/data_cache'
        rgb_path, label_path = (cache/n for n in ('validation_rgb128.npy', 'validation_labels.npy'))
        dll = BUILD/'combined.dll'
        files = (b1_path,big_path,static,rgb_path,label_path,dll,Path(__file__),
                 PIPELINE/'runtime/combined_resident_graph.cu',
                 PIPELINE/'runtime/fixed_gpu_runner.py',PIPELINE/'runtime/gpu_engine_preflight.py')
        protected = {str(p):sha(p) for p in files}
        b1, big = (json.loads(p.read_text()) for p in (b1_path,big_path))
        prep = b1['prepared']['42']
        names = big['models']
        assert names == b1['models'] and len(names) == 213
        assert protected[str(rgb_path)] == b1['validation_rgb_sha256'] == big['validation_rgb_sha256']
        assert prep['calibration_positions'] == big['calibration_positions']
        positions = np.random.default_rng(42).permutation(big['calibration_positions'])[:SAMPLE]
        assert len(set(positions)) == SAMPLE and not set(positions)&set(big['fit_positions'])
        labels = np.load(label_path, allow_pickle=False)[positions]
        raw = np.load(rgb_path, mmap_mode='r', allow_pickle=False)[positions]
        assert raw.shape == (SAMPLE,3,128,128) and labels.shape == (SAMPLE,)
        host = torch.from_numpy(raw).float().div_(255)
        host.sub_(torch.tensor([.5071,.4867,.4408])[None,:,None,None])
        host.div_(torch.tensor([.2675,.2565,.2761])[None,:,None,None])
        device = host.to('cuda')
        records = json.loads(static.read_text())['results']
        pool = sorted((r for r in records if r['name'].endswith('__p0__fp16')), key=lambda r:r['name'])
        assert len(pool) == len({r['architecture'] for r in pool}) == 8
        columns = [names.index(r['name']) for r in pool]

        def trust(p, digest):
            path = Path(p).resolve()
            assert path.drive.upper() == 'C:' and sha(path) == digest
            protected[str(path)] = digest
            return path

        engine_paths = [trust(r['engine_path'],r['engine_sha256']) for r in pool]
        plan = dict(batches=BATCHES, repeats=REPEATS, pool=[r['name'] for r in pool],
                    positions=positions.tolist(), seed=42, reserve_bytes=RESERVE,
                    timing='Normalized GPU-resident input through staging, live TinyCNN and classifier to device predictions. No readback in graph; CPU comparator reads decision scores. Final window synchronization included.',
                    policy='Original seed42 latency penalty and full-pool cost normalization, score maximized over fixed eight-engine subset. No recalibration.',
                    accuracy='Existing classifier validation calibration subset; not independent test accuracy.',
                    energy='Gross NVML GPU-only; startup, input transfer/normalization and final readback excluded.')
        atomic_json(OUT/'plan.json',plan)
        protected[str(OUT/'plan.json')] = sha(OUT/'plan.json')
        state.update(status='running', plan=plan, protected_sha256=protected,
                     gpu=torch.cuda.get_device_name(0))
        save()
        native = C.CDLL(str(dll))
        vp = C.c_void_p
        ip, fp = C.POINTER(C.c_int), C.POINTER(C.c_float)
        for suffix, args in {
            'create':[C.POINTER(vp),vp,vp,vp,vp,vp,C.c_int,ip,fp],
            'begin':[vp,vp,C.c_int], 'end':[vp,vp,C.c_int],
            'instantiate':[vp,ip], 'launch':[vp,vp], 'destroy':[vp],
        }.items():
            function = getattr(native,'combined_'+suffix)
            function.argtypes, function.restype = args, C.c_int
        native.combined_error.restype = C.c_char_p

        def call(suffix, *args):
            if getattr(native, 'combined_'+suffix)(*args):
                raise RuntimeError(suffix+': '+native.combined_error().decode(errors='replace'))

        meter = EnergyCounter()
        assert meter.available, meter.error
        for batch in BATCHES:
            guard.phase = f'setup_b{batch}'
            started = time.perf_counter()
            if batch == 1:
                router_path = trust(prep['engine'],prep['engine_sha256'])
                policy = prep['policies']['latency']
            else:
                spec = big['engines'][str(batch)]
                router_path = trust(spec['engine'],spec['sha256'])
                policy = big['policies']['latency'][str(batch)]
            costs = np.array(policy['costs'], dtype=np.float64)
            chosen_policy = policy['budgets']['1.0']
            assert len(costs)==213 and not chosen_policy['fallback'] and np.all(costs>0)
            offsets = (chosen_policy['penalty']*costs[columns]/costs.max()).astype(np.float32)
            evidence = measure_batch(torch, batch, device, labels, router_path, engine_paths,
                                     columns, offsets, guard, memory, cool, meter, call,
                                     FixedGPURunner, EngineRunner, started, state, save)
            state['batches'][str(batch)] = evidence
            save()
            gc.collect()
            torch.cuda.empty_cache()
            print(f'Batch {batch} complete',flush=True)
        assert len(state['windows']) == len(BATCHES)*REPEATS*2
        for p,digest in protected.items():
            assert sha(p) == digest, p
        state.update(status='complete_pending_cpu_audit', finished_utc=common.utc())
        save()
    except BaseException as exc:
        state.update(status='stopped', error=repr(exc))
        save()
        raise
    finally:
        guard.close()


def measure_batch(torch, batch, device, labels, router_path, engine_paths, columns, offsets,
                  guard, memory, cool, meter, call, Router, Classifier, started, state, save):
    stream = torch.cuda.Stream()
    stage = torch.empty_like(device[:batch])
    shared_logits = torch.empty((batch,100),device='cuda')
    prediction = torch.empty(batch,device='cuda',dtype=torch.int32)
    choice = torch.empty(1,device='cuda',dtype=torch.int32)
    counts = torch.zeros(8,device='cuda',dtype=torch.int32)
    output = torch.empty((SAMPLE//batch,batch),device='cuda',dtype=torch.int32)
    choices_out = torch.empty(SAMPLE//batch,device='cuda',dtype=torch.int32)
    router = Router(router_path)
    assert router.shape == tuple(stage.shape)
    assert router.context.set_tensor_address(router.input,stage.data_ptr())
    probs, = router.outputs.values()
    assert probs.shape == (1,213) and probs.dtype==torch.float32
    engines = []
    extra = 64 * 1024**2
    for path in engine_paths:
        before = memory()
        size = path.stat().st_size
        if before-size-extra < RESERVE:
            raise RuntimeError('Eight-engine pool would exceed VRAM reserve')
        engine = Classifier(path,maximum_batch=batch)
        assert engine.context.set_input_shape(engine.input_name,tuple(stage.shape))
        assert engine.context.get_tensor_shape(engine.output_name)==(batch,100)
        assert engine.context.set_tensor_address(engine.input_name,stage.data_ptr())
        assert engine.context.set_tensor_address(engine.output_name,shared_logits.data_ptr())
        engines.append(engine)
        extra = max(extra,before-memory()-size)
    setup_seconds = time.perf_counter()-started
    vp = C.c_void_p
    handle = vp()

    def enqueue(engine):
        if not engine.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT enqueue failed')

    def capture(engine, index):
        call('begin',handle,stream.cuda_stream,index)
        try:
            enqueue(engine)
        finally:
            call('end',handle,stream.cuda_stream,index)

    def eager_select():
        enqueue(router)
        values = probs.cpu().numpy()[0,columns]
        assert np.isfinite(values).all()
        return int(np.argmax(values-offsets))

    try:
        with torch.inference_mode(),torch.cuda.stream(stream):
            stage.copy_(device[:batch])
            for _ in range(3):
                for runner in [router,*engines]:
                    enqueue(runner)
            stream.synchronize()
            memory()
            graph_started = time.perf_counter()
            call('create',C.byref(handle),probs.data_ptr(),shared_logits.data_ptr(),choice.data_ptr(),
                 prediction.data_ptr(),counts.data_ptr(),batch,(C.c_int*8)(*columns),
                 (C.c_float*8)(*offsets))
            capture(router,-1)
            for i,runner in enumerate(engines):
                capture(runner,i)
            stats = (C.c_int*27)()
            call('instantiate',handle,stats)
            graph_setup = time.perf_counter()-graph_started
            free = memory()
            guard.phase = f'parity_b{batch}'
            ref_logits, ref_predictions, ref_choices = [],[],[]
            for i in range(SAMPLE//batch):
                memory()
                stage.copy_(device[i*batch:(i+1)*batch])
                selected = eager_select()
                enqueue(engines[selected])
                value = shared_logits.cpu().numpy().copy()
                assert np.isfinite(value).all()
                ref_choices.append(selected)
                ref_logits.append(value)
                ref_predictions.append(value.argmax(1))
            ref_predictions, ref_choices = np.asarray(ref_predictions),np.asarray(ref_choices)
            max_error = 0.
            for i in range(SAMPLE//batch):
                memory()
                stage.copy_(device[i*batch:(i+1)*batch])
                shared_logits.fill_(float('nan'));prediction.fill_(-1);choice.fill_(-1);counts.zero_()
                call('launch',handle,stream.cuda_stream)
                result = shared_logits.cpu().numpy()
                assert np.allclose(result,ref_logits[i],atol=1e-4,rtol=1e-4)
                assert int(choice.item())==ref_choices[i]
                assert np.array_equal(prediction.cpu().numpy(),ref_predictions[i])
                assert np.array_equal(counts.cpu().numpy(),np.bincount(ref_choices[i:i+1],minlength=8))
                max_error = max(max_error,float(np.max(np.abs(result-ref_logits[i]))))
            ref_path = OUT/f'b{batch}_reference.npz'
            np.savez_compressed(ref_path, predictions=ref_predictions, choices=ref_choices, labels=labels)
            frequencies = np.bincount(ref_choices,minlength=8)
            def sweep(graph):
                for i in range(SAMPLE//batch):
                    stage.copy_(device[i*batch:(i+1)*batch])
                    if graph:
                        call('launch',handle,stream.cuda_stream)
                        output[i].copy_(prediction)
                        choices_out[i:i+1].copy_(choice)
                    else:
                        selected = eager_select()
                        enqueue(engines[selected])
                        output[i].copy_(shared_logits.argmax(1))
                        choices_out[i].fill_(selected)
                    if i%64==0:
                        guard.check()
                stream.synchronize()

            for repeat in range(REPEATS):
                modes = (False,True) if (repeat+BATCHES.index(batch))%2==0 else (True,False)
                for graph in modes:
                    boundary = cool()
                    guard.phase = 'warmup'
                    sweep(graph)
                    counts.zero_()
                    stream.synchronize()
                    mode = 'combined_graph' if graph else 'resident_cpu_dispatch'
                    phase = f'b{batch}_r{repeat}_{mode}'
                    guard.phase = phase
                    first = guard.reading()
                    meter.start()
                    start = time.perf_counter()
                    sweeps = 0
                    while True:
                        sweep(graph)
                        sweeps += 1
                        elapsed = time.perf_counter()-start
                        if elapsed>=.8 or sweeps>=128:
                            break
                        guard.check()
                    energy = meter.finish(sweeps*SAMPLE)
                    last = guard.reading()
                    guard.phase = 'verification'
                    with guard.lock:
                        readings = [first,last,*(s for s in guard.samples if s['phase']==phase)]
                    assert len(readings)>2 and all(not r['clock_reasons']&32 for r in readings)
                    assert np.isfinite(energy['gpu_energy_j']) and energy['gpu_energy_j']>0
                    result, chosen = output.cpu().numpy(),choices_out.cpu().numpy()
                    assert np.array_equal(result,ref_predictions) and np.array_equal(chosen,ref_choices)
                    branch_counts = counts.cpu().numpy()
                    if graph:
                        assert np.array_equal(branch_counts,frequencies*sweeps)
                    artifact = OUT/(phase+'.npz')
                    np.savez_compressed(artifact,predictions=result,choices=chosen)
                    total_batches = sweeps*SAMPLE//batch
                    row = dict(batch=batch,repeat=repeat,mode=mode,sweeps=sweeps,elapsed=elapsed,
                               ms_per_batch=1000*elapsed/total_batches,
                               gpu_mj_per_batch=1000*energy['gpu_energy_j']/total_batches,
                               energy=energy,branch_counts=branch_counts.tolist(),
                               runtime_engine_loads=0,runtime_engine_evictions=0,cooldown=boundary,
                               first=first,last=last,artifact=str(artifact),sha256=common.digest(artifact))
                    state['windows'].append(row)
                    save()
                    memory()
                    print(f'{phase}: {row["ms_per_batch"]:.3f} ms/batch',flush=True)
            return dict(accuracy=float(100*np.mean(ref_predictions.ravel()==labels)),
                        selected_batches=frequencies.tolist(),
                        switches=int(np.count_nonzero(ref_choices[1:]!=ref_choices[:-1])),
                        max_graph_logit_error=max_error, graph_stats=list(stats),
                        free_after_graph_bytes=free,resident_setup_seconds=setup_seconds,
                        graph_setup_seconds=graph_setup,reference=str(ref_path),
                        reference_sha256=common.digest(ref_path))
    finally:
        if handle.value:
            stream.synchronize()
            call('destroy',handle)


def audit():
    state = json.loads((OUT/'study.json').read_text())
    assert state['status']=='complete_pending_cpu_audit' and not state['test_data_accessed']
    for path,digest in state['protected_sha256'].items():
        assert common.digest(Path(path))==digest, path
    plan = json.loads((OUT/'plan.json').read_text())
    assert plan==state['plan'] and plan['batches']==list(BATCHES)
    assert len(state['windows'])==len(BATCHES)*REPEATS*2
    rows = {(r['batch'],r['repeat'],r['mode']):r for r in state['windows']}
    assert len(rows)==len(state['windows'])
    lines = ['# Combined overhead reductions: eight resident engines + TinyCNN + CUDA Graph',
             '',f"Measured on {state['gpu']}. No accuracy gate was applied.",
             '', '| Batch | Protocol | Validation accuracy (%) | ms/batch | GPU mJ/batch |',
             '|---:|:---|---:|---:|---:|']
    details = []
    for batch in BATCHES:
        proof = state['batches'][str(batch)]
        assert proof['free_after_graph_bytes']>=RESERVE
        assert proof['graph_stats'][1:3]==[0,8]
        assert all(proof['graph_stats'][i:i+2]==[0,0] for i in range(4,27,3))
        assert common.digest(Path(proof['reference']))==proof['reference_sha256']
        with np.load(proof['reference'],allow_pickle=False) as ref:
            preds, choices, labels = ref['predictions'],ref['choices'],ref['labels']
        assert preds.shape==(SAMPLE//batch,batch) and choices.shape==(SAMPLE//batch,)
        assert labels.shape==(SAMPLE,) and np.isclose(proof['accuracy'],100*np.mean(preds.ravel()==labels))
        frequencies = np.bincount(choices,minlength=8)
        assert frequencies.tolist()==proof['selected_batches']
        for mode in ('combined_graph','resident_cpu_dispatch'):
            group = [rows[batch,r,mode] for r in range(REPEATS)]
            for row in group:
                assert common.digest(Path(row['artifact']))==row['sha256']
                with np.load(row['artifact'],allow_pickle=False) as actual:
                    assert np.array_equal(actual['predictions'],preds)
                    assert np.array_equal(actual['choices'],choices)
                assert row['runtime_engine_loads']==row['runtime_engine_evictions']==0
                assert row['energy']['energy_window_images']==SAMPLE*row['sweeps']
                assert all(np.isfinite(row[k]) and row[k]>0 for k in ('ms_per_batch','gpu_mj_per_batch'))
                if mode=='combined_graph':
                    assert row['branch_counts']==(frequencies*row['sweeps']).tolist()
            protocol = '8 resident + GPU dispatch graph' if mode=='combined_graph' else '8 resident + CPU dispatch (control)'
            ms = median(r['ms_per_batch'] for r in group)
            mj = median(r['gpu_mj_per_batch'] for r in group)
            lines.append(f"| {batch} | {protocol} | {proof['accuracy']:.2f} | {ms:.3f} | {mj:.2f} |")
        details.append(f"| {batch} | {np.count_nonzero(frequencies)} | {proof['switches']} | {proof['free_after_graph_bytes']/1024**2:.0f} |")
    lines += ['', 'Three-repeat medians. Same inputs, scores, policy, pool and output boundary in both modes.',
              'Timing begins with normalized inputs already in GPU memory and ends with device predictions; final window synchronization included.',
              'Image decoding, host transfer, preprocessing, readback, model loading and graph capture are outside the timer.',
              'GPU energy is gross NVML GPU energy, not whole-PC energy.',
              '', 'Validation accuracy is descriptive on 1,024 seed-42 calibration images, not held-out test accuracy.',
              'Eight unpruned FP16 engines, one per architecture; actual model use may be lower. The original TinyCNN '
              'penalty was restricted to the eight candidates without recalibration. No historical speedup is inferred from unmatched protocols.',
              '', '| Batch | Models selected | Switches in one sweep | Free VRAM after graph (MiB) |',
              '|---:|---:|---:|---:|', *details,
              '', 'No host nodes inside the conditional graph; CPU submits work. Exactly one classifier branch executed per batch. '
              'Saved final-sweep predictions and choices match the eager reference for every window.']
    with (OUT/'RESULTS.md').open('x',encoding='utf-8') as file:
        file.write('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    import sys
    if sys.argv[1:]==['--audit']:
        audit()
    elif not sys.argv[1:]:
        run()
    else:
        raise SystemExit('Expected no arguments, or --audit')
