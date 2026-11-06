"""Measure larger-batch residency and fixed-route eight-engine cache overhead."""
from collections import OrderedDict
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from pc_tests.model_pipeline.experiments import compare_dispatch_residency_v2 as common
from pc_tests.model_pipeline.experiments.probe_gpu_residency import order_records
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

OUT = PIPELINE / 'reports/larger_residency_20261105'
MIB = 1024**2
RESERVE = 1024 * MIB
BATCHES = (32, 64, 128)
# One model per batch. Reuse distances vary: not all accesses are misses at c6.
TRACE = (0, 1, 0, 2, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 6, 7,
         0, 2, 0, 1, 3, 5, 4, 6, 7, 1, 2, 3, 5, 7, 4, 6)
REPLAYS = 3


def reference_cache_counts(capacity):
    # Independent reference with an ordinary list, least recent first.
    cache = []
    loads = evictions = 0
    for position, i in enumerate(TRACE * (REPLAYS + 1)):
        timed = position >= len(TRACE)
        if i in cache:
            cache.remove(i)
        else:
            loads += timed
            if len(cache) == capacity:
                cache.pop(0)
                evictions += timed
        cache.append(i)
    return loads, evictions


def main():
    import numpy as np
    import torch
    from pc_tests.model_pipeline.runtime import gpu_engine_preflight as trt_runtime
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner

    class HostEngineRunner(trt_runtime.EngineRunner):
        """Original execution path, deserialize from RAM instead of disk."""
        def __init__(self, blob, maximum_batch):
            trt = trt_runtime.trt
            self.logger = trt_runtime.TRT_LOGGER
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(blob)
            if self.engine is None:
                raise RuntimeError('TensorRT deserialization failed')
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError('TensorRT context creation failed')
            self.stream = torch.cuda.Stream()
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
            self.output_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
            assert self.engine.get_tensor_dtype(self.input_name) == trt.float32
            self.input = torch.empty((maximum_batch, 3, 128, 128), device='cuda', dtype=torch.float32)
            self.output = torch.empty((maximum_batch, 100), device='cuda', dtype=torch.float32)
            assert self.context.set_tensor_address(self.input_name, self.input.data_ptr())
            assert self.context.set_tensor_address(self.output_name, self.output.data_ptr())

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'run.lock').open('x') as handle:
        json.dump({'pid': os.getpid(), 'started_utc': common.utc()}, handle)
    common.OUT = OUT
    guard = common.Guard()
    plan = dict(batches=BATCHES, reserve_bytes=RESERVE, route=TRACE, repeats=3,
                timed_replays=REPLAYS, capacities=(6, 8), peak_stop_c=80,
                pool='Eight original unpruned FP16 engines, sorted by name.',
                boundary='Resident synthetic FP32 input to device predictions; timed cache misses deserialize pre-read host bytes.',
                qualifications='Synthetic fixed CPU route, not learned routing or production traffic. Gross GPU energy only. No disk I/O in timer.')
    atomic_json(OUT / 'plan.json', plan)
    state = dict(status='preparing', capacity=[], windows=[], skipped=[], plan=plan)

    def save():
        atomic_json(OUT / 'study.json', state)

    def digest(p):
        return common.digest(Path(p))

    def check_memory():
        guard.check()
        free, total = torch.cuda.mem_get_info()
        if free < RESERVE:
            raise RuntimeError('GPU free memory fell below the reserve; stop without retry')
        return free

    def release():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    def logits(runner, data):
        return runner.run(data).cpu().numpy().copy()

    try:
        state['initial_telemetry'] = guard.start()
        torch.set_num_threads(1)
        static = PIPELINE / 'reports/gpu_static.json'
        prior = PIPELINE / 'reports/correctness_b1_20261028/study.json'
        records = json.loads(static.read_text())['results']
        assert len(records) == 213 and all(r['status'] == 'complete' for r in records)
        assert all(Path(r['engine_path']).resolve().drive.upper() == 'C:' for r in records)
        prepared = json.loads(prior.read_text())['prepared']['42']
        router_path = Path(prepared['engine']).resolve()
        assert router_path.drive.upper() == 'C:' and digest(router_path) == prepared['engine_sha256']
        protected = {str(p): digest(p) for p in (static, prior, router_path, Path(__file__),
                     Path(trt_runtime.__file__), OUT / 'plan.json')}
        state['protected_sha256'] = protected
        state['gpu'] = torch.cuda.get_device_name(0)
        rng = torch.Generator(device='cuda').manual_seed(42)
        inputs = torch.randn((128, 3, 128, 128), generator=rng, device='cuda')
        state['input_sha256'] = hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest()
        router = RouterRunner(str(router_path))
        assert np.isfinite(router.run(inputs[:1]).cpu().numpy()).all()
        release()
        orders = order_records(records)
        pool = sorted((r for r in records if r['name'].endswith('__p0__fp16')), key=lambda r:r['name'])
        assert len(pool) == len({r['architecture'] for r in pool}) == 8
        state['pool'] = [r['name'] for r in pool]

        def trust(record):
            p = str(Path(record['engine_path']).resolve())
            if p not in protected:
                actual = digest(p)
                if actual != record['engine_sha256']:
                    raise RuntimeError('Frozen engine hash mismatch: ' + p)
                protected[p] = actual
            return p

        with torch.inference_mode():
            for batch in BATCHES:
                for order, rows in orders.items():
                    guard.phase = f'capacity:{batch}:{order}'
                    resident, references, accepted = [], [], []
                    release()
                    initial = check_memory()
                    extra = 64 * MIB
                    data = inputs[:batch]
                    row = dict(batch=batch, order=order, initial_free_bytes=initial,
                               models=accepted, status='loading')
                    state['capacity'].append(row)
                    save()
                    for record in rows:
                        before = check_memory()
                        size = Path(record['engine_path']).stat().st_size
                        if before - size - extra < RESERVE:
                            row.update(stop='conservative admission', next_candidate=record['name'],
                                       next_estimate_bytes=size+extra)
                            break
                        runner = trt_runtime.EngineRunner(trust(record), maximum_batch=batch)
                        value = logits(runner, data)
                        assert value.shape == (batch, 100) and np.isfinite(value).all()
                        after = torch.cuda.mem_get_info()[0]
                        if after < RESERVE:
                            del runner
                            release()
                            check_memory()
                            row.update(stop='candidate rolled back', next_candidate=record['name'],
                                       rollback_observed_free_bytes=after)
                            break
                        resident.append(runner)
                        del runner
                        references.append(value)
                        accepted.append(dict(name=record['name'], free_bytes=after,
                                             incremental_bytes=before-after,
                                             sha256=record['engine_sha256']))
                        extra = max(extra, before-after-size)
                        save()
                    else:
                        row['stop'] = 'all engines loaded'
                    for index, expected in enumerate(references):
                        actual = logits(resident[index], data)
                        assert np.allclose(actual, expected, atol=1e-5, rtol=1e-4)
                        check_memory()
                    row.update(status='verified', count=len(resident), verified_free_bytes=check_memory())
                    print(f'Capacity b{batch} {order}: {len(resident)}', flush=True)
                    resident.clear()
                    release()
                    save()

            host_engines = [Path(trust(r)).read_bytes() for r in pool]
            # Same in-memory deserialization and execution for both policies.
            meter = EnergyCounter()
            assert meter.available, meter.error
            for batch in (1, *BATCHES):
                data = inputs[:batch]
                reference = []
                resident = []
                extra = 64 * MIB
                start_setup = time.perf_counter()
                for index, blob in enumerate(host_engines):
                    before = check_memory()
                    if before - len(blob) - extra < RESERVE:
                        break
                    runner = HostEngineRunner(blob, maximum_batch=batch)
                    value = logits(runner, data)
                    assert value.shape == (batch, 100) and np.isfinite(value).all()
                    after = torch.cuda.mem_get_info()[0]
                    if after < RESERVE:
                        del runner
                        release()
                        check_memory()
                        break
                    resident.append(runner)
                    del runner
                    reference.append(value.argmax(1))
                    extra = max(extra, before-after-len(blob))
                setup_seconds = time.perf_counter() - start_setup
                count = len(resident)
                resident.clear()
                release()
                if count != 8:
                    state['skipped'].append(dict(batch=batch, resident_count=count,
                                                reason='eight-model pool did not pass reserve admission'))
                    save()
                    continue
                expected = np.stack([reference[i] for i in TRACE * REPLAYS])
                ref_path = OUT / f'b{batch}_reference.npy'
                np.save(ref_path, expected, allow_pickle=False)
                for repeat in range(3):
                    for capacity in ((6, 8) if (repeat + batch) % 2 == 0 else (8, 6)):
                        cache = OrderedDict()
                        # Pre-allocate output so both modes have the same output boundary.
                        predictions = torch.empty(expected.shape, dtype=torch.int64, device='cuda')
                        def replay(route, output=None):
                            loads = evictions = 0
                            for position, i in enumerate(route):
                                guard.check()
                                if i in cache:
                                    cache.move_to_end(i)
                                else:
                                    if len(cache) == capacity:
                                        torch.cuda.synchronize()
                                        cache.popitem(last=False)
                                        evictions += 1
                                    free = check_memory()
                                    if free - len(host_engines[i]) - extra < RESERVE:
                                        torch.cuda.empty_cache()
                                        if check_memory() - len(host_engines[i]) - extra < RESERVE:
                                            raise RuntimeError('Cache admission would cross reserve')
                                    cache[i] = HostEngineRunner(host_engines[i], maximum_batch=batch)
                                    loads += 1
                                result = cache[i].run(data).argmax(1)
                                if output is not None:
                                    output[position].copy_(result)
                                # One synchronized batch at a time; count full miss overhead.
                                torch.cuda.synchronize()
                                check_memory()
                            return loads, evictions

                        guard.cool()
                        guard.phase = 'warmup'
                        cold_start = time.perf_counter()
                        warm_counts = replay(TRACE)
                        warm_seconds = time.perf_counter() - cold_start
                        phase = f'b{batch}:r{repeat}:c{capacity}'
                        guard.phase = phase
                        start = guard.reading()
                        meter.start()
                        started = time.perf_counter()
                        counts = replay(TRACE * REPLAYS, predictions)
                        seconds = time.perf_counter()-started
                        images = expected.size
                        energy = meter.finish(images)
                        end = guard.reading()
                        guard.phase = 'verification'
                        with guard.lock:
                            samples = [s for s in guard.samples if s['phase'] == phase]
                        observed = predictions.cpu().numpy()
                        out = OUT / f'{phase.replace(":", "_")}_predictions.npy'
                        np.save(out, observed, allow_pickle=False)
                        thermal = sum(bool(s['clock_reasons'] & 32) for s in (start, end, *samples))
                        mismatch = int(np.count_nonzero(observed != expected))
                        accepted_window = bool(not thermal and not mismatch and samples and
                                               energy.get('gpu_energy_j', 0) > 0 and
                                               counts == reference_cache_counts(capacity))
                        state['windows'].append(dict(batch=batch, repeat=repeat, capacity=capacity,
                            images=images, elapsed_s=seconds, ms_per_image=seconds*1000/images,
                            energy=energy, timed_loads=counts[0], timed_evictions=counts[1],
                            untimed_first_trace_s=warm_seconds, first_trace_loads=warm_counts[0],
                            first_trace_evictions=warm_counts[1], eight_engine_setup_s=setup_seconds,
                            thermal_flags=thermal, samples=len(samples), start=start, end=end,
                            prediction_mismatches=mismatch, accepted=accepted_window,
                            prediction_path=str(out), prediction_sha256=digest(out),
                            reference_path=str(ref_path), reference_sha256=digest(ref_path)))
                        print(f'Replay b{batch} r{repeat} c{capacity}: {seconds:.3f}s, loads={counts[0]}', flush=True)
                        cache.clear()
                        del predictions
                        release()
                        save()
                        if not accepted_window:
                            raise RuntimeError('Invalid replay window; evidence retained')
        for path, sha in protected.items():
            assert digest(path) == sha, path
        assert len(state['capacity']) == 9 and all(r['status'] == 'verified' for r in state['capacity'])
        assert len(state['windows']) == 6 * (4-len(state['skipped']))
        for r in state['windows']:
            assert digest(r['prediction_path']) == r['prediction_sha256']
            assert digest(r['reference_path']) == r['reference_sha256']
            assert np.array_equal(np.load(r['prediction_path'], allow_pickle=False),
                                  np.load(r['reference_path'], allow_pickle=False))
        state.update(status='complete', completed_utc=common.utc())
        save()
    except BaseException as exc:
        state.update(status='stopped', error=repr(exc))
        save()
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    main()
