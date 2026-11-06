"""Build the new 213-variant bank, fit routing, and benchmark matched residency modes."""
from pathlib import Path
import gc
import json
import os
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from pc_tests.model_pipeline.experiments import compare_dispatch_residency_v2 as telemetry
from pc_tests.model_pipeline.experiments.run_precision_experiments import file_hash, write_json
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.routers.correctness.correctness_policy import calibrate_correctness
from pc_tests.model_pipeline.routers.correctness.correctness_router import CorrectnessRouter
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
from pc_tests.model_pipeline.runtime.global_candidate_pool import EngineCache
from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner, build_engine, trt
from pc_tests.model_pipeline.runtime.engine_precision_audit import integer_compute_layers


POOL = PIPELINE / 'reports/finetuned_routing_20261106/precision_pool_213'
OUT = POOL / 'pc_residency_benchmark'
DATA = PIPELINE / 'artifacts/pc_precision/data_cache'
BATCHES = (1, 8, 16, 32, 64, 128)
MODES = ('cache2', 'cache8', 'static_fp16', 'static_fp32')
RESERVE = 1024**3


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def wait_for_graphs():
    # Preparation is a dependency, not a timing window. Exit on any upstream error.
    deadline = time.monotonic() + 8*3600
    while time.monotonic() < deadline:
        state = read(POOL / 'quantization_status.json')
        if state['status'] == 'complete':
            assert len(state['completed']) == 63 and len(state['unpruned']) == 24
            return state
        if state['status'] != 'running':
            raise RuntimeError('INT8 preparation did not complete: ' + str(state.get('error')))
        time.sleep(30)
    raise TimeoutError('INT8 preparation did not finish within the dependency window')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    state = dict(status='waiting_for_int8', pid=os.getpid(), expected_candidates=213,
                 built=[], screened=[], windows=[], seed=42, batches=BATCHES,
                 modes=MODES, measurement_images=1024,
                 boundary='pinned host RGB128 to host predicted classes, with live TinyCNN on routed paths',
                 energy='gross NVML GPU energy; not whole-PC energy')

    def save():
        write_json(OUT / 'status.json', state)

    guard = None
    save()
    try:
        source = wait_for_graphs()
        state.update(status='gpu_build_and_validation')
        save()
        telemetry.OUT = OUT
        guard = telemetry.Guard()
        guard.start()
        torch.manual_seed(42)
        torch.set_num_threads(1)
        rng = np.random.default_rng(42)

        def check():
            guard.check()
            if torch.cuda.mem_get_info()[0] < RESERVE:
                raise RuntimeError('Less than 1 GiB free VRAM')

        def cool():
            guard.phase = 'cooldown'
            deadline = time.monotonic() + 180
            while True:
                check()
                row = guard.reading()
                if row['temperature_c'] <= 65 and not row['clock_reasons'] & 96:
                    return
                if time.monotonic() >= deadline:
                    raise RuntimeError('GPU failed to cool for a qualified window')
                time.sleep(1)

        def release():
            gc.collect()
            torch.cuda.empty_cache()

        records = list(source['unpruned'])
        for row in source['completed']:
            for precision in ('fp32', 'fp16', 'int8'):
                is_int = precision == 'int8'
                graph = Path(row['int8_source' if is_int else 'fp_source'])
                assert file_hash(graph) == row['int8_sha256' if is_int else 'fp_sha256']
                name = f"{row['name']}__{precision}"
                target = OUT / 'engines' / f'{name}.engine'
                target.parent.mkdir(exist_ok=True)
                cool()
                state['current'] = name
                save()
                guard.phase = 'build:' + name
                build_engine(graph, target, precision)
                check()
                if is_int:
                    runner = EngineRunner(target)
                    layers = integer_compute_layers(runner.engine.create_engine_inspector().get_engine_information(
                        trt.LayerInformationFormat.JSON))
                    del runner
                    release()
                    if not layers:
                        raise RuntimeError(f'{name}: no verified INT8 compute layers')
                record = dict(name=name, engine=str(target), sha256=file_hash(target))
                records.append(record)
                state['built'].append(record)
                save()
                print(f"Built {len(state['built'])}/189: {name}", flush=True)
        records.sort(key=lambda r: r['name'])
        names = [r['name'] for r in records]
        assert len(names) == len(set(names)) == 213
        for row in records:
            assert file_hash(row['engine']) == row['sha256']
        baseline_fp32 = names.index('resnet101__p0__fp32')
        baseline_fp16 = names.index('resnet101__p0__fp16')
        images = np.load(DATA / 'validation_rgb128.npy', mmap_mode='r', allow_pickle=False)
        labels = np.load(DATA / 'validation_labels.npy', allow_pickle=False)
        assert images.shape == (5000, 3, 128, 128) and np.all(np.bincount(labels) == 50)
        fit, cal = split_validation(labels, 42)
        assert len(fit) == 3000 and len(cal) == 2000 and not set(fit) & set(cal)
        state.update(fit_positions=fit.tolist(), calibration_positions=cal.tolist(), candidates=names)
        save()
        inputs = SharedBatchInputs()
        host = [torch.from_numpy(np.array(images[i:i+64])).pin_memory() for i in range(0, len(labels), 64)]
        correct = np.empty((len(labels), len(names)), dtype=bool)
        costs = {b: [] for b in BATCHES}
        with torch.inference_mode():
            for idx, row in enumerate(records):
                cool()
                guard.phase = 'validation:' + row['name']
                runner = EngineRunner(Path(row['engine']))
                results = []
                for h in host:
                    x = inputs.prepare(h)
                    logits = runner.run(x, copy_input=False)
                    assert torch.isfinite(logits).all().item()
                    results.extend(logits.argmax(1).tolist())
                    check()
                correct[:, idx] = np.array(results) == labels
                # Short validation-only timings estimate costs for policy fitting.
                # They are not reported as qualified performance measurements.
                h128 = torch.from_numpy(np.array(images[cal[:128]])).pin_memory()
                for b in BATCHES:
                    tiles = h128.split(b)
                    runner.run(inputs.prepare(tiles[0]), copy_input=False).argmax(1).cpu()
                    start = time.perf_counter()
                    for tile in tiles:
                        runner.run(inputs.prepare(tile), copy_input=False).argmax(1).cpu()
                    costs[b].append((time.perf_counter()-start) / len(tiles))
                    check()
                state['screened'].append(row['name'])
                save()
                del runner
                release()
                print(f'Validation {idx+1}/213: {row["name"]}', flush=True)
        np.save(OUT / 'validation_correct.npy', correct, allow_pickle=False)
        state.update(status='router_fit', calibration_cost_seconds=costs)
        save()
        router = CorrectnessRouter(len(names)).cuda()
        optimizer = torch.optim.Adam(router.parameters(), lr=.001)
        for epoch in range(20):
            for selection in np.array_split(rng.permutation(fit), 48):
                check()
                x = inputs.prepare(torch.from_numpy(np.array(images[selection])))
                target = torch.from_numpy(correct[selection]).cuda().float()
                optimizer.zero_grad(set_to_none=True)
                loss = binary_cross_entropy_with_logits(router(x), target)
                assert torch.isfinite(loss).item()
                loss.backward()
                optimizer.step()
            print(f'Router epoch {epoch+1}/20', flush=True)
        router.eval()
        torch.save(router.state_dict(), OUT / 'router_weights.pth')
        policies = {}
        with torch.inference_mode():
            for b in BATCHES:
                groups = [cal[i:i+b] for i in range(0, len(cal), b)]
                q = np.concatenate([router(inputs.prepare(torch.from_numpy(np.array(images[g]))))
                                    .sigmoid().mean(0, keepdim=True).cpu().numpy() for g in groups])
                policies[b] = calibrate_correctness(q, np.array([correct[g].sum(0) for g in groups]),
                                                    [len(g) for g in groups], costs[b], baseline_fp32, 1.)
        state.update(status='measurement', policies=policies)
        save()

        # The test subset is fixed without reading any test results; never calibrate on it.
        test_positions = np.random.default_rng(4102).permutation(10000)[:1024]
        raw = np.load(DATA / 'test_rgb128.npy', mmap_mode='r', allow_pickle=False)
        truth = np.load(DATA / 'test_labels.npy', allow_pickle=False)[test_positions]
        assert raw.shape == (10000, 3, 128, 128)
        data = torch.from_numpy(np.array(raw[test_positions])).pin_memory()
        state['test_positions'] = test_positions.tolist()
        energy = EnergyCounter()
        if not energy.available:
            raise RuntimeError('NVML GPU energy counter unavailable')

        def make(row):
            runner = EngineRunner(Path(row['engine']))
            check()
            return runner

        with torch.inference_mode():
            for b in BATCHES:
                groups = data.split(b)
                p = policies[b]
                bias = torch.tensor(costs[b], device='cuda', dtype=torch.float32)
                bias *= p['penalty'] / bias.max()
                for repeat in range(3):
                    reference = None
                    for mode in MODES[repeat:] + MODES[:repeat]:
                        cool()
                        cache = EngineCache(records, 8 if mode == 'cache8' else 2,
                                            make, torch.cuda.synchronize, release)

                        def forward(h):
                            x = inputs.prepare(h)
                            if mode.startswith('static'):
                                index = baseline_fp16 if mode == 'static_fp16' else baseline_fp32
                            elif p['fallback']:
                                index = baseline_fp32
                            else:
                                index = int((router(x).sigmoid().mean(0) - bias).argmax().item())
                            return cache[index].run(x, copy_input=False).argmax(1).cpu().numpy(), index

                        for h in groups[:8]:
                            forward(h)
                            check()
                        token = f'{mode}:b{b}:r{repeat}'
                        guard.phase = token
                        start_reading = guard.reading()
                        load, evict = cache.loads, cache.evictions
                        energy.start()
                        start = time.perf_counter()
                        result, selected = [], []
                        for h in groups:
                            out, index = forward(h)
                            result.append(out)
                            selected.append(index)
                            check()
                        elapsed = time.perf_counter() - start
                        joules = energy.finish(len(truth))
                        end_reading = guard.reading()
                        guard.phase = 'verification'
                        samples = [start_reading, end_reading, *(r for r in guard.samples if r['phase'] == token)]
                        predictions = np.concatenate(result)
                        if mode.startswith('cache'):
                            if reference is None:
                                reference = (selected, predictions)
                            elif selected != reference[0] or not np.array_equal(predictions, reference[1]):
                                raise RuntimeError('Cache-only comparison changed classifier selections or predictions')
                        accepted = (len(samples) > 2 and not any(r['clock_reasons'] & 96 for r in samples)
                                    and joules.get('gpu_energy_j_per_image') is not None)
                        state['windows'].append(dict(batch=b, repeat=repeat, mode=mode, accepted=accepted,
                                                     cache_capacity=cache.capacity, loads=cache.loads-load,
                                                     evictions=cache.evictions-evict, ms_per_image=1000*elapsed/len(truth),
                                                     gpu_energy=joules, test_subset_accuracy=100*float((predictions==truth).mean()),
                                                     selected_counts={names[i]: selected.count(i) for i in set(selected)},
                                                     peak_c=max(r['temperature_c'] for r in samples)))
                        save()
                        cache.clear()
                        print(f'{token} qualified={accepted}', flush=True)
        state['status'] = 'complete' if all(w['accepted'] for w in state['windows']) else 'completed_with_rejected_windows'
    except BaseException as exc:
        state.update(status='stopped', error=repr(exc))
        raise
    finally:
        if guard is not None:
            guard.close()
        save()


if __name__ == '__main__':
    main()
