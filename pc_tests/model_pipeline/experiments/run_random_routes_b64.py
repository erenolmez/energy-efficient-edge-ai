"""Batch-64 random controls, isolated reports and checkpointed GPU timings."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import make_batches
from pc_tests.model_pipeline.experiments.random_route_baselines import random_schedule, selection_by_length, cached_accuracy
from pc_tests.model_pipeline.experiments.run_most_pruned_batches import read_thermal_state


ROOT = REPOSITORY_ROOT
PIPELINE = ROOT / 'pc_tests' / 'model_pipeline'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    for attempt in range(16):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 15:
                raise
            time.sleep(.25)


def frozen_inputs():
    source_path = PIPELINE / 'reports/pc_routes_correctness.json'
    source = json.loads(source_path.read_text())
    static_path = PIPELINE / 'reports/gpu_static.json'
    records = {r['name']: r for r in json.loads(static_path.read_text())['results']}
    names = source['models']
    assert len(names) == 213 and set(names) == set(records)
    cases = [r for r in source['results'] if r['batch_size'] == 64 and r['status'] == 'complete']
    assert len(cases) == 24
    # Start with a predeclared large-allowance case; never select by test wins.
    cases.sort(key=lambda r: (r['seed'], r['objective'] != 'latency', -r['budget_pp']))
    cache = PIPELINE / 'artifacts/pc_precision/data_cache'
    labels = np.load(cache / 'test_labels.npy', allow_pickle=False)
    assert labels.shape == (10000,)
    predictions = np.stack([np.load(PIPELINE / 'artifacts/gpu_static' / (n + '_test_predictions.npy'),
                                   allow_pickle=False) for n in names])
    for index, name in enumerate(names):
        assert abs(100 * np.mean(predictions[index] == labels) - records[name]['test_accuracy']) < 1e-8
    protocol = dict(version=1, batch_size=64, candidates=names, resident_capacity=2, repeats=5,
                    source_sha256=digest(source_path), static_sha256=digest(static_path),
                    labels_sha256=digest(cache / 'test_labels.npy'),
                    protected_b128_sha256=digest(PIPELINE / 'reports/random_b128/random_routes_b128.json'),
                    thermal_reader_sha256=digest(pipeline_source('run_most_pruned_batches.py')),
                    script_sha256=digest(Path(__file__)), helper_sha256=digest(pipeline_source('random_route_baselines.py')),
                    randomization='Uniform over all213; matched permutes source model counts within actual batch lengths. No label-based choices.',
                    timing='Same pinned RGB128 batches: GPU transfer/normalization, decision (fresh CNN or random schedule lookup), engine hash/load/eviction, inference, sync/host predictions. Random schedule generation and input pinning excluded; no cached features or learned decisions.',
                    measurement='One full 10000-image sweep per block, 5 trials, cold capacity2 per block; warmup3; fresh paired static and learned controls interleaved by trial. GPU-only NVML energy.',
                    historical='Original reports remain controls, never overwritten. Frequency counts fixed from original experiment; freshly measured learned counts checked against source.')
    return source, cases, records, cache, labels, predictions, protocol


def task_list(cases):
    tasks = []
    for trial in range(5):
        seen_uniform = set()
        for case in cases:
            # Alternating order reduces fixed-order timing bias.
            roles = ['reference', 'static', 'learned', 'matched']
            if trial % 2:
                roles.reverse()
            for role in roles:
                tasks.append((case, role, trial))
            if case['seed'] not in seen_uniform:
                tasks.append((case, 'uniform', trial))
                seen_uniform.add(case['seed'])
    return tasks


def schedule_for(case, role, trial, batches, names):
    lengths = list(map(len, batches))
    seed = 20261019 + case['seed'] * 100 + trial
    if role == 'uniform':
        return random_schedule(lengths, len(names), seed)
    if role == 'matched':
        counts = selection_by_length(case['measurements'][0], lengths)
        return random_schedule(lengths, len(names), seed + 1000000, counts)
    name = case['baseline'] if role == 'reference' else case['best_eligible_static']
    return np.full(len(batches), names.index(name), dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--max-blocks', type=int)
    args = parser.parse_args()
    output = PIPELINE / 'reports/random_b64'
    source, cases, by_name, cache, labels, predictions, protocol = frozen_inputs()
    names = protocol['candidates']
    report_path = output / 'random_routes_b64.json'
    tasks = task_list(cases)
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(protocol=protocol, results=[])
    if report['protocol'] != protocol:
        raise RuntimeError('Protocol/provenance changed: preserve prior report')
    report['expected_blocks'] = len(tasks)
    batches_by_seed = {seed: make_batches(np.arange(10000), 64, seed + 64) for seed in (42,45,48)}
    offline = []
    for case in cases:
        for role in ('uniform', 'matched'):
            for trial in range(10):
                batches = batches_by_seed[case['seed']]
                choices = schedule_for(case, role, trial, batches, names)
                offline.append(dict(case=case['id'], role=role, trial=trial,
                                    accuracy=cached_accuracy(predictions, labels, batches, choices)))
    save(output / 'cached_accuracy.json', dict(protocol=protocol, trials=offline,
         note='Ten prespecified randomizations; no GPU timing claims. Uniform trials repeat across case references with the same seed.'))
    save(report_path, report)
    print(f'Prepared cached accuracy: {len(offline)} comparisons, {len(tasks)} GPU blocks planned', flush=True)
    if args.prepare_only:
        return
    state_path = PIPELINE / 'artifacts/pc_routers/thermal_guard.json'
    def guard(allow_wait=False):
        state = read_thermal_state(state_path)
        valid = ('monitoring', 'waiting_for_next_checkpoint') if allow_wait else ('monitoring',)
        if state['status'] not in valid:
            raise RuntimeError('Thermal state latched; no restart')
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(state['utc'].replace('Z','+00:00'))).total_seconds()
        if not 0 <= age <= 130 or state['expected_script'] != Path(__file__).name or os.getpid() not in state['tracked_process_ids']:
            raise RuntimeError('Fresh exact thermal guard required')
        return state
    deadline = time.monotonic() + 120
    print(f'Waiting for thermal guard PID {os.getpid()}', flush=True)
    while True:
        try:
            guard()
            break
        except RuntimeError:
            if read_thermal_state(state_path)['status'] != 'monitoring' or time.monotonic() > deadline:
                raise
            time.sleep(1)
    import torch
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    from pc_tests.model_pipeline.runtime.global_candidate_pool import EngineCache
    from pc_tests.model_pipeline.runtime.router_tensorrt import RouterRunner
    from pc_tests.model_pipeline.routers.correctness.correctness_policy import select_correctness
    torch.set_num_threads(1)
    images = np.load(cache / 'test_rgb128.npy', mmap_mode='r', allow_pickle=False)
    assert images.shape == (10000,3,128,128) and images.dtype == np.uint8
    if digest(cache / 'test_rgb128.npy') != source['test_hashes']['test_rgb128.npy']:
        raise RuntimeError('Held-out image hash changed')
    inputs = SharedBatchInputs()
    records = [by_name[n] for n in names]
    def release():
        gc.collect()
        torch.cuda.empty_cache()
    def load(record):
        if digest(record['engine_path']) != record['engine_sha256']:
            raise RuntimeError('Classifier engine hash failed')
        return EngineRunner(record['engine_path'])
    done = {r['id'] for r in report['results'] if r['status'] == 'complete'}
    host_seed, hosts, routers = None, None, {}
    finished = 0
    for case, role, trial in tasks:
        identity = (f"uniform_seed{case['seed']}" if role == 'uniform' else case['id'] + '_' + role) + f'_trial{trial}'
        if identity in done:
            continue
        guard()
        seed = case['seed']
        batches = batches_by_seed[seed]
        if host_seed != seed:
            hosts = None
            release()
            hosts = [torch.from_numpy(np.array(images[b])).pin_memory() for b in batches]
            host_seed = seed
        schedule = schedule_for(case, role, trial, batches, names)
        if role == 'learned' and seed not in routers:
            marker_path = PIPELINE / 'artifacts/pc_routers_correctness' / f'global_correctness_seed{seed}.json'
            if digest(marker_path) != source['training_markers'][marker_path.name]:
                raise RuntimeError('Frozen router marker hash mismatch')
            marker = json.loads(marker_path.read_text())
            assert marker['models'] == names
            routers[seed] = {}
            for size in ('64','16'):
                item = marker['engines'][size]
                if digest(item['engine']) != item['sha256'] or item['torch_max_error'] > 1e-5:
                    raise RuntimeError('Router engine parity/hash failure')
                routers[seed][int(size)] = RouterRunner(item['engine'])
        engines = EngineCache(records, 2, load, torch.cuda.synchronize, release)
        policy = case['policy']
        marker = None
        if role == 'learned':
            marker = json.loads((PIPELINE / 'artifacts/pc_routers_correctness' / f'global_correctness_seed{seed}.json').read_text())
            costs = marker['policies'][case['objective']]['64']['costs']
        selected_batches = np.zeros(213, dtype=int)
        selected_images = np.zeros(213, dtype=int)
        actual = np.empty(10000, dtype=np.int16)
        mismatch = 0
        wall_ms = []
        def forward(index):
            normalized = inputs.prepare(hosts[index])
            if role == 'learned':
                probabilities = routers[seed][len(normalized)].run(normalized.contiguous()).cpu().numpy().copy()
                if probabilities.shape != (1,213):
                    raise RuntimeError('Router must output all 213 candidates')
                chosen = names.index(case['baseline']) if policy['fallback'] else int(select_correctness(probabilities, costs, policy['penalty'])[0])
            else:
                chosen = int(schedule[index])
            result = engines[chosen].run(normalized, copy_input=False).argmax(1).cpu().numpy()
            return chosen, result
        print(f'Start {identity} ({len(done)}/{len(tasks)})', flush=True)
        try:
            with torch.inference_mode():
                for index in range(3):
                    forward(index)
                engines.clear()
                engines.loads = engines.evictions = 0
                torch.cuda.synchronize()
                energy = EnergyCounter()
                if not energy.available:
                    raise RuntimeError('NVML energy unsupported: ' + str(energy.error))
                energy.start()
                for index, positions in enumerate(batches):
                    if index % 10 == 0:
                        guard(allow_wait=True)
                    started = time.perf_counter()
                    chosen, result = forward(index)
                    wall_ms.append((time.perf_counter() - started) * 1000)
                    actual[positions] = result
                    selected_batches[chosen] += 1
                    selected_images[chosen] += len(positions)
                    mismatch += int(np.sum(result != predictions[chosen, positions]))
                metrics = energy.finish(10000)
            accuracy = float(100 * np.mean(actual == labels))
            row = dict(id=identity, status='complete', router=role, source_case=case['id'], seed=seed, trial=trial,
                       objective=case['objective'] if role != 'uniform' else None,
                       budget_pp=case['budget_pp'] if role != 'uniform' else None, batch_size=64,
                       accuracy=accuracy, latency_ms_per_image=float(sum(wall_ms)/10000),
                       elapsed_seconds=float(sum(wall_ms)/1000), batches=157,
                       selected_batches=selected_batches.tolist(), selected_images=selected_images.tolist(),
                       cache_loads=engines.loads, cache_evictions=engines.evictions,
                       cached_prediction_disagreements=mismatch,
                       random_seed=20261019+seed*100+trial+(1000000 if role=='matched' else 0),
                       completed_utc=datetime.now(timezone.utc).isoformat(), **metrics)
            if role in ('uniform','matched'):
                row['cached_expected_accuracy'] = cached_accuracy(predictions, labels, batches, schedule)
                if not np.array_equal(np.bincount(schedule, minlength=213), selected_batches):
                    raise RuntimeError('Random schedule not actually executed')
            if role == 'matched':
                assert row['selected_images'] == case['measurements'][0]['selected_images']
            if role == 'learned':
                row['selection_counts_match_original'] = row['selected_images'] == case['measurements'][0]['selected_images']
            report['results'].append(row)
            save(report_path, report)
            done.add(identity)
            print(f"Finished {identity}: accuracy={accuracy:.2f}% latency={row['latency_ms_per_image']:.3f}ms/img loads={engines.loads} gpu={metrics.get('gpu_temperature_c')}", flush=True)
        finally:
            engines.clear()
        finished += 1
        if guard(allow_wait=True)['status'] != 'monitoring':
            print('Thermal boundary saved; waiting for guard-controlled shutdown', flush=True)
            while True:
                time.sleep(1)
        if args.max_blocks and finished >= args.max_blocks:
            print('Pilot block limit reached; saved checkpoints preserved', flush=True)
            return
    report['status'] = 'complete'
    save(report_path, report)
    print('COMPLETE: all batch64 random comparisons measured', flush=True)


if __name__ == '__main__':
    main()
