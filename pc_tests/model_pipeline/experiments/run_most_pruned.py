"""No-router batch128 static controls: all seven p90 architectures x 3 precisions."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import make_batches, split_validation
from pc_tests.model_pipeline.experiments.run_random_routes import PIPELINE, digest, save


def highest_pruned(records):
    selected = []
    for architecture in sorted({r['architecture'] for r in records}):
        maximum = max(r['prune'] for r in records if r['architecture'] == architecture)
        if maximum == 0:
            continue
        group = [r for r in records if r['architecture'] == architecture and r['prune'] == maximum]
        if len(group) != 3 or {r['precision'] for r in group} != {'fp32','fp16','int8'}:
            raise ValueError('Highest pruning must have all three precisions')
        selected.extend(sorted(group, key=lambda r: ('fp32','fp16','int8').index(r['precision'])))
    if len(selected) != 21:
        raise ValueError('Expected seven pruned architectures x three precisions')
    return selected


def summarize(report, folder):
    rows = report['results']
    ref = [r for r in rows if r['role'] == 'reference']
    complete = report.get('status') == 'complete' and len(rows) == 110
    lines = ['# Most-pruned static controls: batch128', '',
             f"Status: {'COMPLETE' if complete else 'PARTIAL'}; {len(rows)}/110 sweeps saved.", '',
             'Each architecture uses its highest stored pruning level (p90), with FP32/FP16/INT8; '
             'ShuffleNet has no pruned variants and is excluded. No router. Five repeats of '
             '10000 held-out images; common seed42+128 batch ordering; real tail16; same RGB128 '
             'timing boundary and cold cache as the random study. Accuracy is the repeat mean, '
             'latency/energy the repeat medians. Savings below are versus freshly measured '
             'ResNet101 p0 FP32, NOT the eligible-static comparator in the routing matrix. '
             'Negative savings and all accuracy failures are retained. GPU-only NVML energy. '
             'These measurements follow the random study in time; cross-study timing comparisons '
             'remain subject to previously observed throttling and temporal drift.', '',
             '| Architecture | Pruning | Precision | Accuracy | Loss pp | Pass allowances (pp) | ms/image | mJ/image | L saving vs R101 FP32 | E saving vs R101 FP32 |',
             '|---|---:|---|---:|---:|---|---:|---:|---:|---:|']
    summaries = []
    if len(ref) == 5:
        reference_accuracy = np.mean([r['accuracy'] for r in ref])
        reference_latency = np.median([r['latency_ms_per_image'] for r in ref])
        reference_energy = np.median([r['gpu_energy_j_per_image'] for r in ref])
        for name in report['protocol']['models']:
            values = [r for r in rows if r['name'] == name and r['role'] == 'most_pruned']
            if len(values) != 5:
                continue
            accuracy = float(np.mean([r['accuracy'] for r in values]))
            latency = float(np.median([r['latency_ms_per_image'] for r in values]))
            energy = float(np.median([r['gpu_energy_j_per_image'] for r in values]))
            loss = float(reference_accuracy - accuracy)
            passes = [b for b in (.5,1.,1.5,2.) if loss <= b+1e-10]
            item = dict(name=name, accuracy=accuracy, loss_pp=loss, passing_allowances=passes,
                        latency_ms_per_image=latency, energy_mj_per_image=energy*1000,
                        latency_saving_percent=float(100*(reference_latency-latency)/reference_latency),
                        energy_saving_percent=float(100*(reference_energy-energy)/reference_energy),
                        prediction_disagreements=sum(r['cached_prediction_disagreements'] for r in values))
            summaries.append(item)
            first = values[0]
            lines.append(f"| {first['architecture']} | p{first['prune']} | {first['precision'].upper()} | {accuracy:.2f}% | {loss:.2f} | "
                         f"{', '.join(f'{b:g}' for b in passes) or 'none'} | {latency:.3f} | {energy*1000:.3f} | "
                         f"{item['latency_saving_percent']:+.2f}% | {item['energy_saving_percent']:+.2f}% |")
    lines += ['', 'Validation-only eligible choices among these21 (selected before test evaluation):',
              '', '```json', json.dumps(report['protocol']['validation_selected'], indent=2), '```']
    (folder/'most_pruned_table.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save(folder/'most_pruned_summary.json', dict(complete=complete, completed_sweeps=len(rows), rows=summaries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    folder = PIPELINE/'reports/most_pruned_b128'
    path = folder/'most_pruned_routes.json'
    if args.summarize_only:
        summarize(json.loads(path.read_text()), folder)
        return
    static_path = PIPELINE/'reports/gpu_static.json'
    records = json.loads(static_path.read_text())['results']
    selected = highest_pruned(records)
    reference = min([r for r in records if r['prune']==0 and r['precision']=='fp32'],
                    key=lambda r: (-r['validation_accuracy'], r['name']))
    cache = PIPELINE/'artifacts/pc_precision/data_cache'
    validation_labels = np.load(cache/'validation_labels.npy', allow_pickle=False)
    predictions_dir = PIPELINE/'artifacts/gpu_static'
    selected_choices = {}
    for seed in (42,45,48):
        _, positions = split_validation(validation_labels, seed)
        reference_predictions = np.load(predictions_dir/(reference['name']+'_validation_predictions.npy'), allow_pickle=False)
        base_accuracy = 100*np.mean(reference_predictions[positions] == validation_labels[positions])
        for objective, key in [('latency','normalized_host_input_to_logits_batch_ms'), ('energy','gpu_energy_j_per_image')]:
            for allowance in (.5,1.,1.5,2.):
                eligible = []
                for record in selected:
                    pred = np.load(predictions_dir/(record['name']+'_validation_predictions.npy'), allow_pickle=False)
                    if 100*np.mean(pred[positions]==validation_labels[positions]) >= base_accuracy-allowance-1e-10:
                        eligible.append((float(np.median([r[key] for r in record['timings']['128']])), record['name']))
                selected_choices[f'{seed}_{objective}_{allowance:g}'] = min(eligible)[1] if eligible else None
    protocol = dict(version=1, batch_size=128, models=[r['name'] for r in selected], repeats=5,
                    reference=reference['name'], static_sha256=digest(static_path),
                    labels_sha256=digest(cache/'test_labels.npy'), rgb_sha256=digest(cache/'test_rgb128.npy'),
                    script_sha256=digest(Path(__file__)), validation_selected=selected_choices,
                    timing='Same pinned RGB128 -> normalization -> fixed model index -> cold capacity2 hash/load -> inference -> sync; no router; warmup3, per-repeat reference, model order reversed on odd trials.')
    report = json.loads(path.read_text()) if path.exists() else dict(protocol=protocol, results=[], expected_sweeps=110)
    if report['protocol'] != protocol:
        raise RuntimeError('Protocol changed: preserve existing results')
    save(path, report)
    print('Prepared21 highest-pruning static models, 110 sweeps including reference', flush=True)
    if args.prepare_only:
        return
    predecessor = json.loads((PIPELINE/'reports/random_b128/random_routes_b128.json').read_text())
    if predecessor.get('status') != 'complete' or len(predecessor['results']) != 495:
        raise RuntimeError('Random comparison not complete; do not overlap')
    from pc_tests.model_pipeline.routers.correctness.train_correctness_router import require_idle_gpu
    require_idle_gpu()
    state_path = PIPELINE/'artifacts/pc_routers/thermal_guard.json'
    def guard(allow_wait=False):
        state = json.loads(state_path.read_text(encoding='utf-8-sig'))
        if state['status'] not in (('monitoring','waiting_for_next_checkpoint') if allow_wait else ('monitoring',)):
            raise RuntimeError('Thermal state latched; never restart')
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(state['utc'].replace('Z','+00:00'))).total_seconds()
        if not 0 <= age <= 130 or state['expected_script'] != Path(__file__).name or os.getpid() not in state['tracked_process_ids']:
            raise RuntimeError('Fresh exact thermal guard required')
        return state
    deadline = time.monotonic()+120
    print(f'Waiting for exact most-pruned thermal guard PID{os.getpid()}', flush=True)
    while True:
        try:
            guard()
            break
        except RuntimeError:
            if json.loads(state_path.read_text(encoding='utf-8-sig'))['status'] != 'monitoring' or time.monotonic()>deadline:
                raise
            time.sleep(1)
    import torch
    from pc_tests.model_pipeline.runtime.global_candidate_pool import EngineCache
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    torch.set_num_threads(1)
    images = np.load(cache/'test_rgb128.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(cache/'test_labels.npy', allow_pickle=False)
    assert images.shape == (10000,3,128,128) and labels.shape == (10000,)
    batches = make_batches(np.arange(10000),128,42+128)
    hosts = [torch.from_numpy(np.array(images[b])).pin_memory() for b in batches]
    inputs = SharedBatchInputs()
    def release():
        gc.collect()
        torch.cuda.empty_cache()
    def load(record):
        if digest(record['engine_path']) != record['engine_sha256']:
            raise RuntimeError('Engine hash mismatch')
        return EngineRunner(record['engine_path'])
    done = {r['id'] for r in report['results']}
    for trial in range(5):
        order = [reference] + (selected if trial%2==0 else list(reversed(selected)))
        for record in order:
            identity = f"{record['name']}_trial{trial}"
            if identity in done:
                continue
            guard()
            engines = EngineCache([record],2,load,torch.cuda.synchronize,release)
            cached = np.load(predictions_dir/(record['name']+'_test_predictions.npy'), allow_pickle=False)
            actual = np.empty(10000,dtype=np.int16)
            wall = []
            def forward(index):
                normalized = inputs.prepare(hosts[index])
                return engines[0].run(normalized,copy_input=False).argmax(1).cpu().numpy()
            print(f'Start {identity} ({len(done)}/110)', flush=True)
            try:
                with torch.inference_mode():
                    for i in range(3):
                        forward(i)
                    engines.clear()
                    engines.loads=engines.evictions=0
                    torch.cuda.synchronize()
                    energy=EnergyCounter()
                    if not energy.available:
                        raise RuntimeError('NVML energy unavailable')
                    energy.start()
                    for i, positions in enumerate(batches):
                        if i%10==0:
                            guard(allow_wait=True)
                        started=time.perf_counter()
                        actual[positions]=forward(i)
                        wall.append(time.perf_counter()-started)
                    metrics=energy.finish(10000)
                row=dict(id=identity, status='complete', router='most_pruned_static',
                         role='reference' if record['name']==reference['name'] else 'most_pruned',
                         name=record['name'], architecture=record['architecture'],prune=record['prune'],precision=record['precision'],trial=trial,
                         accuracy=float(100*np.mean(actual==labels)),latency_ms_per_image=float(1000*sum(wall)/10000),
                         elapsed_seconds=float(sum(wall)),cache_loads=engines.loads,cache_evictions=engines.evictions,
                         cached_prediction_disagreements=int(np.sum(actual!=cached)),
                         completed_utc=datetime.now(timezone.utc).isoformat(),**metrics)
                report['results'].append(row)
                save(path,report)
                done.add(identity)
                print(f"Finished {identity}: accuracy={row['accuracy']:.2f}% latency={row['latency_ms_per_image']:.3f}ms/img gpu={metrics.get('gpu_temperature_c')}",flush=True)
            finally:
                engines.clear()
            if guard(allow_wait=True)['status']!='monitoring':
                print('Thermal checkpoint saved; awaiting guard shutdown',flush=True)
                while True:
                    time.sleep(1)
    report['status']='complete'
    save(path,report)
    summarize(report,folder)
    print('COMPLETE: most-pruned static comparison110/110',flush=True)


if __name__=='__main__':
    main()
