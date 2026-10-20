"""Isolated no-router p90 controls at batches8/32/64; preserve batch128."""
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
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import make_batches
from pc_tests.model_pipeline.experiments.run_most_pruned import highest_pruned
from pc_tests.model_pipeline.experiments.run_random_routes import PIPELINE, digest, save

BATCH_SIZES = (8,32,64)
FOLDER = PIPELINE/'reports/most_pruned_b8_32_64'
REPORT = FOLDER/'most_pruned_routes.json'
PRE_READFIX_SHA256 = 'c4f27a729496c28701e11561a37a260d898cb1d144fd0e51c99bd7b3f07bf454'


def read_thermal_state(path, attempts=20, delay=.025):
    """Bounded retry for a concurrent guard rewrite; never use cached telemetry."""
    error = None
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding='utf-8-sig'))
        except (json.JSONDecodeError, PermissionError, UnicodeDecodeError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise RuntimeError('Thermal status unreadable after bounded retries; stopping safely') from error


def summarize(report):
    summaries=[]
    lines=['# Most-pruned static controls: batches8,32,64', '',
           f"Status: {report.get('status','partial').upper()}; {len(report['results'])}/330 sweeps.", '',
           '21 p90 configurations (7architectures x FP32/FP16/INT8), five10000-image repeats '
           'per batch size plus five R101 p0 FP32 references at each size. No router. '
           'Seed42+batch shuffle; real tail16 at32/64, no tail at8. Accuracy=repeat mean; '
           'latency and GPU-only NVML energy=repeat medians. Savings use the fresh R101 '
           'p0 FP32 reference at the SAME batch size, not eligible-static routing comparators. '
           'Cold capacity2 engine loading/hash/inference, RGB128 transfer/normalization and '
           'sync included; decode/resize/pinning excluded. All failures/negative savings '
           'retained. Prior GPU throttling and temporal drift limit strong cross-study '
           'latency/energy claims. Batch128 outputs are unchanged.', '',
           '| Batch | Architecture | Pruning | Precision | Accuracy | Loss pp | Passing allowances | ms/image | mJ/image | L saving % | E saving % |',
           '|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|']
    for size in BATCH_SIZES:
        rows=[r for r in report['results'] if r['batch_size']==size]
        refs=[r for r in rows if r['role']=='reference']
        if len(refs)!=5:
            continue
        base_acc=float(np.mean([r['accuracy'] for r in refs]))
        base_l=float(np.median([r['latency_ms_per_image'] for r in refs]))
        base_e=float(np.median([r['gpu_energy_j_per_image'] for r in refs]))
        for name in report['protocol']['models']:
            values=[r for r in rows if r['role']=='most_pruned' and r['name']==name]
            if len(values)!=5:
                continue
            accuracy=float(np.mean([r['accuracy'] for r in values]))
            latency=float(np.median([r['latency_ms_per_image'] for r in values]))
            energy=float(np.median([r['gpu_energy_j_per_image'] for r in values]))
            loss=base_acc-accuracy
            passes=[b for b in (.5,1.,1.5,2.) if loss<=b+1e-10]
            item=dict(name=name,batch_size=size,accuracy=accuracy,loss_pp=loss,
                      passing_allowances=passes,latency_ms_per_image=latency,
                      energy_mj_per_image=energy*1000,
                      latency_saving_percent=100*(base_l-latency)/base_l,
                      energy_saving_percent=100*(base_e-energy)/base_e,
                      cached_prediction_disagreements=sum(r['cached_prediction_disagreements'] for r in values))
            summaries.append(item)
            first=values[0]
            lines.append(f"| {size} | {first['architecture']} | p{first['prune']} | {first['precision'].upper()} | "
                         f"{accuracy:.2f}% | {loss:.2f} | {', '.join(f'{b:g}' for b in passes) or 'none'} | "
                         f"{latency:.3f} | {1000*energy:.3f} | {item['latency_saving_percent']:+.2f} | {item['energy_saving_percent']:+.2f} |")
    (FOLDER/'most_pruned_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(FOLDER/'most_pruned_summary.json',dict(complete=report.get('status')=='complete' and len(summaries)==63,
         completed_sweeps=len(report['results']),expected_sweeps=330,rows=summaries))
    print(f"Summary: {len(report['results'])}/330 sweeps; {len(summaries)}/63 model/batch rows",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--resume-after-read-fix',action='store_true')
    args=parser.parse_args()
    if args.summarize_only:
        summarize(json.loads(REPORT.read_text()))
        return
    static_path=PIPELINE/'reports/gpu_static.json'
    records=json.loads(static_path.read_text())['results']
    selected=highest_pruned(records)
    reference=min([r for r in records if r['prune']==0 and r['precision']=='fp32'],
                  key=lambda r:(-r['validation_accuracy'],r['name']))
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    old_path=PIPELINE/'reports/most_pruned_b128/most_pruned_routes.json'
    old=json.loads(old_path.read_text())
    if old.get('status')!='complete' or len(old['results'])!=110:
        raise RuntimeError('Batch128 predecessor not complete')
    protocol=dict(version=1,batch_sizes=list(BATCH_SIZES),models=[r['name'] for r in selected],
                  repeats=5,resident_capacity=2,reference=reference['name'],
                  static_sha256=digest(static_path),labels_sha256=digest(cache/'test_labels.npy'),
                  rgb_sha256=digest(cache/'test_rgb128.npy'),batch128_report_sha256=digest(old_path),
                  script_sha256=digest(Path(__file__)),
                  model_selection='Highest stored pruning per architecture; no test-based choice; all21 measured.',
                  timing='PinnedRGB128->transfer/normalization->fixed index->coldcache hash/load->inference->sync. warmup3,5repeats,seed42+batch,reversemodelorder on oddtrials.')
    report=json.loads(REPORT.read_text()) if REPORT.exists() else dict(protocol=protocol,results=[],expected_sweeps=330)
    if report['protocol']['script_sha256'] != protocol['script_sha256']:
        original = report['protocol']['script_sha256']
        current = protocol['script_sha256']
        revisions = report.get('code_revisions', [])
        verified_revision = any(r.get('from_sha256') == original and r.get('to_sha256') == current for r in revisions)
        backup = PIPELINE/'artifacts/most_pruned_b8_32_64/run_most_pruned_batches.before_readfix.py'
        expected = dict(protocol, script_sha256=original)
        if (original != PRE_READFIX_SHA256 or digest(backup) != original
                or report['protocol'] != expected
                or not (args.resume_after_read_fix or verified_revision)):
            raise RuntimeError('Unapproved source/protocol change; preserve results')
        if not verified_revision:
            revisions.append(dict(from_sha256=original, to_sha256=current,
                utc=datetime.now(timezone.utc).isoformat(), preserved_sweeps=len(report['results']),
                reason='User authorized continuation after fixing transient empty thermal JSON read. Bounded retry only; scientific inputs, models, batches, timing boundary and prior rows unchanged.'))
            report['code_revisions'] = revisions
        # Preserve the original protocol and explicitly record the repair lineage.
        protocol = expected
    if report['protocol']!=protocol:
        raise RuntimeError('Protocol/provenance changed; preserve report')
    save(REPORT,report)
    print('Prepared21models x3batches x5repeats +15references =330 sweeps',flush=True)
    if args.prepare_only:
        return
    from pc_tests.model_pipeline.routers.correctness.train_correctness_router import require_idle_gpu
    require_idle_gpu()
    state_path=PIPELINE/'artifacts/pc_routers/thermal_guard.json'
    def guard(allow_wait=False):
        state=read_thermal_state(state_path)
        if state['status'] not in (('monitoring','waiting_for_next_checkpoint') if allow_wait else ('monitoring',)):
            raise RuntimeError('Thermal state latched; no restart')
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(state['utc'].replace('Z','+00:00'))).total_seconds()
        if not 0<=age<=130 or state['expected_script']!=Path(__file__).name or os.getpid() not in state['tracked_process_ids']:
            raise RuntimeError('Fresh exact thermal guard required')
        return state
    deadline=time.monotonic()+120
    print(f'Waiting for thermal guard PID{os.getpid()}',flush=True)
    while True:
        try:
            guard()
            break
        except RuntimeError:
            if read_thermal_state(state_path)['status']!='monitoring' or time.monotonic()>deadline:
                raise
            time.sleep(1)
    import torch
    from pc_tests.model_pipeline.runtime.global_candidate_pool import EngineCache
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_engine_preflight import EngineRunner
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter
    torch.set_num_threads(1)
    images=np.load(cache/'test_rgb128.npy',mmap_mode='r',allow_pickle=False)
    labels=np.load(cache/'test_labels.npy',allow_pickle=False)
    assert images.shape==(10000,3,128,128) and labels.shape==(10000,)
    inputs=SharedBatchInputs()
    def release():
        gc.collect()
        torch.cuda.empty_cache()
    def load(record):
        if digest(record['engine_path'])!=record['engine_sha256']:
            raise RuntimeError('Engine hash failed')
        return EngineRunner(record['engine_path'])
    done={r['id'] for r in report['results']}
    for size in BATCH_SIZES:
        batches=make_batches(np.arange(10000),size,42+size)
        assert sum(map(len,batches))==10000 and len(np.unique(np.concatenate(batches)))==10000
        hosts=[torch.from_numpy(np.array(images[b])).pin_memory() for b in batches]
        for trial in range(5):
            order=[reference]+(selected if trial%2==0 else list(reversed(selected)))
            for record in order:
                identity=f"b{size}_{record['name']}_trial{trial}"
                if identity in done:
                    continue
                guard()
                engines=EngineCache([record],2,load,torch.cuda.synchronize,release)
                cached=np.load(PIPELINE/'artifacts/gpu_static'/(record['name']+'_test_predictions.npy'),allow_pickle=False)
                actual=np.empty(10000,dtype=np.int16)
                wall=[]
                def forward(index):
                    normalized=inputs.prepare(hosts[index])
                    return engines[0].run(normalized,copy_input=False).argmax(1).cpu().numpy()
                print(f'Start {identity} ({len(done)}/330)',flush=True)
                try:
                    with torch.inference_mode():
                        for i in range(3):
                            forward(i)
                        engines.clear()
                        engines.loads=engines.evictions=0
                        torch.cuda.synchronize()
                        energy=EnergyCounter()
                        if not energy.available:
                            raise RuntimeError('NVML energy unsupported')
                        energy.start()
                        for i,positions in enumerate(batches):
                            if i%10==0:
                                guard(allow_wait=True)
                            started=time.perf_counter()
                            actual[positions]=forward(i)
                            wall.append(time.perf_counter()-started)
                        metrics=energy.finish(10000)
                    row=dict(id=identity,status='complete',router='most_pruned_static',batch_size=size,
                             role='reference' if record['name']==reference['name'] else 'most_pruned',
                             name=record['name'],architecture=record['architecture'],prune=record['prune'],precision=record['precision'],trial=trial,
                             accuracy=float(100*np.mean(actual==labels)),latency_ms_per_image=float(1000*sum(wall)/10000),
                             elapsed_seconds=float(sum(wall)),cache_loads=engines.loads,cache_evictions=engines.evictions,
                             cached_prediction_disagreements=int(np.sum(actual!=cached)),
                             completed_utc=datetime.now(timezone.utc).isoformat(),**metrics)
                    report['results'].append(row)
                    save(REPORT,report)
                    done.add(identity)
                    print(f"Finished {identity}: accuracy={row['accuracy']:.2f}% latency={row['latency_ms_per_image']:.3f}ms/img gpu={metrics.get('gpu_temperature_c')}",flush=True)
                finally:
                    engines.clear()
                if guard(allow_wait=True)['status']!='monitoring':
                    print('Thermal checkpoint saved; waiting for guard shutdown',flush=True)
                    while True:
                        time.sleep(1)
        del hosts
        release()
        summarize(report)
    report['status']='complete'
    save(REPORT,report)
    summarize(report)
    print('COMPLETE: all330 sweeps at batches8/32/64',flush=True)


if __name__=='__main__':
    main()
