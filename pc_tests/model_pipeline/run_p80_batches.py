"""Isolated p80 static controls at batches8/32/64/128; never modify p90."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
from batch_router_policy import make_batches, split_validation
from run_most_pruned_batches import read_thermal_state
import subprocess
from run_random_routes import PIPELINE, digest, save

BATCH_SIZES = (8,32,64,128)
FOLDER = PIPELINE/'reports/p80_b8_32_64_128'
REPORT = FOLDER/'p80_routes.json'
def select_p80(records):
    """Explicit fixed pruning; no highest-pruning or test-score selection."""
    selected = []
    for architecture in sorted({r['architecture'] for r in records}):
        group = [r for r in records if r['architecture'] == architecture and r['prune'] == 80]
        if not group:
            if any(r['prune'] > 0 for r in records if r['architecture'] == architecture):
                raise ValueError('Pruned architecture missing p80')
            continue
        if len(group) != 3 or {r['precision'] for r in group} != {'fp32','fp16','int8'}:
            raise ValueError('p80 requires all three precisions')
        selected.extend(sorted(group, key=lambda r: ('fp32','fp16','int8').index(r['precision'])))
    if len(selected) != 21:
        raise ValueError('Expected exactly21 p80 configurations')
    return selected


def require_no_overlap():
    script = ("Get-CimInstance Win32_Process | Where-Object { "
              "$_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match "
              "'(run_random_routes(_b64)?|run_most_pruned(_batches)?|run_p80_batches|run_pc_routes|run_correctness_routes|"
              "gpu_static_sweep|train_pc_batch_routers|train_correctness_router)\\.py' } | "
              "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    output = subprocess.run(['powershell.exe','-NoProfile','-Command',script],
                            capture_output=True,text=True,check=True).stdout
    rows = json.loads(output) if output.strip() else []
    if isinstance(rows, dict):
        rows = [rows]
    others = [r for r in rows if r['ProcessId'] not in {os.getpid(),os.getppid()}]
    if others:
        raise RuntimeError('Other GPU study active; p80 must not overlap: '+str(others))


def validation_choices(records, selected, reference, cache):
    """Freeze all p80 choices using calibration labels and prior static costs only."""
    labels = np.load(cache/'validation_labels.npy', allow_pickle=False)
    predictions = {r['name']: np.load(PIPELINE/'artifacts/gpu_static' /
                   (r['name']+'_validation_predictions.npy'), allow_pickle=False)
                   for r in selected + [reference]}
    choices = {}
    for seed in (42,45,48):
        _, positions = split_validation(labels, seed)
        base = 100*np.mean(predictions[reference['name']][positions] == labels[positions])
        accuracy = {r['name']: 100*np.mean(predictions[r['name']][positions] == labels[positions]) for r in selected}
        for size in BATCH_SIZES:
            for objective, field in [('latency','normalized_host_input_to_logits_batch_ms'),
                                     ('energy','gpu_energy_j_per_image')]:
                for budget in (.5,1.,1.5,2.):
                    eligible = [(float(np.median([t[field] for t in r['timings'][str(size)]])),r['name'])
                                for r in selected if accuracy[r['name']] >= base-budget-1e-10]
                    choices[f'{seed}_b{size}_{objective}_{budget:g}'] = min(eligible)[1] if eligible else None
    return choices


def summarize(report):
    summaries=[]
    lines=['# Fixed p80 static controls: batches8,32,64,128', '',
           f"Status: {report.get('status','partial').upper()}; {len(report['results'])}/440 sweeps.", '',
           '21 p80 configurations (7architectures x FP32/FP16/INT8), five10000-image repeats '
           'per batch size plus five R101 p0 FP32 references at each size. No router. '
           'Seed42+batch shuffle; real tail16 at32/64/128, no tail at8. Accuracy=repeat mean; '
           'latency and GPU-only NVML energy=repeat medians. Savings use the fresh R101 '
           'p0 FP32 reference at the SAME batch size, not eligible-static routing comparators. '
           'Cold capacity2 engine loading/hash/inference, RGB128 transfer/normalization and '
           'sync included; decode/resize/pinning excluded. All failures/negative savings '
           'retained. Prior GPU throttling and temporal drift limit strong cross-study '
           'latency/energy claims. Prior p90 and random outputs are unchanged.', '',
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
    (FOLDER/'p80_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(FOLDER/'p80_summary.json',dict(complete=report.get('status')=='complete' and len(summaries)==84,
         completed_sweeps=len(report['results']),expected_sweeps=440,rows=summaries))
    print(f"Summary: {len(report['results'])}/440 sweeps; {len(summaries)}/84 model/batch rows",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.summarize_only:
        summarize(json.loads(REPORT.read_text()))
        return
    static_path=PIPELINE/'reports/gpu_static.json'
    records=json.loads(static_path.read_text())['results']
    selected=select_p80(records)
    reference=min([r for r in records if r['prune']==0 and r['precision']=='fp32'],
                  key=lambda r:(-r['validation_accuracy'],r['name']))
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    protected_paths = [
        PIPELINE/'reports/most_pruned_b128/most_pruned_routes.json',
        PIPELINE/'reports/most_pruned_b8_32_64/most_pruned_routes.json',
        PIPELINE/'reports/random_b128/random_routes_b128.json',
        PIPELINE/'reports/random_b64/random_routes_b64.json']
    predecessor = json.loads(protected_paths[-1].read_text())
    if predecessor.get('status') != 'complete' or len(predecessor['results']) != 495:
        raise RuntimeError('Batch64 random predecessor not complete; do not overlap')
    protected = {str(p.relative_to(PIPELINE)): digest(p) for p in protected_paths}
    protocol=dict(version=1,batch_sizes=list(BATCH_SIZES),models=[r['name'] for r in selected],
                  repeats=5,resident_capacity=2,reference=reference['name'],
                  static_sha256=digest(static_path),labels_sha256=digest(cache/'test_labels.npy'),
                  rgb_sha256=digest(cache/'test_rgb128.npy'),protected_reports=protected,
                  thermal_reader_sha256=digest(Path(__file__).with_name('run_most_pruned_batches.py')),
                  runtime_inputs_sha256=digest(Path(__file__).with_name('gpu_batch_inputs.py')),
                  engine_cache_sha256=digest(Path(__file__).with_name('global_candidate_pool.py')),
                  script_sha256=digest(Path(__file__)),
                  validation_selected=validation_choices(records, selected, reference, cache),
                  model_selection='Fixed p80 per user request; seven architectures x three precisions; no test-based selection.',
                  timing='PinnedRGB128->transfer/normalization->fixed index->coldcache hash/load->inference->sync. warmup3,5repeats,seed42+batch,reversemodelorder on oddtrials.')
    report=json.loads(REPORT.read_text()) if REPORT.exists() else dict(protocol=protocol,results=[],expected_sweeps=440)
    if report['protocol']!=protocol:
        raise RuntimeError('Protocol/provenance changed; preserve report')
    save(REPORT,report)
    print('Prepared21 p80 models x4batches x5repeats +20references =440 sweeps',flush=True)
    if args.prepare_only:
        return
    require_no_overlap()
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
    from global_candidate_pool import EngineCache
    from gpu_batch_inputs import SharedBatchInputs
    from gpu_engine_preflight import EngineRunner
    from gpu_energy import EnergyCounter
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
                print(f'Start {identity} ({len(done)}/440)',flush=True)
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
    print('COMPLETE: all440 p80 sweeps at batches8/32/64/128',flush=True)


if __name__=='__main__':
    main()
