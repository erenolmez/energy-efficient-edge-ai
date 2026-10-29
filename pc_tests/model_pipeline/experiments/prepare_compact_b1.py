"""Queued CPU-only fit/calibration preparation; refuses overlap with PC timing."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

import json
from pathlib import Path
import subprocess
import time
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation,digest
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
from pc_tests.model_pipeline.routers.correctness.compact_cpu import METHODS,FEATURE_NAMES,features,thumbnails,fit


def require_predecessor_done():
    state=json.loads((PIPELINE/'artifacts/pc_routers/thermal_guard.json').read_text(encoding='utf-8-sig'))
    if state.get('status')!='monitoring' or state.get('boundary_start_count') is not None:
        raise RuntimeError('Thermal latch: no experiment or training restart')
    previous=json.loads((PIPELINE/'reports/correctness_b1_20261028/study.json').read_text())
    if previous.get('status')!='complete':
        receipt_path=PIPELINE/'reports/correctness_b1_20261028/stop_receipt.json'
        if not receipt_path.exists():
            raise RuntimeError('Original batch1 comparison still active/incomplete; no real fits during timing')
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('status')!='stopped_by_user' or digest(PIPELINE/'reports/correctness_b1_20261028/study.json')!=receipt['study_sha256']:
            raise RuntimeError('Cancelled predecessor receipt is invalid; never resume it')
        residency=PIPELINE/'reports/gpu_residency_20261028/study.json'
        if not residency.exists() or json.loads(residency.read_text()).get('status')!='complete':
            raise RuntimeError('User-prioritized residency probe must finish first')
    command="Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and (($_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match 'energy-efficient-edge-ai.*(run_|train_|measure_|probe_gpu_residency)') -or ($_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'watch_pc_thermals.ps1')) } | Select-Object ProcessId | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    if result.stdout.strip():
        raise RuntimeError('Another measurement/guard is active; defer CPU fits')


def save(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    require_predecessor_done()
    import joblib
    from threadpoolctl import threadpool_limits
    output=PIPELINE/'artifacts/compact_b1_20261028'
    if output.exists():
        raise RuntimeError('Previous preparation preserved: inspect before any restart')
    cache=PIPELINE/'artifacts/pc_precision/data_cache'
    records,labels,correct,baseline,provenance=load_validation(PIPELINE/'reports/gpu_static.json',PIPELINE/'artifacts/gpu_static',cache)
    images=np.load(cache/'validation_rgb128.npy',mmap_mode='r',allow_pickle=False)
    if images.shape!=(5000,3,128,128):
        raise RuntimeError('Validation image shape invalid')
    output.mkdir()
    report=dict(status='preparing',models=[r['name'] for r in records],baseline=baseline,provenance=provenance,
        source_sha256=digest(Path(__file__)),family_source_sha256=digest(PIPELINE/'routers/correctness/compact_cpu.py'),
        validation_rgb_sha256=digest(cache/'validation_rgb128.npy'),feature_names=FEATURE_NAMES,results=[],
        cpu_energy_measured=False,test_data_accessed=False,gpu_measurements_complete=False,
        note='CPU fitting and calibration predictions only. Deployment adapters/profiling/policy freeze required before testing.')
    path=output/'preparation.json'
    save(path,report)
    with threadpool_limits(limits=1):
        descriptors=np.concatenate([features(np.array(images[i:i+128])) for i in range(0,len(images),128)])
        pixels=np.concatenate([thumbnails(np.array(images[i:i+128])).reshape(-1,192) for i in range(0,len(images),128)])
        for seed in (42,45,48):
            train,calibration=split_validation(labels,seed)
            original=json.loads((PIPELINE/f'artifacts/pc_routers_correctness/global_correctness_seed{seed}.json').read_text())
            if train.tolist()!=original['fit_positions'] or calibration.tolist()!=original['calibration_positions'] or provenance!=original['provenance']:
                raise RuntimeError('Historical split/provenance mismatch')
            for method in METHODS:
                x=pixels if method=='thumbnail_mlp16' else descriptors
                started=time.perf_counter()
                model=fit(method,x[train],correct[train],seed)
                seconds=time.perf_counter()-started
                predictions=model.predict_inputs(x[calibration])
                model_path=output/f'{method}_seed{seed}.joblib'
                joblib.dump(model,model_path,compress=3)
                prediction_path=output/f'{method}_seed{seed}_calibration.npz'
                np.savez_compressed(prediction_path,positions=calibration,probabilities=predictions,correctness=correct[calibration])
                report['results'].append(dict(method=method,seed=seed,fit_positions=train.tolist(),calibration_positions=calibration.tolist(),
                    fit_seconds=seconds,model_file=str(model_path),model_sha256=digest(model_path),serialized_bytes=model_path.stat().st_size,
                    calibration_file=str(prediction_path),calibration_sha256=digest(prediction_path),
                    calibration_brier=float(np.mean((predictions-correct[calibration])**2)),outputs=213))
                save(path,report)
                print(f'PREPARED {method} seed={seed}, {len(report["results"])}/12; not a GPU runtime result',flush=True)
    report['status']='complete'
    save(path,report)


if __name__=='__main__':
    main()
