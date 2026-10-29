"""Isolated repeat of the historical correctness fit with GPU energy counters.

No test data, calibration, TensorRT export, or historical artifacts are modified.
"""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import load_validation, digest
from pc_tests.model_pipeline.routers.scalar.batch_router_policy import split_validation
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json, require_idle_gpu


def read_guard(path, output_dir, now=None):
    # The historical PowerShell guard writes in place: retry only parse races.
    for attempt in range(20):
        try:
            state = json.loads(path.read_text(encoding='utf-8-sig'))
            break
        except (json.JSONDecodeError, OSError):
            if attempt == 19:
                raise RuntimeError('Thermal state persistently unreadable')
            time.sleep(.025)
    if state.get('status') != 'monitoring' or state.get('boundary_start_count') is not None:
        raise RuntimeError('Thermal stop latched: preserve checkpoint, never restart')
    age = ((now or datetime.now(timezone.utc)) - datetime.fromisoformat(state['utc'].replace('Z', '+00:00'))).total_seconds()
    if not 0 <= age <= 130 or state.get('expected_script') != Path(__file__).name or os.getpid() not in state.get('tracked_process_ids', []):
        raise RuntimeError('Fresh thermal guard must track this exact worker')
    if (REPOSITORY_ROOT / state.get('training_marker_directory', '')).resolve() != output_dir.resolve():
        raise RuntimeError('Thermal checkpoint directory mismatch')
    return state


def require_energy(metrics):
    value = metrics.get('gpu_energy_j')
    if value is None or not 0 < value < float('inf'):
        raise RuntimeError('GPU energy counter invalid: ' + str(metrics))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--thermal-state', type=Path, required=True)
    args = parser.parse_args()
    require_idle_gpu()
    # Launcher creates only the directory/logs. Never overwrite a prior run.
    report_path = args.output_dir / 'training_energy.json'
    if report_path.exists() or list(args.output_dir.glob('global_*_seed*.json')):
        raise RuntimeError('Existing measurement preserved; do not restart or overwrite')
    deadline = time.monotonic() + 120
    print('Waiting for exact thermal guard pid=' + str(os.getpid()), flush=True)
    while True:
        try:
            read_guard(args.thermal_state, args.output_dir)
            break
        except RuntimeError:
            state = json.loads(args.thermal_state.read_text(encoding='utf-8-sig'))
            if state.get('status') != 'monitoring' or time.monotonic() >= deadline:
                raise
            time.sleep(1)

    import numpy as np
    import torch
    from torch.nn import functional as F
    from pc_tests.model_pipeline.routers.correctness.correctness_router import CorrectnessRouter
    from pc_tests.model_pipeline.runtime.gpu_batch_inputs import SharedBatchInputs
    from pc_tests.model_pipeline.runtime.gpu_energy import EnergyCounter

    cache = PIPELINE / 'artifacts/pc_precision/data_cache'
    records, labels, correct, baseline, provenance = load_validation(
        PIPELINE / 'reports/gpu_static.json', PIPELINE / 'artifacts/gpu_static', cache)
    audit = json.loads((PIPELINE / 'reports/complementarity/validation_complementarity.json').read_text())
    if audit.get('scope') != 'validation_only' or audit['provenance'] != provenance or audit['models'] != [r['name'] for r in records]:
        raise RuntimeError('Validation audit missing or stale')
    images = np.load(cache / 'validation_rgb128.npy', mmap_mode='r', allow_pickle=False)
    if images.shape != (5000, 3, 128, 128) or images.dtype != np.uint8:
        raise RuntimeError('Unexpected input shape/dtype')
    protected_paths = list((PIPELINE / 'artifacts/pc_routers_correctness').glob('global_correctness_seed*.*'))
    protected = {str(p): digest(p) for p in protected_paths}
    originals = {s: json.loads((PIPELINE / f'artifacts/pc_routers_correctness/global_correctness_seed{s}.json').read_text()) for s in (42,45,48)}
    input_hash = digest(cache / 'validation_rgb128.npy')
    report = dict(status='running', utc=datetime.now(timezone.utc).isoformat(), seeds=[42,45,48], results=[],
        provenance=provenance, validation_rgb_sha256=input_hash, protected_hashes=protected,
        source_hashes={str(p): digest(p) for p in (Path(__file__), PIPELINE / 'routers/correctness/correctness_router.py', PIPELINE / 'runtime/gpu_batch_inputs.py', PIPELINE / 'runtime/gpu_energy.py', PIPELINE / 'routers/correctness/train_correctness_router.py')},
        scope='GPU only; gross NVML counter difference including idle/display; not whole-PC',
        boundary='CUDA context/data validation before measurement; initialization, 20 fit epochs, atomic epoch checkpoint writes inside fit window. Final weights write, calibration, export and supervision generation excluded.',
        historical_energy_reconstructed=False, test_data_accessed=False,
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__)
    atomic_json(report_path, report)
    torch.set_num_threads(1)
    buffers = SharedBatchInputs()
    torch.cuda.init()
    for seed in report['seeds']:
        read_guard(args.thermal_state, args.output_dir)
        fit, calibration = split_validation(labels, seed)
        prior = originals[seed]
        if prior['settings']['epochs'] != 20 or prior['fit_positions'] != fit.tolist() or prior['validation_rgb_sha256'] != input_hash or prior['provenance'] != provenance:
            raise RuntimeError('Historical recipe/input mismatch')
        energy = EnergyCounter()
        if not energy.available:
            raise RuntimeError('NVML total energy counter unavailable: ' + str(energy.error))
        torch.cuda.synchronize()
        energy.start()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = CorrectnessRouter(213).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        targets = torch.from_numpy(correct.astype(np.float32))
        rng = np.random.default_rng(seed)
        history = []
        for epoch in range(20):
            read_guard(args.thermal_state, args.output_dir)
            model.train()
            permutation = rng.permutation(fit)
            losses = []
            for start in range(0, len(permutation), 64):
                positions = permutation[start:start+64]
                normalized = buffers.prepare(torch.from_numpy(np.array(images[positions])).pin_memory())
                logits = model(normalized)
                loss = F.binary_cross_entropy_with_logits(logits, targets[positions].cuda())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)))
            checkpoint = args.output_dir / f'global_checkpoint_epoch{epoch+1}_seed{seed}.pth'
            temporary = checkpoint.with_suffix('.pth.tmp')
            torch.save(dict(weights=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1,
                            settings=prior['settings'], provenance=provenance, rng_state=rng.bit_generator.state), temporary)
            temporary.replace(checkpoint)
            atomic_json(checkpoint.with_suffix('.json'), dict(status='complete', epoch=epoch+1, seed=seed,
                        checkpoint=str(checkpoint.resolve()), sha256=digest(checkpoint)))
            print(f'seed={seed} epoch={epoch+1}/20 bce={history[-1]:.6f}', flush=True)
            read_guard(args.thermal_state, args.output_dir)
        torch.cuda.synchronize()
        metrics = require_energy(energy.finish(20 * len(fit)))
        weights = args.output_dir / f'measured_correctness_seed{seed}.pth'
        torch.save(model.state_dict(), weights)
        report['results'].append(dict(seed=seed, epochs=20, fit_images=len(fit), calibration_images_used=0,
            settings=prior['settings'], training_loss=history, energy=metrics,
            historical_loss_max_abs_difference=max(abs(a-b) for a,b in zip(history, prior['training_loss'])),
            weights=str(weights), weights_sha256=digest(weights)))
        atomic_json(report_path, report)
        print(f"SAVED seed={seed} energy_j={metrics['gpu_energy_j']:.3f} seconds={metrics['energy_window_seconds']:.3f}", flush=True)
        del model, optimizer, targets
        torch.cuda.empty_cache()
    if any(digest(Path(p)) != h for p,h in protected.items()):
        raise RuntimeError('Historical artifact hash changed')
    report['status'] = 'complete'
    report['protected_artifacts_unchanged'] = True
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(report_path, report)
    print('COMPLETE 3/3 measured training seeds; no inference break-even claimed yet', flush=True)


if __name__ == '__main__':
    main()
