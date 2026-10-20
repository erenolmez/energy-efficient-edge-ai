"""Isolated validation-only 213-output training; refuse competing GPU studies."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from audit_validation_complementarity import load_validation, digest
from batch_router_policy import split_validation, make_batches
from correctness_policy import calibrate_correctness


def atomic_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def require_idle_gpu():
    script = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match '(run_pc_routes|run_correctness_routes|gpu_static_sweep|train_pc_batch_routers|train_correctness_router)\\.py' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(rows, dict):
        rows = [rows]
    # A venv launcher parent may have the same command. Identify it explicitly.
    allowed = {os.getpid(), os.getppid()}
    others = [row for row in rows if row['ProcessId'] not in allowed]
    if others:
        raise RuntimeError('GPU experiment already active; correctness training deferred: ' + str([r['ProcessId'] for r in others]))


def require_guard(path, output_dir):
    state = json.loads(path.read_text(encoding='utf-8-sig'))
    if state.get('status') != 'monitoring':
        raise RuntimeError('Thermal state is not monitoring: never restart a latched study')
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(state['utc'].replace('Z', '+00:00'))).total_seconds()
    if not 0 <= age <= 120 or state.get('expected_script') != Path(__file__).name or os.getpid() not in state['tracked_process_ids']:
        raise RuntimeError('Fresh thermal guard must track this exact training process')
    guarded = Path(__file__).resolve().parents[2] / state['training_marker_directory']
    if guarded.resolve() != output_dir.resolve():
        raise RuntimeError('Thermal guard checkpoint directory mismatch')


def costs_for(records, size, objective):
    key = 'normalized_host_input_to_logits_batch_ms' if objective == 'latency' else 'gpu_energy_j_per_image'
    values = np.array([[r[key] for r in row['timings'][str(size)]] for row in records])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Complete positive measured costs required')
    return np.median(values, axis=1) * (size if objective == 'energy' else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('static-report', 'static-dir', 'cache-dir', 'output-dir', 'audit-report', 'thermal-state'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seeds', default='42,45,48')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--guard-startup-seconds', type=int, default=30)
    args = p.parse_args()
    if args.epochs <= 0:
        raise ValueError('Positive epochs required')
    require_idle_gpu()  # Before importing GPU runtime or touching CUDA.
    records, labels, correct, baseline, provenance = load_validation(args.static_report, args.static_dir, args.cache_dir)
    audit = json.loads(args.audit_report.read_text())
    if audit['provenance'] != provenance or audit.get('scope') != 'validation_only':
        raise RuntimeError('Audit is missing or stale')
    names = [r['name'] for r in records]
    if audit['models'] != names:
        raise RuntimeError('Audit candidate ordering mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Awaiting verified thermal guard for PID {os.getpid()}', flush=True)
    deadline = time.monotonic() + args.guard_startup_seconds
    while True:
        try:
            require_guard(args.thermal_state, args.output_dir)
            break
        except RuntimeError:
            state = json.loads(args.thermal_state.read_text(encoding='utf-8-sig'))
            if state['status'] != 'monitoring' or time.monotonic() >= deadline:
                raise
            time.sleep(1)
    import torch
    from torch.nn import functional as F
    from correctness_router import CorrectnessRouter, BatchCorrectnessRouter
    from gpu_batch_inputs import SharedBatchInputs
    from router_tensorrt import build_router, RouterRunner
    torch.set_num_threads(1)
    images = np.load(args.cache_dir / 'validation_rgb128.npy', mmap_mode='r', allow_pickle=False)
    if images.shape != (5000, 3, 128, 128) or images.dtype != np.uint8:
        raise ValueError('Expected fixed validation RGB128 identity')
    image_hash = digest(args.cache_dir / 'validation_rgb128.npy')
    buffers = SharedBatchInputs()
    sizes = (8, 16, 32, 64, 128)
    baseline_index = names.index(baseline)
    for seed in map(int, args.seeds.split(',')):
        started = time.perf_counter()
        fit, calibration = split_validation(labels, seed)
        settings = dict(seed=seed, epochs=args.epochs, model='tiny_shared_encoder_213_correctness_logits',
                        loss='unweighted BCEWithLogits', learning_rate=.001, fit_images=3000, calibration_images=2000)
        marker = args.output_dir / f'global_correctness_seed{seed}.json'
        if marker.exists():
            prior = json.loads(marker.read_text())
            if prior.get('status') == 'complete' and prior.get('settings') == settings and prior.get('provenance') == provenance and prior.get('validation_rgb_sha256') == image_hash and digest(prior['weights']) == prior['weights_sha256'] and all(digest(e['engine']) == e['sha256'] for e in prior['engines'].values()):
                print('Skip verified ' + marker.stem, flush=True)
                continue
            raise RuntimeError('Existing incomplete/stale training preserved; use a new isolated output directory')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = CorrectnessRouter(213).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        targets = torch.from_numpy(correct.astype(np.float32))
        rng = np.random.default_rng(seed)
        history = []
        for epoch in range(args.epochs):
            require_guard(args.thermal_state, args.output_dir)
            model.train()
            permutation = rng.permutation(fit)
            losses = []
            for start in range(0, len(permutation), 64):
                positions = permutation[start:start + 64]
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
            torch.save(dict(weights=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1, settings=settings,
                            provenance=provenance, rng_state=rng.bit_generator.state), temporary)
            temporary.replace(checkpoint)
            atomic_json(checkpoint.with_suffix('.json'), dict(status='complete', epoch=epoch+1, checkpoint=str(checkpoint.resolve()), sha256=digest(checkpoint)))
            print(f'seed={seed} epoch={epoch+1} fit_bce={history[-1]:.6f}', flush=True)
        model.eval()
        weights = args.output_dir / f'global_correctness_seed{seed}.pth'
        temporary = weights.with_suffix('.pth.tmp')
        torch.save(model.state_dict(), temporary)
        temporary.replace(weights)
        # Export fixed batches, including true partial tails. Never duplicate
        # images when averaging correctness probabilities.
        shapes = sorted(set(sizes) | {len(calibration) % s for s in sizes if len(calibration) % s} | {10000 % s for s in sizes if 10000 % s})
        wrapped = BatchCorrectnessRouter(model).eval()
        engines, runners = {}, {}
        for size in shapes:
            require_guard(args.thermal_state, args.output_dir)
            path, error = build_router(wrapped, 'normalized_images', args.output_dir / 'engines',
                                       f'correctness_seed{seed}_b{size}', input_shape=(size, 3, 128, 128))
            engines[str(size)] = dict(engine=str(path.resolve()), sha256=digest(path), torch_max_error=error)
            runners[size] = RouterRunner(path)
        policies = {}
        calibration_brier = []
        for size in sizes:
            batches = make_batches(calibration, size, seed + size)
            q, counts, lengths = [], [], []
            with torch.inference_mode():
                for batch in batches:
                    normalized = buffers.prepare(torch.from_numpy(np.array(images[batch])).pin_memory())
                    value = runners[len(batch)].run(normalized.contiguous()).cpu().numpy()[0]
                    q.append(value.copy())
                    counts.append(correct[batch].sum(0))
                    lengths.append(len(batch))
            q, counts, lengths = np.asarray(q), np.asarray(counts), np.asarray(lengths)
            calibration_brier.append(float(np.mean((q - counts / lengths[:, None])**2)))
            for objective in ('latency', 'energy'):
                costs = costs_for(records, size, objective)
                policies.setdefault(objective, {})[str(size)] = dict(costs=costs.tolist(), budgets={str(b): calibrate_correctness(q, counts, lengths, costs, baseline_index, b) for b in (.5, 1., 1.5, 2.)})
        atomic_json(marker, dict(status='complete', protocol_version=1, models=names, baseline=baseline,
            settings=settings, provenance=provenance, validation_rgb_sha256=image_hash,
            fit_positions=fit.tolist(), calibration_positions=calibration.tolist(),
            weights=str(weights.resolve()), weights_sha256=digest(weights), engines=engines,
            training_loss=history, calibration_batch_brier=calibration_brier, policies=policies,
            elapsed_seconds=time.perf_counter()-started,
            note='Three shared correctness fits, independent energy/latency calibration. No test data opened; measured static cost estimates exclude runtime loading and router overhead. End-to-end deployment still required.'))
        del model, wrapped, runners
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
