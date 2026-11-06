"""Split the Adam campaign and collect verified models without replacing PC files."""
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if HERE.parent.name == 'scripts' else HERE.parent
PIPE = Path('pc_tests/model_pipeline')
BANK = 'fresh_prune_adam_bank_20261105'
PACKAGES = ('torch', 'torchvision', 'torch-pruning', 'numpy', 'Pillow')
ESTIMATE = dict(resnet18=15, resnet34=35, resnet50=55, resnet101=80,
                efficientnet_b0=35, mobilenetv3_small=22, mobilenetv3_large=32)
EXPERIMENTS = ('fresh_prune_adam_bank', 'fresh_prune_adam', 'adam_recovery_curve',
               'adam_post_prune_pilot', 'compare_dispatch_residency_v2', 'prune_finetune',
               'train_baselines')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(path, expected):
    require(path.is_file() and digest(path) == expected, f'Missing or mismatched file: {path}')


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)


def versions():
    return {p: importlib.metadata.version(p).split('+', 1)[0] for p in PACKAGES}


def allocate(cases, completed, interrupted, workers):
    shards = {f'hpc{i}': [] for i in range(workers)} | {'pc': []}
    weight = dict.fromkeys(shards, 0)
    for case in sorted(cases, key=lambda x: -ESTIMATE[x['architecture']]):
        cid = case['id']
        if cid in completed:
            continue
        owner = 'pc' if cid == interrupted else min(weight, key=weight.get)
        shards[owner].append(cid)
        weight[owner] += ESTIMATE[case['architecture']]
    return shards


def old_worker_stopped():
    query = ("@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object CommandLine -match ' -m pc_tests\\.model_pipeline\\.experiments\\.fresh_prune_adam_bank( |$)').Count")
    count = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command', query], text=True)
    require(int(count.strip()) == 0, 'Original PC worker is still running; no handoff is allowed.')


def allocate_tail(cases, completed, active, workers, size):
    ids = [c['id'] for c in cases]
    require(active in ids, 'Cannot determine the active PC case')
    prefix, tail = ids[:-size], cases[-size:]
    require(active in prefix and set(completed) <= set(prefix),
            'PC has already reached the requested HPC tail; choose a shorter tail')
    shards = {f'hpc{i}': [] for i in range(workers)}
    weight = dict.fromkeys(shards, 0)
    for case in sorted(reversed(tail), key=lambda c: -ESTIMATE[c['architecture']]):
        owner = min(weight, key=weight.get)
        shards[owner].append(case['id'])
        weight[owner] += ESTIMATE[case['architecture']]
    shards['pc'] = [cid for cid in prefix if cid not in completed]
    return shards


def prepare(out, workers, tail_size=None):
    if tail_size is None:
        old_worker_stopped()
    original = REPO / PIPE / 'reports' / BANK
    campaign = read(original / 'campaign.json')
    verify(original / 'plan.json', campaign['plan_sha256'])
    plan = read(original / 'plan.json')
    require(len(plan['cases']) == 63 and not plan['test_data_accessed'], 'Unexpected original plan')
    recipe = {key: plan[key] for key in ('split_sha256', 'optimizer', 'lr', 'seed', 'image_size',
              'amp', 'label_smoothing', 'weight_decay', 'scheduler', 'stop')}
    prior = {row['id']: row for row in campaign['completed']}
    require(len(prior) == len(campaign['completed']), 'Duplicate old PC result')
    for spec in plan['cases']:
        cid = spec['id']
        report_path = original / 'cases' / cid / 'study.json'
        if cid not in prior and report_path.is_file():
            report = read(report_path)
            if report['status'] == 'complete':
                require(report['spec'] == spec and report['split_sha256'] == recipe['split_sha256'],
                        f'Incompatible completed legacy result: {cid}')
                prior[cid] = dict(id=cid, best=report['best'], report=str(report_path),
                                  report_sha256=digest(report_path))
    for row in prior.values():
        verify(Path(row['report']), row['report_sha256'])
        verify(Path(row['best']['path']), row['best']['sha256'])
    cases = []
    inputs = {}
    for row in plan['cases']:
        case = dict(row)
        for key in ('baseline', 'original_sidecar'):
            source = Path(row[key])
            case[key] = source.relative_to(REPO).as_posix()
            if row['id'] not in prior:
                inputs[case[key]] = plan['protected_sha256'][str(source)]
        cases.append(case)
    require(set(prior) <= {c['id'] for c in cases}, 'Unknown prior case')
    split = PIPE / 'artifacts/baselines/cifar100_split_seed42.json'
    inputs[split.as_posix()] = plan['protected_sha256'][str(REPO / split)]
    sources = [PIPE / 'paths.py', PIPE / 'runtime/atomic_checkpoint.py',
               PIPE / 'analysis/summarize_pruning.py']
    sources += [PIPE / 'experiments' / (name + '.py') for name in EXPERIMENTS]
    sources += [p.relative_to(REPO) for p in (REPO / 'model_pipeline').glob('*.py')]
    for folder in (Path('pc_tests'), PIPE, PIPE/'experiments', PIPE/'runtime', PIPE/'analysis'):
        if (REPO / folder / '__init__.py').is_file():
            sources.append(folder / '__init__.py')
    sources = {p.as_posix(): digest(REPO / p) for p in sources}
    for path, expected in plan['protected_sha256'].items():
        verify(Path(path), expected)
    for path, expected in inputs.items():
        verify(REPO / path, expected)
    if tail_size is not None:
        require(campaign['status'] == 'running' and workers <= tail_size < len(cases),
                'Tail mode requires the original PC campaign to be running')
        shards = allocate_tail(cases, prior, campaign['active'], workers, tail_size)
    else:
        shards = allocate(cases, prior, campaign['active'], workers)
    require(all(shards.values()), 'Not enough remaining cases for this many workers')
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'prior_pc.json', prior)
    write(out / 'handoff.json', dict(recipe=recipe, cases=cases, shards=shards, sources=sources,
          inputs=inputs, versions=versions(), runner_sha256=digest(Path(__file__)),
          legacy_pc_plan_sha256=campaign['plan_sha256'] if tail_size is not None else None,
          split=split.as_posix(), prior_ids=sorted(prior)))
    wanted = {cid for name, ids in shards.items() if name != 'pc' for cid in ids}
    hpc_inputs = {split.as_posix()}
    for case in cases:
        if case['id'] in wanted:
            hpc_inputs.update((case['baseline'], case['original_sidecar']))
    with tarfile.open(out / 'hpc-input.tar.gz', 'x:gz', compresslevel=1) as archive:
        for name in sorted(set(sources) | hpc_inputs):
            archive.add(REPO / name, arcname=name, recursive=False)
        archive.add(out / 'handoff.json', arcname='handoff.json')
        archive.add(Path(__file__), arcname='split_finetune.py')
        slurm = Path(__file__).with_name('split_finetune.sbatch').read_bytes().replace(b'\r\n', b'\n')
        entry = tarfile.TarInfo('split_finetune.sbatch')
        entry.size = len(slurm)
        archive.addfile(entry, io.BytesIO(slurm))
    print('Preserved', len(prior), 'completed cases; assigned:', {k: len(v) for k, v in shards.items()})
    if tail_size is not None:
        current = read(original / 'campaign.json')
        require(not ({r['id'] for r in current['completed']} | {current['active']}) & wanted,
                'PC reached the tail during packaging; do not upload this bundle')
        print('HPC tail starts at', cases[-tail_size]['id'], '; the PC worker was not changed.')


def publish(out, case, owner, plan_hash, summary):
    export = out / 'export'
    export.mkdir(exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix='publication-', dir=out))
    checkpoint = Path(summary['best']['path'])
    shutil.copyfile(checkpoint, pending / 'best.pth')
    shutil.copyfile(summary['report'], pending / 'study.json')
    verify(pending / 'best.pth', summary['best']['sha256'])
    verify(pending / 'study.json', summary['report_sha256'])
    write(pending / 'receipt.json', dict(id=case['id'], owner=owner, plan_sha256=plan_hash,
          checkpoint_sha256=summary['best']['sha256'], report_sha256=summary['report_sha256']))
    pending.rename(export / case['id'])


def worker(root, handoff, owner, data, out):
    plan_hash = digest(handoff)
    plan = read(handoff)
    verify(Path(__file__), plan['runner_sha256'])
    require(versions() == plan['versions'], f"Dependency versions must match PC: {plan['versions']}")
    if owner == 'pc':
        require(not plan.get('legacy_pc_plan_sha256'), 'Tail mode uses the existing PC worker; do not start another')
        old_worker_stopped()
    else:
        require(os.name != 'nt' and os.environ.get('SLURM_JOB_ID'), 'HPC workers require a Slurm allocation')
    chosen = [next(c for c in plan['cases'] if c['id'] == cid) for cid in plan['shards'][owner]]
    for path, sha in plan['sources'].items():
        verify(root / path, sha)
    needed = {plan['split']} | {c[k] for c in chosen for k in ('baseline', 'original_sidecar')}
    for path in needed:
        verify(root / path, plan['inputs'][path])
    sys.path.insert(0, str(root))
    import copy
    import time
    import torch
    from torch.utils.data import Subset
    from model_pipeline.data import evaluation_transform, training_transform, split_fingerprint, stratified_indices
    from pc_tests.model_pipeline.experiments import fresh_prune_adam_bank as bank
    from pc_tests.model_pipeline.experiments.adam_post_prune_pilot import TrainOnlyCIFAR100

    require(torch.cuda.device_count() == 1, 'Expected exactly one visible CUDA GPU')
    probe = torch.ones(32, device='cuda')
    require(float(probe.sum()) == 32, 'CUDA preflight failed')
    del probe
    torch.set_num_threads(1)
    train_data = TrainOnlyCIFAR100(root=str(data), train=True, download=False, transform=training_transform(128))
    val_data = copy.copy(train_data)
    val_data.transform = evaluation_transform(128)
    train, val = stratified_indices(train_data.targets, 5000, 42)
    split = read(root / plan['split'])
    require(train == split['train_indices'] and val == split['validation_indices'] and
            split_fingerprint(train, val) == plan['recipe']['split_sha256'], 'Validation split mismatch')
    val_loader = bank.make_loader(Subset(val_data, val), 32, 2, False, 43)
    train = Subset(train_data, train)
    if owner == 'pc':
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'pc.lock').open('x'):
            pass
    attempt = out / 'attempts' / owner
    attempt.mkdir(parents=True, exist_ok=False)
    bank.OUT = bank.telemetry.OUT = attempt
    bank.MODELS = attempt / 'models'
    guard = bank.Guard() if owner == 'pc' else argparse.Namespace(phase='setup')

    def check():
        if owner == 'pc':
            guard.check()
        require(torch.cuda.mem_get_info()[0] >= 1024**3, 'Below 1 GiB VRAM reserve')

    def pause_heat():
        check()
        if owner == 'pc' and guard.last['temperature_c'] >= 76:
            torch.cuda.synchronize()
            deadline = time.monotonic() + 180
            while guard.last['temperature_c'] > 68:
                check()
                require(time.monotonic() < deadline, 'PC cooldown timed out')
                time.sleep(.5)

    try:
        if owner == 'pc':
            guard.start()
        for case in chosen:
            require(not (out / 'export' / case['id']).exists(), 'Refusing to overwrite an export')
            spec = dict(case, baseline=str(root / case['baseline']), original_sidecar=str(root / case['original_sidecar']))
            local = dict(plan['recipe'], protected_sha256={spec['baseline']: plan['inputs'][case['baseline']]})
            summary = bank.run_case(spec, local, train, val_loader, guard, check, pause_heat)
            publish(out, case, owner, plan_hash, summary)
            print('Published', case['id'], flush=True)
    finally:
        if owner == 'pc':
            guard.close()


def audit_export(path, case, owner, plan_hash, recipe):
    receipt = read(path / 'receipt.json')
    require((receipt['id'], receipt['owner'], receipt['plan_sha256']) == (case['id'], owner, plan_hash),
            f'Unexpected receipt: {path}')
    verify(path / 'best.pth', receipt['checkpoint_sha256'])
    verify(path / 'study.json', receipt['report_sha256'])
    report = read(path / 'study.json')
    spec = report['spec']
    require(all(spec[k] == case[k] for k in ('id', 'architecture', 'level', 'batch')) and
            report['status'] == 'complete' and report['split_sha256'] == recipe['split_sha256'] and
            report['best']['sha256'] == receipt['checkpoint_sha256'], f'Incompatible report: {path}')
    rows, stop = report['rows'], recipe['stop']
    require(stop['min_epochs'] <= len(rows) <= stop['max_epochs'], 'Invalid epoch count')
    anchor, significant, trigger = -1., 0, None
    for epoch, row in enumerate(rows, 1):
        val = row['validation']
        cm = val['confusion']
        require(row['epoch'] == epoch and row['images'] == 45000 and row['steps'] == math.ceil(45000/case['batch'])
                and 0 <= row['skipped_amp_updates'] < row['steps'] and len(cm) == 100 and
                all(len(r) == 100 and sum(r) == 50 and all(type(v) is int and v >= 0 for v in r) for r in cm) and
                val['accuracy'] == sum(cm[i][i] for i in range(100))/50 and
                all(math.isfinite(x) for x in (val['loss'], row['train_loss'])), f'Invalid epoch {epoch}: {path}')
        if val['accuracy'] >= anchor + stop['min_delta_pp'] - 1e-9:
            anchor, significant = val['accuracy'], epoch
        if epoch >= stop['min_epochs'] and epoch - significant >= stop['patience'] and trigger is None:
            trigger = epoch
    require((trigger == len(rows) and report['stop_reason'] == 'validation patience') or
            (trigger is None and len(rows) == stop['max_epochs'] and report['stop_reason'] == 'epoch cap'),
            f'Wrong stopping rule: {path}')
    best = max(rows, key=lambda r: r['validation']['accuracy'])
    require((best['epoch'], best['validation']['accuracy']) == (report['best']['epoch'], report['best']['accuracy']),
            f'Incorrect best checkpoint metadata: {path}')
    return best['validation']['accuracy']


def collect(handoff, remote, dest, *, batch_mode=False, local_exports=None):
    require(re.fullmatch(r'[\w.@-]+:/[\w./-]+', remote) and not remote.startswith('-'),
            'Use user@host:/absolute/path/to/results/export (no spaces)')
    require(not dest.exists(), 'Destination already exists; will not overwrite')
    plan, plan_hash = read(handoff), digest(handoff)
    prior = read(handoff.with_name('prior_pc.json'))
    owners = {cid: owner for owner, ids in plan['shards'].items() for cid in ids}
    expected = {cid for cid, owner in owners.items() if owner != 'pc'}
    require(len(owners) == sum(map(len, plan['shards'].values())) and
            set(prior) == set(plan['prior_ids']) and not set(prior) & set(owners) and
            set(prior) | set(owners) == {c['id'] for c in plan['cases']}, 'Invalid assignment coverage')
    legacy = {}
    if plan.get('legacy_pc_plan_sha256'):
        original = REPO / PIPE / 'reports' / BANK
        verify(original / 'plan.json', plan['legacy_pc_plan_sha256'])
        state = read(original / 'campaign.json')
        require(state['plan_sha256'] == plan['legacy_pc_plan_sha256'], 'Original PC plan changed')
        legacy = {row['id']: row for row in state['completed']}
        missing = set(plan['shards']['pc']) - set(legacy)
        require(not missing, 'PC prefix not yet complete: ' + ', '.join(sorted(missing)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='finetune-import-', dir=dest.parent) as temp:
        staging = Path(temp) / 'remote'
        scp_options = ['-B', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=20',
                       '-o', 'ServerAliveCountMax=3'] if batch_mode else []
        if local_exports is None:
            subprocess.run(['scp', *scp_options, '-r', remote, str(staging)], check=True)
        else:
            shutil.copytree(local_exports, staging)
        require({p.name for p in staging.iterdir()} == expected, 'Missing or unexpected HPC case folders')
        rows = []
        for case in plan['cases']:
            cid = case['id']
            if cid in prior or (legacy and owners[cid] == 'pc'):
                old = prior[cid] if cid in prior else legacy[cid]
                checkpoint = Path(old['best']['path'])
                verify(checkpoint, old['best']['sha256'])
                verify(Path(old['report']), old['report_sha256'])
                report = read(Path(old['report']))
                require(report['status'] == 'complete' and report['best'] == old['best'] and
                        report['split_sha256'] == plan['recipe']['split_sha256'], 'Invalid PC result: ' + cid)
                if cid not in prior:
                    spec = report['spec']
                    require(all(spec[k] == case[k] for k in ('id', 'architecture', 'level', 'batch')),
                            'Wrong PC case: ' + cid)
                accuracy, owner = old['best']['accuracy'], 'pc-original'
            else:
                owner = owners[cid]
                folder = handoff.parent / 'pc-results' / 'export' / cid if owner == 'pc' else staging / cid
                accuracy = audit_export(folder, case, owner, plan_hash, plan['recipe'])
                checkpoint = folder / 'best.pth'
            if owner.startswith('pc'):
                target = staging / cid / 'best.pth'
                target.parent.mkdir()
                shutil.copyfile(checkpoint, target)
                verify(target, digest(checkpoint))
            rows.append((cid, owner, accuracy, str(dest / cid / 'best.pth')))
        with (staging / 'model_index.csv').open('x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('case', 'owner', 'best_validation_accuracy', 'checkpoint'))
            writer.writerows(rows)
        staging.rename(dest)
    duplicates = sorted(set(legacy) & expected)
    if duplicates:
        print('Kept the assigned HPC checkpoint in the index; left duplicate PC files untouched:', ', '.join(duplicates))
    print(f'Verified {len(expected)} HPC checkpoint hashes; {len(rows)}-case model index: {dest / "model_index.csv"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest='cmd', required=True)
    p = cmd.add_parser('prepare')
    p.add_argument('out', type=Path)
    p.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--tail-cases', type=int, help='Reserve a tail for HPC without stopping the original PC worker')
    w = cmd.add_parser('worker')
    for field in ('root', 'handoff', 'data', 'out'):
        w.add_argument('--' + field, type=Path, required=True)
    w.add_argument('--owner', required=True)
    c = cmd.add_parser('collect')
    c.add_argument('handoff', type=Path)
    c.add_argument('remote')
    c.add_argument('dest', type=Path)
    a = parser.parse_args()
    if a.cmd == 'prepare':
        prepare(a.out.resolve(), a.workers, a.tail_cases)
    elif a.cmd == 'worker':
        worker(a.root.resolve(), a.handoff.resolve(), a.owner, a.data.resolve(), a.out.resolve())
    else:
        collect(a.handoff.resolve(), a.remote, a.dest.resolve())


if __name__ == '__main__':
    main()
