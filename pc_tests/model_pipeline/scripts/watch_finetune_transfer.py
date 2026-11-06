"""User-run, read-only readiness checks followed by one verified SCP import."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time

if __package__:
    from . import split_finetune as runner
else:
    import split_finetune as runner


REMOTE_PROBE = '''
import getpass, json, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1])
complete = sorted(p.parent.name for p in root.glob('*/receipt.json')
                  if (p.parent / 'study.json').is_file() and (p.parent / 'best.pth').is_file())
live = None
if sys.argv[2:]:
    query = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i %T'],
                           capture_output=True, text=True, timeout=15)
    if query.returncode == 0:
        live = {line.split()[0]: line.split()[1] for line in query.stdout.splitlines()
                if len(line.split()) == 2 and line.split()[0] in sys.argv[2:]}
print(json.dumps(dict(complete=complete, live=live)))
'''

REMOTE_STREAM = '''
import getpass, json, subprocess, sys, tarfile, time
from pathlib import Path

root, cases, jobs, seconds = Path(sys.argv[1]), set(json.loads(sys.argv[2])), sys.argv[4:], int(sys.argv[3])
files = ('receipt.json', 'study.json', 'best.pth')
while True:
    completed = {p.parent.name for p in root.glob('*/receipt.json')
                 if all((p.parent / name).is_file() for name in files)}
    print(f'HPC exports: {len(cases & completed)}/{len(cases)}', file=sys.stderr, flush=True)
    if completed - cases:
        raise SystemExit('Unexpected HPC exports; no transfer.')
    if cases == completed:
        break
    if jobs:
        try:
            q = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i'],
                               capture_output=True, text=True, timeout=15)
            if q.returncode == 0 and not set(q.stdout.split()) & set(jobs):
                raise SystemExit('Slurm jobs left the queue with missing exports; inspect logs.')
        except subprocess.TimeoutExpired:
            pass
    time.sleep(seconds)
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
    for case in sorted(cases):
        for name in files:
            archive.add(root / case / name, arcname=case + '/' + name, recursive=False)
'''


def snapshot(handoff):
    plan = runner.read(handoff)
    pc = set(plan['prior_ids']) | set(plan['shards']['pc'])
    hpc = {cid for owner, ids in plan['shards'].items() if owner != 'pc' for cid in ids}
    runner.require(pc and hpc and not pc & hpc and pc | hpc == {c['id'] for c in plan['cases']},
                   'Invalid handoff assignment')
    runner.require(plan.get('legacy_pc_plan_sha256'), 'This watcher requires the live-tail handoff')
    state = runner.read(runner.REPO / runner.PIPE / 'reports' / runner.BANK / 'campaign.json')
    runner.require(state['plan_sha256'] == plan['legacy_pc_plan_sha256'], 'PC campaign plan changed')
    done = {row['id'] for row in state['completed']} | set(plan['prior_ids'])
    return pc, hpc, done, state


def endpoint(remote):
    runner.require(re.fullmatch(r'[\w.@-]+:/[\w./-]+', remote) and not remote.startswith('-'),
                   'Expected user@host:/absolute/export/path')
    return remote.split(':', 1)


def probe(remote, jobs, unattended):
    host, path = endpoint(remote)
    command = shlex.join(['python3', '-B', '-c', REMOTE_PROBE, path, *jobs])
    options = ['-o', 'ConnectTimeout=15']
    if unattended:
        options += ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                    '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2']
    try:
        process = subprocess.run(['ssh', *options, host, command], capture_output=True,
                                 text=True, timeout=90 if unattended else None, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('HPC status check failed. Unattended mode requires working SSH '
                           'authentication and an already trusted host key. No files transferred.') from exc
    return json.loads(process.stdout)


def unpack(archive, folder, cases):
    expected = {f'{case}/{name}' for case in cases for name in ('receipt.json', 'study.json', 'best.pth')}
    seen = set()
    with tarfile.open(archive, 'r:') as source:
        for member in source:
            runner.require(member.isfile() and member.name in expected - seen,
                           'Unexpected or duplicate file in received archive')
            seen.add(member.name)
            path = folder / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as inp, path.open('xb') as out:
                shutil.copyfileobj(inp, out)
    runner.require(seen == expected, 'Received archive is incomplete')


def single_login(handoff, remote, dest, jobs, interval):
    runner.require(not dest.exists(), 'Destination already exists; will not overwrite')
    pc, hpc, done, state = snapshot(handoff)
    host, path = endpoint(remote)
    cmd = shlex.join(['python3', '-B', '-c', REMOTE_STREAM, path, json.dumps(sorted(hpc)),
                     str(interval * 60), *jobs])
    print(f'PC {len(pc & done)}/{len(pc)}. Authenticate to HPC once when prompted.', flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='finetune-ssh-', dir=dest.parent) as name:
        stage = Path(name)
        archive = stage / 'exports.tar'
        with archive.open('xb') as out:
            subprocess.run(['ssh', '-T', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
                            '-o', 'ServerAliveCountMax=3', host, cmd], stdout=out, check=True)
        exports = stage / 'exports'
        unpack(archive, exports, hpc)
        while True:
            pc, _, done, state = snapshot(handoff)
            print(f'HPC downloaded; PC {len(pc & done)}/{len(pc)} ({state["status"]}).', flush=True)
            if pc <= done:
                runner.collect(handoff, remote, dest, local_exports=exports)
                return
            runner.require(state['status'] not in ('stopped', 'failed'), 'PC stopped before finishing its prefix')
            time.sleep(interval * 60)


def run(handoff, remote, dest, jobs, interval):
    runner.require(not dest.exists(), 'Destination already exists; will not overwrite')
    while True:
        pc, hpc, done, state = snapshot(handoff)
        status = probe(remote, jobs, interval is not None)
        exported, live = set(status['complete']), status['live']
        runner.require(not exported - hpc, 'Unexpected HPC exports: ' + ', '.join(sorted(exported - hpc)))
        print(f'{datetime.now().astimezone():%Y-%m-%d %H:%M %Z}: '
              f'PC {len(pc & done)}/{len(pc)} ({state["status"]}, active: {state.get("active")}); '
              f'HPC {len(exported)}/{len(hpc)} exports; jobs: {live if live is not None else "unverified"}',
              flush=True)
        if pc <= done and hpc == exported:
            runner.collect(handoff, remote, dest, batch_mode=interval is not None)
            return
        if pc - done and state['status'] in ('stopped', 'failed'):
            raise RuntimeError('PC worker stopped before completing its prefix; no transfer.')
        if hpc - exported and live == {}:
            raise RuntimeError('None of the specified Slurm jobs is still in the queue and exports '
                               'are incomplete. Inspect their logs; no transfer.')
        if interval is None:
            return
        time.sleep(interval * 60)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('handoff', type=Path)
    cli.add_argument('remote')
    cli.add_argument('dest', type=Path)
    cli.add_argument('--jobs', nargs='+', default=[])
    cli.add_argument('--wait-minutes', type=int, help='Check repeatedly until ready; needs noninteractive SSH')
    cli.add_argument('--single-login', action='store_true', help='Authenticate once and keep that SSH session open')
    args = cli.parse_args()
    if any(not re.fullmatch(r'[0-9]+', job) for job in args.jobs):
        cli.error('Job IDs must be numeric')
    if args.wait_minutes is not None and args.wait_minutes < 5:
        cli.error('Set at least five minutes between checks')
    try:
        if args.single_login:
            single_login(args.handoff.resolve(), args.remote, args.dest.resolve(), args.jobs, args.wait_minutes or 10)
        else:
            run(args.handoff.resolve(), args.remote, args.dest.resolve(), args.jobs, args.wait_minutes)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        cli.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
