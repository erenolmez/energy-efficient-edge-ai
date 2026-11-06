"""CPU-only closure audit of saved 100/300/600-step fits; never reruns training."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
from statistics import median
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.analysis.audit_router_warmup import digest, interval
from pc_tests.model_pipeline.experiments.compare_router_warmup_dose import slots as planned_slots

PIPE = Path(__file__).resolve().parents[1]
ART = PIPE / 'artifacts/router_warmup_dose_20261103'
OUT = PIPE / 'reports/router_warmup_dose_20261103'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def positive(value):
    assert math.isfinite(value) and value > 0, value


def identity_ms(identity):
    found = re.fullmatch(r'/Date\((\d+)\)/', identity['CreationDate'])
    assert found, identity
    return int(found[1])


def main():
    raw = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command',
        "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | Select-Object ProcessId,ExecutablePath,CommandLine,CreationDate) | ConvertTo-Json -Depth 4"], text=True)
    live = json.loads(raw) if raw.strip() else []
    if isinstance(live, dict):
        live = [live]
    # Audit itself may be Python; no measured GPU experiment may overlap it.
    assert not any(any(name in (p.get('CommandLine') or '') for name in (
        'compare_router_warmup.py', 'compare_router_warmup_dose.py',
        'compare_dispatch_residency', 'probe_native_conditional_graph')) for p in live)
    state, plan = load(ART/'study.json'), load(ART/'plan.json')
    assert state['status'] == 'complete_pending_audit' and state['active_child_pid'] is None
    assert state['test_data_accessed'] is False
    assert digest(ART/'plan.json') == state['plan_sha256']
    for name, expected in plan['protected_sha256'].items():
        assert digest(name) == expected, name
    slots, results = plan['slots'], state['results']
    assert slots == planned_slots()
    assert plan['epochs'] == 20 and plan['batch_size'] == 64
    assert plan['optimizer'] == 'Adam' and plan['lr'] == .001
    assert plan['warmup_doses'] == [100,300,600] and plan['idle_seconds'] == 60
    assert len(slots) == len(results) == 18
    assert len({s['id'] for s in slots}) == 18
    assert Counter((r['seed'], r['warmup_steps']) for r in results) == Counter({(s,d):2 for s in (42,45,48) for d in (100,300,600)})
    launch = load(ART/'launch_identity.json')
    assert len(launch['controller']) == 2
    for ident in launch['controller']:
        assert 'compare_router_warmup_dose.py' in ident['CommandLine']
        assert ident['ExecutablePath'] and identity_ms(ident) > 0
    rows, hashes, identities = [], {}, []
    epochs = 0
    for index, (slot, row) in enumerate(zip(slots, results)):
        folder = ART/slot['id']
        assert all(row[k] == v for k,v in slot.items())
        assert load(folder/'result.json') == row
        assert row['status'] == 'complete' and not row['test_data_accessed']
        assert row['warmup_rng_restored'] and row['warmup']['steps'] == slot['warmup_steps']
        worker, launcher = load(folder/'worker_identity.json'), load(ART/(slot['id']+'.identity.json'))
        assert worker['ProcessId'] == row['pid']
        for ident in (worker, launcher):
            assert ident['CommandLine'].rstrip().endswith(f'--slot {index}')
            assert 'compare_router_warmup_dose.py' in ident['CommandLine']
            assert ident['ExecutablePath'].lower().endswith('python.exe')
        created = identity_ms(worker)
        started = datetime.fromisoformat(row['started_utc']).timestamp()*1000
        assert identity_ms(launcher) <= created <= started
        assert started-created < 120000
        identities.append(dict(slot=slot['id'], worker=worker, launcher=launcher,
            worker_creation_unix_ms=created,
            qualification='Worker self-captured CIM executable/command line/creation; launcher independently captured. Not a post-hoc live worker match.'))
        assert row['start_temperature_c'] <= 70 and not row['start_clock_reasons'] & 32
        for metric, images in ((row['fit_energy'],60000), (row['warmup']['energy'],64*slot['warmup_steps']),
                               (row['warmup']['setup_and_loop_energy'],64*slot['warmup_steps'])):
            for key in ('gpu_energy_j','energy_window_seconds','gpu_average_watts'):
                positive(metric[key])
            assert metric['energy_window_images'] == images
            assert math.isclose(metric['gpu_energy_j']/metric['energy_window_seconds'],metric['gpu_average_watts'],rel_tol=1e-9)
        assert len(row['epochs']) == 20
        for number, epoch in enumerate(row['epochs'],1):
            assert epoch['epoch'] == number and math.isfinite(epoch['bce']) and epoch['bce'] >= 0
            positive(epoch['training_seconds']); positive(epoch['checkpoint_seconds'])
            checkpoint = folder/f'epoch_{number:02d}.pth'
            marker = load(checkpoint.with_suffix('.json'))
            assert marker['status'] == 'complete' and marker['epoch'] == number and marker['seed'] == slot['seed']
            actual = digest(checkpoint)
            assert actual == marker['sha256']
            hashes[str(checkpoint)] = actual
            epochs += 1
        assert hashes[str(folder/'epoch_20.pth')] == row['final_checkpoint_sha256']
        traces = [json.loads(line) for line in (folder/'telemetry.jsonl').read_text().splitlines()]
        assert traces and all(t['temperature_c'] is not None and t['temperature_c'] < 85 for t in traces)
        stamps = [t['monotonic'] for t in traces]
        assert all(b > a for a,b in zip(stamps,stamps[1:]))
        fit = [t for t in traces if t['window'].startswith(('epoch_','fit_'))]
        assert fit
        fe, we, total = row['fit_energy'], row['warmup']['energy'], row['warmup']['setup_and_loop_energy']
        assert total['gpu_energy_j'] >= we['gpu_energy_j']
        rows.append(dict(id=slot['id'],seed=slot['seed'],dose=slot['warmup_steps'],block=slot['block'],position=slot['position'],
            fit_j=fe['gpu_energy_j'],fit_seconds=fe['energy_window_seconds'],warmup_loop_j=we['gpu_energy_j'],
            warmup_setup_loop_j=total['gpu_energy_j'],warmup_setup_loop_seconds=total['energy_window_seconds'],
            setup_loop_plus_fit_j=total['gpu_energy_j']+fe['gpu_energy_j'],
            final_bce=row['epochs'][-1]['bce'],loss_history=[e['bce'] for e in row['epochs']],
            initial_hash=row['initial_weights_sha256'],order_hash=row['first_order_sha256'],
            peak_c=max(t['temperature_c'] for t in traces),telemetry_samples=len(traces),
            max_telemetry_gap_s=max((b-a for a,b in zip(stamps,stamps[1:])), default=0),
            fit_thermal_samples=sum(bool(t['throttle_reasons']&32) for t in fit),
            fit_power_cap_samples=sum(bool(t['throttle_reasons']&4) for t in fit)))
    parity = {}
    for seed in (42,45,48):
        group = [r for r in rows if r['seed']==seed]
        assert len({r['initial_hash'] for r in group}) == len({r['order_hash'] for r in group}) == 1
        parity[str(seed)] = dict(initial_and_first_order_match=True,
            max_epoch_bce_spread=max(max(r['loss_history'][e] for r in group)-min(r['loss_history'][e] for r in group) for e in range(20)))
    summary = []
    for seed in (42,45,48):
        for dose in (100,300,600):
            group = [r for r in rows if r['seed']==seed and r['dose']==dose]
            summary.append(dict(seed=seed,dose=dose,**{key:interval([r[key] for r in group]) for key in ('fit_j','fit_seconds','warmup_setup_loop_j','setup_loop_plus_fit_j','final_bce')}))
    paired=[]
    for seed in (42,45,48):
        for block in (1,2):
            base=next(r for r in rows if r['seed']==seed and r['block']==block and r['dose']==100)
            for dose in (300,600):
                other=next(r for r in rows if r['seed']==seed and r['block']==block and r['dose']==dose)
                paired.append(dict(seed=seed,block=block,dose=dose,
                    fit_j_delta=other['fit_j']-base['fit_j'],fit_seconds_delta=other['fit_seconds']-base['fit_seconds'],
                    setup_loop_plus_fit_j_delta=other['setup_loop_plus_fit_j']-base['setup_loop_plus_fit_j']))
    OUT.mkdir(parents=True,exist_ok=True)
    receipt=dict(status='complete_audited_qualified',audit_completed=True,integrity_checks_passed=True,
        checked_utc=datetime.now(timezone.utc).isoformat(),fits=18,epochs=epochs,
        plan_sha256=digest(ART/'plan.json'),study_sha256=digest(ART/'study.json'),protected_hashes_verified=plan['protected_sha256'],
        checkpoint_hashes=hashes,worker_identities=identities,live_process_snapshot=live,
        rows=rows,seed_dose_summary=summary,paired_comparisons=paired,seed_parity=parity,
        peak_c=max(r['peak_c'] for r in rows),fit_thermal_samples=sum(r['fit_thermal_samples'] for r in rows),
        fit_power_cap_samples=sum(r['fit_power_cap_samples'] for r in rows),
        max_telemetry_gap_s=max(r['max_telemetry_gap_s'] for r in rows),
        qualifications=['Only two repeats per seed/dose; dose order not fully counterbalanced within every seed.',
          'NVML 1-second sampling is coarse for short phases; no continuous unsampled thermal guarantee.',
          'Warm-up setup+loop includes dummy initialization/deletion/RNG restoration, excludes earlier CUDA context/input preparation.',
          'Fit includes initialization, training, checkpoint/result writes; GPU-only gross energy, not whole-PC or deployment payback.',
          'CIM worker receipt captured from running worker, but controller did not independently capture actual redirected child; retained evidence stronger than original study, not tamper-proof provenance.',
          'Nondeterministic training BCE differences retained; no accuracy evaluation, recalibration, export or test-data use.'])
    (OUT/'audited_summary.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    lines=['# Longer router warm-up: completed audit','',
        '18/18 fits and 360/360 checkpoint hashes verified. All protected sources/data/weights match the frozen plan. Actual worker CIM receipts match the result PID, executable, exact slot command and creation-before-start chronology. All seed initialization/order hashes match. No measurements were rerun.','',
        '| Seed | Dummy steps | Fit GPU J (median) | Fit seconds | Warm-up setup+loop GPU J | Setup+loop + fit GPU J |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['seed']} | {s['dose']} | {s['fit_j']['median']:.2f} | {s['fit_seconds']['median']:.3f} | {s['warmup_setup_loop_j']['median']:.2f} | {s['setup_loop_plus_fit_j']['median']:.2f} |")
    lines += ['', 'Two repeats per seed/dose; JSON retains ranges, individual results and within-block comparisons against 100 steps.', '',
        f"Peak {receipt['peak_c']} C; fit-labelled samples: {receipt['fit_thermal_samples']} software-thermal, {receipt['fit_power_cap_samples']} power-cap. Maximum telemetry gap {receipt['max_telemetry_gap_s']:.3f} seconds.", '',
        '## Interpretation', '',
        'Compare paired fit and warm-up-inclusive boundaries separately. Extra dummy steps have an energy cost; a lower fit-only number is not a net saving. Results are descriptive of this session, not proof of a universally optimal warm-up length. Do not pool absolute numbers with the earlier cold/warm study as a matched experiment.', '',
        '## Qualifications', '', *['- '+q for q in receipt['qualifications']], '',
        'The original 36-fit study audit is also closed, qualified by missing actual-worker creation receipts; those historical records cannot be reconstructed. This later study does not erase that limitation.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4.3))
    for ax,key,title in zip(axes,('fit_j','fit_seconds','setup_loop_plus_fit_j'),('Fit-window GPU energy (J)','Fit-window duration (s)','Warm-up setup+loop + fit GPU energy (J)')):
        for seed,color in ((42,'#227c91'),(45,'#c66a32'),(48,'#76509b')):
            group=[s for s in summary if s['seed']==seed]
            ax.errorbar([s['dose'] for s in group],[s[key]['median'] for s in group],
                yerr=[[s[key]['median']-s[key]['minimum'] for s in group],[s[key]['maximum']-s[key]['median'] for s in group]],
                fmt='o-',capsize=3,label=f'Seed {seed}',color=color)
        ax.set_title(title,fontsize=9);ax.set_xticks([100,300,600]);ax.set_xlabel('Dummy warm-up steps');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('18 saved fits: medians/ranges of two repeats per seed/dose; GPU-only energy',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'warmup_dose_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps({k:receipt[k] for k in ('status','fits','epochs','peak_c','fit_thermal_samples','fit_power_cap_samples','max_telemetry_gap_s')}))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
