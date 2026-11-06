"""Saved36-fit audit and paired summaries; no GPU or test-data work."""
from pathlib import Path
from collections import Counter
from statistics import median
import hashlib
import json
import math
import sys

REPO=Path(__file__).resolve().parents[3]
PIPE=REPO/'pc_tests/model_pipeline'
ART=PIPE/'artifacts/router_warmup_20261103'
OUT=PIPE/'reports/router_warmup_20261103'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def interval(values):
    return dict(median=median(values),minimum=min(values),maximum=max(values))


def audit():
    state=json.loads((ART/'study.json').read_text())
    plan=json.loads((ART/'plan.json').read_text())
    assert state['status']=='complete_pending_audit' and not state['test_data_accessed']
    assert digest(ART/'plan.json')==state['plan_sha256']
    for p,h in plan['protected_sha256'].items():
        assert digest(p)==h,p
    rows=state['results']
    assert len(rows)==36 and [r['id'] for r in rows]==[r['id'] for r in plan['slots']]
    assert set(Counter((r['seed'],r['condition']) for r in rows).values())=={6}
    audited=[]
    checkpoints=0
    identity_gaps=[]
    for row,slot in zip(rows,plan['slots']):
        directory=ART/row['id']
        assert all(row[k]==slot[k] for k in slot)
        actual=json.loads((directory/'result.json').read_text())
        assert actual==row and row['status']=='complete'
        assert len(row['epochs'])==20 and [e['epoch'] for e in row['epochs']]==list(range(1,21))
        assert not row['test_data_accessed']
        identity=json.loads((ART/(row['id']+'.identity.json')).read_text())
        assert 'compare_router_warmup.py' in identity['CommandLine'] and '--slot' in identity['CommandLine'] and identity['CreationDate']
        if identity['ProcessId']!=row['pid']:
            identity_gaps.append(dict(slot=row['id'],recorded_launcher_pid=identity['ProcessId'],
                                      self_reported_worker_pid=row['pid'],
                                      limitation='Windows venv launcher identity captured; actual redirected worker creation timestamp/executable not independently captured. Do not infer exact worker identity from PID alone.'))
        assert row['start_temperature_c']<=70 and not(row['start_clock_reasons']&32)
        m=row['fit_energy']
        for k in ('gpu_energy_j','energy_window_seconds','gpu_average_watts'):
            assert math.isfinite(m[k]) and m[k]>0
        assert m['energy_window_images']==60000
        assert math.isclose(m['gpu_energy_j']/m['energy_window_seconds'],m['gpu_average_watts'],rel_tol=1e-9)
        for e in row['epochs']:
            assert math.isfinite(e['bce']) and e['bce']>=0 and e['training_seconds']>0 and e['checkpoint_seconds']>0
            checkpoint=directory/f'epoch_{e["epoch"]:02d}.pth'
            marker=json.loads(checkpoint.with_suffix('.json').read_text())
            assert marker['epoch']==e['epoch'] and marker['seed']==row['seed'] and digest(checkpoint)==marker['sha256']
            checkpoints+=1
        assert digest(directory/'epoch_20.pth')==row['final_checkpoint_sha256']
        if row['condition']=='warm':
            assert row['warmup_rng_restored'] and row['warmup']['steps']==100
        else:
            assert row['warmup']['status']=='not_applicable_cold_control'
        traces=[json.loads(t) for t in (directory/'telemetry.jsonl').read_text().splitlines()]
        assert traces and all(t['temperature_c'] is not None and t['temperature_c']<85 for t in traces)
        assert any(t['window'].startswith('epoch_') for t in traces)
        item={k:row[k] for k in ('id','block','position','seed','condition','start_temperature_c','initial_weights_sha256','first_order_sha256')}
        warm=row['warmup'].get('energy',{})
        item.update(fit_j=m['gpu_energy_j'],fit_seconds=m['energy_window_seconds'],mean_gpu_watts=m['gpu_average_watts'],
                    checkpoint_seconds=sum(e['checkpoint_seconds'] for e in row['epochs']),
                    final_bce=row['epochs'][-1]['bce'],loss_history=[e['bce'] for e in row['epochs']],
                    warmup_loop_j=warm.get('gpu_energy_j'),warmup_loop_seconds=row['warmup'].get('seconds'),
                    warmup_energy_unavailable=warm.get('energy_unavailable_reason'),
                    peak_gpu_c=max(t['temperature_c'] for t in traces),telemetry_samples=len(traces),
                    fit_thermal_flag_samples=sum(bool(t['throttle_reasons']&32) for t in traces if t['window'].startswith(('epoch_','fit_'))),
                    fit_power_cap_samples=sum(bool(t['throttle_reasons']&4) for t in traces if t['window'].startswith(('epoch_','fit_'))))
        audited.append(item)
    parity={}
    for seed in (42,45,48):
        rr=[r for r in audited if r['seed']==seed]
        assert len({r['initial_weights_sha256'] for r in rr})==1
        assert len({r['first_order_sha256'] for r in rr})==1
        parity[str(seed)]=dict(initial_weights_match=True,first_epoch_order_match=True,
                              final_bce_range=interval([r['final_bce'] for r in rr]),
                              max_epoch_bce_spread=max(max(r['loss_history'][i] for r in rr)-min(r['loss_history'][i] for r in rr) for i in range(20)))
    groups=[]
    for seed in (42,45,48):
        for condition in ('cold','warm'):
            rr=[r for r in audited if r['seed']==seed and r['condition']==condition]
            groups.append(dict(seed=seed,condition=condition,**{k:interval([r[k] for r in rr]) for k in ('fit_j','fit_seconds','mean_gpu_watts','checkpoint_seconds','final_bce')}))
    pairs=[]
    for block in range(1,7):
        for seed in (42,45,48):
            cold=next(r for r in audited if r['block']==block and r['seed']==seed and r['condition']=='cold')
            warm=next(r for r in audited if r['block']==block and r['seed']==seed and r['condition']=='warm')
            pairs.append(dict(block=block,seed=seed,position=cold['position'],
                              fit_j_warm_minus_cold=warm['fit_j']-cold['fit_j'],
                              fit_seconds_warm_minus_cold=warm['fit_seconds']-cold['fit_seconds'],
                              fit_energy_reduction_pct=100*(1-warm['fit_j']/cold['fit_j']),
                              warm_fit_plus_dummy_loop_minus_cold_j=None if warm['warmup_loop_j'] is None else warm['fit_j']+warm['warmup_loop_j']-cold['fit_j']))
    order=[]
    for position in (1,2,3):
        for condition in ('cold','warm'):
            rr=[r for r in audited if r['position']==position and r['condition']==condition]
            order.append(dict(position=position,condition=condition,fit_j=interval([r['fit_j'] for r in rr]),fit_seconds=interval([r['fit_seconds'] for r in rr])))
    receipt=dict(audit_passed=not identity_gaps,numerical_and_checkpoint_audit_passed=True,
                 status='qualified_identity_provenance_gap' if identity_gaps else 'complete_audited',
                 exact_worker_identity_audit_passed=not identity_gaps,identity_gaps=identity_gaps,
                 fits=36,epochs=checkpoints,plan_sha256=digest(ART/'plan.json'),study_sha256=digest(ART/'study.json'),
                 protected_sources_data_old_artifacts_unchanged=True,seed_initialization_and_order_parity=parity,
                 peak_gpu_c=max(r['peak_gpu_c'] for r in audited),thermal_stop=False,
                 fit_thermal_flag_samples=sum(r['fit_thermal_flag_samples'] for r in audited),
                 fit_power_cap_flag_samples=sum(r['fit_power_cap_samples'] for r in audited),
                 warmup_loop_unavailable=sum(r['condition']=='warm' and r['warmup_loop_j'] is None for r in audited),
                 rows=audited,seed_condition_summary=groups,paired_comparisons=pairs,order_position_summary=order,
                 qualification='Fresh-process first-use control, not cold hardware; warm-up primes CPU/optimizer/runtime and GPU together. GPU-only fit energy, instrumentation differs from historical. Dummy-loop energy excludes dummy model/optimizer initialization and setup; no complete warm-up or whole-PC energy saving claim. Initial/order hashes match, but nondeterministic training loss differences retained. No test/calibration/export/deployment accuracy or payback claim.')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'audited_summary.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def outputs(receipt):
    groups=receipt['seed_condition_summary']
    lines=['# Router-training warm-up consistency: audited results', '',
           '36/36 fits,720/720 epoch checkpoints and original source/data/weight/report hashes verified. Initial model weights and first-epoch input order match for every seed across both conditions. No historical result replaced. Numerical integrity checks pass; the overall audit is QUALIFIED, not fully passed, because exact redirected-worker Windows creation identities were not saved.', '',
           '| Seed | Cold-process fit J | Warmed fit J | Cold fit seconds | Warmed fit seconds |',
           '|---|---:|---:|---:|---:|']
    for seed in (42,45,48):
        cold=next(g for g in groups if g['seed']==seed and g['condition']=='cold')
        warm=next(g for g in groups if g['seed']==seed and g['condition']=='warm')
        lines.append(f"| {seed} | {cold['fit_j']['median']:.2f} | {warm['fit_j']['median']:.2f} | {cold['fit_seconds']['median']:.3f} | {warm['fit_seconds']['median']:.3f} |")
    lines+=['', 'Medians of6 fits per seed/condition. Cold means a fresh process with no dummy training, not a power-cycled GPU. All repeat ranges and paired differences are retained in audited_summary.json.', '',
            '## Interpretation', '',
            'Standardized warm-up reduces measured fit-window time/energy across seeds in this controlled session. Seed42 is not consistently the most expensive seed once order/conditions are balanced. This supports a first-use/runtime-initialization explanation for part of the earlier anomaly, rather than treating seed42 as intrinsically inefficient.',
            'The warm-up constructs a disposable model and Adam optimizer before the fit. It primes CPU/Python/optimizer initialization as well as CUDA kernels and GPU allocation caches. Therefore the result does not isolate a specifically GPU-only warm-up mechanism.',
            'The historical270.437/201.185/195.008J remain separate session measurements. New checkpoint/result/hash instrumentation and fresh-process design differ from the old shared process; do not replace the old payback numerator or claim an exact retrospective correction.', '',
            '## Warm-up is not free', '',
            'Fit-window savings exclude the preceding dummy warm-up. Its100-step loop time/energy is retained separately and added in paired comparisons where available, but dummy model/optimizer initialization and setup are outside that loop counter. Those sums are incomplete full-setup costs, not complete deployment/training energy savings.',
            'All energy is gross reported GPU energy, including idle/display load. CPU, storage and the rest of the PC are unmeasured.', '',
            '## Controls and limitations', '',
            f"Peak GPU temperature was{receipt['peak_gpu_c']}C. Fit-labelled telemetry recorded{receipt['fit_thermal_flag_samples']} software-thermal and{receipt['fit_power_cap_flag_samples']} software-power-cap samples. No85C safety stop occurred; no hardware settings changed.",
            'Initial weights and first data order match, but floating-point/nondeterministic loss differences can remain; their maximum epoch spread is reported per seed. This is not an accuracy evaluation or a best-router search.',
            f"Identity provenance gap: {len(receipt['identity_gaps'])}/36 slot receipts captured the Windows venv launcher rather than the redirected worker PID recorded inside result.json. Script/slot/launcher creation data are present, but actual worker creation timestamps cannot be recovered from these receipts. No rerun or retrospective identity fabrication is performed; a fully identity-audited repeat would need corrected capture and separate approval.",
            'Seed x condition and order-position summaries are descriptive, with6 repetitions per cell, not a final causal guarantee. Starting thermal/clock conditions and fixed60-second idle intervals are retained. Full per-phase energy and complete setup energy were not measured.', '',
            'Recommendation: standardize and disclose warm-up and measured boundaries in future router-energy studies, balance seed order, and continue reporting setup/warm-up separately. No extra measurements are launched by this analysis.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sys.path.insert(0,str(PIPE/'artifacts/plot_dependencies'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,('fit_j','fit_seconds'),('GPU fit-window energy (J)','Fit-window duration (s)')):
        for condition,color,offset in [('cold','#d57937',-.12),('warm','#267b80',.12)]:
            for index,seed in enumerate((42,45,48)):
                g=next(g for g in groups if g['seed']==seed and g['condition']==condition)[metric]
                ax.errorbar(index+offset,g['median'],yerr=[[g['median']-g['minimum']],[g['maximum']-g['median']]],fmt='o',capsize=4,color=color,label=('Fresh process, no dummy warm-up' if condition=='cold' else 'After100 dummy steps') if index==0 else None)
        ax.set_xticks(range(3),['42','45','48']);ax.set_xlabel('Router seed');ax.set_title(title);ax.grid(alpha=.2)
    handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='upper center',ncol=2,frameon=False)
    fig.suptitle('20epochs per fit; median/range of6 fits per condition. Warm-up energy excluded here.',y=.88,fontsize=10)
    fig.tight_layout(rect=(0,0,1,.8));fig.savefig(OUT/'warmup_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    colors={42:'#267b80',45:'#d57937',48:'#526bb0'}
    for seed in (42,45,48):
        rr=[r for r in receipt['paired_comparisons'] if r['seed']==seed]
        for ax,metric,title in zip(axes,('fit_j_warm_minus_cold','fit_seconds_warm_minus_cold'),('Paired warmed minus cold energy (J)','Paired warmed minus cold duration (s)')):
            ax.plot([r['block'] for r in rr],[r[metric] for r in rr],'o-',color=colors[seed],label=f'Seed{seed}')
            ax.axhline(0,color='gray',lw=1);ax.set_title(title,fontsize=10);ax.set_xlabel('Balanced-order block');ax.set_xticks(range(1,7));ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Fit-window differences only: negative is lower after warm-up; dummy warm-up costs separate',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'paired_differences.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    receipt=audit()
    outputs(receipt)
    print(json.dumps({k:receipt[k] for k in ('audit_passed','fits','epochs','peak_gpu_c','fit_thermal_flag_samples','fit_power_cap_flag_samples','warmup_loop_unavailable')}))
    print(json.dumps(receipt['seed_condition_summary'],indent=2))
