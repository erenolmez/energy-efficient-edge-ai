"""Version2 CPU audit: preserve original frozen audit and reconcile async edge samples."""
import sys
from pathlib import Path
if not __package__: sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import hashlib
import json
import math
import subprocess
from collections import Counter
from statistics import median
from datetime import datetime,timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.experiments.remaining_work_design import graph_windows,plateau_windows
from pc_tests.model_pipeline.analysis.audit_telemetry_boundaries_v2 import reconcile

ROOT=PIPELINE/'artifacts/remaining_work_20261104'
REPORT=PIPELINE/'reports/remaining_work_20261104_audit_v2'


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def positive(value):
    assert math.isfinite(value) and value>0,value


def audit_one(kind,live):
    tag='remaining_graph_performance_20261104' if kind=='graph' else 'remaining_plateau_20261104'
    art=PIPELINE/'artifacts'/tag;out=PIPELINE/'reports'/(tag+'_audit_v2');out.mkdir(parents=True,exist_ok=True)
    state,plan=load(art/'study.json'),load(art/'plan.json')
    assert state['status']=='complete_pending_audit',state.get('error')
    assert not state['test_data_accessed']
    expected=graph_windows() if kind=='graph' else plateau_windows()
    assert plan['windows']==expected and len(state['rows'])==len(expected)
    assert digest(art/'plan.json')==state['plan_sha256']
    for name,value in state['protected_hashes'].items(): assert digest(name)==value,name
    worker,launch=load(art/'worker_identity.json'),load(art/'launch_identity.json')
    assert worker['ProcessId']==state['pid'] and worker in launch['processes']
    for ident in launch['processes']:
        assert ident['ExecutablePath'] and ident['creation_unix_ms']>0
        assert launch['expected_script'] in ident['CommandLine']
        assert not any(all(row.get(k)==ident[k] for k in ('ProcessId','ExecutablePath','CommandLine','creation_unix_ms')) for row in live)
    refs=PIPELINE/'artifacts/dispatch_residency_v2_20261104'
    if kind=='graph':
        assert len(state['independent_parity'])==256
        assert Counter(r['mode'] for r in state['independent_parity'])=={'cuda_graph':128,'gpu_conditional':128}
        assert state['graph_node_counts']['parent'][1:]==[0,2]
        assert all(state['graph_node_counts'][k][0]>1 and state['graph_node_counts'][k][1:]==[0,0] for k in ('branchA','branchB'))
        for name,h in state['reference_sha256'].items(): assert digest(refs/name)==h
        expected_predictions=np.load(refs/'reference_predictions.npy',allow_pickle=False)
        expected_choices=np.load(refs/'reference_choices.npy',allow_pickle=False)
        for mode in ('cuda_graph','gpu_conditional'):
            parity=[r for r in state['independent_parity'] if r['mode']==mode]
            assert sorted(r['input_index'] for r in parity)==list(range(128))
            for p in parity:
                assert p['choice']==int(expected_choices[p['input_index']]) and p['prediction']==int(expected_predictions[p['input_index']])
                assert math.isfinite(p['max_abs_logit_error'])
    else:
        for r in state['reference_files'].values(): assert digest(r['path'])==r['sha256']
        assert all(v==[[1,3,128,128],[32,3,128,128],[128,3,128,128]] for v in state['engine_profiles'].values())
    rows=state['rows']
    for index,(r,spec) in enumerate(zip(rows,expected)):
        assert r['index']==index and all(r[k]==v for k,v in spec.items())
        assert r['accepted'] and r['prediction_mismatches']==0 and r['software_thermal_samples']==0
        assert r['timed_telemetry_samples']>0
        assert r['cooldown']['seconds']>=15 and r['cooldown']['reading']['temperature_c']<=55
        assert not r['cooldown']['reading']['clock_reasons']&32
        for key in ('seconds','ms_per_image','stream_span_ms'): positive(r[key])
        energy=r['gpu_energy']
        for key in ('gpu_energy_j','gpu_energy_j_per_image','gpu_average_watts','energy_window_seconds'): positive(energy[key])
        assert energy['energy_window_images']==r['requests']
        assert math.isclose(energy['gpu_energy_j']/r['requests'],energy['gpu_energy_j_per_image'],rel_tol=1e-9)
        assert math.isclose(energy['gpu_energy_j']/energy['energy_window_seconds'],energy['gpu_average_watts'],rel_tol=1e-9)
        assert math.isclose(r['seconds']*1000/r['requests'],r['ms_per_image'],rel_tol=1e-9)
        assert digest(r['predictions_file'])==r['predictions_file_sha256']
        observed=np.load(r['predictions_file'],allow_pickle=False)
        if kind=='graph':
            assert r['requests']==128 and observed.shape==(128,) and np.array_equal(observed,expected_predictions)
            if r['mode']=='cuda_graph': assert r['graph_branch_counts']==np.bincount(expected_choices,minlength=2).tolist()
        else:
            reference=np.load(state['reference_files'][f'{r["name"]}_b{r["batch"]}']['path'],allow_pickle=False)
            assert observed.shape==(r['cycles'],128) and r['requests']==128*r['cycles']
            assert np.all(observed==reference.argmax(1)[None,:])
            assert 1<=r['seconds']<=10
    telemetry=[json.loads(line) for line in (art/'telemetry.jsonl').read_text().splitlines()]
    assert telemetry and all('error' not in t and t['temperature_c']<80 for t in telemetry)
    timestamps=[datetime.fromisoformat(t['utc']).timestamp() for t in telemetry]
    gaps=[b-a for a,b in zip(timestamps,timestamps[1:])]
    assert gaps and min(gaps)>0 and max(gaps)<2
    timed=[t for t in telemetry if t['phase'].startswith('window:')]
    assert not any(t['clock_reasons']&32 for t in timed)
    reconciliations=[]
    for index,row in enumerate(rows):
        samples=[t for t in timed if t['phase']==f'window:{index}']
        reconciliations.append(reconcile(row,samples))
    result=dict(status='audited_qualified',audit_completed=True,kind=kind,windows=len(rows),requests=sum(r['requests'] for r in rows),
        exact_schedule_predictions_hashes_identities_verified=True,rows=rows,peak_c=max(t['temperature_c'] for t in telemetry),
        timed_powercap_samples=sum(r['conservative_powercap_samples'] for r in reconciliations),
        original_worker_powercap_samples=sum(r['software_powercap_samples'] for r in rows),timed_thermal_samples=0,
        telemetry_boundary_reconciliation=reconciliations,
        audit_revision='v2: phase-envelope samples conservatively retained; original snapshot counts unchanged',
        audit_source_sha256=digest(__file__),boundary_helper_sha256=digest(Path(__file__).with_name('audit_telemetry_boundaries_v2.py')),
        telemetry_samples=len(telemetry),max_telemetry_gap_seconds=max(gaps),
        checked_utc=datetime.now(timezone.utc).isoformat(),study_sha256=digest(art/'study.json'),plan_sha256=digest(art/'plan.json'),
        qualifications=['Gross NVML GPU energy includes idle/display and excludes CPU/restPC; short-window counter uncertainty.',
            'No hardware clock/power/fan intervention. Power-cap and frequency-ramp flags retained.',
            '128 repeated validation inputs and finite representative model subset; not deployment accuracy or a universal causal hardware claim.',
            'CPU submits; CUDA-event span includes host gaps. Every measured prediction saved and checked.',
            'Graph uses explicit CUDA If bodies around standalone engines; original invalid TensorRT capture timings excluded.'] if kind=='graph' else [
            'Fixed-model two-architecture FP16 diagnostic, not routed mix/cache/grouping benchmark; no universal plateau-cause claim.',
            'Same opt32 dynamic profile and128 repeated validation inputs; no dedicated batch-profile ablation or hardware intervention.',
            'GPUresident and host-to-logits boundaries differ intentionally and must not be pooled.',
            'Warmup immediately after cool-start gate, before timed window; sampled thermal/clock evidence does not establish unsampled behavior.',
            'Gross GPU-only energy, short intervals and CPU launch/sync instrumentation qualified.'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if kind=='graph':
        paired=[];lines=['# Matched CUDA Graph performance: completed audit','',
            '72 planned windows; six repeats per variant; all128 predictions/window and graph selected-only counters verified. Input/output boundaries are matched within each comparison. No invalid original graph speedups reused.','',
            '| Input / output | Graph vs baseline | Median latency change | Median GPU-energy change |','|---|---|---:|---:|']
        for residence in ('host','gpu'):
            for egress in ('per_image_cpu','deferred_gpu'):
                for baseline in ('cpu_dispatch','gpu_conditional'):
                    group=[]
                    for rep in range(6):
                        rr=[r for r in rows if r['repeat']==rep and r['residence']==residence and r['egress']==egress]
                        a=next(r for r in rr if r['mode']=='cuda_graph');b=next(r for r in rr if r['mode']==baseline)
                        group.append(dict(repeat=rep,residence=residence,egress=egress,baseline=baseline,
                            latency_change_pct=100*(a['seconds']/b['seconds']-1),
                            gpu_energy_change_pct=100*(a['gpu_energy']['gpu_energy_j']/b['gpu_energy']['gpu_energy_j']-1)))
                    paired.extend(group)
                    lines.append(f"| {residence} / {egress} | {baseline} | {median(r['latency_change_pct'] for r in group):+.2f}% | {median(r['gpu_energy_change_pct'] for r in group):+.2f}% |")
        result['paired_comparisons']=paired
        fig,axes=plt.subplots(1,2,figsize=(11,5.6))
        labels=[]
        for i,(res,eg,base) in enumerate(dict.fromkeys((r['residence'],r['egress'],r['baseline']) for r in paired)):
            group=[r for r in paired if (r['residence'],r['egress'],r['baseline'])==(res,eg,base)]
            labels.append(f'{res}/{eg}\nvs {base}')
            for ax,key in zip(axes,('latency_change_pct','gpu_energy_change_pct')):
                values=[r[key] for r in group];ax.scatter(values,[i]*6,s=15,alpha=.5);ax.plot(median(values),i,'kD',ms=5)
        for ax,title in zip(axes,('Wall latency change (%)','Gross GPU energy change (%)')):
            ax.axvline(0,color='gray');ax.set_title(title);ax.set_yticks(range(len(labels)),labels,fontsize=7);ax.grid(alpha=.2)
        fig.suptitle('CUDA Graph relative to matched baselines: negative = lower; dots = six paired repeats')
        fig.tight_layout();fig.savefig(out/'paired_graph_performance.png',dpi=180);plt.close(fig)
        lines+=['','Negative percentage means lower cost. All paired repeats retained in JSON; graph branch-counter instrumentation included. Native-If selected-only profiler evidence is from the protected earlier session, while current-session independent parity is rechecked.']
    else:
        groups=[];lines=['# Fixed-model batch-size follow-up: completed audit','',
            '36 balanced planned windows, three repeats per model/boundary/batch. Same128 inputs per cycle; all timed predictions verified against independent same-batch references. Original engines unchanged.','',
            '| Model | Boundary | Batch | Median ms/image | Median GPU mJ/image |','|---|---|---:|---:|---:|']
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for model_index,name in enumerate(('resnet101__p0__fp16','efficientnet_b0__p0__fp16')):
            for boundary in ('gpu_resident','host_to_logits'):
                values=[]
                for batch in (32,64,128):
                    group=[r for r in rows if r['name']==name and r['boundary']==boundary and r['batch']==batch]
                    item=dict(name=name,boundary=boundary,batch=batch,ms_image=median(r['ms_per_image'] for r in group),
                        gpu_mj_image=1000*median(r['gpu_energy']['gpu_energy_j_per_image'] for r in group),
                        min_ms_image=min(r['ms_per_image'] for r in group),max_ms_image=max(r['ms_per_image'] for r in group))
                    groups.append(item);values.append(item)
                    lines.append(f"| {name} | {boundary} | {batch} | {item['ms_image']:.4f} | {item['gpu_mj_image']:.3f} |")
                for ax,key,title in zip(axes[model_index],('ms_image','gpu_mj_image'),('ms/image','GPU mJ/image')):
                    ax.plot([v['batch'] for v in values],[v[key] for v in values],'o-',label=boundary)
                    ax.set_title(name+' / '+title,fontsize=9);ax.set_xticks([32,64,128]);ax.set_xlabel('Batch');ax.grid(alpha=.2);ax.legend(fontsize=8)
        result['groups']=groups;fig.suptitle('Fixed-model follow-up: medians of three repeats; GPU-only energy')
        fig.tight_layout();fig.savefig(out/'batch_plateau_followup.png',dpi=180);plt.close(fig)
        lines+=['','Interpretation must separate models and boundaries. This controls classifier identity, sample/order and warm-up while recording thermal/clock conditions; it does not isolate every hardware cause, nor test dedicated batch-size engine profiles. A null or mixed result is retained, not automatically retried.']
    lines+=['','## Preserved asynchronous telemetry boundary qualification','',
        'The original audit is retained unchanged. This separately versioned audit reconciles saved snapshot counts/flags against timestamp-supported edge candidates, while conservatively retaining ALL phase-labelled samples plus boundary readings for thermal/power qualification. Exact timer/snapshot membership is not reconstructed. No raw measurement, flag, source or plan was edited.',
        f"Original worker power-cap count {result['original_worker_powercap_samples']}; conservative phase-envelope count {result['timed_powercap_samples']}. Narrow25ms edge tolerance is a declared audit rule, not an assertion of timer precision. All other integrity checks remain required."]
    lines+=['',f"Peak {result['peak_c']}C; timed thermal flags0; timed power-cap samples {result['timed_powercap_samples']}; maximum telemetry gap {max(gaps):.3f}s.",'','## Qualifications','',*['- '+q for q in result['qualifications']]]
    (out/'audited_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(kind=kind,status=result['status'],windows=len(rows),report=str(out/'RESULTS.md'),plot=str(out/('paired_graph_performance.png' if kind=='graph' else 'batch_plateau_followup.png')))


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=('graph','plateau','all'),default='all');args=parser.parse_args()
    raw=subprocess.check_output(['powershell.exe','-NoProfile','-Command',
        '@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match \'^python(w)?\\.exe$\' } | Select-Object ProcessId,ExecutablePath,CommandLine,@{n="creation_unix_ms";e={[DateTimeOffset]::new($_.CreationDate).ToUnixTimeMilliseconds()}}) | ConvertTo-Json -Depth 5'],text=True)
    live=json.loads(raw) if raw.strip() else []
    if isinstance(live,dict): live=[live]
    assert not any(any(name in (r.get('CommandLine') or '') for name in ('measure_remaining_graph.py','measure_remaining_plateau.py')) for r in live),'GPU worker active'
    campaign_plan=load(ROOT/'plan.json')
    for p,h in campaign_plan['protected_sources'].items(): assert digest(p)==h,p
    results=[audit_one(kind,live) for kind in (('graph','plateau') if args.kind=='all' else (args.kind,))]
    REPORT.mkdir(parents=True,exist_ok=True)
    receipt=dict(status='audits_complete_pending_visual_review' if args.kind=='all' else args.kind+'_audit_complete_pending_visual_review',results=results,checked_utc=datetime.now(timezone.utc).isoformat(),
        hpc='Blocked on external scheduler/log evidence; no SSH/job intervention.',
        original_warmup='Audit closed, historical worker identity limitation irrecoverable.',
        longer_warmup='Audit closed qualified, no new rerun needed.',
        residency_switching='Closed qualified; order/capacity limitations retained.')
    (REPORT/'audited_summary.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
