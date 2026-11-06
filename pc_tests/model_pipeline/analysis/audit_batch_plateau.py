"""Audit isolated validation diagnostic and plot saved results; no CUDA work."""
from pathlib import Path
from statistics import median
from collections import Counter
import hashlib
import json
import math
import sys

REPO=Path(__file__).resolve().parents[3]
PIPE=REPO/'pc_tests/model_pipeline'
ART=PIPE/'artifacts/batch_plateau_20261103'
OUT=PIPE/'reports/batch_plateau_20261103'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def audit():
    state=json.loads((ART/'study.json').read_text())
    plan=json.loads((ART/'plan.json').read_text())
    assert state['status']=='complete_pending_audit'
    assert digest(ART/'plan.json')==state['plan_sha256']
    assert state['test_accessed'] is False
    for path, expected in plan['sources_sha256'].items():
        assert digest(path)==expected, path
    for name, expected in plan['data_sha256'].items():
        assert digest(PIPE/'artifacts/pc_precision/data_cache'/name)==expected,name
    records=json.loads((PIPE/'reports/gpu_static.json').read_text())['results']
    for r in records:
        p=Path(r['engine_path'])
        assert p.drive.upper()=='C:' and digest(p)==r['engine_sha256'],r['name']
    marker=json.loads((PIPE/'artifacts/pc_routers_correctness/global_correctness_seed42.json').read_text())
    for b in ('32','64','128'):
        e=marker['engines'][b]
        assert digest(e['engine'])==e['sha256']
    rows=state['results']
    ids={f'r{r}_b{b}_{m}' for r in range(3) for b in (32,64,128) for m in ('route','eligible','reference')}
    assert len(rows)==27 and {r['id'] for r in rows}==ids
    telemetry=[json.loads(x) for x in (ART/'telemetry.jsonl').read_text().splitlines()]
    assert telemetry and all(t['temperature_c'] is not None and t['temperature_c']<85 for t in telemetry)
    assert all(a['monotonic']<b['monotonic'] for a,b in zip(telemetry,telemetry[1:]))
    per_window=[]
    for row in rows:
        m=row['measurement']
        assert m['test_images']==512 # Legacy measurement key; validation inputs in this study.
        assert sum(m['selected_images'])==512
        assert sum(m['selected_batches'])==512//row['batch']
        assert m['runtime_engine_evictions']==0
        for key in ('gpu_energy_j','gpu_energy_j_per_image','energy_window_seconds','rgb_input_to_prediction_seconds'):
            assert math.isfinite(m[key]) and m[key]>0
        assert m['energy_window_images']>=512 and m['energy_window_images']%512==0
        assert math.isclose(m['gpu_energy_j']/m['energy_window_images'],m['gpu_energy_j_per_image'],rel_tol=1e-10)
        samples=[t for t in telemetry if t['window']==row['id'] and row['start_monotonic']<=t['monotonic']<=row['end_monotonic']]
        assert samples, row['id']
        per_window.append(dict(id=row['id'],batch=row['batch'],mode=row['mode'],repeat=row['repeat'],
                               first_sweep_ms_image=m['rgb_input_to_prediction_seconds']*1000/512,
                               energy_window_ms_image=m['energy_window_seconds']*1000/m['energy_window_images'],
                               gpu_mj_image=m['gpu_energy_j_per_image']*1000,
                               validation_accuracy=m['test_accuracy'],loads=m['runtime_engine_loads'],
                               evictions=m['runtime_engine_evictions'],telemetry_samples=len(samples),
                               sm_clock_mhz_range=[min(t['sm_clock_mhz'] for t in samples),max(t['sm_clock_mhz'] for t in samples)],
                               thermal_flag_samples=sum(bool(t['throttle_reasons']&32) for t in samples),
                               power_flag_samples=sum(bool(t['throttle_reasons']&4) for t in samples)))
    for b in (32,64,128):
        rr=[r for r in rows if r['mode']=='route' and r['batch']==b]
        assert all(r['measurement']['selected_images']==rr[0]['measurement']['selected_images'] for r in rr)
    groups=[]
    for b in (32,64,128):
        for mode in ('route','eligible','reference'):
            rr=[r for r in per_window if r['batch']==b and r['mode']==mode]
            item=dict(batch=b,mode=mode)
            for metric in ('first_sweep_ms_image','energy_window_ms_image','gpu_mj_image','validation_accuracy'):
                v=[r[metric] for r in rr]
                item[metric]=dict(median=median(v),minimum=min(v),maximum=max(v))
            groups.append(item)
    receipt=dict(audit_passed=True,windows=27,source_data_router_and_213_classifier_hashes_verified=True,
                 all_choice_counts_and_repeat_consistency_verified=True,raw_study_sha256=digest(ART/'study.json'),
                 plan_sha256=digest(ART/'plan.json'),telemetry_sha256=digest(ART/'telemetry.jsonl'),
                 telemetry_samples=len(telemetry),temperature_range_c=[min(t['temperature_c'] for t in telemetry),max(t['temperature_c'] for t in telemetry)],
                 reason_bitmask_counts=dict(Counter(t['throttle_reasons'] for t in telemetry)),
                 software_thermal_flag_samples=sum(bool(t['throttle_reasons']&32) for t in telemetry),
                 software_power_cap_flag_samples=sum(bool(t['throttle_reasons']&4) for t in telemetry),
                 sampler_max_gap_seconds=max(b['monotonic']-a['monotonic'] for a,b in zip(telemetry,telemetry[1:])),
                 measurement_marker_limitation='Window IDs and start/end wrap warm-up and measured loop together. No exact measured-loop NVML timestamps were saved; do not attribute a sample to kernel execution or compute per-phase energy.',
                 no_saved_prediction_arrays='Legacy routine retained accuracy and model counts, not per-image prediction arrays; diagnostic only, not a new full accuracy audit.',
                 decoded_flags_source='https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html',
                 gpu_only=True,test_data_accessed=False,groups=groups,per_window=per_window)
    (OUT/'diagnostic_audit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt,telemetry


def outputs(receipt,telemetry):
    groups=receipt['groups']
    lines=['# PC routed batch plateau: validation diagnostic', '',
           'Audit passed27/27 windows; original source/data/router and213 classifier-engine hashes verified. Three repeats, fixed seed42, original policies,213 choices/cache2,512 validation images. No test data or training. All results retained.', '',
           '| Batch | Path | First-sweep ms/image | Energy-window ms/image | GPU mJ/image |',
           '|---|---|---:|---:|---:|']
    labels=dict(route='Full routing',eligible='Eligible static',reference='R101 FP32')
    for g in groups:
        lines.append(f"| {g['batch']} | {labels[g['mode']]} | {g['first_sweep_ms_image']['median']:.4f} | {g['energy_window_ms_image']['median']:.4f} | {g['gpu_mj_image']['median']:.3f} |")
    lines+=['', 'Table values are medians across3 repeats; full min/max and individual windows are in diagnostic_audit.json.', '',
            '## What this establishes', '',
            'The historical October29 curve worsens after32. In this new short diagnostic the routed first-sweep medians decrease slightly from32 to128, with overlapping repeat ranges, while GPU energy is lower at64 than32/128. That difference is retained, not forced to reproduce the historical curve.',
            'All measured windows have zero evictions. Routed paths load two models initially and static paths one; the two-resident cache is not repeatedly thrashing in these windows.',
            f"GPU telemetry spans {receipt['temperature_range_c'][0]}–{receipt['temperature_range_c'][1]}C. Software-thermal slowdown bit32 appears in {receipt['software_thermal_flag_samples']}/{receipt['telemetry_samples']} samples, and software-power-cap bit4 in {receipt['software_power_cap_flag_samples']}/{receipt['telemetry_samples']}. They may overlap; idle/setup/warm-up samples are included.",
            'These are explicit reported clock-limiting reasons, stronger evidence than temperature alone. The85C safety latch did not fire; a GPU can reduce clocks well below that emergency threshold. We did not alter clocks, power limits or fan settings.',
            'Clock/power/thermal conditions vary during the session. This supports treating the original plateau as hardware/session-sensitive, but does not isolate a sole causal mechanism or prove that the old session had the same flags.', '',
            '## Important measurement boundaries', '',
            'First-sweep latency includes initial measured engine loads after20 warm-up batches because the historical routine clears the cache before measurement. GPU energy covers repeated sweeps for at least3 seconds; initial loading is amortized differently. Therefore do not divide GPU energy by first-sweep latency to infer power or combine them as one steady-state sample.',
            'The energy-window ms/image column uses its own actual duration/image denominator. Lower energy with higher first-sweep latency can reflect this boundary difference. It is not evidence of free routing overhead.',
            'Telemetry window IDs encompass warm-up plus measurement, not exact kernel or energy subphases. One-second sampling cannot resolve millisecond transfers or kernels. Selected-engine CUDA-event spans can include host/cache waits.',
            'Raw accuracy keys inherited the name test_accuracy, but these inputs are validation images, partly previously used in fitting/calibration. No held-out accuracy, test-based selection, deployment-gain or payback claim is made.',
            'No synchronized CPU/full-PC energy, PCIe bandwidth or memory-bandwidth counter was measured. Further causal experiments would need explicit warm/cold and phase markers, fixed model schedules and matched telemetry conditions; not automatically launched.', '',
            '## Separate next protocol', '',
            'WARMUP_PROTOCOL.md prepares the seed42 router-training consistency question. No training repeat has been launched.']
    (OUT/'DIAGNOSTIC_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sys.path.insert(0,str(PIPE/'artifacts/plot_dependencies'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.6))
    for mode,color in [('route','#267b80'),('eligible','#366db0'),('reference','#d57937')]:
        for ax,metric,title in zip(axes,('first_sweep_ms_image','energy_window_ms_image','gpu_mj_image'),
                                    ('First sweep incl. initial loads\n(ms/image)','Aggregate energy window\n(ms/image)','GPU energy\n(mJ/image)')):
            rr=[g for g in groups if g['mode']==mode]
            med=[g[metric]['median'] for g in rr]
            ax.errorbar([g['batch'] for g in rr],med,yerr=[[g[metric]['median']-g[metric]['minimum'] for g in rr],[g[metric]['maximum']-g[metric]['median'] for g in rr]],fmt='o-',color=color,label=labels[mode],capsize=3)
            ax.set_xticks([32,64,128]);ax.set_xscale('log',base=2);ax.set_xticks([32,64,128],['32','64','128'])
            ax.set_title(title,fontsize=10);ax.set_xlabel('Input batch size');ax.grid(alpha=.2)
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='upper center',ncol=3,frameon=False)
    fig.suptitle('Validation diagnostic: seed42, frozen policies,213 choices/cache2; median/range of3 repeats',y=.89,fontsize=10)
    fig.tight_layout(rect=(0,0,1,.80));fig.savefig(OUT/'diagnostic_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True)
    origin=telemetry[0]['monotonic'];x=[t['monotonic']-origin for t in telemetry]
    axes[0].plot(x,[t['temperature_c'] for t in telemetry],color='#c36b34');axes[0].set_ylabel('GPU temperature (C)')
    axes[1].plot(x,[t['sm_clock_mhz'] for t in telemetry],color='#267b80');axes[1].set_ylabel('SM clock (MHz)')
    for bit,label,color in [(32,'Software thermal flag','#c55353'),(4,'Software power-cap flag','#426ca7')]:
        axes[2].scatter([v for v,t in zip(x,telemetry) if t['throttle_reasons']&bit],[1 if bit==32 else .5 for t in telemetry if t['throttle_reasons']&bit],label=label,s=10,color=color)
    axes[2].set_yticks([]);axes[2].set_ylim(.2,1.3);axes[2].legend(loc='upper left');axes[2].set_xlabel('Seconds from first telemetry sample (setup + warm-up + measured loops)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Read-only GPU telemetry: reported clock limiting, not a phase-resolved causal estimate',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'telemetry.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    receipt,telemetry=audit()
    outputs(receipt,telemetry)
    print(json.dumps({k:receipt[k] for k in ('audit_passed','windows','telemetry_samples','temperature_range_c','software_thermal_flag_samples','software_power_cap_flag_samples','sampler_max_gap_seconds')}))
