"""Saved-results-only figures, gated payback and final investigation index."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import json, math, sys
from pathlib import Path
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
sys.path.insert(0, str(PIPELINE/'artifacts/plot_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=PIPELINE/'reports/compact_b1_20261028/final_summary'
    out.mkdir(exist_ok=True)
    def read(study, filename='audited_summary.json'):
        p=PIPELINE/'reports'/study/filename
        return json.loads(p.read_text())
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','svg'): fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
    batch=read('batch8_grouping_20261029')['results']
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for mode,color in [('batch_choice','#4477aa'),('per_image','#ee7733')]:
        r=[next(x for x in batch if x['seed']==seed and x['mode']==mode) for seed in (42,45,48)]
        for ax,key in zip(axes,('accuracy','median_ms_per_image','median_gpu_j_per_image')):
            ax.plot([42,45,48],[x[key] for x in r],'o-',label=mode.replace('_',' '),color=color)
            ax.set_xticks([42,45,48]);ax.set_xlabel('Router seed');ax.grid(alpha=.2)
    axes[0].axhline(84.8,color='black',ls=':',label='1 pp accuracy gate')
    for ax,title in zip(axes,('Test accuracy (%)','Total latency (ms/image)','GPU energy (J/image)')):ax.set_title(title)
    axes[0].legend(fontsize=8);fig.suptitle('Batch 8: per-image grouping costs include gather, inference and scatter',y=1.03)
    save(fig,'batch8_grouping')
    deployment=read('supervision_deployment_20261029')
    training=read('router_supervision_screen_20261029','study.json')
    raw=read('supervision_deployment_20261029','study.json')
    controls={r['case']:r for r in raw['full_sweeps']}
    payback=[]
    for row in deployment['results']:
        fit=next((f for f in training['fits'] if f['name']==row['method']),None)
        energy=fit['gpu_fit_energy']['gpu_energy_j'] if fit else 270.437
        for baseline,control in [('reference',controls['reference']),('eligible',controls['eligible_'+str(row['calibration_images'])])]:
            delta=control['gpu_j_per_image']-row['gpu_j_per_image']
            valid=bool(row['accuracy_pass'] and delta>0)
            payback.append(dict(case=row['case'],baseline=baseline,training_gpu_j=energy,saved_gpu_j_per_image=delta,
                accuracy_pass=row['accuracy_pass'],valid=valid,payback_images=math.ceil(energy/delta) if valid else None,
                reason='valid GPU-fit-only estimate' if valid else ('accuracy failure' if not row['accuracy_pass'] else 'nonpositive GPU-energy saving')))
    atomic_json(out/'supervision_payback.json',dict(results=payback,scope='GPU fit only; init/checkpoints/internal tune included where used; supervision/calibration/export/CPU excluded. Separate training/inference sessions; descriptive estimates, not precise uncertainty.'))
    names=[f['name'] for f in training['fits']]+['original_control']
    labels=['Fit 1000\n20 epochs','Fit 2000\n20 epochs','Fit 3000\n20 epochs','Internal tune\nunweighted 60','Internal tune\nbalanced 60','Original\ncontrol']
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for n,color in [(500,'#228833'),(1000,'#4477aa'),(2000,'#cc6677')]:
        group=[next((r for r in deployment['results'] if r['method']==name and r['calibration_images']==n),None) for name in names]
        for ax,key in zip(axes,('accuracy_loss_pp','latency_saving_vs_eligible_pct','gpu_energy_saving_vs_eligible_pct')):
            ax.plot(range(6),[r[key] if r else np.nan for r in group],'o-',label=f'{n} calibration',color=color)
    for ax,title in zip(axes,('Accuracy loss (pp)','Latency saving vs eligible (%)','GPU-energy saving vs eligible (%)')):
        ax.set_title(title);ax.set_xticks(range(6),labels,fontsize=7,rotation=30,ha='right');ax.grid(alpha=.2)
    axes[0].axhline(1,color='black',ls=':');axes[1].axhline(0,color='black',ls=':');axes[2].axhline(0,color='black',ls=':')
    axes[0].legend(fontsize=7);fig.suptitle('Supervision deployment: seed 42, one full test sweep per predeclared setting',y=1.03)
    save(fig,'supervision_deployment')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for f in training['fits']:
        losses=[r['loss'] for r in training['results'] if r.get('kind')=='epoch' and r.get('fit')==f['name']]
        axes[0].plot(range(1,len(losses)+1),losses,label=f['name'])
    axes[0].set(xlabel='Epoch',ylabel='Training loss',title='Loss curves (balanced loss has different weighting)');axes[0].legend(fontsize=6)
    for baseline,color in [('reference','#4477aa'),('eligible','#ee7733')]:
        group=[r for r in payback if r['baseline']==baseline and r['valid']]
        axes[1].scatter(range(len(group)),[r['payback_images'] for r in group],label=f'{baseline}: {len(group)}/16 valid',color=color)
    axes[1].set(xlabel='Valid predeclared cases (not a ranking)',ylabel='Images to repay GPU fit energy',title='Accuracy + positive-energy gates required');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'supervision_training_payback')
    conditional=read('gpu_conditional_20261029');dispatch=read('dispatch_controls_20261029')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,source,title in [(axes[0],conditional['measurements'],'Two-branch conditional prototype'),(axes[1],dispatch['selection'],'Selection implementation control')]:
        keys=list(source); med=[source[k]['median_ms'] for k in keys]
        errors=np.array([[m-source[k]['range_ms'][0],source[k]['range_ms'][1]-m] for k,m in zip(keys,med)]).T
        ax.bar(range(len(keys)),med,color=['#4477aa','#ee7733','#999999'],yerr=errors,capsize=4)
        ax.set_xticks(range(len(keys)),[k.replace('_','\n') for k in keys],fontsize=8);ax.set(ylabel='Total latency (ms/image)',title=title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Validation-only engineering: matched final CPU class output; ranges are not confidence intervals',y=1.03)
    save(fig,'dispatch_conditional')
    lines=['# Final compact-routing investigation summary','',
        'All authorized bounded measurement investigations are complete. The cancelled multi-day batch-1 matrix remains cancelled (8 full sweeps plus 6 preparation/pilot rows); the new studies are not an equivalent full 213-choice/capacity-2 matrix.','',
        '## Main findings','',
        '- Compact resident screen: 5/15 router cases pass accuracy; none also saves both latency and GPU energy versus eligible static.',
        '- Switching confirmation: 7/15 accuracy passes. Seed42 resident graphs recorded 9.07% latency and 4.20% GPU-energy savings versus graph static. Cache-aware policies collapsed to constant R101 FP16; those gains are not dynamic-selection evidence.',
        '- Batch8 per-image grouping improves accuracy but costs 2.77–3.28 ms/image versus 0.77–0.97 for one choice per batch. Only seed42 per-image passes the 1-point allowance.',
        f"- Final supervision screen: {sum(r['accuracy_pass'] for r in deployment['results'])}/16 accuracy passes; no case combines accuracy compliance with positive latency and GPU-energy savings versus eligible static.",
        '- Genuine conditional execution verified only the selected branch runs, but measured 4.123 ms versus 3.955 ms CPU dispatch. It is an engineering prototype, not a validated 1-point routing policy.',
        '- Dedicated R34 INT8 remains quarantined: 40/100 changed probe classes; exact numerical cause unresolved. Original engines retained.',
        '', '## Coverage and audit index','']
    studies=['correctness_training_energy_20261028','gpu_residency_20261028','b1_engine_profiles_20261028','b1_int8_diagnostic_20261028','compact_validation_20261028','compact_runtime_profile_20261028','gpu_dispatch_20261028','compact_resident_screen_20261028','compact_switching_pilot_20261028','compact_switching_confirmation_20261028','router_supervision_screen_20261029','batch8_grouping_20261029','dispatch_controls_20261029','gpu_conditional_20261029','supervision_deployment_20261029']
    # Historical diagnostic had a prose audit but no machine-readable receipt.
    diagnostic_path=PIPELINE/'reports/b1_int8_diagnostic_20261028/study.json'
    diagnostic=json.loads(diagnostic_path.read_text())
    assert diagnostic['status']=='complete' and diagnostic['test_data_opened'] is False
    assert diagnostic['source_sha256']==digest(PIPELINE/'experiments/diagnose_b1_int8.py')
    dr={r['id']:r for r in diagnostic['results']}
    assert set(dr)=={'original_b1','original_b32','dedicated_b1'}
    assert len(set(diagnostic['validation_positions']))==100
    assert dr['original_b1']['predictions']==dr['original_b32']['predictions']
    assert sum(a!=b for a,b in zip(dr['original_b1']['predictions'],dr['dedicated_b1']['predictions']))==40
    profile=read('b1_engine_profiles_20261028','study.json')
    build=next(r for r in profile['results'] if r['id']=='build_resnet34__p0__int8')
    assert digest(build['engine_path'])==dr['dedicated_b1']['engine_sha256']==build['engine_sha256']
    def find_engine(value):
        if isinstance(value,dict):
            if value.get('name')=='resnet34__p0__int8' and 'engine_path' in value:return value
            for v in value.values():
                found=find_engine(v)
                if found:return found
        if isinstance(value,list):
            for v in value:
                found=find_engine(v)
                if found:return found
    original=find_engine(read('','gpu_static.json'))
    assert digest(original['engine_path'])==dr['original_b1']['engine_sha256']==dr['original_b32']['engine_sha256']
    atomic_json(diagnostic_path.parent/'audited_summary.json',dict(audit_passed=True,report_sha256=digest(diagnostic_path),rows=3,changed_predictions=40,
        note='Source/engine hashes and saved predictions audited. Raw logits were not retained by original diagnostic; recorded maximum-logit differences cannot independently be recomputed. Dedicated INT8 remains quarantined; numerical root cause unresolved.'))
    receipts={}
    for study in studies:
        p=PIPELINE/'reports'/study/'audited_summary.json'
        assert p.exists(), str(p)
        audit=json.loads(p.read_text());assert audit.get('audit_passed',False), str(p)
        receipts[str(p)]=digest(p);lines.append(f'- [{study}](../../{study}/audited_summary.json): audited.')
    lines+=['','## Scope and limitations','',
        'GPU-only energy excludes CPU and the rest of the PC. No verified CPU/system energy meter was available through the checked interfaces. No CPU-router training payback is claimed. Fourteen historical short setup-energy intervals remain unavailable, not zero; separate aggregate load/context/cleanup cycles were measured instead.',
        'Static timing varied substantially even for paired same-classifier controls. Separate sessions, one-repeat screens and three-repeat descriptive ranges do not justify general causal or precise uncertainty claims. All seeds, failures, negative savings and constant-policy collapse remain in the underlying tables.',
        'Training payback is training GPU joules divided by comparator-minus-routed GPU joules/image, only when accuracy passes and saving is positive. Training scope excludes supervision/calibration/export and is not whole-pipeline deployment cost. Existing larger-batch results and payback plots are reused without reruns under correctness_b1_20261028/existing_only.',
        'Batch-1 engine specialization, residency, host-byte caching, cleanup/reuse, switching-aware calibration, CUDA Graphs, selection precision/readback, cold/warm costs, fit/calibration quantity, loss/epochs, batch grouping and thermal/session controls were investigated. No new external data, classifier retraining, power changes or HPC work was performed.',
        'Final GPU-resident class output was not compared against host-output baselines: the same final host-class boundary was retained for fair comparisons.','',
        '## Figures','',*[f'![{n}]({n}.png)' for n in ('batch8_grouping','supervision_deployment','supervision_training_payback','dispatch_conditional')]]
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(out/'analysis_receipt.json',dict(audit_receipts=receipts,source_sha256=digest(Path(__file__)),plots_generated=True,visual_review_pending=True,regression_receipt_pending=True))
    print(str(out))


if __name__=='__main__':main()
