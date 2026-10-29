"""Build presentation assets from saved results only; never launches inference."""
import hashlib,json,math,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
PIPE=REPO/'pc_tests/model_pipeline'
sys.path.insert(0,str(PIPE/'artifacts/plot_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
IMAGES=ROOT/'images';IMAGES.mkdir(exist_ok=True)
sources={}
def read(relative):
    path=PIPE/'reports'/relative;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
old=read('correctness_b1_20261028/existing_only/combined_summary.json')
assert old['audit_passed'] and len(old['rows'])==120
compact=read('compact_resident_screen_20261028/audited_summary.json')
assert compact['audit_passed']
training=read('correctness_training_energy_20261028/audited_summary.json')
resident_raw=read('compact_resident_screen_20261028/study.json')
deployment=read('supervision_deployment_20261029/audited_summary.json')
payback=read('compact_b1_20261028/final_summary/supervision_payback.json')
switching=read('compact_switching_confirmation_20261028/audited_summary.json')
assert switching['audit_passed'] and switching['full_sweeps']==81
for date in ('2026-10-13','2026-10-20'):
    p=ROOT.parent/date/'progress_report.tex';sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('uniform_batches_savings.tex','p80_savings.tex'):
    p=ROOT.parent/'2026-10-20'/name;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for r in compact['results']:
    if r['method']=='tinycnn':rows.append(dict(batch=1,seed=r['seed'],accuracy=r['accuracy'],passed=r['accuracy_pass'],ms_image=r['ms_per_image'],ms_batch=r['ms_per_image'],j_image=r['gpu_j_per_image']))
for r in old['rows']:
    if r['objective']=='latency' and r['budget_pp']==1:
        rows.append(dict(batch=r['batch_size'],seed=r['seed'],accuracy=r['accuracy'],passed=r['accuracy_passed'],ms_image=r['route_ms_per_image'],ms_batch=r['route_ms_per_batch'],j_image=r['route_j_per_image']))
assert len(rows)==18 and {(r['batch'],r['seed']) for r in rows}=={(b,s) for b in (1,8,16,32,64,128) for s in (42,45,48)}
batches=[1,8,16,32,64,128]
baseline_rows=[]
for baseline,method in [('eligible_static','eligible_static'),('reference_fp32','reference')]:
    for r in compact['results']:
        if r['method']==method:
            baseline_rows.append(dict(series=baseline,batch=1,seed=r['seed'],accuracy=r['accuracy'],ms_image=r['ms_per_image'],j_image=r['gpu_j_per_image']))
    for r in old['rows']:
        if r['objective']=='latency' and r['budget_pp']==1:
            control=r[baseline]
            baseline_rows.append(dict(series=baseline,batch=r['batch_size'],seed=r['seed'],accuracy=control['accuracy'],ms_image=control['ms_per_image'],j_image=control['j_per_image']))
assert len(baseline_rows)==36
series=[('learned','Learned TinyCNN','#187f80','o',-.14),('eligible_static','Eligible static','#4267aa','s',0),('reference_fp32','R101 FP32 reference','#db783c','^',.14)]
fig,axes=plt.subplots(1,3,figsize=(11.2,2.7))
for ax,key,title,scale in zip(axes,['accuracy','ms_image','j_image'],['Test accuracy (%)','Latency (ms/image)','GPU energy (mJ/image)'],[1,1,1000]):
    for name,label,color,marker,offset in series:
        source=rows if name=='learned' else [r for r in baseline_rows if r['series']==name]
        for b in batches:
            values=[r[key]*scale for r in source if r['batch']==b];assert len(values)==3
            med=np.median(values)
            ax.errorbar(b*2**offset,med,yerr=[[med-min(values)],[max(values)-med]],fmt=marker,mfc='white' if b==1 else color,mec=color,ecolor=color,capsize=2.5,markersize=4,label=label if b==1 else None)
    ax.set_xscale('log',base=2);ax.set_xticks(batches,['1*','8','16','32','64','128']);ax.set_xlabel('Input batch size');ax.set_title(title);ax.grid(alpha=.2)
    if key!='accuracy':ax.set_yscale('log')
    else:
        ax.axhline(84.8,color='#999999',ls=':',lw=1)
        ax.axhline(83.8,color='#999999',ls='--',lw=1)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.015),fontsize=9)
fig.tight_layout(rect=(0,0,1,.90));fig.savefig(IMAGES/'batch_comparison.pdf');fig.savefig(IMAGES/'batch_comparison.png',dpi=180);plt.close(fig)
lines=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Batch & Protocol & Accuracy (\%) & ms/batch & GPU mJ/batch & Pass 1 pp & Pass 2 pp\\',r'\midrule']
def interval(v,precision=2):return f'{min(v):.{precision}f}--{max(v):.{precision}f}'
for b in batches:
    rr=[r for r in rows if r['batch']==b]
    lines.append(f"{b} & {'6 resident' if b==1 else '213/cache2'} & {interval([r['accuracy'] for r in rr])} & {interval([r['ms_batch'] for r in rr])} & {interval([r['j_image']*1000*10000/math.ceil(10000/b) for r in rr],1)} & {sum(r['passed'] for r in rr)}/3 & {sum(r['accuracy']>=83.8-1e-10 for r in rr)}/3"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(ROOT/'batch_statistics.tex').write_text('\n'.join(lines),encoding='utf-8')
lines=[r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',r'Router & Accuracy (\%) & ms/image & GPU mJ/image & Pass 1 pp & Pass 2 pp & Device\\',r'\midrule']
for key,label in [('tree_d3','Depth 3 tree'),('logistic','Logistic'),('thumbnail_mlp16','Thumbnail MLP'),('boosted_d2','Boosted depth 2'),('tinycnn','TinyCNN control'),('eligible_static','Eligible static')]:
    rr=[r for r in compact['results'] if r['method']==key];assert len(rr)==3
    lines.append(f"{label} & {interval([r['accuracy'] for r in rr])} & {interval([r['ms_per_image'] for r in rr])} & {interval([1000*r['gpu_j_per_image'] for r in rr],1)} & {sum(r['accuracy_pass'] for r in rr)}/3 & {sum(r['accuracy_loss_pp']<=2+1e-10 for r in rr)}/3 & {'GPU' if key=='tinycnn' else ('--' if key=='eligible_static' else 'CPU')}"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(ROOT/'compact_statistics.tex').write_text('\n'.join(lines),encoding='utf-8')
lines=[r'\begin{tabularx}{\linewidth}{@{}Xrrrrc@{}}',r'\toprule',r'TinyCNN runtime mode & ms/image & GPU mJ/image & Pass 1 pp & Pass 2 pp & Selection\\',r'\midrule']
variants=[('file_static_c2','Engine files, capacity 2'),('host_static_reuse_c2','RAM bytes + reuse, capacity 2'),('resident_eager_c6','Six resident, eager'),('resident_graph_c6','Six resident, CUDA Graphs'),('host_aware_c2','RAM + switching-aware, capacity 2')]
for key,label in variants:
    rr=[r for r in switching['results'] if r['variant']==key and r['is_routed']]
    assert len(rr)==3 and {r['seed'] for r in rr}=={42,45,48}
    lines.append(f"{label} & {interval([r['median_ms_per_image'] for r in rr])} & {interval([1000*r['median_gpu_j_per_image'] for r in rr],1)} & {sum(r['all_repeats_pass'] for r in rr)}/3 & {sum(r['accuracy_loss_pp']<=2+1e-10 for r in rr)}/3 & {'Constant' if all(r['constant_classifier_selection'] for r in rr) else 'Dynamic'}"+r'\\')
lines += [r'\bottomrule',r'\end{tabularx}'];(ROOT/'switching_statistics.tex').write_text('\n'.join(lines),encoding='utf-8')
explanations=[
    ('file_static_c2','1. Load engine files; keep only two models on GPU.', 'When the selected model is missing from GPU memory, read its engine file, create its execution context and evict a model if necessary. Selection is dynamic, but repeated switching is expensive.'),
    ('host_static_reuse_c2','2. Cache engine bytes in CPU RAM and reuse allocations.', 'Keep verified engine bytes in system RAM to avoid repeated disk reads, reuse buffers and reduce forced cleanup. Only two models are resident on GPU, so misses still require GPU deserialization. The same per-seed choices are preserved.'),
    ('resident_eager_c6','3. Keep all six models on GPU; execute normally.', 'Load every classifier, its context and buffers before timing. The router can switch among them without warm-run loading or eviction. Eager means ordinary per-inference execution calls, without a captured graph.'),
    ('resident_graph_c6','4. Keep all six models on GPU and replay CUDA Graphs.', 'In addition to residency, capture GPU work once and replay it to reduce repeated launch overhead. Seed42 recorded9.07 percent latency and4.20 percent GPU-energy savings against graph-enabled eligible static. It meets1pp; all three seeds meet2pp, but not every repeat improves timing.'),
    ('host_aware_c2','5. Penalize model switches during selection.', 'Combine RAM-byte caching with a policy that charges the measured cost of a cache miss. All three seeds then choose R101 FP16 for every image: switching disappears, but selection is constant rather than input-dependent. This is not evidence that dynamic routing beats static selection.')]
paragraphs=[]
for variant,title,explanation in explanations:
    rr=[r for r in switching['results'] if r['variant']==variant and r['is_routed']]
    costs=f"Measured total latency: {interval([r['median_ms_per_image'] for r in rr])}ms/image; GPU energy: {interval([1000*r['median_gpu_j_per_image'] for r in rr],1)}mJ/image."
    paragraphs.append(r'\textbf{'+title+r'} '+explanation.replace('percent',r'\%')+' '+costs)
(ROOT/'runtime_explanations.tex').write_text('\n\n'.join(paragraphs)+'\n',encoding='utf-8')
compact_routes=[r for r in compact['results'] if r['method'] not in ('reference','eligible_static')]
sensitivity=dict(scope='Descriptive2pp reassessment of unchanged1pp-calibrated policies and controls; not a new run or calibration.',compact_cases=len(compact_routes),
    compact_accuracy_passes_1pp=sum(r['accuracy_pass'] for r in compact_routes),compact_accuracy_passes_2pp=sum(r['accuracy_loss_pp']<=2+1e-10 for r in compact_routes),
    compact_joint_savings_passes_1pp=sum(r['accuracy_pass'] and r['latency_saving_vs_eligible_pct']>0 and r['energy_saving_vs_eligible_pct']>0 for r in compact_routes),
    compact_joint_savings_passes_2pp=sum(r['accuracy_loss_pp']<=2+1e-10 and r['latency_saving_vs_eligible_pct']>0 and r['energy_saving_vs_eligible_pct']>0 for r in compact_routes))
(ROOT/'accuracy_sensitivity.tex').write_text(r'\textbf{Study A:1pp versus2pp.} '+f"At1pp, {sensitivity['compact_accuracy_passes_1pp']}/15 cases pass accuracy and none also saves both latency/GPU energy versus eligible static. At2pp, all{sensitivity['compact_accuracy_passes_2pp']}/15 pass accuracy and{sensitivity['compact_joint_savings_passes_2pp']} also show both savings. This is a looser acceptance check on unchanged policies and1pp-selected static controls, not a newly optimized2pp experiment.\n",encoding='utf-8')
graph42=next(r for r in switching['results'] if r['seed']==42 and r['variant']=='resident_graph_c6')
assert graph42['eligible_comparison']['valid_gpu_fit_payback_images']==28621
optimized_example=(r'\textbf{Optimized Study B is a different case.} '+
    f"Seed42 resident-Graph routing uses{graph42['median_gpu_j_per_image']:.7f}GPUJ/image and passes accuracy; repayment is{graph42['eligible_comparison']['valid_gpu_fit_payback_images']:,}images versus graph-enabled eligible static. "
    +r'Do not compare this directly with Study A\textquotesingle s2,399images versus FP32: the runtime, baseline and measurement session differ.\n')
(ROOT/'optimized_payback.tex').write_text(optimized_example.replace('\\n','\n'),encoding='utf-8')
fit_j={r['seed']:r['joules'] for r in training['rows']}
batch1_payback=[]
for r in compact['results']:
    if r['method']!='tinycnn':continue
    item=dict(seed=r['seed'],batch_size=1,accuracy_passed=r['accuracy_pass'])
    for baseline,method in [('eligible_static','eligible_static'),('reference_fp32','reference')]:
        control=next(x for x in compact['results'] if x['seed']==r['seed'] and x['method']==method)
        saving=control['gpu_j_per_image']-r['gpu_j_per_image']
        item[baseline]=dict(training_j=fit_j[r['seed']],saved_j_per_image=saving,
            valid_payback_images=math.ceil(fit_j[r['seed']]/saving) if r['accuracy_pass'] and saving>0 else None)
    batch1_payback.append(item)
assert len(batch1_payback)==3
assert [r['reference_fp32']['valid_payback_images'] for r in batch1_payback if r['seed']==42]==[2399]
cnn42=next(r for r in compact['results'] if r['method']=='tinycnn' and r['seed']==42)
ref42=next(r for r in compact['results'] if r['method']=='reference' and r['seed']==42)
eligible42=next(r for r in compact['results'] if r['method']=='eligible_static' and r['seed']==42)
batch1_example=(r'\textbf{Batch1 example (Study A, seed42).} '+f"Fitting uses{fit_j[42]:.3f}J; routed energy is{cnn42['gpu_j_per_image']:.7f}J/image versus{ref42['gpu_j_per_image']:.7f}J/image for R101 FP32: "
    +r'$\lceil '+f"{fit_j[42]:.3f}/({ref42['gpu_j_per_image']:.7f}-{cnn42['gpu_j_per_image']:.7f})"+r'\rceil=2,399$ images. '
    +f"Eligible static uses only{eligible42['gpu_j_per_image']:.7f}J/image, so routing cannot repay energy against it. Other seeds fail the original accuracy gate. CPU-router fit energy is unmeasured, so no CPU-router payback is claimed.\n")
(ROOT/'batch1_example.tex').write_text(batch1_example,encoding='utf-8')
fig,ax=plt.subplots(figsize=(10.8,2.7))
stats=[]
for i,b in enumerate(batches):
    rr=batch1_payback if b==1 else [r for r in old['rows'] if r['batch_size']==b]
    baseline_stats={}
    for baseline,color,offset in [('eligible_static','#187f80',-.12),('reference_fp32','#db783c',.12)]:
        vals=[r[baseline]['valid_payback_images'] for r in rr if r[baseline]['valid_payback_images'] is not None]
        for j,v in enumerate(vals):ax.scatter(i+offset+(j-(len(vals)-1)/2)*.006,v,facecolors='none' if b==1 else color,edgecolors=color,s=26 if b==1 else 16,alpha=.8)
        ax.text(i+offset,1.01, str(len(vals))+'/'+str(len(rr)),transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color=color)
        baseline_stats[baseline]=dict(valid=len(vals),minimum=min(vals) if vals else None,maximum=max(vals) if vals else None)
    vals=[r['eligible_static']['valid_payback_images'] for r in rr if r['eligible_static']['valid_payback_images'] is not None]
    stats.append(dict(batch=b,settings=len(rr),accuracy_passes=sum(r['accuracy_passed'] for r in rr),eligible_valid=len(vals),minimum=min(vals) if vals else None,maximum=max(vals) if vals else None,baselines=baseline_stats))
ax.axvline(.5,color='#999999',lw=1,ls=':')
ax.set_xticks(range(6),['1*','8','16','32','64','128']);ax.set_xlabel('Input batch size (* batch1: six resident models; historical batches: 213 choices/cache2)');ax.set_ylabel('Images to repay GPU fit');ax.set_yscale('log');ax.grid(alpha=.2)
ax.plot([],[],'o',color='#187f80',label='Eligible static');ax.plot([],[],'o',color='#db783c',label='R101 FP32');ax.legend(loc='lower right',fontsize=8)
fig.tight_layout();fig.savefig(IMAGES/'payback_comparison.pdf');fig.savefig(IMAGES/'payback_comparison.png',dpi=180);plt.close(fig)
lines=[r'\begin{tabular}{@{}rll ll@{}}',r'\toprule',r'& \multicolumn{2}{c}{Compared with eligible static} & \multicolumn{2}{c}{Compared with R101 FP32}\\',r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',r'Batch & Settings repaying & Images before repayment & Settings repaying & Images before repayment\\',r'\midrule']
def payback_cells(stats,total):
    if not stats['valid']:return 'No payback',r'\textemdash'
    span=f"{stats['minimum']:,}" if stats['minimum']==stats['maximum'] else f"{stats['minimum']:,}--{stats['maximum']:,}"
    return f"{stats['valid']} of {total}",span
for r in stats:
    a,ar=payback_cells(r['baselines']['eligible_static'],r['settings']);b,br=payback_cells(r['baselines']['reference_fp32'],r['settings'])
    lines.append(f"{'1*' if r['batch']==1 else r['batch']} & {a} & {ar} & {b} & {br}"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(ROOT/'payback_statistics.tex').write_text('\n'.join(lines),encoding='utf-8')
example=next(r for r in old['rows'] if r['seed']==48 and r['batch_size']==64 and r['objective']=='energy' and r['budget_pp']==1)
assert example['accuracy_passed'] and example['eligible_static']['valid_payback_images'] is not None
e=example['eligible_static'];assert math.ceil(example['training_j']/e['saved_j_per_image'])==e['valid_payback_images']
text=(r'\textbf{Worked measured example (historical batch64, energy objective, seed48,1pp).} '
      +f"Accuracy is{example['accuracy']:.2f}\\%, so the allowance passes. Eligible static uses{e['j_per_image']:.7f}J/image and routing uses{example['route_j_per_image']:.7f}J/image. "
      +f"Saving is{e['saved_j_per_image']:.7f}J/image; with{example['training_j']:.3f}J of fit energy, repayment is{e['valid_payback_images']:,}images.\n")
(ROOT/'payback_example.tex').write_text(text,encoding='utf-8')
for name in ('batch8_grouping','supervision_deployment','dispatch_conditional'):
    p=PIPE/'reports/compact_b1_20261028/final_summary'/(name+'.png')
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,IMAGES/p.name)
snapshot=dict(sources_sha256=sources,fixed1pp_latency_rows=rows,comparison_baseline_rows=baseline_rows,payback_by_batch_all_settings=stats,batch1_payback=batch1_payback,worked_example=example,
              compact_rows=compact['results'],accuracy_sensitivity=sensitivity,switching=switching,optimized_graph_seed42=graph42,fit_energy=training,supervision=deployment,supervision_payback=payback,
              note='Historical 213-choice/capacity2 vs new six-resident classifiers are not a matched batch-size experiment. No inference reruns.')
(ROOT/'report_data.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
print(json.dumps({'payback_by_batch':stats,'example':text},indent=2))
