"""Scientific figures for audited batch-one screen; no implicit protocol merging."""
import sys,json
from pathlib import Path
PIPELINE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PIPELINE/'artifacts/plot_dependencies'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    folder=PIPELINE/'reports/compact_resident_screen_20261028'
    summary=json.loads((folder/'audited_summary.json').read_text());assert summary['audit_passed']
    rows=summary['results'];methods=['tree_d3','logistic','thumbnail_mlp16','boosted_d2','tinycnn']
    labels=['Depth-3 tree','Logistic','Small MLP','Boosted trees','TinyCNN']
    colors=['#2274a5','#ee964b','#32936f']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    specifications=[('accuracy_loss_pp','Accuracy loss vs R101 FP32 (pp)',1.),
                    ('latency_saving_vs_eligible_pct','Latency saving vs eligible static (%)',0.),
                    ('energy_saving_vs_eligible_pct','GPU-energy saving vs eligible static (%)',0.),
                    ('decision_ms','Router/selection wall time (ms/image)',None)]
    for ax,(key,title,boundary) in zip(axes.flat,specifications):
        for j,seed in enumerate((42,45,48)):
            group=[next(r for r in rows if r['seed']==seed and r['method']==m) for m in methods]
            bars=ax.bar(np.arange(5)+(j-1)*.25,[r[key] for r in group],width=.23,label=f'Seed {seed}',color=colors[j])
            for bar,r in zip(bars,group):
                if not r['accuracy_pass']:bar.set_hatch('///');bar.set_edgecolor('#333333')
        if boundary is not None:ax.axhline(boundary,color='#444444',ls='--',lw=1)
        ax.set_xticks(range(5),labels,rotation=20,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0,0].legend(ncols=3,fontsize=9)
    fig.suptitle('Batch 1: compact routers and TinyCNN, same six resident original classifiers',fontsize=13)
    fig.text(.5,.015,'All three seeds; one 10,000-image sweep per case. Hatching = accuracy allowance failed. GPU energy excludes CPU energy.\nPositive savings mean lower cost; warm inference only, startup separate. Preliminary screen, not the old 213-choice/capacity-2 protocol.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95))
    for ext in ('png','svg'):fig.savefig(folder/f'compact_comparison.{ext}',bbox_inches='tight')
    plt.close(fig)
    train=json.loads((PIPELINE/'reports/correctness_training_energy_20261028/audited_summary.json').read_text())
    joules={r['seed']:r['joules'] for r in train['rows']}
    x=np.linspace(0,12000,300);fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,baseline,title in zip(axes,['reference','eligible'],['Fresh R101 FP32 baseline','Fresh eligible-static baseline']):
        for color,seed in zip(colors,(42,45,48)):
            routed=next(r for r in rows if r['seed']==seed and r['method']=='tinycnn')
            control=next(r for r in rows if r['seed']==seed and r['method']==('reference' if baseline=='reference' else 'eligible_static'))
            saving=control['gpu_j_per_image']-routed['gpu_j_per_image']
            y=x*saving-joules[seed]
            ax.plot(x,y,color=color,ls='-' if routed['accuracy_pass'] else '--',label=f"Seed {seed}"+(' (accuracy FAIL)' if not routed['accuracy_pass'] else ' (accuracy pass)'))
            valid=routed['fit_gpu_payback_images_vs_'+baseline]
            if valid is not None:ax.plot(valid,0,'o',color=color);ax.annotate(f'{valid:,} images',(valid,0),xytext=(6,10),textcoords='offset points',fontsize=9)
        ax.axhline(0,color='#444444',lw=1);ax.set_title(title);ax.set_xlabel('Deployment inferences');ax.set_ylabel('Net GPU energy saved after fit energy (J)');ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('TinyCNN GPU-fit energy payback: all seeds shown, accuracy gate retained')
    fig.text(.5,.015,'Dashed curves are NOT valid payback claims because held-out accuracy failed. Fit-only GPU energy; excludes CPU, supervision, calibration, export and startup.\nTraining and inference were measured in different sessions. These are linear projections from preliminary one-sweep results, not lifetime measurements.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.12,1,.94))
    for ext in ('png','svg'):fig.savefig(folder/f'tinycnn_fit_payback.{ext}',bbox_inches='tight')
    plt.close(fig)
    print('Created compact_comparison and tinycnn_fit_payback (PNG/SVG).')

if __name__=='__main__':main()
