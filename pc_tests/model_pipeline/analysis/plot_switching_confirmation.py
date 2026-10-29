"""Mode-matched confirmation plots, show every seed and accuracy failure."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/plot_dependencies'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    folder=ROOT/'reports/compact_switching_confirmation_20261028'
    s=json.loads((folder/'audited_summary.json').read_text());assert s['audit_passed']
    variants=['file_static_c2','host_static_reuse_c2','host_aware_c2','resident_eager_c6','resident_graph_c6']
    labels=['File / cache2','Host + reuse / cache2','Host + cache-aware / cache2','Resident6 / eager','Resident6 / graph']
    colors=['#2274a5','#ed964b','#32936f']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    for ax,metric,title,boundary in zip(axes.flat,['accuracy_loss_pp','latency','energy','loads'],
        ['Accuracy loss vs same-mode R101 FP32 (pp)','Latency saving vs same-mode eligible static (%)','GPU-energy saving vs same-mode eligible static (%)','Classifier loads within each 10,000-image sweep'],[1,0,0,None]):
        for j,seed in enumerate((42,45,48)):
            rows=[next(r for r in s['results'] if r['seed']==seed and r['variant']==v) for v in variants]
            values=[r['accuracy_loss_pp'] if metric=='accuracy_loss_pp' else r['median_loads'] if metric=='loads' else r['eligible_comparison']['latency_saving_percent' if metric=='latency' else 'gpu_energy_saving_percent'] for r in rows]
            bars=ax.bar(np.arange(5)+(j-1)*.25,values,width=.23,color=colors[j],label=f'Seed {seed}')
            for bar,row in zip(bars,rows):
                if not row['accuracy_pass']:bar.set_hatch('///');bar.set_edgecolor('#333333')
                if row['constant_classifier_selection']:bar.set_edgecolor('black');bar.set_linewidth(1.5)
        if boundary is not None:ax.axhline(boundary,ls='--',lw=1,color='#555555')
        ax.set_xticks(range(5),labels,rotation=22,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0,0].legend(ncols=3)
    fig.suptitle('Batch-one switching confirmation: all seeds, three repeats, original six-model pool',fontsize=13)
    fig.text(.5,.02,'Hatching = accuracy allowance failed. Black outlines on cache-aware bars = constant R101 FP16 selection, not dynamic specialization.\nGraph cases use graph static controls; others use eager controls. Medians, GPU-only energy; substantial session/timing variation limits causal claims.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.95))
    for ext in ('png','svg'):fig.savefig(folder/f'switching_comparison.{ext}',dpi=145,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.7))
    for ax,seed in zip(axes,(42,45,48)):
        rows=[r for r in s['results'] if r['seed']==seed and r['is_routed']]
        for i,row in enumerate(rows):
            y=row['median_ms_per_image'];lo,hi=row['latency_range_ms']
            ax.errorbar(row['accuracy'],y,yerr=[[y-lo],[hi-y]],fmt='o' if row['accuracy_pass'] else 'x',capsize=3,label=labels[variants.index(row['variant'])])
        ax.set_title(f'Seed {seed}');ax.set_xlabel('Test accuracy (%)');ax.set_ylabel('Latency (ms/image)');ax.grid(alpha=.18)
    axes[0].legend(fontsize=7)
    fig.suptitle('Accuracy–latency trade-off; error bars are observed three-repeat ranges, not confidence intervals')
    fig.tight_layout(rect=(0,0,1,.91))
    for ext in ('png','svg'):fig.savefig(folder/f'accuracy_latency_ranges.{ext}',dpi=145,bbox_inches='tight')
    plt.close(fig)
    print('Created two confirmation figures, PNG/SVG.')

if __name__=='__main__':main()
