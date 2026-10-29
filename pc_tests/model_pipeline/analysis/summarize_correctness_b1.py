"""CPU-only audit, batch plots and seed-specific fit-energy payback."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stats
import sys

PIPELINE=Path(__file__).resolve().parents[1]
OUTPUT=PIPELINE/'reports/correctness_b1_20261028'
BATCHES=(1,8,16,32,64,128)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def payback(training_j, static_j_per_image, routed_j_per_image, accuracy_passed):
    if not all(math.isfinite(v) and v>0 for v in (training_j,static_j_per_image,routed_j_per_image)):
        raise ValueError('Finite positive GPU energies required')
    saving=static_j_per_image-routed_j_per_image
    crossing=math.ceil(training_j/saving) if saving>0 else None
    valid=crossing if accuracy_passed else None
    return dict(saved_j_per_image=saving,mathematical_crossing_images=crossing,
                valid_payback_images=valid,
                no_valid_payback_reason=('accuracy allowance failed' if not accuracy_passed else
                                        'no positive energy saving' if saving<=0 else None))


def med(rows,key):
    values=[r[key] for r in rows]
    if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in values):
        raise ValueError('Invalid measurement '+key)
    return stats.median(values)


def summarize(seed,batch,objective,budget,route,reference,eligible,chosen,training_j,source):
    assert all(len(x)==5 for x in (route,reference,eligible))
    for measurements in (route,reference,eligible):
        assert sorted(r['repeat'] for r in measurements)==list(range(5))
        for row in measurements:
            assert row['test_images']==10000
            assert sum(row['selected_images'])==10000
    accuracy=stats.mean(r['test_accuracy'] for r in route)
    loss=stats.mean(r['test_accuracy'] for r in reference)-accuracy
    passed=loss<=budget+1e-10
    routed=med(route,'gpu_energy_j_per_image')
    seconds=med(route,'rgb_input_to_prediction_seconds')
    result=dict(seed=seed,batch_size=batch,objective=objective,budget_pp=budget,accuracy=accuracy,
        accuracy_loss_pp=loss,accuracy_passed=passed,route_ms_per_image=seconds/10,
        route_ms_per_batch=med(route,'mean_batch_ms'),route_j_per_image=routed,
        route_latency_repeat_range_ms_per_image=[min(r['rgb_input_to_prediction_seconds'] for r in route)/10,max(r['rgb_input_to_prediction_seconds'] for r in route)/10],
        route_energy_repeat_range_j_per_image=[min(r['gpu_energy_j_per_image'] for r in route),max(r['gpu_energy_j_per_image'] for r in route)],
        training_j=training_j,eligible_static=chosen,measurement_source=source)
    for name,measurements in (('reference_fp32',reference),('eligible_static',eligible)):
        energy=med(measurements,'gpu_energy_j_per_image')
        base_seconds=med(measurements,'rgb_input_to_prediction_seconds')
        result[name]=dict(accuracy=stats.mean(r['test_accuracy'] for r in measurements),j_per_image=energy,
            ms_per_image=base_seconds/10,latency_saving_percent=100*(base_seconds-seconds)/base_seconds,
            energy_saving_percent=100*(energy-routed)/energy,
            **payback(training_j,energy,routed,passed))
    return result


def collect(existing_only=False):
    energy_path=PIPELINE/'reports/correctness_training_energy_20261028/audited_summary.json'
    energy=json.loads(energy_path.read_text())
    assert energy['audit_passed'] and energy['original_artifacts_unchanged']
    raw_training=PIPELINE/'artifacts/correctness_training_energy_20261028/training_energy.json'
    assert digest(raw_training)==energy['measurement_sha256']
    training={r['seed']:r['joules'] for r in energy['rows']}
    old_path=PIPELINE/'reports/pc_routes_correctness.json'
    old=json.loads(old_path.read_text())
    assert len(old['results'])==120 and all(r['status']=='complete' for r in old['results'])
    rows=[]
    for item in old['results']:
        rows.append(summarize(item['seed'],item['batch_size'],item['objective'],item['budget_pp'],item['measurements'],
            item['comparator_measurements'][item['baseline']],item['comparator_measurements'][item['best_eligible_static']],
            item['best_eligible_static'],training[item['seed']],'historical larger-batch report'))
    inputs={str(old_path):digest(old_path),str(energy_path):digest(energy_path),str(raw_training):digest(raw_training)}
    if not existing_only:
        path=OUTPUT/'study.json'
        study=json.loads(path.read_text())
        assert study['status']=='complete' and study['protected_unchanged']
        assert len(study['results'])==366 and len({r['id'] for r in study['results']})==366
        for group in ('protected_hashes','source_hashes','test_hashes'):
            for filename,expected in study[group].items():
                assert digest(filename)==expected,filename
        for seed in (42,45,48):
            prepared=study['prepared'][str(seed)]
            assert len(prepared['calibration_positions'])==2000
            assert len(prepared['fit_positions'])==3000
            assert not set(prepared['calibration_positions']) & set(prepared['fit_positions'])
            assert prepared['torch_max_error']<=1e-5 and prepared['validation_max_error']<=1e-5
            assert digest(prepared['engine'])==prepared['engine_sha256']
            for objective in ('latency','energy'):
                for budget in (.5,1.,1.5,2.):
                    selected=[r for r in study['results'] if r['kind']=='full' and r['seed']==seed and r['objective']==objective and r['budget_pp']==budget]
                    assert len(selected)==15
                    measurements={role:[r['measurement'] for r in selected if r['role']==role] for role in ('reference','eligible_static','learned')}
                    for value in measurements['learned']:
                        assert len(value['selected_images'])==213
                    rows.append(summarize(seed,1,objective,budget,measurements['learned'],measurements['reference'],measurements['eligible_static'],
                        prepared['comparators'][objective][str(budget)],training[seed],'new batch-one session'))
        inputs[str(path)]=digest(path)
    assert len(rows)==(120 if existing_only else 144)
    assert len({(r['seed'],r['batch_size'],r['objective'],r['budget_pp']) for r in rows})==len(rows)
    return sorted(rows,key=lambda r:(r['batch_size'],r['objective'],r['budget_pp'],r['seed'])),inputs


def plots(rows,directory):
    # Keep plotting packages isolated from the CUDA environment.
    sys.path.insert(0,str(PIPELINE/'artifacts/plot_dependencies'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib import ticker
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    available=sorted({r['batch_size'] for r in rows})
    colors={'latency':'#2166ac','energy':'#b35806'}
    figures=[]
    def finish(fig,name):
        fig.savefig(directory/(name+'.png'),bbox_inches='tight')
        fig.savefig(directory/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)
        figures.append(name+'.png')
    # Fixed allowance; all seeds, not the best test seed.
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for objective in ('latency','energy'):
        for ax,key,multiplier in zip(axes,('accuracy','route_ms_per_image','route_j_per_image'),(1,1,1000)):
            values=[]
            for batch in available:
                group=[r for r in rows if r['batch_size']==batch and r['objective']==objective and r['budget_pp']==1.]
                y=[r[key]*multiplier for r in group]
                center=stats.median(y);values.append(center)
                ax.errorbar(batch,center,yerr=[[center-min(y)],[max(y)-center]],color=colors[objective],capsize=3)
                for r in group:
                    ax.scatter(batch,r[key]*multiplier,marker='o' if r['accuracy_passed'] else 'x',s=32,color=colors[objective])
            ax.plot(available,values,color=colors[objective],label=objective+' policy')
    for ax,title,ylabel in zip(axes,('Accuracy','End-to-end latency','GPU energy'),('Test accuracy (%)','ms / image','mJ / image')):
        ax.set(title=title,xlabel='Batch size',ylabel=ylabel,xscale='log',xticks=available)
        ax.set_xticklabels(available);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    session_note='Batch 1 is a new session; larger batches are saved historical measurements' if 1 in available else 'Saved larger-batch measurements only; batch 1 is not included yet'
    fig.suptitle('1-point allowance | all seeds; line=median, bars=seed range; x=accuracy failure\n'+session_note,fontsize=11)
    finish(fig,'batch_size_accuracy_latency_energy')
    # No savings-baseline ambiguity: each heatmap explicitly uses eligible static.
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    budgets=(.5,1.,1.5,2.)
    for column,objective in enumerate(('latency','energy')):
        for line,metric in enumerate(('latency','energy')):
            ax=axes[line,column]
            grid=np.array([[stats.median(r['eligible_static'][metric+'_saving_percent'] for r in rows if r['batch_size']==b and r['objective']==objective and r['budget_pp']==a) for b in available] for a in budgets])
            limit=max(1,float(np.max(np.abs(grid))))
            im=ax.imshow(grid,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
            for i,a in enumerate(budgets):
                for j,b in enumerate(available):
                    passed=sum(r['accuracy_passed'] for r in rows if r['batch_size']==b and r['objective']==objective and r['budget_pp']==a)
                    ax.text(j,i,f'{grid[i,j]:+.1f}%\nP {passed}/3',ha='center',va='center',fontsize=8,
                            color='white' if abs(grid[i,j])>limit*.6 else 'black')
            ax.set(title=f'{objective.capitalize()} policy: {metric} saving',xlabel='Batch size',ylabel='Accuracy allowance (pp)',xticks=range(len(available)),yticks=range(4))
            ax.set_xticklabels(available);ax.set_yticklabels(budgets)
            fig.colorbar(im,ax=ax,shrink=.7,label='Median saving (%)')
    fig.suptitle('Savings against each setting\'s validation-selected eligible static\nP = number of seeds meeting the test accuracy allowance; negatives and failures retained',fontsize=11)
    finish(fig,'eligible_static_savings_heatmaps')
    # Scatter plots retain each setting; no test-selected winners.
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for batch,color in zip(available,plt.cm.viridis(np.linspace(.08,.9,len(available)))):
        group=[r for r in rows if r['batch_size']==batch]
        for ax,key,multiplier in zip(axes,('route_ms_per_image','route_j_per_image'),(1,1000)):
            for passed,marker in ((True,'o'),(False,'x')):
                chosen=[r for r in group if r['accuracy_passed']==passed]
                ax.scatter([r[key]*multiplier for r in chosen],[r['accuracy'] for r in chosen],s=26,color=color,marker=marker,alpha=.7,label=f'b{batch}' if passed else None)
    for ax,label in zip(axes,('End-to-end ms / image','GPU mJ / image')):
        ax.set(xlabel=label,ylabel='Test accuracy (%)',xscale='log');ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10,numticks=4))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    controls=[r for r in rows if r['seed']==45 and r['objective']=='latency' and r['budget_pp']==1.]
    for ax,key,multiplier in zip(axes,('ms_per_image','j_per_image'),(1,1000)):
        ax.scatter([r['eligible_static'][key]*multiplier for r in controls],
            [r['eligible_static']['accuracy'] for r in controls],marker='D',s=50,
            facecolors='none',edgecolors='black',label='Static controls: seed45/latency/1pp')
    axes[0].legend(ncols=2,fontsize=8)
    fig.suptitle('Every measured routing setting | o=accuracy pass, x=accuracy failure\nCost axis is logarithmic; separate sessions are not controlled hardware speedup comparisons',fontsize=11)
    finish(fig,'accuracy_cost_tradeoffs')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,comparator,title in zip(axes,('eligible_static','reference_fp32'),('Eligible static comparator','R101 FP32 reference')):
        for seed,marker in zip((42,45,48),('o','s','^')):
            for objective in ('latency','energy'):
                group=[r for r in rows if r['seed']==seed and r['objective']==objective and r['budget_pp']==1.]
                valid=[r for r in group if r[comparator]['valid_payback_images'] is not None]
                ax.plot([r['batch_size'] for r in valid],[r[comparator]['valid_payback_images'] for r in valid],linestyle='none',marker=marker,color=colors[objective],label=f'{objective}, seed {seed}')
        ax.set(title=title,xlabel='Batch size',ylabel='Images to recover fit energy',yscale='log',xscale='log',xticks=available)
        ax.set_xticklabels(available);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('1-point allowance: valid fit-energy payback only\nMissing points mean accuracy failure or no positive energy saving; all cases remain in the data',fontsize=11)
    finish(fig,'payback_by_batch')
    # Fixed seed/objective/budget chosen in advance, not the best test result.
    chosen=[r for r in rows if r['seed']==45 and r['objective']=='latency' and r['budget_pp']==1.]
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,r in zip(axes.flat,chosen):
        base=r['eligible_static']['j_per_image'];route=r['route_j_per_image'];training=r['training_j']
        cross=r['eligible_static']['mathematical_crossing_images']
        maximum=min(max(10000,(cross or 50000)*1.4),10000000)
        x=np.linspace(0,maximum,200)
        ax.plot(x,base*x,label='Static')
        ax.plot(x,training+route*x,label='Training + routed inference')
        if cross and cross<=maximum:
            ax.axvline(cross,color='gray',linestyle=':',label=f'Crossing {cross:,} images')
        ax.set(title=f"Batch {r['batch_size']} | accuracy {'PASS' if r['accuracy_passed'] else 'FAIL'}",xlabel='Images processed',ylabel='Cumulative GPU energy (J)')
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    fig.suptitle('Seed 45, latency policy, 1-point allowance | eligible-static comparator\nRouting starts at its measured 201.185 J fit cost; a crossing is not valid if accuracy fails',fontsize=11)
    finish(fig,'cumulative_gpu_energy')
    return figures,matplotlib.__version__


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing-only',action='store_true',help='CPU-only preview of saved larger batches, separate output subdirectory')
    args=parser.parse_args()
    rows,inputs=collect(args.existing_only)
    directory=OUTPUT/'existing_only' if args.existing_only else OUTPUT
    directory.mkdir(parents=True,exist_ok=True)
    figures,version=plots(rows,directory)
    result=dict(audit_passed=True,complete=not args.existing_only,settings=len(rows),rows=rows,input_hashes=inputs,
        analysis_sha256=digest(__file__),figures=figures,matplotlib_version=version,
        caveat='GPU-only fit-energy repeat, not full preparation/whole-PC; historical inference and new training/b1 are separate sessions with nonstationary GPU conditions. No test-label tuning or retrospective best-seed selection. Seed variation mixes training/split/batching. No precise statistical confidence claim.',
        per_image_cached_prediction_parity='Not established for batch1: actual runtime accuracy is used; no cached predictions substitute for timed inference.')
    (directory/'combined_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fields=('seed','batch_size','objective','budget_pp','accuracy','accuracy_loss_pp','accuracy_passed','route_ms_per_image','route_ms_per_batch','route_j_per_image','training_j','eligible_static')
    extra=tuple(c+'_'+k for c in ('eligible_static','reference_fp32') for k in ('latency_saving_percent','energy_saving_percent','saved_j_per_image','valid_payback_images','mathematical_crossing_images','no_valid_payback_reason'))
    with (directory/'combined_summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields+extra);writer.writeheader()
        for r in rows:
            writer.writerow({**{k:r[k] for k in fields},**{c+'_'+k:r[c][k] for c in ('eligible_static','reference_fp32') for k in ('latency_saving_percent','energy_saving_percent','saved_j_per_image','valid_payback_images','mathematical_crossing_images','no_valid_payback_reason')}})
    lines=['# Batch-size comparison and fit-energy payback','',result['caveat'],'',
        'Payback images = ceil(seed-specific training J / (static J/image - routed J/image)). Calculate independently for every batch, seed, objective, allowance and comparator. Null valid payback is retained for accuracy failures or nonpositive savings. No three-seed cost is charged to one deployed router.','',
        '| Batch | Settings | Accuracy passes | Positive energy saving vs eligible static | Valid payback range (images) |',
        '|---|---:|---:|---:|---:|']
    for batch in sorted({r['batch_size'] for r in rows}):
        group=[r for r in rows if r['batch_size']==batch]
        pay=[r['eligible_static']['valid_payback_images'] for r in group if r['eligible_static']['valid_payback_images'] is not None]
        span=f'{min(pay):,} to {max(pay):,}' if pay else 'None'
        lines.append(f"| {batch} | {len(group)} | {sum(r['accuracy_passed'] for r in group)} | {sum(r['eligible_static']['saved_j_per_image']>0 for r in group)} | {span} |")
    lines+=['','Plots show all settings or a fixed1pp allowance, not the most favorable test-selected configuration. Detailed CSV/JSON retain every seed, negative and failed accuracy case.','',
        'Full-batch payback in batches is ceil(training J / (batch size x saved J/image)); image counts in the JSON are continuous-throughput estimates rounded up, not padded batch counts.']
    (directory/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit_passed=True,settings=len(rows),complete=not args.existing_only,figures=figures,directory=str(directory)),indent=2))


if __name__=='__main__':
    main()
