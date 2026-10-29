"""Audit real nativeIf lazybranch execution; engineeringbias isnotdeploymentpolicy."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
from pathlib import Path
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

def main():
    path=PIPELINE/'reports/gpu_conditional_20261029/study.json';s=json.loads(path.read_text());a=PIPELINE/'artifacts/gpu_conditional_20261029'
    assert s['status']=='complete' and not s['test_data_opened']
    assert s['source_sha256']==digest(PIPELINE/'experiments/probe_gpu_conditional.py')
    assert s['helper_sha256']==digest(PIPELINE/'runtime/conditional_onnx.py')
    assert s['fixedrunner_sha256']==digest(PIPELINE/'runtime/fixed_gpu_runner.py')
    assert s['predecessor_sha256']==digest(PIPELINE/'reports/dispatch_controls_20261029/audited_summary.json')
    for p,h in s['protected_hashes'].items():assert digest(p)==h
    assert s['prototype_supported'] and s['branch_execution_verified'] and len(s['results'])==15 and len({r['id'] for r in s['results']})==15
    for r in s['results']:
        if r['kind'] in ('branch_build','conditional_build'):assert digest(r['engine'])==r['engine_sha256']
        if r['kind']=='conditional_build':assert digest(a/'conditional_fp32.onnx')==r['graph_sha256']
    parity=s['parity'];assert parity['images']==512 and parity['class_changes']==parity['cpu_vs_gpu_choice_mismatches']==0 and parity['branches_seen']==[0,1]
    assert parity['max_abs_logit_error']<=1e-3
    structure=s['lazy_structure'];assert all(b['compute_layers']>0 and b['all_compute_lazy'] for b in structure['branches'].values())
    trace=next(r for r in s['results'] if r['kind']=='branch_trace');assert trace['verified']
    assert {r['chosen'] for r in trace['traces']}=={0,1}
    for r in trace['traces']:
        active='A__' if r['chosen']==0 else 'B__';inactive='B__' if r['chosen']==0 else 'A__'
        assert r['compute_ms_by_branch'][active]>0 and r['compute_ms_by_branch'][inactive]<=1e-7
    windows={};allpred=[]
    for mode in ('cpu_dispatch','gpu_conditional','both_eager'):
        rows=[r for r in s['results'] if r['kind']=='window' and r['mode']==mode]
        assert len(rows)==3 and {r['repeat'] for r in rows}=={0,1,2}
        for r in rows:assert r['images']==len(r['predictions'])==512 and r['seconds']>0 and r['gpu_energy']['gpu_energy_j']>0
        allpred.extend(r['predictions'] for r in rows)
        windows[mode]=dict(median_ms=float(np.median([r['ms_per_image'] for r in rows])),range_ms=[min(r['ms_per_image'] for r in rows),max(r['ms_per_image'] for r in rows)],
            median_gpu_j_per_image=float(np.median([r['gpu_energy']['gpu_energy_j_per_image'] for r in rows])))
    assert all(p==allpred[0] for p in allpred)
    result=dict(audit_passed=True,report_sha256=digest(path),rows=15,branch_execution_verified=True,parity=parity,measurements=windows,
        gpu_vs_cpu_latency_saving_percent=100*(1-windows['gpu_conditional']['median_ms']/windows['cpu_dispatch']['median_ms']),
        gpu_vs_both_eager_latency_saving_percent=100*(1-windows['gpu_conditional']['median_ms']/windows['both_eager']['median_ms']),
        scope='TwoFP32classifiers/sameoldseed42CNN, actualnativeIfGPUconditionandtypedbranchoutput, onlyselectedclassifiercomputes.512validationimages,3repeats. No intermediatebranchIDreadback intimedGPUmode; finalCPUclassmatched.',
        limitations='Engineeringmedianbias fromcalibrationdeliberatelyexercisesbothbranches; NOT validated1ppdeploymentpolicy orheldouttestaccuracy. Both-eager isnegativecontrol, notstandardrouterbaseline. GenuineGPUcontrol neednotbefasterthanCPUdispatch. Runtimeprofilerremovedbynewcontextbeforetiming. GPU-onlyenergy excludesCPU. No general213modelorallarchitectureclaim.')
    atomic_json(path.parent/'audited_summary.json',result)
    lines=['# Native GPU-conditional two-classifier prototype','',result['scope'],'',result['limitations'],'',
        '| Implementation | Median ms/image | GPU J/image |','|---|---:|---:|']
    for mode,r in windows.items():lines.append(f"| {mode} | {r['median_ms']:.3f} | {r['median_gpu_j_per_image']:.4f} |")
    lines+=['',f"Conditional vs CPU dispatch latency saving: {result['gpu_vs_cpu_latency_saving_percent']:+.2f}%. Conditional vs both-eager negative control: {result['gpu_vs_both_eager_latency_saving_percent']:+.2f}%.",
        '',f"Chosenbranchlogit maximum absolute difference {parity['max_abs_logit_error']:.8f}; zero class/choice mismatches on512validationimages. Bothbranchestested; unselectedConv/Gemm profilemass zero within1e-7ms threshold."]
    (path.parent/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
