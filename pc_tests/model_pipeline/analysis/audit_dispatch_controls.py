"""Audit fixed-count selection, same-engine pairs and aggregatecoldenergy."""
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
    path=PIPELINE/'reports/dispatch_controls_20261029/study.json';s=json.loads(path.read_text());a=PIPELINE/'artifacts/dispatch_controls_20261029'
    assert s['status']=='complete' and len(s['results'])==21 and len({r['id'] for r in s['results']})==21 and not s['test_data_opened']
    assert s['source_sha256']==digest(PIPELINE/'experiments/profile_dispatch_controls.py')
    assert s['predecessor_sha256']==digest(PIPELINE/'reports/batch8_grouping_20261029/audited_summary.json')
    parity=next(r for r in s['results'] if r['kind']=='selector_parity')
    assert parity['real_images']==512 and parity['near_tie_fixtures']==105
    assert parity['engine_sha256']==digest(a/'selector_fp32.engine') and parity['graph_sha256']==digest(a/'selector_fp32.onnx')
    pairs=[]
    for r in [r for r in s['results'] if r['kind']=='same_classifier_pair']:
        assert r['images_per_mode']==512 and len(r['blocks'])==32
        for block in r['blocks']:
            assert block['static']['predictions']==block['forced_routed']['predictions'] and len(block['static']['predictions'])==16
        static=sum(b['static']['seconds'] for b in r['blocks'])*1000/512
        route=sum(b['forced_routed']['seconds'] for b in r['blocks'])*1000/512
        pairs.append(dict(repeat=r['repeat'],static_ms=static,forced_routed_ms=route,increment_ms=route-static,telemetry=r['telemetry']))
    assert sorted(r['repeat'] for r in pairs)==list(range(5))
    selection={}
    for mode in ('cpu64','torch64','trt32'):
        group=[r for r in s['results'] if r['kind']=='selector_window' and r['mode']==mode]
        assert len(group)==3 and {r['repeat'] for r in group}=={0,1,2}
        for r in group:assert r['images']==sum(r['selected_images'])==len(r['predictions'])==512 and r['gpu_energy']['gpu_energy_j']>0
        selection[mode]=dict(median_ms=float(np.median([r['ms_per_image'] for r in group])),range_ms=[min(r['ms_per_image'] for r in group),max(r['ms_per_image'] for r in group)],median_gpu_j_per_image=float(np.median([r['gpu_energy']['gpu_energy_j_per_image'] for r in group])))
    if parity['fp32_real_choice_mismatches']==0:
        prediction_arrays=[r['predictions'] for r in s['results'] if r['kind']=='selector_window']
        assert all(p==prediction_arrays[0] for p in prediction_arrays)
    cold={}
    for mode in ('file','host'):
        group=[r for r in s['results'] if r['kind']=='aggregate_cold' and r['mode']==mode]
        assert len(group)==3 and {r['repeat'] for r in group}=={0,1,2}
        for r in group:assert r['seconds']>=2 and r['loads']>0 and r['minimum_free_bytes']>=1024**3 and r['gpu_energy']['gpu_energy_j']>0
        cold[mode]=dict(median_seconds_per_load=float(np.median([r['seconds']/r['loads'] for r in group])),median_gpu_j_per_load_cycle=float(np.median([r['gpu_j_per_load'] for r in group])))
    out=dict(audit_passed=True,report_sha256=digest(path),rows=21,parity=parity,paired_same_classifier=pairs,selection=selection,aggregate_cold=cold,
        cpu_system_sensor_availability=s['cpu_system_sensor_availability'],cpu_energy_measured=False,whole_pc_energy_measured=False,
        caveats='Validation-onlyengineering, nottestaccuracy. GPUonlyenergy excludesCPU. Fixedsameclassifier/context interleaving showedpositiveCNN/readback overhead inall5pairs, but staticlatencyvaried4.10–7.12ms. Selector3repeat rangesoverlap; do notclaimuniversalFP32speedup.15/105near-tiedFP32fixturesdifferedfromFP64 despite0/512realchoicechanges. Aggregatecoldenergy isload+context+cleanupcycles>=2s, notpurestartuporperimage; oldunavailableindividualreadingsnotreplaced.')
    atomic_json(path.parent/'audited_summary.json',out)
    lines=['# Dispatch controls and cold GPU energy','',out['caveats'],'','| Selector | Median ms/image | GPU J/image |','|---|---:|---:|']
    for m,r in selection.items():lines.append(f"| {m} | {r['median_ms']:.3f} | {r['median_gpu_j_per_image']:.4f} |")
    lines+=['','| Cold mode | Seconds/load+cleanup | GPU J/load+cleanup |','|---|---:|---:|']
    for m,r in cold.items():lines.append(f"| {m} | {r['median_seconds_per_load']:.3f} | {r['median_gpu_j_per_load_cycle']:.3f} |")
    lines+=['','No verifiedCPU/systemenergy sensor wasavailablethrough thetwo checkedWMIproviders. This doesnotassert that no othermetercouldexist; noCPU/fullPCenergy result orCPUtrainingpaybackreported.']
    (path.parent/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(audit_passed=True,selection=selection,cold=cold,paired_overheads_ms=[r['increment_ms'] for r in pairs]),indent=2))

if __name__=='__main__':main()
