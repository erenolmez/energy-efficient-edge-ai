"""Audit fixed validation switching pilots; no heldout savings inference."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
from pathlib import Path
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json

def main():
    path=PIPELINE/'reports/compact_switching_pilot_20261028/study.json';s=json.loads(path.read_text())
    assert s['status']=='complete' and len(s['results'])==8 and len({r['id'] for r in s['results']})==8
    assert not s['test_data_opened'] and s['source_sha256']==digest(PIPELINE/'experiments/pilot_compact_switching.py')
    for p,h in s['dependencies'].items():assert digest(p)==h
    assert s['predecessor_sha256']==digest(PIPELINE/'reports/compact_resident_screen_20261028/audited_summary.json')
    assert s['profile_sha256']==digest(PIPELINE/'reports/compact_runtime_profile_20261028/study.json')
    assert s['router_engine_sha256']==digest(PIPELINE/'artifacts/correctness_b1_20261028/engines/correctness_seed42_b1.engine')
    pilots=[r for r in s['results'] if r['kind']=='pilot'];assert len(pilots)==7
    for row in pilots:
        assert row['images']==sum(row['selected_images'])==256 and row['seconds']>0 and row['gpu_energy']['gpu_energy_j']>0
        if row['capacity']==6:assert row['runtime_loads']==row['evictions']==0
    # Mechanical optimizations should preserve the static policy's selected counts/classes.
    static=[r for r in pilots if not r['cache_aware']]
    assert all(r['selected_images']==static[0]['selected_images'] and r['calibration_probe_accuracy']==static[0]['calibration_probe_accuracy'] for r in static)
    aware=[r for r in pilots if r['cache_aware']]
    assert aware[0]['selected_images']==aware[1]['selected_images']
    summary=dict(audit_passed=True,report_sha256=digest(path),pilots=pilots,
        note='One256image calibration pilot pervariant, fixedorder; timing variation/warmup effects not isolated. Cache-aware policy changes choices, accuracy here is calibration-only. No heldout/fulltest/accuracy-preserving savings claim. Final memory reserve checked every64images; perstep peak not captured.')
    atomic_json(path.parent/'audited_summary.json',summary)
    lines=['# Validation-only switching pilot','',summary['note'],'',
           '| Variant | ms/image | GPU J/image | Loads | Evictions |','|---|---:|---:|---:|---:|']
    for r in pilots:lines.append(f"| {r['id']} | {r['ms_per_image']:.3f} | {r['gpu_energy']['gpu_energy_j_per_image']:.4f} | {r['runtime_loads']} | {r['evictions']} |")
    (path.parent/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Audited7validation-only switching pilots; fulltest confirmation pending.')

if __name__=='__main__':main()
