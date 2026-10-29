"""Verify immutable completed work and authorize explicit checkpoint-aware resume."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import hashlib,json,subprocess
from datetime import datetime,timezone
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    artifacts=PIPELINE/'artifacts/compact_switching_confirmation_20261028'
    backup=artifacts/'recovery_checkpoint41';path=PIPELINE/'reports/compact_switching_confirmation_20261028/study.json'
    receipt_path=artifacts/'resume_receipt.json'
    if receipt_path.exists():raise RuntimeError('Recovery already prepared; inspect, never blindly repeat')
    state=json.loads(path.read_text());old=json.loads((backup/'study_failed.json').read_text())
    assert state==old and state['status']=='failed' and 'PermissionError' in state['error']
    assert state['source_sha256']==digest(backup/'confirm_compact_switching_original.py')
    thermal=json.loads((PIPELINE/'artifacts/pc_routers/thermal_guard.json').read_text(encoding='utf-8-sig'))
    assert thermal['status']=='monitoring' and thermal.get('boundary_start_count') is None
    assert json.loads((PIPELINE/'reports/experiment_pause.json').read_text())['status']=='resumed_by_user'
    command="Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and (($_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match 'energy-efficient-edge-ai.*(confirm_|pilot_|run_|train_|measure_|probe_|profile_|diagnose_|refresh_|prepare_compact)') -or ($_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'watch_pc_thermals.ps1')) } | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    if result.stdout.strip():raise RuntimeError('Worker/guard still active; refuse overlap: '+result.stdout)
    assert digest(artifacts/'frozen_plan.json')==digest(backup/'frozen_plan.json')==state['frozen_plan_sha256']
    for group in ('dependencies','protected_hashes','test_hashes'):
        for filename,expected in state[group].items():assert digest(filename)==expected
    plan=json.loads((artifacts/'frozen_plan.json').read_text())
    for p in plan['plans']:
        assert digest(PIPELINE/f'artifacts/correctness_b1_20261028/engines/correctness_seed{p["seed"]}_b1.engine')==p['router_engine_sha256']
    rows=state['full_sweeps'];assert len(rows)==41
    completed={(r['seed'],r['repeat'],r['variant']) for r in rows};assert len(completed)==41
    labels=np.load(PIPELINE/'artifacts/pc_precision/data_cache/test_labels.npy',allow_pickle=False)
    for row in rows:
        assert digest(row['runtime_file'])==row['runtime_sha256']
        with np.load(row['runtime_file'],allow_pickle=False) as data:
            assert data['predictions'].shape==data['choices'].shape==data['order'].shape==(10000,)
            assert np.array_equal(np.sort(data['order']),np.arange(10000))
            assert np.array_equal(np.bincount(data['choices'],minlength=6),row['selected_images'])
            assert np.isclose(100*(data['predictions']==labels).mean(),row['accuracy'],atol=1e-10)
        chunks=[r for r in state['results'] if r.get('kind')=='test_chunk' and (r['seed'],r['repeat'],r['variant'])==(row['seed'],row['repeat'],row['variant'])]
        assert len(chunks)==20 and sorted(r['completed_images'] for r in chunks)==list(range(500,10001,500))
        assert np.isclose(sum(r['seconds'] for r in chunks),row['seconds'])
        assert np.isclose(sum(r['gpu_energy']['gpu_energy_j'] for r in chunks),row['gpu_energy_j'])
    partial=[r for r in state['results'] if r.get('kind')=='test_chunk' and (r['seed'],r['repeat'],r['variant']) not in completed]
    assert len(partial)==5 and len({(r['seed'],r['repeat'],r['variant']) for r in partial})==1
    assert len(list(artifacts.glob('*.npz')))==41
    current_source=PIPELINE/'experiments/confirm_compact_switching.py'
    revision=dict(utc=datetime.now(timezone.utc).isoformat(),old_source_sha256=state['source_sha256'],new_source_sha256=digest(current_source),
        reason='Bounded atomic-save reader-lock retry and explicit saved-policy/whole-sweep resume; no scientific policy/settings change',
        original_source=str(backup/'confirm_compact_switching_original.py'),original_failed_report_sha256=digest(backup/'study_failed.json'),
        preserved_full_sweeps_sha256=canonical(rows),preserved_full_sweeps=41,interrupted_chunks=5,
        writer_sha256=digest(PIPELINE/'runtime/atomic_checkpoint.py'),test_scope='5CPUtests passed: validwrite,transientlock,persistentlock,othererror,nonfinite')
    receipt=dict(status='authorized',authority='Explicit human instruction to continue after reported file-access failure',revision=revision,
        frozen_plan_sha256=state['frozen_plan_sha256'],remaining_full_sweeps=40,
        incomplete_sweep='Restart this one full10000-image sweep only; old5partial chunks retained separately, not reused or mixed into new timing',
        shutdown_authority='Latest human instruction: graceful shutdown if thermal stop triggers or all authorized tasks complete; not merely this81sweep stage',
        cpu_temperature_available=False)
    atomic_json(receipt_path,receipt)
    state['interrupted_attempts']=[dict(error=state['error'],results=partial,source_sha256=state['source_sha256'],
        note='Partial prediction arrays were not saved; retained historical chunks excluded from resumed full-sweep totals')]
    state['results']=[r for r in state['results'] if r not in partial]
    state['code_revisions']=[revision];state['status']='resume_authorized';state['prior_error']=state.pop('error')
    state['recovery_receipt_sha256']=digest(receipt_path)
    atomic_json(path,state)
    print(json.dumps(dict(status=state['status'],preserved_full_sweeps=41,preserved_chunks=820,archived_partial_chunks=5,remaining=40),indent=2))

if __name__=='__main__':main()
