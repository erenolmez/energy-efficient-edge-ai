"""Explicit user-approved continuation: new plateau stage only, never graph rerun."""
import sys
from pathlib import Path
if not __package__: sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
import os
import subprocess
import time
from pc_tests.model_pipeline.paths import PIPELINE,REPOSITORY_ROOT
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json
from pc_tests.model_pipeline.experiments.compare_dispatch_residency_v2 import digest,utc
from pc_tests.model_pipeline.experiments.run_remaining_work import capture

OUT=PIPELINE/'artifacts/remaining_work_continuation_v2_20261104'
REPORT=PIPELINE/'reports/remaining_work_continuation_v2_20261104'


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('x') as f: json.dump(dict(pid=os.getpid(),utc=utc()),f)
    state=dict(status='preparing',pid=os.getpid(),started_utc=utc(),test_data_accessed=False,
        authority='Explicit user reply: Yes do that, approving versioned CPU audit then plateau only if it passes.',
        graph_rerun=False,active_identity=None)
    atomic_json(OUT/'study.json',state)
    try:
        atomic_json(OUT/'worker_identity.json',capture(os.getpid(),'continue_remaining_work_v2.py'))
        old=PIPELINE/'artifacts/remaining_work_20261104'
        assert load(old/'study.json')['status']=='stopped_error'
        original_plan=load(old/'plan.json')
        for p,h in original_plan['protected_sources'].items(): assert digest(p)==h,p
        graph_receipt_path=PIPELINE/'reports/remaining_graph_performance_20261104_audit_v2/audited_summary.json'
        graph=load(graph_receipt_path)
        assert graph['audit_completed'] and graph['windows']==72 and graph['exact_schedule_predictions_hashes_identities_verified']
        assert graph['timed_thermal_samples']==0 and graph['status']=='audited_qualified'
        assert graph['study_sha256']==digest(PIPELINE/'artifacts/remaining_graph_performance_20261104/study.json')
        assert (PIPELINE/'reports/remaining_graph_performance_20261104_audit_v2/visual_review.json').exists()
        atomic_json(OUT/'graph_approval_snapshot.json',dict(receipt=graph,receipt_sha256=digest(graph_receipt_path),
            visual_review=load(PIPELINE/'reports/remaining_graph_performance_20261104_audit_v2/visual_review.json')))
        art=PIPELINE/'artifacts/remaining_plateau_20261104'
        assert not art.exists(), 'Plateau already started; no automatic restart'
        sources=[Path(__file__),PIPELINE/'analysis/audit_remaining_work_v2.py',PIPELINE/'analysis/audit_telemetry_boundaries_v2.py',
            PIPELINE/'experiments/measure_remaining_plateau.py',PIPELINE/'experiments/remaining_work_design.py',
            PIPELINE/'experiments/remaining_work_windows.py',old/'plan.json',old/'study.json',graph_receipt_path,
            OUT/'graph_approval_snapshot.json']
        plan=dict(authority=state['authority'],stage='plateau_only',expected_windows=36,
            no_original_controller_restart=True,no_graph_recollection=True,
            original_campaign_plan_sha256=digest(old/'plan.json'),protected_sha256={str(p):digest(p) for p in sources},
            original_protected_sources=original_plan['protected_sources'],
            completion='Plateau worker exits, version2 CPU audit passes, monitor verifies final plot before closure.')
        atomic_json(OUT/'plan.json',plan);state['plan_sha256']=digest(OUT/'plan.json')
        art.mkdir()
        script=PIPELINE/'experiments/measure_remaining_plateau.py'
        state.update(status='measuring',phase='plateau');atomic_json(OUT/'study.json',state)
        with (art/'run.stdout.log').open('x') as stdout,(art/'run.stderr.log').open('x') as stderr:
            child=subprocess.Popen([sys.executable,'-u',str(script)],cwd=REPOSITORY_ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            time.sleep(2)
            identity=capture(child.pid,'measure_remaining_plateau.py');atomic_json(art/'launch_identity.json',identity)
            state['active_identity']=identity;atomic_json(OUT/'study.json',state)
            code=child.wait()
        state['active_identity']=None
        if code: raise RuntimeError('Plateau child failed; preserve and stop without retry')
        plateau=load(art/'study.json')
        assert plateau['status']=='complete_pending_audit' and len(plateau['rows'])==36
        state.update(status='cpu_audit',phase='plateau_audit');atomic_json(OUT/'study.json',state)
        with (OUT/'audit.stdout.log').open('x') as stdout,(OUT/'audit.stderr.log').open('x') as stderr:
            audit=subprocess.run(['C:/Users/Eren/radioconda/python.exe',str(PIPELINE/'analysis/audit_remaining_work_v2.py'),'--kind','plateau'],cwd=REPOSITORY_ROOT,stdout=stdout,stderr=stderr)
        if audit.returncode: raise RuntimeError('Plateau audit failed; preserve, no retry')
        for p,h in {**plan['protected_sha256'],**plan['original_protected_sources']}.items(): assert digest(p)==h,p
        final=load(PIPELINE/'reports/remaining_plateau_20261104_audit_v2/audited_summary.json')
        assert final['audit_completed'] and final['windows']==36
        REPORT.mkdir(parents=True,exist_ok=True)
        atomic_json(REPORT/'audited_summary.json',dict(status='audits_complete_pending_visual_review',
            graph_windows=72,plateau_windows=36,
            graph_receipt=str(graph_receipt_path),graph_receipt_sha256=digest(graph_receipt_path),
            plateau_receipt=str(PIPELINE/'reports/remaining_plateau_20261104_audit_v2/audited_summary.json'),
            plateau_receipt_sha256=digest(PIPELINE/'reports/remaining_plateau_20261104_audit_v2/audited_summary.json'),
            all_original_and_continuation_hashes_verified=True,finished_utc=utc(),
            hpc='Still blocked pending external scheduler/log evidence. No remote intervention.'))
        state.update(status='complete_pending_visual_audit',finished_utc=utc(),phase='await_agent_plot_review')
    except BaseException as exc:
        state.update(status='stopped_error',error=repr(exc),finished_utc=utc());raise
    finally: atomic_json(OUT/'study.json',state)


if __name__=='__main__': main()
