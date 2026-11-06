"""Exclusive finite campaign: graph measurement/audit, plateau measurement/audit."""
import sys
from pathlib import Path
if not __package__: sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
import os
import subprocess
import time
from pc_tests.model_pipeline.paths import PIPELINE,REPOSITORY_ROOT
from pc_tests.model_pipeline.experiments.compare_dispatch_residency_v2 import digest,utc
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json

OUT=PIPELINE/'artifacts/remaining_work_20261104'


def capture(pid,script):
    command=f'@(Get-CimInstance Win32_Process | Where-Object {{ ($_.ProcessId -eq {pid} -or $_.ParentProcessId -eq {pid}) -and $_.Name -eq "python.exe" }} | Select-Object ProcessId,ExecutablePath,CommandLine,@{{n="creation_unix_ms";e={{[DateTimeOffset]::new($_.CreationDate).ToUnixTimeMilliseconds()}}}}) | ConvertTo-Json -Depth 4'
    raw=subprocess.check_output(['powershell.exe','-NoProfile','-Command',command],text=True)
    rows=json.loads(raw) if raw.strip() else []
    if isinstance(rows,dict): rows=[rows]
    assert rows and all(script in p['CommandLine'] and p['creation_unix_ms']>0 for p in rows)
    return dict(utc=utc(),expected_script=script,processes=rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('x') as f: json.dump(dict(pid=os.getpid(),utc=utc()),f)
    state=dict(status='preparing',pid=os.getpid(),started_utc=utc(),stages=[],test_data_accessed=False)
    atomic_json(OUT/'study.json',state)
    try:
        atomic_json(OUT/'worker_identity.json',capture(os.getpid(),'run_remaining_work.py'))
        sources=[Path(__file__),PIPELINE/'experiments/measure_remaining_graph.py',PIPELINE/'experiments/measure_remaining_plateau.py',
            PIPELINE/'experiments/remaining_work_windows.py',PIPELINE/'experiments/remaining_work_design.py',
            PIPELINE/'analysis/audit_remaining_work.py',PIPELINE/'runtime/native_conditional_graph_v3.cu',
            PIPELINE/'artifacts/native_conditional_graph_v3_20261104/native_conditional_graph.dll',
            PIPELINE/'experiments/compare_dispatch_residency_v2.py',PIPELINE/'runtime/gpu_energy.py']
        plan=dict(stages=[dict(kind='graph',script='measure_remaining_graph.py',artifact='remaining_graph_performance_20261104',windows=72),
            dict(kind='plateau',script='measure_remaining_plateau.py',artifact='remaining_plateau_20261104',windows=36)],
            protected_sources={str(p):digest(p) for p in sources},
            scope='Only explicitly authorized PC follow-ups; no HPC/SSH/W&B changes, training/test/Jetson/E:/hardware changes or shutdown.',
            completion='Each child exits before its CPU audit; all audits must pass then agent visually reviews plots. No automatic retry/resume after errors or shutdown.')
        assert all(not (PIPELINE/'artifacts'/s['artifact']).exists() for s in plan['stages']), 'Old/new study directory already exists; no restart'
        atomic_json(OUT/'plan.json',plan);state['plan_sha256']=digest(OUT/'plan.json')
        for stage in plan['stages']:
            for p,h in plan['protected_sources'].items(): assert digest(p)==h,p
            art=PIPELINE/'artifacts'/stage['artifact'];art.mkdir()
            script=PIPELINE/'experiments'/stage['script']
            state.update(status='measuring',current=stage);atomic_json(OUT/'study.json',state)
            with (art/'run.stdout.log').open('x') as stdout,(art/'run.stderr.log').open('x') as stderr:
                process=subprocess.Popen([sys.executable,'-u',str(script)],cwd=REPOSITORY_ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                time.sleep(2)
                receipt=capture(process.pid,stage['script']);atomic_json(art/'launch_identity.json',receipt)
                state['active_identity']=receipt;atomic_json(OUT/'study.json',state)
                code=process.wait()
            state['active_identity']=None
            if code: raise RuntimeError(f'{stage["kind"]} worker failed ({code}); preserve artifacts, no retry')
            result=json.loads((art/'study.json').read_text())
            assert result['status']=='complete_pending_audit' and len(result['rows'])==stage['windows']
            state.update(status='cpu_audit',current=stage);atomic_json(OUT/'study.json',state)
            with (OUT/f'{stage["kind"]}_audit.stdout.log').open('x') as stdout,(OUT/f'{stage["kind"]}_audit.stderr.log').open('x') as stderr:
                audit=subprocess.run(['C:/Users/Eren/radioconda/python.exe',str(PIPELINE/'analysis/audit_remaining_work.py'),'--kind',stage['kind']],cwd=REPOSITORY_ROOT,stdout=stdout,stderr=stderr)
            if audit.returncode: raise RuntimeError(f'{stage["kind"]} audit failed; preserve and request diagnosis, no next GPU stage')
            state['stages'].append(dict(**stage,status='audited_pending_visual_review'))
            atomic_json(OUT/'study.json',state)
        with (OUT/'final_audit.stdout.log').open('x') as stdout,(OUT/'final_audit.stderr.log').open('x') as stderr:
            audit=subprocess.run(['C:/Users/Eren/radioconda/python.exe',str(PIPELINE/'analysis/audit_remaining_work.py')],cwd=REPOSITORY_ROOT,stdout=stdout,stderr=stderr)
        if audit.returncode: raise RuntimeError('Final campaign audit failed; preserve, no retry')
        state.update(status='complete_pending_visual_audit',finished_utc=utc(),active_identity=None)
    except BaseException as exc:
        state.update(status='stopped_error',error=repr(exc),finished_utc=utc());raise
    finally: atomic_json(OUT/'study.json',state)


if __name__=='__main__': main()
