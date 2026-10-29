"""Record the explicitly scoped publication files; never launches experiments."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).splitlines()
new=git('ls-files','--others','--exclude-standard')
groups=['pc_tests/model_pipeline/'+x+'/' for x in ('analysis','experiments','runtime','routers/correctness','scripts')]
excluded={'request_completion_shutdown.py','record_compact_completion.py','snapshot_experiment_pause.py','resume_experiment_checkpoint.py'}
selected={p for p in new if any(p.startswith(g) for g in groups) and Path(p).suffix in ('.py','.ps1') and Path(p).name not in excluded}
previous=ROOT/'code_manifest.json'
if previous.exists():
    for record in json.loads(previous.read_text())['files']:
        rel=record['path']
        if rel in ('progress_reports/2026-10-29/main.tex','progress_reports/2026-10-29/main.pdf'):
            rel=rel.replace('/main.','/progress_report.')
        selected.add(rel)
tests=('ablation_design','atomic_checkpoint','b1_engine_profiles','compact_b1','compact_subset','conditional_onnx','correctness_b1','correctness_payback','correctness_training_energy','device_selection','experiment_guard','gpu_residency','grouped_dispatch')
selected.update('tests/test_'+n+'.py' for n in tests)
studies=('b1_engine_profiles_20261028','b1_int8_diagnostic_20261028','batch8_grouping_20261029','compact_b1_20261028','compact_resident_screen_20261028','compact_runtime_profile_20261028','compact_switching_confirmation_20261028','compact_switching_pilot_20261028','compact_validation_20261028','correctness_training_energy_20261028','dispatch_controls_20261029','gpu_conditional_20261029','gpu_dispatch_20261028','gpu_residency_20261028','router_supervision_screen_20261029','supervision_deployment_20261029')
small_names={'audited_summary.json','README.md','results.md','comparison.csv','SETTINGS_ABLATIONS.md'}
for study in studies:
    for p in (REPO/'pc_tests/model_pipeline/reports'/study).iterdir():
        if p.is_file() and p.name in small_names:selected.add(p.relative_to(REPO).as_posix())
for rel in ('pc_tests/model_pipeline/reports/correctness_b1_20261028/existing_only','pc_tests/model_pipeline/reports/compact_b1_20261028/final_summary'):
    for p in (REPO/rel).iterdir():
        if p.is_file() and p.suffix in ('.png','.svg','.md','.csv','.json') and p.name not in ('analysis_receipt.json','completion_receipt.json','shutdown_receipt.json'):
            selected.add(p.relative_to(REPO).as_posix())
selected.add('pc_tests/model_pipeline/reports/correctness_b1_20261028/README.md')
selected.add('pc_tests/model_pipeline/reports/correctness_b1_20261028/stop_receipt.json')
selected.update(p for p in new if p.startswith('progress_reports/2026-10-29/'))
selected.discard('progress_reports/2026-10-29/code_manifest.json')
for p in selected:
    assert Path(p).suffix not in ('.engine','.pth','.npz','.npy','.onnx')
    assert '/artifacts/' not in p and Path(p).name!='study.json'
files=[]
for rel in sorted(selected):
    p=REPO/rel
    assert p.is_file() and p.stat().st_size<1_000_000,rel
    files.append(dict(path=rel,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert sum(r['bytes'] for r in files)<15_000_000
manifest=dict(scope='29 October report, related PC experiment/runtime/analysis/launcher code, CPU tests and small audited numerical summaries. Excludes unrelated changes, datasets, engines/weights/prediction arrays, raw study checkpoints and shutdown/pause receipts.',
    verification=dict(cpu_tests_passed=38,cuda_devices_hidden=True,report_pages=3,report_verifier_passed=True),
    limitations='Recompiling supplied TeX needs only report assets; replaying measurements/audits needs separately retained local data/engines/raw reports. Local launchers contain exact-identity/thermal controls and may request shutdown; do not run as generic examples. Cancelled full batch1 baseline remains cancelled.',files=files)
(ROOT/'code_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(files),bytes=sum(r['bytes'] for r in files)),indent=2))
