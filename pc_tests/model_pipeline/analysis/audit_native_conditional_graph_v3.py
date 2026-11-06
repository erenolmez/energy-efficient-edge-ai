"""CPU-only independent audit after the bounded graph correctness probe exits."""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import subprocess
import numpy as np

PIPE=Path(__file__).resolve().parents[1]
ART=PIPE/'artifacts/native_conditional_graph_v3_20261104'
OUT=PIPE/'reports/native_conditional_graph_v3_20261104'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def main():
    raw=subprocess.check_output(['powershell.exe','-NoProfile','-Command',
        '@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match \'^python(w)?\\.exe$\' } | Select-Object ProcessId,ExecutablePath,CommandLine,@{n="creation_unix_ms";e={[DateTimeOffset]::new($_.CreationDate).ToUnixTimeMilliseconds()}}) | ConvertTo-Json -Depth 5'],text=True)
    live=json.loads(raw) if raw.strip() else []
    if isinstance(live,dict): live=[live]
    assert not any('probe_native_conditional_graph' in (r.get('CommandLine') or '') for r in live), 'Probe still active'
    state=load(ART/'probe.json')
    assert state['status']=='correctness_verified_pending_audit',state.get('error',state['status'])
    assert state['test_data_accessed'] is False and state['instantiation_succeeded']
    worker=load(ART/'worker_identity.json'); launch=load(ART/'launch_identity.json')
    assert worker['ProcessId']==state['pid'] and worker in launch['processes']
    assert len(launch['processes'])==2
    for saved in launch['processes']:
        assert saved['creation_unix_ms']>0 and 'probe_native_conditional_graph_v3.py' in saved['CommandLine']
        assert saved['ExecutablePath'].endswith('python.exe')
        assert not any(all(p.get(k)==saved[k] for k in ('ProcessId','ExecutablePath','CommandLine','creation_unix_ms')) for p in live)
    for path,expected in state['protected_hashes'].items():
        assert digest(path)==expected,path
    for name,expected in state['replay_files_sha256'].items():
        assert digest(ART/name)==expected,name
    counts=state['graph_node_counts']
    assert counts['parent'][1:]==[0,2]
    assert all(counts[b][0]>1 and counts[b][1:]==[0,0] for b in ('branchA','branchB'))
    rows=state['verified']
    assert len(rows)==state['verified_replays']==256
    assert len({(r['pass_id'],r['input_index']) for r in rows})==256
    for p in (0,1):
        assert sorted(r['input_index'] for r in rows if r['pass_id']==p)==list(range(128))
    refs=PIPE/'artifacts/dispatch_residency_v2_20261104'
    logits=np.load(ART/'verified_logits.npy',allow_pickle=False)
    branches=np.load(ART/'verified_branch_counts.npy',allow_pickle=False)
    ref_logits=np.load(refs/'reference_logits.npy',allow_pickle=False)
    ref_choices=np.load(refs/'reference_choices.npy',allow_pickle=False)
    ref_predictions=np.load(refs/'reference_predictions.npy',allow_pickle=False)
    assert logits.shape==(256,100) and branches.shape==(256,2)
    assert np.isfinite(logits).all()
    maxerr=0
    for n,r in enumerate(rows):
        i=r['input_index']
        assert r['choice']==int(ref_choices[i]) and r['prediction']==int(ref_predictions[i])
        assert int(np.argmax(logits[n]))==r['prediction']
        expected=np.zeros(2,dtype=np.int32);expected[r['choice']]=1
        assert np.array_equal(branches[n],expected) and r['branch_counts']==expected.tolist()
        assert np.allclose(logits[n],ref_logits[i],atol=1e-3,rtol=1e-4)
        error=float(np.max(np.abs(logits[n]-ref_logits[i])))
        assert np.isclose(error,r['max_abs_logit_error'])
        maxerr=max(maxerr,error)
    assert np.isclose(maxerr,state['max_abs_logit_error'])
    telemetry=[json.loads(line) for line in (ART/'telemetry.jsonl').read_text().splitlines()]
    assert telemetry and all('error' not in t and t['temperature_c']<80 for t in telemetry)
    assert any(t['phase']=='graph_correctness' for t in telemetry)
    times=[datetime.fromisoformat(t['utc']).timestamp() for t in telemetry]
    gaps=[b-a for a,b in zip(times,times[1:])]
    assert all(g>0 for g in gaps) and max(gaps)<2
    dot=(ART/'native_graph.dot').read_text()
    assert dot.count('Conditional Type| IF')==2
    receipt=dict(status='correctness_repair_verified_audited_qualified',audit_completed=True,
        checked_utc=datetime.now(timezone.utc).isoformat(),verified_replays=256,
        choice_prediction_or_selected_branch_mismatches=0,max_abs_logit_error=maxerr,
        branch_selections=dict(Counter(str(r['choice']) for r in rows)),graph_node_counts=counts,
        protected_hash_count=len(state['protected_hashes']),protected_hashes_verified=True,
        identities_match_launch_receipt=True,probe_processes_exited=True,live_python_snapshot=live,
        telemetry_samples=len(telemetry),max_gap_seconds=max(gaps),peak_c=max(t['temperature_c'] for t in telemetry),
        thermal_flag_samples=sum(bool(t['clock_reasons']&32) for t in telemetry),
        power_cap_flag_samples=sum(bool(t['clock_reasons']&4) for t in telemetry),
        probe_sha256=digest(ART/'probe.json'),replay_files_sha256=state['replay_files_sha256'],
        dot_sha256=digest(ART/'native_graph.dot'),
        qualifications=['Correctness/support probe only, not a latency/energy comparison.',
            'CPU submits launches, stages inputs and validates outputs; decisions/selected inference/argmax in graph run on GPU.',
            'Two FP32 classifiers, fixed median-biased engineering policy and 128 validation inputs; not an accuracy-budget policy or whole 213-model graph.',
            'CUDA12.6 helper/MSVC14.44 unsupported-compiler flag; mixed PyTorch cu118/TensorRT runtime configuration.',
            'Original TensorRT native-If engine is still not capturable; this repair uses explicit CUDA conditional nodes with standalone engines.',
            'Earlier invalid graph speedups remain invalid and excluded; no performance conclusion follows here.'])
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'audited_summary.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    lines=['# CUDA Graph correctness repair verified','',
        f"256/256 independent poisoned-output replays passed: exact routing choice, prediction and selected-only branch counters; full-logit tolerance passed (maximum absolute error {maxerr:.9g}). Both 128-input passes were checked, including shuffled order. Saved replay logits/counters rechecked independently after process exit.",'',
        'Driver-level node inspection passed recursively: parent has two conditional nodes; both classifier body graphs are nonempty; no host callback nodes. Explicit graph instantiation and replay succeeded. Protected hashes, reference/replay hashes and exact worker launch identities match. Worker/launcher exited.','',
        f"Telemetry: {len(telemetry)} samples, peak {receipt['peak_c']} C, maximum gap {max(gaps):.3f} s, {receipt['thermal_flag_samples']} thermal-flag and {receipt['power_cap_flag_samples']} power-cap samples (entire probe; not timed measurements).",'',
        '## What was fixed','',
        'Revision2 localized CUDA runtime cudaGraphNodeGetType error999 in parent-node inspection before instantiation. Revision3 uses checked driver cuGraphNodeGetType instead, preserving rather than skipping structure checks. Captured standalone engines are joined using explicit CUDA conditional bodies. Earlier failed probes remain preserved.','',
        '## Scope and qualifications','',*['- '+q for q in receipt['qualifications']], '',
        '[NVIDIA driver graph API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html); [matching CUDA12.6 GetType error reproduction](https://forums.developer.nvidia.com/t/cannot-get-node-type-of-conditional-nodes-with-cudagraphnodegettype/311412).']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
