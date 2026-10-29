"""Validate staged publication scope, byte hashes and common credential patterns."""
import ast,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
manifest_path='progress_reports/2026-10-29/code_manifest.json'
manifest=json.loads((REPO/manifest_path).read_text())
staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).splitlines())
expected={r['path'] for r in manifest['files']}|{manifest_path}
renamed={'progress_reports/2026-10-29/main.tex','progress_reports/2026-10-29/main.pdf'}
assert staged<=expected|renamed
secret=re.compile(rb'(?:sk-[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')
for record in manifest['files']:
    rel=record['path'];data=subprocess.check_output(['git','show',':'+rel],cwd=REPO)
    assert hashlib.sha256(data).hexdigest()==record['sha256'],rel
    assert len(data)==record['bytes'],rel
    assert not secret.search(data),rel
    assert not rel.startswith(('jetson_tests/','hpc/','progress_reports/2026-10-20/'))
    assert not any(t in rel for t in ('/artifacts/','.engine','.pth','.onnx','.npz','.npy','study.json','shutdown_receipt'))
    if rel.endswith('.py'):ast.parse(data.decode('utf-8-sig'),filename=rel)
report=json.loads((ROOT/'report_checks.json').read_text())
assert report['passed'] and report['pages']==3
assert hashlib.sha256((ROOT/'progress_report.pdf').read_bytes()).hexdigest()==report['pdf_sha256']
print(json.dumps(dict(staged_files=len(staged),byte_hashes_verified=len(manifest['files']),python_syntax='passed',credential_pattern_scan='passed',raw_artifact_exclusion='passed',pdf_pages=3),indent=2))
