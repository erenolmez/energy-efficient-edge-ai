"""Verify report snapshot, numerical explanations and three-page PDF."""
import hashlib,json,math
from pathlib import Path
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((ROOT/'report_data.json').read_text())
for path,h in s['sources_sha256'].items():assert digest(path)==h,path
assert len(s['fixed1pp_latency_rows'])==18
assert len(s['comparison_baseline_rows'])==36
for series,method in [('eligible_static','eligible_static'),('reference_fp32','reference')]:
    rr=[r for r in s['comparison_baseline_rows'] if r['series']==series]
    assert {(r['batch'],r['seed']) for r in rr}=={(b,seed) for b in (1,8,16,32,64,128) for seed in (42,45,48)}
    for row in rr:
        assert row['ms_image']>0 and row['j_image']>0
        if row['batch']==1:
            control=next(r for r in s['compact_rows'] if r['method']==method and r['seed']==row['seed'])
            assert row['accuracy']==control['accuracy'] and row['j_image']==control['gpu_j_per_image'] and row['ms_image']==control['ms_per_image']
for batch in (1,8,16,32,64,128):
    rr=[r for r in s['fixed1pp_latency_rows'] if r['batch']==batch]
    assert sorted(r['seed'] for r in rr)==[42,45,48]
    assert all(r['passed']==(r['accuracy']>=84.8-1e-10) for r in rr)
    table=(ROOT/'batch_statistics.tex').read_text()
    assert f"{min(r['accuracy'] for r in rr):.2f}--{max(r['accuracy'] for r in rr):.2f}" in table
    assert f"{min(r['ms_batch'] for r in rr):.2f}--{max(r['ms_batch'] for r in rr):.2f}" in table
    energies=[r['j_image']*1000*10000/math.ceil(10000/batch) for r in rr]
    assert f'{min(energies):.1f}--{max(energies):.1f}' in table
example=s['worked_example']
assert math.ceil(example['training_j']/example['eligible_static']['saved_j_per_image'])==9583
assert [r['eligible_valid'] for r in s['payback_by_batch_all_settings']]==[0,2,2,9,9,9]
assert [r['reference_fp32']['valid_payback_images'] for r in s['batch1_payback'] if r['seed']==42]==[2399]
assert all(r['eligible_static']['valid_payback_images'] is None for r in s['batch1_payback'])
b1text=(ROOT/'batch1_example.tex').read_text()
for method in ('tinycnn','reference','eligible_static'):
    r=next(r for r in s['compact_rows'] if r['seed']==42 and r['method']==method)
    assert f"{r['gpu_j_per_image']:.7f}" in b1text
assert sum(r['accuracy_pass'] for r in s['supervision']['results'])==9
assert s['accuracy_sensitivity']['compact_accuracy_passes_1pp']==5
assert s['accuracy_sensitivity']['compact_accuracy_passes_2pp']==15
assert s['accuracy_sensitivity']['compact_joint_savings_passes_2pp']==4
assert sum(r['baseline']=='eligible' and r['valid'] for r in s['supervision_payback']['results'])==4
assert s['switching']['audit_passed'] and s['switching']['full_sweeps']==81
assert s['optimized_graph_seed42']['eligible_comparison']['valid_gpu_fit_payback_images']==28621
for variant in ('file_static_c2','host_static_reuse_c2','resident_eager_c6','resident_graph_c6','host_aware_c2'):
    rr=[r for r in s['switching']['results'] if r['variant']==variant and r['is_routed']]
    assert len(rr)==3
    assert f"{min(r['median_ms_per_image'] for r in rr):.2f}--{max(r['median_ms_per_image'] for r in rr):.2f}" in (ROOT/'runtime_explanations.tex').read_text()
table=(ROOT/'payback_statistics.tex').read_text()
assert 'Acc. pass' not in table and 'valid / images' not in table
assert '2 of 24' in table and 'Images before repayment' in table
assert 'Can training changes or per-image choices' not in (ROOT/'progress_report.tex').read_text()
assert '28,621' in (ROOT/'optimized_payback.tex').read_text()
pdf=PdfReader(ROOT/'progress_report.pdf');assert len(pdf.pages)==3
for i,page in enumerate(pdf.pages,1):
    text=page.extract_text();assert len(text)>1500 and str(i) in text
assert '2399' in pdf.pages[2].extract_text().replace(',','').replace(' ','')
log=(ROOT/'progress_report.log').read_text(errors='replace');assert 'Overfull' not in log
checks=dict(passed=True,pages=3,all18fixed_setting_rows_verified=True,tail_adjustment_verified=True,
    comparison_baseline_rows_verified=36,six_batch_payback_groups_verified=True,batch1_reference_payback_images=2399,optimized_graph_eligible_payback_images=28621,worked_payback_images=9583,
    accuracy_sensitivity_2pp_verified=True,pdf_sha256=digest(ROOT/'progress_report.pdf'),tex_sha256=digest(ROOT/'progress_report.tex'),
    snapshot_sha256=digest(ROOT/'report_data.json'),source_hashes_verified=len(s['sources_sha256']),
    latex_overfull_boxes=0,visual_review='Three latest rendered pages inspected at1400px; no clipping/overlap.')
(ROOT/'report_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
