"""CPU-only provenance/reserve/count audit and readable capacity summary."""
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
    output=PIPELINE/'reports/gpu_residency_20261028'
    path=output/'study.json'
    report=json.loads(path.read_text())
    assert report['status']=='complete'
    assert report['test_labels_opened'] is False and report['training_performed'] is False
    assert digest(PIPELINE/'reports/gpu_static.json')==report['static_report_sha256']
    assert digest(PIPELINE/'reports/correctness_b1_20261028/study.json')==report['stopped_study_sha256']
    assert digest(PIPELINE/'experiments/probe_gpu_residency.py')==report['source_sha256']
    assert digest(PIPELINE/'runtime/gpu_engine_preflight.py')==report['engine_runner_sha256']
    static=json.loads((PIPELINE/'reports/gpu_static.json').read_text())
    by_name={r['name']:r for r in static['results']}
    assert set(report['orders'])=={'smallest_first','largest_first','architecture_round_robin'}
    assert len({r['id'] for r in report['results']})==len(report['results'])
    summary=[]
    for name,order in report['orders'].items():
        rows=[r for r in report['results'] if r['order']==name]
        assert len(rows)==order['count']==len(order['resident_models'])
        assert len(set(order['resident_models']))==order['count']
        assert order['all_resident_verified'] and order['maximum_retained_logits_difference']<=1e-5
        assert order['free_after_verification_bytes']>=report['reserve_bytes']
        assert [r['count'] for r in rows]==list(range(1,len(rows)+1))
        for row in rows:
            assert row['free_bytes']>=report['reserve_bytes']
            assert row['engine_sha256']==by_name[row['added_model']]['engine_sha256']
            assert digest(Path(by_name[row['added_model']]['engine_path']))==row['engine_sha256']
        summary.append(dict(order=name,usable_classifier_engines=order['count'],
            final_used_mib=order['used_bytes']/1024**2,final_free_mib=order['free_after_verification_bytes']/1024**2,
            stop_reason=order['stop_reason'],all_resident_verified=True))
    result=dict(audit_passed=True,report_sha256=digest(path),rows=summary,reserve_mib=1024,
        batch_size=1,gpu=report['gpu'],router_additionally_resident=True,
        claim='Demonstrated safe counts under frozen orders/reserve; not absolute maximum or speedup.')
    atomic_json(output/'audited_summary.json',result)
    lines=['# Measured safe GPU residency','',report['gpu']+'; batch1, tinyrouter also resident; 1GiBfree reserve.','',
        '| Loading order | Usable classifier engines | GPU used (MiB) | GPU free after verification (MiB) |',
        '|---|---:|---:|---:|']
    lines += [f"| {r['order']} | {r['usable_classifier_engines']} | {r['final_used_mib']:.0f} | {r['final_free_mib']:.0f} |" for r in summary]
    lines += ['',result['claim'],'','All retained engines were kept alive and re-executed with synthetic-input parity. Counts vary with model mix, context allocation, other GPU consumers and the safety reserve. No classifier accuracy or routing latency/energy saving is established by this capacity test.','',
        'The stopped multi-day baseline remains preserved and must not be resumed.']
    (output/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
