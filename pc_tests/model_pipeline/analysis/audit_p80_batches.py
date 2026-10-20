"""Audit saved p80 evidence only; no inference, training or protocol changes."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import json
import math
import statistics as st
from collections import Counter
from pathlib import Path
from pc_tests.model_pipeline.experiments.run_p80_batches import PIPELINE, FOLDER, BATCH_SIZES, select_p80, validation_choices
from pc_tests.model_pipeline.experiments.run_random_routes import digest
from pc_tests.model_pipeline.provenance import verify_source


def main():
    report = json.loads((FOLDER/'p80_routes.json').read_text())
    protocol = report['protocol']
    rows = report['results']
    assert report['status'] == 'complete' and len(rows) == report['expected_sweeps'] == 440
    assert len({r['id'] for r in rows}) == 440
    assert Counter(r['batch_size'] for r in rows) == {b:110 for b in BATCH_SIZES}
    expected = {f'b{b}_{name}_trial{t}' for b in BATCH_SIZES for t in range(5)
                for name in protocol['models']+[protocol['reference']]}
    assert {r['id'] for r in rows} == expected
    cache = PIPELINE/'artifacts/pc_precision/data_cache'
    checks = {'static_sha256': PIPELINE/'reports/gpu_static.json',
              'labels_sha256': cache/'test_labels.npy', 'rgb_sha256': cache/'test_rgb128.npy',
              'script_sha256': PIPELINE/'run_p80_batches.py',
              'thermal_reader_sha256': PIPELINE/'run_most_pruned_batches.py',
              'runtime_inputs_sha256': PIPELINE/'gpu_batch_inputs.py',
              'engine_cache_sha256': PIPELINE/'global_candidate_pool.py'}
    for key, path in checks.items():
        if path.suffix == '.py':
            verify_source(path.name, protocol[key])
        else:
            assert digest(path) == protocol[key], (key, path)
    for path, value in protocol['protected_reports'].items():
        assert digest(PIPELINE/path) == value, path
    records = json.loads(checks['static_sha256'].read_text())['results']
    selected = select_p80(records)
    reference = next(r for r in records if r['name'] == protocol['reference'])
    assert [r['name'] for r in selected] == protocol['models']
    assert validation_choices(records, selected, reference, cache) == protocol['validation_selected']
    for record in selected+[reference]:
        engine = Path(record['engine_path'])
        assert engine.drive.upper() == 'C:'
        assert digest(engine) == record['engine_sha256'], record['name']
    for row in rows:
        assert row['status'] == 'complete' and row['cache_loads'] == 1 and row['cache_evictions'] == 0
        assert 0 <= row['accuracy'] <= 100
        assert math.isfinite(row['latency_ms_per_image']) and row['latency_ms_per_image'] > 0
        assert math.isfinite(row['gpu_energy_j_per_image']) and row['gpu_energy_j_per_image'] > 0
        assert math.isclose(row['elapsed_seconds']/10, row['latency_ms_per_image'])
        assert row['prune'] == (0 if row['role'] == 'reference' else 80)
    summary = json.loads((FOLDER/'p80_summary.json').read_text())
    assert summary['complete'] and len(summary['rows']) == 84
    assert len({(r['name'],r['batch_size']) for r in summary['rows']}) == 84
    for item in summary['rows']:
        group = [r for r in rows if r['name']==item['name'] and r['batch_size']==item['batch_size']]
        refs = [r for r in rows if r['role']=='reference' and r['batch_size']==item['batch_size']]
        assert len(group) == len(refs) == 5 and {r['trial'] for r in group} == set(range(5))
        latency = st.median(r['latency_ms_per_image'] for r in group)
        energy = st.median(r['gpu_energy_j_per_image'] for r in group)
        base_l = st.median(r['latency_ms_per_image'] for r in refs)
        base_e = st.median(r['gpu_energy_j_per_image'] for r in refs)
        accuracy = st.mean(r['accuracy'] for r in group)
        loss = st.mean(r['accuracy'] for r in refs)-accuracy
        for value, calculated in [(item['accuracy'],accuracy), (item['loss_pp'],loss),
                                  (item['latency_ms_per_image'],latency),
                                  (item['energy_mj_per_image'],1000*energy),
                                  (item['latency_saving_percent'],100*(base_l-latency)/base_l),
                                  (item['energy_saving_percent'],100*(base_e-energy)/base_e)]:
            assert math.isclose(value,calculated,abs_tol=1e-9)
        assert item['passing_allowances'] == [b for b in (.5,1,1.5,2) if loss <= b+1e-10]
    original = json.loads((PIPELINE/'reports/pc_routes_correctness.json').read_text())
    previous = {}
    for batch in BATCH_SIZES:
        group = [r for r in original['results'] if r['batch_size']==batch]
        previous[str(batch)] = {key:[min(r[key] for r in group), max(r[key] for r in group)]
                               for key in ('latency_saving_vs_p0_fp32_percent',
                                           'energy_saving_vs_p0_fp32_percent')}
        previous[str(batch)]['accuracy_range'] = [min(st.mean(m['test_accuracy'] for m in r['measurements']) for r in group),
                                                   max(st.mean(m['test_accuracy'] for m in r['measurements']) for r in group)]
    audit = dict(complete=True,unique_sweeps=440,sweeps_per_batch=110,summary_rows=84,
                 all_source_input_engine_protected_hashes_verified=True,
                 frozen_validation_choices_verified=True,
                 null_validation_choices=sum(v is None for v in protocol['validation_selected'].values()),
                 cached_prediction_disagreements=sum(r['cached_prediction_disagreements'] for r in rows),
                 evaluations=4400000,accuracy_range=[min(r['accuracy'] for r in summary['rows']),max(r['accuracy'] for r in summary['rows'])],
                 latency_saving_range=[min(r['latency_saving_percent'] for r in summary['rows']),max(r['latency_saving_percent'] for r in summary['rows'])],
                 energy_saving_range=[min(r['energy_saving_percent'] for r in summary['rows']),max(r['energy_saving_percent'] for r in summary['rows'])],
                 accuracy_passing_model_batch_rows=sum(bool(r['passing_allowances']) for r in summary['rows']),
                 negative_latency_rows=sum(r['latency_saving_percent']<0 for r in summary['rows']),
                 negative_energy_rows=sum(r['energy_saving_percent']<0 for r in summary['rows']),
                 original_oct13_fixed_fp32_comparison=previous)
    (FOLDER/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
