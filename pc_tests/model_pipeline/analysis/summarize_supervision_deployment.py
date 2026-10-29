"""Audit the predeclared seed42 deployment screen without selecting test policies."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import csv
import json
from pathlib import Path
import numpy as np
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.runtime.atomic_checkpoint import atomic_json


def main():
    path = PIPELINE / 'reports/supervision_deployment_20261029/study.json'
    artifacts = PIPELINE / 'artifacts/supervision_deployment_20261029'
    study = json.loads(path.read_text())
    assert study['status'] == 'complete'
    assert len(study['full_sweeps']) == 20 and len(study['results']) == 406
    assert len({r['id'] for r in study['results']}) == 406
    assert study['source_sha256'] == digest(PIPELINE / 'experiments/validate_supervision_deployment.py')
    assert study['predecessor_sha256'] == digest(PIPELINE / 'reports/gpu_conditional_20261029/audited_summary.json')
    training_path = PIPELINE / 'reports/router_supervision_screen_20261029/study.json'
    assert study['training_sha256'] == digest(training_path)
    training = json.loads(training_path.read_text())
    assert study['frozen_plan_sha256'] == digest(artifacts / 'frozen_plan.json')
    plan = json.loads((artifacts / 'frozen_plan.json').read_text())
    for group in ('protected_hashes', 'test_hashes'):
        for p, h in study[group].items():
            assert digest(p) == h
    assert len(study['prepared']) == 6
    for p in study['prepared']:
        assert digest(p['engine']) == p['engine_sha256']
        if p['name'] != 'original_control':
            assert 0 <= p['random_parity_error'] <= 1e-5
            assert 0 <= p['real_parity_error'] <= 1e-5
            fit = next(f for f in training['fits'] if f['name'] == p['name'])
            assert p['weights_sha256'] == fit['weights_sha256'] == digest(fit['weights'])
    rows = {r['case']: r for r in study['full_sweeps']}
    assert len(rows) == 20 and set(rows) == set(plan['settings'])
    labels = np.load(PIPELINE / 'artifacts/pc_precision/data_cache/test_labels.npy', allow_pickle=False)
    for case, row in rows.items():
        chunks = [r for r in study['results'] if r.get('case') == case]
        assert len(chunks) == 20
        assert sorted(r['completed_images'] for r in chunks) == list(range(500, 10001, 500))
        for r in chunks:
            assert r['images'] == 500 and r['observed_free_bytes'] >= 1024**3
            assert np.isfinite(r['seconds']) and r['seconds'] > 0
            assert np.isfinite(r['gpu_energy']['gpu_energy_j']) and r['gpu_energy']['gpu_energy_j'] > 0
        assert np.isclose(sum(r['seconds'] for r in chunks), row['seconds'])
        assert np.isclose(sum(r['gpu_energy']['gpu_energy_j'] for r in chunks), row['gpu_energy_j'])
        assert digest(row['runtime_file']) == row['runtime_sha256']
        with np.load(row['runtime_file'], allow_pickle=False) as data:
            assert all(data[k].shape == (10000,) for k in ('predictions', 'choices', 'order'))
            assert np.array_equal(np.sort(data['order']), np.arange(10000))
            assert np.array_equal(data['order'], np.random.default_rng(43).permutation(10000))
            assert np.all((data['choices'] >= 0) & (data['choices'] < 6))
            assert np.array_equal(np.bincount(data['choices'], minlength=6), row['selected_images'])
            assert np.isclose(100 * (data['predictions'] == labels).mean(), row['accuracy'])
            if row['method'] is None:
                assert np.all(data['choices'] == plan['settings'][case]['index'])
        assert sum(row['selected_images']) == 10000
        assert np.isclose(row['gpu_j_per_image'], row['gpu_energy_j'] / 10000)
        assert np.isclose(row['ms_per_image'], row['seconds'] / 10)
    output = []
    for case, r in rows.items():
        if r['method'] is None:
            continue
        ref = rows['reference']
        static = rows['eligible_' + str(r['calibration_images'])]
        item = dict(case=case, method=r['method'], calibration_images=r['calibration_images'],
                    accuracy=r['accuracy'], accuracy_loss_pp=ref['accuracy']-r['accuracy'],
                    accuracy_pass=ref['accuracy']-r['accuracy'] <= 1 + 1e-10,
                    ms_per_image=r['ms_per_image'], gpu_j_per_image=r['gpu_j_per_image'],
                    distinct_models=sum(n > 0 for n in r['selected_images']),
                    validation_fallback=plan['settings'][case]['policy']['fallback'])
        for label, comparator in [('reference', ref), ('eligible', static)]:
            item['latency_saving_vs_' + label + '_pct'] = 100 * (1-r['ms_per_image']/comparator['ms_per_image'])
            item['gpu_energy_saving_vs_' + label + '_pct'] = 100 * (1-r['gpu_j_per_image']/comparator['gpu_j_per_image'])
        output.append(item)
    summary = dict(audit_passed=True, report_sha256=digest(path), full_sweeps=20, full_chunks=400,
                   preparations=6, results=output,
                   note='Seed42 only, one repeat per predeclared case; no test-based policy selection. Six original resident eager classifiers, GPU-only energy excludes CPU/restPC. Preserve accuracy failures and negative savings. Separate-session timing variation prevents a general causal speedup claim. Fit payback must use the corresponding fit energy and positive savings plus accuracy compliance.')
    atomic_json(path.parent / 'audited_summary.json', summary)
    with (path.parent / 'comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
