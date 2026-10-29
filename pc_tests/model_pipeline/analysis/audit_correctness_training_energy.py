"""CPU-only audit of the isolated October 28 fit-energy measurement."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json
import math
from pathlib import Path
import statistics
from pc_tests.model_pipeline.paths import PIPELINE
from pc_tests.model_pipeline.analysis.audit_validation_complementarity import digest
from pc_tests.model_pipeline.routers.correctness.train_correctness_router import atomic_json


def main():
    directory = PIPELINE / 'artifacts/correctness_training_energy_20261028'
    path = directory / 'training_energy.json'
    report = json.loads(path.read_text())
    assert report['status'] == 'complete'
    assert report['test_data_accessed'] is False
    assert [r['seed'] for r in report['results']] == [42,45,48]
    for group in ('source_hashes', 'protected_hashes'):
        for filename, expected in report[group].items():
            assert digest(Path(filename)) == expected, filename
    rows = []
    for result in report['results']:
        seed = result['seed']
        assert result['epochs'] == 20 and len(result['training_loss']) == 20
        assert all(math.isfinite(v) for v in result['training_loss'])
        assert result['fit_images'] == 3000 and result['calibration_images_used'] == 0
        historical = json.loads((PIPELINE / f'artifacts/pc_routers_correctness/global_correctness_seed{seed}.json').read_text())
        assert result['settings'] == historical['settings']
        assert report['provenance'] == historical['provenance']
        assert digest(Path(result['weights'])) == result['weights_sha256']
        for epoch in range(1,21):
            checkpoint = directory / f'global_checkpoint_epoch{epoch}_seed{seed}.json'
            marker = json.loads(checkpoint.read_text())
            assert marker['epoch'] == epoch and marker['status'] == 'complete'
            assert digest(Path(marker['checkpoint'])) == marker['sha256']
        energy = result['energy']
        for key in ('gpu_energy_j', 'energy_window_seconds', 'gpu_average_watts'):
            assert math.isfinite(energy[key]) and energy[key] > 0
        assert energy['energy_window_images'] == 60000
        assert math.isclose(energy['gpu_average_watts'] * energy['energy_window_seconds'], energy['gpu_energy_j'])
        rows.append(dict(seed=seed, joules=energy['gpu_energy_j'], seconds=energy['energy_window_seconds'],
                         final_bce=result['training_loss'][-1],
                         max_loss_difference_from_historical=result['historical_loss_max_abs_difference']))
    summary = dict(audit_passed=True, measurement_sha256=digest(path), rows=rows,
        median_joules_per_fit=statistics.median(r['joules'] for r in rows),
        min_joules_per_fit=min(r['joules'] for r in rows), max_joules_per_fit=max(r['joules'] for r in rows),
        total_three_fit_joules=sum(r['joules'] for r in rows),
        median_seconds_per_fit=statistics.median(r['seconds'] for r in rows),
        original_artifacts_unchanged=True, epoch_checkpoints_verified=60,
        scope=report['scope'], boundary=report['boundary'],
        break_even_computed=False, exact_historical_weight_reproduction_claimed=False)
    output = PIPELINE / 'reports/correctness_training_energy_20261028'
    atomic_json(output / 'audited_summary.json', summary)
    lines = ['# Audited correctness-router training energy', '',
             'New fit-only GPU measurements, not historical energy reconstruction.', '',
             '| Seed | GPU energy (J) | Fit window (s) | Final BCE |',
             '|---|---:|---:|---:|']
    lines += [f"| {r['seed']} | {r['joules']:.3f} | {r['seconds']:.3f} | {r['final_bce']:.6f} |" for r in rows]
    lines += ['', f"Median single-fit energy: {summary['median_joules_per_fit']:.3f} J; range {summary['min_joules_per_fit']:.3f}-{summary['max_joules_per_fit']:.3f} J.",
              f"All three experimental fits together: {summary['total_three_fit_joules']:.3f} J. Do not charge all three to deployment of one seed.", '',
              'All 60 epoch checkpoint hashes, historical artifact hashes, source hashes and fit settings verified. Training losses are close but not bitwise identical to the historical runs.', '',
              report['scope'] + '.', '', report['boundary'], '',
              'These are three seeds in a single session, not repeated energy trials for each seed. Seed 42 also bears first-fit initialization costs. No precise uncertainty or full-preparation claim is made.', '',
              'Break-even remains uncomputed: use total joules per deployed fit divided by positive matched end-to-end GPU joules saved per image at acceptable accuracy. Training-presentation energy is not inference energy.']
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
