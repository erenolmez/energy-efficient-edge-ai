"""Auditable batch128 random-control tables, with partial status explicit."""
import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    folder = root / 'reports/random_b128'
    report = json.loads((folder / 'random_routes_b128.json').read_text())
    cached = json.loads((folder / 'cached_accuracy.json').read_text())
    source = json.loads((root / 'reports/pc_routes_correctness.json').read_text())
    cases = sorted([r for r in source['results'] if r['batch_size'] == 128],
                   key=lambda r: (r['objective'], r['budget_pp'], r['seed']))
    cpu_lines = ['# Batch128 random-selection accuracy (cached predictions)', '',
                 'Descriptive accuracy only; no timing inferred from cached model costs. Ten prespecified '
                 'random assignments per source case. Matched random preserves model counts and the '
                 '16-image tail choice; full batches are reassigned without image/label information. '
                 'Uniform selects equally over all213, including failed-accuracy quantization variants. '
                 'Uniform draws are shared across cases with the same batch-order seed; repeated cells '
                 'are not independent trials. Learned accuracy below is the original frozen report, not a new measurement.', '',
                 '| Objective | Seed | Budget pp | Original learned accuracy | Matched random mean (range) | Uniform random mean (range) |',
                 '|---|---:|---:|---:|---|---|']
    for case in cases:
        stats = []
        for role in ('matched','uniform'):
            a = [r['accuracy'] for r in cached['trials'] if r['case'] == case['id'] and r['role'] == role]
            assert len(a) == 10
            stats.append(f'{np.mean(a):.2f}% ({min(a):.2f}–{max(a):.2f}%)')
        old = np.mean([r['test_accuracy'] for r in case['measurements']])
        cpu_lines.append(f"| {case['objective']} | {case['seed']} | {case['budget_pp']:g} | {old:.2f}% | " + ' | '.join(stats) + ' |')
    (folder / 'cached_accuracy_table.md').write_text('\n'.join(cpu_lines)+'\n', encoding='utf-8')
    results = report['results']
    complete = len(results) == report['expected_blocks'] and report.get('status') == 'complete'
    lines = ['# Batch128 measured random-routing comparison', '',
             f"Status: {'COMPLETE' if complete else 'PARTIAL'}; {len(results)}/{report['expected_blocks']} saved full-test blocks.", '',
             'All timings are fresh, same run and input boundary, capacity2 cold cache, all213 logically '
             'selectable; random controls do not execute the neural router. Schedule generation is offline; '
             'dispatch lookup, transfer/normalization, hash/load/eviction/inference/sync are measured. '
             'Static/learned/matched order alternates by trial. Five trials, each10000 images. '
             'Latency and GPU energy are medians over trials; accuracy is the trial mean. SD is over '
             'five randomized assignments, not independent test datasets. Energy is NVML GPU-only. '
             'Loss is versus the freshly measured R101 p0 FP32 reference; L/E savings are versus '
             'the original validation-selected eligible static identity, freshly measured for each case. '
             'Uniform has no budget or objective, and its measurements are reused across references.', '',
             '| Objective | Seed | Budget pp | Policy | Accuracy mean ± SD | Loss pp | Accuracy pass (trials/5) | ms/image | L saving % | mJ/image | E saving % |',
             '|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|']
    summaries = []
    for case in cases:
        grouped = {}
        for role in ('reference','static','learned','matched','uniform'):
            grouped[role] = [r for r in results if r['router'] == role and
                             (r['seed'] == case['seed'] if role == 'uniform' else r['source_case'] == case['id'])]
        if not all(len(v) == 5 for v in grouped.values()):
            continue
        reference = np.mean([r['accuracy'] for r in grouped['reference']])
        base_l = np.median([r['latency_ms_per_image'] for r in grouped['static']])
        base_e = np.median([r['gpu_energy_j_per_image'] for r in grouped['static']])
        for role in ('learned','matched','uniform'):
            rows = grouped[role]
            accuracy = np.array([r['accuracy'] for r in rows])
            latency = float(np.median([r['latency_ms_per_image'] for r in rows]))
            energy = float(np.median([r['gpu_energy_j_per_image'] for r in rows]))
            loss = float(reference - accuracy.mean())
            passed = int(np.sum(reference - accuracy <= case['budget_pp'] + 1e-10))
            item = dict(case=case['id'], policy=role, objective=case['objective'], seed=case['seed'],
                        budget_pp=case['budget_pp'], accuracy=float(accuracy.mean()),
                        accuracy_sd=float(accuracy.std(ddof=1)), accuracy_loss_pp=loss,
                        mean_accuracy_pass=loss <= case['budget_pp']+1e-10, passing_trials=passed,
                        median_latency_ms_per_image=latency, median_energy_mj_per_image=1000*energy,
                        latency_saving_percent=float(100*(base_l-latency)/base_l),
                        energy_saving_percent=float(100*(base_e-energy)/base_e),
                        eligible_static=case['best_eligible_static'],
                        prediction_disagreements=sum(r['cached_prediction_disagreements'] for r in rows),
                        learned_counts_match=all(r.get('selection_counts_match_original', True) for r in rows))
            summaries.append(item)
            lines.append(f"| {case['objective']} | {case['seed']} | {case['budget_pp']:g} | {role} | "
                         f"{item['accuracy']:.2f} ± {item['accuracy_sd']:.2f}% | {loss:.2f} | "
                         f"{'pass' if item['mean_accuracy_pass'] else 'FAIL'} ({passed}/5) | {latency:.3f} | "
                         f"{item['latency_saving_percent']:+.2f} | {1000*energy:.3f} | {item['energy_saving_percent']:+.2f} |")
    (folder / 'random_comparison_table.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (folder / 'random_comparison_summary.json').write_text(json.dumps(dict(complete=complete, rows=summaries,
        completed_blocks=len(results), expected_blocks=report['expected_blocks']),indent=2)+'\n', encoding='utf-8')
    print(f'Summary: {len(results)}/{report["expected_blocks"]} blocks, {len(summaries)//3}/24 fully paired cases; complete={complete}')


if __name__ == '__main__':
    main()
