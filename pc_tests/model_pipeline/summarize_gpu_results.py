"""Summarize actual TensorRT GPU measurements, including failures and partial runs."""
import argparse
import json
from pathlib import Path
import numpy as np


def comparison_rows(report):
    completed = [row for row in report['results'] if row['status'] == 'complete']
    baselines = {row['architecture']: row for row in completed if row['prune'] == 0 and row['precision'] == 'fp32'}
    rows = []
    for record in completed:
        baseline = baselines.get(record['architecture'])
        for size, repeats in record['timings'].items():
            row = {key: record[key] for key in ('name', 'architecture', 'prune', 'precision', 'test_accuracy', 'validation_accuracy')}
            row.update(batch_size=int(size), engine_mib=record['engine_bytes']/1024**2,
                       integer_compute_layer_count=len(record['integer_compute_layers']))
            for label, key in [('forward_ms', 'forward_batch_ms'),
                               ('host_to_logits_ms', 'normalized_host_input_to_logits_batch_ms'),
                               ('gpu_j_per_image', 'gpu_energy_j_per_image')]:
                values = [repeat.get(key) for repeat in repeats]
                row[label] = float(np.median(values)) if all(value is not None for value in values) else None
                row[label + '_repeat_min'] = float(min(values)) if row[label] is not None else None
                row[label + '_repeat_max'] = float(max(values)) if row[label] is not None else None
                if baseline and size in baseline['timings']:
                    base_values = [repeat.get(key) for repeat in baseline['timings'][size]]
                    base = float(np.median(base_values)) if all(value is not None for value in base_values) else None
                    row[label + '_saving_vs_p0_percent'] = 100*(base-row[label])/base if base and row[label] is not None else None
                else:
                    row[label + '_saving_vs_p0_percent'] = None
            row['accuracy_loss_vs_p0_pp'] = baseline['test_accuracy']-record['test_accuracy'] if baseline else None
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report = json.loads((args.report_dir / 'gpu_static.json').read_text())
    completed = [row for row in report['results'] if row['status'] == 'complete']
    failures = [row for row in report['results'] if row['status'] == 'failed']
    if not args.allow_partial and (len(completed) != 213 or failures):
        raise RuntimeError('Full GPU report requires 213 successful targets')
    rows = comparison_rows(report)
    (args.report_dir / 'gpu_static_comparison.json').write_text(json.dumps({'completed': len(completed),
        'expected': 213, 'partial': len(completed) != 213, 'rows': rows}, indent=2)+'\n', encoding='utf-8')
    lines = ['# PC TensorRT static results', '', f'Completed GPU targets: {len(completed)}/213. Recorded execution failures: {len(failures)}.', '',
             'This report is partial.' if len(completed) != 213 else 'The static sweep is complete; this alone does not complete routing experiments.', '',
             'All targets use TensorRT on RTX 2060. Each completed target has 10,000 test predictions and 5,000 validation predictions; timings use five repeats at batches 1, 8, 16, 32, 64 and 128.', '',
             'Forward time is device-resident GPU execution. Host-to-logits time includes transfer of normalized input and host logits, but excludes normalization/decoding/resizing. It is not the complete routed pipeline boundary. GPU energy is measured by the NVML cumulative counter over at least three seconds; it includes idle/display load and excludes the rest of the PC.', '',
             'INT8 engines are mixed-precision deployments of calibrated QDQ graphs. The inspector must identify integer convolution/matrix inputs; neither an INT8 filename nor QDQ nodes alone establish integer execution. Accuracy failures are retained even when execution succeeds.', '',
             '## Batch 32 snapshot', '',
             '| Model | Test accuracy | Forward ms | Host-to-logits ms | GPU J/image | Integer compute layers |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['batch_size'] != 32:
            continue
        energy = 'unavailable' if row['gpu_j_per_image'] is None else f"{row['gpu_j_per_image']:.5f}"
        lines.append(f"| {row['name']} | {row['test_accuracy']:.2f}% | {row['forward_ms']:.3f} | {row['host_to_logits_ms']:.3f} | {energy} | {row['integer_compute_layer_count']} |")
    lines += ['', '## Failures', '']
    lines += [f"- {row['name']}: {row['error']}" for row in failures] or ['No execution failures recorded. This does not mean all targets meet accuracy budgets.']
    lines += ['', 'All measured batch sizes and repeat ranges are in [gpu_static_comparison.json](gpu_static_comparison.json). Comparisons use the same architecture’s measured p0 FP32. Latency and energy remain separate quantities. No Jetson results are used.', '']
    (args.report_dir / 'gpu_static_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Summarized {len(completed)}/213 GPU targets and {len(rows)} batch records')


if __name__ == '__main__':
    main()
