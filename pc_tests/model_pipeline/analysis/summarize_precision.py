"""Summarize measured PC precision results without rerunning inference."""
from __future__ import annotations
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source


import argparse
import csv
import html
import json
from pathlib import Path


def comparison_rows(report):
    completed = [r for r in report['results'] if r['status'] == 'complete']
    fp32 = {(r['architecture'], r['prune']): r for r in completed if r['precision'] == 'fp32'}
    rows = []
    for result in completed:
        arch, prune = result['architecture'], result['prune']
        if (arch, 0) not in fp32 or (arch, prune) not in fp32:
            continue
        rows.append({
            'architecture': arch, 'prune': prune, 'precision': result['precision'],
            'test_accuracy': result['test_accuracy'],
            'p0_fp32_accuracy': fp32[arch, 0]['test_accuracy'],
            'loss_vs_p0_pp': fp32[arch, 0]['test_accuracy'] - result['test_accuracy'],
            'precision_loss_pp': fp32[arch, prune]['test_accuracy'] - result['test_accuracy'],
            'onnx_mib': result['onnx_bytes'] / 1024**2,
            'test_images': result['test_images'], 'runtime': result['runtime'],
            'evaluation_seconds': result['evaluation_seconds'],
        })
    return sorted(rows, key=lambda r: (r['architecture'], r['prune'], r['precision']))


def plot(rows, metric, title, destination):
    architectures = sorted({r['architecture'] for r in rows})
    colors = {'fp32': '#2364aa', 'fp16': '#df7a15', 'int8': '#408340'}
    height = ((len(architectures) + 1) // 2) * 255 + 60
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<g font-family="Arial,sans-serif" font-size="12" fill="#222">',
                f'<text x="25" y="24" font-size="18">{html.escape(title)}</text>']
    for i, architecture in enumerate(architectures):
        group = [r for r in rows if r['architecture'] == architecture]
        left, top = (i % 2) * 480 + 58, (i // 2) * 255 + 80
        maximum = 100 if metric == 'test_accuracy' else max(r[metric] for r in group) * 1.1
        maximum = max(maximum, 1)
        elements.append(f'<text x="{left}" y="{top-20}" font-size="14">{architecture}</text>')
        for fraction in (0, .25, .5, .75, 1):
            y = top + 155 * (1 - fraction)
            elements.extend([f'<path d="M{left},{y}h370" stroke="#ddd"/>',
                             f'<text x="{left-8}" y="{y+4}" text-anchor="end">{maximum*fraction:.1f}</text>'])
        for prune in range(0, 100, 10):
            elements.append(f'<text x="{left+prune/90*370}" y="{top+175}" text-anchor="middle">{prune}</text>')
        elements.append(f'<text x="{left+185}" y="{top+193}" text-anchor="middle">Requested pruning (%)</text>')
        for precision, color in colors.items():
            series = sorted([r for r in group if r['precision'] == precision], key=lambda r: r['prune'])
            points = [(left+r['prune']/90*370, top+155*(1-r[metric]/maximum)) for r in series]
            elements.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="2"/>')
            elements.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>' for x,y in points)
    for i, (precision, color) in enumerate(colors.items()):
        elements.append(f'<text x="{600+i*105}" y="25" fill="{color}">{precision.upper()}</text>')
    elements.append('</g></svg>')
    destination.write_text('\n'.join(elements), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report = json.loads((args.report_dir / 'precision_accuracy.json').read_text())
    completed = [r for r in report['results'] if r['status'] == 'complete']
    if not args.allow_partial and (len(completed) != 213 or any(r['test_images'] != 10000 for r in completed)):
        raise ValueError('Full report requires all 213 model/precision evaluations on 10000 images')
    rows = comparison_rows(report)
    with (args.report_dir / 'precision_accuracy.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, 'test_accuracy', 'CIFAR-100 test accuracy (%)', args.report_dir / 'precision_accuracy.svg')
    plot(rows, 'onnx_mib', 'ONNX graph size (MiB)', args.report_dir / 'precision_size.svg')
    lines = ['# PC precision evaluation', '',
             f"Completed: {len(completed)}/213 evaluations. Each result uses all 10,000 CIFAR-100 test images.", '',
             'INT8 uses 1,024 fixed training images for MinMax calibration, with symmetric signed activations and per-channel weights. Conv/Gemm/MatMul paths use QDQ; other operators remain floating point.', '',
             'FP32 and INT8 accuracy run in ONNX Runtime on CPU; FP16 accuracy runs in PyTorch on RTX 2060. Runtime differences mean evaluation duration is not a fair precision speed comparison. These are not TensorRT engines or Jetson power measurements.', '',
             'All losses below use the freshly measured FP32 p0 of the same architecture. Precision loss isolates conversion at the same pruning level. Negative loss means higher accuracy.', '',
             '## Unpruned models', '',
             '| Architecture | FP32 | FP16 | INT8 |', '|---|---:|---:|---:|']
    for arch in sorted({r['architecture'] for r in rows}):
        values = {r['precision']: r['test_accuracy'] for r in rows if r['architecture'] == arch and r['prune'] == 0}
        lines.append('| ' + arch + ' | ' + ' | '.join(f"{values[p]:.2f}%" if p in values else 'pending' for p in ('fp32','fp16','int8')) + ' |')
    lines.extend(['', '## Conversion sensitivity', ''])
    for precision in ('fp16', 'int8'):
        subset = [r for r in rows if r['precision'] == precision]
        if subset:
            worst = max(subset, key=lambda r: r['precision_loss_pp'])
            mean = sum(r['precision_loss_pp'] for r in subset) / len(subset)
            lines.append(f"{precision.upper()}: mean accuracy loss at the same pruning level is {mean:.2f} points; largest loss is {worst['precision_loss_pp']:.2f} points for {worst['architecture']} p{worst['prune']}.")
            lines.append('')
    lines.extend(['Severe INT8 losses indicate that this fixed post-training calibration policy is not deployment-ready for those models. Calibration alternatives should be selected on validation data, not this test set. Quantization-aware training and longer pruning recovery are possible follow-up experiments, not results of this run.', '',
                  '## Complete comparison', '',
                  '| Architecture | Pruning | Precision | Accuracy | Loss vs p0 (pp) | Precision loss (pp) | ONNX MiB |',
                  '|---|---:|---|---:|---:|---:|---:|'])
    for r in rows:
        lines.append(f"| {r['architecture']} | p{r['prune']} | {r['precision']} | {r['test_accuracy']:.2f}% | {r['loss_vs_p0_pp']:.2f} | {r['precision_loss_pp']:.2f} | {r['onnx_mib']:.2f} |")
    lines.extend(['', '[Accuracy figure](precision_accuracy.svg) · [Size figure](precision_size.svg) · [CSV](precision_accuracy.csv)', ''])
    (args.report_dir / 'precision_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote comparison tables and figures for {len(rows)} results')


if __name__ == '__main__':
    main()
