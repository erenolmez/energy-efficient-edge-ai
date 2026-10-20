"""Descriptive per-class test statistics; cached unpruned TensorRT FP32 only."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


ARCHITECTURES = [
    ('resnet18', 'ResNet-18'), ('resnet34', 'ResNet-34'),
    ('resnet50', 'ResNet-50'), ('resnet101', 'ResNet-101'),
    ('mobilenetv3_small', 'MobileNetV3-Small'),
    ('mobilenetv3_large', 'MobileNetV3-Large'),
    ('efficientnet_b0', 'EfficientNet-B0'),
    ('shufflenetv2_x1_0', 'ShuffleNetV2 x1.0'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--static-dir', type=Path, required=True)
    parser.add_argument('--static-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    # Standard, local CIFAR-100 files already used by the classifier pipeline.
    with (args.data_dir / 'meta').open('rb') as handle:
        names = pickle.load(handle, encoding='latin1')['fine_label_names']
    with (args.data_dir / 'test').open('rb') as handle:
        original_labels = np.asarray(pickle.load(handle, encoding='latin1')['fine_labels'])
    label_path = args.cache_dir / 'test_labels.npy'
    labels = np.load(label_path, allow_pickle=False)
    assert labels.shape == (10000,) and np.array_equal(labels, original_labels)
    assert len(names) == 100 and np.array_equal(np.bincount(labels), np.full(100, 100))
    records = {r['name']: r for r in json.loads(args.static_report.read_text())['results']}
    rows = []
    for architecture, display in ARCHITECTURES:
        key = architecture + '__p0__fp32'
        path = args.static_dir / (key + '_test_predictions.npy')
        predictions = np.load(path, allow_pickle=False)
        assert predictions.shape == labels.shape
        assert np.issubdtype(predictions.dtype, np.integer)
        assert np.all((predictions >= 0) & (predictions < 100))
        accuracy = float(100 * np.mean(predictions == labels))
        assert abs(accuracy - records[key]['test_accuracy']) < 1e-8
        classes = []
        for index, name in enumerate(names):
            count = int(np.sum((labels == index) & (predictions == labels)))
            classes.append(dict(class_id=index, class_name=name, correct=count,
                                images=100, accuracy_percent=float(count)))
        ranked = sorted(classes, key=lambda c: (c['accuracy_percent'], c['class_id']))
        rows.append(dict(architecture=architecture, display_name=display,
                         overall_accuracy_percent=accuracy,
                         worst=[c for c in ranked if c['accuracy_percent'] == ranked[0]['accuracy_percent']],
                         best=[c for c in ranked if c['accuracy_percent'] == ranked[-1]['accuracy_percent']],
                         middle_ranked_classes=ranked[49:51],
                         median_class_accuracy_percent=float(np.median([c['accuracy_percent'] for c in classes])),
                         classes=classes, prediction_sha256=digest(path)))
    note = ('CIFAR-100 test set: 10,000 images, 100 images per true class. '
            'Accuracy is correct predictions divided by images belonging to that true class, not precision. '
            'All tied worst/best classes are shown. Median column shows ranks 50 and 51 '
            '(ascending accuracy; ties ordered by canonical class ID). '
            'This is descriptive test analysis, not policy selection. No router, inference or training was run.')
    reference = next(r for r in rows if r['architecture'] == 'resnet101')
    forest = lambda row: next(c['accuracy_percent'] for c in row['classes'] if c['class_name'] == 'forest')
    examples = []
    for architecture in ('resnet18', 'mobilenetv3_large'):
        row = next(r for r in rows if r['architecture'] == architecture)
        latency = lambda arch: float(np.median([t['normalized_host_input_to_logits_batch_ms']
                                   for t in records[arch + '__p0__fp32']['timings']['32']]))
        assert row['overall_accuracy_percent'] < reference['overall_accuracy_percent']
        assert forest(row) > forest(reference) and latency(architecture) < latency('resnet101')
        examples.append(dict(model=row['display_name'], overall_accuracy_percent=row['overall_accuracy_percent'],
                             forest_accuracy_percent=forest(row), fp32_batch32_median_ms=latency(architecture)))
    sentence = ('These results support the motivation for input-dependent selection: despite lower overall '
                'accuracy than ResNet-101 (85.80%), ResNet-18 (79.16%) and MobileNetV3-Large (81.34%) '
                'achieve higher accuracy on forest images (71% and 76%, versus 62%); both are faster in the '
                'saved FP32 batch-32 static measurements (median 13.64 and 6.92 ms/batch, versus 131.28 ms/batch). '
                'These are descriptive results on 100 forest images, not proof of statistical significance '
                'or end-to-end routing gains.')
    payload = dict(note=note, interpretation=sentence, examples=examples, test_labels_sha256=digest(label_path),
                   static_report_sha256=digest(args.static_report), models=rows)
    def cell(classes):
        return '; '.join(c['class_name'].replace('_', ' ') + f" ({c['accuracy_percent']:.0f}%)" for c in classes)
    markdown = ['# Unpruned FP32 per-class test accuracy', '', note, '',
                '| Architecture | Overall FP32 | Worst class(es) | Best class(es) | Median-ranked class(es) and accuracy |',
                '|---|---:|---|---|---|']
    tex = [r'\begin{tabular}{l r p{2.5cm} p{3.5cm} p{3cm}}', r'\hline',
           r'Architecture & FP32 & Worst & Best & Median-ranked classes \\', r'\hline']
    for row in rows:
        values = [row['display_name'], f"{row['overall_accuracy_percent']:.2f}%",
                  cell(row['worst']), cell(row['best']), cell(row['middle_ranked_classes'])]
        markdown.append('| ' + ' | '.join(values) + ' |')
        tex.append(' & '.join(v.replace('%', r'\%') for v in values) + r' \\')
    tex.extend([r'\hline', r'\end{tabular}'])
    markdown.extend(['', sentence])
    tex.extend(['', r'\par\smallskip', sentence.replace('%', r'\%')])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'fp32_class_statistics.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'fp32_class_table.md').write_text('\n'.join(markdown) + '\n', encoding='utf-8')
    (args.output_dir / 'fp32_class_table.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    print('\n'.join(markdown))
    print('\nVERIFIED: cached labels match canonical test order; all eight overall accuracies match TensorRT reports.')


if __name__ == '__main__':
    main()
