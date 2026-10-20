"""One row per architecture/pruning pair; cached TensorRT FP32/FP16/INT8."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from pc_tests.model_pipeline.analysis.summarize_fp32_classes import ARCHITECTURES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'cache-dir', 'static-dir', 'static-report', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    with (args.data_dir / 'meta').open('rb') as stream:
        names = pickle.load(stream, encoding='latin1')['fine_label_names']
    with (args.data_dir / 'test').open('rb') as stream:
        canonical = np.asarray(pickle.load(stream, encoding='latin1')['fine_labels'])
    labels_path = args.cache_dir / 'test_labels.npy'
    labels = np.load(labels_path, allow_pickle=False)
    assert labels.shape == (10000,) and np.array_equal(labels, canonical)
    assert len(names) == 100 and np.array_equal(np.bincount(labels), np.full(100, 100))
    records = json.loads(args.static_report.read_text())['results']
    by_name = {r['name']: r for r in records}
    assert len(by_name) == len(records) == 213
    precisions = ('fp32', 'fp16', 'int8')
    groups = {(r['architecture'], r['prune']) for r in records}
    assert len(groups) == 71
    display = dict(ARCHITECTURES)
    order = {arch: i for i, (arch, _) in enumerate(ARCHITECTURES)}
    rows = []
    for arch, prune in sorted(groups, key=lambda x: (order[x[0]], x[1])):
        row = dict(architecture=arch, display_name=display[arch], prune=prune, precisions={})
        for precision in precisions:
            name = f'{arch}__p{prune}__{precision}'
            record = by_name[name]
            path = args.static_dir / (name + '_test_predictions.npy')
            predictions = np.load(path, allow_pickle=False)
            assert predictions.shape == labels.shape and np.issubdtype(predictions.dtype, np.integer)
            assert np.all((predictions >= 0) & (predictions < 100))
            correct = np.bincount(labels[predictions == labels], minlength=100)
            accuracy = float(correct.sum() / 100)
            assert abs(accuracy - record['test_accuracy']) < 1e-8, name
            classes = [dict(class_id=i, class_name=names[i], correct=int(c), images=100,
                            accuracy_percent=float(c)) for i, c in enumerate(correct)]
            ranked = sorted(classes, key=lambda c: (c['accuracy_percent'], c['class_id']))
            row['precisions'][precision] = dict(name=name, overall_accuracy_percent=accuracy,
                worst=[c for c in ranked if c['correct'] == ranked[0]['correct']],
                best=[c for c in ranked if c['correct'] == ranked[-1]['correct']],
                middle_ranked_classes=ranked[49:51], classes=classes,
                prediction_sha256=digest(path))
        rows.append(row)
    note = ('All 213 actual TensorRT configurations, grouped into 71 architecture/pruning rows. '
            'p0 means unpruned; pN is the stored pruning setting, not a measured parameter/MAC reduction. '
            'Each class has 100 test images. Class accuracy is recall for the true class. '
            'Worst/best ties are all retained, including zero-accuracy classes. Median-ranked classes '
            'are positions 50 and 51 after sorting by accuracy, then canonical class ID. '
            'Within each class-statistics cell, entries are FP32, FP16, INT8 in that order. '
            'Descriptive held-out test statistics only; no router or new inference/training.')
    def cell(classes):
        return '; '.join(c['class_name'].replace('_', ' ') + f" ({c['correct']}%)" for c in classes)
    md = ['# All-model per-class test statistics', '', note, '',
          '| Architecture | Pruning | FP32 accuracy | FP16 accuracy | INT8 accuracy | Worst class(es) | Best class(es) | Median-ranked class(es) |',
          '|---|---|---:|---:|---:|---|---|---|']
    tex = [r'% Standalone table fragment: requires longtable and landscape page layout.',
           r'\begingroup\scriptsize', r'\setlength{\tabcolsep}{3pt}',
           r'\begin{longtable}{p{2.8cm} r r r r p{5.0cm} p{4.2cm} p{4.2cm}}',
           r'\hline', r'Architecture & Pruning & FP32 & FP16 & INT8 & Worst classes & Best classes & Median-ranked classes \\',
           r'\hline\endhead']
    for row in rows:
        p = row['precisions']
        values = [row['display_name'], f"p{row['prune']}"]
        values += [f"{p[precision]['overall_accuracy_percent']:.2f}%" for precision in precisions]
        texvalues = values.copy()
        for key in ('worst', 'best', 'middle_ranked_classes'):
            entries = [precision.upper() + ': ' + cell(p[precision][key]) for precision in precisions]
            values.append('<br>'.join(entries))
            texvalues.append(r'\newline '.join(entries))
        md.append('| ' + ' | '.join(values) + ' |')
        tex.append(' & '.join(v.replace('%', r'\%') for v in texvalues) + r' \\ \hline')
    tex += [r'\end{longtable}', r'\endgroup']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(note=note, configurations=213, table_rows=71,
                   test_labels_sha256=digest(labels_path), static_report_sha256=digest(args.static_report), models=rows)
    for filename, content in [('all_213_class_statistics.json', json.dumps(payload, indent=2)),
                              ('all_213_class_table.md', '\n'.join(md)),
                              ('all_213_class_table.tex', '\n'.join(tex))]:
        (args.output_dir / filename).write_text(content + '\n', encoding='utf-8')
    print('VERIFIED: all 213 cached predictions match recorded accuracies and canonical label order; 71 rows.')
    print('Rows by architecture:', {arch: sum(r['architecture'] == arch for r in rows) for arch in display})
    print('Saved separate all_213 files; original fp32 files untouched.')


if __name__ == '__main__':
    main()
