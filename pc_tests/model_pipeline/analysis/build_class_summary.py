"""Audit saved class counts and build the November 6 report tables."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HERE = REPO / 'progress_reports/2026-11-06'
PC = REPO / 'pc_tests/model_pipeline/reports/fresh_prune_adam_bank_20261105'
BANK = Path.home() / '.codex/finetune-handoff/imported'
ARCH = dict(resnet18='ResNet18', resnet34='ResNet34', resnet50='ResNet50',
            resnet101='ResNet101', mobilenetv3_small='MobileNetV3 Small',
            mobilenetv3_large='MobileNetV3 Large', efficientnet_b0='EfficientNet-B0')
LEVELS = range(10, 100, 10)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    if sys.argv[1:] == ['--test']:
        build_test_summary()
        return
    state = read(PC / 'campaign.json')
    labels = state['class_names']
    assert len(labels) == len(set(labels)) == 100
    local = {r['id']: r for r in state['completed']}
    plan = read(PC / 'plan.json')
    split = plan['split_sha256']
    assert digest(PC / 'plan.json') == state['plan_sha256']
    with (BANK / 'model_index.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 63 and {r['case'] for r in rows} == {
        f'{a}__p{p}' for a in ARCH for p in LEVELS}
    models = {}
    for row in rows:
        name = row['case']
        folder = BANK / name
        if row['owner'] == 'pc-original':
            proof = local[name]
            report_path = Path(proof['report'])
            checkpoint_hash = proof['best']['sha256']
        else:
            proof = read(folder / 'receipt.json')
            assert (proof['id'], proof['owner']) == (name, row['owner'])
            checkpoint_hash = proof['checkpoint_sha256']
            report_path = folder / 'study.json'
        assert digest(report_path) == proof['report_sha256'], name
        assert (folder / 'best.pth').resolve() == Path(row['checkpoint']).resolve()
        assert digest(folder / 'best.pth') == checkpoint_hash, name
        report = read(report_path)
        assert report['status'] == 'complete' and report['split_sha256'] == split, name
        assert report['best']['sha256'] == checkpoint_hash
        best = max(report['rows'], key=lambda r: r['validation']['accuracy'])
        assert best['epoch'] == report['best']['epoch']
        cm = best['validation']['confusion']
        assert len(cm) == 100 and all(len(r) == 100 and sum(r) == 50 and
                                      all(type(x) is int and x >= 0 for x in r) for r in cm)
        counts = [r[i] for i, r in enumerate(cm)]
        overall = sum(counts) / 50
        assert overall == float(row['best_validation_accuracy']) == best['validation']['accuracy']
        models[name] = dict(accuracy=overall, epoch=best['epoch'], correct_by_class=counts,
                            report_sha256=proof['report_sha256'], checkpoint_sha256=checkpoint_hash)

    render(models, labels, examples_per_class=50, output=HERE / 'class_summary.tex')
    payload = dict(split_sha256=split, examples_per_class=50, class_names=labels, models=models)
    (HERE / 'class_validation_audit.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print(f'Audited {len(models)} models; owners: {dict(Counter(r["owner"] for r in rows))}')


def build_test_summary():
    with (HERE / 'class_test_run.jsonl').open(encoding='utf-8') as f:
        provenance, *rows = map(json.loads, f)
    assert provenance['dataset'] == 'CIFAR-100 official test'
    assert provenance['images'] == 10000 and provenance['examples_per_class'] == 100
    prior = read(HERE / 'class_validation_audit.json')
    assert provenance['class_names'] == prior['class_names']
    models = {r['name']: r for r in rows}
    assert len(rows) == len(models) == 63 and set(models) == set(prior['models'])
    for name, result in models.items():
        counts = result['correct_by_class']
        assert len(counts) == 100 and all(type(n) is int and 0 <= n <= 100 for n in counts)
        assert result['accuracy'] == sum(counts) / 100
        assert result['checkpoint_sha256'] == prior['models'][name]['checkpoint_sha256']
    representative = {a: max(LEVELS, key=lambda p: prior['models'][f'{a}__p{p}']['accuracy']) for a in ARCH}
    assert set(representative.values()) == {10}
    render(models, provenance['class_names'], examples_per_class=100,
           output=HERE / 'class_test_summary.tex', representative=representative)
    print(f'Validated all {len(models)} official-test records')


def render(models, labels, *, examples_per_class, output, representative=None):
    scale = 100 // examples_per_class

    def label(i):
        return labels[i].replace('_', ' ')

    def extreme(values, fn):
        n = fn(values)
        names = [label(i) for i, v in enumerate(values) if v == n]
        note = f' (+{len(names)-2} ties)' if len(names) > 2 else ''
        return f'{" / ".join(names[:2])}{note} ({scale*n}\\%)'

    def values(a, p):
        return models[f'{a}__p{p}']['correct_by_class']

    lines = [r'\begin{center}\small',
             r'\begin{tabularx}{\linewidth}{@{}lrrXX@{}}\toprule',
             r'Architecture & Pruning & Overall & Worst class & Best class(es) \\ \midrule']
    for a, title in ARCH.items():
        level = representative[a] if representative else max(LEVELS, key=lambda p: models[f'{a}__p{p}']['accuracy'])
        v = values(a, level)
        acc = models[f'{a}__p{level}']['accuracy']
        lines.append(f'{title} & {level}\\% & {acc:.2f}\\% & {extreme(v,min)} & {extreme(v,max)} \\\\')
    lines += [r'\bottomrule\end{tabularx}', r'\end{center}',
              r'\textbf{How does the weakest class change as pruning increases?}',
              r'\begin{center}\small\setlength{\tabcolsep}{4.2pt}',
              r'\begin{tabular}{@{}lrrrrrrrrr@{}}\toprule',
              'Requested pruning (\\%) & ' + ' & '.join(map(str, LEVELS)) + r' \\ \midrule']
    for a, title in ARCH.items():
        worst = [scale*min(values(a,p)) for p in LEVELS]
        cells = [rf'\cellcolor{{teal!{int(n*0.65)}}}{n}' for n in worst]
        lines.append(title + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule\end{tabular}', r'\end{center}',
              r'\smallcap{Each cell is the lowest class accuracy (\%) at that pruning level; darker means higher accuracy. The weakest class can change from cell to cell.}']
    if examples_per_class == 100:
        lines += [r'\textbf{How does the strongest class change as pruning increases?}',
                  r'\begin{center}\small\setlength{\tabcolsep}{4.2pt}',
                  r'\begin{tabular}{@{}lrrrrrrrrr@{}}\toprule',
                  'Requested pruning (\\%) & ' + ' & '.join(map(str, LEVELS)) + r' \\ \midrule']
        for a, title in ARCH.items():
            best = [max(values(a, p)) for p in LEVELS]
            cells = [rf'\cellcolor{{teal!{int(n*0.65)}}}{n}' for n in best]
            lines.append(title + ' & ' + ' & '.join(cells) + r' \\')
        lines += [r'\bottomrule\end{tabular}', r'\end{center}',
                  r'\smallcap{Each cell is the highest class accuracy (\%) at that pruning level, using the same color scale as above. The strongest class can change from cell to cell.}']

    # Compare architectures at matched requested pruning; avoid mistaking
    # model-count imbalance or a different pruning dose for class specialization.
    examples = []
    for p in (10, 50, 90):
        columns = list(zip(*(values(a, p) for a in ARCH)))
        spans = [max(col)-min(col) for col in columns]
        i = max(range(100), key=spans.__getitem__)
        c = columns[i]
        lo = '/'.join(title for title, n in zip(ARCH.values(),c) if n == min(c))
        hi = '/'.join(title for title, n in zip(ARCH.values(),c) if n == max(c))
        examples.append(f'{p}\\% & {label(i)} & {hi} ({scale*max(c)}\\%) & {lo} ({scale*min(c)}\\%) \\\\')
    lines += [r'\textbf{Where do models disagree most on class accuracy?} At each selected pruning level, this is the class with the largest accuracy gap between models.',
              r'\begin{center}\small\begin{tabularx}{\linewidth}{@{}rlXX@{}}\toprule',
              r'Pruning & Class & Highest accuracy & Lowest accuracy \\ \midrule', *examples,
              r'\bottomrule\end{tabularx}\end{center}']
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
