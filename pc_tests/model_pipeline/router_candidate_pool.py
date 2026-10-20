"""Bound resident router pools using validation accuracy and measured PC costs."""
import numpy as np


def shortlist(records, maximum=8):
    """Keep p0 FP32, cheap conditional choices, and accuracy-budget choices.

    Static results retain every candidate. This deployment shortlist limits
    simultaneously resident engines on the 6-GB GPU; no test accuracy is read.
    Cheap candidates are retained even if they fail every static accuracy budget.
    """
    if not records or maximum < 4:
        raise ValueError('A nonempty architecture and pool capacity >=4 are required')
    architecture = records[0]['architecture']
    if any(row['architecture'] != architecture for row in records):
        raise ValueError('Candidate pool must belong to one architecture')
    baseline_name = f'{architecture}__p0__fp32'
    baseline = next(row for row in records if row['name'] == baseline_name)
    if any('validation_accuracy' not in row for row in records):
        raise ValueError('Validation accuracy required for candidate selection')
    latency = {}
    for row in records:
        latency[row['name']] = float(np.median([
            repeat['normalized_host_input_to_logits_batch_ms'] / int(size)
            for size, repeats in row['timings'].items() if int(size) > 1
            for repeat in repeats]))
    ordered = sorted(records, key=lambda row: (latency[row['name']], -row['validation_accuracy'], row['name']))
    by_name = {row['name']: row for row in records}
    selected = [baseline_name]

    def add(name):
        if name not in selected and len(selected) < maximum:
            selected.append(name)

    # Reserve two cheap routes regardless of aggregate accuracy; routers may
    # use them conditionally on easy batches.
    for row in ordered[:2]:
        add(row['name'])
    for budget in (.5, 1., 1.5, 2.):
        eligible = [row for row in ordered if row['validation_accuracy'] >= baseline['validation_accuracy'] - budget]
        add(eligible[0]['name'])
        energy_eligible = []
        for row in eligible:
            energy = [repeat.get('gpu_energy_j_per_image') for repeats in row['timings'].values() for repeat in repeats]
            if all(value is not None and value > 0 for value in energy):
                energy_eligible.append((float(np.median(energy)), row['name']))
        if energy_eligible:
            add(min(energy_eligible)[1])
    # Fill remaining slots at intermediate accuracy levels, not just the most
    # accurate or most heavily pruned end of the pool.
    for target in np.linspace(baseline['validation_accuracy'], ordered[0]['validation_accuracy'], maximum):
        row = min(records, key=lambda row: (abs(row['validation_accuracy'] - target), latency[row['name']]))
        add(row['name'])
    for row in ordered:
        add(row['name'])
    return [by_name[name] for name in sorted(selected)]
