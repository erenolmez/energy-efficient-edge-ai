"""Validation-only batch routing targets and accuracy-constrained policies."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import numpy as np


def split_validation(labels, seed=42):
    """Reserve 40% of each class for policy calibration, never test images."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    fit, calibration = [], []
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        rng.shuffle(positions)
        boundary = int(len(positions) * .6)
        fit.extend(positions[:boundary])
        calibration.extend(positions[boundary:])
    return np.asarray(fit), np.asarray(calibration)


def make_batches(positions, batch_size, seed, shuffles=1):
    if batch_size < 1 or shuffles < 1:
        raise ValueError('Positive batch size and shuffle count required')
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(shuffles):
        order = np.array(positions, copy=True)
        rng.shuffle(order)
        batches.extend(order[start:start + batch_size] for start in range(0, len(order), batch_size))
    return batches


def batch_targets(batches, correct, costs, baseline_index):
    """Average cheapest-correct rank: supervise the whole batch directly."""
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(costs, kind='stable')
    baseline_rank = int(np.flatnonzero(order == baseline_index)[0])
    any_correct = correct[:, order].any(1)
    ranks = correct[:, order].argmax(1)
    ranks[~any_correct] = baseline_rank
    targets = np.asarray([ranks[batch].mean() / max(1, len(order) - 1) for batch in batches])
    counts = np.asarray([correct[batch].sum(0) for batch in batches])
    lengths = np.asarray([len(batch) for batch in batches])
    return targets, counts, lengths


def select(scores, policy):
    scores = np.asarray(scores)
    if policy['fallback']:
        return np.full(len(scores), policy['baseline_index'], dtype=int)
    ranks = np.clip(np.floor((scores + policy['bias']) * len(policy['order'])),
                    0, len(policy['order']) - 1).astype(int)
    return np.asarray(policy['order'])[ranks]


def calibrate(scores, counts, lengths, costs, baseline_index, budget_pp):
    """Choose a threshold bias using calibration labels and measured batch costs.

    Costs are per complete batch, not per image. Partial tails retain their actual
    accuracy weight; end-to-end evaluation separately measures their real cost.
    """
    scores, counts = np.asarray(scores), np.asarray(counts)
    lengths, costs = np.asarray(lengths), np.asarray(costs, dtype=float)
    if budget_pp < 0 or not np.isfinite(scores).all() or not np.isfinite(costs).all():
        raise ValueError('Finite scores/costs and nonnegative accuracy budget required')
    if counts.shape != (len(scores), len(costs)) or len(lengths) != len(scores) or lengths.sum() <= 0:
        raise ValueError('Inconsistent calibration arrays')
    baseline_accuracy = 100 * counts[:, baseline_index].sum() / lengths.sum()
    policy = {'fallback': True, 'baseline_index': int(baseline_index),
              'order': np.argsort(costs, kind='stable').tolist(), 'bias': 0.,
              'budget_pp': float(budget_pp), 'calibration_accuracy': float(baseline_accuracy),
              'estimated_batch_cost': float(costs[baseline_index])}
    for bias in np.linspace(-1, 1, 1601):
        candidate = {**policy, 'fallback': False, 'bias': float(bias)}
        choices = select(scores, candidate)
        accuracy = 100 * counts[np.arange(len(scores)), choices].sum() / lengths.sum()
        cost = float(costs[choices].mean())
        if accuracy + 1e-10 >= baseline_accuracy - budget_pp and cost < policy['estimated_batch_cost']:
            policy = {**candidate, 'calibration_accuracy': float(accuracy), 'estimated_batch_cost': cost}
    return policy
