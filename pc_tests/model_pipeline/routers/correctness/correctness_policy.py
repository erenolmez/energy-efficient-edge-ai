"""Model-specific probabilities; no scalar rank or test-label policy tuning."""
# Allow both package imports and direct invocation of this organized entry point.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path as pipeline_source

import numpy as np


def select_correctness(probabilities, costs, penalty):
    q, costs = np.asarray(probabilities), np.asarray(costs)
    if q.ndim != 2 or q.shape[1] != len(costs) or not np.isfinite(q).all():
        raise ValueError('Finite batch-by-candidate probabilities required')
    if (q < 0).any() or (q > 1).any() or not np.isfinite(costs).all() or (costs <= 0).any() or not np.isfinite(penalty) or penalty < 0:
        raise ValueError('Invalid probabilities, costs or penalty')
    return (q - penalty * costs[None, :] / costs.max()).argmax(1)


def calibrate_correctness(probabilities, counts, lengths, costs, baseline, budget_pp):
    counts, lengths, costs = np.asarray(counts), np.asarray(lengths), np.asarray(costs)
    q = np.asarray(probabilities)
    if q.shape != counts.shape or len(lengths) != len(q) or not len(q) or not np.isfinite(lengths).all() or (lengths <= 0).any() or not np.isfinite(budget_pp) or budget_pp < 0:
        raise ValueError('Inconsistent calibration data')
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts > lengths[:, None]).any():
        raise ValueError('Invalid correctness counts')
    minimum = 100 * counts[:, baseline].sum() / lengths.sum() - budget_pp
    best = {'fallback': True, 'baseline_index': int(baseline), 'penalty': 0.,
            'estimated_batch_cost': float(costs[baseline]), 'budget_pp': float(budget_pp)}
    for penalty in np.r_[0., np.geomspace(.0001, 100., 401)]:
        choices = select_correctness(q, costs, penalty)
        accuracy = 100 * counts[np.arange(len(q)), choices].sum() / lengths.sum()
        cost = float(costs[choices].mean())
        if accuracy + 1e-10 >= minimum and cost < best['estimated_batch_cost']:
            best = dict(fallback=False, baseline_index=int(baseline), penalty=float(penalty),
                        estimated_batch_cost=cost, calibration_accuracy=float(accuracy), budget_pp=float(budget_pp))
    return best
