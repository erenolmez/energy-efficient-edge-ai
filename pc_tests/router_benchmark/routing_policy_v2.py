"""Calibrate routing utility against the explicit, unpruned p0 baseline.

Predictors may estimate each candidate's probability of correctness or its
correctness difference from p0. Subtracting a common p0 prediction does not
change the decision. Neither model index nor cost is assumed to encode strength.
"""

from numbers import Integral

import numpy as np


LAMBDA_GRID = np.concatenate(([0.0], np.logspace(-4, 1, 301)))


def _validate_inputs(predictions, costs, baseline_index, *, allow_empty=False):
    predictions = np.asarray(predictions, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] == 0:
        raise ValueError("predictions must have shape (samples, models)")
    if not allow_empty and predictions.shape[0] == 0:
        raise ValueError("calibration requires at least one sample")
    if costs.ndim != 1 or costs.shape[0] != predictions.shape[1]:
        raise ValueError("costs must contain one value per prediction column")
    if not np.isfinite(predictions).all():
        raise ValueError("predictions must be finite")
    if not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError("costs must be finite and strictly positive")
    if (isinstance(baseline_index, (bool, np.bool_))
            or not isinstance(baseline_index, Integral)
            or not 0 <= baseline_index < predictions.shape[1]):
        raise ValueError("baseline_index must identify a prediction column")
    return predictions, costs


def _select(predictions, costs, penalty, baseline_index):
    utility = predictions - penalty * (costs / costs[baseline_index])
    # Put the baseline first for exact utility ties, followed by cheaper models.
    # Exact duplicate nonbaseline columns with equal costs remain interchangeable;
    # their final tie is resolved by column index because no model IDs are given.
    order = np.asarray(sorted(range(len(costs)), key=lambda index: (
        index != baseline_index, costs[index], index,
    )))
    return order[np.argmax(utility[:, order], axis=1)]


def calibrate_utility(predictions, correct, costs, baseline_index, max_drop=0.01):
    """Find the cheapest feasible policy on calibration samples only.

    ``predictions`` and ``correct`` have shape (N, M), with one column per
    candidate model. ``correct`` contains bools or binary 0/1 outcomes.
    ``max_drop`` is an absolute accuracy fraction (0.01 = one percentage point).

    Returns a JSON-serializable dict accepted by :func:`select_utility`.
    Feasibility uses measured calibration accuracy, not predicted accuracy.
    This is a calibration constraint, not an accuracy guarantee on unseen data.
    """
    predictions, costs = _validate_inputs(predictions, costs, baseline_index)
    correct = np.asarray(correct)
    if correct.shape != predictions.shape:
        raise ValueError("correct must have the same shape as predictions")
    if not np.isin(correct, [0, 1]).all():
        raise ValueError("correct must contain only bools or binary 0/1 values")
    correct = correct.astype(bool)
    if not np.isscalar(max_drop) or not np.isfinite(max_drop) or not 0 <= max_drop <= 1:
        raise ValueError("max_drop must be a finite fraction between zero and one")

    baseline_accuracy = float(correct[:, baseline_index].mean())
    minimum_accuracy = baseline_accuracy - float(max_drop)
    selected_rows = np.arange(len(predictions))

    # Explicitly include always-p0, even if no lambda would select p0 everywhere.
    best = {
        "lambda": 0.0,
        "fallback": True,
        "minimum_accuracy": minimum_accuracy,
        "calibration_accuracy": baseline_accuracy,
        "cost": float(costs[baseline_index]),
        "baseline_accuracy": baseline_accuracy,
    }
    best_key = (best["cost"], -baseline_accuracy, 0, 0.0)

    for penalty in LAMBDA_GRID:
        choices = _select(predictions, costs, float(penalty), baseline_index)
        accuracy = float(correct[selected_rows, choices].mean())
        if accuracy + 1e-12 < minimum_accuracy:
            continue
        cost = float(costs[choices].mean())
        candidate_key = (cost, -accuracy, 1, float(penalty))
        if candidate_key < best_key:
            best_key = candidate_key
            best = {
                "lambda": float(penalty),
                "fallback": False,
                "minimum_accuracy": minimum_accuracy,
                "calibration_accuracy": accuracy,
                "cost": cost,
                "baseline_accuracy": baseline_accuracy,
            }
    return best


def select_utility(predictions, costs, policy, baseline_index):
    """Apply an already-calibrated policy without observing ground truth.

    Use the same candidate model identities and matching costs as during
    calibration; columns may be reordered when baseline_index is updated.
    Empty sample batches are accepted and return an empty integer array.
    """
    predictions, costs = _validate_inputs(
        predictions, costs, baseline_index, allow_empty=True,
    )
    if not isinstance(policy, dict) or "lambda" not in policy or "fallback" not in policy:
        raise ValueError("policy must contain lambda and fallback")
    penalty = policy["lambda"]
    if not np.isscalar(penalty) or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("policy lambda must be finite and nonnegative")
    if not isinstance(policy["fallback"], (bool, np.bool_)):
        raise ValueError("policy fallback must be a boolean")
    if policy["fallback"]:
        return np.full(len(predictions), baseline_index, dtype=np.int64)
    if len(predictions) == 0:
        return np.empty(0, dtype=np.int64)
    return _select(predictions, costs, float(penalty), baseline_index)
