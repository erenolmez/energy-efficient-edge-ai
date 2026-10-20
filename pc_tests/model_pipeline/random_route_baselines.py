"""Frozen, image-independent schedules for random-routing controls."""
import numpy as np


def selection_by_length(row, lengths):
    """Recover full/tail model counts without using any test labels."""
    lengths = np.asarray(lengths)
    counts = np.asarray(row['selected_batches'], dtype=int)
    images = np.asarray(row['selected_images'], dtype=int)
    full = int(lengths.max())
    if len(lengths) != counts.sum() or lengths.sum() != images.sum():
        raise ValueError('Batch/image counts do not match source run')
    tail_lengths = lengths[lengths != full]
    if len(tail_lengths) > 1:
        raise ValueError('At most one partial tail supported')
    result = {}
    if len(tail_lengths):
        tail = int(tail_lengths[0])
        remainder = images % full
        found = np.flatnonzero(remainder)
        if len(found) != 1 or remainder[found[0]] != tail:
            raise ValueError('Ambiguous tail identity')
        tail_counts = np.zeros_like(counts)
        tail_counts[found[0]] = 1
        result[tail] = tail_counts
        counts = counts - tail_counts
    if (counts < 0).any() or not np.array_equal(counts * full +
            sum((n * c for n, c in result.items()), np.zeros_like(counts)), images):
        raise ValueError('Invalid full/tail counts')
    result[full] = counts
    return result


def random_schedule(lengths, candidates, seed, counts_by_length=None):
    """Uniform candidates, or a uniform permutation of fixed counts per size."""
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths)
    if counts_by_length is None:
        return rng.integers(candidates, size=len(lengths), dtype=np.int64)
    choices = np.empty(len(lengths), dtype=np.int64)
    for length in sorted(set(lengths)):
        positions = np.flatnonzero(lengths == length)
        counts = np.asarray(counts_by_length[int(length)])
        if len(counts) != candidates or counts.sum() != len(positions) or (counts < 0).any():
            raise ValueError('Invalid frequency-matching counts')
        choices[positions] = rng.permutation(np.repeat(np.arange(candidates), counts))
    return choices


def cached_accuracy(predictions, labels, batches, choices):
    correct = sum(int(np.sum(predictions[int(model), batch] == labels[batch]))
                  for batch, model in zip(batches, choices))
    return 100 * correct / len(labels)
