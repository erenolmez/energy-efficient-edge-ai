"""Fit/calibration-only nested class-balanced subset design and loss weights."""
import numpy as np

def nested_subsets(labels,positions,sizes,seed=42):
    labels=np.asarray(labels);positions=np.asarray(positions,dtype=np.int64)
    if len(set(positions.tolist()))!=len(positions):raise ValueError('Unique positions required')
    classes=np.unique(labels[positions]);rng=np.random.default_rng(seed)
    groups={c:rng.permutation(positions[labels[positions]==c]) for c in classes}
    result={}
    for size in sizes:
        if size<=0 or size%len(classes):raise ValueError('Exact class balance required')
        n=size//len(classes)
        if any(len(g)<n for g in groups.values()):raise ValueError('Insufficient class support')
        result[size]=rng.permutation(np.concatenate([groups[c][:n] for c in classes]))
    return result

def positive_weights(targets):
    y=np.asarray(targets,dtype=np.float32)
    if y.ndim!=2 or not len(y) or not np.isin(y,(0,1)).all():raise ValueError('Binary nonempty targetmatrix required')
    positive=y.sum(0)
    return np.clip((len(y)-positive)/np.maximum(positive,1),.1,10).astype(np.float32)
